"""FastAPI application for the local LeRobot console."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as error:
    raise ImportError("Install the local console with `pip install 'lerobot[console]'`") from error

from pydantic import BaseModel, Field, field_validator

from evomind_lerobot.calibration_service import CalibrationService
from evomind_lerobot.catalog import hardware_catalog
from evomind_lerobot.collection_store import (
    CollectionStore,
    CollectionTaskConflictError,
    CollectionTaskNotFoundError,
    local_today,
)
from evomind_lerobot.dataset_browser import (
    DatasetNotFoundError,
    DatasetUnavailableError,
    dataset_detail,
    dataset_episode,
    dataset_video_path,
    datasets_catalog,
)
from evomind_lerobot.device_config import (
    DeviceConfiguration,
    load_device_configuration,
    save_device_configuration,
)
from evomind_lerobot.discovery import hardware_inventory, runtime_inventory
from evomind_lerobot.events import EventBroker, Operation, Phase
from evomind_lerobot.feetech_service import (
    FeetechActionRequest,
    FeetechScanRequest,
    FeetechSnapshotRequest,
    run_feetech_action,
    scan_feetech,
    snapshot_feetech,
)
from evomind_lerobot.identification import (
    camera_previews,
    poll_motion_identification,
    start_motion_identification,
    stop_motion_identification,
)
from evomind_lerobot.jobs import JobManager
from evomind_lerobot.piper_service import (
    PiperActionRequest,
    PiperScanRequest,
    close_piper_session,
    run_piper_action,
    scan_piper,
    snapshot_piper,
)
from evomind_lerobot.robot_assets import (
    RobotModelNotFoundError,
    robot_asset_content_type,
    robot_asset_path,
    robot_model_manifest,
)
from evomind_lerobot.runtime_service import (
    CollectionStartRequest,
    HardwareBusyError,
    PolicyInspectRequest,
    PolicyPreloadRequest,
    RecordingStartRequest,
    ReplayStartRequest,
    RolloutStartRequest,
    RuntimeCommandRequest,
    RuntimeService,
    TeleoperationStartRequest,
    inspect_policy_compatibility,
    require_local_policy,
)
from evomind_lerobot.workspace import workspace_inventory


class MotionStartRequest(BaseModel):
    model: str = Field(min_length=1)
    excluded_ids: list[str] = Field(default_factory=list)


class CalibrationStartRequest(BaseModel):
    alias: str = Field(min_length=1)


class CalibrationAutoStartRequest(BaseModel):
    alias: str | None = Field(default=None, min_length=1)


class CollectionTaskCreateRequest(BaseModel):
    work_date: date = Field(default_factory=local_today)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    target_duration_s: float = Field(gt=0, le=604_800)
    num_episodes: int = Field(default=20, ge=1, le=10_000)
    episode_time_s: int = Field(default=30, ge=1, le=86_400)
    reset_time_s: int = Field(default=10, ge=0, le=86_400)
    fps: int = Field(default=30, ge=1, le=60)
    collection_method: Literal["manual", "policy"] = "manual"
    policy_path: str = ""
    rollout_strategy: Literal[
        "episodic",
        "sentry",
        "highlight",
        "dagger_corrections",
        "dagger_continuous",
        "episodic_dagger",
    ] = "episodic_dagger"
    inference: Literal["sync", "rtc"] = "sync"
    duration_s: int = Field(default=120, ge=1, le=86_400)
    ring_buffer_seconds: int = Field(default=10, ge=1, le=300)

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("内容不能为空")
        return value


class CollectionTaskUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    target_duration_s: float = Field(gt=0, le=604_800)
    num_episodes: int = Field(default=20, ge=1, le=10_000)
    episode_time_s: int = Field(default=30, ge=1, le=86_400)
    reset_time_s: int = Field(default=10, ge=0, le=86_400)
    fps: int = Field(default=30, ge=1, le=60)
    collection_method: Literal["manual", "policy"] = "manual"
    policy_path: str = ""
    rollout_strategy: Literal[
        "episodic",
        "sentry",
        "highlight",
        "dagger_corrections",
        "dagger_continuous",
        "episodic_dagger",
    ] = "episodic_dagger"
    inference: Literal["sync", "rtc"] = "sync"
    duration_s: int = Field(default=120, ge=1, le=86_400)
    ring_buffer_seconds: int = Field(default=10, ge=1, le=300)

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("内容不能为空")
        return value


def _collection_task_settings(
    body: CollectionTaskCreateRequest | CollectionTaskUpdateRequest,
) -> dict[str, object]:
    policy_path = require_local_policy(body.policy_path) if body.collection_method == "policy" else ""
    return {
        "collection_method": body.collection_method,
        "policy_path": policy_path,
        "rollout_strategy": body.rollout_strategy,
        "inference": body.inference,
        "duration_s": body.duration_s,
        "ring_buffer_seconds": body.ring_buffer_seconds,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def create_app():
    events = EventBroker()
    jobs = JobManager(events)
    collection_store = CollectionStore()
    calibration = CalibrationService(events, jobs)
    runtime = RuntimeService(events, jobs, collection_store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Evomind LeRobot Console", version="0.1.0", lifespan=lifespan)
    app.state.events = events
    app.state.jobs = jobs
    app.state.calibration = calibration
    app.state.runtime = runtime
    app.state.collection_store = collection_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:3001",
            "http://localhost:3001",
        ],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status():
        current = jobs.current
        return {
            "ready": True,
            "lerobot_version": _package_version("lerobot"),
            "runtime": runtime_inventory(),
            "current_job": current.as_dict() if current else None,
            "event": events.latest.as_dict(),
        }

    @app.get("/api/devices")
    def devices():
        return hardware_inventory()

    @app.get("/api/catalog")
    def catalog():
        return hardware_catalog()

    @app.get("/api/workspace")
    def workspace():
        return workspace_inventory()

    @app.get("/api/collection/tasks")
    def collection_tasks(work_date: date | None = None):
        return collection_store.list_tasks(work_date or local_today())

    @app.post("/api/collection/tasks")
    def create_collection_task(body: CollectionTaskCreateRequest):
        try:
            policy_settings = _collection_task_settings(body)
            return collection_store.create_task(
                work_date=body.work_date,
                name=body.name,
                description=body.description,
                target_duration_s=body.target_duration_s,
                num_episodes=body.num_episodes,
                episode_time_s=body.episode_time_s,
                reset_time_s=body.reset_time_s,
                fps=body.fps,
                **policy_settings,
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except CollectionTaskConflictError as error:
            raise HTTPException(409, str(error)) from error

    @app.put("/api/collection/tasks/{task_id}")
    def update_collection_task(task_id: str, body: CollectionTaskUpdateRequest):
        try:
            policy_settings = _collection_task_settings(body)
            return collection_store.update_task(
                task_id,
                name=body.name,
                description=body.description,
                target_duration_s=body.target_duration_s,
                num_episodes=body.num_episodes,
                episode_time_s=body.episode_time_s,
                reset_time_s=body.reset_time_s,
                fps=body.fps,
                **policy_settings,
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except CollectionTaskNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except CollectionTaskConflictError as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/api/collection/tasks/{task_id}")
    def delete_collection_task(task_id: str):
        try:
            collection_store.delete_task(task_id)
        except CollectionTaskNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except CollectionTaskConflictError as error:
            raise HTTPException(409, str(error)) from error
        return {"ok": True}

    @app.get("/api/collection/progress")
    def collection_progress(work_date: date | None = None, window: int = 7):
        if window not in {7, 30}:
            raise HTTPException(400, "趋势窗口只能是 7 或 30 天")
        selected_date = work_date or local_today()
        progress = collection_store.progress(selected_date, window)
        active_session = progress["active_session"]
        if active_session and active_session["work_date"] == selected_date.isoformat():
            active_session["event"] = runtime.status()["event"]
        else:
            progress["active_session"] = None
        return progress

    @app.get("/api/datasets")
    def datasets():
        return datasets_catalog(runtime.active_dataset_id)

    @app.get("/api/dataset/detail")
    def read_dataset_detail(dataset_id: str):
        try:
            return dataset_detail(dataset_id, runtime.active_dataset_id)
        except DatasetNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except DatasetUnavailableError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/dataset/episode")
    def read_dataset_episode(dataset_id: str, episode: int = 0):
        try:
            return dataset_episode(
                dataset_id,
                episode,
                active_dataset_id=runtime.active_dataset_id,
            )
        except DatasetNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except DatasetUnavailableError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/dataset/video")
    def read_dataset_video(dataset_id: str, episode: int, camera: str):
        try:
            path = dataset_video_path(
                dataset_id,
                episode,
                camera,
                active_dataset_id=runtime.active_dataset_id,
            )
        except DatasetNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except DatasetUnavailableError as error:
            raise HTTPException(409, str(error)) from error
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/dataset/robot-model/{model}")
    def read_robot_model(model: str):
        try:
            return robot_model_manifest(model)
        except RobotModelNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/dataset/robot-assets/{asset_id}/{path:path}")
    def read_robot_asset(asset_id: str, path: str):
        try:
            asset = robot_asset_path(asset_id, path)
        except RobotModelNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(
            asset,
            media_type=robot_asset_content_type(asset),
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/config")
    def read_configuration():
        configuration = load_device_configuration()
        return configuration.model_dump() if configuration else None

    @app.put("/api/config")
    def write_configuration(configuration: DeviceConfiguration):
        if jobs.current is not None:
            raise HTTPException(409, "硬件任务运行中，不能修改设备配置")
        catalog_data = hardware_catalog()
        profiles = {profile["id"]: profile for profile in catalog_data["systems"]}
        profile = profiles.get(configuration.profile_id)
        if profile is None:
            raise HTTPException(400, f"Unknown device profile: {configuration.profile_id}")
        if configuration.robot_type != profile["robot_type"]:
            raise HTTPException(400, "Robot type does not match the selected device profile")
        if configuration.teleoperator_type != profile["teleoperator_type"]:
            raise HTTPException(400, "Teleoperator type does not match the selected device profile")

        def profile_bindings(kind: str):
            transport = profile[f"{kind}_transport"]
            source = configuration.can_bindings if transport == "socketcan" else configuration.serial_bindings
            return [binding for binding in source if binding.kind == kind]

        for kind in ("robot", "teleoperator"):
            expected_transport = profile[f"{kind}_transport"]
            unexpected = (
                [binding for binding in configuration.serial_bindings if binding.kind == kind]
                if expected_transport == "socketcan"
                else [binding for binding in configuration.can_bindings if binding.kind == kind]
            )
            if unexpected:
                raise HTTPException(
                    400,
                    f"{kind.capitalize()} bindings must use {expected_transport}",
                )

        robot_bindings = profile_bindings("robot")
        teleoperator_bindings = profile_bindings("teleoperator")
        if (
            robot_bindings
            and profile["robot_ports"] is not None
            and len(robot_bindings) != profile["robot_ports"]
        ):
            raise HTTPException(400, f"This device requires {profile['robot_ports']} robot ports")
        if (
            teleoperator_bindings
            and profile["teleoperator_ports"] is not None
            and len(teleoperator_bindings) != profile["teleoperator_ports"]
        ):
            raise HTTPException(
                400,
                f"This device requires {profile['teleoperator_ports']} teleoperator ports",
            )

        inventory = hardware_inventory()
        serial_devices = {device["id"]: device for device in inventory["serial"]}
        can_devices = {device["id"]: device for device in inventory["socketcan"]}
        camera_devices = {device["id"]: device for device in inventory["cameras"]}
        current_configuration = load_device_configuration()
        unchanged_serial = (
            {(binding.alias, binding.id, binding.port) for binding in current_configuration.serial_bindings}
            if current_configuration
            else set()
        )
        unchanged_can = (
            {(binding.alias, binding.id) for binding in current_configuration.can_bindings}
            if current_configuration
            else set()
        )
        unchanged_cameras = (
            {(binding.alias, binding.id, binding.port) for binding in current_configuration.camera_bindings}
            if current_configuration
            else set()
        )
        for binding in configuration.serial_bindings:
            if (binding.alias, binding.id, binding.port) in unchanged_serial:
                continue
            device = serial_devices.get(binding.id)
            if device is None or binding.port != device["path"]:
                raise HTTPException(400, f"Serial device is no longer connected: {binding.id}")
        for binding in configuration.can_bindings:
            if (binding.alias, binding.id) in unchanged_can:
                continue
            if binding.id not in can_devices:
                raise HTTPException(400, f"SocketCAN device is no longer connected: {binding.id}")
        for binding in configuration.camera_bindings:
            if (binding.alias, binding.id, binding.port) in unchanged_cameras:
                continue
            device = camera_devices.get(binding.id)
            if device is None or binding.port != device["path"]:
                raise HTTPException(400, f"Camera is no longer connected: {binding.id}")
            if binding.driver != device["driver"]:
                raise HTTPException(400, f"Camera driver does not match the connected device: {binding.id}")
            if binding.serial_number and binding.serial_number != device["serial_number"]:
                raise HTTPException(400, f"Camera serial number does not match the connected device: {binding.id}")

        save_device_configuration(configuration)
        return configuration.model_dump()

    @app.post("/api/maintenance/feetech/scan")
    async def scan_feetech_servos(body: FeetechScanRequest):
        events.publish(Operation.DIAGNOSTICS, Phase.STARTING, "正在扫描飞特舵机")
        try:
            result = await asyncio.to_thread(scan_feetech, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            events.publish(Operation.DIAGNOSTICS, Phase.FAILED, str(error))
            raise HTTPException(400, str(error)) from error
        events.publish(
            Operation.DIAGNOSTICS,
            Phase.COMPLETED,
            f"发现 {len(result['motors'])} 个飞特舵机",
            data={"device_id": body.device_id, "motor_count": len(result["motors"])},
        )
        return result

    @app.post("/api/maintenance/feetech/action")
    async def control_feetech_servo(body: FeetechActionRequest):
        if body.action in {"set_id", "set_baudrate"}:
            if jobs.current is not None:
                raise HTTPException(409, "硬件任务运行中，不能修改舵机配置")
            await asyncio.to_thread(stop_motion_identification)
        events.publish(
            Operation.DIAGNOSTICS,
            Phase.RUNNING,
            f"执行舵机操作：{body.action}",
            data={"device_id": body.device_id, "motor_id": body.motor_id},
        )
        try:
            result = await asyncio.to_thread(run_feetech_action, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            events.publish(Operation.DIAGNOSTICS, Phase.FAILED, str(error))
            raise HTTPException(400, str(error)) from error
        message = (
            f"舵机 ID {body.motor_id} 已修改为 {body.value}"
            if body.action == "set_id"
            else "舵机操作完成"
        )
        events.publish(Operation.DIAGNOSTICS, Phase.COMPLETED, message)
        return result

    @app.post("/api/maintenance/feetech/snapshot")
    async def snapshot_feetech_servos(body: FeetechSnapshotRequest):
        try:
            return await asyncio.to_thread(snapshot_feetech, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/maintenance/piper/scan")
    async def scan_piper_arm(body: PiperScanRequest):
        if jobs.current is not None:
            raise HTTPException(409, "硬件任务运行中，不能打开维修工具")
        await asyncio.to_thread(stop_motion_identification)
        events.publish(Operation.DIAGNOSTICS, Phase.STARTING, "正在读取 PiperX")
        try:
            result = await asyncio.to_thread(scan_piper, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            events.publish(Operation.DIAGNOSTICS, Phase.FAILED, str(error))
            raise HTTPException(400, str(error)) from error
        events.publish(
            Operation.DIAGNOSTICS,
            Phase.COMPLETED,
            "PiperX 状态读取完成",
            data={"device_id": body.device_id, "interface": result["interface"]},
        )
        return result

    @app.post("/api/maintenance/piper/snapshot")
    async def snapshot_piper_arm(body: PiperScanRequest):
        try:
            return await asyncio.to_thread(snapshot_piper, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/maintenance/piper/close")
    async def close_piper_arm():
        await asyncio.to_thread(close_piper_session)
        return {"status": "closed"}

    @app.post("/api/maintenance/piper/action")
    async def control_piper_arm(body: PiperActionRequest):
        if jobs.current is not None:
            raise HTTPException(409, "硬件任务运行中，不能执行维修操作")
        events.publish(
            Operation.DIAGNOSTICS,
            Phase.RUNNING,
            f"执行 PiperX 操作：{body.action}",
            data={"device_id": body.device_id, "motor_id": body.motor_id},
        )
        try:
            result = await asyncio.to_thread(run_piper_action, body)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            events.publish(Operation.DIAGNOSTICS, Phase.FAILED, str(error))
            raise HTTPException(400, str(error)) from error
        events.publish(Operation.DIAGNOSTICS, Phase.COMPLETED, "PiperX 操作完成")
        return result

    @app.get("/api/calibration/status")
    def calibration_status():
        return calibration.status()

    @app.post("/api/calibration/auto/start")
    def calibration_auto_start(body: CalibrationAutoStartRequest):
        try:
            return calibration.start_auto(body.alias)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/calibration/manual/start")
    def calibration_manual_start(body: CalibrationStartRequest):
        try:
            return calibration.start_manual(body.alias)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/calibration/manual/advance")
    def calibration_manual_advance():
        try:
            return calibration.advance_manual()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/runtime/status")
    def runtime_status():
        return runtime.status()

    @app.post("/api/runtime/teleoperation/start")
    def runtime_teleoperation_start(body: TeleoperationStartRequest):
        try:
            close_piper_session()
            return runtime.start(Operation.TELEOPERATION, body)
        except HardwareBusyError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/recording/start")
    def runtime_recording_start(body: RecordingStartRequest):
        try:
            close_piper_session()
            return runtime.start(Operation.RECORDING, body)
        except CollectionTaskNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except CollectionTaskConflictError as error:
            raise HTTPException(409, str(error)) from error
        except (HardwareBusyError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/collection/start")
    def runtime_collection_start(body: CollectionStartRequest):
        try:
            close_piper_session()
            return runtime.start_collection(body)
        except CollectionTaskNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except (CollectionTaskConflictError, HardwareBusyError) as error:
            raise HTTPException(409, str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/runtime/rollout/start")
    def runtime_rollout_start(body: RolloutStartRequest):
        try:
            close_piper_session()
            return runtime.start(Operation.ROLLOUT, body)
        except HardwareBusyError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/policy/inspect")
    def runtime_policy_inspect(body: PolicyInspectRequest):
        try:
            local_path = require_local_policy(body.policy_path)
            return inspect_policy_compatibility(body.model_copy(update={"policy_path": local_path}))
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/runtime/policy/residency")
    def runtime_policy_residency():
        return runtime.policy_residency()

    @app.post("/api/runtime/policy/preload")
    def runtime_policy_preload(body: PolicyPreloadRequest):
        try:
            return runtime.preload_policy(body)
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/policy/unload")
    def runtime_policy_unload():
        try:
            return runtime.unload_policy()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/replay/start")
    def runtime_replay_start(body: ReplayStartRequest):
        try:
            close_piper_session()
            return runtime.start(Operation.REPLAY, body)
        except HardwareBusyError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/runtime/command")
    def runtime_command(body: RuntimeCommandRequest):
        try:
            return runtime.command(body.command)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/calibration/stop")
    def calibration_stop():
        try:
            return calibration.stop()
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/identify/motion/start")
    async def identify_motion_start(body: MotionStartRequest):
        await asyncio.to_thread(close_piper_session)
        return await asyncio.to_thread(
            start_motion_identification,
            body.model,
            set(body.excluded_ids),
        )

    @app.get("/api/identify/motion/poll")
    async def identify_motion_poll():
        try:
            return await asyncio.to_thread(poll_motion_identification)
        except RuntimeError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/identify/motion/stop")
    async def identify_motion_stop():
        return await asyncio.to_thread(stop_motion_identification)

    @app.post("/api/identify/cameras")
    async def identify_cameras():
        return await asyncio.to_thread(camera_previews)

    @app.websocket("/api/events")
    async def event_stream(websocket: WebSocket):
        await websocket.accept()
        queue = events.subscribe()
        try:
            await websocket.send_json(events.latest.as_dict())
            while True:
                event = await queue.get()
                await websocket.send_json(event.as_dict())
        except WebSocketDisconnect:
            pass
        finally:
            events.unsubscribe(queue)

    source_root = Path(__file__).resolve().parents[2]
    web_root = Path(os.environ.get("EVOMIND_LEROBOT_WEB_ROOT", source_root / "web/dist/client"))
    if web_root.joinpath("index.html").is_file():
        app.mount("/", StaticFiles(directory=web_root, html=True), name="console")

    return app


app = create_app()
