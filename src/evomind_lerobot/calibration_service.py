"""Calibration lifecycle for the local HTTP runtime."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evomind_lerobot.calibration import (
    SO101_DEVICE_TYPES,
    CalibrationDeviceConfig,
    CalibrationStoppedError,
    run_native_manual_calibration,
    run_so101_auto_calibration,
)
from evomind_lerobot.device_config import (
    DeviceConfiguration,
    SerialBinding,
    calibration_path,
    device_type,
    load_device_configuration,
)
from evomind_lerobot.events import EventBroker, Operation, Phase
from evomind_lerobot.jobs import JobManager


@dataclass
class CalibrationSnapshot:
    state: str = "idle"
    mode: str = ""
    phase: str = ""
    alias: str = ""
    message: str = ""
    prompt_id: str = ""
    calibration_id: str = ""
    path: str = ""
    motor: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _supports_auto_calibration(configuration: DeviceConfiguration, binding: SerialBinding) -> bool:
    return device_type(configuration, binding) in SO101_DEVICE_TYPES


def _supports_manual_calibration(configuration: DeviceConfiguration, binding: SerialBinding) -> bool:
    return device_type(configuration, binding) in SO101_DEVICE_TYPES | {
        "so100_follower",
        "so100_leader",
    }


def _calibration_files(configuration: DeviceConfiguration) -> dict[str, Path]:
    return {
        binding.alias: calibration_path(configuration, binding)
        for binding in configuration.serial_bindings
    }


class CalibrationService:
    def __init__(self, events: EventBroker, jobs: JobManager) -> None:
        self._events = events
        self._jobs = jobs
        self._lock = threading.RLock()
        self._snapshot = CalibrationSnapshot()
        self._auto_snapshots: dict[str, CalibrationSnapshot] = {}
        self._auto_pending: set[str] = set()
        self._auto_failed = False
        self._auto_job_id: str | None = None
        self._stop_event = threading.Event()
        self._prompt_event = threading.Event()
        self._commands: set[str] = set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot.as_dict()
            auto_snapshots = {
                alias: run_snapshot.as_dict()
                for alias, run_snapshot in self._auto_snapshots.items()
            }
        configuration = load_device_configuration()
        files = _calibration_files(configuration) if configuration else {}
        snapshot["devices"] = (
            {
                alias: {
                    "available": path.is_file(),
                    "path": str(path),
                    "run": auto_snapshots.get(alias),
                }
                for alias, path in files.items()
            }
        )
        return snapshot

    def start_auto(self, alias: str | None = None) -> dict[str, Any]:
        configuration = load_device_configuration()
        if configuration is None:
            raise ValueError("请先完成设备配置")
        eligible_bindings = [
            binding
            for binding in configuration.serial_bindings
            if _supports_auto_calibration(configuration, binding)
        ]
        if not eligible_bindings:
            raise ValueError("当前设备不支持自动校准")
        if alias is not None:
            eligible_bindings = [binding for binding in eligible_bindings if binding.alias == alias]
            if not eligible_bindings:
                raise ValueError(f"设备不支持自动校准：{alias}")

        with self._lock:
            active = self._snapshot.state in {"starting", "running", "stopping"}
            if active and self._snapshot.mode != "auto":
                raise RuntimeError("已有手动校准正在运行")
            if self._snapshot.state == "stopping":
                raise RuntimeError("自动校准正在停止")
            if not active:
                job = self._jobs.acquire(Operation.CALIBRATION, "正在自动校准")
                self._stop_event.clear()
                self._prompt_event.clear()
                self._commands.clear()
                self._auto_job_id = job.id
                self._auto_pending.clear()
                self._auto_failed = False
                self._auto_snapshots.clear()

            bindings = [
                binding
                for binding in eligible_bindings
                if binding.alias not in self._auto_snapshots
            ]
            if not bindings:
                raise RuntimeError("选择的机械臂已在当前自动校准批次中")
            self._auto_pending.update(binding.alias for binding in bindings)
            self._auto_snapshots.update({
                binding.alias: CalibrationSnapshot(
                    state="starting",
                    mode="auto",
                    phase="preparing",
                    alias=binding.alias,
                    message="等待自动校准启动。",
                    calibration_id=calibration_path(configuration, binding).stem,
                    path=str(calibration_path(configuration, binding)),
                )
                for binding in bindings
            })
            count = len(self._auto_pending)
            message = (
                f"正在自动校准 {bindings[0].alias}。"
                if count == 1
                else f"正在并行自动校准 {count} 只机械臂。"
            )
            if active:
                self._update(state="running", phase="preparing", message=message)
            else:
                self._snapshot = CalibrationSnapshot(
                    state="starting",
                    mode="auto",
                    phase="preparing",
                    message=message,
                )

        for binding in bindings:
            path = calibration_path(configuration, binding)
            threading.Thread(
                target=self._run_auto,
                args=(configuration, binding, path),
                name=f"evomind-calibration:{binding.alias}",
                daemon=True,
            ).start()
        return self.status()

    def start_manual(self, alias: str) -> dict[str, Any]:
        configuration, binding = self._resolve_binding(alias)
        if not _supports_manual_calibration(configuration, binding):
            raise ValueError("当前设备暂不支持网页手动校准")

        with self._lock:
            if self._snapshot.state in {"starting", "running", "stopping"}:
                raise RuntimeError("已有校准正在运行")
            job = self._jobs.acquire(Operation.CALIBRATION, f"正在手动校准 {alias}")
            path = calibration_path(configuration, binding)
            self._auto_snapshots.clear()
            self._auto_pending.clear()
            self._auto_failed = False
            self._auto_job_id = None
            self._stop_event.clear()
            self._prompt_event.clear()
            self._commands.clear()
            self._snapshot = CalibrationSnapshot(
                state="starting",
                mode="manual",
                phase="preparing",
                alias=alias,
                message="正在连接设备。",
                calibration_id=path.stem,
                path=str(path),
            )

        thread = threading.Thread(
            target=self._run_manual,
            args=(job.id, configuration, binding, path),
            name=f"evomind-manual-calibration:{alias}",
            daemon=True,
        )
        thread.start()
        return self.status()

    def advance_manual(self) -> dict[str, Any]:
        with self._lock:
            if self._snapshot.mode != "manual" or self._snapshot.state not in {"starting", "running"}:
                raise RuntimeError("当前没有运行中的手动校准")
            if self._snapshot.prompt_id == "calibration_middle":
                self._prompt_event.set()
            elif self._snapshot.phase == "recording_ranges":
                self._commands.add("finish_calibration_range")
            else:
                raise RuntimeError("当前步骤不需要确认")
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._snapshot.state not in {"starting", "running", "stopping"}:
                raise RuntimeError("当前没有运行中的校准")
            self._stop_event.set()
            self._commands.add("stop")
            self._prompt_event.set()
            for snapshot in self._auto_snapshots.values():
                if snapshot.state in {"starting", "running"}:
                    snapshot.state = "stopping"
                    snapshot.phase = "stopping"
                    snapshot.message = "正在停止自动校准。"
                    snapshot.updated_at = int(time.time())
            self._update(state="stopping", phase="stopping", message="正在停止校准。")
        self._events.publish(
            Operation.CALIBRATION,
            Phase.STOPPING,
            "正在停止校准",
            data=self.status(),
        )
        return self.status()

    def manual_emit(self, operation: str, phase: str, data: dict[str, Any]) -> None:
        del operation
        messages = {
            "starting": "正在准备 LeRobot 手动校准。",
            "connecting": "正在连接设备。",
            "running": "手动校准已开始。",
            "recording_ranges": "请依次将所有关节移动到完整行程，然后完成范围记录。",
            "ranges_recorded": "已记录关节范围，正在保存校准文件。",
            "stopping": "正在断开设备。",
            "completed": "手动校准已完成。",
        }
        with self._lock:
            if self._snapshot.mode != "manual":
                return
            self._update(
                state="done" if phase == "completed" else "running",
                phase=phase,
                prompt_id="",
                message=messages.get(phase, self._snapshot.message),
            )
            snapshot = self._snapshot.as_dict()
        self._events.publish(
            Operation.CALIBRATION,
            Phase.COMPLETED if phase == "completed" else Phase.RUNNING,
            self._snapshot.message,
            data={**snapshot, **data},
        )

    def manual_prompt(self, prompt_id: str, message: str) -> str:
        if prompt_id == "calibration_mismatch":
            return "c"
        with self._lock:
            self._prompt_event.clear()
            self._update(
                state="running",
                phase="positioning",
                prompt_id=prompt_id,
                message="请将机械臂放到各关节活动范围的中间位置。",
            )
            snapshot = self._snapshot.as_dict()
        self._events.publish(
            Operation.CALIBRATION,
            Phase.RUNNING,
            snapshot["message"],
            data=snapshot,
        )
        self._prompt_event.wait()
        if self._stop_event.is_set():
            raise CalibrationStoppedError("Stopped by user.")
        return ""

    def take_manual_commands(self) -> set[str]:
        with self._lock:
            commands = set(self._commands)
            self._commands.clear()
        return commands

    def _resolve_binding(self, alias: str) -> tuple[DeviceConfiguration, SerialBinding]:
        configuration = load_device_configuration()
        if configuration is None:
            raise ValueError("请先完成设备配置")
        binding = next((item for item in configuration.serial_bindings if item.alias == alias), None)
        if binding is None:
            raise ValueError(f"未找到设备：{alias}")
        return configuration, binding

    def _update(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self._snapshot, key, value)
        self._snapshot.updated_at = int(time.time())

    def _on_auto_event(self, alias: str, event: dict[str, Any]) -> None:
        phase = str(event.get("phase") or "running")
        with self._lock:
            snapshot = self._auto_snapshots[alias]
            for key, value in {
                "state": "done" if phase == "done" else "running",
                "phase": phase,
                "message": str(event.get("message") or self._snapshot.message),
                "motor": str(event.get("motor") or ""),
                "path": str(event.get("path") or snapshot.path),
                "profile": event.get("profile") or snapshot.profile,
            }.items():
                setattr(snapshot, key, value)
            snapshot.updated_at = int(time.time())
            self._update(
                state="running",
                phase=phase,
                message=f"正在并行自动校准：{len(self._auto_pending)} 只机械臂运行中。",
            )
            event_snapshot = snapshot.as_dict()
            event_message = snapshot.message
        phase_map = {
            "preparing": Phase.PREPARING,
            "probing": Phase.PROBING,
            "saving": Phase.SAVING,
            "done": Phase.COMPLETED,
        }
        self._events.publish(
            Operation.CALIBRATION,
            phase_map.get(phase, Phase.RUNNING),
            f"{alias}：{event_message}",
            data=event_snapshot,
        )

    def _run_auto(
        self,
        configuration: DeviceConfiguration,
        binding: SerialBinding,
        path: Path,
    ) -> None:
        try:
            result = run_so101_auto_calibration(
                CalibrationDeviceConfig(
                    arm_type=device_type(configuration, binding),
                    role="leader" if binding.kind == "teleoperator" else "follower",
                    port=binding.port,
                    calibration_dir=str(path.parent),
                    calibration_id=path.stem,
                ),
                stop_event=self._stop_event,
                on_event=lambda event: self._on_auto_event(binding.alias, event),
            )
            with self._lock:
                snapshot = self._auto_snapshots[binding.alias]
                snapshot.state = "done"
                snapshot.phase = "done"
                snapshot.message = "自动校准已完成。"
                snapshot.path = result.calibration_path
                snapshot.profile = result.profile
                snapshot.error = ""
                snapshot.updated_at = int(time.time())
            self._finish_auto(binding.alias, failed=False)
        except CalibrationStoppedError:
            with self._lock:
                snapshot = self._auto_snapshots[binding.alias]
                snapshot.state = "stopped"
                snapshot.phase = "stopped"
                snapshot.message = "自动校准已停止。"
                snapshot.updated_at = int(time.time())
            self._finish_auto(binding.alias, failed=False)
        except Exception as error:
            with self._lock:
                snapshot = self._auto_snapshots[binding.alias]
                snapshot.state = "error"
                snapshot.phase = "error"
                snapshot.message = str(error)
                snapshot.error = str(error)
                snapshot.updated_at = int(time.time())
            self._finish_auto(binding.alias, failed=True)

    def _finish_auto(self, alias: str, *, failed: bool) -> None:
        with self._lock:
            self._auto_pending.discard(alias)
            self._auto_failed = self._auto_failed or failed
            if self._auto_pending:
                self._update(
                    state="running",
                    phase="probing",
                    message=f"正在并行自动校准：{len(self._auto_pending)} 只机械臂运行中。",
                )
                return
            any_stopped = any(snapshot.state == "stopped" for snapshot in self._auto_snapshots.values())
            final_state = "error" if self._auto_failed else "stopped" if any_stopped else "done"
            final_message = "部分机械臂自动校准失败。" if self._auto_failed else "自动校准已停止。" if any_stopped else "全部机械臂自动校准已完成。"
            self._update(
                state=final_state,
                phase=final_state,
                message=final_message,
                error=final_message if self._auto_failed else "",
            )
            job_id = self._auto_job_id
            self._auto_job_id = None
        if job_id is not None:
            self._jobs.release(
                job_id,
                failed=self._auto_failed,
                message=final_message,
                data=self.status(),
            )

    def _run_manual(
        self,
        job_id: str,
        configuration: DeviceConfiguration,
        binding: SerialBinding,
        path: Path,
    ) -> None:
        bridge = _ManualCalibrationBridge(self)
        try:
            result = run_native_manual_calibration(
                CalibrationDeviceConfig(
                    arm_type=device_type(configuration, binding),
                    role="leader" if binding.kind == "teleoperator" else "follower",
                    port=binding.port,
                    calibration_dir=str(path.parent),
                    calibration_id=path.stem,
                ),
                bridge,
            )
            with self._lock:
                self._update(
                    state="done",
                    phase="done",
                    prompt_id="",
                    message="手动校准已完成。",
                    path=result.calibration_path,
                    profile=result.profile,
                    error="",
                )
            self._jobs.release(job_id, message="手动校准已完成", data=self.status())
        except CalibrationStoppedError:
            with self._lock:
                self._update(state="stopped", phase="stopped", prompt_id="", message="手动校准已停止。")
            self._jobs.release(job_id, message="手动校准已停止", data=self.status())
        except Exception as error:
            if self._stop_event.is_set():
                with self._lock:
                    self._update(state="stopped", phase="stopped", prompt_id="", message="手动校准已停止。")
                self._jobs.release(job_id, message="手动校准已停止", data=self.status())
                return
            with self._lock:
                self._update(state="error", phase="error", prompt_id="", message=str(error), error=str(error))
            self._jobs.release(job_id, failed=True, message=str(error), data=self.status())


class _ManualCalibrationBridge:
    def __init__(self, service: CalibrationService) -> None:
        self._service = service

    def emit(self, operation: str, phase: str, data: dict[str, Any]) -> None:
        self._service.manual_emit(operation, phase, data)

    def take_commands(self) -> set[str]:
        return self._service.take_manual_commands()

    def prompt(self, prompt_id: str, message: str) -> str:
        return self._service.manual_prompt(prompt_id, message)
