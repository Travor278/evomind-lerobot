"""Run native LeRobot workflows behind the local HTTP runtime."""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import signal
import sqlite3
import threading
from pathlib import Path
from queue import Empty
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evomind_lerobot.collection_store import CollectionStore, CollectionStoreError
from evomind_lerobot.device_config import (
    CameraBinding,
    CanBinding,
    DeviceConfiguration,
    SerialBinding,
    calibration_path,
    load_device_configuration,
    runtime_id,
)
from evomind_lerobot.events import EventBroker, Operation, Phase
from evomind_lerobot.jobs import HardwareBusyError, JobManager
from evomind_lerobot.workspace import datasets_inventory, policies_inventory


class TeleoperationStartRequest(BaseModel):
    fps: int = Field(default=30, ge=1, le=60)


class RecordingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)


class CollectionStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)


class RecordingExecutionRequest(RecordingStartRequest):
    fps: int = Field(ge=1, le=60)
    num_episodes: int = Field(ge=1, le=10_000)
    episode_time_s: int = Field(ge=1, le=86_400)
    reset_time_s: int = Field(ge=0, le=86_400)


class RolloutStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_path: str = Field(min_length=1)
    strategy: Literal[
        "base",
        "episodic",
        "sentry",
        "highlight",
        "dagger_corrections",
        "dagger_continuous",
        "episodic_dagger",
    ] = "base"
    inference: Literal["sync", "rtc"] = "sync"
    task: str = Field(min_length=1, max_length=500)
    dataset_name: str = Field(default="rollout_policy-rollout", min_length=1, max_length=80)
    fps: int = Field(default=30, ge=1, le=60)
    duration_s: int = Field(default=120, ge=1, le=86_400)
    num_episodes: int = Field(default=10, ge=1, le=10_000)
    episode_time_s: int = Field(default=30, ge=1, le=86_400)
    reset_time_s: int = Field(default=10, ge=0, le=86_400)
    ring_buffer_seconds: int = Field(default=10, ge=1, le=300)
    return_to_initial_position: bool = False


class PolicyInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_path: str = Field(min_length=1)


class PolicyPreloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_path: str = Field(min_length=1)


class ReplayStartRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    episode: int = Field(default=0, ge=0)


class RuntimeCommandRequest(BaseModel):
    command: Literal[
        "stop",
        "finish_episode",
        "rerecord_episode",
        "pause_resume",
        "correction",
        "toggle_highlight",
    ]


_MESSAGES = {
    ("teleoperation", "starting"): "正在启动遥操作",
    ("teleoperation", "connecting"): "正在连接遥操作设备和机械臂",
    ("teleoperation", "running"): "遥操作运行中",
    ("teleoperation", "stopping"): "正在停止遥操作",
    ("teleoperation", "completed"): "遥操作已结束",
    ("recording", "starting"): "正在启动数据采集",
    ("recording", "connecting"): "正在连接采集设备",
    ("recording", "running"): "数据采集中",
    ("recording", "resetting"): "正在重置场景",
    ("recording", "saving"): "正在保存 Episode",
    ("recording", "stopping"): "正在停止数据采集",
    ("recording", "completed"): "数据采集已结束",
    ("rollout", "starting"): "正在加载 Policy",
    ("rollout", "connecting"): "正在连接推理设备",
    ("rollout", "running"): "Policy 推理运行中",
    ("rollout", "stopping"): "正在停止推理",
    ("rollout", "completed"): "推理已结束",
    ("replay", "starting"): "正在加载回放数据",
    ("replay", "connecting"): "正在连接回放设备",
    ("replay", "running"): "数据回放中",
    ("replay", "stopping"): "正在停止回放",
    ("replay", "completed"): "回放已结束",
}


class ProcessRuntimeBridge:
    """RuntimeBridge implementation used inside a workflow process."""

    def __init__(self, event_queue: Any, command_queue: Any) -> None:
        self._event_queue = event_queue
        self._command_queue = command_queue

    def emit(self, operation: str, phase: str, data: dict[str, Any]) -> None:
        self._event_queue.put({"kind": "event", "operation": operation, "phase": phase, "data": data})

    def take_commands(self) -> set[str]:
        commands: set[str] = set()
        while True:
            try:
                commands.add(self._command_queue.get_nowait())
            except Empty:
                return commands

    def prompt(self, prompt_id: str, message: str) -> str:
        raise RuntimeError(f"当前网页尚未处理运行时确认：{prompt_id} · {message}")


def _repo_id(value: str) -> str:
    repo_id = value.strip()
    if not repo_id:
        raise ValueError("数据集 repo id 不能为空")
    return repo_id


def _rollout_repo_id(value: str) -> str:
    """Ensure rollout datasets follow LeRobot's required basename convention."""
    repo_id = _repo_id(value)
    namespace, separator, name = repo_id.rpartition("/")
    if not name.startswith("rollout_"):
        name = f"rollout_{name}"
    return f"{namespace}/{name}" if separator else name


def _policy_path(value: str) -> str:
    """Accept either a Hub repo id, local path, or a pasted Hugging Face URL."""
    path = value.strip().rstrip("/")
    match = re.fullmatch(r"https?://huggingface\.co/([^/]+/[^/]+)(?:/.*)?", path)
    return match.group(1) if match else path


def require_local_policy(value: str) -> str:
    """Resolve a policy selected from the local workspace inventory."""
    selected = value.strip()
    for policy in policies_inventory():
        if selected in {policy["id"], policy["path"]}:
            return policy["path"]
    raise ValueError("选择的 Policy 不在本机模型目录中")


def _recording_dataset_name(task: dict[str, Any]) -> str:
    """Build a readable, filesystem-safe dataset name from the selected daily task."""
    name = re.sub(r"[^\w.-]+", "-", str(task["name"]).strip(), flags=re.UNICODE).strip("-._")
    task_prefix = str(task["id"]).split("-", 1)[0]
    return f"{name[:48] or 'recording'}_{task_prefix}"


def _camera_config(binding: CameraBinding, fps: int) -> dict[str, Any]:
    if binding.driver == "intelrealsense":
        return {
            "type": "intelrealsense",
            "serial_number_or_name": binding.serial_number,
            "fps": min(fps, 30),
            "width": 640,
            "height": 480,
            "warmup_s": 2,
            "use_rgb": True,
            "use_depth": False,
        }
    return {
        "type": "opencv",
        "index_or_path": binding.port,
        "fps": min(fps, 30),
        "width": 640,
        "height": 480,
        # LeRobot's OpenCV default is one second. Cameras are connected
        # sequentially, so a three-second value adds roughly nine seconds for
        # this workstation's three-camera setup before inference can start.
        "warmup_s": 1,
        "fourcc": "MJPG",
    }


def _camera_key(alias: str, side: str) -> str:
    prefix = f"{side}_"
    return alias.removeprefix(prefix) if side in {"left", "right"} else alias


def _device_bindings(configuration: DeviceConfiguration, kind: str) -> list[SerialBinding | CanBinding]:
    return [
        binding
        for binding in [*configuration.serial_bindings, *configuration.can_bindings]
        if binding.kind == kind
    ]


def _binding_port(binding: SerialBinding | CanBinding) -> str:
    return binding.port if isinstance(binding, SerialBinding) else binding.id


def _robot_payload(configuration: DeviceConfiguration, fps: int, *, cameras: bool) -> dict[str, Any]:
    bindings = _device_bindings(configuration, "robot")
    payload: dict[str, Any] = {
        "type": configuration.robot_type,
        "id": runtime_id(configuration, "robot"),
    }

    camera_bindings = configuration.camera_bindings if cameras else []
    dual = {binding.side for binding in bindings} >= {"left", "right"}
    if dual:
        supports_top_level_cameras = configuration.robot_type == "bi_so_follower"
        for side in ("left", "right"):
            binding = next(item for item in bindings if item.side == side)
            side_cameras = {
                _camera_key(camera.alias, side): _camera_config(camera, fps)
                for camera in camera_bindings
                if camera.side == side
                or (not supports_top_level_cameras and side == "right" and camera.side == "single")
            }
            payload[f"{side}_arm_config"] = {
                "port": _binding_port(binding),
                "cameras": side_cameras,
            }
        # Bimanual LeRobot robots expose top-level cameras without a left/right
        # prefix.  Keeping environment cameras here also avoids silently dropping
        # them on serial dual-arm configurations.
        if supports_top_level_cameras:
            payload["cameras"] = {
                camera.alias: _camera_config(camera, fps)
                for camera in camera_bindings
                if camera.side == "single"
            }
    elif bindings:
        payload["port"] = _binding_port(bindings[0])
        payload["cameras"] = {camera.alias: _camera_config(camera, fps) for camera in camera_bindings}
    return payload


def _configured_camera_feature_name(
    configuration: DeviceConfiguration,
    camera: CameraBinding,
) -> str:
    """Return the feature name emitted by the configured robot at runtime."""
    bindings = _device_bindings(configuration, "robot")
    dual = {binding.side for binding in bindings} >= {"left", "right"}
    if dual and camera.side in {"left", "right"}:
        return f"{camera.side}_{_camera_key(camera.alias, camera.side)}"
    if dual and configuration.robot_type != "bi_so_follower":
        return f"right_{camera.alias}"
    return camera.alias


def _configured_visual_features(configuration: DeviceConfiguration) -> set[str]:
    """Return policy-facing visual keys without opening cameras or serial ports."""
    return {
        f"observation.images.{_configured_camera_feature_name(configuration, camera)}"
        for camera in configuration.camera_bindings
    }


def _configured_vector_dimensions(configuration: DeviceConfiguration) -> tuple[int | None, int | None]:
    """Infer fixed arm vector dimensions for compatibility checks."""
    robot_count = len(_device_bindings(configuration, "robot"))
    if configuration.robot_type in {"so100_follower", "so101_follower", "bi_so_follower"}:
        dimension = 6 * robot_count
        return dimension, dimension
    return None, None


def _camera_rename_map(
    configuration: DeviceConfiguration,
    expected: set[str],
    provided: set[str],
) -> dict[str, str]:
    """Resolve unambiguous camera aliases from hardware slots to policy inputs."""
    missing = expected - provided
    extra = provided - expected
    if not missing or not extra:
        return {}

    slot_by_alias = {slot.alias: slot for slot in configuration.camera_slots}
    slot_by_feature = {
        f"observation.images.{_configured_camera_feature_name(configuration, camera)}": slot_by_alias[
            camera.alias
        ]
        for camera in configuration.camera_bindings
        if camera.alias in slot_by_alias
    }
    rename_map: dict[str, str] = {}
    for source in sorted(extra):
        slot = slot_by_feature.get(source)
        if slot is None:
            continue
        candidates = []
        for target in missing:
            leaf = target.rsplit(".", 1)[-1].lower()
            if slot.kind == "environment":
                matches = any(token in leaf for token in ("base", "front", "environment", "top"))
            else:
                matches = "wrist" in leaf and slot.side in {"left", "right"} and slot.side in leaf
            if matches:
                candidates.append(target)
        if len(candidates) == 1:
            rename_map[source] = candidates[0]
            missing.remove(candidates[0])
    return rename_map


def _normalizer_feature_dim(policy_path: str, feature_name: str) -> int | None:
    """Read the checkpoint's effective (pre-padding) feature size from saved stats."""
    path = Path(policy_path)
    if not path.is_dir():
        return None
    try:
        from safetensors import safe_open

        for state_file in path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"):
            with safe_open(state_file, framework="pt", device="cpu") as tensors:
                tensor_keys = tensors.keys()
                for statistic in ("q01", "mean", "min"):
                    key = f"{feature_name}.{statistic}"
                    if key in tensor_keys:
                        shape = tensors.get_slice(key).get_shape()
                        return int(shape[-1]) if shape else None
    except (OSError, RuntimeError, ValueError):
        logging.info("Could not inspect normalizer dimensions for %s", policy_path, exc_info=True)
    return None


def _teleoperator_payload(configuration: DeviceConfiguration) -> dict[str, Any] | None:
    if configuration.teleoperator_type is None:
        return None
    bindings = _device_bindings(configuration, "teleoperator")
    payload: dict[str, Any] = {
        "type": configuration.teleoperator_type,
        "id": runtime_id(configuration, "teleoperator"),
    }
    dual = {binding.side for binding in bindings} >= {"left", "right"}
    if dual:
        for side in ("left", "right"):
            binding = next(item for item in bindings if item.side == side)
            payload[f"{side}_arm_config"] = {"port": _binding_port(binding)}
    elif bindings:
        payload["port"] = _binding_port(bindings[0])
    return payload


def _configuration() -> DeviceConfiguration:
    configuration = load_device_configuration()
    if configuration is None or not (configuration.serial_bindings or configuration.can_bindings):
        raise ValueError("请先完成设备识别")
    return configuration


def _require_calibration(configuration: DeviceConfiguration, kind: str) -> None:
    configured_type = configuration.robot_type if kind == "robot" else configuration.teleoperator_type or ""
    if "piperx" in configured_type:
        return
    missing = [
        binding.alias
        for binding in configuration.serial_bindings
        if binding.kind == kind and not calibration_path(configuration, binding).is_file()
    ]
    if missing:
        raise ValueError(f"请先校准设备：{', '.join(missing)}")


def _decode_hardware(
    configuration: DeviceConfiguration,
    fps: int,
    *,
    cameras: bool = True,
    include_teleoperator: bool = True,
):
    import draccus

    from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
        OpenCVCameraConfig as _OpenCVCameraConfig,
    )
    from lerobot.cameras.realsense.configuration_realsense import (  # noqa: F401
        RealSenseCameraConfig as _RealSenseCameraConfig,
    )
    from lerobot.robots.config import RobotConfig
    from lerobot.scripts import lerobot_teleoperate as _core_hardware_configs  # noqa: F401
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    _require_calibration(configuration, "robot")
    robot = draccus.decode(RobotConfig, _robot_payload(configuration, fps, cameras=cameras))
    teleop_payload = _teleoperator_payload(configuration) if include_teleoperator else None
    if teleop_payload:
        _require_calibration(configuration, "teleoperator")
    teleop = draccus.decode(TeleoperatorConfig, teleop_payload) if teleop_payload else None
    return robot, teleop


def _execute_teleoperation(payload: dict[str, Any]) -> None:
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate

    request = TeleoperationStartRequest.model_validate(payload)
    robot, teleop = _decode_hardware(_configuration(), request.fps, cameras=False)
    if teleop is None:
        raise ValueError("当前设备没有遥操作设备")
    teleoperate(
        TeleoperateConfig(
            robot=robot,
            teleop=teleop,
            fps=request.fps,
            display_data=False,
        )
    )


def _execute_recording(payload: dict[str, Any]) -> None:
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.scripts.lerobot_record import RecordConfig, record

    task_description = str(payload.pop("_task_description"))
    dataset_name = str(payload.pop("_dataset_name"))
    request = RecordingExecutionRequest.model_validate(payload)
    robot, teleop = _decode_hardware(_configuration(), request.fps)
    if teleop is None:
        raise ValueError("当前设备没有遥操作设备")
    record(
        RecordConfig(
            robot=robot,
            teleop=teleop,
            dataset=DatasetRecordConfig(
                repo_id=_repo_id(dataset_name),
                single_task=task_description,
                fps=request.fps,
                num_episodes=request.num_episodes,
                episode_time_s=request.episode_time_s,
                reset_time_s=request.reset_time_s,
                push_to_hub=False,
                streaming_encoding=True,
                encoder_threads=2,
                encoder_queue_maxsize=30,
                rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            ),
            display_data=False,
            play_sounds=False,
        )
    )


def inspect_policy_compatibility(request: PolicyInspectRequest) -> dict[str, Any]:
    """Inspect checkpoint metadata and compare it with configured hardware.

    This deliberately downloads only metadata/configuration files.  It never
    loads model weights, opens cameras, connects serial ports, or moves a robot.
    """
    from lerobot import policies as _policy_configs  # noqa: F401
    from lerobot.configs import FeatureType, PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    path = _policy_path(request.policy_path)
    policy = PreTrainedConfig.from_pretrained(path)
    configuration = _configuration()

    expected_visuals = {
        key for key, feature in policy.input_features.items() if feature.type == FeatureType.VISUAL
    }
    provided_visuals = _configured_visual_features(configuration)
    rename_map = _camera_rename_map(configuration, expected_visuals, provided_visuals)
    renamed_visuals = {rename_map.get(key, key) for key in provided_visuals}

    state_feature = policy.input_features.get("observation.state")
    action_feature = policy.output_features.get("action")
    padded_state_dim = state_feature.shape[-1] if state_feature and state_feature.shape else None
    state_dim = _normalizer_feature_dim(path, "observation.state") or padded_state_dim
    action_dim = action_feature.shape[-1] if action_feature and action_feature.shape else None
    hardware_state_dim, hardware_action_dim = _configured_vector_dimensions(configuration)

    issues: list[str] = []
    missing_visuals = sorted(expected_visuals - renamed_visuals)
    if missing_visuals:
        issues.append(f"缺少模型摄像头输入：{', '.join(missing_visuals)}")
    if hardware_state_dim is not None and state_dim is not None and hardware_state_dim != state_dim:
        issues.append(f"状态维度不匹配：模型 {state_dim}，设备 {hardware_state_dim}")
    if hardware_action_dim is not None and action_dim is not None and hardware_action_dim != action_dim:
        issues.append(f"动作维度不匹配：模型 {action_dim}，设备 {hardware_action_dim}")

    revision: str | None = None
    size_bytes: int | None = None
    if not os.path.exists(path):
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(path, files_metadata=True)
            revision = info.sha
            sizes = [sibling.size for sibling in info.siblings if sibling.size is not None]
            size_bytes = sum(sizes) if sizes else None
        except Exception:
            logging.info("Could not read optional Hub model metadata for %s", path, exc_info=True)

    return {
        "policy_path": path,
        "policy_type": policy.type,
        "revision": revision,
        "size_bytes": size_bytes,
        "state_dim": state_dim,
        "padded_state_dim": padded_state_dim if padded_state_dim != state_dim else None,
        "action_dim": action_dim,
        "hardware_state_dim": hardware_state_dim,
        "hardware_action_dim": hardware_action_dim,
        "expected_visuals": sorted(expected_visuals),
        "provided_visuals": sorted(provided_visuals),
        "rename_map": rename_map,
        "supports_rtc": policy.type in {"pi0", "pi05", "pi0_fast"}
        or bool(getattr(policy, "rtc_enabled", False)),
        "compatible": not issues,
        "issues": issues,
    }


def _execute_rollout(
    payload: dict[str, Any],
    *,
    preloaded_policy: Any | None = None,
    hardware_session: Any | None = None,
    keep_hardware_connected: bool = False,
) -> Any | None:
    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.rollout import context as rollout_context
    from lerobot.rollout.configs import (
        BaseStrategyConfig,
        DAggerPedalConfig,
        DAggerStrategyConfig,
        EpisodicDAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        RolloutConfig,
        SentryStrategyConfig,
    )
    from lerobot.scripts.lerobot_rollout import run_rollout

    request = RolloutStartRequest.model_validate(payload)
    configuration = _configuration()
    inspection = inspect_policy_compatibility(PolicyInspectRequest(policy_path=request.policy_path))
    if not inspection["compatible"]:
        raise ValueError("Policy 与当前设备不兼容：" + "；".join(inspection["issues"]))

    needs_teleop = request.strategy in {
        "episodic",
        "dagger_corrections",
        "dagger_continuous",
        "episodic_dagger",
    }
    robot, teleop = _decode_hardware(
        configuration,
        request.fps,
        include_teleoperator=needs_teleop,
    )
    hardware_key = (repr(robot), repr(teleop))
    connected_hardware = None
    if hardware_session is not None:
        if hardware_session.get("key") == hardware_key:
            connected_hardware = hardware_session.get("hardware")
        else:
            logging.info("Hardware configuration changed; releasing the previous resident session")
            _disconnect_hardware_session(hardware_session)
    policy_path = _policy_path(request.policy_path)
    policy = PreTrainedConfig.from_pretrained(policy_path)
    policy.pretrained_path = policy_path

    dataset = None
    if request.strategy != "base":
        dataset = DatasetRecordConfig(
            repo_id=_rollout_repo_id(request.dataset_name),
            single_task=request.task,
            fps=request.fps,
            num_episodes=request.num_episodes,
            episode_time_s=request.episode_time_s,
            reset_time_s=request.reset_time_s,
            push_to_hub=False,
            streaming_encoding=True,
            encoder_threads=2,
            encoder_queue_maxsize=30,
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        )

    pedal_device = os.environ.get("EVOMIND_DAGGER_PEDAL_DEVICE", "").strip()
    dagger_input = (
        {
            "input_device": "pedal",
            "pedal": DAggerPedalConfig(
                device_path=pedal_device,
                intervention=os.environ.get("EVOMIND_DAGGER_PEDAL_KEY", "*").strip() or "*",
            ),
        }
        if pedal_device
        else {}
    )

    strategy = {
        "base": BaseStrategyConfig(),
        "episodic": EpisodicStrategyConfig(),
        "sentry": SentryStrategyConfig(),
        "highlight": HighlightStrategyConfig(ring_buffer_seconds=request.ring_buffer_seconds),
        "dagger_corrections": DAggerStrategyConfig(
            num_episodes=request.num_episodes,
            record_autonomous=False,
            **dagger_input,
        ),
        "dagger_continuous": DAggerStrategyConfig(
            num_episodes=request.num_episodes,
            record_autonomous=True,
            **dagger_input,
        ),
        "episodic_dagger": EpisodicDAggerStrategyConfig(
            num_episodes=request.num_episodes,
            **dagger_input,
        ),
    }[request.strategy]
    inference = _rollout_inference_config(request.inference, policy)
    rollout_config = RolloutConfig(
        robot=robot,
        teleop=teleop if needs_teleop else None,
        policy=policy,
        strategy=strategy,
        inference=inference,
        dataset=dataset,
        fps=request.fps,
        duration=request.duration_s,
        task=request.task,
        rename_map=inspection["rename_map"],
        return_to_initial_position=request.return_to_initial_position,
        display_data=False,
        play_sounds=False,
    )
    original_loader = None
    if preloaded_policy is not None:
        original_loader = rollout_context._load_pretrained_policy
        rollout_context._load_pretrained_policy = lambda _config: preloaded_policy
    try:
        hardware = run_rollout(
            rollout_config,
            connected_hardware=connected_hardware,
            keep_hardware_connected=keep_hardware_connected,
        )
    finally:
        if original_loader is not None:
            rollout_context._load_pretrained_policy = original_loader
    if keep_hardware_connected and hardware is not None:
        return {"key": hardware_key, "hardware": hardware}
    return None


def _rollout_inference_config(backend: Literal["sync", "rtc"], policy_config: Any) -> Any:
    """Build the web rollout backend using Evo Studio's validated RTC continuity settings."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceConfig, SyncInferenceConfig

    if backend == "sync":
        return SyncInferenceConfig()

    chunk_size = int(getattr(policy_config, "chunk_size", 20))
    if chunk_size <= 0:
        raise ValueError(f"Policy chunk_size must be positive, got {chunk_size}")
    execution_horizon = int(getattr(policy_config, "rtc_execution_horizon", min(20, chunk_size)))
    max_guidance_weight = float(getattr(policy_config, "rtc_max_guidance_weight", 5.0))
    return RTCInferenceConfig(
        rtc=RTCConfig(
            execution_horizon=min(execution_horizon, chunk_size),
            max_guidance_weight=max_guidance_weight,
        ),
        queue_threshold=min(30, chunk_size - 1),
    )


def _dataset(dataset_id: str) -> dict[str, Any]:
    available = {item["id"]: item for item in datasets_inventory()}
    if dataset_id not in available:
        raise ValueError("选择的数据集不在本地数据目录中")
    return available[dataset_id]


def _execute_replay(payload: dict[str, Any]) -> None:
    from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay

    request = ReplayStartRequest.model_validate(payload)
    dataset = _dataset(request.dataset_id)
    configuration = _configuration()
    if dataset.get("robot_type") and dataset["robot_type"] != configuration.robot_type:
        raise ValueError(
            f"数据集设备类型 {dataset['robot_type']} 与当前设备 {configuration.robot_type} 不一致"
        )
    if request.episode >= dataset["episodes"]:
        raise ValueError("Episode 超出数据集范围")
    robot, _ = _decode_hardware(
        configuration,
        dataset["fps"],
        cameras=False,
        include_teleoperator=False,
    )
    replay(
        ReplayConfig(
            robot=robot,
            dataset=DatasetReplayConfig(
                repo_id=dataset["id"],
                episode=request.episode,
            ),
            play_sounds=False,
        )
    )


_EXECUTORS = {
    "teleoperation": _execute_teleoperation,
    "recording": _execute_recording,
    "rollout": _execute_rollout,
    "replay": _execute_replay,
}


def _run_workflow(
    operation: str,
    payload: dict[str, Any],
    event_queue: Any,
    command_queue: Any,
) -> None:
    from lerobot.utils.runtime_bridge import use_runtime_bridge

    bridge = ProcessRuntimeBridge(event_queue, command_queue)
    try:
        with use_runtime_bridge(bridge):
            _EXECUTORS[operation](payload)
    except BaseException as error:
        event_queue.put({"kind": "exit", "error": str(error) or error.__class__.__name__})
    else:
        event_queue.put({"kind": "exit", "error": ""})


def _load_resident_policy(policy_path: str) -> tuple[Any, dict[str, Any]]:
    """Load one local policy and keep it on its configured accelerator."""
    import torch

    from lerobot import policies as _policy_configs  # noqa: F401
    from lerobot.configs import PreTrainedConfig
    from lerobot.rollout.context import _load_pretrained_policy
    from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    configured_device = policy_config.device
    device = (
        configured_device
        if configured_device and is_torch_device_available(configured_device)
        else auto_select_torch_device().type
    )
    policy = _load_pretrained_policy(policy_config).to(device)
    policy.eval()
    warmup_details = _warmup_resident_policy(policy, policy_config)
    allocated_bytes = torch.cuda.memory_allocated() if str(device).startswith("cuda") else None
    return policy, {
        "policy_path": policy_path,
        "policy_type": policy_config.type,
        "device": str(device),
        "allocated_bytes": allocated_bytes,
        **warmup_details,
    }


def _warmup_resident_policy(policy: Any, policy_config: Any) -> dict[str, Any]:
    """Make a resident OpenPI JAX policy inference-ready before reporting it loaded.

    OpenPI compiles lazily on its first request.  Warming both the plain and RTC
    request paths here keeps that compilation out of the hardware rollout and
    guarantees that a successful preload has already produced valid actions.
    Dummy observations never reach the robot.
    """
    if getattr(policy_config, "type", None) != "openpi_jax":
        return {}

    import time

    import torch

    from lerobot.configs import FeatureType
    from lerobot.utils.constants import OBS_STATE

    batch: dict[str, Any] = {"task": [str(getattr(policy_config, "prompt", ""))]}
    for key, feature in policy_config.input_features.items():
        shape = tuple(feature.shape)
        if feature.type == FeatureType.VISUAL:
            batch[key] = torch.zeros((1, *shape), dtype=torch.uint8)
        elif key == OBS_STATE:
            batch[key] = torch.zeros((1, *shape), dtype=torch.float32)

    if OBS_STATE not in batch:
        raise ValueError(f"OpenPI JAX warmup requires {OBS_STATE}")

    started = time.perf_counter()
    actions = policy.predict_action_chunk(batch)
    warmup_inferences = 1

    if bool(getattr(policy_config, "rtc_enabled", False)):
        execution_horizon = max(1, int(getattr(policy_config, "rtc_execution_horizon", 1)))
        prefix = actions[0, :execution_horizon]
        policy.predict_action_chunk(
            batch,
            inference_delay=0,
            prev_chunk_left_over=prefix,
        )
        warmup_inferences += 1

    policy.reset()
    warmup_s = time.perf_counter() - started
    logging.info(
        "Resident OpenPI JAX policy warmed up (%d inference paths in %.2fs)",
        warmup_inferences,
        warmup_s,
    )
    server_metadata = getattr(getattr(policy, "_client", None), "metadata", {})
    model_load = server_metadata.get("evomind_load") if isinstance(server_metadata, dict) else None
    return {
        "inference_ready": True,
        "warmup_inferences": warmup_inferences,
        "warmup_s": round(warmup_s, 3),
        **({"model_load": dict(model_load)} if isinstance(model_load, dict) else {}),
    }


def _stop_external_policy_server(policy_path: str) -> None:
    """Stop the external server explicitly owned by a local policy config."""
    import json
    import subprocess

    config_path = Path(policy_path) / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "openpi_jax":
        return

    command = config.get("server_stop_command") or []
    if not command:
        logging.warning("OpenPI JAX policy has no server_stop_command: %s", policy_path)
        return
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("OpenPI JAX server_stop_command must be a non-empty string list")

    timeout_s = float(config.get("server_stop_timeout_s", 30.0))
    logging.info("Stopping resident OpenPI JAX server: %s", command)
    subprocess.run(command, check=True, timeout=timeout_s)


def _run_policy_resident(
    policy_path: str,
    event_queue: Any,
    control_queue: Any,
    command_queue: Any,
) -> None:
    """Own a GPU policy and an optional connected collection hardware session."""
    from lerobot.utils.runtime_bridge import use_runtime_bridge

    try:
        policy, details = _load_resident_policy(policy_path)
    except BaseException as error:
        event_queue.put({"kind": "resident_error", "error": str(error) or error.__class__.__name__})
        return
    event_queue.put({"kind": "resident_ready", "details": details})

    bridge = ProcessRuntimeBridge(event_queue, command_queue)
    hardware_session = None
    while True:
        message = control_queue.get()
        if message.get("kind") == "shutdown":
            _disconnect_hardware_session(hardware_session)
            del policy
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logging.info("Could not explicitly clear accelerator cache", exc_info=True)
            return
        if message.get("kind") == "release_hardware":
            _disconnect_hardware_session(hardware_session)
            hardware_session = None
            event_queue.put({"kind": "hardware_session", "ready": False})
            continue
        if message.get("kind") == "prepare_hardware":
            try:
                hardware_session = _prepare_hardware_session(
                    fps=int(message["fps"]),
                    include_teleoperator=bool(message["include_teleoperator"]),
                    hardware_session=hardware_session,
                )
            except BaseException as error:
                _disconnect_hardware_session(hardware_session)
                hardware_session = None
                event_queue.put(
                    {
                        "kind": "hardware_session",
                        "ready": False,
                        "error": str(error) or error.__class__.__name__,
                    }
                )
            else:
                event_queue.put({"kind": "hardware_session", "ready": True})
            continue
        if message.get("kind") != "rollout":
            continue
        keep_hardware = bool(message.get("keep_hardware"))
        if hardware_session is not None and not keep_hardware:
            _disconnect_hardware_session(hardware_session)
            hardware_session = None
        try:
            with use_runtime_bridge(bridge):
                hardware_session = _execute_rollout(
                    message["payload"],
                    preloaded_policy=policy,
                    hardware_session=hardware_session,
                    keep_hardware_connected=keep_hardware,
                )
        except BaseException as error:
            _disconnect_hardware_session(hardware_session)
            hardware_session = None
            event_queue.put({"kind": "hardware_session", "ready": False})
            event_queue.put({"kind": "job_exit", "error": str(error) or error.__class__.__name__})
        else:
            event_queue.put({"kind": "hardware_session", "ready": hardware_session is not None})
            event_queue.put({"kind": "job_exit", "error": ""})


def _release_collection_device(device: Any, *, preserve_pose: bool) -> None:
    """Release partial serial devices without enabling/disabling torque on failure."""
    if device is None:
        return
    if not preserve_pose:
        if device.is_connected:
            device.disconnect()
        return

    errors = []

    def release(call):
        try:
            call()
        except Exception as error:
            errors.append(error)
            logging.exception("Could not release part of a failed collection connection")

    arms = [getattr(device, name, None) for name in ("left_arm", "right_arm")]
    if any(arm is not None for arm in arms):
        for arm in arms:
            if arm is not None:
                release(lambda arm=arm: _release_collection_device(arm, preserve_pose=True))
    else:
        bus = getattr(device, "bus", None)
        if bus is not None:
            if bus.is_connected:
                release(lambda: bus.disconnect(disable_torque=False))
            for camera in getattr(device, "cameras", {}).values():
                if camera.is_connected:
                    release(camera.disconnect)
        elif device.is_connected:
            release(device.disconnect)
    if errors:
        raise errors[0]


def _disconnect_hardware_session(session: Any | None, *, failed_connection: bool = False) -> None:
    """Release a cached robot/camera/teleoperator session; safe to call repeatedly."""
    if session is None:
        return
    hardware = session.get("hardware") if isinstance(session, dict) else session
    if hardware is None:
        return
    robot = hardware.robot_wrapper.inner
    preserve_pose = failed_connection or getattr(robot, "motor_bus_recovery_failed", False) is True
    try:
        _release_collection_device(robot, preserve_pose=preserve_pose)
    finally:
        _release_collection_device(hardware.teleop, preserve_pose=preserve_pose)


def _prepare_hardware_session(
    *,
    fps: int,
    include_teleoperator: bool,
    hardware_session: Any | None,
) -> dict[str, Any]:
    """Connect collection devices before recording so first action startup is immediate."""
    from lerobot.robots import make_robot_from_config
    from lerobot.rollout.context import HardwareContext
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.teleoperators import make_teleoperator_from_config

    configuration = _configuration()
    robot_config, teleop_config = _decode_hardware(
        configuration,
        fps,
        include_teleoperator=include_teleoperator,
    )
    hardware_key = (repr(robot_config), repr(teleop_config))
    if hardware_session is not None and hardware_session.get("key") == hardware_key:
        hardware = hardware_session.get("hardware")
        robot = hardware.robot_wrapper.inner
        teleop = hardware.teleop
        robot_ready = bool(robot.is_connected)
        teleop_ready = teleop_config is None or bool(teleop and teleop.is_connected)
        if robot_ready and teleop_ready and getattr(robot, "motor_bus_recovery_failed", False) is not True:
            try:
                # Open handles alone do not prove the motor feedback still works.
                robot.get_observation()
            except (OSError, RuntimeError, ValueError):
                logging.exception("Cached collection hardware failed its feedback check; rebuilding")
                _disconnect_hardware_session(hardware_session, failed_connection=True)
                hardware_session = None
            else:
                logging.info("Collection hardware session is already prepared and feedback is available")
                return hardware_session

    _disconnect_hardware_session(hardware_session)
    logging.info("Preparing collection robot and cameras (%s)", robot_config.type)
    robot = make_robot_from_config(robot_config)
    teleop = None
    try:
        robot.connect()
        initial_obs = robot.get_observation()
        initial_position = {key: value for key, value in initial_obs.items() if key.endswith(".pos")}
        if teleop_config is not None:
            logging.info("Preparing collection teleoperator (%s)", teleop_config.type)
            teleop = make_teleoperator_from_config(teleop_config)
            teleop.connect()
    except BaseException:
        # Aggregate is_connected can be false while one arm, serial port, or
        # camera is already open. Always unwind the components we constructed.
        for device in (teleop, robot):
            try:
                _release_collection_device(device, preserve_pose=True)
            except Exception:
                logging.exception("Cleanup after collection prepare failed; retaining original connection error")
        raise

    hardware = HardwareContext(
        robot_wrapper=ThreadSafeRobot(robot),
        teleop=teleop,
        initial_position=initial_position,
        disconnect_on_teardown=False,
    )
    logging.info("Collection hardware session prepared")
    return {"key": hardware_key, "hardware": hardware}


class RuntimeService:
    """Own one native LeRobot workflow process at a time."""

    def __init__(
        self,
        events: EventBroker,
        jobs: JobManager,
        collection_store: CollectionStore | None = None,
    ) -> None:
        self._events = events
        self._jobs = jobs
        self._collection_store = collection_store
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.RLock()
        self._process: multiprocessing.Process | None = None
        self._event_queue: Any = None
        self._command_queue: Any = None
        self._job_id = ""
        self._operation: Operation | None = None
        self._latest: dict[str, Any] | None = None
        self._active_dataset_id: str | None = None
        self._tracked_collection = False
        self._using_resident = False
        self._resident_process: multiprocessing.Process | None = None
        self._resident_event_queue: Any = None
        self._resident_control_queue: Any = None
        self._resident_command_queue: Any = None
        self._resident_details: dict[str, Any] | None = None
        self._resident_loading_path: str | None = None
        self._resident_hardware_ready = False

    @property
    def active_dataset_id(self) -> str | None:
        with self._lock:
            return self._active_dataset_id if self._tracked_collection else None

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            return {
                "running": bool(self._operation and process and process.is_alive()),
                "job_id": self._job_id or None,
                "operation": self._operation.value if self._operation else None,
                "event": self._latest,
                "policy_residency": self.policy_residency(),
                "hardware_session": {
                    "state": "ready" if self._resident_hardware_ready else "empty",
                },
            }

    def policy_residency(self) -> dict[str, Any]:
        with self._lock:
            process = self._resident_process
            ready = bool(process and process.is_alive() and self._resident_details)
            loading = bool(process and process.is_alive() and self._resident_loading_path)
            return {
                "state": "ready" if ready else "loading" if loading else "empty",
                **({"policy_path": self._resident_loading_path} if loading else {}),
                **(self._resident_details or {}),
            }

    def preload_policy(self, request: PolicyPreloadRequest) -> dict[str, Any]:
        policy_path = require_local_policy(request.policy_path)
        with self._lock:
            if self._operation is not None:
                raise RuntimeError("请先结束当前运行任务")
            process = self._resident_process
            if (
                process
                and process.is_alive()
                and self._resident_details
                and self._resident_details.get("policy_path") == policy_path
            ):
                return self.policy_residency()
            if process and process.is_alive() and self._resident_loading_path:
                raise RuntimeError("模型正在预加载")
        self.unload_policy()

        event_queue = self._context.Queue()
        control_queue = self._context.Queue()
        command_queue = self._context.Queue()
        process = self._context.Process(
            target=_run_policy_resident,
            args=(policy_path, event_queue, control_queue, command_queue),
            name="evomind-policy-resident",
        )
        with self._lock:
            self._resident_process = process
            self._resident_event_queue = event_queue
            self._resident_control_queue = control_queue
            self._resident_command_queue = command_queue
            self._resident_details = None
            self._resident_loading_path = policy_path
        try:
            process.start()
        except Exception:
            self._discard_resident(terminate=False)
            raise
        try:
            item = event_queue.get(timeout=1800)
        except Empty as error:
            self._discard_resident(terminate=True)
            raise RuntimeError("模型预加载超时") from error
        if item.get("kind") != "resident_ready":
            message = str(item.get("error") or "模型预加载失败")
            process.join(timeout=5)
            self._discard_resident(terminate=False)
            raise RuntimeError(message)
        with self._lock:
            self._resident_details = item["details"]
            self._resident_loading_path = None
        return self.policy_residency()

    def unload_policy(self) -> dict[str, Any]:
        with self._lock:
            if self._operation is not None and self._using_resident:
                raise RuntimeError("请先结束正在使用该模型的任务")
            process = self._resident_process
            control_queue = self._resident_control_queue
            policy_path = (self._resident_details or {}).get("policy_path")
        if process and process.is_alive() and control_queue is not None:
            control_queue.put({"kind": "shutdown"})
            process.join(timeout=10)
        self._discard_resident(terminate=bool(process and process.is_alive()))
        if policy_path:
            _stop_external_policy_server(policy_path)
        return self.policy_residency()

    def release_hardware_session(self) -> dict[str, Any]:
        """Disconnect idle collection hardware while keeping the policy resident."""
        with self._lock:
            if self._operation is not None:
                raise RuntimeError("请先停止当前运行任务")
            process = self._resident_process
            control_queue = self._resident_control_queue
            event_queue = self._resident_event_queue
            ready = self._resident_hardware_ready
        if not ready:
            return {"state": "empty"}
        if not process or not process.is_alive() or control_queue is None or event_queue is None:
            with self._lock:
                self._resident_hardware_ready = False
            return {"state": "empty"}
        control_queue.put({"kind": "release_hardware"})
        try:
            item = event_queue.get(timeout=15)
        except Empty as error:
            raise RuntimeError("设备会话释放超时") from error
        if item.get("kind") != "hardware_session" or item.get("ready"):
            raise RuntimeError("设备会话释放失败")
        with self._lock:
            self._resident_hardware_ready = False
        return {"state": "empty"}

    def prepare_collection_hardware(self, request: CollectionStartRequest) -> dict[str, Any]:
        """Prepare the selected policy task's devices without starting inference or recording."""
        if self._collection_store is None:
            raise RuntimeError("采集进度账本未初始化")
        task = self._collection_store.require_today_task(request.task_id)
        if task["collection_method"] != "policy":
            raise ValueError("人工采集任务不需要预连接 Policy 设备")
        with self._lock:
            if self._operation is not None:
                raise RuntimeError("请先停止当前运行任务")
            process = self._resident_process
            control_queue = self._resident_control_queue
            event_queue = self._resident_event_queue
            resident_path = (self._resident_details or {}).get("policy_path")
        local_policy = require_local_policy(task["policy_path"])
        if not process or not process.is_alive() or resident_path != local_policy:
            raise RuntimeError("请先把当前任务的 Policy 预加载到显存")
        control_queue.put(
            {
                "kind": "prepare_hardware",
                "fps": task["fps"],
                "include_teleoperator": task["rollout_strategy"]
                in {"episodic", "dagger_corrections", "dagger_continuous", "episodic_dagger"},
            }
        )
        try:
            item = event_queue.get(timeout=60)
        except Empty as error:
            raise RuntimeError("采集设备准备超时") from error
        ready = item.get("kind") == "hardware_session" and bool(item.get("ready"))
        with self._lock:
            self._resident_hardware_ready = ready
        if not ready:
            raise RuntimeError(str(item.get("error") or "采集设备准备失败"))
        return {"state": "ready"}

    def close(self) -> None:
        """Release a resident model when the local console shuts down."""
        with self._lock:
            process = self._resident_process
            control_queue = self._resident_control_queue
            using_resident = self._using_resident
            policy_path = (self._resident_details or {}).get("policy_path")
        if process and process.is_alive() and control_queue is not None and not using_resident:
            control_queue.put({"kind": "shutdown"})
            process.join(timeout=10)
        self._discard_resident(terminate=bool(process and process.is_alive()))
        if policy_path and not using_resident:
            try:
                _stop_external_policy_server(policy_path)
            except Exception:
                logging.exception("Could not stop the resident policy server during shutdown")

    def _discard_resident(self, *, terminate: bool) -> None:
        with self._lock:
            process = self._resident_process
        if terminate and process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        with self._lock:
            self._resident_process = None
            self._resident_event_queue = None
            self._resident_control_queue = None
            self._resident_command_queue = None
            self._resident_details = None
            self._resident_loading_path = None
            self._resident_hardware_ready = False

    def start(self, operation: Operation, request: BaseModel) -> dict[str, Any]:
        if operation.value not in _EXECUTORS:
            raise ValueError(f"不支持的运行任务：{operation.value}")
        collection_task: dict[str, Any] | None = None
        if operation is Operation.RECORDING:
            if self._collection_store is None or not isinstance(request, RecordingStartRequest):
                raise RuntimeError("采集进度账本未初始化")
            collection_task = self._collection_store.require_today_task(request.task_id)
            if collection_task["collection_method"] != "manual":
                raise ValueError("该任务是 Policy 采集任务，请使用统一采集入口")
        if operation is Operation.ROLLOUT and isinstance(request, RolloutStartRequest):
            request = request.model_copy(update={"policy_path": require_local_policy(request.policy_path)})
        return self._start(operation, request, collection_task)

    def start_collection(self, request: CollectionStartRequest) -> dict[str, Any]:
        if self._collection_store is None:
            raise RuntimeError("采集进度账本未初始化")
        task = self._collection_store.require_today_task(request.task_id)
        if task["collection_method"] == "manual":
            return self._start(Operation.RECORDING, RecordingStartRequest(task_id=task["id"]), task)

        local_policy = require_local_policy(task["policy_path"])
        inspection = inspect_policy_compatibility(PolicyInspectRequest(policy_path=local_policy))
        if not inspection["compatible"]:
            raise ValueError("Policy 与当前设备不兼容：" + "；".join(inspection["issues"]))
        rollout_request = RolloutStartRequest(
            policy_path=local_policy,
            strategy=task["rollout_strategy"],
            inference=task["inference"],
            task=task["description"],
            dataset_name=_recording_dataset_name(task),
            fps=task["fps"],
            duration_s=task["duration_s"],
            num_episodes=task["num_episodes"],
            episode_time_s=task["episode_time_s"],
            reset_time_s=task["reset_time_s"],
            ring_buffer_seconds=task["ring_buffer_seconds"],
        )
        return self._start(Operation.ROLLOUT, rollout_request, task)

    def _start(
        self,
        operation: Operation,
        request: BaseModel,
        collection_task: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = request.model_dump()
        keep_hardware_session = bool(operation is Operation.ROLLOUT and collection_task is not None)
        if not keep_hardware_session:
            self.release_hardware_session()
        with self._lock:
            resident_path = (
                (self._resident_details or {}).get("policy_path")
                if self._resident_process and self._resident_process.is_alive()
                else None
            )
        if operation is Operation.ROLLOUT and resident_path and resident_path != payload.get("policy_path"):
            raise ValueError("显存中驻留的是另一个 Policy，请先卸载或预加载当前 Policy")
        if operation is Operation.RECORDING and collection_task is not None:
            payload.update(
                {
                    "fps": collection_task["fps"],
                    "num_episodes": collection_task["num_episodes"],
                    "episode_time_s": collection_task["episode_time_s"],
                    "reset_time_s": collection_task["reset_time_s"],
                }
            )
            payload["_task_description"] = collection_task["description"]
            payload["_dataset_name"] = _recording_dataset_name(collection_task)
        job = self._jobs.acquire(operation, f"正在启动 {operation.value}")
        if collection_task is not None:
            try:
                execution_request: RecordingExecutionRequest | RolloutStartRequest
                if operation is Operation.RECORDING:
                    execution_request = RecordingExecutionRequest.model_validate(
                        {key: value for key, value in payload.items() if not key.startswith("_")}
                    )
                    dataset_name = payload["_dataset_name"]
                else:
                    execution_request = RolloutStartRequest.model_validate(payload)
                    dataset_name = execution_request.dataset_name
                self._collection_store.start_session(
                    job.id,
                    collection_task["id"],
                    dataset_name,
                    execution_request,
                )
            except Exception:
                self._jobs.release(job.id, failed=True, message="采集任务启动失败")
                raise
        with self._lock:
            resident_process = self._resident_process
            resident_path = (self._resident_details or {}).get("policy_path")
            use_resident = bool(
                operation is Operation.ROLLOUT
                and resident_process
                and resident_process.is_alive()
                and resident_path == payload.get("policy_path")
            )
            if use_resident:
                event_queue = self._resident_event_queue
                command_queue = self._resident_command_queue
                control_queue = self._resident_control_queue
                process = resident_process
            else:
                event_queue = self._context.Queue()
                command_queue = self._context.Queue()
                control_queue = None
                process = self._context.Process(
                    target=_run_workflow,
                    args=(operation.value, payload, event_queue, command_queue),
                    name=f"evomind-{operation.value}",
                )
        with self._lock:
            self._process = process
            self._event_queue = event_queue
            self._command_queue = command_queue
            self._job_id = job.id
            self._operation = operation
            self._tracked_collection = collection_task is not None
            self._using_resident = use_resident
            self._latest = self._events.latest.as_dict()
        try:
            if use_resident:
                control_queue.put(
                    {
                        "kind": "rollout",
                        "payload": payload,
                        "keep_hardware": keep_hardware_session,
                    }
                )
            else:
                process.start()
        except Exception:
            self._clear()
            if collection_task is not None and self._collection_store is not None:
                self._collection_store.finish_session(job.id, failed=True, error="运行任务启动失败")
            self._jobs.release(job.id, failed=True, message="运行任务启动失败")
            raise
        threading.Thread(target=self._monitor, name=f"monitor-{operation.value}", daemon=True).start()
        return self.status()

    def command(self, command: str) -> dict[str, Any]:
        with self._lock:
            process = self._process
            command_queue = self._command_queue
            operation = self._operation
        if process is None or not process.is_alive() or command_queue is None:
            raise RuntimeError("当前没有运行中的任务")
        recording_commands = {"finish_episode", "rerecord_episode"}
        rollout_commands = recording_commands | {"pause_resume", "correction", "toggle_highlight"}
        if command != "stop" and operation is Operation.RECORDING and command not in recording_commands:
            raise ValueError("当前采集任务不支持这个操作")
        if command != "stop" and operation is Operation.ROLLOUT and command not in rollout_commands:
            raise ValueError("当前 Rollout 不支持这个操作")
        if command != "stop" and operation not in {Operation.RECORDING, Operation.ROLLOUT}:
            raise ValueError("当前任务不支持这个操作")
        command_queue.put(command)
        if command == "stop" and operation is Operation.ROLLOUT:
            os.kill(process.pid, signal.SIGINT)
        return self.status()

    def _monitor(self) -> None:
        error = ""
        while True:
            with self._lock:
                process = self._process
                event_queue = self._event_queue
                job_id = self._job_id
                operation = self._operation
                tracked_collection = self._tracked_collection
                using_resident = self._using_resident
            if process is None or event_queue is None or operation is None:
                return
            try:
                item = event_queue.get(timeout=0.2)
            except Empty:
                if process.is_alive():
                    continue
                if using_resident:
                    error = "模型驻留进程意外退出"
                break
            if item["kind"] in {"exit", "job_exit"}:
                error = item["error"]
                break
            if item["kind"] == "hardware_session":
                with self._lock:
                    self._resident_hardware_ready = bool(item.get("ready"))
                continue
            phase = Phase(item["phase"])
            message = _MESSAGES.get((item["operation"], item["phase"]), item["phase"])
            if tracked_collection and operation is Operation.ROLLOUT:
                message = {
                    "starting": "正在加载采集 Policy",
                    "connecting": "正在连接 Policy 采集设备",
                    "running": "Policy 数据采集中",
                    "stopping": "正在停止 Policy 采集",
                    "completed": "Policy 采集已结束",
                }.get(item["phase"], message)
            recovery = item["data"].get("hardware_recovery")
            if recovery == "retrying":
                message = f"设备断联，正在重连（{item['data']['attempt']}/{item['data']['max_attempts']}）"
            elif recovery == "recovered":
                message = "连接已恢复，已丢弃断联帧并清空旧动作"
            elif recovery == "failed":
                message = "设备重连失败，本次运行已停止，断联帧未保存"
            event = self._events.publish(
                Operation(item["operation"]),
                phase,
                message,
                job_id=job_id,
                data=item["data"],
            )
            if tracked_collection and self._collection_store is not None:
                try:
                    data = item["data"]
                    if data.get("repo_id"):
                        with self._lock:
                            self._active_dataset_id = str(data["repo_id"])
                    self._collection_store.update_session_repo_id(job_id, data.get("repo_id"))
                    if data.get("stage") == "episode_saved":
                        self._collection_store.save_episode(job_id, data)
                except (CollectionStoreError, KeyError, TypeError, ValueError, sqlite3.Error):
                    logging.exception("Failed to persist collection progress")
            with self._lock:
                self._latest = event.as_dict()
        if not using_resident:
            process.join(timeout=1)
        if error:
            event = self._events.publish(
                operation,
                Phase.FAILED,
                error,
                job_id=job_id,
            )
            with self._lock:
                self._latest = event.as_dict()
        if tracked_collection and self._collection_store is not None:
            self._collection_store.finish_session(job_id, failed=bool(error), error=error)
        self._clear()
        self._jobs.release(
            job_id,
            failed=bool(error),
            message=error or _MESSAGES.get((operation.value, "completed"), "任务已完成"),
        )

    def _clear(self) -> None:
        with self._lock:
            self._process = None
            self._event_queue = None
            self._command_queue = None
            self._job_id = ""
            self._operation = None
            self._active_dataset_id = None
            self._tracked_collection = False
            self._using_resident = False


__all__ = [
    "HardwareBusyError",
    "CollectionStartRequest",
    "PolicyInspectRequest",
    "PolicyPreloadRequest",
    "RecordingStartRequest",
    "ReplayStartRequest",
    "RolloutStartRequest",
    "RuntimeCommandRequest",
    "RuntimeService",
    "TeleoperationStartRequest",
    "inspect_policy_compatibility",
    "require_local_policy",
]
