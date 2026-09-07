"""PiperX diagnostics and conservative maintenance controls."""

from __future__ import annotations

import time
from contextlib import suppress
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from evomind_lerobot.device_config import DeviceConfiguration, load_device_configuration
from evomind_lerobot.discovery import hardware_inventory

PIPER_DEVICE_TYPES = {
    "piperx_follower",
    "piperx_leader",
    "bi_piperx_follower",
    "bi_piperx_leader",
}
PIPER_JOINT_LIMITS = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}
MAX_MAINTENANCE_STEP_DEGREES = 5.0


class PiperScanRequest(BaseModel):
    device_id: str = Field(min_length=1)


class PiperActionRequest(PiperScanRequest):
    action: Literal["enable", "disable", "move"]
    motor_id: int = Field(default=7, ge=1, le=7)
    value: float | None = None
    confirmed: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> PiperActionRequest:
        if self.action in {"enable", "move"} and not self.confirmed:
            raise ValueError("This PiperX action requires confirmation")
        if self.action == "move":
            if self.motor_id not in PIPER_JOINT_LIMITS or self.value is None:
                raise ValueError("Joint movement requires a joint number and target angle")
            minimum, maximum = PIPER_JOINT_LIMITS[self.motor_id]
            if not minimum <= self.value <= maximum:
                raise ValueError("Joint target exceeds the PiperX limit")
        return self


def supports_piper_maintenance(configuration: DeviceConfiguration) -> bool:
    configured_types = {configuration.robot_type, configuration.teleoperator_type}
    return bool(configured_types & PIPER_DEVICE_TYPES)


def _require_piper_configuration() -> None:
    configuration = load_device_configuration()
    if configuration is None:
        raise ValueError("请先配置设备")
    if not supports_piper_maintenance(configuration):
        raise ValueError("当前设备没有 PiperX 维修工具")


def _can_device(device_id: str) -> dict[str, Any]:
    device = next(
        (item for item in hardware_inventory()["socketcan"] if item["id"] == device_id),
        None,
    )
    if device is None:
        raise ValueError("SocketCAN device is no longer connected")
    from lerobot_robot_evomind_piper.can_setup import ensure_can_interface_ready

    ensure_can_interface_ready(str(device["interface"]))
    refreshed = next(
        (
            item
            for item in hardware_inventory()["socketcan"]
            if item["id"] == device_id
        ),
        None,
    )
    if refreshed is None:
        raise ValueError("SocketCAN device is no longer connected")
    if not refreshed["up"] or refreshed["bitrate"] != 1_000_000:
        raise ValueError("这个 CAN 接口无法以 1 Mbit/s 启用")
    return refreshed


def _number(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _timestamp(message: Any) -> float:
    try:
        return float(getattr(message, "time_stamp", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _decimal(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _faults(status: Any) -> list[str]:
    labels = {
        "voltage_too_low": "电压过低",
        "motor_overheating": "电机过温",
        "driver_overcurrent": "驱动器过流",
        "driver_overheating": "驱动器过温",
        "collision_status": "碰撞保护",
        "driver_error_status": "驱动器异常",
        "stall_status": "堵转保护",
    }
    return [label for field, label in labels.items() if bool(getattr(status, field, False))]


class PiperMaintenanceSession:
    def __init__(self, device: dict[str, Any]) -> None:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        self.device = dict(device)
        self.arm = C_PiperInterface_V2(
            can_name=str(device["interface"]),
            judge_flag=False,
            can_auto_init=True,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
            logger_level=LogLevel.WARNING,
        )

    def connect(self, timeout_s: float = 0.8) -> None:
        self.arm.ConnectPort(can_init=True, piper_init=False)
        self.arm.SearchPiperFirmwareVersion()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._has_feedback():
                return
            time.sleep(0.02)

    def close(self) -> None:
        self.arm.DisconnectPort()

    def _has_feedback(self) -> bool:
        return any(
            _timestamp(message) > 0
            for message in (
                self.arm.GetArmJointMsgs(),
                self.arm.GetArmJointCtrl(),
                self.arm.GetArmLowSpdInfoMsgs(),
                self.arm.GetArmStatus(),
            )
        )

    def snapshot(self) -> dict[str, Any]:
        joint_feedback = self.arm.GetArmJointMsgs()
        joint_control = self.arm.GetArmJointCtrl()
        if _timestamp(joint_feedback) > 0:
            joint_state = joint_feedback.joint_state
            feedback_source = "feedback"
        elif _timestamp(joint_control) > 0:
            joint_state = joint_control.joint_ctrl
            feedback_source = "control"
        else:
            joint_state = None
            feedback_source = "none"

        low_speed = self.arm.GetArmLowSpdInfoMsgs()
        enabled = list(self.arm.GetArmEnableStatus())
        motors = []
        for motor_id in range(1, 7):
            telemetry = getattr(low_speed, f"motor_{motor_id}", None)
            foc_status = getattr(telemetry, "foc_status", None)
            motors.append(
                {
                    "id": motor_id,
                    "position": (
                        _number(getattr(joint_state, f"joint_{motor_id}", 0)) * 1e-3
                        if joint_state is not None
                        else None
                    ),
                    "voltage": _number(getattr(telemetry, "vol", 0)) * 0.1,
                    "driver_temperature": _number(getattr(telemetry, "foc_temp", 0)),
                    "motor_temperature": _number(getattr(telemetry, "motor_temp", 0)),
                    "current": _number(getattr(telemetry, "bus_current", 0)) * 0.001,
                    "enabled": bool(
                        getattr(foc_status, "driver_enable_status", False)
                        or (motor_id <= len(enabled) and enabled[motor_id - 1])
                    ),
                    "faults": _faults(foc_status),
                }
            )

        arm_message = self.arm.GetArmStatus()
        arm_status = getattr(arm_message, "arm_status", None)
        gripper_feedback = self.arm.GetArmGripperMsgs()
        gripper_control = self.arm.GetArmGripperCtrl()
        if _timestamp(gripper_feedback) > 0:
            gripper = gripper_feedback.gripper_state
        elif _timestamp(gripper_control) > 0:
            gripper = gripper_control.gripper_ctrl
        else:
            gripper = None

        firmware = self.arm.GetPiperFirmwareVersion()
        if not isinstance(firmware, str):
            firmware = ""
        return {
            "device_id": self.device["id"],
            "interface": self.device["interface"],
            "firmware": firmware,
            "feedback_source": feedback_source,
            "can_fps": _decimal(self.arm.GetCanFps()),
            "status": {
                "available": _timestamp(arm_message) > 0,
                "ctrl_mode": _number(getattr(arm_status, "ctrl_mode", 0)),
                "arm_status": _number(getattr(arm_status, "arm_status", 0)),
                "mode": _number(getattr(arm_status, "mode_feed", 0)),
                "error_code": _number(getattr(arm_status, "err_code", 0)),
            },
            "motors": motors,
            "gripper": {
                "available": gripper is not None,
                "position": abs(_number(getattr(gripper, "grippers_angle", 0))) * 1e-3,
                "effort": _number(getattr(gripper, "grippers_effort", 0)) * 1e-3,
            },
        }

    def action(self, request: PiperActionRequest) -> dict[str, Any]:
        if request.action == "enable":
            self.arm.EnableArm(request.motor_id)
        elif request.action == "disable":
            self.arm.DisableArm(request.motor_id)
        else:
            current = self.snapshot()
            if current["feedback_source"] != "feedback":
                raise ValueError("只有收到从臂状态反馈时才能调整关节")
            if not all(motor["enabled"] for motor in current["motors"]):
                raise ValueError("调整关节前需要使能全部关节")
            positions = [motor["position"] for motor in current["motors"]]
            if any(position is None for position in positions):
                raise ValueError("未收到完整的关节位置")
            target = float(request.value)
            current_position = float(positions[request.motor_id - 1])
            if abs(target - current_position) > MAX_MAINTENANCE_STEP_DEGREES:
                raise ValueError("单次关节调整不能超过 5°")
            positions[request.motor_id - 1] = target
            self.arm.ModeCtrl(0x01, 0x01, 10, 0x00)
            self.arm.JointCtrl(*(round(float(position) * 1000) for position in positions))
        time.sleep(0.15)
        return self.snapshot()


_SESSION_LOCK = Lock()
_SESSION: PiperMaintenanceSession | None = None


def scan_piper(request: PiperScanRequest) -> dict[str, Any]:
    global _SESSION
    _require_piper_configuration()
    device = _can_device(request.device_id)
    with _SESSION_LOCK:
        if _SESSION is not None:
            with suppress(Exception):
                _SESSION.close()
        session = PiperMaintenanceSession(device)
        try:
            session.connect()
            result = session.snapshot()
        except Exception:
            with suppress(Exception):
                session.close()
            _SESSION = None
            raise
        _SESSION = session
        return result


def snapshot_piper(request: PiperScanRequest) -> dict[str, Any]:
    _require_piper_configuration()
    with _SESSION_LOCK:
        if _SESSION is None or _SESSION.device["id"] != request.device_id:
            raise ValueError("请先读取这个 PiperX 控制器")
        return _SESSION.snapshot()


def run_piper_action(request: PiperActionRequest) -> dict[str, Any]:
    _require_piper_configuration()
    with _SESSION_LOCK:
        if _SESSION is None or _SESSION.device["id"] != request.device_id:
            raise ValueError("请先读取这个 PiperX 控制器")
        return _SESSION.action(request)


def close_piper_session() -> None:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            with suppress(Exception):
                _SESSION.close()
        _SESSION = None
