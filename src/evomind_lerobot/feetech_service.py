"""Feetech servo diagnostics backed by LeRobot's motor bus."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from evomind_lerobot.device_config import DeviceConfiguration, load_device_configuration
from evomind_lerobot.discovery import hardware_inventory
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import MODEL_BAUDRATE_TABLE, MODEL_NUMBER_TABLE
from lerobot.motors.motors_bus import Motor, MotorNormMode

DEFAULT_BAUDRATE = 1_000_000
SUPPORTED_BAUDRATES = tuple(MODEL_BAUDRATE_TABLE["sts3215"])
_MODEL_BY_NUMBER = {number: model for model, number in MODEL_NUMBER_TABLE.items()}
_BUS_LOCK = Lock()
FEETECH_DEVICE_TYPES = {
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "bi_so_follower",
    "bi_so_leader",
}


class FeetechScanRequest(BaseModel):
    device_id: str = Field(min_length=1)
    baudrate: int = DEFAULT_BAUDRATE


class FeetechMotorTarget(BaseModel):
    id: int = Field(ge=0, le=253)
    model: str = Field(min_length=1)


class FeetechSnapshotRequest(FeetechScanRequest):
    motors: list[FeetechMotorTarget] = Field(min_length=1)


class FeetechActionRequest(FeetechScanRequest):
    motor_id: int = Field(ge=0, le=253)
    action: Literal["torque_enable", "torque_disable", "move", "set_id", "set_baudrate"]
    value: int | None = None
    confirmed: bool = False

    @model_validator(mode="after")
    def validate_action(self) -> FeetechActionRequest:
        if self.action in {"move", "set_id", "set_baudrate"} and not self.confirmed:
            raise ValueError("This action requires confirmation")
        if self.action == "move" and (self.value is None or not 0 <= self.value <= 4095):
            raise ValueError("Position must be between 0 and 4095")
        if self.action == "set_id" and (self.value is None or not 0 <= self.value <= 253):
            raise ValueError("Motor ID must be between 0 and 253")
        if self.action == "set_baudrate" and self.value not in SUPPORTED_BAUDRATES:
            raise ValueError("Unsupported baudrate")
        return self


def supports_feetech_maintenance(configuration: DeviceConfiguration) -> bool:
    configured_types = {configuration.robot_type, configuration.teleoperator_type}
    return bool(configured_types & FEETECH_DEVICE_TYPES)


def _require_feetech_configuration() -> None:
    configuration = load_device_configuration()
    if configuration is None:
        raise ValueError("请先配置设备")
    if not supports_feetech_maintenance(configuration):
        raise ValueError("当前设备没有飞特舵机维修工具")


def _device_path(device_id: str) -> str:
    serial_devices = hardware_inventory()["serial"]
    device = next((item for item in serial_devices if item["id"] == device_id), None)
    if device is None:
        raise ValueError("Serial device is no longer connected")
    return str(device["path"])


def _motors(found: dict[int, int]) -> dict[str, Motor]:
    motors: dict[str, Motor] = {}
    for motor_id, model_number in found.items():
        model = _MODEL_BY_NUMBER.get(model_number)
        if model:
            motors[f"motor_{motor_id}"] = Motor(motor_id, model, MotorNormMode.RANGE_M100_100)
    return motors


def _target_motors(targets: list[FeetechMotorTarget]) -> dict[str, Motor]:
    motors = {
        f"motor_{target.id}": Motor(target.id, target.model, MotorNormMode.RANGE_M100_100)
        for target in targets
    }
    if len(motors) != len(targets):
        raise ValueError("Motor IDs must be unique")
    unsupported = [motor.model for motor in motors.values() if motor.model not in MODEL_NUMBER_TABLE]
    if unsupported:
        raise ValueError(f"Unsupported Feetech motor model: {unsupported[0]}")
    return motors


@contextmanager
def _open_bus(device_id: str, baudrate: int) -> Iterator[FeetechMotorsBus]:
    if baudrate not in SUPPORTED_BAUDRATES:
        raise ValueError("Unsupported baudrate")
    path = _device_path(device_id)
    bus = FeetechMotorsBus(path, {})
    bus.connect(handshake=False)
    try:
        bus.set_baudrate(baudrate)
        yield bus
    finally:
        bus.disconnect(disable_torque=False)


def _read_motor(bus: FeetechMotorsBus, name: str, motor: Motor, model_number: int) -> dict[str, Any]:
    def read(register: str) -> int:
        return int(bus.read(register, name, normalize=False, num_retry=1))

    return {
        "id": motor.id,
        "model": motor.model,
        "model_number": model_number,
        "position": read("Present_Position"),
        "velocity": read("Present_Velocity"),
        "load": read("Present_Load"),
        "voltage": read("Present_Voltage") / 10,
        "temperature": read("Present_Temperature"),
        "current": read("Present_Current") if motor.model != "scs0009" else None,
        "torque_enabled": bool(read("Torque_Enable")),
        "moving": bool(read("Moving")),
    }


def scan_feetech(request: FeetechScanRequest) -> dict[str, Any]:
    _require_feetech_configuration()
    with _BUS_LOCK:
        with _open_bus(request.device_id, request.baudrate) as probe:
            found = probe.broadcast_ping(num_retry=1, raise_on_error=True) or {}
            path = probe.port
        motors = _motors(found)
        if not motors:
            return {"device_id": request.device_id, "baudrate": request.baudrate, "motors": []}

        bus = FeetechMotorsBus(path, motors)
        bus.connect(handshake=False)
        try:
            bus.set_baudrate(request.baudrate)
            states = [_read_motor(bus, name, motor, found[motor.id]) for name, motor in motors.items()]
        finally:
            bus.disconnect(disable_torque=False)
        return {"device_id": request.device_id, "baudrate": request.baudrate, "motors": states}


def snapshot_feetech(request: FeetechSnapshotRequest) -> dict[str, Any]:
    _require_feetech_configuration()
    motors = _target_motors(request.motors)
    path = _device_path(request.device_id)
    with _BUS_LOCK:
        bus = FeetechMotorsBus(path, motors)
        bus.connect(handshake=False)
        try:
            bus.set_baudrate(request.baudrate)
            positions = bus.sync_read("Present_Position", normalize=False, num_retry=1)
        finally:
            bus.disconnect(disable_torque=False)
    return {
        "positions": [{"id": motor.id, "position": int(positions[name])} for name, motor in motors.items()]
    }


def run_feetech_action(request: FeetechActionRequest) -> dict[str, Any]:
    _require_feetech_configuration()
    with _BUS_LOCK:
        with _open_bus(request.device_id, request.baudrate) as probe:
            found = probe.broadcast_ping(num_retry=1, raise_on_error=True) or {}
            path = probe.port
        model_number = found.get(request.motor_id)
        model = _MODEL_BY_NUMBER.get(model_number) if model_number is not None else None
        if model is None:
            raise ValueError(f"Motor {request.motor_id} was not found or is unsupported")
        assert model_number is not None
        if request.action == "set_id":
            assert request.value is not None
            _validate_id_change(found, request.motor_id, request.value)

        name = f"motor_{request.motor_id}"
        motor = Motor(request.motor_id, model, MotorNormMode.RANGE_M100_100)
        bus = FeetechMotorsBus(path, {name: motor})
        bus.connect(handshake=False)
        try:
            bus.set_baudrate(request.baudrate)
            _apply_action(bus, name, request)
        finally:
            bus.disconnect(disable_torque=False)

        if request.action == "set_id":
            assert request.value is not None
            with _open_bus(request.device_id, request.baudrate) as probe:
                verified = probe.broadcast_ping(num_retry=2, raise_on_error=True) or {}
            _verify_id_change(verified, request.motor_id, request.value, model_number)
            return {
                "status": "completed",
                "rescan_required": True,
                "old_id": request.motor_id,
                "new_id": request.value,
                "model": model,
            }

    if request.action == "set_baudrate":
        return {"status": "completed", "rescan_required": True}
    return scan_feetech(FeetechScanRequest(device_id=request.device_id, baudrate=request.baudrate))


def _validate_id_change(found: dict[int, int], old_id: int, new_id: int) -> None:
    if new_id == old_id:
        raise ValueError("新 ID 必须与当前 ID 不同")
    if new_id in found:
        raise ValueError(f"ID {new_id} 已被总线上的其他舵机使用")


def _verify_id_change(found: dict[int, int], old_id: int, new_id: int, model_number: int) -> None:
    if found.get(new_id) != model_number:
        raise RuntimeError(f"ID 写入后未检测到新 ID {new_id}，请重新扫描总线")
    if old_id in found:
        raise RuntimeError(f"ID 写入后旧 ID {old_id} 仍然存在，请检查是否连接了重复 ID 的舵机")


def _apply_action(bus: FeetechMotorsBus, name: str, request: FeetechActionRequest) -> None:
    if request.action == "torque_enable":
        bus.enable_torque(name, num_retry=1)
    elif request.action == "torque_disable":
        bus.disable_torque(name, num_retry=1)
    elif request.action == "move":
        bus.write("Goal_Position", name, request.value, normalize=False, num_retry=1)
    elif request.action == "set_id":
        assert request.value is not None
        bus.disable_torque(name, num_retry=1)
        bus.write("ID", name, request.value, normalize=False, num_retry=1)
    elif request.action == "set_baudrate":
        model = bus.motors[name].model
        bus.disable_torque(name, num_retry=1)
        encoded = MODEL_BAUDRATE_TABLE[model][request.value]
        bus.write("Baud_Rate", name, encoded, normalize=False, num_retry=1)
