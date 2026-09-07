"""Persistent local device declaration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SerialBinding(StrictModel):
    id: str
    port: str
    alias: str = Field(min_length=1)
    kind: Literal["robot", "teleoperator"]
    side: Literal["single", "left", "right"]


class CanBinding(StrictModel):
    id: str
    alias: str = Field(min_length=1)
    kind: Literal["robot", "teleoperator"]
    side: Literal["single", "left", "right"]


class CameraBinding(StrictModel):
    id: str
    port: str
    alias: str = Field(min_length=1)
    side: Literal["single", "left", "right"]
    driver: Literal["opencv", "intelrealsense"]
    serial_number: str | None

    @model_validator(mode="after")
    def validate_driver_identity(self) -> CameraBinding:
        if self.driver == "intelrealsense" and not self.serial_number:
            raise ValueError("RealSense camera bindings require a serial number")
        return self


class CameraSlot(StrictModel):
    alias: str = Field(min_length=1)
    kind: Literal["wrist", "environment"]
    side: Literal["single", "left", "right"]


class DeviceConfiguration(StrictModel):
    profile_id: str = Field(min_length=1)
    robot_type: str = Field(min_length=1)
    teleoperator_type: str | None = None
    camera_slots: list[CameraSlot] = Field(default_factory=list)
    serial_bindings: list[SerialBinding] = Field(default_factory=list)
    can_bindings: list[CanBinding] = Field(default_factory=list)
    camera_bindings: list[CameraBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bindings(self) -> DeviceConfiguration:
        aliases = [binding.alias for binding in self.serial_bindings]
        aliases.extend(binding.alias for binding in self.can_bindings)
        aliases.extend(binding.alias for binding in self.camera_bindings)
        if len(aliases) != len(set(aliases)):
            raise ValueError("Device aliases must be unique")
        serial_ids = [binding.id for binding in self.serial_bindings]
        can_ids = [binding.id for binding in self.can_bindings]
        camera_ids = [binding.id for binding in self.camera_bindings]
        if (
            len(serial_ids) != len(set(serial_ids))
            or len(can_ids) != len(set(can_ids))
            or len(camera_ids) != len(set(camera_ids))
        ):
            raise ValueError("A physical device can only be bound once")
        slot_aliases = [slot.alias for slot in self.camera_slots]
        if len(slot_aliases) != len(set(slot_aliases)):
            raise ValueError("Camera slot aliases must be unique")
        if self.camera_bindings and set(slot_aliases) != {binding.alias for binding in self.camera_bindings}:
            raise ValueError("Camera bindings must match the declared camera slots")
        if (
            any(
                binding.kind == "teleoperator"
                for binding in [*self.serial_bindings, *self.can_bindings]
            )
            and self.teleoperator_type is None
        ):
            raise ValueError("A teleoperator type is required for teleoperator bindings")
        for kind in ("robot", "teleoperator"):
            kind_bindings = [
                binding
                for binding in [*self.serial_bindings, *self.can_bindings]
                if binding.kind == kind
            ]
            if len(kind_bindings) == 2 and {binding.side for binding in kind_bindings} != {"left", "right"}:
                raise ValueError(f"Dual {kind} bindings must contain one left and one right device")
            if len(kind_bindings) == 1 and kind_bindings[0].side != "single":
                raise ValueError(f"Single {kind} binding must use side=single")
        return self


def device_type(configuration: DeviceConfiguration, binding: SerialBinding) -> str:
    if binding.kind == "robot":
        return configuration.robot_type
    if configuration.teleoperator_type is None:
        raise ValueError("遥操作设备类型未配置")
    return configuration.teleoperator_type


def calibration_id(configuration: DeviceConfiguration, binding: SerialBinding) -> str:
    sides = {
        item.side
        for item in configuration.serial_bindings
        if item.kind == binding.kind
    }
    suffix = f"_{binding.side}" if {"left", "right"} <= sides else ""
    return f"{runtime_id(configuration, binding.kind)}{suffix}"


def runtime_id(configuration: DeviceConfiguration, kind: str) -> str:
    identities = sorted(
        binding.id
        for binding in [*configuration.serial_bindings, *configuration.can_bindings]
        if binding.kind == kind
    )
    if not identities:
        raise ValueError(f"没有已绑定的 {kind} 设备")
    digest = hashlib.sha256("\n".join(identities).encode()).hexdigest()[:10]
    return f"{configuration.profile_id}-{digest}"


def calibration_directory(configuration: DeviceConfiguration, binding: SerialBinding) -> Path:
    category = ROBOTS if binding.kind == "robot" else TELEOPERATORS
    configured_type = device_type(configuration, binding)
    native_type = {
        "so100_follower": "so_follower",
        "so101_follower": "so_follower",
        "bi_so_follower": "so_follower",
        "so100_leader": "so_leader",
        "so101_leader": "so_leader",
        "bi_so_leader": "so_leader",
    }.get(configured_type, configured_type)
    return HF_LEROBOT_CALIBRATION / category / native_type


def calibration_path(configuration: DeviceConfiguration, binding: SerialBinding) -> Path:
    return calibration_directory(configuration, binding) / f"{calibration_id(configuration, binding)}.json"


def _configuration_path() -> Path:
    configured = os.environ.get("EVOMIND_LEROBOT_DEVICE_CONFIG")
    if configured:
        return Path(configured)
    return Path.home() / ".config/evomind-lerobot/device.json"


def load_device_configuration() -> DeviceConfiguration | None:
    path = _configuration_path()
    if not path.exists():
        return None
    return DeviceConfiguration.model_validate_json(path.read_text(encoding="utf-8"))


def save_device_configuration(configuration: DeviceConfiguration) -> None:
    path = _configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(configuration.model_dump(), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise
