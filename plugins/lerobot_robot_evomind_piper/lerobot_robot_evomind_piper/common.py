"""PiPER SDK helpers ported from Evo-RL."""

from __future__ import annotations

import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)
PIPER_ROLE_LEADER = 0xFA
PIPER_ROLE_FOLLOWER = 0xFC
_SYS_CLASS_NET = Path("/sys/class/net")


def resolve_piper_can_interface(serial_number: str) -> str:
    """Return the SocketCAN interface for a USB-CAN adapter serial number."""
    for interface_path in _SYS_CLASS_NET.iterdir():
        try:
            if (interface_path / "type").read_text().strip() != "280":
                continue
        except OSError:
            continue
        # Some USB hubs do not propagate ID_SERIAL_SHORT into the net device's
        # udev properties. The USB interface parent still exposes the adapter
        # serial directly through sysfs.
        try:
            usb_serial_path = (interface_path / "device").resolve().parent / "serial"
            usb_serial = usb_serial_path.read_text().strip()
        except OSError:
            usb_serial = ""
        if usb_serial == serial_number:
            return interface_path.name
        try:
            result = subprocess.run(  # nosec B607
                ["udevadm", "info", "--query=property", f"--path={interface_path}"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("`udevadm` is required to find the PiPER USB-CAN adapter.") from exc
        if f"ID_SERIAL_SHORT={serial_number}" in result.stdout.splitlines():
            return interface_path.name
    raise RuntimeError(f"No PiPER USB-CAN adapter found with serial '{serial_number}'.")


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `piper_sdk`. Install the PiperX plugin dependencies first."
        ) from exc


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    confirmed_reads = 0
    while time.monotonic() < deadline:
        # EnablePiper() reads cached status before sending its command. Send the
        # command first and require two fresh all-enabled reads so a USB reconnect
        # cannot make stale SDK state look like a successful enable.
        arm.EnableArm(7)
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(interval_s, remaining_s))
        enabled = list(arm.GetArmEnableStatus())
        confirmed_reads = confirmed_reads + 1 if enabled and all(enabled) else 0
        if confirmed_reads >= 2:
            return True
    return False


def set_piper_role(arm: Any, role: int) -> None:
    arm.MasterSlaveConfig(role, 0x00, 0x00, 0x00)
