"""Configure classic 1 Mbit/s SocketCAN interfaces for PiperX."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from .common import resolve_piper_can_interface

CAN_BITRATE = 1_000_000


def can_interfaces() -> list[str]:
    interfaces = []
    for path in sorted(Path("/sys/class/net").iterdir()):
        try:
            if (path / "type").read_text().strip() == "280":
                interfaces.append(path.name)
        except OSError:
            continue
    return interfaces


def _ip(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["ip", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"ip {' '.join(arguments)} failed: {detail}")
    return result


def interface_status(interface: str) -> tuple[bool, int | None]:
    result = _ip("-details", "link", "show", interface)
    flags_match = re.search(r"<([^>]+)>", result.stdout)
    flags = set(flags_match.group(1).split(",")) if flags_match else set()
    bitrate_match = re.search(r"\bbitrate\s+(\d+)", result.stdout)
    bitrate = int(bitrate_match.group(1)) if bitrate_match else None
    return "UP" in flags, bitrate


def configure(interface: str) -> None:
    _ip("link", "set", interface, "down", check=False)
    _ip("link", "set", interface, "type", "can", "bitrate", str(CAN_BITRATE))
    _ip("link", "set", interface, "up")
    up, bitrate = interface_status(interface)
    if not up or bitrate != CAN_BITRATE:
        raise RuntimeError(
            f"SocketCAN {interface} did not become ready: up={up}, bitrate={bitrate}"
        )


def ensure_can_interface_ready(interface: str) -> str:
    up, bitrate = interface_status(interface)
    if not up or bitrate != CAN_BITRATE:
        configure(interface)
    return interface


def ensure_piper_can_ready(serial_number: str) -> str:
    interface = resolve_piper_can_interface(serial_number)
    return ensure_can_interface_ready(interface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("interfaces", nargs="*", help="默认配置检测到的全部 SocketCAN 接口")
    arguments = parser.parse_args()
    selected = arguments.interfaces or can_interfaces()
    if not selected:
        raise SystemExit("未发现 SocketCAN 接口")
    for interface in selected:
        ensure_can_interface_ready(interface)
        print(f"configured {interface}: classic CAN {CAN_BITRATE} bit/s")
