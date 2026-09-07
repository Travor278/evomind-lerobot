"""Read-only local hardware inventory."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _serial_devices() -> list[dict[str, str]]:
    root = Path("/dev/serial/by-id")
    if not root.exists():
        return []
    return [
        {
            "id": path.name,
            "path": str(path),
            "device": str(path.resolve()),
        }
        for path in sorted(root.iterdir())
        if path.is_symlink()
    ]


def _socketcan_devices() -> list[dict[str, Any]]:
    devices = []
    network_root = Path("/sys/class/net")
    if not network_root.exists():
        return devices
    for path in sorted(network_root.iterdir()):
        try:
            if (path / "type").read_text().strip() != "280":
                continue
        except OSError:
            continue
        properties = subprocess.run(
            ["udevadm", "info", "--query=property", f"--path={path}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        serial_number = next(
            (
                line.removeprefix("ID_SERIAL_SHORT=")
                for line in properties
                if line.startswith("ID_SERIAL_SHORT=")
            ),
            "",
        )
        device_path = next(
            (
                line.removeprefix("ID_PATH=")
                for line in properties
                if line.startswith("ID_PATH=")
            ),
            "",
        )
        # Identification must include every SocketCAN interface.  Prefer the
        # adapter serial for a reboot-stable identity, then the physical USB
        # path, and only fall back to the current interface name when udev has
        # neither property (for example, a virtual CAN interface).
        stable_id = serial_number or device_path or path.name
        detail = subprocess.run(
            ["ip", "-details", "link", "show", path.name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        bitrate_match = re.search(r"\bbitrate\s+(\d+)", detail)
        flags_match = re.search(r"<([^>]+)>", detail)
        flags = set(flags_match.group(1).split(",")) if flags_match else set()
        devices.append(
            {
                "id": stable_id,
                "serial_number": serial_number,
                "interface": path.name,
                "state": (path / "operstate").read_text().strip(),
                "up": "UP" in flags,
                "bitrate": int(bitrate_match.group(1)) if bitrate_match else None,
            }
        )
    return devices


def _video_devices() -> list[dict[str, str]]:
    devices = []
    for path in sorted(Path("/dev").glob("video*")):
        name_path = Path("/sys/class/video4linux") / path.name / "name"
        name = name_path.read_text().strip() if name_path.exists() else path.name
        devices.append({"id": path.name, "path": str(path), "name": name})
    return devices


def _video_aliases() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    aliases: list[dict[str, list[str]]] = []
    for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        grouped: dict[str, list[str]] = {}
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_symlink():
                    grouped.setdefault(str(path.resolve()), []).append(str(path))
        aliases.append(grouped)
    return aliases[0], aliases[1]


def _usb_device_parent(path: Path) -> Path:
    return next(
        (
            candidate
            for candidate in (path, *path.parents)
            if (candidate / "idVendor").exists() and (candidate / "idProduct").exists()
        ),
        path,
    )


def _realsense_serials_by_usb_device() -> dict[str, str]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return {}

    serials = {}
    try:
        devices = rs.context().query_devices()
    except RuntimeError:
        return serials
    for device in devices:
        if not (
            device.supports(rs.camera_info.physical_port)
            and device.supports(rs.camera_info.serial_number)
        ):
            continue
        physical_port = Path(device.get_info(rs.camera_info.physical_port))
        usb_device = _usb_device_parent(physical_port)
        serials[str(usb_device)] = device.get_info(rs.camera_info.serial_number)
    return serials


def _camera_capture_path(
    resolved_video: str,
    by_id: dict[str, list[str]],
    by_path: dict[str, list[str]],
) -> tuple[int, str]:
    id_aliases = by_id.get(resolved_video, [])
    rgb_alias = next((alias for alias in id_aliases if "-rgb-video-index0" in alias), None)
    if rgb_alias:
        return 0, rgb_alias
    index_zero_alias = next((alias for alias in id_aliases if alias.endswith("-video-index0")), None)
    if index_zero_alias:
        return 1, index_zero_alias
    path_index_zero = next(
        (alias for alias in by_path.get(resolved_video, []) if alias.endswith("-video-index0")),
        None,
    )
    if path_index_zero:
        return 2, path_index_zero
    return 3, resolved_video


def _camera_devices(video_devices: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id, by_path = _video_aliases()
    cameras: dict[str, dict[str, Any]] = {}
    for video in video_devices:
        device_path = (Path("/sys/class/video4linux") / video["id"] / "device").resolve()
        # A single physical camera may expose video nodes from several USB
        # interfaces (for example, RealSense depth and RGB endpoints). Group
        # those interfaces by their nearest USB device parent instead of
        # presenting them as separate cameras.
        usb_device = _usb_device_parent(device_path)
        key = str(usb_device)
        resolved_video = str(Path(video["path"]).resolve())
        score, capture_path = _camera_capture_path(resolved_video, by_id, by_path)
        camera = cameras.setdefault(
            key,
            {
                "name": video["name"],
                "serial_number": (usb_device / "serial").read_text().strip()
                if (usb_device / "serial").exists()
                else "",
                "paths": [],
                "candidates": [],
            },
        )
        camera["paths"].append(video["path"])
        camera["candidates"].append((score, capture_path, resolved_video))

    realsense_serials = _realsense_serials_by_usb_device()
    result = []
    for usb_device, camera in cameras.items():
        candidates = sorted(camera.pop("candidates"), key=lambda item: (item[0], item[1]))
        selected_path = candidates[0][1]
        capture_paths = list(
            dict.fromkeys(
                [candidate[1] for candidate in candidates]
                + [candidate[2] for candidate in candidates]
            )
        )
        driver = "intelrealsense" if "realsense" in camera["name"].lower() else "opencv"
        serial_number = camera.pop("serial_number")
        if driver == "intelrealsense":
            serial_number = realsense_serials.get(usb_device, "")
        result.append(
            {
                "id": selected_path,
                "name": camera["name"],
                "path": selected_path,
                "driver": driver,
                "serial_number": serial_number,
                "paths": camera["paths"],
                "capture_paths": capture_paths,
            }
        )
    return result


def hardware_inventory() -> dict[str, Any]:
    video_devices = _video_devices()
    return {
        "serial": _serial_devices(),
        "socketcan": _socketcan_devices(),
        "video": video_devices,
        "cameras": _camera_devices(video_devices),
        "platform": os.uname().sysname,
        "hostname": os.environ.get("EVOMIND_WORKSTATION_NAME") or os.uname().nodename,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.splitlines()[0].strip() or None


def runtime_inventory() -> dict[str, Any]:
    data_root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    existing_root = data_root if data_root.exists() else Path.home()
    disk = shutil.disk_usage(existing_root)
    return {
        "hostname": os.environ.get("EVOMIND_WORKSTATION_NAME") or os.uname().nodename,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "lerobot_version": _package_version("lerobot"),
        "torch_version": _package_version("torch"),
        "gpu": _gpu_name(),
        "data_root": str(data_root),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
    }
