"""Local hardware identification sessions used by the device wizard."""

from __future__ import annotations

import base64
import threading
import time
from typing import Any

from evomind_lerobot.discovery import hardware_inventory
from evomind_lerobot.hardware_motion import HardwareMotionSession, PiperMotionSession

_motion_session: HardwareMotionSession | PiperMotionSession | None = None
_motion_lock = threading.Lock()


def _serial_candidates(excluded_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "stable_id": device["id"],
            "path": device["path"],
            "device": device["device"],
            "bus_type": "",
            "motor_ids": [],
            "delta": 0,
            "moved": False,
            "motion_error": "",
        }
        for device in hardware_inventory()["serial"]
        if device["id"] not in excluded_ids
    ]


def _socketcan_candidates(excluded_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "stable_id": device["id"],
            "path": device["interface"],
            "device": device["interface"],
            "bus_type": "socketcan",
            "motor_ids": [],
            "delta": 0,
            "moved": False,
            "motion_error": "",
        }
        for device in hardware_inventory()["socketcan"]
        if device["id"] not in excluded_ids
    ]


def start_motion_identification(model: str, excluded_ids: set[str]) -> dict[str, Any]:
    global _motion_session
    with _motion_lock:
        if _motion_session is not None:
            _motion_session.stop()
        session = (
            PiperMotionSession(
                _socketcan_candidates(excluded_ids),
                release_motors=not excluded_ids,
            )
            if model.strip().lower().replace("-", "").replace("_", "") == "piperx"
            else HardwareMotionSession(_serial_candidates(excluded_ids), model)
        )
        readable_count = session.start()
        _motion_session = session
        return {
            "status": "watching",
            "readable_count": readable_count,
            "ports": session.payloads(),
        }


def poll_motion_identification() -> dict[str, Any]:
    with _motion_lock:
        if _motion_session is None:
            raise RuntimeError("机械臂识别尚未开始")
        return {"ports": _motion_session.poll()}


def stop_motion_identification() -> dict[str, str]:
    global _motion_session
    with _motion_lock:
        if _motion_session is not None:
            _motion_session.stop()
        _motion_session = None
    return {"status": "stopped"}


def _read_camera_frame(capture: Any, timeout_seconds: float = 1.5) -> Any | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        frame_ok, frame = capture.read()
        if frame_ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def camera_previews() -> list[dict[str, Any]]:
    import cv2

    previews = []
    for camera in hardware_inventory()["cameras"]:
        preview_data_url = None
        preview_error = "摄像头已识别，但无法读取画面"
        for capture_path in camera.get("capture_paths", [camera["path"]]):
            capture = cv2.VideoCapture(capture_path)
            try:
                if not capture.isOpened():
                    continue
                frame = _read_camera_frame(capture)
                if frame is None:
                    continue
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    continue
                preview_data_url = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                )
                preview_error = None
                break
            finally:
                capture.release()
        previews.append(
            {
                "id": camera["id"],
                "name": camera["name"],
                "path": camera["path"],
                "driver": camera["driver"],
                "serial_number": camera["serial_number"],
                "paths": camera["paths"],
                "preview_data_url": preview_data_url,
                "preview_error": preview_error,
            }
        )
    return previews
