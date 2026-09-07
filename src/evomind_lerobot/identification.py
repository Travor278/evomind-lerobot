"""Local hardware identification sessions used by the device wizard."""

from __future__ import annotations

import base64
import multiprocessing as mp
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


def _read_camera_frame(capture: Any, timeout_seconds: float = 3.0) -> Any | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        frame_ok, frame = capture.read()
        if frame_ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def _encode_preview(frame: Any) -> str | None:
    import cv2

    encoded_ok, encoded = cv2.imencode(".jpg", frame)
    if not encoded_ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _opencv_preview(camera: dict[str, Any]) -> str | None:
    import cv2

    # Opening sibling V4L nodes can select depth/metadata endpoints and reset a
    # marginal USB hub. Use only the selected RGB endpoint and MJPEG bandwidth.
    capture = cv2.VideoCapture(camera["path"], cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = _read_camera_frame(capture)
        return _encode_preview(frame) if frame is not None else None
    finally:
        capture.release()


def _realsense_preview(camera: dict[str, Any]) -> str | None:
    import numpy as np
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if camera.get("serial_number"):
        config.enable_device(camera["serial_number"])
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    try:
        pipeline.start(config)
        started = True
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return _encode_preview(np.asanyarray(color_frame.get_data()))
    finally:
        if started:
            pipeline.stop()


def _camera_preview_worker(camera: dict[str, Any], connection: Any) -> None:
    try:
        preview = (
            _realsense_preview(camera) if camera["driver"] == "intelrealsense" else _opencv_preview(camera)
        )
        connection.send(preview)
    except Exception:
        connection.send(None)
    finally:
        connection.close()


def _camera_preview_with_timeout(camera: dict[str, Any], timeout_seconds: float = 6.0) -> str | None:
    # Kernel-level V4L2 reads can ignore thread deadlines after a disconnect.
    # A short-lived process gives the web service a reliable upper bound.
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_camera_preview_worker, args=(camera, child), daemon=True)
    process.start()
    child.close()
    try:
        return parent.recv() if parent.poll(timeout_seconds) else None
    except (EOFError, OSError):
        return None
    finally:
        parent.close()
        process.join(timeout=0.2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def camera_previews() -> list[dict[str, Any]]:
    previews = []
    for camera in hardware_inventory()["cameras"]:
        preview_data_url = None
        preview_error = "摄像头已识别，但无法读取画面"
        preview_data_url = _camera_preview_with_timeout(camera)
        if preview_data_url is not None:
            preview_error = None
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
