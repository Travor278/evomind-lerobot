"""Identify a serial-connected arm by watching its motor positions."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_BAUDRATE = 1_000_000
DEFAULT_MOTOR_IDS = list(range(1, 7))
MOTION_THRESHOLD = 50
PIPER_MOTION_THRESHOLD = 1000

_FEETECH_POS_ADDR = 56
_DYNAMIXEL_POS_ADDR = 132


class PortProber(Protocol):
    protocol: str
    label: str

    def probe(self, port_path: str, baudrate: int, motor_ids: list[int]) -> list[int]: ...

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int
    ) -> dict[int, int]: ...


@contextmanager
def _suppress_stderr():
    saved_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)


class FeetechProber:
    protocol = "feetech"
    label = "Feetech"

    def probe(self, port_path: str, baudrate: int, motor_ids: list[int]) -> list[int]:
        import scservo_sdk as scs

        handler = scs.PortHandler(port_path)
        try:
            if not handler.openPort():
                return []
        except OSError:
            return []
        try:
            handler.setBaudRate(baudrate)
            packet = scs.PacketHandler(0)
            return [
                motor_id
                for motor_id in motor_ids
                if packet.read2ByteTxRx(handler, motor_id, _FEETECH_POS_ADDR)[1]
                == scs.COMM_SUCCESS
            ]
        finally:
            handler.closePort()

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int
    ) -> dict[int, int]:
        import scservo_sdk as scs

        handler = scs.PortHandler(port_path)
        if not handler.openPort():
            return {}
        try:
            handler.setBaudRate(baudrate)
            packet = scs.PacketHandler(0)
            positions = {}
            for motor_id in motor_ids:
                value, result, _ = packet.read2ByteTxRx(handler, motor_id, _FEETECH_POS_ADDR)
                if result == scs.COMM_SUCCESS:
                    positions[motor_id] = int(value)
            return positions
        finally:
            handler.closePort()


class DynamixelProber:
    protocol = "dynamixel"
    label = "Dynamixel"

    def probe(self, port_path: str, baudrate: int, motor_ids: list[int]) -> list[int]:
        import dynamixel_sdk as dxl

        handler = dxl.PortHandler(port_path)
        try:
            if not handler.openPort():
                return []
        except OSError:
            return []
        try:
            handler.setBaudRate(baudrate)
            packet = dxl.PacketHandler(2.0)
            return [
                motor_id
                for motor_id in motor_ids
                if packet.read4ByteTxRx(handler, motor_id, _DYNAMIXEL_POS_ADDR)[1]
                == dxl.COMM_SUCCESS
            ]
        finally:
            handler.closePort()

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int
    ) -> dict[int, int]:
        import dynamixel_sdk as dxl

        handler = dxl.PortHandler(port_path)
        if not handler.openPort():
            return {}
        try:
            handler.setBaudRate(baudrate)
            packet = dxl.PacketHandler(2.0)
            positions = {}
            for motor_id in motor_ids:
                value, result, _ = packet.read4ByteTxRx(handler, motor_id, _DYNAMIXEL_POS_ADDR)
                if result == dxl.COMM_SUCCESS:
                    positions[motor_id] = int(value)
            return positions
        finally:
            handler.closePort()


_PROBERS: dict[str, type[PortProber]] = {
    "feetech": FeetechProber,
    "dynamixel": DynamixelProber,
}

_MODEL_PROBE_PROTOCOLS = {
    "so101": "feetech",
    "so100": "feetech",
    "biso": "feetech",
    "koch": "dynamixel",
    "omx": "dynamixel",
}


def _probe_protocol(model: str) -> str:
    model_key = model.strip().lower().replace("-", "").replace("_", "")
    try:
        return _MODEL_PROBE_PROTOCOLS[model_key]
    except KeyError as error:
        raise ValueError(f"当前本体不支持自动串口识别：{model}") from error


def detect_motion(baseline: dict[int, int], current: dict[int, int]) -> int:
    return sum(
        abs(current[motor_id] - base_value)
        for motor_id, base_value in baseline.items()
        if motor_id in current
    )


def _result_id(result: dict[str, Any]) -> str:
    return str(result.get("stable_id") or "")


def resolve_active_motion(
    results: list[dict[str, Any]], active_id: str = ""
) -> tuple[list[dict[str, Any]], str]:
    moved = [result for result in results if result.get("moved")]
    strongest = max(moved, key=lambda item: int(item.get("delta", 0)), default=None)
    next_active_id = _result_id(strongest) if strongest else ""
    if not next_active_id and any(_result_id(item) == active_id for item in results):
        next_active_id = active_id
    normalized = [
        {**result, "moved": _result_id(result) == next_active_id} for result in results
    ]
    return normalized, next_active_id


@dataclass
class MotionDetector:
    path: str
    prober: PortProber
    motor_ids: list[int]
    baudrate: int = DEFAULT_BAUDRATE
    baseline: dict[int, int] = field(default_factory=dict)
    last_positions: dict[int, int] = field(default_factory=dict)

    def capture_baseline(self) -> dict[int, int]:
        with _suppress_stderr():
            self.baseline = self.prober.read_positions(
                self.path, self.motor_ids, self.baudrate
            )
        self.last_positions = dict(self.baseline)
        return dict(self.baseline)

    def poll(self) -> dict[str, Any]:
        with _suppress_stderr():
            current = self.prober.read_positions(self.path, self.motor_ids, self.baudrate)
        reference = self.last_positions or self.baseline
        delta = detect_motion(reference, current)
        self.last_positions = dict(current)
        return {"delta": delta, "moved": delta > MOTION_THRESHOLD}


class HardwareMotionSession:
    def __init__(self, candidates: list[dict[str, Any]], model: str) -> None:
        self._candidates = [dict(candidate) for candidate in candidates]
        self._model = model
        self._detectors: dict[str, MotionDetector] = {}
        self._active_id = ""

    def start(self) -> int:
        for candidate in self._candidates:
            self._prepare_candidate(candidate)
        return len(self._detectors)

    def stop(self) -> None:
        self._detectors = {}
        self._active_id = ""

    def poll(self) -> list[dict[str, Any]]:
        results = []
        for candidate in self._candidates:
            stable_id = str(candidate["stable_id"])
            detector = self._detectors.get(stable_id)
            result = self._payload(candidate)
            if detector is not None:
                try:
                    result.update(detector.poll())
                except (OSError, RuntimeError) as error:
                    result["motion_error"] = f"读取电机失败：{error}"
            results.append(result)
        normalized, self._active_id = resolve_active_motion(results, self._active_id)
        return normalized

    def payloads(self) -> list[dict[str, Any]]:
        return [self._payload(candidate) for candidate in self._candidates]

    def _prepare_candidate(self, candidate: dict[str, Any]) -> None:
        path = str(candidate.get("path") or "")
        if not path or not os.path.exists(path):
            candidate["motion_error"] = "串口路径不可用"
            return
        if not os.access(path, os.R_OK | os.W_OK):
            candidate["motion_error"] = "串口不可读写"
            return
        protocol = _probe_protocol(self._model)
        prober = _PROBERS[protocol]()
        try:
            with _suppress_stderr():
                found_ids = prober.probe(path, DEFAULT_BAUDRATE, DEFAULT_MOTOR_IDS)
            if not found_ids:
                candidate["motion_error"] = "未能读取电机位置"
                return
            detector = MotionDetector(path, prober, found_ids)
            baseline = detector.capture_baseline()
            if not baseline:
                candidate["motion_error"] = "未能读取电机位置"
                return
            detector.motor_ids = sorted(baseline)
            candidate.update({"bus_type": protocol, "motor_ids": detector.motor_ids})
            self._detectors[str(candidate["stable_id"])] = detector
        except ImportError:
            candidate["motion_error"] = f"缺少 {prober.label} SDK"
        except (OSError, RuntimeError) as error:
            candidate["motion_error"] = f"读取电机失败：{error}"

    @staticmethod
    def _payload(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "stable_id": candidate.get("stable_id", ""),
            "device": candidate.get("device", ""),
            "bus_type": candidate.get("bus_type", ""),
            "motor_ids": candidate.get("motor_ids", []),
            "delta": candidate.get("delta", 0),
            "moved": candidate.get("moved", False),
            "motion_error": candidate.get("motion_error", ""),
        }


class PiperMotionDetector:
    """Watch Piper feedback without changing role or sending motion targets."""

    def __init__(self, interface: str) -> None:
        self.interface = interface
        self.arm: Any | None = None
        self.last_positions: dict[str, dict[int, int]] = {}

    def start(self, *, release_motors: bool, timeout_s: float = 0.4) -> None:
        from lerobot_robot_evomind_piper.can_setup import ensure_can_interface_ready
        from piper_sdk import C_PiperInterface_V2, LogLevel

        interface = ensure_can_interface_ready(self.interface)
        self.arm = C_PiperInterface_V2(
            can_name=interface,
            judge_flag=False,
            can_auto_init=True,
            logger_level=LogLevel.WARNING,
        )
        self.arm.ConnectPort(can_init=True, piper_init=False)
        if release_motors:
            self._disable_motors()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            positions = self._read_positions()
            if positions:
                self.last_positions = positions
                return
            time.sleep(0.02)

        # Some PiperX roles remain silent on CAN while stationary. Keep the
        # passive listener eligible so later movement can identify the arm.
        self.last_positions = {}

    def _disable_motors(self, timeout_s: float = 0.8) -> None:
        """Release all joints before asking the operator to move the arm."""
        status = self.arm.GetArmLowSpdInfoMsgs()
        previous_timestamp = float(getattr(status, "time_stamp", 0.0))
        self.arm.DisableArm(7)

        deadline = time.monotonic() + timeout_s
        saw_updated_status = False
        while time.monotonic() < deadline:
            status = self.arm.GetArmLowSpdInfoMsgs()
            timestamp = float(getattr(status, "time_stamp", 0.0))
            if timestamp > previous_timestamp:
                saw_updated_status = True
                if not any(self.arm.GetArmEnableStatus()):
                    return
            time.sleep(0.02)
        if saw_updated_status:
            raise RuntimeError("PiperX 电机失能未确认")

    def _read_positions(self) -> dict[str, dict[int, int]]:
        feedback = self.arm.GetArmJointMsgs()
        control = self.arm.GetArmJointCtrl()
        positions = {}
        for source, message, state in (
            ("feedback", feedback, feedback.joint_state),
            ("control", control, control.joint_ctrl),
        ):
            if message.time_stamp > 0:
                positions[source] = {
                    index: int(getattr(state, f"joint_{index}"))
                    for index in DEFAULT_MOTOR_IDS
                }
        return positions

    def poll(self) -> dict[str, Any]:
        current = self._read_positions()
        delta = max(
            (
                detect_motion(self.last_positions.get(source, {}), positions)
                for source, positions in current.items()
            ),
            default=0,
        )
        for source, positions in current.items():
            self.last_positions[source] = positions
        return {"delta": delta, "moved": delta > PIPER_MOTION_THRESHOLD}

    def stop(self) -> None:
        if self.arm is not None:
            self.arm.DisconnectPort()
            self.arm = None


class PiperMotionSession:
    def __init__(
        self, candidates: list[dict[str, Any]], *, release_motors: bool
    ) -> None:
        self._candidates = [dict(candidate) for candidate in candidates]
        self._release_motors = release_motors
        self._detectors: dict[str, PiperMotionDetector] = {}
        self._active_id = ""

    def start(self) -> int:
        for candidate in self._candidates:
            stable_id = str(candidate["stable_id"])
            detector = PiperMotionDetector(str(candidate["device"]))
            try:
                detector.start(release_motors=self._release_motors)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                candidate["motion_error"] = f"读取 PiperX 失败：{error}"
                with suppress(Exception):
                    detector.stop()
                continue
            candidate.update({"bus_type": "socketcan", "motor_ids": DEFAULT_MOTOR_IDS})
            self._detectors[stable_id] = detector
        return len(self._detectors)

    def stop(self) -> None:
        for detector in self._detectors.values():
            with suppress(Exception):
                detector.stop()
        self._detectors = {}
        self._active_id = ""

    def poll(self) -> list[dict[str, Any]]:
        results = []
        for candidate in self._candidates:
            result = HardwareMotionSession._payload(candidate)
            detector = self._detectors.get(str(candidate["stable_id"]))
            if detector is not None:
                try:
                    result.update(detector.poll())
                except (OSError, RuntimeError, ValueError) as error:
                    result["motion_error"] = f"读取 PiperX 失败：{error}"
            results.append(result)
        normalized, self._active_id = resolve_active_motion(results, self._active_id)
        return normalized

    def payloads(self) -> list[dict[str, Any]]:
        return [HardwareMotionSession._payload(candidate) for candidate in self._candidates]
