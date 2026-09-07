"""In-process calibration helpers for the local Evomind runtime.

This module intentionally exposes library functions instead of terminal-oriented
``input()`` flows. The app backend owns the HTTP/UI state machine and calls these
helpers from a worker thread.

The SO-101 automatic probing implementation is carried forward from the
Apache-2.0 licensed MINT-SJTU/RoboClaw calibration workflow used by the
archived local application.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any

from lerobot.utils.runtime_bridge import RuntimeBridge, use_runtime_bridge

logger = logging.getLogger(__name__)

SO101_DEVICE_TYPES = {
    "so101_follower",
    "so101_leader",
    "bi_so_follower",
    "bi_so_leader",
}


@dataclass(frozen=True)
class CalibrationDeviceConfig:
    arm_type: str
    role: str
    port: str
    calibration_dir: Path | str
    calibration_id: str


@dataclass(frozen=True)
class CalibrationResult:
    calibration_path: str
    profile: dict[str, dict[str, int]]


class CalibrationStoppedError(RuntimeError):
    """Raised when the local app asks a calibration run to stop."""


def run_so101_auto_calibration(
    config: CalibrationDeviceConfig,
    *,
    stop_event: Event | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> CalibrationResult:
    """Run the SO-101 automatic probing calibration and persist its profile JSON."""

    if config.arm_type not in SO101_DEVICE_TYPES:
        raise RuntimeError(f"Automatic calibration only supports SO-101, got {config.arm_type}.")
    calibration_dir = Path(config.calibration_dir).expanduser()
    calibration_dir.mkdir(parents=True, exist_ok=True)
    path = calibration_dir / f"{config.calibration_id}.json"

    def emit(phase: str, message: str, **extra: Any) -> None:
        if on_event:
            on_event({
                "phase": phase,
                "message": message,
                "calibration_id": config.calibration_id,
                "calibration_dir": str(calibration_dir),
                "path": str(path),
                **extra,
            })

    emit("preparing", "正在准备 SO-101 自动校准。")
    calibrator = _SO101AutoCalibrator(config.port, stop_event=stop_event, on_event=emit)
    profile = calibrator.run()
    _save_profile(config, profile)
    if not path.is_file():
        raise RuntimeError(f"Calibration did not save {path.name}.")
    saved_profile = _read_profile(path)
    emit("done", "自动校准已完成。", profile=saved_profile)
    return CalibrationResult(
        calibration_path=str(path),
        profile=saved_profile,
    )


def run_native_manual_calibration(
    config: CalibrationDeviceConfig,
    runtime_bridge: RuntimeBridge,
) -> CalibrationResult:
    """Run LeRobot's native interactive calibration through the embedded runtime bridge."""

    from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate

    calibration_dir = Path(config.calibration_dir).expanduser()
    calibration_dir.mkdir(parents=True, exist_ok=True)
    path = calibration_dir / f"{config.calibration_id}.json"
    device_config = _native_device_config(config)
    workflow = (
        CalibrateConfig(teleop=device_config)
        if config.role == "leader"
        else CalibrateConfig(robot=device_config)
    )
    with use_runtime_bridge(runtime_bridge):
        calibrate(workflow)
    if not path.is_file():
        raise RuntimeError(f"Calibration did not save {path.name}.")
    return CalibrationResult(calibration_path=str(path), profile=_read_profile(path))


def _native_device_config(config: CalibrationDeviceConfig) -> Any:
    calibration_dir = Path(config.calibration_dir).expanduser()
    if config.role == "leader":
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

        return SOLeaderTeleopConfig(
            id=config.calibration_id,
            calibration_dir=calibration_dir,
            port=config.port,
        )
    if config.role == "follower":
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        return SOFollowerRobotConfig(
            id=config.calibration_id,
            calibration_dir=calibration_dir,
            port=config.port,
            cameras={},
        )
    raise RuntimeError(f"Unsupported calibration role: {config.role}")


def _make_device(config: CalibrationDeviceConfig) -> Any:
    if config.role == "leader":
        from lerobot.teleoperators.utils import make_teleoperator_from_config

        return make_teleoperator_from_config(_native_device_config(config))
    if config.role == "follower":
        from lerobot.robots.utils import make_robot_from_config

        return make_robot_from_config(_native_device_config(config))
    raise RuntimeError(f"Unsupported calibration role: {config.role}")


def _save_profile(
    config: CalibrationDeviceConfig,
    profile: dict[str, dict[str, int]],
) -> Path:
    from lerobot.motors import MotorCalibration

    device = _make_device(config)
    device.calibration = {
        motor: MotorCalibration(
            id=int(values["id"]),
            drive_mode=int(values["drive_mode"]),
            homing_offset=int(values["homing_offset"]),
            range_min=int(values["range_min"]),
            range_max=int(values["range_max"]),
        )
        for motor, values in profile.items()
    }
    device._save_calibration()
    return Path(device.calibration_fpath)


def _read_profile(path: Path) -> dict[str, dict[str, int]]:
    import draccus

    from lerobot.motors import MotorCalibration

    with path.open() as stream, draccus.config_type("json"):
        profile = draccus.load(dict[str, MotorCalibration], stream)
    return {motor: asdict(calibration) for motor, calibration in profile.items()}


HALF_TURN = 2048
POSITION_MIN = 0
POSITION_MAX = 4095
PROBE_STEP = 20
PROBE_INTERVAL = 0.04
MOVE_STEP = 16
MOVE_INTERVAL = 0.02
MOVE_TOL = 40
SAT_DELTA = 2
SAT_CYCLES = 6
CLAMP_EDGE_DIST = 50
WRAP_THRESHOLD = 1000
RETREAT_TICKS = 50
PROBE_MAX_TICKS = 250
MOVE_MAX_TICKS = 500
EEPROM_COMMIT_DELAY = 0.05
PROBE_TORQUE_LIMIT = 600
TRANSIENT_ATTEMPTS = 5
TRANSIENT_DELAY = 0.05
SAFETY_MARGIN_TICKS = 20
M2M3_CENTER_TOL = 150


@dataclass
class _ProbeResult:
    motor_id: int
    hard_min: int
    hard_max: int
    applied_min: int
    applied_max: int
    homing_offset: int = 0
    drive_mode: int = 0


@dataclass
class _ProbeState:
    direction: int = 0
    start_pos: int = 0
    last_pos: int = 0
    goal: int = 0
    stalled: int = 0
    done: bool = True
    result_pos: int = 0
    reason: str = ""


def _check_auto_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise CalibrationStoppedError("Stopped by user.")


def _retry(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    last: Exception | None = None
    for index in range(TRANSIENT_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except CalibrationStoppedError:
            raise
        except (ConnectionError, RuntimeError) as exc:
            last = exc
            logger.warning("[cal:retry] %s (%s/%s): %s", label, index + 1, TRANSIENT_ATTEMPTS, exc)
            time.sleep(TRANSIENT_DELAY)
    assert last is not None
    raise last


def _read_pos(bus: Any, name: str) -> int:
    return int(_retry(f"read Present_Position {name}", bus.read, "Present_Position", name, normalize=False))


def _write_goal(bus: Any, name: str, goal: int) -> None:
    _retry(f"write Goal_Position {name}", bus.write, "Goal_Position", name, goal, normalize=False, num_retry=2)


class _MotorProber:
    def __init__(self, bus: Any, name: str, motor_id: int, *, stop_event: Event | None = None) -> None:
        self._bus = bus
        self._name = name
        self._motor_id = motor_id
        self._stop_event = stop_event
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False
        self._orig_min: int | None = None
        self._orig_max: int | None = None
        self._orig_homing: int | None = None
        self._orig_torque_limit: int | None = None
        self._probe = _ProbeState()
        self._last_goal_written: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def motor_id(self) -> int:
        return self._motor_id

    @property
    def min_pos(self) -> int:
        return self._min_pos

    @property
    def max_pos(self) -> int:
        return self._max_pos

    @property
    def center(self) -> int:
        return (self._min_pos + self._max_pos) // 2

    def needs_more(self) -> bool:
        return not (self._min_real and self._max_real)

    def _check_stopped(self) -> None:
        _check_auto_stopped(self._stop_event)

    def _set_goal(self, goal: int) -> None:
        _write_goal(self._bus, self._name, goal)
        self._last_goal_written = goal

    def _anchor_frame_here(self) -> None:
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(
            f"Torque_Enable=128 {self._name}",
            self._bus.write,
            "Torque_Enable",
            self._name,
            128,
            normalize=False,
            num_retry=2,
        )
        time.sleep(0.1)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write, "Operating_Mode", self._name, 0, normalize=False)

    def prepare(self) -> None:
        self._check_stopped()
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        self._orig_min = int(_retry(
            f"read Min_Position_Limit {self._name}",
            self._bus.read,
            "Min_Position_Limit",
            self._name,
            normalize=False,
        ))
        self._orig_max = int(_retry(
            f"read Max_Position_Limit {self._name}",
            self._bus.read,
            "Max_Position_Limit",
            self._name,
            normalize=False,
        ))
        self._orig_homing = int(_retry(
            f"read Homing_Offset {self._name}",
            self._bus.read,
            "Homing_Offset",
            self._name,
            normalize=False,
        ))
        self._orig_torque_limit = int(_retry(
            f"read Torque_Limit {self._name}",
            self._bus.read,
            "Torque_Limit",
            self._name,
            normalize=False,
        ))
        self._anchor_frame_here()
        _retry(f"write Min_Position_Limit {self._name}", self._bus.write, "Min_Position_Limit", self._name, POSITION_MIN, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        _retry(f"write Max_Position_Limit {self._name}", self._bus.write, "Max_Position_Limit", self._name, POSITION_MAX, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        _retry(f"write Torque_Limit {self._name}", self._bus.write, "Torque_Limit", self._name, PROBE_TORQUE_LIMIT, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False

    def reset_center(self) -> None:
        self._check_stopped()
        self._anchor_frame_here()
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False

    def capture_current_as_center(self) -> None:
        self._check_stopped()
        self._anchor_frame_here()
        self._min_pos = POSITION_MIN
        self._max_pos = POSITION_MAX
        self._min_real = True
        self._max_real = True

    def restore_orig_limits(self) -> None:
        if self._orig_min is None or self._orig_max is None:
            return
        try:
            _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
            _retry(f"write Min_Position_Limit {self._name}", self._bus.write, "Min_Position_Limit", self._name, self._orig_min, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(f"write Max_Position_Limit {self._name}", self._bus.write, "Max_Position_Limit", self._name, self._orig_max, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            if self._orig_homing is not None:
                _retry(f"write Homing_Offset {self._name}", self._bus.write, "Homing_Offset", self._name, self._orig_homing, normalize=False)
                time.sleep(EEPROM_COMMIT_DELAY)
            if self._orig_torque_limit is not None:
                _retry(f"write Torque_Limit {self._name}", self._bus.write, "Torque_Limit", self._name, self._orig_torque_limit, normalize=False)
                time.sleep(EEPROM_COMMIT_DELAY)
        except Exception:
            logger.exception("[cal:prober] %s restore failed", self._name)

    def start_probe(self, direction: int) -> None:
        assert direction in (-1, +1)
        self._check_stopped()
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write, "Operating_Mode", self._name, 0, normalize=False)
        pos = _read_pos(self._bus, self._name)
        self._probe = _ProbeState(direction=direction, start_pos=pos, last_pos=pos, goal=pos, stalled=0, done=False)
        self._set_goal(pos)
        _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=3)
        time.sleep(0.05)

    def probe_tick(self) -> bool:
        state = self._probe
        if state.done:
            return True
        self._check_stopped()
        state.goal = max(POSITION_MIN, min(POSITION_MAX, state.goal + state.direction * PROBE_STEP))
        try:
            self._set_goal(state.goal)
            pos = _read_pos(self._bus, self._name)
        except Exception as exc:
            logger.warning("[cal:probe] %s tick failed: %s", self._name, exc)
            return False
        if abs(pos - state.last_pos) > WRAP_THRESHOLD:
            edge = POSITION_MIN if state.direction < 0 else POSITION_MAX
            self._finish_probe(edge, "wrap", state.last_pos)
            return True
        if abs(pos - state.last_pos) < SAT_DELTA:
            state.stalled += 1
            if state.stalled >= SAT_CYCLES:
                edge = POSITION_MIN if state.direction < 0 else POSITION_MAX
                travel = pos - state.start_pos
                dir_ok = (
                    (state.direction > 0 and travel > 0)
                    or (state.direction < 0 and travel < 0)
                    or abs(travel) < 30
                )
                reason = "clamp" if (abs(pos - edge) < CLAMP_EDGE_DIST or not dir_ok) else "real"
                self._finish_probe(pos, reason, pos)
                return True
        else:
            state.stalled = 0
        state.last_pos = pos
        return False

    def _finish_probe(self, report_pos: int, reason: str, actual_pos: int) -> None:
        state = self._probe
        park_pos = actual_pos - state.direction * RETREAT_TICKS if reason in {"real", "wrap"} else actual_pos
        park_pos = max(POSITION_MIN, min(POSITION_MAX, park_pos))
        try:
            self._set_goal(park_pos)
        except Exception as exc:
            logger.warning("[cal:probe] %s park failed: %s", self._name, exc)
        state.done = True
        state.result_pos = report_pos
        state.reason = reason
        if state.direction < 0:
            self._min_pos = report_pos
            self._min_real = reason == "real"
        else:
            self._max_pos = report_pos
            self._max_real = reason == "real"

    def is_probe_done(self) -> bool:
        return self._probe.done

    def probe(self, direction: int) -> tuple[int, str]:
        self.start_probe(direction)
        for _ in range(PROBE_MAX_TICKS):
            if self.probe_tick():
                break
            time.sleep(PROBE_INTERVAL)
        return self._probe.result_pos, self._probe.reason

    def move_to(self, target: int, *, tol: int = MOVE_TOL) -> None:
        self._check_stopped()
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write, "Operating_Mode", self._name, 0, normalize=False)
        pos = _read_pos(self._bus, self._name)
        goal = pos
        self._set_goal(pos)
        _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=3)
        time.sleep(0.05)
        for _ in range(MOVE_MAX_TICKS):
            self._check_stopped()
            if goal != target:
                step = max(-MOVE_STEP, min(MOVE_STEP, target - goal))
                goal += step
                try:
                    self._set_goal(goal)
                except Exception as exc:
                    logger.warning("[cal:move] %s goal failed: %s", self._name, exc)
            try:
                pos = _read_pos(self._bus, self._name)
            except Exception as exc:
                logger.warning("[cal:move] %s read failed: %s", self._name, exc)
                time.sleep(MOVE_INTERVAL)
                continue
            if goal == target and abs(pos - target) < tol:
                return
            time.sleep(MOVE_INTERVAL)
        logger.warning("[cal:move] %s -> %s budget exhausted", self._name, target)

    def refresh_hold(self) -> None:
        try:
            pos = _read_pos(self._bus, self._name)
            self._set_goal(pos)
            _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=2)
        except Exception as exc:
            logger.warning("[cal:prober] %s refresh failed: %s", self._name, exc)

    def release(self) -> None:
        if self._orig_torque_limit is not None:
            try:
                _retry(f"restore Torque_Limit {self._name}", self._bus.write, "Torque_Limit", self._name, self._orig_torque_limit, normalize=False)
            except Exception as exc:
                logger.warning("[cal:prober] %s torque restore failed: %s", self._name, exc)
        try:
            _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        except Exception as exc:
            logger.warning("[cal:prober] %s release failed: %s", self._name, exc)

    def run_full(self, max_iter: int = 4) -> None:
        for _ in range(max_iter):
            self.probe(-1)
            self.probe(+1)
            if not self.needs_more():
                break
            self.move_to(self.center)
            self.reset_center()
        self.move_to(self.center)


def _paired_probe(p1: _MotorProber, d1: int, p2: _MotorProber, d2: int) -> None:
    assert d1 == -d2
    p1.start_probe(d1)
    p2.start_probe(d2)
    for _ in range(PROBE_MAX_TICKS):
        _check_auto_stopped(p1._stop_event)
        first_done = p1.is_probe_done() or p1.probe_tick()
        second_done = p2.is_probe_done() or p2.probe_tick()
        if first_done and second_done:
            return
        time.sleep(PROBE_INTERVAL)


def _paired_iter_probe(
    p1: _MotorProber,
    p2: _MotorProber,
    max_iter: int = 4,
    refresh_holds: list[_MotorProber] | None = None,
) -> None:
    refresh_holds = refresh_holds or []
    for _ in range(max_iter):
        _paired_probe(p1, -1, p2, +1)
        for hold in refresh_holds:
            hold.refresh_hold()
        _paired_probe(p1, +1, p2, -1)
        for hold in refresh_holds:
            hold.refresh_hold()
        if not (p1.needs_more() or p2.needs_more()):
            return
        _concurrent_move([(p1, p1.center), (p2, p2.center)])
        p1.reset_center()
        p2.reset_center()
        for hold in refresh_holds:
            hold.refresh_hold()


def _concurrent_move(pairs: list[tuple[_MotorProber, int]], *, tol: int = MOVE_TOL) -> None:
    if not pairs:
        return
    for prober, _ in pairs:
        _retry(f"disable_torque {prober.name}", prober._bus.disable_torque, prober.name, num_retry=3)
        _retry(f"write Operating_Mode {prober.name}", prober._bus.write, "Operating_Mode", prober.name, 0, normalize=False)
    goals: dict[_MotorProber, int] = {}
    for prober, _ in pairs:
        start = prober._last_goal_written if prober._last_goal_written is not None else _read_pos(prober._bus, prober.name)
        goals[prober] = start
        prober._set_goal(start)
        _retry(f"enable_torque {prober.name}", prober._bus.enable_torque, prober.name, num_retry=3)
    time.sleep(0.05)
    done: dict[_MotorProber, bool] = {prober: False for prober, _ in pairs}
    for _ in range(MOVE_MAX_TICKS):
        pairs[0][0]._check_stopped()
        for prober, target in pairs:
            if done[prober] or goals[prober] == target:
                continue
            step = max(-MOVE_STEP, min(MOVE_STEP, target - goals[prober]))
            goals[prober] += step
            try:
                prober._set_goal(goals[prober])
            except Exception as exc:
                logger.warning("[cal:move] %s goal failed: %s", prober.name, exc)
        for prober, target in pairs:
            if done[prober]:
                continue
            try:
                pos = _read_pos(prober._bus, prober.name)
            except Exception as exc:
                logger.warning("[cal:move] %s read failed: %s", prober.name, exc)
                continue
            if goals[prober] == target and abs(pos - target) < tol:
                done[prober] = True
        if all(done.values()):
            return
        time.sleep(MOVE_INTERVAL)


class _SO101AutoCalibrator:
    ARM_MOTORS = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
    }
    GRIPPER_NAME = "gripper"
    GRIPPER_ID = 6
    WRIST_ROLL_NAME = "wrist_roll"
    WRIST_ROLL_ID = 5
    ALL_MOTORS = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "gripper",
        "wrist_roll",
    )

    def __init__(
        self,
        port: str,
        *,
        stop_event: Event | None = None,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        self._port = port
        self._stop_event = stop_event
        self._on_event = on_event
        motors = {
            name: Motor(id=motor_id, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
            for name, motor_id in self.ARM_MOTORS.items()
        }
        motors[self.GRIPPER_NAME] = Motor(id=self.GRIPPER_ID, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
        motors[self.WRIST_ROLL_NAME] = Motor(id=self.WRIST_ROLL_ID, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
        self._bus = FeetechMotorsBus(port=port, motors=motors)
        self._probers: dict[str, _MotorProber] = {}

    def _emit(self, phase: str, message: str, **extra: Any) -> None:
        if self._on_event:
            self._on_event(phase, message, **extra)

    def _check_stopped(self) -> None:
        _check_auto_stopped(self._stop_event)

    def run(self) -> dict[str, dict[str, int]]:
        self._emit("preparing", "正在连接 SO-101 电机总线。")
        self._bus.connect(handshake=True)
        self._probers = {
            "shoulder_pan": _MotorProber(self._bus, "shoulder_pan", 1, stop_event=self._stop_event),
            "shoulder_lift": _MotorProber(self._bus, "shoulder_lift", 2, stop_event=self._stop_event),
            "elbow_flex": _MotorProber(self._bus, "elbow_flex", 3, stop_event=self._stop_event),
            "wrist_flex": _MotorProber(self._bus, "wrist_flex", 4, stop_event=self._stop_event),
            "gripper": _MotorProber(self._bus, "gripper", self.GRIPPER_ID, stop_event=self._stop_event),
            "wrist_roll": _MotorProber(self._bus, "wrist_roll", self.WRIST_ROLL_ID, stop_event=self._stop_event),
        }
        try:
            self._prepare_all()
            self._run_sequence()
            results = self._build_results()
            self._move_to_final_pose()
            self._apply_results(results)
            for prober in self._probers.values():
                prober.release()
            return {
                name: {
                    "id": result.motor_id,
                    "drive_mode": result.drive_mode,
                    "homing_offset": result.homing_offset,
                    "range_min": result.applied_min,
                    "range_max": result.applied_max,
                }
                for name, result in results.items()
            }
        except Exception:
            logger.exception("[cal] failed; restoring original EEPROM limits")
            self._restore_via_fresh_bus()
            raise
        finally:
            try:
                self._bus.disconnect(disable_torque=True)
            except Exception as exc:
                logger.warning("[cal] disconnect failed: %s", exc)

    def _prepare_all(self) -> None:
        self._emit("probing", "正在准备电机 EEPROM 范围。", current_arm="")
        for name in self.ALL_MOTORS:
            self._check_stopped()
            self._emit("probing", f"正在准备 {name}。", motor=name)
            self._probers[name].prepare()

    def _run_sequence(self) -> None:
        prober = self._probers
        self._emit("probing", "正在探测夹爪活动范围。", motor=self.GRIPPER_NAME)
        prober[self.GRIPPER_NAME].run_full()
        self._emit("probing", "正在探测 wrist_flex 负向范围。", motor="wrist_flex")
        prober["wrist_flex"].probe(-1)
        self._emit("probing", "正在探测 shoulder_pan 范围。", motor="shoulder_pan")
        prober["shoulder_pan"].run_full()
        self._emit("probing", "正在联动探测 shoulder_lift / elbow_flex。", motor="shoulder_lift")
        _paired_iter_probe(prober["shoulder_lift"], prober["elbow_flex"], refresh_holds=[prober["wrist_flex"]])
        prober["wrist_flex"].refresh_hold()
        _concurrent_move(
            [
                (prober["shoulder_lift"], prober["shoulder_lift"].center),
                (prober["elbow_flex"], prober["elbow_flex"].center),
            ],
            tol=M2M3_CENTER_TOL,
        )
        self._emit("probing", "正在探测 wrist_flex 正向范围。", motor="wrist_flex")
        prober["wrist_flex"].probe(+1)
        self._emit("probing", "正在记录 wrist_roll 当前中心位。", motor=self.WRIST_ROLL_NAME)
        prober[self.WRIST_ROLL_NAME].capture_current_as_center()

    def _build_results(self) -> dict[str, _ProbeResult]:
        results = {}
        for name in self.ALL_MOTORS:
            self._check_stopped()
            prober = self._probers[name]
            homing_offset = int(_retry(f"read Homing_Offset {name}", self._bus.read, "Homing_Offset", name, normalize=False))
            if name == self.WRIST_ROLL_NAME:
                results[name] = _ProbeResult(
                    motor_id=prober.motor_id,
                    hard_min=POSITION_MIN,
                    hard_max=POSITION_MAX,
                    applied_min=POSITION_MIN,
                    applied_max=POSITION_MAX,
                    homing_offset=homing_offset,
                    drive_mode=0,
                )
                continue
            applied_min = max(POSITION_MIN, prober.min_pos + SAFETY_MARGIN_TICKS)
            applied_max = min(POSITION_MAX, prober.max_pos - SAFETY_MARGIN_TICKS)
            if applied_min >= applied_max:
                raise RuntimeError(f"{name}: safety margin collapses range: [{prober.min_pos}, {prober.max_pos}]")
            results[name] = _ProbeResult(
                motor_id=prober.motor_id,
                hard_min=prober.min_pos,
                hard_max=prober.max_pos,
                applied_min=applied_min,
                applied_max=applied_max,
                homing_offset=homing_offset,
                drive_mode=0,
            )
        return results

    def _apply_results(self, results: dict[str, _ProbeResult]) -> None:
        self._emit("saving", "正在写入 SO-101 EEPROM 限位。")
        for name, result in results.items():
            self._check_stopped()
            _retry(f"disable_torque {name}", self._bus.disable_torque, name, num_retry=3)
            _retry(f"write Min_Position_Limit {name}", self._bus.write, "Min_Position_Limit", name, result.applied_min, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(f"write Max_Position_Limit {name}", self._bus.write, "Max_Position_Limit", name, result.applied_max, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(f"enable_torque {name}", self._bus.enable_torque, name, num_retry=3)

    def _move_to_final_pose(self) -> None:
        prober = self._probers
        self._emit("saving", "正在移动到自动校准结束姿态。")
        wrist_flex_target = (prober["wrist_flex"].max_pos + prober["wrist_flex"].center) // 2
        _concurrent_move(
            [
                (prober["shoulder_pan"], prober["shoulder_pan"].center),
                (prober["shoulder_lift"], prober["shoulder_lift"].min_pos + SAFETY_MARGIN_TICKS),
                (prober["elbow_flex"], prober["elbow_flex"].max_pos - SAFETY_MARGIN_TICKS),
                (prober["wrist_flex"], wrist_flex_target),
                (prober[self.GRIPPER_NAME], prober[self.GRIPPER_NAME].min_pos + SAFETY_MARGIN_TICKS),
            ],
            tol=M2M3_CENTER_TOL,
        )

    def _restore_via_fresh_bus(self) -> None:
        try:
            self._bus.disconnect(disable_torque=True)
        except Exception as exc:
            logger.warning("[cal] disconnect before restore failed: %s", exc)
        time.sleep(0.3)
        try:
            self._bus.connect(handshake=False)
        except Exception:
            logger.exception("[cal] could not reconnect bus to restore EEPROM")
            return
        for name in self.ALL_MOTORS:
            prober = self._probers.get(name)
            if prober:
                prober.restore_orig_limits()
