#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time
from collections import deque
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.runtime_bridge import runtime_prompt

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._motor_bus_recovery_duration_s = 0.0
        self._recovery_callback = None
        self._recovery_cancelled = lambda: False
        self._recovery_events = deque()
        self.motor_bus_recovery_failed = False
        self._preserve_fault_torque = False
        self._recovery_resume_pose = None
        self._last_positions = None
        self.bus = self._make_motor_bus()
        self.cameras = make_cameras_from_configs(config.cameras)

    def _make_motor_bus(self) -> FeetechMotorsBus:
        # choose normalization mode depending on config if available
        norm_mode_body = (
            MotorNormMode.DEGREES if self.config.use_degrees else MotorNormMode.RANGE_M100_100
        )
        return FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    def consume_motor_bus_recovery_duration_s(self) -> float:
        """Return and clear the last recovery pause so rollout timing can exclude it."""
        duration_s = self._motor_bus_recovery_duration_s
        self._motor_bus_recovery_duration_s = 0.0
        return duration_s

    def set_motor_bus_recovery_callback(self, callback, cancelled=None) -> None:
        """Rollout hook: pause inference before rebuilding transport; never reads this robot."""
        self._recovery_callback = callback
        self._recovery_cancelled = cancelled or (lambda: False)

    def preserve_recovery_pose_on_disconnect(self) -> None:
        # Leave the existing torque state alone. This cannot guarantee a pose if
        # a motor lost power, and deliberately does not re-enable a failed motor.
        self._preserve_fault_torque = True

    def disconnect_after_motor_fault(self) -> None:
        """Release owned handles even when only part of the hardware is connected."""
        self.preserve_recovery_pose_on_disconnect()
        try:
            if self.bus.is_connected:
                self.bus.disconnect(False)
        finally:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()

    def _validate_recovery_positions(self, positions):
        if set(positions) != set(self.bus.motors) or not all(
            math.isfinite(float(value)) for value in positions.values()
        ):
            raise ConnectionError(f"{self}: incomplete or non-finite recovery position feedback")
        return positions

    def _stable_recovery_positions(self, initial):
        initial = self._validate_recovery_positions(initial)
        positions = initial
        count = self.config.read_reconnect_stable_reads
        interval = self.config.read_reconnect_stable_interval_s
        threshold = self.config.read_reconnect_stable_max_delta
        if count < 2 or not math.isfinite(interval) or interval < 0 or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Invalid motor recovery stability configuration")
        for _ in range(count - 1):
            if self._recovery_cancelled():
                raise ConnectionError("Motor recovery cancelled by operator")
            time.sleep(interval)
            positions = self._validate_recovery_positions(
                self.bus.sync_read("Present_Position", num_retry=0)
            )
            if any(abs(float(positions[k]) - float(initial[k])) > threshold for k in initial):
                raise ConnectionError(f"{self}: position still moving during recovery validation")
        return positions

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            if getattr(self.cameras[cam], "use_rgb", True):
                features[cam] = (self.cameras[cam].height, self.cameras[cam].width, 3)
            if getattr(self.cameras[cam], "use_depth", False):
                features[f"{cam}_depth"] = (self.cameras[cam].height, self.cameras[cam].width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = runtime_prompt(
                "calibration_mismatch",
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        runtime_prompt(
            "calibration_middle",
            f"Move {self} to the middle of its range of motion and press ENTER....",
        )
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        num_retry = self.config.num_write_retries
        with self.bus.torque_disabled(num_retry=num_retry):
            self.bus.configure_motors(num_retry=num_retry)
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=num_retry)
                self.bus.write(
                    "P_Coefficient", motor, self.config.position_p_coefficient, num_retry=num_retry
                )
                self.bus.write(
                    "I_Coefficient", motor, self.config.position_i_coefficient, num_retry=num_retry
                )
                self.bus.write(
                    "D_Coefficient", motor, self.config.position_d_coefficient, num_retry=num_retry
                )

                if motor == "gripper":
                    self.bus.write(
                        "Max_Torque_Limit", motor, 500, num_retry=num_retry
                    )  # 50% of max torque to avoid burnout
                    self.bus.write(
                        "Protection_Current", motor, 250, num_retry=num_retry
                    )  # 50% of max current to avoid burnout
                    self.bus.write(
                        "Overload_Torque", motor, 25, num_retry=num_retry
                    )  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            runtime_prompt(
                "configure_motor",
                f"Connect the controller board to the '{motor}' motor only and press enter.",
            )
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def _read_present_positions(self) -> dict[str, float]:
        """Read joint positions, rebuilding a wedged motor transport when needed."""
        try:
            positions = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            self._last_positions = dict(positions)
            return positions
        except ConnectionError as initial_error:
            last_error = initial_error

        recovery_started = time.perf_counter()
        self.motor_bus_recovery_failed = True  # Cleared only after all recovery checks succeed.
        if self._recovery_callback is not None:
            self._recovery_callback("recovering", self.config.port, str(last_error))
        now = time.monotonic()
        while self._recovery_events and now - self._recovery_events[0] > self.config.read_reconnect_window_s:
            self._recovery_events.popleft()
        self._recovery_events.append(now)
        if len(self._recovery_events) > self.config.read_reconnect_max_events:
            raise ConnectionError(f"{self}: recurrent motor failures; recovery budget exceeded, preserving partial episode") from last_error
        for attempt in range(1, self.config.read_reconnect_attempts + 1):
            if self._recovery_cancelled():
                raise ConnectionError("Motor recovery cancelled by operator") from last_error
            backoff_s = min(
                self.config.read_reconnect_backoff_s * (2 ** (attempt - 1)),
                self.config.read_reconnect_max_backoff_s,
            )
            logger.warning(
                "%s motor read failed after retries; rebuilding %s in %.2fs (%d/%d)",
                self,
                self.config.port,
                backoff_s,
                attempt,
                self.config.read_reconnect_attempts,
            )
            stage = "close_transport"
            try:
                if self.bus.is_connected:
                    self.bus.disconnect(False)
                if backoff_s:
                    time.sleep(backoff_s)
                if self._recovery_cancelled():
                    raise ConnectionError("Motor recovery cancelled by operator")
                # A new bus also creates fresh PortHandler/PacketHandler/GroupSyncRead objects.
                # Merely reopening the old PortHandler can leave the SDK transport wedged.
                self.bus = self._make_motor_bus()
                stage = "open_and_handshake"
                self.bus.connect()
                stage = "read_all_motors"
                positions = self.bus.sync_read(
                    "Present_Position", num_retry=self.config.num_read_retries
                )
                stage = "validate_before_enable"
                positions = self._stable_recovery_positions(positions)
                # If the motor controller briefly browned out, torque may have reset. Latch the
                # measured pose before enabling torque so recovery cannot jump to an old target.
                stage = "latch_current_pose"
                self.bus.sync_write("Goal_Position", positions)
                stage = "enable_torque"
                self.bus.enable_torque(num_retry=self.config.num_write_retries)
                stage = "validate_after_enable"
                positions = self._stable_recovery_positions(positions)
            except (ConnectionError, OSError, RuntimeError) as error:
                last_error = error
                logger.warning("%s recovery attempt %d failed at %s: %s", self, attempt, stage, error)
                continue
            recovery_duration_s = time.perf_counter() - recovery_started
            self._motor_bus_recovery_duration_s += recovery_duration_s
            self.motor_bus_recovery_failed = False
            self._recovery_resume_pose = dict(positions)
            self._last_positions = dict(positions)
            logger.warning(
                "%s motor bus recovered on %s after %.2fs; current pose latched",
                self,
                self.config.port,
                recovery_duration_s,
            )
            return positions

        raise ConnectionError(
            f"{self} motor bus did not recover on {self.config.port} after "
            f"{self.config.read_reconnect_attempts} reopen attempt(s)"
        ) from last_error

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        start = time.perf_counter()
        obs_dict = self._read_present_positions()
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # A current-pose hold does not consume the guard. The first genuinely new
        # policy/human target must not jump away after a communication gap.
        if self._recovery_resume_pose is not None and goal_pos:
            if not all(math.isfinite(float(value)) for value in goal_pos.values()):
                self.motor_bus_recovery_failed = True
                raise ConnectionError(f"{self}: non-finite recovered target; refusing motion")
            delta = max(abs(float(value) - float(self._recovery_resume_pose[key])) for key, value in goal_pos.items())
            if not math.isfinite(delta) or delta > self.config.read_reconnect_resume_max_delta:
                self.motor_bus_recovery_failed = True
                raise ConnectionError(f"{self}: first recovered target jumps {delta:.2f}; refusing motion, realign before restarting")
            if delta > 0.25:
                self._recovery_resume_pose = None

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            recovery_before = self._motor_bus_recovery_duration_s
            present_pos = self._read_present_positions()
            if self._motor_bus_recovery_duration_s != recovery_before:
                # This target was computed before the read/reconnect. Do not
                # finish sending it after recovery merely because it was clipped.
                self.motor_bus_recovery_failed = True
                raise ConnectionError(f"{self}: recovery during action write; stale target discarded, saving partial episode")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        try:
            self.bus.sync_write("Goal_Position", goal_pos)
        except (ConnectionError, OSError) as error:
            self.motor_bus_recovery_failed = True
            if self._recovery_callback is not None:
                self._recovery_callback("write_failed", self.config.port, str(error))
            raise
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        preserve = self._preserve_fault_torque or (
            self.motor_bus_recovery_failed and self._recovery_callback is not None
        )
        if preserve:
            logger.warning("%s: closing faulted transport without homing or changing torque; pose cannot be guaranteed without feedback", self)
        self.bus.disconnect(False if preserve else self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
