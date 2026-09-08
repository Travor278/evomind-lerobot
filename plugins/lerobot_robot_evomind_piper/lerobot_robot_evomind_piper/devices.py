"""LeRobot PiperX plugin with Evo-RL-compatible runtime behavior."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .can_setup import ensure_piper_can_ready
from .common import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    PIPER_JOINT_NAMES,
    PIPER_ROLE_FOLLOWER,
    PIPER_ROLE_LEADER,
    get_piper_sdk,
    milli_to_unit,
    parse_piper_log_level,
    set_piper_role,
    unit_to_milli,
    wait_enable_piper,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PiperXFollowerConfigBase:
    port: str
    judge_flag: bool = False
    can_auto_init: bool = True
    log_level: str = "WARNING"
    startup_sleep_s: float = 0.1
    speed_ratio: int = 100
    high_follow: bool = True
    enable_on_connect: bool = True
    enable_timeout_s: float = 3.0
    sync_gripper: bool = True
    gripper_effort_default: int = 1000
    gripper_status_code: int = 0x01
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    disable_on_disconnect: bool = False


def _validate_piper_follower_config(config: PiperXFollowerConfigBase) -> None:
    if not 0 <= config.speed_ratio <= 100:
        raise ValueError("`speed_ratio` must be between 0 and 100.")
    if config.enable_timeout_s < 0:
        raise ValueError("`enable_timeout_s` must be >= 0.")
    if config.startup_sleep_s < 0:
        raise ValueError("`startup_sleep_s` must be >= 0.")
    if not 0 <= config.gripper_effort_default <= 5000:
        raise ValueError("`gripper_effort_default` must be between 0 and 5000.")
    if config.gripper_status_code not in {0x00, 0x01, 0x02, 0x03}:
        raise ValueError("`gripper_status_code` must be one of 0x00, 0x01, 0x02, 0x03.")


@RobotConfig.register_subclass("piperx_follower")
@dataclass(kw_only=True)
class PiperXFollowerConfig(RobotConfig, PiperXFollowerConfigBase):
    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_piper_follower_config(self)


@dataclass(kw_only=True)
class PiperXLeaderConfigBase:
    port: str
    judge_flag: bool = False
    can_auto_init: bool = True
    log_level: str = "WARNING"
    startup_sleep_s: float = 0.1
    manual_control: bool = True
    sync_gripper: bool = True
    gripper_effort_default: int = 1000
    gripper_status_code: int = 0x01
    command_speed_ratio: int = 100
    command_high_follow: bool = True
    mode_refresh_interval_s: float = 1.0
    enable_timeout_s: float = 3.0
    disable_on_disconnect: bool = False


def _validate_piper_leader_config(config: PiperXLeaderConfigBase) -> None:
    if not 0 <= config.command_speed_ratio <= 100:
        raise ValueError("`command_speed_ratio` must be between 0 and 100.")
    if config.mode_refresh_interval_s < 0:
        raise ValueError("`mode_refresh_interval_s` must be >= 0.")
    if config.enable_timeout_s < 0:
        raise ValueError("`enable_timeout_s` must be >= 0.")
    if config.startup_sleep_s < 0:
        raise ValueError("`startup_sleep_s` must be >= 0.")
    if not 0 <= config.gripper_effort_default <= 5000:
        raise ValueError("`gripper_effort_default` must be between 0 and 5000.")
    if config.gripper_status_code not in {0x00, 0x01, 0x02, 0x03}:
        raise ValueError("`gripper_status_code` must be one of 0x00, 0x01, 0x02, 0x03.")


@TeleoperatorConfig.register_subclass("piperx_leader")
@dataclass(kw_only=True)
class PiperXLeaderConfig(TeleoperatorConfig, PiperXLeaderConfigBase):
    def __post_init__(self) -> None:
        _validate_piper_leader_config(self)


@RobotConfig.register_subclass("bi_piperx_follower")
@dataclass(kw_only=True)
class BiPiperXFollowerConfig(RobotConfig):
    left_arm_config: PiperXFollowerConfigBase
    right_arm_config: PiperXFollowerConfigBase


@TeleoperatorConfig.register_subclass("bi_piperx_leader")
@dataclass(kw_only=True)
class BiPiperXLeaderConfig(TeleoperatorConfig):
    left_arm_config: PiperXLeaderConfigBase
    right_arm_config: PiperXLeaderConfigBase
    process_isolation: bool = True


class PiperXFollower(Robot):
    config_class = PiperXFollowerConfig
    name = "piperx_follower"

    def __init__(self, config: PiperXFollowerConfig):
        super().__init__(config)
        self.config = config
        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=ensure_piper_can_ready(config.port),
            judge_flag=config.judge_flag,
            can_auto_init=config.can_auto_init,
            logger_level=parse_piper_log_level(config.log_level),
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            camera: (
                self.config.cameras[camera].height,
                self.config.cameras[camera].width,
                3,
            )
            for camera in self.cameras
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(camera.is_connected for camera in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm.ConnectPort()
        connected_cameras = []
        try:
            if self.config.startup_sleep_s > 0:
                time.sleep(self.config.startup_sleep_s)
            self._is_connected = True
            self.configure()
            if self.config.enable_on_connect and not wait_enable_piper(
                self.arm, self.config.enable_timeout_s
            ):
                logger.warning("Piper follower did not report enabled state before timeout.")
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
        except Exception:
            self.arm.DisconnectPort()
            for camera in connected_cameras:
                camera.disconnect()
            self._is_connected = False
            raise
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
        mit_mode = 0xAD if self.config.high_follow else 0x00
        self.arm.MotionCtrl_2(0x01, 0x01, self.config.speed_ratio, mit_mode)

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joint_message = self.arm.GetArmJointMsgs()
        joint_state = getattr(joint_message, "joint_state", None)
        observation: RobotObservation = {}
        for joint_name in PIPER_JOINT_NAMES:
            raw_value = getattr(joint_state, joint_name, 0)
            observation[f"{joint_name}.pos"] = milli_to_unit(raw_value)
        gripper_message = self.arm.GetArmGripperMsgs()
        gripper_state = getattr(gripper_message, "gripper_state", None)
        observation["gripper.pos"] = abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))
        for camera_key, camera in self.cameras.items():
            observation[camera_key] = camera.async_read()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent_action: dict[str, float] = {}
        has_all_joints = all(key in action for key in PIPER_JOINT_ACTION_KEYS)
        if has_all_joints:
            joint_targets = [action[key] for key in PIPER_JOINT_ACTION_KEYS]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)
            sent_action.update(
                {
                    key: milli_to_unit(raw)
                    for key, raw in zip(PIPER_JOINT_ACTION_KEYS, joint_commands, strict=True)
                }
            )
        elif any(key in action for key in PIPER_JOINT_ACTION_KEYS):
            logger.debug("Ignoring partial Piper joint action. Need all six joint keys to send command.")
        if self.config.sync_gripper and "gripper.pos" in action:
            gripper_pos_raw = unit_to_milli(action["gripper.pos"])
            self.arm.GripperCtrl(
                gripper_pos_raw,
                self.config.gripper_effort_default,
                self.config.gripper_status_code,
                0x00,
            )
            sent_action["gripper.pos"] = milli_to_unit(gripper_pos_raw)
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self.arm.DisableArm(7)
        finally:
            self.arm.DisconnectPort()
            for camera in self.cameras.values():
                camera.disconnect()
            self._is_connected = False
            logger.info("%s disconnected.", self)


class PiperXLeader(Teleoperator):
    config_class = PiperXLeaderConfig
    name = "piperx_leader"

    def __init__(self, config: PiperXLeaderConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._manual_control_enabled: bool | None = None
        self._manual_action: RobotAction | None = None
        self._last_control_joint_timestamp = 0.0
        self._last_control_gripper_timestamp = 0.0
        self._last_mode_refresh_t = 0.0
        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=ensure_piper_can_ready(config.port),
            judge_flag=config.judge_flag,
            can_auto_init=config.can_auto_init,
            logger_level=parse_piper_log_level(config.log_level),
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm.ConnectPort()
        if self.config.startup_sleep_s > 0:
            time.sleep(self.config.startup_sleep_s)
        self._is_connected = True
        self._manual_control_enabled = None
        self._manual_action = None
        try:
            self.configure()
        except Exception:
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            self._manual_action = None
            raise
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def _send_command_mode(self) -> None:
        mit_mode = 0xAD if self.config.command_high_follow else 0x00
        self.arm.MotionCtrl_2(0x01, 0x01, self.config.command_speed_ratio, mit_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_command_mode_if_needed(self) -> None:
        interval_s = self.config.mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_command_mode()

    def _send_gripper_ctrl(self, gripper_pos_raw: int, enabled: bool) -> None:
        self.arm.GripperCtrl(
            gripper_pos_raw,
            self.config.gripper_effort_default if enabled else 0,
            self.config.gripper_status_code if enabled else 0x00,
            0x00,
        )

    def _set_gripper_enabled(self, enabled: bool) -> None:
        gripper_pos_raw = 0
        try:
            gripper_message = self.arm.GetArmGripperMsgs()
            gripper_state = getattr(gripper_message, "gripper_state", None)
            if gripper_state is not None:
                gripper_pos_raw = abs(int(getattr(gripper_state, "grippers_angle", 0)))
        except Exception:
            logger.debug("Could not read current gripper angle before setting enable=%s.", enabled)
        self._send_gripper_ctrl(gripper_pos_raw, enabled)

    def set_manual_control(self, enabled: bool) -> None:
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if enabled == self._manual_control_enabled:
            return
        self._manual_control_enabled = None
        if enabled:
            self._manual_action = None
            seed_action = None
            try:
                set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
                seed_action = self._wait_for_feedback_action()
            finally:
                set_piper_role(self.arm, PIPER_ROLE_LEADER)
            if seed_action is None:
                raise RuntimeError(
                    f"[{self.config.port}] no complete Piper feedback received while initializing "
                    "manual control."
                )
            self._manual_action = seed_action
            self._last_control_joint_timestamp = self.arm.GetArmJointCtrl().time_stamp
            self._last_control_gripper_timestamp = (
                self.arm.GetArmGripperCtrl().time_stamp if self.config.sync_gripper else 0.0
            )
        else:
            self._manual_action = None
            set_piper_role(self.arm, PIPER_ROLE_FOLLOWER)
            self._send_command_mode()
            if not wait_enable_piper(self.arm, self.config.enable_timeout_s):
                raise RuntimeError(
                    f"[{self.config.port}] Piper leader did not enable after switching to follower role."
                )
            if self.config.sync_gripper:
                self._set_gripper_enabled(True)
        self._manual_control_enabled = enabled

    def enable_torque(self) -> None:
        """Put the leader in command mode so rollout can align it to the follower."""
        self.set_manual_control(False)

    def disable_torque(self) -> None:
        """Return the leader to backdrivable manual-control mode."""
        self.set_manual_control(True)

    def configure(self) -> None:
        self.set_manual_control(self.config.manual_control)

    def _read_joint_from_feedback(self) -> dict[str, float] | None:
        message = self.arm.GetArmJointMsgs()
        if message.time_stamp <= 0:
            return None
        return {
            f"{joint_name}.pos": milli_to_unit(getattr(message.joint_state, joint_name))
            for joint_name in PIPER_JOINT_NAMES
        }

    def _read_gripper_from_feedback(self) -> float | None:
        message = self.arm.GetArmGripperMsgs()
        if message.time_stamp <= 0:
            return None
        return abs(milli_to_unit(message.gripper_state.grippers_angle))

    def _wait_for_feedback_action(self) -> RobotAction | None:
        deadline = time.monotonic() + self.config.enable_timeout_s
        while True:
            action = self._read_joint_from_feedback()
            gripper_pos = self._read_gripper_from_feedback() if self.config.sync_gripper else 0.0
            if action is not None and gripper_pos is not None:
                action["gripper.pos"] = gripper_pos
                return action
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    def _read_manual_action(self) -> RobotAction:
        if self._manual_action is None:
            return {}
        joint_message = self.arm.GetArmJointCtrl()
        if joint_message.time_stamp > 0 and joint_message.time_stamp != self._last_control_joint_timestamp:
            self._manual_action.update(
                {
                    f"{joint_name}.pos": milli_to_unit(getattr(joint_message.joint_ctrl, joint_name))
                    for joint_name in PIPER_JOINT_NAMES
                }
            )
            self._last_control_joint_timestamp = joint_message.time_stamp
        if self.config.sync_gripper:
            gripper_message = self.arm.GetArmGripperCtrl()
            if (
                gripper_message.time_stamp > 0
                and gripper_message.time_stamp != self._last_control_gripper_timestamp
            ):
                self._manual_action["gripper.pos"] = abs(
                    milli_to_unit(gripper_message.gripper_ctrl.grippers_angle)
                )
                self._last_control_gripper_timestamp = gripper_message.time_stamp
        return dict(self._manual_action)

    def _read_raw_action(self) -> RobotAction:
        if self._manual_control_enabled is True:
            return self._read_manual_action()
        if self._manual_control_enabled is False:
            action = self._read_joint_from_feedback()
            gripper_pos = self._read_gripper_from_feedback() if self.config.sync_gripper else None
        else:
            raise RuntimeError(f"[{self.config.port}] Piper leader control mode is unknown.")
        if action is None:
            return {}
        if self.config.sync_gripper:
            if gripper_pos is None:
                return {}
            action["gripper.pos"] = gripper_pos
        else:
            action["gripper.pos"] = 0.0
        return action

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self._read_raw_action()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self.set_manual_control(False)
        self._refresh_command_mode_if_needed()
        has_all_joints = all(key in feedback for key in PIPER_JOINT_ACTION_KEYS)
        if has_all_joints:
            joint_targets = [feedback[key] for key in PIPER_JOINT_ACTION_KEYS]
            joint_commands = [unit_to_milli(value) for value in joint_targets]
            self.arm.JointCtrl(*joint_commands)
        if self.config.sync_gripper and "gripper.pos" in feedback:
            self._send_gripper_ctrl(unit_to_milli(feedback["gripper.pos"]), enabled=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.set_manual_control(True)
            if self.config.disable_on_disconnect:
                self.arm.DisableArm(7)
        finally:
            self.arm.DisconnectPort()
            self._is_connected = False
            self._manual_control_enabled = None
            self._manual_action = None
            logger.info("%s disconnected.", self)


class BiPiperXFollower(Robot):
    config_class = BiPiperXFollowerConfig
    name = "bi_piperx_follower"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "speed_ratio",
        "high_follow",
        "enable_on_connect",
        "enable_timeout_s",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "cameras",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, side_config: PiperXFollowerConfigBase, side: str):
        kwargs = {name: getattr(side_config, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        return PiperXFollowerConfig(**kwargs)

    def __init__(self, config: BiPiperXFollowerConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = PiperXFollower(self._build_arm_config(config.left_arm_config, "left"))
        self.right_arm = PiperXFollower(self._build_arm_config(config.right_arm_config, "right"))
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm._motors_ft.items()},
            **{f"right_{key}": value for key, value in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm._cameras_ft.items()},
            **{f"right_{key}": value for key, value in self.right_arm._cameras_ft.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.left_arm.connect()
        self.right_arm.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = {}
        observation.update({f"left_{key}": value for key, value in self.left_arm.get_observation().items()})
        observation.update({f"right_{key}": value for key, value in self.right_arm.get_observation().items()})
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action: RobotAction = {}
        right_action: RobotAction = {}
        for key, value in action.items():
            if key.startswith("left_"):
                left_action[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_action[key.removeprefix("right_")] = value
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()


def _bi_piper_leader_worker(connection, arm_config) -> None:
    arm = PiperXLeader(arm_config)
    try:
        while True:
            request = connection.recv()
            command = request["command"]
            if command == "__close__":
                break
            try:
                result = getattr(arm, command)(*request.get("args", ()), **request.get("kwargs", {}))
                connection.send({"ok": True, "result": result})
            except Exception as exc:
                connection.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        with suppress(Exception):
            if arm.is_connected:
                arm.disconnect()
        connection.close()


class _PiperLeaderProcessProxy:
    def __init__(self, arm_config: PiperXLeaderConfig):
        self._arm_config = arm_config
        self._ctx = mp.get_context("spawn")
        self._parent_conn = None
        self._process = None
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        parent_connection, child_connection = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_bi_piper_leader_worker,
            args=(child_connection, self._arm_config),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._parent_conn = parent_connection
        self._process = process

    def _call(self, command: str, *args, **kwargs):
        self._ensure_process()
        assert self._parent_conn is not None
        self._parent_conn.send({"command": command, "args": args, "kwargs": kwargs})
        response = self._parent_conn.recv()
        if response["ok"]:
            return response.get("result")
        raise RuntimeError(
            f"bi_piper leader worker command '{command}' failed: {response['error']}\n{response['traceback']}"
        )

    def connect(self) -> None:
        try:
            self._call("connect")
            self._is_connected = True
        except Exception:
            self.disconnect()
            raise

    def configure(self) -> None:
        self._call("configure")

    def setup_motors(self) -> None:
        self._call("setup_motors")

    def set_manual_control(self, enabled: bool) -> None:
        self._call("set_manual_control", enabled)

    def get_action(self) -> RobotAction:
        return self._call("get_action")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self._call("send_feedback", feedback)

    def disconnect(self) -> None:
        if self._process is None:
            self._is_connected = False
            return
        if self._parent_conn is not None:
            with suppress(Exception):
                if self._is_connected:
                    self._call("disconnect")
            with suppress(Exception):
                self._parent_conn.send({"command": "__close__"})
            with suppress(Exception):
                self._parent_conn.close()
        if self._process.is_alive():
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._parent_conn = None
        self._process = None
        self._is_connected = False


class BiPiperXLeader(Teleoperator):
    config_class = BiPiperXLeaderConfig
    name = "bi_piperx_leader"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "manual_control",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "command_speed_ratio",
        "command_high_follow",
        "mode_refresh_interval_s",
        "enable_timeout_s",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, side_config: PiperXLeaderConfigBase, side: str):
        kwargs = {name: getattr(side_config, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        return PiperXLeaderConfig(**kwargs)

    def __init__(self, config: BiPiperXLeaderConfig):
        super().__init__(config)
        self.config = config
        left_config = self._build_arm_config(config.left_arm_config, "left")
        right_config = self._build_arm_config(config.right_arm_config, "right")
        if config.process_isolation:
            self.left_arm = _PiperLeaderProcessProxy(left_config)
            self.right_arm = _PiperLeaderProcessProxy(right_config)
        else:
            self.left_arm = PiperXLeader(left_config)
            self.right_arm = PiperXLeader(right_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.feedback_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.feedback_features.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.left_arm.connect()
        self.right_arm.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def set_manual_control(self, enabled: bool) -> None:
        self.left_arm.set_manual_control(enabled)
        self.right_arm.set_manual_control(enabled)

    def enable_torque(self) -> None:
        """Put both leaders in command mode for smooth policy-to-human handover."""
        self.set_manual_control(False)

    def disable_torque(self) -> None:
        """Return both leaders to backdrivable manual-control mode."""
        self.set_manual_control(True)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action: RobotAction = {}
        action.update({f"left_{key}": value for key, value in self.left_arm.get_action().items()})
        action.update({f"right_{key}": value for key, value in self.right_arm.get_action().items()})
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        left_feedback: dict[str, Any] = {}
        right_feedback: dict[str, Any] = {}
        for key, value in feedback.items():
            if key.startswith("left_"):
                left_feedback[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_feedback[key.removeprefix("right_")] = value
        self.left_arm.send_feedback(left_feedback)
        self.right_arm.send_feedback(right_feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
