"""No hardware: exercise the actual Piper plugin with a fake SDK and clock."""
from types import SimpleNamespace as S

import pytest

from lerobot_robot_evomind_piper import devices, handover
from lerobot_robot_evomind_piper.common import PIPER_JOINT_NAMES, PIPER_ROLE_FOLLOWER, PIPER_ROLE_LEADER


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        assert duration >= 0
        self.now += max(duration, 0.000001)


class SDK:
    def __init__(self, clock, trace, side="left", *, frozen_joint=False,
                 frozen_gripper=False, frozen_enable=False, missing_motor=False):
        self.clock, self.trace, self.side = clock, trace, side
        self.frozen_joint, self.frozen_gripper = frozen_joint, frozen_gripper
        self.frozen_enable, self.missing_motor = frozen_enable, missing_motor
        self.enabled = False
        self.positions = [1000, 2000, -3000, 4000, -5000, 6000]
        self.gripper = 17000

    def log(self, command, *values):
        self.trace.append((self.clock.now, self.side, command, values))

    def GetArmJointMsgs(self):
        return S(time_stamp=1.0 if self.frozen_joint else self.clock.now,
                 joint_state=S(**dict(zip(PIPER_JOINT_NAMES, self.positions, strict=True))))

    def GetArmGripperMsgs(self):
        return S(time_stamp=1.0 if self.frozen_gripper else self.clock.now,
                 gripper_state=S(grippers_angle=self.gripper))

    def GetArmLowSpdInfoMsgs(self):
        return S(time_stamp=1.0 if self.frozen_enable else self.clock.now,
                 **{f"motor_{i}": S(foc_status=S(driver_enable_status=(
                     self.enabled and not (self.missing_motor and i == 6)))) for i in range(1, 7)})

    def GetArmJointCtrl(self):
        return S(time_stamp=self.clock.now)

    def GetArmGripperCtrl(self):
        return S(time_stamp=self.clock.now)

    def MasterSlaveConfig(self, *args):
        self.log("role", *args)

    def MotionCtrl_2(self, *args):
        self.log("mode", *args)

    def JointCtrl(self, *args):
        self.log("target", *args)

    def GripperCtrl(self, *args):
        self.log("gripper", *args)

    def EnableArm(self, *args):
        self.log("enable", *args)
        self.enabled = True


@pytest.fixture
def rig(monkeypatch):
    clock, trace = Clock(), []
    monkeypatch.setattr(handover, "time", clock)
    monkeypatch.setattr(devices, "time", clock)

    def make(side="left", **sdk_options):
        leader = object.__new__(devices.PiperXLeader)
        leader.config = devices.PiperXLeaderConfig(port=side, enable_timeout_s=0.8)
        leader._is_connected = True
        leader._manual_control_enabled = True
        leader._manual_action = None
        leader._last_mode_refresh_t = 0.0
        leader.arm = SDK(clock, trace, side, **sdk_options)
        return leader
    return clock, trace, make


def test_seed_current_joints_and_gripper_before_enable(rig):
    _, trace, make = rig
    leader = make()
    leader.set_manual_control(False)
    commands = [row[2] for row in trace]
    assert commands.index("mode") < commands.index("target") < commands.index("enable")
    assert commands.index("gripper") < commands.index("enable")
    assert all(row[3] == tuple(leader.arm.positions) for row in trace if row[2] == "target")
    assert all(row[3][0] == 17000 for row in trace if row[2] == "gripper")
    assert commands.count("target") > commands.count("enable")
    assert leader._manual_control_enabled is False
    # Do not re-send a role change after the holding target has been enabled.
    assert [r[3][0] for r in trace if r[2] == "role"] == [PIPER_ROLE_FOLLOWER]


def test_left_has_a_hold_before_waiting_for_right(rig):
    _, trace, make = rig
    bi = object.__new__(devices.BiPiperXLeader)
    bi.left_arm, bi.right_arm = make("left"), make("right")
    bi.set_manual_control(False)
    for side in ("left", "right"):
        first_enable = next(i for i, r in enumerate(trace) if r[1:3] == (side, "enable"))
        assert any(r[1:3] == (side, "target") for r in trace[:first_enable])
    right_start = next(i for i, r in enumerate(trace) if r[1] == "right")
    assert any(r[1:3] == ("left", "target") for r in trace[:right_start])


@pytest.mark.parametrize("fault", ["frozen_joint", "frozen_gripper"])
def test_stale_feedback_never_enables_or_writes_a_target(rig, fault):
    _, trace, make = rig
    leader = make(**{fault: True})
    with pytest.raises(RuntimeError, match="no fresh Piper feedback"):
        leader.set_manual_control(False)
    assert not any(r[2] in {"target", "enable", "mode", "gripper"} for r in trace)
    assert trace[-1][3][0] == PIPER_ROLE_LEADER
    assert leader._manual_control_enabled is None


@pytest.mark.parametrize("fault", ["frozen_enable", "missing_motor"])
def test_unconfirmed_enable_returns_to_manual_role(rig, fault):
    _, trace, make = rig
    leader = make(**{fault: True})
    with pytest.raises(RuntimeError, match="enable feedback did not advance"):
        leader.set_manual_control(False)
    assert trace[-1][2] == "role"
    assert trace[-1][3][0] == PIPER_ROLE_LEADER
    assert leader._manual_control_enabled is None


def test_nonfinite_joint_feedback_never_reaches_actuator(rig):
    _, trace, make = rig
    leader = make()
    leader.arm.positions[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        leader.set_manual_control(False)
    assert not any(r[2] in {"target", "enable"} for r in trace)


def test_manual_release_uses_new_feedback_without_position_commands(rig):
    _, trace, make = rig
    leader = make()
    leader._manual_control_enabled = False
    leader.set_manual_control(True)
    assert leader._manual_control_enabled is True
    assert leader._manual_action["joint_1.pos"] == 1.0
    assert leader._manual_action["gripper.pos"] == 17.0
    assert not any(r[2] in {"target", "enable", "mode", "gripper"} for r in trace)


def test_same_mode_request_does_not_reenable(rig):
    _, trace, make = rig
    leader = make()
    leader.set_manual_control(False)
    count = len(trace)
    leader.set_manual_control(False)
    assert len(trace) == count


def test_feedback_accepts_advancing_timestamps_not_just_positive(rig):
    clock, _, make = rig
    leader = make()
    begin = clock.now
    action = leader._wait_for_feedback_action()
    assert clock.now >= begin + 0.019
    assert action["joint_6.pos"] == 6.0


def test_no_gripper_configuration_does_not_wait_for_gripper(rig):
    _, trace, make = rig
    leader = make(frozen_gripper=True)
    leader.config.sync_gripper = False
    leader.set_manual_control(False)
    assert not any(r[2] == "gripper" for r in trace)
