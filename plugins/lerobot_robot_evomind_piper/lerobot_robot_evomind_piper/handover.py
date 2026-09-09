"""Feedback and hold checks used only for Piper leader handovers.

SDK getters return mutable cached messages, not blocking hardware reads.
Require advancing timestamps; never infer fresh feedback from nonzero values.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy

from .common import PIPER_JOINT_NAMES, milli_to_unit


def _feedback_snapshot(arm, sync_gripper: bool):
    joints = deepcopy(arm.GetArmJointMsgs())
    joint_stamp = float(joints.time_stamp)
    if not math.isfinite(joint_stamp) or joint_stamp <= 0:
        return None
    action = {
        f"{name}.pos": milli_to_unit(getattr(joints.joint_state, name)) for name in PIPER_JOINT_NAMES
    }
    gripper_stamp = 0.0
    action["gripper.pos"] = 0.0
    if sync_gripper:
        gripper = deepcopy(arm.GetArmGripperMsgs())
        gripper_stamp = float(gripper.time_stamp)
        if not math.isfinite(gripper_stamp) or gripper_stamp <= 0:
            return None
        action["gripper.pos"] = abs(milli_to_unit(gripper.gripper_state.grippers_angle))
    if not all(math.isfinite(value) for value in action.values()):
        raise ValueError("Piper handover feedback contains non-finite positions")
    return (joint_stamp, gripper_stamp), action


def wait_fresh_feedback(arm, timeout_s: float, *, sync_gripper: bool):
    """Require two advancing snapshots after this call, including the gripper.

The SDK exposes aggregate joint timestamps, not one timestamp per joint;
this guards against a frozen cache, not every possible partial CAN failure.
"""
    initial = _feedback_snapshot(arm, sync_gripper)
    last = initial[0] if initial is not None else (0.0, 0.0)
    confirmed = 0
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        snapshot = _feedback_snapshot(arm, sync_gripper)
        if snapshot is not None:
            stamps, action = snapshot
            if stamps[0] > last[0] and (not sync_gripper or stamps[1] > last[1]):
                last = stamps
                confirmed += 1
                if confirmed >= 2:
                    return action
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return None


def enable_with_position_hold(arm, hold, timeout_s: float) -> bool:
    """Seed before EnableArm and refresh the same target while waiting.

Enable-state snapshots must advance twice and report all six motors enabled.
No thread is introduced: the existing per-arm worker remains the only writer.
"""
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_stamp = float(arm.GetArmLowSpdInfoMsgs().time_stamp)
    confirmed = 0
    next_enable = time.monotonic()
    next_check = next_enable + 0.2
    while time.monotonic() < deadline:
        hold()
        now = time.monotonic()
        if now >= next_enable:
            arm.EnableArm(7)
            next_enable = now + 0.2
        if now >= next_check:
            status = deepcopy(arm.GetArmLowSpdInfoMsgs())
            stamp = float(status.time_stamp)
            if math.isfinite(stamp) and stamp > last_stamp:
                last_stamp = stamp
                enabled = all(
                    getattr(status, f"motor_{i}").foc_status.driver_enable_status for i in range(1, 7)
                )
                confirmed = confirmed + 1 if enabled else 0
                if confirmed >= 2:
                    hold()
                    return True
            else:
                confirmed = 0
            next_check = now + 0.2
        time.sleep(min(1 / 30, max(0.0, deadline - time.monotonic())))
    return False
