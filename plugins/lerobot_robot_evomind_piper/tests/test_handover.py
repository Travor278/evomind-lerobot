from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from lerobot_robot_evomind_piper.common import PIPER_ACTION_KEYS, wait_enable_piper
from lerobot_robot_evomind_piper.devices import BiPiperXLeader, PiperXLeader, _PiperLeaderProcessProxy

from lerobot.common.control_utils import teleop_supports_feedback
from lerobot.rollout import DAggerStrategyConfig
from lerobot.rollout.strategies import DAggerPhase, DAggerStrategy


def _piper_arm() -> MagicMock:
    arm = MagicMock()
    arm.action_features = dict.fromkeys(PIPER_ACTION_KEYS, float)
    arm.feedback_features = dict.fromkeys(PIPER_ACTION_KEYS, float)
    arm.is_connected = True
    return arm


def _bi_piper_leader() -> BiPiperXLeader:
    leader = object.__new__(BiPiperXLeader)
    leader.left_arm = _piper_arm()
    leader.right_arm = _piper_arm()
    return leader


def test_single_piper_leader_reports_actuated_feedback_and_switches_roles() -> None:
    leader = object.__new__(PiperXLeader)
    leader.set_manual_control = MagicMock()

    assert set(leader.feedback_features) == set(PIPER_ACTION_KEYS)
    assert teleop_supports_feedback(leader)

    leader.enable_torque()
    leader.set_manual_control.assert_called_once_with(False)

    leader.disable_torque()
    leader.set_manual_control.assert_called_with(True)


def test_bi_piper_leader_reports_actuated_feedback_and_switches_roles() -> None:
    leader = _bi_piper_leader()

    assert set(leader.feedback_features) == {
        *(f"left_{key}" for key in PIPER_ACTION_KEYS),
        *(f"right_{key}" for key in PIPER_ACTION_KEYS),
    }
    assert teleop_supports_feedback(leader)

    leader.enable_torque()
    leader.left_arm.set_manual_control.assert_called_once_with(False)
    leader.right_arm.set_manual_control.assert_called_once_with(False)

    leader.disable_torque()
    leader.left_arm.set_manual_control.assert_called_with(True)
    leader.right_arm.set_manual_control.assert_called_with(True)


def test_wait_enable_requires_fresh_consecutive_status_reads() -> None:
    arm = MagicMock()
    arm.GetArmEnableStatus.side_effect = [[False] * 6, [True] * 6, [True] * 6]

    assert wait_enable_piper(arm, timeout_s=1.0, retry_interval_s=0.01)
    assert arm.EnableArm.call_count == 3


def test_bi_piper_leader_rolls_back_partial_mode_switch() -> None:
    leader = _bi_piper_leader()
    leader.right_arm.set_manual_control.side_effect = [RuntimeError("right failed"), None]

    with pytest.raises(RuntimeError, match="right failed"):
        leader.set_manual_control(False)

    assert leader.left_arm.set_manual_control.call_args_list[0].args == (False,)
    assert leader.left_arm.set_manual_control.call_args_list[-1].args == (True,)
    assert leader.right_arm.set_manual_control.call_args_list[-1].args == (True,)


def test_bi_piper_leader_overlaps_isolated_action_reads() -> None:
    calls = MagicMock()
    left = object.__new__(_PiperLeaderProcessProxy)
    right = object.__new__(_PiperLeaderProcessProxy)
    left._is_connected = True
    right._is_connected = True
    left._begin_call = calls.left_begin
    right._begin_call = calls.right_begin
    left._finish_call = calls.left_finish
    right._finish_call = calls.right_finish
    left._finish_call.return_value = {"joint_1.pos": 1.0}
    right._finish_call.return_value = {"joint_1.pos": 2.0}
    leader = object.__new__(BiPiperXLeader)
    leader.left_arm = left
    leader.right_arm = right

    action = leader.get_action()

    assert action == {"left_joint_1.pos": 1.0, "right_joint_1.pos": 2.0}
    assert calls.mock_calls == [
        call.left_begin("get_action"),
        call.right_begin("get_action"),
        call.left_finish("get_action"),
        call.right_finish("get_action"),
    ]


def test_dagger_moves_piper_leader_to_follower_instead_of_the_reverse(monkeypatch) -> None:
    import lerobot.rollout.strategies.dagger as dagger_module

    leader = _bi_piper_leader()
    robot = MagicMock()
    engine = MagicMock()
    move_leader = MagicMock()
    move_follower = MagicMock()
    monkeypatch.setattr(dagger_module, "teleop_smooth_move_to", move_leader)
    monkeypatch.setattr(dagger_module, "follower_smooth_move_to", move_follower)

    context = SimpleNamespace(
        hardware=SimpleNamespace(teleop=leader, robot_wrapper=robot),
        processors=SimpleNamespace(
            teleop_action_processor=MagicMock(),
            robot_action_processor=MagicMock(),
        ),
    )
    action = {f"left_{key}": 1.0 for key in PIPER_ACTION_KEYS}
    action.update({f"right_{key}": 2.0 for key in PIPER_ACTION_KEYS})
    strategy = DAggerStrategy(DAggerStrategyConfig())

    strategy._apply_transition(
        DAggerPhase.AUTONOMOUS,
        DAggerPhase.PAUSED,
        engine,
        MagicMock(),
        context,
        action,
    )
    move_leader.assert_called_once_with(leader, action)

    strategy._apply_transition(
        DAggerPhase.PAUSED,
        DAggerPhase.CORRECTING,
        engine,
        MagicMock(),
        context,
        action,
    )
    move_follower.assert_not_called()
    robot.send_action.assert_not_called()


def test_dagger_handover_failure_releases_leader_and_stays_paused(monkeypatch) -> None:
    import lerobot.rollout.strategies.dagger as dagger_module

    leader = _bi_piper_leader()
    monkeypatch.setattr(
        dagger_module,
        "teleop_smooth_move_to",
        MagicMock(side_effect=RuntimeError("handover failed")),
    )
    context = SimpleNamespace(
        hardware=SimpleNamespace(teleop=leader, robot_wrapper=MagicMock()),
        processors=SimpleNamespace(
            teleop_action_processor=MagicMock(),
            robot_action_processor=MagicMock(),
        ),
    )
    strategy = DAggerStrategy(DAggerStrategyConfig())

    strategy._apply_transition(
        DAggerPhase.AUTONOMOUS,
        DAggerPhase.PAUSED,
        MagicMock(),
        MagicMock(),
        context,
        {f"left_{key}": 1.0 for key in PIPER_ACTION_KEYS},
    )

    leader.left_arm.set_manual_control.assert_called_with(True)
    leader.right_arm.set_manual_control.assert_called_with(True)
