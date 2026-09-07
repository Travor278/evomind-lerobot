from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from lerobot_robot_evomind_piper.common import PIPER_ACTION_KEYS
from lerobot_robot_evomind_piper.devices import BiPiperXLeader, PiperXLeader

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
