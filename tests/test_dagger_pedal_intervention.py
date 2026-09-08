from __future__ import annotations

from unittest.mock import MagicMock


def test_single_pedal_starts_correction_then_resumes_policy() -> None:
    from lerobot.rollout.strategies.dagger import DAggerEvents, DAggerPhase

    events = DAggerEvents()

    # A pedal press cannot take over before the phone has paused the policy
    # and aligned the leader arm.
    events.request_transition("pedal_intervention")
    assert events.consume_transition() is None

    events.request_transition("pause_resume")
    assert events.consume_transition() == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    events.request_transition("pedal_intervention")
    assert events.consume_transition() == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    events.request_transition("pedal_intervention")
    assert events.consume_transition() == (DAggerPhase.CORRECTING, DAggerPhase.AUTONOMOUS)


def test_startup_hold_latches_current_robot_pose() -> None:
    from lerobot.rollout import DAggerStrategy, DAggerStrategyConfig

    robot = MagicMock()
    robot.action_features = {"left_joint.pos": float, "right_joint.pos": float}
    robot.get_observation.return_value = {
        "left_joint.pos": 1.25,
        "right_joint.pos": -2.5,
        "observation.images.camera": object(),
    }

    strategy = DAggerStrategy(DAggerStrategyConfig())
    hold_action = strategy._capture_hold_action(robot)

    assert hold_action == {"left_joint.pos": 1.25, "right_joint.pos": -2.5}
    robot.send_action.assert_called_once_with(hold_action)
