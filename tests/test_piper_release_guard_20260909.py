from types import SimpleNamespace as S
from unittest.mock import MagicMock

import pytest

from lerobot.rollout import DAggerStrategyConfig
from lerobot.rollout.strategies import dagger


@pytest.mark.parametrize("resume_policy", [False, True])
def test_failed_release_remains_paused_without_finalizing_dataset(monkeypatch, resume_policy):
    strategy = dagger.DAggerStrategy(DAggerStrategyConfig())
    teleop, engine, robot, dataset = (MagicMock() for _ in range(4))
    teleop.disable_torque.side_effect = RuntimeError("no fresh position feedback")
    ctx = S(hardware=S(teleop=teleop, robot_wrapper=robot), data=S(dataset=dataset))
    monkeypatch.setattr(dagger, "teleop_supports_feedback", lambda _: True)
    old = dagger.DAggerPhase.CORRECTING if resume_policy else dagger.DAggerPhase.PAUSED
    new = dagger.DAggerPhase.AUTONOMOUS if resume_policy else dagger.DAggerPhase.CORRECTING
    strategy._events.phase = new
    strategy._apply_transition(old, new, engine, MagicMock(), ctx, {"joint.pos": 1.0})
    assert strategy._events.phase == dagger.DAggerPhase.PAUSED
    assert strategy._handover_error == "no fresh position feedback"
    engine.pause.assert_called_once()
    dataset.finalize.assert_not_called()
    dataset.clear_episode_buffer.assert_not_called()
    robot.send_action.assert_not_called()
