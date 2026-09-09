import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from lerobot.rollout import EpisodicDAggerStrategy, EpisodicDAggerStrategyConfig
from lerobot.rollout.strategies import dagger


def test_stopped_events_drop_pending_press():
    events = dagger.DAggerEvents()
    events.request_transition("pedal")
    events.stop_recording.set()
    assert events.consume_transition() is None
    events.request_transition("pedal")
    assert events.consume_transition() is None


def test_stop_during_sync_never_unlocks_or_enters_correction(monkeypatch):
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    strategy._engine, strategy._interpolator = MagicMock(), MagicMock()
    teleop, robot = MagicMock(), MagicMock()
    teleop.action_features = {"joint.pos": float}
    stop = threading.Event()
    ctx = SimpleNamespace(hardware=SimpleNamespace(teleop=teleop, robot_wrapper=robot),
                          runtime=SimpleNamespace(shutdown_event=stop))
    monkeypatch.setattr(dagger, "teleop_supports_feedback", lambda _: True)
    monkeypatch.setattr(dagger, "teleop_smooth_move_to", lambda *_: stop.set())
    strategy._prepare_teleop_reset(ctx, {"joint.pos": 1.0})
    assert strategy._events.phase == dagger.DAggerPhase.PAUSED
    teleop.disable_torque.assert_not_called()
    strategy._engine.resume.assert_not_called()
    robot.send_action.assert_not_called()


def test_stop_before_transition_does_not_resume_policy():
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    stop = threading.Event()
    stop.set()
    ctx = SimpleNamespace(runtime=SimpleNamespace(shutdown_event=stop))
    engine = MagicMock()
    strategy._apply_transition(dagger.DAggerPhase.CORRECTING, dagger.DAggerPhase.AUTONOMOUS,
                               engine, MagicMock(), ctx, None)
    engine.resume.assert_not_called()
    assert strategy._events.phase == dagger.DAggerPhase.PAUSED


def test_alignment_drops_pending_extra_press(monkeypatch):
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    events = strategy._events
    events.request_transition("pedal")
    old, new = events.consume_transition()
    events.request_transition("pedal")
    def align(*_):
        events.request_transition("pedal")
    monkeypatch.setattr(strategy, "_perform_transition", align)
    strategy._apply_transition(old, new, MagicMock(), MagicMock(), SimpleNamespace(), None)
    assert events.consume_transition() is None
    assert events.phase == dagger.DAggerPhase.PAUSED


@pytest.mark.parametrize("stopped", [False, True])
def test_explicit_stop_skips_home_but_normal_finish_preserves_config(monkeypatch, stopped):
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    stop = threading.Event()
    if stopped:
        stop.set()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_event=stop, cfg=SimpleNamespace(
            play_sounds=False, return_to_initial_position=True)),
        hardware=SimpleNamespace(), data=SimpleNamespace(dataset=None))
    teardown = MagicMock()
    monkeypatch.setattr(strategy, "_teardown_hardware", teardown)
    strategy.teardown(ctx)
    assert teardown.call_args.kwargs["return_to_initial_position"] is (not stopped)
