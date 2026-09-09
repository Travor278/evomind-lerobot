from __future__ import annotations

import contextlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.utils.pedal import PedalPressFilter


def test_pedal_listener_closes_device_and_stops(monkeypatch):
    from lerobot.utils import pedal as module

    device = MagicMock()
    device.name = "Test FootSwitch"
    pending = [SimpleNamespace(type=1, code=30, value=1, keycode="KEY_A")]
    received = threading.Event()
    monkeypatch.setitem(
        sys.modules,
        "evdev",
        SimpleNamespace(
            InputDevice=lambda _: device, ecodes=SimpleNamespace(EV_KEY=1), categorize=lambda event: event
        ),
    )

    def select_ready(*args):
        if pending:
            return [device], [], []
        time.sleep(0.005)
        return [], [], []

    device.read.side_effect = lambda: [pending.pop(0)]
    monkeypatch.setattr(module.select, "select", select_ready)
    listener = module.start_pedal_listener(lambda _: received.set(), "/dev/input/fake-pedal")
    try:
        assert received.wait(1)
    finally:
        listener.stop()
    assert not listener.is_alive()
    device.grab.assert_called_once()
    device.close.assert_called_once()
    assert listener.status["state"] == "stopped"


def test_pedal_permission_failure_is_structured(monkeypatch):
    from lerobot.utils import pedal as module

    def denied(_):
        raise PermissionError("permission denied")

    monkeypatch.setitem(sys.modules, "evdev", SimpleNamespace(
        InputDevice=denied, categorize=lambda event: event, ecodes=SimpleNamespace(EV_KEY=1)
    ))
    listener = module.start_pedal_listener(lambda _: None, "/dev/input/fake-pedal")
    assert listener.status["state"] == "reconnecting"
    assert "permission denied" in listener.status["error"]
    listener.stop()
    assert not listener.is_alive()


def test_web_bridge_does_not_disable_the_pedal(monkeypatch):
    from lerobot.rollout.configs import DAggerStrategyConfig
    from lerobot.rollout.strategies import dagger as module

    strategy = module.DAggerStrategy(DAggerStrategyConfig(num_episodes=1))
    ctx = SimpleNamespace(
        data=SimpleNamespace(dataset_features={}, dataset=None),
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(fps=30, play_sounds=False, return_to_initial_position=False)
        ),
        hardware=SimpleNamespace(),
    )
    listener = MagicMock()
    monkeypatch.setattr(strategy, "_init_engine", lambda _: None)
    monkeypatch.setattr(strategy, "_teardown_hardware", lambda *a, **kw: None)
    monkeypatch.setattr(module, "runtime_bridge_active", lambda: True)
    monkeypatch.setattr(module, "resolve_pedal_device", lambda: "/dev/input/fake-pedal")
    monkeypatch.setattr(module, "estimate_max_episode_seconds", lambda *a, **kw: 30)
    init = MagicMock(return_value=listener)
    monkeypatch.setattr(module, "_init_dagger_pedal", init)
    strategy.setup(ctx)
    strategy.teardown(ctx)
    assert init.call_args.args[1].device_path == "/dev/input/fake-pedal"
    listener.stop.assert_called_once()


def test_pedal_ignores_hold_repeat_and_contact_bounce():
    pedal = PedalPressFilter()
    assert pedal.accept("KEY_A", 1, 0)
    assert not pedal.accept("KEY_A", 1, 1)
    assert not pedal.accept("KEY_A", 2, 1.1)
    assert not pedal.accept("KEY_A", 0, 1.2)
    assert pedal.accept("KEY_A", 1, 1.3)
    assert not pedal.accept("KEY_A", 0, 1.31)
    assert not pedal.accept("KEY_A", 1, 1.32)
    assert not pedal.accept("KEY_A", 0, 1.33)
    assert pedal.accept("KEY_A", 1, 1.6)


def test_single_pedal_cycles_all_three_phases(monkeypatch):
    from lerobot.rollout.configs import DAggerPedalConfig
    from lerobot.rollout.strategies import dagger as module

    callbacks = []
    monkeypatch.setattr(module, "start_pedal_listener", lambda callback, **_: callbacks.append(callback))
    events = module.DAggerEvents()
    module._init_dagger_pedal(events, DAggerPedalConfig())
    phases = []
    for _ in range(3):
        callbacks[0]("KEY_A")
        phases.append(events.consume_transition()[1].value)
    assert phases == ["paused", "correcting", "autonomous"]
    events.transitioning.set()
    callbacks[0]("KEY_A")
    assert events.consume_transition() is None


def test_manual_press_ends_episode_and_separate_press_skips_reset(monkeypatch):
    from lerobot.scripts import lerobot_record as module

    robot = MagicMock()
    event = threading.Event()
    events = {
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
        "_pedal_pressed": event,
    }
    monkeypatch.setattr(module, "take_runtime_commands", lambda: set())
    event.set()
    module.record_loop(
        robot,
        events,
        30,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        dataset=SimpleNamespace(fps=30),
        control_time_s=10,
    )
    assert not event.is_set()
    assert not events["exit_early"]
    event.set()
    module.record_loop(robot, events, 30, lambda x: x, lambda x: x, lambda x: x, control_time_s=10)
    assert not event.is_set()
    robot.get_observation.assert_not_called()
    robot.send_action.assert_not_called()


def test_policy_pedal_records_human_frames_only_after_sync(monkeypatch):
    from lerobot.rollout import EpisodicDAggerStrategy, EpisodicDAggerStrategyConfig
    from lerobot.rollout.strategies import dagger, episodic_dagger as module

    order = []
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    strategy._engine = MagicMock()
    strategy._engine.pause.side_effect = lambda: order.append("pause")
    strategy._engine.resume.side_effect = lambda: order.append("resume")
    strategy._interpolator = MagicMock()
    strategy._interpolator.get_control_interval.return_value = 0.001
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.5}
    teleop = MagicMock()
    teleop.get_action.return_value = {"joint.pos": 0.7}
    teleop.disable_torque.side_effect = lambda: order.append("unlock")
    frames = []
    dataset = SimpleNamespace(add_frame=frames.append)
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=teleop),
        data=SimpleNamespace(dataset=dataset, dataset_features={}),
        processors=SimpleNamespace(
            robot_observation_processor=lambda x: x,
            teleop_action_processor=lambda x: x[0],
            robot_action_processor=lambda x: x[0],
        ),
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                fps=30,
                interpolation_multiplier=1,
                use_torch_compile=False,
                dataset=SimpleNamespace(single_task="test"),
                play_sounds=False,
            ),
            shutdown_event=threading.Event(),
        ),
    )
    commands = iter([set(), {"pedal"}, {"pedal"}, {"pedal"}, {"finish_episode"}])
    monkeypatch.setattr(module, "take_runtime_commands", lambda: next(commands))
    monkeypatch.setattr(module, "build_dataset_frame", lambda _f, value, *, prefix: {prefix: value})
    monkeypatch.setattr(module, "send_next_action", lambda *_: {"joint.pos": 0.5})
    monkeypatch.setattr(dagger, "teleop_supports_feedback", lambda _: True)
    monkeypatch.setattr(dagger, "teleop_smooth_move_to", lambda *_: order.append("sync"))
    monkeypatch.setattr(strategy, "_process_observation_and_notify", lambda _p, obs: obs)
    monkeypatch.setattr(strategy, "_handle_warmup", lambda *_: False)
    monkeypatch.setattr(strategy, "_log_telemetry", lambda *_: None)
    outcome, _ = strategy._run_episode(ctx, episode=1, total_episodes=1, duration_s=10)
    assert outcome == "finished"
    assert [f["intervention"].item() for f in frames] == [False, True, False]
    assert order.index("pause") < order.index("sync") < order.index("unlock")
    assert order[-2:] == ["resume", "unlock"]
    assert strategy._engine.reset.call_count == 2


def test_failed_leader_sync_cannot_enter_human_control(monkeypatch):
    from lerobot.rollout.configs import DAggerStrategyConfig
    from lerobot.rollout.strategies import dagger as module

    strategy = module.DAggerStrategy(DAggerStrategyConfig())
    engine, interpolator, teleop = MagicMock(), MagicMock(), MagicMock()
    teleop.action_features = {"joint.pos": float}
    ctx = SimpleNamespace(hardware=SimpleNamespace(teleop=teleop, robot_wrapper=MagicMock()))
    monkeypatch.setattr(module, "teleop_supports_feedback", lambda _: True)
    monkeypatch.setattr(module, "teleop_smooth_move_to", MagicMock(side_effect=RuntimeError("sync failed")))
    for _ in range(2):
        strategy._events.request_transition("pedal")
        old, new = strategy._events.consume_transition()
        strategy._apply_transition(old, new, engine, interpolator, ctx, {"joint.pos": 0.5})
    assert strategy._events.phase == module.DAggerPhase.PAUSED
    assert strategy._handover_error == "sync failed"
    engine.resume.assert_not_called()


def test_pause_before_first_policy_action_aligns_to_observation(monkeypatch):
    from lerobot.rollout.configs import DAggerStrategyConfig
    from lerobot.rollout.strategies import dagger as module

    strategy = module.DAggerStrategy(DAggerStrategyConfig())
    teleop, robot = MagicMock(), MagicMock()
    teleop.action_features = {"joint.pos": float}
    robot.get_observation.return_value = {"joint.pos": 0.7}
    ctx = SimpleNamespace(hardware=SimpleNamespace(teleop=teleop, robot_wrapper=robot))
    monkeypatch.setattr(module, "teleop_supports_feedback", lambda _: True)
    sync = MagicMock()
    monkeypatch.setattr(module, "teleop_smooth_move_to", sync)
    strategy._events.request_transition("pedal")
    old, new = strategy._events.consume_transition()
    strategy._apply_transition(old, new, MagicMock(), MagicMock(), ctx, None)
    sync.assert_called_once_with(teleop, {"joint.pos": 0.7})
    assert strategy._handover_error is None


def test_failed_sync_cannot_bypass_guard_via_episode_reset():
    from lerobot.rollout import EpisodicDAggerStrategy, EpisodicDAggerStrategyConfig
    from lerobot.rollout.strategies.dagger import DAggerPhase

    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig())
    strategy._events.phase = DAggerPhase.PAUSED
    strategy._handover_error = "sync failed"
    strategy._engine = MagicMock()
    strategy._interpolator = MagicMock()
    ctx = SimpleNamespace(hardware=SimpleNamespace(teleop=MagicMock(), robot_wrapper=MagicMock()))
    with pytest.raises(RuntimeError, match="Cannot enter teleoperation reset"):
        strategy._prepare_teleop_reset(ctx, {"joint.pos": 0.5})
    assert strategy._events.phase == DAggerPhase.PAUSED


def test_rtc_pause_drains_inflight_chunk_before_reset(monkeypatch):
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import rtc as module

    entered, release, paused = threading.Event(), threading.Event(), threading.Event()
    policy = MagicMock()

    def predict(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return torch.zeros((1, 4, 1))

    policy.predict_action_chunk.side_effect = predict
    pre = MagicMock(side_effect=lambda x: x)
    pre.steps = []
    post = MagicMock(side_effect=lambda x: x)
    engine = module.RTCInferenceEngine(
        policy, pre, post, SimpleNamespace(robot_type="fake"), RTCConfig(), {}, "test", 30, "cpu"
    )
    monkeypatch.setattr(module, "build_dataset_frame", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "prepare_observation_for_inference", lambda *_: {})
    engine.start()
    engine.notify_observation({"state": 1})
    engine.resume()
    waiter = None
    try:
        assert entered.wait(3)
        waiter = threading.Thread(target=lambda: (engine.pause(), paused.set()))
        waiter.start()
        assert not paused.wait(0.05)
        release.set()
        assert paused.wait(3)
        engine.reset()
        assert engine.action_queue.qsize() == 0
        assert engine._obs_holder["obs"] is None
        engine.resume()
        time.sleep(0.03)
        assert policy.predict_action_chunk.call_count == 1
    finally:
        release.set()
        if waiter is not None:
            waiter.join(timeout=3)
        engine.stop()


@pytest.mark.parametrize("rerecord", [False, True])
def test_record_saves_before_reset_and_does_not_reset_after_last_episode(monkeypatch, rerecord):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts import lerobot_record as module

    order = []
    robot, teleop, dataset = MagicMock(), MagicMock(), MagicMock()
    robot.name = "fake"
    robot.cameras = {}
    robot.action_features = robot.observation_features = {"joint.pos": float}
    dataset.num_episodes = 0
    dataset.writer.episode_buffer = {"size": 0}

    def save():
        order.append("save")
        dataset.num_episodes += 1
        dataset.writer.episode_buffer["size"] = 0

    def clear():
        dataset.writer.episode_buffer["size"] = 0

    dataset.save_episode.side_effect = save
    dataset.clear_episode_buffer.side_effect = clear
    events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}

    def loop(**kwargs):
        if kwargs.get("dataset") is not None:
            order.append("record")
            dataset.writer.episode_buffer["size"] = 3
            if rerecord and order.count("record") == 1:
                events["rerecord_episode"] = True
        else:
            order.append("reset")
            assert dataset.writer.episode_buffer["size"] == 0

    monkeypatch.setattr(module, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(module, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(module.LeRobotDataset, "create", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, "record_loop", loop)
    monkeypatch.setattr(module, "init_keyboard_listener", lambda: (None, events))
    monkeypatch.setattr(module, "resolve_pedal_device", lambda *_: None)
    monkeypatch.setattr(module, "VideoEncodingManager", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(module, "init_logging", lambda: None)
    monkeypatch.setattr(module, "log_say", lambda *args, **kwargs: None)
    cfg = module.RecordConfig(
        robot=SimpleNamespace(type="fake"),
        teleop=SimpleNamespace(type="fake"),
        dataset=DatasetRecordConfig(
            repo_id="test/offline",
            single_task="test",
            video=False,
            push_to_hub=False,
            num_episodes=1 if rerecord else 2,
        ),
        play_sounds=False,
    )
    module.record.__wrapped__(cfg)
    assert order == (
        ["record", "reset", "record", "save"] if rerecord else ["record", "save", "reset", "record", "save"]
    )
