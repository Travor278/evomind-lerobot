from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def run_replay(monkeypatch, *, fps=10, override=None, costs=None, frames=12, stop_at=None, invalid=False):
    from lerobot.scripts import lerobot_replay as module

    clock = SimpleNamespace(now=0.0)
    sent, events = [], []
    costs = iter(costs or [0.001] * frames)
    rows = [{"action": np.array([float(i)])} for i in range(frames)]
    if invalid:
        rows[0]["action"][0] = np.nan

    def selected(_column):
        for row in rows:
            clock.now += 0.03  # Parquet/torch conversion belongs before robot connection.
            yield row

    dataset = SimpleNamespace(
        fps=fps, num_frames=frames, features={"action": {"names": ["joint.pos"]}}, select_columns=selected
    )
    robot = MagicMock()
    robot.action_features = {"joint.pos": float}
    robot.get_observation.side_effect = AssertionError("Replay does not need observation I/O")

    def send(action):
        sent.append((clock.now, action))
        clock.now += next(costs)

    robot.send_action.side_effect = send

    def emit(operation, phase, **data):
        events.append((phase, data))
        clock.now += 0.02  # Status/IPC overhead must not accumulate in every deadline.

    monkeypatch.setattr(module, "LeRobotDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(module, "init_logging", lambda: None)
    monkeypatch.setattr(module, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "emit_runtime_event", emit)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(module, "precise_sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    monkeypatch.setattr(
        module,
        "take_runtime_commands",
        lambda: {"stop"} if stop_at is not None and len(sent) >= stop_at else set(),
    )
    cfg = module.ReplayConfig(
        robot=SimpleNamespace(type="fake"),
        dataset=module.DatasetReplayConfig(repo_id="test/offline", episode=0, fps=override),
        play_sounds=False,
    )
    if invalid:
        with pytest.raises(ValueError, match="non-finite"):
            module.replay.__wrapped__(cfg)
        robot.connect.assert_not_called()
    else:
        module.replay.__wrapped__(cfg)
        robot.disconnect.assert_called_once()
    return sent, events


def test_replay_uses_recorded_fps_without_io_drift(monkeypatch):
    sent, events = run_replay(monkeypatch)
    np.testing.assert_allclose(np.diff([t for t, _ in sent]), 0.1, atol=1e-9)
    assert [a["joint.pos"] for _, a in sent] == list(range(12))
    assert events[-1][1]["target_fps"] == 10
    assert events[-1][1]["frame"] == 12


def test_replay_honors_explicit_fps(monkeypatch):
    sent, _ = run_replay(monkeypatch, override=20)
    np.testing.assert_allclose(np.diff([t for t, _ in sent]), 0.05, atol=1e-9)


def test_replay_does_not_burst_after_a_slow_send(monkeypatch):
    sent, events = run_replay(monkeypatch, frames=4, costs=[0.25, 0.001, 0.001, 0.001])
    np.testing.assert_allclose(np.diff([t for t, _ in sent]), [0.25, 0.1, 0.1], atol=1e-9)
    assert events[-1][1]["deadline_misses"] == 1


def test_replay_stop_reports_actual_sent_frames(monkeypatch):
    sent, events = run_replay(monkeypatch, stop_at=3)
    assert len(sent) == 3
    assert events[-1][1]["stopped"] is True
    assert events[-1][1]["frame"] == 3
    assert events[-1][1]["total_frames"] == 12


def test_replay_rejects_bad_actions_before_connect(monkeypatch):
    run_replay(monkeypatch, invalid=True)


def test_web_replay_passes_selected_root_and_fps_without_cameras(monkeypatch):
    from evomind_lerobot import runtime_service
    from lerobot.scripts import lerobot_replay

    monkeypatch.setattr(
        runtime_service,
        "_dataset",
        lambda _: {
            "id": "test/offline",
            "path": "/actual/local/dataset",
            "robot_type": "fake",
            "episodes": 2,
            "fps": 30,
        },
    )
    monkeypatch.setattr(runtime_service, "_configuration", lambda: SimpleNamespace(robot_type="fake"))
    decode = MagicMock(return_value=(SimpleNamespace(type="fake"), None))
    monkeypatch.setattr(runtime_service, "_decode_hardware", decode)
    replay = MagicMock()
    monkeypatch.setattr(lerobot_replay, "replay", replay)
    runtime_service._execute_replay({"dataset_id": "test/offline", "episode": 1, "fps": 15})
    cfg = replay.call_args.args[0]
    assert cfg.dataset.root == "/actual/local/dataset"
    assert cfg.dataset.episode == 1
    assert cfg.dataset.fps == 15
    assert decode.call_args.kwargs == {"cameras": False, "include_teleoperator": False}
