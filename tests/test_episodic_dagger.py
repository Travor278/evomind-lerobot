from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


def _context(*, num_episodes: int = 1):
    dataset_cfg = SimpleNamespace(
        num_episodes=num_episodes,
        episode_time_s=30,
        reset_time_s=10,
        single_task="insert screw",
    )
    cfg = SimpleNamespace(
        dataset=dataset_cfg,
        duration=120,
        fps=30,
        interpolation_multiplier=1,
        play_sounds=False,
    )
    dataset = MagicMock()
    dataset.num_episodes = 0
    dataset.writer.episode_buffer = {"size": 0}

    def clear_episode_buffer():
        dataset.writer.episode_buffer["size"] = 0

    dataset.clear_episode_buffer.side_effect = clear_episode_buffer
    return SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=threading.Event()),
        data=SimpleNamespace(dataset=dataset),
    )


def test_episodic_dagger_saves_each_complete_round(monkeypatch) -> None:
    import lerobot.rollout.strategies.episodic_dagger as module
    from lerobot.rollout import EpisodicDAggerStrategy, EpisodicDAggerStrategyConfig

    ctx = _context(num_episodes=2)
    # Episodic DAgger is bounded by episode count and per-episode time, not by
    # the legacy rollout-wide duration field.
    ctx.runtime.cfg.duration = 1e-9
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig(num_episodes=2))
    strategy._engine = MagicMock()
    strategy._interpolator = MagicMock()

    def run_episode(*_args, **_kwargs):
        ctx.data.dataset.writer.episode_buffer["size"] = 12
        return "finished", {"joint.pos": 1.0}

    saved = []

    def save_episode(dataset, _ctx, *, strategy):
        saved.append(strategy)
        dataset.num_episodes += 1
        dataset.writer.episode_buffer["size"] = 0

    monkeypatch.setattr(module, "VideoEncodingManager", lambda _dataset: contextlib.nullcontext())
    monkeypatch.setattr(module, "save_episode_and_emit", save_episode)
    monkeypatch.setattr(strategy, "_run_episode", run_episode)
    monkeypatch.setattr(strategy, "_prepare_teleop_reset", MagicMock())
    monkeypatch.setattr(strategy, "_run_reset", MagicMock())

    strategy.run(ctx)

    assert saved == ["episodic_dagger", "episodic_dagger"]
    strategy._prepare_teleop_reset.assert_called_once()
    strategy._run_reset.assert_called_once()
    strategy._engine.pause.assert_called_once()


def test_episodic_dagger_discards_rerecorded_round(monkeypatch) -> None:
    import lerobot.rollout.strategies.episodic_dagger as module
    from lerobot.rollout import EpisodicDAggerStrategy, EpisodicDAggerStrategyConfig

    ctx = _context()
    strategy = EpisodicDAggerStrategy(EpisodicDAggerStrategyConfig(num_episodes=1))
    strategy._engine = MagicMock()
    strategy._interpolator = MagicMock()
    outcomes = iter(("rerecord", "finished"))

    def run_episode(*_args, **_kwargs):
        ctx.data.dataset.writer.episode_buffer["size"] = 8
        return next(outcomes), {"joint.pos": 1.0}

    saved = []

    def save_episode(dataset, _ctx, *, strategy):
        saved.append(strategy)
        dataset.num_episodes += 1
        dataset.writer.episode_buffer["size"] = 0

    monkeypatch.setattr(module, "VideoEncodingManager", lambda _dataset: contextlib.nullcontext())
    monkeypatch.setattr(module, "save_episode_and_emit", save_episode)
    monkeypatch.setattr(strategy, "_run_episode", run_episode)
    monkeypatch.setattr(strategy, "_prepare_teleop_reset", MagicMock())
    monkeypatch.setattr(strategy, "_run_reset", MagicMock())

    strategy.run(ctx)

    assert saved == ["episodic_dagger"]
    ctx.data.dataset.clear_episode_buffer.assert_called_once()
    strategy._run_reset.assert_called_once()


def test_episodic_dagger_records_control_source(monkeypatch) -> None:
    import lerobot.rollout.strategies.episodic_dagger as module
    from lerobot.rollout import EpisodicDAggerStrategy

    dataset = MagicMock()
    monkeypatch.setattr(
        module,
        "build_dataset_frame",
        lambda _features, value, *, prefix: {f"{prefix}.value": value},
    )

    EpisodicDAggerStrategy._record_frame(
        dataset,
        {},
        "observation",
        "action",
        "task",
        intervention=True,
    )

    frame = dataset.add_frame.call_args.args[0]
    assert frame["task"] == "task"
    assert frame["intervention"].tolist() == [True]
