"""Exercise native replay using real dataset actions and a robot with no hardware I/O."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_replay as replay_module
from lerobot.utils.constants import ACTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    dataset = LeRobotDataset(
        args.dataset.name, root=args.dataset, episodes=[args.episode], download_videos=False
    )
    columns = dataset.select_columns([ACTION, "timestamp"])
    rows = columns.select(range(min(args.frames, len(columns))))
    names = dataset.features[ACTION]["names"]
    values = np.asarray([np.asarray(row[ACTION]) for row in rows])
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sample = SimpleNamespace(
        fps=dataset.fps,
        features=dataset.features,
        num_frames=len(rows),
        select_columns=lambda _: rows.select_columns(ACTION),
    )

    class OfflineRobot:
        action_features = dict.fromkeys(names, float)

        def __init__(self):
            self.actions = []
            self.sent_at = []

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_observation(self):
            raise AssertionError("Replay must not poll robot observations")

        def send_action(self, action):
            self.sent_at.append(time.perf_counter())
            self.actions.append([action[name] for name in names])
            time.sleep(0.001)  # Stand-in for command transmission, never opens a device.
            return action

    robot = OfflineRobot()
    config = replay_module.ReplayConfig(
        robot=SimpleNamespace(type="offline_probe"),
        dataset=replay_module.DatasetReplayConfig(
            repo_id=args.dataset.name, root=args.dataset, episode=args.episode, fps=args.fps
        ),
        play_sounds=False,
    )
    with (
        patch.object(replay_module, "make_robot_from_config", return_value=robot),
        patch.object(replay_module, "LeRobotDataset", return_value=sample),
    ):
        replay_module.replay.__wrapped__(config)
    np.testing.assert_array_equal(np.asarray(robot.actions), values)
    intervals = np.diff(robot.sent_at)
    report = {
        "dataset": str(args.dataset),
        "episode": args.episode,
        "frames": len(rows),
        "dataset_fps": dataset.fps,
        "target_fps": args.fps or dataset.fps,
        "effective_fps": float(1 / intervals.mean()),
        "interval_p50_ms": float(np.percentile(intervals, 50) * 1000),
        "interval_p95_ms": float(np.percentile(intervals, 95) * 1000),
        "stored_timestamp_interval_ms": float(np.median(np.diff(timestamps)) * 1000),
        "action_values_unchanged": True,
        "hardware_access": False,
        "note": "Recorded timestamps are the dataset timeline, not proof of original wall-clock capture rate.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
