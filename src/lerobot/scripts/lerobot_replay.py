# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.runtime_bridge import emit_runtime_event, take_runtime_commands
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # None preserves the dataset's recorded frame rate; an override changes playback speed.
    fps: float | None = None

    def __post_init__(self):
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("Replay fps must be finite and positive")


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    emit_runtime_event(
        "replay",
        "starting",
        robot_type=cfg.robot.type,
        repo_id=cfg.dataset.repo_id,
        episode=cfg.dataset.episode,
    )

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Materialize only action vectors before connecting. No parquet row conversion,
    # image decoding or observation read belongs in the replay control loop.
    names = dataset.features[ACTION]["names"]
    rows = dataset.select_columns(ACTION)
    actions = np.asarray([np.asarray(row[ACTION]) for row in rows], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(names) or not len(actions):
        raise ValueError("Episode actions must be a non-empty [frames, action dimensions] array")
    if not np.isfinite(actions).all():
        raise ValueError("Episode contains non-finite actions")
    if len(set(names)) != len(names) or set(names) != set(robot.action_features):
        raise ValueError("Episode action names do not match the configured robot")
    fps = cfg.dataset.fps if cfg.dataset.fps is not None else dataset.fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset replay fps must be finite and positive")
    interval = 1 / fps

    runtime_context = {
        "repo_id": cfg.dataset.repo_id,
        "episode": cfg.dataset.episode,
        "target_fps": fps,
        "dataset_fps": dataset.fps,
    }
    emit_runtime_event("replay", "connecting", **runtime_context)
    robot.connect()

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        emit_runtime_event("replay", "running", frame=0, total_frames=len(actions), **runtime_context)
        last_status_t = time.perf_counter()
        deadline = last_status_t
        started = last_status_t
        sent_frames = 0
        overruns = 0
        stopped = False
        first_send = None
        last_send = started
        for action_array in actions:
            if "stop" in take_runtime_commands():
                stopped = True
                break
            precise_sleep(max(deadline - time.perf_counter(), 0.0))
            if "stop" in take_runtime_commands():
                stopped = True
                break
            action = dict(zip(names, action_array.tolist(), strict=True))
            # The default processor is identity and requires no robot observation.
            processed_action = robot_action_processor((action, {}))

            last_send = time.perf_counter()
            if first_send is None:
                first_send = last_send
            _ = robot.send_action(processed_action)

            sent_frames += 1
            now = time.perf_counter()
            deadline += interval
            if now > deadline:
                # Rebase after slow hardware rather than sending catch-up bursts.
                overruns += 1
                deadline = now
            if now - last_status_t >= 0.5:
                emit_runtime_event(
                    "replay",
                    "running",
                    frame=sent_frames,
                    total_frames=len(actions),
                    elapsed_s=now - started,
                    deadline_misses=overruns,
                    effective_fps=(sent_frames - 1) / max(last_send - first_send, 1e-9),
                    **runtime_context,
                )
                last_status_t = now
        if not stopped:
            # Let the final frame occupy its period before disconnecting the robot.
            precise_sleep(max(deadline - time.perf_counter(), 0.0))
        elapsed_s = time.perf_counter() - started
    finally:
        emit_runtime_event("replay", "stopping", **runtime_context)
        robot.disconnect()
    emit_runtime_event(
        "replay",
        "completed",
        frame=sent_frames,
        total_frames=len(actions),
        stopped=stopped,
        deadline_misses=overruns,
        elapsed_s=elapsed_s,
        effective_fps=max(sent_frames - 1, 0) / max(last_send - (first_send or started), 1e-9),
        **runtime_context,
    )


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
