# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Episode-oriented DAgger rollout with human intervention in every round."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import numpy as np

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.runtime_bridge import emit_runtime_event, take_runtime_commands
from lerobot.utils.utils import log_say

from ..configs import EpisodicDAggerStrategyConfig
from ..context import RolloutContext
from .core import save_episode_and_emit, send_next_action
from .dagger import DAggerPhase, DAggerStrategy

logger = logging.getLogger(__name__)


class EpisodicDAggerStrategy(DAggerStrategy):
    """Record complete episodes while allowing Policy/human handovers.

    Unlike continuous DAgger, episode boundaries are explicit: each round has
    a configured maximum duration, can be ended or discarded by the operator,
    and is followed by an unrecorded teleoperation reset phase.
    """

    config: EpisodicDAggerStrategyConfig

    def run(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        if dataset_cfg is None:
            raise RuntimeError("Episodic DAgger requires a dataset configuration")

        dataset = ctx.data.dataset
        if dataset is None:
            raise RuntimeError("Episodic DAgger dataset was not initialized")

        recorded_episodes = 0

        with VideoEncodingManager(dataset):
            try:
                while (
                    recorded_episodes < dataset_cfg.num_episodes
                    and not self._events.stop_recording.is_set()
                    and not ctx.runtime.shutdown_event.is_set()
                ):
                    outcome, last_action = self._run_episode(
                        ctx,
                        episode=recorded_episodes + 1,
                        total_episodes=dataset_cfg.num_episodes,
                        duration_s=dataset_cfg.episode_time_s,
                    )

                    if outcome == "stopped":
                        break

                    should_reset = outcome == "rerecord" or recorded_episodes < dataset_cfg.num_episodes - 1

                    if should_reset:
                        self._prepare_teleop_reset(ctx, last_action)
                        self._emit_reset_phase(recorded_episodes + 1, dataset_cfg.num_episodes)
                        self._run_reset(
                            ctx,
                            duration_s=dataset_cfg.reset_time_s,
                        )

                    if outcome == "rerecord":
                        dataset.clear_episode_buffer()
                        continue

                    if self._episode_has_frames(dataset):
                        save_episode_and_emit(dataset, ctx, strategy="episodic_dagger")
                        self._needs_push.set()
                        recorded_episodes += 1

            finally:
                self._engine.pause()
                # Preserve a partial episode when the process is interrupted.
                with contextlib.suppress(Exception):
                    if self._episode_has_frames(dataset):
                        save_episode_and_emit(dataset, ctx, strategy="episodic_dagger")
                        self._needs_push.set()

    def _run_episode(
        self,
        ctx: RolloutContext,
        *,
        episode: int,
        total_episodes: int,
        duration_s: float,
    ) -> tuple[str, dict[str, Any] | None]:
        engine = self._engine
        interpolator = self._interpolator
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        features = ctx.data.dataset_features
        cfg = ctx.runtime.cfg

        engine.reset()
        interpolator.reset()
        self._events.reset()
        engine.resume()

        control_interval = interpolator.get_control_interval(cfg.fps)
        record_stride = max(1, cfg.interpolation_multiplier)
        task = cfg.dataset.single_task if cfg.dataset else cfg.task
        episode_started = time.perf_counter()
        record_tick = 0
        last_action: dict[str, Any] | None = None

        self._emit_episode_phase(DAggerPhase.AUTONOMOUS, episode, total_episodes)
        log_say(f"Recording episode {episode}", cfg.play_sounds)

        while (
            time.perf_counter() - episode_started < duration_s
            and not self._events.stop_recording.is_set()
            and not ctx.runtime.shutdown_event.is_set()
        ):
            loop_started = time.perf_counter()
            commands = take_runtime_commands()

            if "rerecord_episode" in commands:
                return "rerecord", last_action
            if "finish_episode" in commands:
                return "finished", last_action
            if "pause_resume" in commands:
                self._events.request_transition("pause_resume")
            if "correction" in commands:
                self._events.request_transition("correction")

            transition = self._events.consume_transition()
            if transition is not None:
                old_phase, new_phase = transition
                self._apply_transition(old_phase, new_phase, engine, interpolator, ctx, last_action)
                self._emit_episode_phase(new_phase, episode, total_episodes)
                if new_phase == DAggerPhase.AUTONOMOUS:
                    last_action = None

            phase = self._events.phase
            obs = robot.get_observation()

            if phase == DAggerPhase.CORRECTING:
                obs_processed = ctx.processors.robot_observation_processor(obs)
                teleop_action = teleop.get_action()
                processed_teleop = ctx.processors.teleop_action_processor((teleop_action, obs))
                robot_action = ctx.processors.robot_action_processor((processed_teleop, obs))
                robot.send_action(robot_action)
                last_action = robot_action
                self._log_telemetry(obs_processed, processed_teleop, ctx.runtime)
                if record_tick % record_stride == 0:
                    self._record_frame(
                        dataset,
                        features,
                        obs_processed,
                        processed_teleop,
                        task,
                        intervention=True,
                    )
                record_tick += 1
            elif phase == DAggerPhase.PAUSED:
                if last_action:
                    robot.send_action(last_action)
            else:
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)
                if self._handle_warmup(cfg.use_torch_compile, loop_started, control_interval):
                    continue
                action = send_next_action(obs_processed, obs, ctx, interpolator)
                if action is not None:
                    last_action = ctx.processors.robot_action_processor((action, obs))
                    self._log_telemetry(obs_processed, action, ctx.runtime)
                    if record_tick % record_stride == 0:
                        self._record_frame(
                            dataset,
                            features,
                            obs_processed,
                            action,
                            task,
                            intervention=False,
                        )
                    record_tick += 1

            elapsed = time.perf_counter() - loop_started
            precise_sleep(max(control_interval - elapsed, 0.0))

        if ctx.runtime.shutdown_event.is_set() or self._events.stop_recording.is_set():
            return "stopped", last_action
        return "finished", last_action

    @staticmethod
    def _record_frame(dataset, features, observation, action, task: str, *, intervention: bool) -> None:
        obs_frame = build_dataset_frame(features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(features, action, prefix=ACTION)
        dataset.add_frame(
            {
                **obs_frame,
                **action_frame,
                "task": task,
                "intervention": np.array([intervention], dtype=bool),
            }
        )

    def _prepare_teleop_reset(
        self,
        ctx: RolloutContext,
        last_action: dict[str, Any] | None,
    ) -> None:
        """Enter human control using the same smooth handover as DAgger."""
        phase = self._events.phase
        if phase == DAggerPhase.AUTONOMOUS:
            self._apply_transition(
                DAggerPhase.AUTONOMOUS,
                DAggerPhase.PAUSED,
                self._engine,
                self._interpolator,
                ctx,
                last_action,
            )
            self._events.phase = DAggerPhase.PAUSED
            phase = DAggerPhase.PAUSED
        if phase == DAggerPhase.PAUSED:
            self._apply_transition(
                DAggerPhase.PAUSED,
                DAggerPhase.CORRECTING,
                self._engine,
                self._interpolator,
                ctx,
                last_action,
            )
            self._events.phase = DAggerPhase.CORRECTING

    def _run_reset(
        self,
        ctx: RolloutContext,
        *,
        duration_s: float,
    ) -> None:
        """Let the operator reset the scene without writing dataset frames."""
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        control_interval = 1.0 / ctx.runtime.cfg.fps
        started = time.perf_counter()

        while (
            time.perf_counter() - started < duration_s
            and not ctx.runtime.shutdown_event.is_set()
        ):
            loop_started = time.perf_counter()
            commands = take_runtime_commands()
            if "finish_episode" in commands:
                break

            obs = robot.get_observation()
            teleop_action = teleop.get_action()
            processed = ctx.processors.teleop_action_processor((teleop_action, obs))
            robot_action = ctx.processors.robot_action_processor((processed, obs))
            robot.send_action(robot_action)
            precise_sleep(max(control_interval - (time.perf_counter() - loop_started), 0.0))

    def _emit_episode_phase(self, phase: DAggerPhase, episode: int, total_episodes: int) -> None:
        emit_runtime_event(
            "rollout",
            "running",
            strategy="episodic_dagger",
            rollout_phase=phase.value,
            episode=episode,
            total_episodes=total_episodes,
            records_data=True,
            record_autonomous=True,
        )

    @staticmethod
    def _emit_reset_phase(episode: int, total_episodes: int) -> None:
        emit_runtime_event(
            "rollout",
            "running",
            strategy="episodic_dagger",
            rollout_phase="resetting",
            episode=episode,
            total_episodes=total_episodes,
            records_data=False,
            record_autonomous=True,
        )

    @staticmethod
    def _episode_has_frames(dataset) -> bool:
        return int(dataset.writer.episode_buffer["size"]) > 0
