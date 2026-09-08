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

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


class _EpisodeActivityGate:
    """Retain a short pre-roll and open recording on sustained follower motion."""

    def __init__(self, *, threshold: float, consecutive_frames: int, pre_roll_frames: int) -> None:
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self._pending: deque[dict[str, Any]] = deque(maxlen=max(1, pre_roll_frames))
        self._above_threshold = 0
        self.total_frames_seen = 0
        self.trimmed_frames = 0
        self.active = False
        self.just_activated = False

    @staticmethod
    def _state(frame: dict[str, Any]) -> np.ndarray:
        value = frame["observation.state"]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def push(self, frame: dict[str, Any], *, force_start: bool = False) -> list[dict[str, Any]]:
        self.just_activated = False
        self.total_frames_seen += 1
        if self.active:
            return [frame]

        state = self._state(frame)
        # Compare against the oldest frame in the pre-roll window instead of
        # the episode's initial pose. This detects actual recent motion while
        # ignoring slow servo creep that can accumulate over a long warm-up.
        reference = self._state(self._pending[0]) if self._pending else state
        self._pending.append(frame)
        displacement = float(np.max(np.abs(state - reference))) if state.size else 0.0
        self._above_threshold = self._above_threshold + 1 if displacement >= self.threshold else 0
        if not force_start and self._above_threshold < self.consecutive_frames:
            return []

        self.active = True
        self.just_activated = True
        ready = list(self._pending)
        self._pending.clear()
        self.trimmed_frames = max(0, self.total_frames_seen - len(ready))
        return ready


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

        with VideoEncodingManager(dataset), ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="episodic-dagger-save"
        ) as save_executor:
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
                    has_frames = outcome != "rerecord" and self._episode_has_frames(dataset)
                    save_future = None

                    if should_reset:
                        self._prepare_teleop_reset(ctx, last_action)
                        self._emit_reset_phase(recorded_episodes + 1, dataset_cfg.num_episodes)
                        # Streaming encoding has already written nearly all video frames. Finish the
                        # MP4 containers, parquet and metadata while the operator resets the scene,
                        # rather than exposing that disk work as dead time before the next episode.
                        if has_frames:
                            save_future = save_executor.submit(
                                save_episode_and_emit, dataset, ctx, strategy="episodic_dagger"
                            )
                        try:
                            self._run_reset(
                                ctx,
                                duration_s=dataset_cfg.reset_time_s,
                            )
                        finally:
                            if save_future is not None:
                                save_future.result()

                    if outcome == "rerecord":
                        dataset.clear_episode_buffer()
                        continue

                    if has_frames:
                        if save_future is None:
                            save_episode_and_emit(dataset, ctx, strategy="episodic_dagger")
                        self._needs_push.set()
                        recorded_episodes += 1

            finally:
                try:
                    self._engine.pause()
                except Exception:
                    logger.exception("Could not pause engine; still preserving recorded frames")
                # Preserve a partial episode when the process is interrupted.
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
        last_action = self._capture_hold_action(robot)
        activity_gate = None
        if self.config.trim_leading_idle:
            activity_gate = _EpisodeActivityGate(
                threshold=self.config.motion_start_threshold,
                consecutive_frames=self.config.motion_start_consecutive_frames,
                pre_roll_frames=max(1, round(self.config.motion_start_pre_roll_s * cfg.fps / record_stride)),
            )

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
                # Keep the last human/policy target active until the freshly
                # reset RTC queue has produced a replacement action.

            phase = self._events.phase
            obs = robot.get_observation()
            recovery_s, recovery_hold = self._handle_motor_bus_recovery(
                robot,
                obs,
                phase,
                engine,
                interpolator,
                strategy="episodic_dagger",
                runtime_data={"episode": episode, "total_episodes": total_episodes},
            )
            if recovery_s > 0:
                # A reconnect pause is neither task time nor a valid dataset frame.
                episode_started += recovery_s
                if recovery_hold is not None:
                    last_action = recovery_hold
                continue

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
                        activity_gate=activity_gate,
                        force_start=True,
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
                            activity_gate=activity_gate,
                        )
                    record_tick += 1
                elif last_action:
                    robot.send_action(last_action)

            elapsed = time.perf_counter() - loop_started
            precise_sleep(max(control_interval - elapsed, 0.0))

        if ctx.runtime.shutdown_event.is_set() or self._events.stop_recording.is_set():
            return "stopped", last_action
        return "finished", last_action

    @staticmethod
    def _record_frame(
        dataset,
        features,
        observation,
        action,
        task: str,
        *,
        intervention: bool,
        activity_gate: _EpisodeActivityGate | None = None,
        force_start: bool = False,
    ) -> None:
        obs_frame = build_dataset_frame(features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(features, action, prefix=ACTION)
        frame = {
            **obs_frame,
            **action_frame,
            "task": task,
            "intervention": np.array([intervention], dtype=bool),
        }
        ready = activity_gate.push(frame, force_start=force_start) if activity_gate is not None else [frame]
        for ready_frame in ready:
            dataset.add_frame(ready_frame)
        if activity_gate is not None and activity_gate.just_activated and activity_gate.trimmed_frames:
            logger.info("Dynamic episode trim removed %d leading idle frames", activity_gate.trimmed_frames)

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
            recovery_s, _ = self._handle_motor_bus_recovery(
                robot,
                obs,
                DAggerPhase.CORRECTING,
                self._engine,
                self._interpolator,
                strategy="episodic_dagger",
            )
            if recovery_s > 0:
                started += recovery_s
                continue
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
