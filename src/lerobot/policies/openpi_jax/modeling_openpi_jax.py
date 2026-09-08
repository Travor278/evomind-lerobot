"""LeRobot adapter for an OpenPI JAX websocket policy server."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
from torch import Tensor
from websockets.sync.client import ClientConnection, connect

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from .configuration_openpi_jax import OpenPIJAXConfig

logger = logging.getLogger(__name__)


def _pack_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported OpenPI array dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


def _unpack_array(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class _OpenPIClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0) -> None:
        self.uri = f"ws://{host}:{port}"
        self.timeout_s = timeout_s
        self.connection: ClientConnection | None = None
        self.metadata: dict[str, Any] = {}

    def connect(self) -> None:
        connection = connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=self.timeout_s,
        )
        self.connection = connection
        self.metadata = _unpackb(connection.recv())

    def close(self) -> None:
        if self.connection is not None:
            # websockets 17.0.1 can hang in its close-frame serializer during
            # interpreter shutdown. The OpenPI protocol is request/response and
            # doesn't require an application-level close handshake.
            transport = getattr(self.connection, "socket", None)
            if transport is not None:
                transport.close()
            else:
                self.connection.close()
            self.connection = None

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            self.connect()
        assert self.connection is not None
        self.connection.send(_packb(observation))
        response = self.connection.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference server error: {response}")
        return _unpackb(response)


class OpenPIJAXPolicy(PreTrainedPolicy):
    """A parameter-free proxy whose numerical model lives in a JAX server process."""

    config_class = OpenPIJAXConfig
    name = "openpi_jax"

    def __init__(self, config: OpenPIJAXConfig) -> None:
        super().__init__(config)
        self._client: _OpenPIClient | None = None
        self._action_queue: deque[Tensor] = deque()
        self._action_q01: np.ndarray | None = None
        self._action_q99: np.ndarray | None = None
        if config.rtc_enabled:
            self._action_q01, self._action_q99 = self._load_action_stats(config.action_stats_path)
        if config.connect_on_load:
            self._ensure_client()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs: Any,
    ) -> OpenPIJAXPolicy:
        """Load proxy metadata without looking for a PyTorch model.safetensors file."""
        if config is None:
            config = OpenPIJAXConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, OpenPIJAXConfig):
            raise TypeError(f"Expected OpenPIJAXConfig, got {type(config).__name__}")
        return cls(config).eval()

    @property
    def type(self) -> str:
        return self.config.type

    def get_optim_params(self) -> dict[str, Tensor]:
        return {}

    def reset(self) -> None:
        self._action_queue.clear()

    def supports_rtc(self) -> bool:
        return bool(self.config.rtc_enabled)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise RuntimeError("OpenPI JAX websocket policies are inference-only")

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        del kwargs
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            for action in chunk[0, : self.config.n_action_steps]:
                self._action_queue.append(action)
        return self._action_queue.popleft().unsqueeze(0)

    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        observation = self._to_openpi_observation(batch)
        if self.config.rtc_enabled and prev_chunk_left_over is not None:
            observation["_rtc_prev_actions_normalized"] = self._normalize_rtc_prefix(prev_chunk_left_over)
            observation["_rtc_inference_delay"] = int(inference_delay)
            observation["_rtc_has_prev"] = True
        elif self.config.rtc_enabled:
            observation["_rtc_prev_actions_normalized"] = np.zeros(
                (self.config.rtc_model_horizon, self.config.rtc_model_action_dim), dtype=np.float32
            )
            observation["_rtc_inference_delay"] = 0
            observation["_rtc_has_prev"] = False

        result = self._infer_with_reconnect(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected_dim = self.config.output_features["action"].shape[-1]
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(f"OpenPI returned action shape {actions.shape}; expected [T, {expected_dim}]")
        if not np.isfinite(actions).all():
            raise ValueError("OpenPI returned non-finite actions")
        return torch.from_numpy(actions.copy()).unsqueeze(0)

    def _ensure_client(self) -> _OpenPIClient:
        if self._client is not None and self._client.connection is not None:
            return self._client

        client = _OpenPIClient(self.config.server_host, self.config.server_port)
        try:
            client.connect()
        except (OSError, TimeoutError):
            if not self.config.server_start_command:
                raise ConnectionError(
                    f"OpenPI JAX server is not listening at {client.uri} and no start command is configured"
                ) from None
            logger.info("Starting OpenPI JAX server: %s", self.config.server_start_command)
            subprocess.run(
                self.config.server_start_command,
                check=True,
                timeout=self.config.server_start_timeout_s,
            )
            deadline = time.monotonic() + self.config.server_start_timeout_s
            while True:
                try:
                    client.connect()
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for OpenPI JAX server at {client.uri}"
                        ) from None
                    time.sleep(1.0)
        self._client = client
        logger.info("Connected to OpenPI JAX server %s metadata=%s", client.uri, client.metadata)
        return client

    def _infer_with_reconnect(self, observation: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            return client.infer(observation)
        except Exception:
            logger.warning("OpenPI request failed; reconnecting once", exc_info=True)
            client.close()
            return self._ensure_client().infer(observation)

    def _to_openpi_observation(self, batch: dict[str, Any]) -> dict[str, Any]:
        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError(f"OpenPI JAX policy requires tensor input {OBS_STATE}")
        state_array = state.detach().cpu().float().numpy()
        if state_array.ndim == 2:
            if state_array.shape[0] != 1:
                raise ValueError("OpenPI JAX rollout supports batch size 1")
            state_array = state_array[0]

        images: dict[str, np.ndarray] = {}
        for policy_key, server_key in self.config.image_map.items():
            value = batch.get(policy_key)
            if not isinstance(value, Tensor):
                raise ValueError(f"OpenPI JAX policy is missing image tensor {policy_key}")
            image = value.detach().cpu()
            if image.ndim == 4:
                if image.shape[0] != 1:
                    raise ValueError("OpenPI JAX rollout supports batch size 1")
                image = image[0]
            if image.ndim != 3:
                raise ValueError(f"Image {policy_key} must be rank 3, got {tuple(image.shape)}")
            if image.shape[0] in (1, 3, 4):
                image = image.permute(1, 2, 0)
            image_array = image.numpy()
            if image_array.dtype != np.uint8:
                scale = 255.0 if image_array.size and float(np.nanmax(image_array)) <= 1.5 else 1.0
                image_array = np.clip(image_array * scale, 0, 255).astype(np.uint8)
            images[server_key] = np.ascontiguousarray(image_array)

        task = batch.get("task") if self.config.use_runtime_prompt else self.config.prompt
        task = task or self.config.prompt
        if isinstance(task, (list, tuple)):
            task = task[0]
        return {
            "state": np.asarray(state_array, dtype=np.float32),
            "images": images,
            "prompt": str(task or self.config.prompt),
        }

    @staticmethod
    def _load_action_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = payload.get("norm_stats", payload)["actions"]
        return np.asarray(stats["q01"], dtype=np.float32), np.asarray(stats["q99"], dtype=np.float32)

    def _normalize_rtc_prefix(self, actions: Tensor) -> np.ndarray:
        if self._action_q01 is None or self._action_q99 is None:
            raise RuntimeError("RTC action normalization statistics are not loaded")
        values = actions.detach().cpu().float().numpy()
        if values.ndim == 3:
            values = values[0]
        normalized = (values - self._action_q01) / (self._action_q99 - self._action_q01 + 1e-6)
        normalized = normalized * 2.0 - 1.0
        padded = np.zeros((self.config.rtc_model_horizon, self.config.rtc_model_action_dim), dtype=np.float32)
        steps = min(len(normalized), self.config.rtc_model_horizon)
        dims = min(normalized.shape[-1], self.config.rtc_model_action_dim)
        padded[:steps, :dims] = normalized[:steps, :dims]
        return padded

    def __del__(self) -> None:
        if self._client is not None:
            self._client.close()
