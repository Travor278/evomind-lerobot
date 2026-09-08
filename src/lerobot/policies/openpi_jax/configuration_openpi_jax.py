"""Configuration for an OpenPI JAX checkpoint served outside the LeRobot process."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _default_input_features() -> dict[str, PolicyFeature]:
    return {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.right_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }


def _default_output_features() -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))}


def _default_image_map() -> dict[str, str]:
    return {
        "observation.images.base_0_rgb": "camera_front",
        "observation.images.left_wrist_0_rgb": "camera_wrist_left",
        "observation.images.right_wrist_0_rgb": "camera_wrist_right",
    }


@PreTrainedConfig.register_subclass("openpi_jax")
@dataclass
class OpenPIJAXConfig(PreTrainedConfig):
    """Metadata needed to expose an OpenPI JAX checkpoint as a LeRobot policy."""

    input_features: dict[str, PolicyFeature] = field(default_factory=_default_input_features)
    output_features: dict[str, PolicyFeature] = field(default_factory=_default_output_features)
    device: str | None = "cpu"

    checkpoint_path: str = ""
    prompt: str = ""
    use_runtime_prompt: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = 8100
    server_start_command: list[str] = field(default_factory=list)
    server_start_timeout_s: float = 180.0
    server_stop_command: list[str] = field(default_factory=list)
    server_stop_timeout_s: float = 30.0
    connect_on_load: bool = True

    chunk_size: int = 50
    n_action_steps: int = 25
    image_map: dict[str, str] = field(default_factory=_default_image_map)

    rtc_enabled: bool = True
    rtc_config: RTCConfig | None = None
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_model_horizon: int = 50
    rtc_model_action_dim: int = 32
    action_stats_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.checkpoint_path:
            raise ValueError("OpenPI JAX policy requires checkpoint_path")
        if not self.prompt:
            raise ValueError("OpenPI JAX policy requires prompt")
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.rtc_enabled and not self.action_stats_path:
            raise ValueError("RTC-enabled OpenPI JAX policy requires action_stats_path")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig()

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            raise ValueError(f"OpenPI JAX input_features must include {OBS_STATE}")
        if ACTION not in self.output_features:
            raise ValueError(f"OpenPI JAX output_features must include {ACTION}")
        missing_images = set(self.image_map) - set(self.input_features)
        if missing_images:
            raise ValueError(f"image_map contains unknown policy inputs: {sorted(missing_images)}")
