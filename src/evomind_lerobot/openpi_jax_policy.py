"""Registration helpers for external OpenPI JAX checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.openpi_jax.configuration_openpi_jax import OpenPIJAXConfig
from lerobot.policies.openpi_jax.processor_openpi_jax import make_openpi_jax_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

PolicyBackend = Literal["pytorch", "jax", "unknown"]


def detect_policy_backend(policy_path: str | Path, config: dict[str, Any] | None = None) -> PolicyBackend:
    """Identify the numerical backend from files and registered policy metadata."""
    path = Path(policy_path)
    config = config or {}
    if config.get("type") == "openpi_jax":
        return "jax"

    checkpoint_path = Path(str(config.get("checkpoint_path") or path))
    if (checkpoint_path / "params").is_dir():
        return "jax"
    if (path / "model.safetensors").is_file():
        return "pytorch"
    return "unknown"


def register_openpi_jax_policy(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    prompt: str,
    action_stats_path: str | Path,
    server_host: str = "127.0.0.1",
    server_port: int = 8100,
    server_start_command: list[str] | None = None,
    state_dim: int = 12,
    action_dim: int = 12,
    image_size: int = 224,
    image_map: dict[str, str] | None = None,
    chunk_size: int = 50,
    n_action_steps: int = 25,
    rtc_enabled: bool = True,
) -> Path:
    """Create a lightweight LeRobot policy directory pointing at a JAX checkpoint."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if detect_policy_backend(checkpoint) != "jax":
        raise ValueError(f"No OpenPI JAX params directory found under {checkpoint}")

    stats = Path(action_stats_path).expanduser().resolve()
    if rtc_enabled and not stats.is_file():
        raise ValueError(f"RTC action statistics not found: {stats}")

    destination = Path(output_path).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_image_map = image_map or {
        "observation.images.base_0_rgb": "camera_front",
        "observation.images.left_wrist_0_rgb": "camera_wrist_left",
        "observation.images.right_wrist_0_rgb": "camera_wrist_right",
    }
    inputs = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    inputs.update(
        {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size))
            for key in resolved_image_map
        }
    )
    config = OpenPIJAXConfig(
        input_features=inputs,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        checkpoint_path=str(checkpoint),
        prompt=prompt,
        server_host=server_host,
        server_port=server_port,
        server_start_command=list(server_start_command or []),
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        image_map=resolved_image_map,
        rtc_enabled=rtc_enabled,
        action_stats_path=str(stats),
    )
    config.validate_features()
    config.save_pretrained(destination)
    preprocessor, postprocessor = make_openpi_jax_pre_post_processors(config)
    preprocessor.save_pretrained(destination, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(destination, config_filename="policy_postprocessor.json")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an OpenPI JAX checkpoint in EvoMind")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--action-stats", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8100)
    parser.add_argument("--server-start-command-json", default="[]")
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-map-json", default="")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=25)
    parser.add_argument("--disable-rtc", action="store_true")
    args = parser.parse_args()

    command = json.loads(args.server_start_command_json)
    image_map = json.loads(args.image_map_json) if args.image_map_json else None
    output = register_openpi_jax_policy(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        prompt=args.prompt,
        action_stats_path=args.action_stats,
        server_host=args.server_host,
        server_port=args.server_port,
        server_start_command=command,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        image_size=args.image_size,
        image_map=image_map,
        chunk_size=args.chunk_size,
        n_action_steps=args.action_steps,
        rtc_enabled=not args.disable_rtc,
    )
    print(output)


if __name__ == "__main__":
    main()
