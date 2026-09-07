"""Benchmark one local LeRobot checkpoint without connecting robot hardware."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _observation(policy_config) -> dict[str, np.ndarray]:
    from lerobot.configs import FeatureType

    observation: dict[str, np.ndarray] = {}
    for name, feature in policy_config.input_features.items():
        shape = tuple(int(item) for item in feature.shape)
        if feature.type == FeatureType.VISUAL:
            channels, height, width = shape
            if channels not in {1, 3, 4}:
                raise ValueError(f"Unsupported visual shape for {name}: {shape}")
            observation[name] = np.zeros((height, width, channels), dtype=np.uint8)
        elif feature.type == FeatureType.STATE:
            observation[name] = np.zeros(shape, dtype=np.float32)
    if not observation:
        raise ValueError("Checkpoint declares no benchmarkable observation features")
    return observation


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    import torch

    from lerobot.common.control_utils import predict_action
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.torch_threads)

    load_started = time.perf_counter()
    policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_config.device = str(device)
    policy_config.pretrained_path = str(checkpoint)
    if args.precision != "checkpoint" and hasattr(policy_config, "dtype"):
        policy_config.dtype = args.precision
    if hasattr(policy_config, "compile_model"):
        policy_config.compile_model = args.torch_compile
    if hasattr(policy_config, "compile_mode"):
        policy_config.compile_mode = args.compile_mode
    policy = get_policy_class(policy_config.type).from_pretrained(str(checkpoint), config=policy_config)
    policy = policy.to(device)
    policy.eval()
    overrides = {"device_processor": {"device": str(device)}}
    tokenizer = checkpoint / "tokenizer"
    if tokenizer.is_dir():
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=overrides,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_time_s = time.perf_counter() - load_started
    observation = _observation(policy_config)

    def infer_once() -> tuple[float, list[float]]:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        action = predict_action(
            {name: value.copy() for name, value in observation.items()},
            policy,
            device,
            preprocessor,
            postprocessor,
            use_amp=args.amp,
            task=args.task,
            robot_type=args.robot_type,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - started) * 1000, action.detach().float().cpu().reshape(-1).tolist()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    first_inference_ms, first_action = infer_once()
    for _ in range(args.warmup_runs):
        infer_once()
    latencies = [infer_once()[0] for _ in range(args.measured_runs)]
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "captured_at": captured_at,
        "environment_label": args.environment_label,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "driver": args.driver,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "load_time_s": load_time_s,
        "first_inference_ms": first_inference_ms,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "peak_memory_bytes": peak_memory,
        "notes": args.notes,
        "inference": {
            "device": str(device),
            "precision": getattr(policy_config, "dtype", None),
            "attention_backend": "eager" if policy_config.type in {"pi0", "pi05"} else "checkpoint",
            "torch_compile": {
                "enabled": args.torch_compile,
                "mode": args.compile_mode if args.torch_compile else None,
                "backend": "inductor" if args.torch_compile else None,
                "dynamic": None,
            },
            "environment_variables": {},
        },
        "versions": {
            "source": "validated_inference",
            "captured_at": captured_at,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.version() else None,
            "pytorch": _version("torch"),
            "transformers": _version("transformers"),
            "triton": _version("triton"),
            "jax": None,
            "jaxlib": None,
        },
        "samples_ms": latencies,
        **({"first_action": first_action, "seed": args.seed} if args.record_action else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--environment-label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--driver")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--robot-type", default="bi_piperx_follower")
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--precision", choices=("checkpoint", "float32", "bfloat16"), default="checkpoint")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--record-action", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.measured_runs <= 0 or args.torch_threads <= 0:
        parser.error("warmup-runs must be non-negative; measured-runs and torch-threads must be positive")
    result = benchmark(args)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
