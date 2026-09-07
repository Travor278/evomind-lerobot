from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from evomind_lerobot import policy_runtime
from evomind_lerobot.policy_runtime import (
    BenchmarkRecord,
    ExecutableEnvironment,
    InferenceSettings,
    PolicyRuntimeManifest,
    TrainingEnvironment,
    inspect_runtime_manifest,
    load_runtime_manifest,
    runtime_launch_spec,
    save_runtime_manifest,
)


def pytorch_manifest(**kwargs) -> PolicyRuntimeManifest:
    defaults = {
        "framework": "pytorch",
        "training": TrainingEnvironment(
            python="3.12.13",
            cuda="12.8",
            cudnn="91002",
            pytorch="2.10.0",
            transformers="5.0.0",
            triton="3.6.0",
        ),
        "environment": ExecutableEnvironment(kind="current"),
        "inference": InferenceSettings(precision="bfloat16", attention_backend="eager"),
    }
    defaults.update(kwargs)
    return PolicyRuntimeManifest(**defaults)


def installed_versions() -> dict[str, str | None]:
    return {
        "python": "3.12.8",
        "pytorch": "2.10.0",
        "jax": None,
        "jaxlib": None,
        "transformers": "5.0.0",
        "triton": "3.6.0",
    }


def test_manifest_round_trip_and_refuses_accidental_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    manifest = pytorch_manifest(
        benchmarks=[
            BenchmarkRecord(
                captured_at=datetime(2026, 9, 7, tzinfo=UTC),
                environment_label="torch-2.10-cu128",
                device="cuda:0",
                gpu="RTX 4090",
                warmup_runs=10,
                measured_runs=100,
                load_time_s=11.2,
                first_inference_ms=1850,
                latency_p50_ms=92,
                latency_p95_ms=98,
                peak_memory_bytes=8_000_000_000,
            )
        ]
    )

    path = save_runtime_manifest(checkpoint, manifest)

    assert path.name == "evomind-runtime.json"
    assert load_runtime_manifest(checkpoint) == manifest
    with pytest.raises(FileExistsError):
        save_runtime_manifest(checkpoint, manifest)

    args = policy_runtime._parser().parse_args(
        [
            "configure",
            str(checkpoint),
            "--rollout-backend",
            "rtc",
            "--precision",
            "bfloat16",
            "--torch-compile",
            "disabled",
        ]
    )
    assert args.handler(args) == 0
    configured = load_runtime_manifest(checkpoint)
    assert configured.inference.rollout_backend == "rtc"
    assert configured.inference.precision == "bfloat16"
    assert configured.benchmarks == manifest.benchmarks


def test_manifest_requires_framework_versions_and_immutable_container() -> None:
    with pytest.raises(ValidationError, match="training.pytorch"):
        PolicyRuntimeManifest(framework="pytorch", training=TrainingEnvironment(python="3.12"))
    with pytest.raises(ValidationError, match="image@sha256"):
        ExecutableEnvironment(kind="container", reference="evomind/pi05:latest")


def test_reads_training_precision_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "policy": {
                    "type": "pi05",
                    "dtype": "float32",
                    "use_amp": False,
                    "compile_model": False,
                    "compile_mode": "max-autotune",
                }
            }
        ),
        encoding="utf-8",
    )

    assert policy_runtime.checkpoint_training_settings(checkpoint) == {
        "precision": "float32",
        "use_amp": False,
        "attention_backend": "eager",
        "torch_compile": False,
        "torch_compile_mode": "max-autotune",
        "settings_source": "train_config.json",
    }


def test_manifest_inspection_reports_current_environment_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "pretrained_model"
    save_runtime_manifest(checkpoint, pytorch_manifest())
    monkeypatch.setattr(policy_runtime, "current_environment", installed_versions)

    matching = inspect_runtime_manifest(checkpoint)
    assert matching["status"] == "configured"
    assert matching["compatibility"] == {"compatible": True, "issues": []}

    monkeypatch.setattr(
        policy_runtime,
        "current_environment",
        lambda: {**installed_versions(), "pytorch": "2.9.1", "python": "3.11.9"},
    )
    mismatch = inspect_runtime_manifest(checkpoint)
    assert mismatch["compatibility"]["compatible"] is False
    assert any("Python" in issue for issue in mismatch["compatibility"]["issues"])
    assert any("pytorch" in issue for issue in mismatch["compatibility"]["issues"])
    with pytest.raises(ValueError, match="does not match checkpoint"):
        runtime_launch_spec(checkpoint)


def test_python_environment_resolves_relative_to_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "pretrained_model"
    executable = checkpoint / "runtime" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    save_runtime_manifest(
        checkpoint,
        pytorch_manifest(
            environment=ExecutableEnvironment(kind="python", reference="runtime/bin/python"),
            inference=InferenceSettings(environment_variables={"TORCH_LOGS": "recompiles"}),
        ),
    )
    monkeypatch.setattr(policy_runtime, "current_environment", installed_versions)

    launch = runtime_launch_spec(checkpoint)

    assert launch.python_executable == str(executable)
    assert launch.environment_variables == {"TORCH_LOGS": "recompiles"}
    assert inspect_runtime_manifest(checkpoint)["manifest"]["inference"]["environment_variables"] == {
        "TORCH_LOGS": "<configured>"
    }


def test_manifest_inspection_is_resilient_to_missing_and_invalid_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy_runtime, "current_environment", installed_versions)
    checkpoint = tmp_path / "pretrained_model"

    assert inspect_runtime_manifest(checkpoint)["status"] == "missing"
    checkpoint.mkdir()
    (checkpoint / "evomind-runtime.json").write_text("{not-json", encoding="utf-8")
    invalid = inspect_runtime_manifest(checkpoint)
    assert invalid["status"] == "invalid"
    assert invalid["error"]


def test_workspace_inventory_includes_checkpoint_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "policies" / "run-1" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"type": "pi05"}), encoding="utf-8")
    save_runtime_manifest(checkpoint, pytorch_manifest())
    constants = types.ModuleType("lerobot.utils.constants")
    constants.HF_LEROBOT_HOME = tmp_path
    utils = types.ModuleType("lerobot.utils")
    utils.constants = constants
    monkeypatch.setitem(sys.modules, "lerobot.utils", utils)
    monkeypatch.setitem(sys.modules, "lerobot.utils.constants", constants)
    workspace = importlib.import_module("evomind_lerobot.workspace")
    workspace = importlib.reload(workspace)
    monkeypatch.setattr(policy_runtime, "current_environment", installed_versions)

    policies = workspace.policies_inventory()

    assert len(policies) == 1
    assert policies[0]["id"] == str(Path("run-1") / "pretrained_model")
    assert policies[0]["runtime"]["status"] == "configured"
    assert policies[0]["runtime"]["manifest"]["framework"] == "pytorch"
