"""Checkpoint-local runtime environment metadata.

Each policy checkpoint may contain an ``evomind-runtime.json`` file next to
``config.json``.  The manifest records the environment used to train the
checkpoint, the executable environment that should run it, inference tuning,
and reproducible offline benchmark results.

This module deliberately keeps discovery cheap: listing policies never imports
PyTorch or JAX and therefore does not initialize an accelerator in the web
server process.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUNTIME_MANIFEST_NAME = "evomind-runtime.json"
_VERSION_PACKAGES = {
    "pytorch": "torch",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "transformers": "transformers",
    "triton": "triton",
}


class TrainingEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["training_capture", "checkpoint_metadata", "validated_inference"] = "training_capture"
    captured_at: datetime | None = None
    python: str = Field(min_length=1)
    cuda: str | None = None
    cudnn: str | None = None
    pytorch: str | None = None
    jax: str | None = None
    jaxlib: str | None = None
    transformers: str | None = None
    triton: str | None = None
    precision: str | None = None
    use_amp: bool | None = None
    attention_backend: str | None = None
    torch_compile: bool | None = None
    torch_compile_mode: str | None = None
    settings_source: str | None = None


class CompileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: str | None = None
    backend: str | None = None
    dynamic: bool | None = None


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cuda"
    precision: str = "bfloat16"
    attention_backend: str = "checkpoint"
    rollout_backend: Literal["sync", "rtc"] = "sync"
    torch_compile: CompileSettings = Field(default_factory=CompileSettings)
    environment_variables: dict[str, str] = Field(default_factory=dict)


class ExecutableEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["current", "python", "uv", "conda", "container"] = "current"
    reference: str | None = None
    working_directory: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> ExecutableEnvironment:
        if self.kind != "current" and not (self.reference and self.reference.strip()):
            raise ValueError(f"environment.reference is required for kind={self.kind}")
        if self.kind == "uv" and self.reference and Path(self.reference).name != "uv.lock":
            raise ValueError("uv environments must reference a uv.lock file")
        if self.kind == "container" and self.reference and "@sha256:" not in self.reference:
            raise ValueError("container environments must use an immutable image@sha256:... reference")
        return self


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captured_at: datetime
    environment_label: str = Field(min_length=1)
    device: str = Field(min_length=1)
    gpu: str | None = None
    driver: str | None = None
    warmup_runs: int = Field(ge=0)
    measured_runs: int = Field(gt=0)
    load_time_s: float = Field(ge=0)
    first_inference_ms: float = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    notes: str | None = None
    versions: TrainingEnvironment | None = None
    inference: InferenceSettings | None = None
    samples_ms: list[float] = Field(default_factory=list)


class PolicyRuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    framework: Literal["pytorch", "jax"]
    training: TrainingEnvironment
    environment: ExecutableEnvironment = Field(default_factory=ExecutableEnvironment)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    benchmarks: list[BenchmarkRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_framework_version(self) -> PolicyRuntimeManifest:
        if self.framework == "pytorch" and not self.training.pytorch:
            raise ValueError("training.pytorch is required for a PyTorch checkpoint")
        if self.framework == "jax" and not (self.training.jax and self.training.jaxlib):
            raise ValueError("training.jax and training.jaxlib are required for a JAX checkpoint")
        return self


@dataclass(frozen=True)
class RuntimeLaunchSpec:
    python_executable: str
    environment_variables: dict[str, str]
    environment_kind: str
    environment_reference: str | None
    working_directory: str | None
    manifest: PolicyRuntimeManifest | None


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def current_environment() -> dict[str, str | None]:
    """Return cheap package metadata without importing accelerator frameworks."""
    return {
        "python": platform.python_version(),
        **{field: _package_version(distribution) for field, distribution in _VERSION_PACKAGES.items()},
    }


def capture_training_environment(
    framework: Literal["pytorch", "jax"],
    *,
    source: Literal["training_capture", "checkpoint_metadata", "validated_inference"] = "training_capture",
) -> TrainingEnvironment:
    """Capture versions from the active training or inference environment."""
    installed = current_environment()
    cuda = None
    cudnn = None
    if framework == "pytorch":
        try:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, torch; "
                        "print(json.dumps({'cuda': torch.version.cuda, "
                        "'cudnn': torch.backends.cudnn.version()}))"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            details = json.loads(probe.stdout)
            cuda = details.get("cuda")
            cudnn_version = details.get("cudnn")
            cudnn = str(cudnn_version) if cudnn_version is not None else None
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return TrainingEnvironment(
        source=source,
        captured_at=datetime.now(UTC),
        python=str(installed["python"]),
        cuda=cuda,
        cudnn=cudnn,
        pytorch=installed["pytorch"],
        jax=installed["jax"],
        jaxlib=installed["jaxlib"],
        transformers=installed["transformers"],
        triton=installed["triton"],
    )


def load_runtime_manifest(checkpoint: Path) -> PolicyRuntimeManifest:
    path = checkpoint / RUNTIME_MANIFEST_NAME
    return PolicyRuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def checkpoint_training_settings(checkpoint: Path) -> dict[str, Any]:
    """Read training precision evidence saved alongside a checkpoint."""
    for filename in ("train_config.json", "config.json"):
        path = checkpoint / filename
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        policy = value.get("policy", value)
        if not isinstance(policy, dict):
            continue
        policy_type = str(policy.get("type") or value.get("type") or "")
        precision = policy.get("dtype")
        use_amp = policy.get("use_amp")
        compile_enabled = policy.get("compile_model")
        compile_mode = policy.get("compile_mode")
        attention_backend = policy.get("attention_backend") or policy.get("attn_implementation")
        if not attention_backend and policy_type in {"pi0", "pi05"}:
            attention_backend = "eager"
        if any(item is not None for item in (precision, use_amp, compile_enabled, attention_backend)):
            return {
                "precision": str(precision) if precision is not None else None,
                "use_amp": bool(use_amp) if use_amp is not None else None,
                "attention_backend": str(attention_backend) if attention_backend is not None else None,
                "torch_compile": bool(compile_enabled) if compile_enabled is not None else None,
                "torch_compile_mode": str(compile_mode) if compile_mode is not None else None,
                "settings_source": filename,
            }
    return {}


def save_runtime_manifest(
    checkpoint: Path, manifest: PolicyRuntimeManifest, *, overwrite: bool = False
) -> Path:
    path = checkpoint / RUNTIME_MANIFEST_NAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"Runtime manifest already exists: {path}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _major_minor(value: str) -> tuple[int, int] | None:
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _reference_availability(checkpoint: Path, environment: ExecutableEnvironment) -> bool | None:
    if environment.kind == "current":
        return True
    reference = environment.reference or ""
    if environment.kind == "python":
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = checkpoint / path
        return path.is_file()
    if environment.kind == "uv":
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = checkpoint / path
        return path.is_file() and _environment_python(path.parent / ".venv").is_file()
    if environment.kind == "conda":
        return shutil.which("conda") is not None
    if environment.kind == "container":
        return shutil.which("docker") is not None or shutil.which("podman") is not None
    return None


def _environment_python(prefix: Path) -> Path:
    windows = prefix / "Scripts" / "python.exe"
    return windows if windows.is_file() else prefix / "bin" / "python"


def _resolve_path(checkpoint: Path, reference: str) -> Path:
    path = Path(reference).expanduser()
    return path if path.is_absolute() else (checkpoint / path).resolve()


def _conda_python(reference: str) -> Path:
    prefix = Path(reference).expanduser()
    if prefix.is_dir():
        return _environment_python(prefix)
    conda = shutil.which("conda")
    if conda is None:
        raise ValueError("Conda executable was not found")
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        environments = json.loads(result.stdout).get("envs", [])
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ValueError(f"Could not list Conda environments: {error}") from error
    match = next((Path(item) for item in environments if Path(item).name == reference), None)
    if match is None:
        raise ValueError(f"Conda environment was not found: {reference}")
    return _environment_python(match)


def runtime_launch_spec(checkpoint: Path) -> RuntimeLaunchSpec:
    """Resolve the interpreter and environment that must own policy imports."""
    inspection = inspect_runtime_manifest(checkpoint)
    if inspection["status"] == "missing":
        return RuntimeLaunchSpec(str(Path(sys.executable).resolve()), {}, "current", None, None, None)
    if inspection["status"] == "invalid":
        raise ValueError(f"Invalid {RUNTIME_MANIFEST_NAME}: {inspection['error']}")

    manifest = load_runtime_manifest(checkpoint)
    if manifest.framework == "jax":
        raise ValueError(
            "JAX runtime metadata is valid, but this LeRobot rollout path requires a JAX policy runner adapter"
        )
    environment = manifest.environment
    if environment.kind == "current":
        issues = inspection["compatibility"]["issues"]
        if issues:
            raise ValueError("Current environment does not match checkpoint: " + "; ".join(issues))
        executable = Path(sys.executable).resolve()
    elif environment.kind == "python":
        executable = _resolve_path(checkpoint, environment.reference or "")
    elif environment.kind == "uv":
        lockfile = _resolve_path(checkpoint, environment.reference or "")
        if not lockfile.is_file():
            raise ValueError(f"uv.lock was not found: {lockfile}")
        executable = _environment_python(lockfile.parent / ".venv")
    elif environment.kind == "conda":
        executable = _conda_python(environment.reference or "")
    else:
        raise ValueError(
            "Container runtime manifests are recorded but require the isolated container runner "
            f"before they can execute: {environment.reference}"
        )
    if not executable.is_file():
        raise ValueError(f"Runtime Python executable was not found: {executable}")
    working_directory = environment.working_directory
    if working_directory:
        resolved_working_directory = _resolve_path(checkpoint, working_directory)
        if not resolved_working_directory.is_dir():
            raise ValueError(f"Runtime working directory was not found: {resolved_working_directory}")
        working_directory = str(resolved_working_directory)
    return RuntimeLaunchSpec(
        python_executable=str(executable),
        environment_variables=dict(manifest.inference.environment_variables),
        environment_kind=environment.kind,
        environment_reference=environment.reference,
        working_directory=working_directory,
        manifest=manifest,
    )


def validate_active_runtime(checkpoint: Path) -> PolicyRuntimeManifest | None:
    """Validate the selected child environment, including its CUDA runtime."""
    path = checkpoint / RUNTIME_MANIFEST_NAME
    if not path.is_file():
        return None
    manifest = load_runtime_manifest(checkpoint)
    if manifest.framework == "jax":
        raise ValueError("A JAX policy runner adapter is required for this checkpoint")
    actual = capture_training_environment(manifest.framework)
    issues: list[str] = []
    if _major_minor(actual.python) != _major_minor(manifest.training.python):
        issues.append(f"Python {actual.python} != {manifest.training.python}")
    for field in ("pytorch", "cuda", "cudnn", "transformers", "triton"):
        expected = getattr(manifest.training, field)
        observed = getattr(actual, field)
        if expected and observed != expected:
            issues.append(f"{field} {observed or 'not installed'} != {expected}")
    if issues:
        raise ValueError(
            "Selected runtime does not match checkpoint training environment: " + "; ".join(issues)
        )
    return manifest


def inspect_runtime_manifest(checkpoint: Path) -> dict[str, Any]:
    """Read and validate one checkpoint manifest without breaking inventory scans."""
    path = checkpoint / RUNTIME_MANIFEST_NAME
    installed = current_environment()
    base: dict[str, Any] = {
        "manifest_path": str(path),
        "status": "missing",
        "error": None,
        "manifest": None,
        "current": installed,
        "compatibility": {"compatible": None, "issues": []},
        "environment_available": None,
    }
    if not path.is_file():
        return base
    try:
        manifest = load_runtime_manifest(checkpoint)
    except (OSError, ValueError) as error:
        return {**base, "status": "invalid", "error": str(error)}

    issues: list[str] = []
    if manifest.environment.kind == "current":
        expected_python = _major_minor(manifest.training.python)
        actual_python = _major_minor(str(installed["python"]))
        if expected_python and actual_python and expected_python != actual_python:
            issues.append(
                f"Python {installed['python']} does not match training Python {manifest.training.python}"
            )
        fields = ["transformers", "triton"]
        fields.append("pytorch" if manifest.framework == "pytorch" else "jax")
        if manifest.framework == "jax":
            fields.append("jaxlib")
        for field in fields:
            expected = getattr(manifest.training, field)
            actual = installed[field]
            if expected and actual != expected:
                issues.append(f"{field} {actual or 'not installed'} does not match training {expected}")
        compatible: bool | None = not issues
    else:
        compatible = None

    public_manifest = manifest.model_dump(mode="json")
    public_manifest["inference"]["environment_variables"] = dict.fromkeys(
        manifest.inference.environment_variables, "<configured>"
    )
    return {
        **base,
        "status": "configured",
        "manifest": public_manifest,
        "compatibility": {"compatible": compatible, "issues": issues},
        "environment_available": _reference_availability(checkpoint, manifest.environment),
    }


def _load_benchmark(path: Path) -> BenchmarkRecord:
    value = json.loads(path.read_text(encoding="utf-8"))
    versions = value.get("versions")
    if isinstance(versions, dict):
        versions.setdefault("source", "validated_inference")
        versions.setdefault("captured_at", value.get("captured_at"))
    inference = value.get("inference")
    if isinstance(inference, dict) and isinstance(inference.get("torch_compile"), bool):
        enabled = inference["torch_compile"]
        value["inference"] = {
            "device": value.get("device", "cuda"),
            "precision": inference.get("precision", "checkpoint"),
            "attention_backend": inference.get("attention_backend", "checkpoint"),
            "torch_compile": {
                "enabled": enabled,
                "mode": inference.get("compile_mode") if enabled else None,
                "backend": "inductor" if enabled else None,
                "dynamic": None,
            },
            "environment_variables": {},
        }
    allowed = BenchmarkRecord.model_fields
    return BenchmarkRecord.model_validate({key: item for key, item in value.items() if key in allowed})


def _capture_command(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    training = capture_training_environment(args.framework, source=args.environment_source).model_copy(
        update=checkpoint_training_settings(checkpoint)
    )
    environment = ExecutableEnvironment(kind=args.environment_kind, reference=args.environment_reference)
    inference = InferenceSettings(
        device=args.device,
        precision=args.precision,
        attention_backend=args.attention_backend,
        rollout_backend=args.rollout_backend,
        torch_compile=CompileSettings(enabled=args.torch_compile, mode=args.torch_compile_mode),
    )
    benchmarks = [_load_benchmark(Path(item)) for item in args.benchmark]
    manifest = PolicyRuntimeManifest(
        framework=args.framework,
        training=training,
        environment=environment,
        inference=inference,
        benchmarks=benchmarks,
    )
    path = save_runtime_manifest(checkpoint, manifest, overwrite=args.force)
    print(path)
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = load_runtime_manifest(checkpoint)
    record = BenchmarkRecord(
        captured_at=datetime.now(UTC),
        environment_label=args.environment_label,
        device=args.device,
        gpu=args.gpu,
        driver=args.driver,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        load_time_s=args.load_time_s,
        first_inference_ms=args.first_inference_ms,
        latency_p50_ms=args.latency_p50_ms,
        latency_p95_ms=args.latency_p95_ms,
        peak_memory_bytes=args.peak_memory_bytes,
        notes=args.notes,
    )
    updated = manifest.model_copy(update={"benchmarks": [*manifest.benchmarks, record]})
    save_runtime_manifest(checkpoint, updated, overwrite=True)
    print(checkpoint / RUNTIME_MANIFEST_NAME)
    return 0


def _configure_command(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    manifest = load_runtime_manifest(checkpoint)
    inference_updates = {
        name: value
        for name, value in (
            ("device", args.device),
            ("precision", args.precision),
            ("attention_backend", args.attention_backend),
            ("rollout_backend", args.rollout_backend),
        )
        if value is not None
    }
    compile_settings = manifest.inference.torch_compile
    if args.torch_compile is not None or args.torch_compile_mode is not None:
        compile_settings = compile_settings.model_copy(
            update={
                "enabled": args.torch_compile == "enabled"
                if args.torch_compile is not None
                else compile_settings.enabled,
                "mode": args.torch_compile_mode
                if args.torch_compile_mode is not None
                else compile_settings.mode,
            }
        )
        inference_updates["torch_compile"] = compile_settings
    inference = manifest.inference.model_copy(update=inference_updates)
    training = manifest.training
    if args.refresh_training_settings:
        training = training.model_copy(update=checkpoint_training_settings(checkpoint))
    updated = manifest.model_copy(update={"inference": inference, "training": training})
    save_runtime_manifest(checkpoint, updated, overwrite=True)
    print(checkpoint / RUNTIME_MANIFEST_NAME)
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    result = inspect_runtime_manifest(checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "configured" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage checkpoint-local EvoMind runtime manifests")
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="capture the active environment into a checkpoint")
    capture.add_argument("checkpoint")
    capture.add_argument("--framework", choices=("pytorch", "jax"), required=True)
    capture.add_argument(
        "--environment-source",
        choices=("training_capture", "checkpoint_metadata", "validated_inference"),
        default="training_capture",
    )
    capture.add_argument(
        "--environment-kind", choices=("current", "python", "uv", "conda", "container"), default="current"
    )
    capture.add_argument("--environment-reference")
    capture.add_argument("--device", default="cuda")
    capture.add_argument("--precision", default="bfloat16")
    capture.add_argument("--attention-backend", default="checkpoint")
    capture.add_argument("--rollout-backend", choices=("sync", "rtc"), default="sync")
    capture.add_argument("--torch-compile", action="store_true")
    capture.add_argument("--torch-compile-mode")
    capture.add_argument("--benchmark", action="append", default=[], help="benchmark JSON file to include")
    capture.add_argument("--force", action="store_true")
    capture.set_defaults(handler=_capture_command)

    benchmark = commands.add_parser("record-benchmark", help="append an offline benchmark result")
    benchmark.add_argument("checkpoint")
    benchmark.add_argument("--environment-label", required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--gpu")
    benchmark.add_argument("--driver")
    benchmark.add_argument("--warmup-runs", type=int, required=True)
    benchmark.add_argument("--measured-runs", type=int, required=True)
    benchmark.add_argument("--load-time-s", type=float, required=True)
    benchmark.add_argument("--first-inference-ms", type=float, required=True)
    benchmark.add_argument("--latency-p50-ms", type=float, required=True)
    benchmark.add_argument("--latency-p95-ms", type=float, required=True)
    benchmark.add_argument("--peak-memory-bytes", type=int)
    benchmark.add_argument("--notes")
    benchmark.set_defaults(handler=_benchmark_command)

    configure = commands.add_parser("configure", help="update runtime selection without losing benchmarks")
    configure.add_argument("checkpoint")
    configure.add_argument("--device")
    configure.add_argument("--precision")
    configure.add_argument("--attention-backend")
    configure.add_argument("--rollout-backend", choices=("sync", "rtc"))
    configure.add_argument("--torch-compile", choices=("enabled", "disabled"))
    configure.add_argument("--torch-compile-mode")
    configure.add_argument("--refresh-training-settings", action="store_true")
    configure.set_defaults(handler=_configure_command)

    validate = commands.add_parser("validate", help="validate and inspect a checkpoint manifest")
    validate.add_argument("checkpoint")
    validate.set_defaults(handler=_validate_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
