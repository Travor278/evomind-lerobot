"""Robot model manifests and packaged assets for dataset playback."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any


class RobotModelNotFoundError(RuntimeError):
    pass


ROBOT_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "robots"

_PIPERX_JOINT_LIMITS_DEGREES = {
    "joint1": [-150.0, 150.0],
    "joint2": [0.0, 180.0],
    "joint3": [-170.0, 0.0],
    "joint4": [-100.0, 100.0],
    "joint5": [-70.0, 70.0],
    "joint6": [-120.0, 120.0],
}

_ROBOT_MODELS: dict[str, dict[str, Any]] = {
    "so101": {
        "asset_id": "so101",
        "urdf_path": "so101_new_calib.urdf",
        "joint_order": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "feature_mapping": {"joint_pattern": "{side}_{joint}.pos", "gripper_pattern": None},
        "units": {"joint": "degree", "gripper": "degree"},
        "gripper": None,
        "limits": {
            "shoulder_pan": [-110.0, 162.79],
            "shoulder_lift": [-100.0, 100.0],
            "elbow_flex": [-96.83, 96.83],
            "wrist_flex": [-95.0, 95.0],
            "wrist_roll": [-157.21, 162.79],
            "gripper": [-10.0, 100.0],
        },
        "scene": {"left_base_xyz": [0.0, 0.115, 0.0], "right_base_xyz": [0.0, -0.115, 0.0]},
    },
    "piperx": {
        "asset_id": "piperx",
        "urdf_path": "piper_x_description.urdf",
        "joint_order": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "feature_mapping": {
            "joint_pattern": "{side}_joint_{index}.pos",
            "gripper_pattern": "{side}_gripper.pos",
        },
        "units": {"joint": "degree", "gripper": "millimeter"},
        "gripper": {
            "source_range": [0.0, 100.0],
            "joints": [
                {"name": "joint7", "scale": 0.0005, "offset": 0.0},
                {"name": "joint8", "scale": -0.0005, "offset": 0.0},
            ],
        },
        "limits": _PIPERX_JOINT_LIMITS_DEGREES,
        "scene": {"left_base_xyz": [0.0, 0.30, 0.0], "right_base_xyz": [0.0, -0.30, 0.0]},
    },
}

_MODEL_ALIASES = {
    "piper": "piperx",
    "bi_piper": "piperx",
    "bi_piperx": "piperx",
    "piper_follower": "piperx",
    "bi_piper_follower": "piperx",
    "piperx_follower": "piperx",
    "bi_piperx_follower": "piperx",
    "so101_follower": "so101",
    "bi_so101_follower": "so101",
    "so_follower": "so101",
    "bi_so_follower": "so101",
}


def normalize_robot_model(robot_type: str) -> str | None:
    key = robot_type.strip().lower().replace("-", "_")
    if key in _ROBOT_MODELS:
        return key
    return _MODEL_ALIASES.get(key)


@cache
def robot_model_manifest(robot_type: str) -> dict[str, Any]:
    model = normalize_robot_model(robot_type)
    if model is None:
        raise RobotModelNotFoundError(f"暂不支持 {robot_type or '未知型号'} 的 3D 回放")
    spec = _ROBOT_MODELS[model]
    asset_id = spec["asset_id"]
    root = _asset_root(asset_id)
    files = [_asset_payload(path, root) for path in sorted(root.iterdir()) if path.is_file()]
    base_url = f"/api/dataset/robot-assets/{asset_id}/"
    urdf_path = spec["urdf_path"]
    urdf_version = next(item["sha256"] for item in files if item["path"] == urdf_path)
    return {
        "model": model,
        "asset_base_url": base_url,
        "urdf_url": f"{base_url}{urdf_path}?v={urdf_version}",
        "files": files,
        **spec,
    }


def robot_asset_path(asset_id: str, relative_path: str) -> Path:
    root = _asset_root(_safe_segment(asset_id))
    target = (root / _safe_relative_path(relative_path)).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise RobotModelNotFoundError("3D 模型资源不存在")
    return target


def robot_asset_content_type(path: Path) -> str:
    if path.suffix.lower() == ".urdf":
        return "application/xml"
    if path.suffix.lower() == ".glb":
        return "model/gltf-binary"
    if path.suffix.lower() in {".md", ".txt"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _asset_root(asset_id: str) -> Path:
    root = (ROBOT_ASSET_ROOT / _safe_segment(asset_id)).resolve()
    if not root.is_dir():
        raise RobotModelNotFoundError(f"3D 模型资源包 {asset_id} 不存在")
    return root


def _asset_payload(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_type": robot_asset_content_type(path),
    }


def _safe_segment(value: str) -> str:
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if not value.strip() or path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise RobotModelNotFoundError("3D 模型资源路径无效")
    return path.parts[0]


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RobotModelNotFoundError("3D 模型资源路径无效")
    return path.as_posix()


__all__ = [
    "RobotModelNotFoundError",
    "normalize_robot_model",
    "robot_asset_content_type",
    "robot_asset_path",
    "robot_model_manifest",
]
