"""Read-only inventory of local datasets and policy checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_HOME


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _datasets() -> list[dict[str, Any]]:
    root = HF_LEROBOT_HOME
    if not root.is_dir():
        return []
    datasets = []
    info_paths = set(root.glob("*/meta/info.json")) | set(root.glob("*/*/meta/info.json"))
    for info_path in sorted(info_paths):
        info = _read_json(info_path)
        episodes = int(info["total_episodes"])
        frames = int(info["total_frames"])
        fps = int(info["fps"])
        if episodes <= 0 or frames <= 0 or fps <= 0:
            continue
        dataset_root = info_path.parents[1]
        relative = dataset_root.relative_to(root)
        if relative.parts[0] in {"calibration", "hub", "policies"}:
            continue
        datasets.append(
            {
                "id": str(relative),
                "path": str(dataset_root),
                "episodes": episodes,
                "frames": frames,
                "fps": fps,
                "tasks": int(info["total_tasks"]),
                "robot_type": str(info.get("robot_type") or ""),
            }
        )
    return datasets


def _policies() -> list[dict[str, str]]:
    root = HF_LEROBOT_HOME / "policies"
    if not root.is_dir():
        return []
    policies = []
    for config_path in sorted(root.glob("**/pretrained_model/config.json")):
        policy_root = config_path.parent
        config = _read_json(config_path)
        policies.append(
            {
                "id": str(policy_root.relative_to(root)),
                "path": str(policy_root),
                "type": str(config["type"]),
            }
        )
    return policies


def workspace_inventory() -> dict[str, Any]:
    return {"datasets": datasets_inventory(), "policies": policies_inventory()}


def datasets_inventory() -> list[dict[str, Any]]:
    return _datasets()


def policies_inventory() -> list[dict[str, str]]:
    return _policies()
