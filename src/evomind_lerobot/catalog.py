"""Discover robot and teleoperator types registered with LeRobot."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

_SYSTEM_DEFINITIONS = (
    ("so101", "SO-101", "so101_follower", "so101_leader", 1, 1),
    ("so100", "SO-100", "so100_follower", "so100_leader", 1, 1),
    ("bi_so", "SO 双臂", "bi_so_follower", "bi_so_leader", 2, 2),
    ("openarm", "OpenArm", "openarm_follower", "openarm_leader", 1, 1),
    ("bi_openarm", "OpenArm 双臂", "bi_openarm_follower", "bi_openarm_leader", 2, 2),
    ("rebot", "reBot", "rebot_b601_follower", "rebot_102_leader", 1, 1),
    ("bi_rebot", "reBot 双臂", "bi_rebot_b601_follower", "bi_rebot_102_leader", 2, 2),
    ("omx", "OMX", "omx_follower", "omx_leader", 1, 1),
    ("koch", "Koch", "koch_follower", "koch_leader", 1, 1),
    ("reachy2", "Reachy 2", "reachy2", "reachy2_teleoperator", 0, 0),
    ("unitree_g1", "Unitree G1", "unitree_g1", "unitree_g1", 0, 0),
)


def _load_config_modules(package: ModuleType) -> None:
    root = Path(package.__file__).parent
    for path in sorted(root.rglob("config*.py")):
        relative = path.relative_to(root).with_suffix("")
        module_name = f"{package.__name__}.{'.'.join(relative.parts)}"
        importlib.import_module(module_name)


def _label(type_name: str) -> str:
    return type_name.replace("_", " ").title()


def _profile(
    profile_id: str,
    label: str,
    robot_type: str,
    teleoperator_type: str | None,
    robot_ports: int | None,
    teleoperator_ports: int | None,
    robot_transport: str | None = "serial",
    teleoperator_transport: str | None = "serial",
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "label": label,
        "robot_type": robot_type,
        "teleoperator_type": teleoperator_type,
        "robot_ports": robot_ports,
        "teleoperator_ports": teleoperator_ports,
        "robot_transport": robot_transport if robot_ports else None,
        "teleoperator_transport": teleoperator_transport if teleoperator_ports else None,
    }


def _system_profiles(robots: set[str], teleoperators: set[str]) -> list[dict[str, Any]]:
    profiles = []
    covered = set()
    for definition in _SYSTEM_DEFINITIONS:
        profile_id, label, robot_type, teleoperator_type, robot_ports, teleoperator_ports = definition
        if robot_type not in robots or teleoperator_type not in teleoperators:
            continue
        profiles.append(
            _profile(
                profile_id,
                label,
                robot_type,
                teleoperator_type,
                robot_ports,
                teleoperator_ports,
            )
        )
        covered.add(robot_type)

    piper_definitions = (
        ("piperx", "PiperX", "piperx_follower", "piperx_leader", 1, 1),
        (
            "bi_piperx",
            "PiperX 双臂",
            "bi_piperx_follower",
            "bi_piperx_leader",
            2,
            2,
        ),
    )
    for definition in piper_definitions:
        profile_id, label, robot_type, teleoperator_type, robot_ports, teleoperator_ports = definition
        if robot_type not in robots or teleoperator_type not in teleoperators:
            continue
        profiles.append(
            _profile(
                profile_id,
                label,
                robot_type,
                teleoperator_type,
                robot_ports,
                teleoperator_ports,
                "socketcan",
                "socketcan",
            )
        )
        covered.add(robot_type)

    for robot_type in sorted(robots - covered):
        leader_type = robot_type.replace("_follower", "_leader")
        teleoperator_type = leader_type if leader_type in teleoperators else None
        profiles.append(
            _profile(robot_type, _label(robot_type), robot_type, teleoperator_type, None, None)
        )
    return profiles


@lru_cache(maxsize=1)
def hardware_catalog() -> dict[str, list[dict[str, Any]]]:
    import lerobot.robots
    import lerobot.teleoperators
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    _load_config_modules(lerobot.robots)
    _load_config_modules(lerobot.teleoperators)
    register_third_party_plugins()
    robots = set(RobotConfig.get_known_choices())
    teleoperators = set(TeleoperatorConfig.get_known_choices())
    return {"systems": _system_profiles(robots, teleoperators)}
