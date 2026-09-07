from pathlib import Path

import pytest

from evomind_lerobot.robot_assets import (
    RobotModelNotFoundError,
    normalize_robot_model,
    robot_asset_content_type,
    robot_asset_path,
    robot_model_manifest,
)


@pytest.mark.parametrize(
    ("robot_type", "expected"),
    [
        ("so101_follower", "so101"),
        ("bi_so_follower", "so101"),
        ("piper", "piperx"),
        ("bi_piperx_follower", "piperx"),
    ],
)
def test_robot_model_aliases(robot_type: str, expected: str) -> None:
    assert normalize_robot_model(robot_type) == expected


@pytest.mark.parametrize("robot_type", ["so101", "piperx"])
def test_robot_model_manifest_references_packaged_assets(robot_type: str) -> None:
    manifest = robot_model_manifest(robot_type)

    assert manifest["model"] == robot_type
    assert manifest["urdf_url"].startswith(f"/api/dataset/robot-assets/{manifest['asset_id']}/")
    assert {item["path"] for item in manifest["files"]} >= {manifest["urdf_path"], "model.glb"}
    assert robot_asset_path(manifest["asset_id"], manifest["urdf_path"]).is_file()
    assert robot_asset_content_type(Path("model.glb")) == "model/gltf-binary"


@pytest.mark.parametrize("path", ["../model.glb", "/tmp/model.glb", "folder/../../model.glb"])
def test_robot_asset_path_rejects_traversal(path: str) -> None:
    with pytest.raises(RobotModelNotFoundError, match="路径无效"):
        robot_asset_path("so101", path)


def test_unknown_robot_model_is_not_silently_substituted() -> None:
    with pytest.raises(RobotModelNotFoundError, match="暂不支持"):
        robot_model_manifest("unknown_robot")
