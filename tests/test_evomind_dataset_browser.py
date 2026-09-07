import json

import pytest

import evomind_lerobot.dataset_browser as browser
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def test_local_only_metadata_never_pulls_from_hub(tmp_path, monkeypatch) -> None:
    pulls = []

    def missing_metadata(self) -> None:
        raise FileNotFoundError("missing local episode metadata")

    monkeypatch.setattr(LeRobotDatasetMetadata, "_load_metadata", missing_metadata)
    monkeypatch.setattr(
        LeRobotDatasetMetadata,
        "_pull_from_repo",
        lambda *args, **kwargs: pulls.append((args, kwargs)),
    )

    with pytest.raises(FileNotFoundError, match="missing local episode metadata"):
        LeRobotDatasetMetadata("local/dataset", root=tmp_path, local_files_only=True)

    assert pulls == []


def test_catalog_keeps_unreadable_local_dataset(tmp_path, monkeypatch) -> None:
    dataset_id = "broken-dataset"
    root = tmp_path / dataset_id
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_frames": 120,
                "total_episodes": 2,
                "robot_type": "bi_piperx_follower",
                "features": {
                    "observation.images.environment": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(browser, "_dataset_roots", lambda: [(dataset_id, root)])

    def unreadable_metadata(*args, **kwargs):
        raise browser.DatasetUnavailableError("元数据不可加载")

    monkeypatch.setattr(browser, "_metadata", unreadable_metadata)

    assert browser.datasets_catalog() == [
        {
            "id": dataset_id,
            "path": str(root),
            "episodes": 2,
            "frames": 120,
            "fps": 30,
            "duration_s": 4.0,
            "tasks": [],
            "robot_type": "bi_piperx_follower",
            "recorded_on": browser.datetime.fromtimestamp((meta / "info.json").stat().st_mtime)
            .astimezone()
            .date()
            .isoformat(),
            "camera_count": 1,
            "status": "unreadable",
            "available": False,
            "error": "元数据不可加载",
        }
    ]


def test_local_data_validation_rejects_corrupt_parquet(tmp_path) -> None:
    parquet = tmp_path / "data/chunk-000/file-000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"unfinished collection")

    with pytest.raises(browser.DatasetUnavailableError, match="数据文件不可读取"):
        browser._validate_local_data(tmp_path, expected_frames=1)
