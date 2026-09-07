"""Read-only local LeRobot dataset browsing helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE


class DatasetBrowserError(RuntimeError):
    pass


class DatasetNotFoundError(DatasetBrowserError):
    pass


class DatasetUnavailableError(DatasetBrowserError):
    pass


def _dataset_roots() -> list[tuple[str, Path]]:
    root = HF_LEROBOT_HOME
    if not root.is_dir():
        return []
    info_paths = set(root.glob("*/meta/info.json")) | set(root.glob("*/*/meta/info.json"))
    datasets: list[tuple[str, Path]] = []
    for info_path in sorted(info_paths):
        dataset_root = info_path.parents[1]
        relative = dataset_root.relative_to(root)
        if relative.parts[0] in {"calibration", "hub", "policies"}:
            continue
        datasets.append((str(relative), dataset_root))
    return datasets


def _resolve_dataset(dataset_id: str) -> Path:
    available = dict(_dataset_roots())
    root = available.get(dataset_id)
    if root is None:
        raise DatasetNotFoundError("本地数据集不存在")
    resolved = root.resolve()
    if not resolved.is_relative_to(HF_LEROBOT_HOME.resolve()):
        raise DatasetNotFoundError("数据集路径不在本地 LeRobot 目录中")
    return resolved


def _metadata(dataset_id: str, root: Path):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    try:
        return LeRobotDatasetMetadata(dataset_id, root=root, token=False, local_files_only=True)
    except Exception as error:
        raise DatasetUnavailableError(f"元数据不可加载：{error}") from error


def _validate_local_data(root: Path, expected_frames: int) -> None:
    """Validate local parquet footers without reading frame payloads."""
    import pyarrow.parquet as pq

    paths = sorted((root / "data").glob("*/*.parquet"))
    if expected_frames > 0 and not paths:
        raise DatasetUnavailableError("数据文件缺失")

    actual_frames = 0
    for path in paths:
        try:
            actual_frames += pq.read_metadata(path).num_rows
        except Exception as error:
            relative = path.relative_to(root)
            raise DatasetUnavailableError(f"数据文件不可读取：{relative}：{error}") from error
    if actual_frames != expected_frames:
        raise DatasetUnavailableError(
            f"数据帧数不一致：info.json 记录 {expected_frames}，本地 Parquet 实际 {actual_frames}"
        )


def _task_names(metadata: Any) -> list[str]:
    if metadata.tasks is None:
        return []
    return [str(value) for value in metadata.tasks.index.tolist()]


def _feature_names(feature: dict[str, Any], dimension: int, key: str) -> list[str]:
    names = feature.get("names")
    if isinstance(names, list) and len(names) == dimension:
        return [str(name) for name in names]
    return [f"{key}_{index}" for index in range(dimension)]


def _camera_resolution(feature: dict[str, Any]) -> str | None:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None
    values = [int(value) for value in shape]
    if len(values) >= 3 and values[0] <= 4 and values[1] > 4 and values[2] > 4:
        height, width = values[1], values[2]
    else:
        height, width = values[0], values[1]
    return f"{width}×{height}"


def _episode_row(metadata: Any, episode_index: int) -> dict[str, Any]:
    if episode_index < 0 or episode_index >= metadata.total_episodes:
        raise DatasetNotFoundError("Episode 超出数据集范围")
    if metadata.episodes is None:
        raise DatasetUnavailableError("Episode 元数据缺失")
    row = metadata.episodes[episode_index]
    if int(row["episode_index"]) != episode_index:
        matches = metadata.episodes.filter(
            lambda item: int(item["episode_index"]) == episode_index,
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        if len(matches) == 0:
            raise DatasetNotFoundError("Episode 不存在")
        row = matches[0]
    return row


def datasets_catalog(active_dataset_id: str | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset_id, root in _dataset_roots():
        active = dataset_id == active_dataset_id
        info_path = root / "meta/info.json"
        fps = 0
        frames = 0
        episodes = 0
        tasks: list[str] = []
        robot_type = ""
        recorded_on = ""
        camera_count = 0
        status = "recording" if active else "ready"
        error_message = ""
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = int(info["fps"])
            frames = int(info["total_frames"])
            episodes = int(info["total_episodes"])
            features = info.get("features", {})
            camera_keys = [
                key for key, feature in features.items() if feature.get("dtype") in {"video", "image"}
            ]
            robot_type = str(info.get("robot_type") or "")
            recorded_on = datetime.fromtimestamp(info_path.stat().st_mtime).astimezone().date().isoformat()
            camera_count = len(camera_keys)
            if not active:
                metadata = _metadata(dataset_id, root)
                tasks = _task_names(metadata)
                _validate_local_data(root, frames)
                for episode_index in range(metadata.total_episodes):
                    for video_key in metadata.video_keys:
                        video_path = root / metadata.get_video_file_path(episode_index, video_key)
                        if not video_path.is_file():
                            status = "incomplete"
                            error_message = f"缺少视频文件：{video_key} / Episode {episode_index}"
                            break
                    if error_message:
                        break
        except (DatasetBrowserError, FileNotFoundError, KeyError, TypeError, ValueError, OSError) as error:
            status = "unreadable"
            error_message = str(error)
        results.append(
            {
                "id": dataset_id,
                "path": str(root),
                "episodes": episodes,
                "frames": frames,
                "fps": fps,
                "duration_s": frames / fps if fps else 0,
                "tasks": tasks,
                "robot_type": robot_type,
                "recorded_on": recorded_on,
                "camera_count": camera_count,
                "status": status,
                "available": status == "ready",
                "error": error_message,
            }
        )
    return results


def dataset_detail(dataset_id: str, active_dataset_id: str | None = None) -> dict[str, Any]:
    if dataset_id == active_dataset_id:
        raise DatasetUnavailableError("数据集仍在采集中，结束后才能查看")
    root = _resolve_dataset(dataset_id)
    metadata = _metadata(dataset_id, root)
    episodes = []
    for index in range(metadata.total_episodes):
        row = _episode_row(metadata, index)
        length = int(row["length"])
        episodes.append(
            {
                "episode_index": int(row["episode_index"]),
                "frames": length,
                "duration_s": length / metadata.fps if metadata.fps else 0,
                "tasks": [str(value) for value in row.get("tasks", [])],
            }
        )
    cameras = [
        {
            "key": key,
            "label": key.rsplit(".", 1)[-1],
            "resolution": _camera_resolution(metadata.features[key]),
            "depth": key in metadata.depth_keys,
        }
        for key in metadata.camera_keys
    ]
    return {
        "id": dataset_id,
        "path": str(root),
        "robot_type": metadata.robot_type,
        "fps": metadata.fps,
        "frames": metadata.total_frames,
        "duration_s": metadata.total_frames / metadata.fps if metadata.fps else 0,
        "tasks": _task_names(metadata),
        "cameras": cameras,
        "episodes": episodes,
    }


def _to_matrix(values: list[Any], frame_count: int) -> np.ndarray:
    if len(values) == 0:
        return np.empty((frame_count, 0), dtype=np.float64)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix.reshape(matrix.shape[0], -1)


def _sample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit, dtype=np.int64))


def dataset_episode(
    dataset_id: str,
    episode_index: int,
    *,
    active_dataset_id: str | None = None,
    max_points: int = 10_000,
) -> dict[str, Any]:
    if dataset_id == active_dataset_id:
        raise DatasetUnavailableError("数据集仍在采集中，结束后才能查看")
    root = _resolve_dataset(dataset_id)
    metadata = _metadata(dataset_id, root)
    episode = _episode_row(metadata, episode_index)

    from lerobot.datasets.io_utils import load_nested_dataset

    try:
        frame_data = load_nested_dataset(root / "data", episodes=[episode_index]).with_format(None)
    except Exception as error:
        raise DatasetUnavailableError(f"Episode 数据不可加载：{error}") from error
    available_columns = set(frame_data.column_names)
    timestamps = np.asarray(frame_data["timestamp"], dtype=np.float64)
    if timestamps.size:
        timestamps -= timestamps[0]
    action = _to_matrix(frame_data[ACTION] if ACTION in available_columns else [], len(timestamps))
    state = _to_matrix(frame_data[OBS_STATE] if OBS_STATE in available_columns else [], len(timestamps))
    indices = _sample_indices(len(timestamps), max_points)

    action_names = _feature_names(metadata.features.get(ACTION, {}), action.shape[1], "action")
    state_names = _feature_names(metadata.features.get(OBS_STATE, {}), state.shape[1], "state")
    series = []
    for dimension in range(max(action.shape[1], state.shape[1])):
        action_name = action_names[dimension] if dimension < len(action_names) else ""
        state_name = state_names[dimension] if dimension < len(state_names) else ""
        label = action_name or state_name or f"joint_{dimension}"
        if action_name and state_name and action_name != state_name:
            label = f"{action_name} / {state_name}"
        series.append(
            {
                "label": label,
                "action_name": action_name,
                "state_name": state_name,
                "action": action[indices, dimension].tolist() if dimension < action.shape[1] else [],
                "state": state[indices, dimension].tolist() if dimension < state.shape[1] else [],
            }
        )

    videos = []
    for key in metadata.video_keys:
        video_path = root / metadata.get_video_file_path(episode_index, key)
        if not video_path.is_file():
            raise DatasetUnavailableError(f"缺少视频文件：{key}")
        query = urlencode({"dataset_id": dataset_id, "episode": episode_index, "camera": key})
        videos.append(
            {
                "key": key,
                "label": key.rsplit(".", 1)[-1],
                "url": f"/api/dataset/video?{query}",
                "from_timestamp": float(episode.get(f"videos/{key}/from_timestamp", 0)),
                "to_timestamp": float(
                    episode.get(
                        f"videos/{key}/to_timestamp",
                        episode.get(f"videos/{key}/from_timestamp", 0)
                        + int(episode["length"]) / metadata.fps,
                    )
                ),
            }
        )
    length = int(episode["length"])
    return {
        "dataset_id": dataset_id,
        "robot_type": metadata.robot_type,
        "episode_index": episode_index,
        "frames": length,
        "duration_s": length / metadata.fps if metadata.fps else 0,
        "fps": metadata.fps,
        "tasks": [str(value) for value in episode.get("tasks", [])],
        "timestamps": timestamps[indices].tolist(),
        "series": series,
        "videos": videos,
    }


def dataset_video_path(
    dataset_id: str,
    episode_index: int,
    camera: str,
    *,
    active_dataset_id: str | None = None,
) -> Path:
    if dataset_id == active_dataset_id:
        raise DatasetUnavailableError("数据集仍在采集中，结束后才能查看")
    root = _resolve_dataset(dataset_id)
    metadata = _metadata(dataset_id, root)
    _episode_row(metadata, episode_index)
    if camera not in metadata.video_keys:
        raise DatasetNotFoundError("摄像头不存在")
    path = (root / metadata.get_video_file_path(episode_index, camera)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DatasetNotFoundError("视频文件不存在")
    return path


__all__ = [
    "DatasetBrowserError",
    "DatasetNotFoundError",
    "DatasetUnavailableError",
    "dataset_detail",
    "dataset_episode",
    "dataset_video_path",
    "datasets_catalog",
]
