"""Durable local collection plans and recording progress."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


class CollectionStoreError(RuntimeError):
    """Base error for collection progress persistence."""


class CollectionTaskNotFoundError(CollectionStoreError):
    pass


class CollectionTaskConflictError(CollectionStoreError):
    pass


def local_today() -> date:
    return datetime.now(LOCAL_TIMEZONE).date()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _state_database_path() -> Path:
    configured = os.environ.get("EVOMIND_LEROBOT_STATE_DB")
    if configured:
        return Path(configured)
    return Path.home() / ".config/evomind-lerobot/state.sqlite3"


class CollectionStore:
    """SQLite-backed source of truth for daily plans and saved episodes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _state_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_tasks (
                    id TEXT PRIMARY KEY,
                    work_date TEXT NOT NULL,
                    name TEXT NOT NULL COLLATE NOCASE,
                    description TEXT NOT NULL,
                    target_duration_s REAL NOT NULL CHECK (target_duration_s > 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (work_date, name)
                )
                """
            )
            task_columns = {row["name"] for row in connection.execute("PRAGMA table_info(daily_tasks)")}
            task_setting_columns = {
                "num_episodes": "INTEGER NOT NULL DEFAULT 20",
                "episode_time_s": "INTEGER NOT NULL DEFAULT 30",
                "reset_time_s": "INTEGER NOT NULL DEFAULT 10",
                "fps": "INTEGER NOT NULL DEFAULT 30",
                "collection_method": "TEXT NOT NULL DEFAULT 'manual'",
                "policy_path": "TEXT NOT NULL DEFAULT ''",
                "rollout_strategy": "TEXT NOT NULL DEFAULT 'episodic_dagger'",
                "inference": "TEXT NOT NULL DEFAULT 'sync'",
                "duration_s": "INTEGER NOT NULL DEFAULT 120",
                "ring_buffer_seconds": "INTEGER NOT NULL DEFAULT 10",
            }
            for column, definition in task_setting_columns.items():
                if column not in task_columns:
                    connection.execute(f"ALTER TABLE daily_tasks ADD COLUMN {column} {definition}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_sessions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES daily_tasks(id) ON DELETE RESTRICT,
                    dataset_name TEXT NOT NULL,
                    repo_id TEXT,
                    fps INTEGER NOT NULL,
                    num_episodes INTEGER NOT NULL,
                    episode_time_s REAL NOT NULL,
                    reset_time_s REAL NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            session_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(collection_sessions)")
            }
            session_setting_columns = {
                "collection_method": "TEXT NOT NULL DEFAULT 'manual'",
                "rollout_strategy": "TEXT",
                "policy_path": "TEXT",
            }
            for column, definition in session_setting_columns.items():
                if column not in session_columns:
                    connection.execute(
                        f"ALTER TABLE collection_sessions ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_episodes (
                    session_id TEXT NOT NULL REFERENCES collection_sessions(id) ON DELETE CASCADE,
                    episode_index INTEGER NOT NULL,
                    frames INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    saved_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, episode_index)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_tasks_work_date ON daily_tasks(work_date)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection_sessions_task_id ON collection_sessions(task_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection_sessions_status ON collection_sessions(status)"
            )
            connection.execute(
                "UPDATE collection_sessions SET status = 'interrupted', ended_at = ? WHERE status = 'running'",
                (now,),
            )
            connection.execute("PRAGMA optimize")

    @staticmethod
    def _task_payload(row: sqlite3.Row) -> dict[str, Any]:
        values = dict(row)
        duration = float(values.get("actual_duration_s", 0))
        target = float(values["target_duration_s"])
        episodes = int(values.get("episode_count", 0))
        sessions = int(values.get("session_count", 0))
        active_sessions = int(values.get("active_session_count", 0))
        return {
            "id": values["id"],
            "work_date": values["work_date"],
            "name": values["name"],
            "description": values["description"],
            "target_duration_s": target,
            "num_episodes": int(values["num_episodes"]),
            "episode_time_s": int(values["episode_time_s"]),
            "reset_time_s": int(values["reset_time_s"]),
            "fps": int(values["fps"]),
            "collection_method": str(values["collection_method"]),
            "policy_path": str(values["policy_path"]),
            "rollout_strategy": str(values["rollout_strategy"]),
            "inference": str(values["inference"]),
            "duration_s": int(values["duration_s"]),
            "ring_buffer_seconds": int(values["ring_buffer_seconds"]),
            "actual_duration_s": duration,
            "episode_count": episodes,
            "session_count": sessions,
            "progress_percent": duration / target * 100 if target > 0 else 0,
            "completed": duration >= target,
            "locked": sessions > 0,
            "collecting": active_sessions > 0,
            "created_at": values["created_at"],
            "updated_at": values["updated_at"],
        }

    def create_task(
        self,
        *,
        work_date: date,
        name: str,
        description: str,
        target_duration_s: float,
        num_episodes: int,
        episode_time_s: int,
        reset_time_s: int,
        fps: int,
        collection_method: str = "manual",
        policy_path: str = "",
        rollout_strategy: str = "episodic_dagger",
        inference: str = "sync",
        duration_s: int = 120,
        ring_buffer_seconds: int = 10,
    ) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO daily_tasks
                        (id, work_date, name, description, target_duration_s, num_episodes,
                         episode_time_s, reset_time_s, fps, collection_method, policy_path,
                         rollout_strategy, inference, duration_s, ring_buffer_seconds,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        work_date.isoformat(),
                        name.strip(),
                        description.strip(),
                        target_duration_s,
                        num_episodes,
                        episode_time_s,
                        reset_time_s,
                        fps,
                        collection_method,
                        policy_path,
                        rollout_strategy,
                        inference,
                        duration_s,
                        ring_buffer_seconds,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise CollectionTaskConflictError("当天已存在同名任务") from error
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT t.*,
                       COALESCE(SUM(e.duration_s), 0) AS actual_duration_s,
                       COUNT(e.episode_index) AS episode_count,
                       COUNT(DISTINCT s.id) AS session_count,
                       COUNT(DISTINCT CASE WHEN s.status = 'running' THEN s.id END)
                           AS active_session_count
                FROM daily_tasks t
                LEFT JOIN collection_sessions s ON s.task_id = t.id
                LEFT JOIN collection_episodes e ON e.session_id = s.id
                WHERE t.id = ?
                GROUP BY t.id
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise CollectionTaskNotFoundError("采集任务不存在")
        return self._task_payload(row)

    def require_today_task(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task["work_date"] != local_today().isoformat():
            raise CollectionTaskConflictError("只能开始今天计划中的采集任务")
        return task

    def list_tasks(self, work_date: date) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT t.*,
                       COALESCE(SUM(e.duration_s), 0) AS actual_duration_s,
                       COUNT(e.episode_index) AS episode_count,
                       COUNT(DISTINCT s.id) AS session_count,
                       COUNT(DISTINCT CASE WHEN s.status = 'running' THEN s.id END)
                           AS active_session_count
                FROM daily_tasks t
                LEFT JOIN collection_sessions s ON s.task_id = t.id
                LEFT JOIN collection_episodes e ON e.session_id = s.id
                WHERE t.work_date = ?
                GROUP BY t.id
                ORDER BY t.created_at
                """,
                (work_date.isoformat(),),
            ).fetchall()
        return [self._task_payload(row) for row in rows]

    def update_task(
        self,
        task_id: str,
        *,
        name: str,
        description: str,
        target_duration_s: float,
        num_episodes: int,
        episode_time_s: int,
        reset_time_s: int,
        fps: int,
        collection_method: str,
        policy_path: str,
        rollout_strategy: str,
        inference: str,
        duration_s: int,
        ring_buffer_seconds: int,
    ) -> dict[str, Any]:
        current = self.get_task(task_id)
        if current["collecting"]:
            raise CollectionTaskConflictError("任务正在采集，不能修改")
        if current["locked"] and collection_method != current["collection_method"]:
            raise CollectionTaskConflictError("已有采集记录后不能修改采集方式")
        try:
            with self._connect() as connection:
                updated = connection.execute(
                    """
                    UPDATE daily_tasks
                    SET name = ?, description = ?, target_duration_s = ?, num_episodes = ?, episode_time_s = ?,
                        reset_time_s = ?, fps = ?, collection_method = ?, policy_path = ?,
                        rollout_strategy = ?, inference = ?, duration_s = ?,
                        ring_buffer_seconds = ?, updated_at = ?
                    WHERE id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM collection_sessions
                          WHERE task_id = ? AND status = 'running'
                      )
                    """,
                    (
                        name.strip(),
                        description.strip(),
                        target_duration_s,
                        num_episodes,
                        episode_time_s,
                        reset_time_s,
                        fps,
                        collection_method,
                        policy_path,
                        rollout_strategy,
                        inference,
                        duration_s,
                        ring_buffer_seconds,
                        _utc_now(),
                        task_id,
                        task_id,
                    ),
                )
                if updated.rowcount == 0:
                    raise CollectionTaskConflictError("任务正在采集，不能修改")
        except sqlite3.IntegrityError as error:
            raise CollectionTaskConflictError("当天已存在同名任务") from error
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task["locked"]:
            raise CollectionTaskConflictError("已有采集记录的任务不能删除")
        with self._connect() as connection:
            connection.execute("DELETE FROM daily_tasks WHERE id = ?", (task_id,))

    def start_session(self, session_id: str, task_id: str, dataset_name: str, request: Any) -> None:
        task = self.require_today_task(task_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO collection_sessions
                    (id, task_id, dataset_name, fps, num_episodes, episode_time_s,
                     reset_time_s, collection_method, rollout_strategy, policy_path,
                     status, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    session_id,
                    task_id,
                    dataset_name,
                    request.fps,
                    request.num_episodes,
                    request.episode_time_s,
                    request.reset_time_s,
                    task["collection_method"],
                    task["rollout_strategy"] if task["collection_method"] == "policy" else None,
                    task["policy_path"] if task["collection_method"] == "policy" else None,
                    _utc_now(),
                ),
            )

    def update_session_repo_id(self, session_id: str, repo_id: str | None) -> None:
        if not repo_id:
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE collection_sessions SET repo_id = ? WHERE id = ?",
                (repo_id, session_id),
            )

    def save_episode(self, session_id: str, data: dict[str, Any]) -> None:
        episode_index = int(data["episode_index"])
        frames = int(data["frames"])
        fps = float(data["fps"])
        if episode_index < 0 or frames < 0 or fps <= 0:
            raise CollectionStoreError("Episode 保存事件包含无效的索引、帧数或 FPS")
        duration_s = frames / fps
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO collection_episodes
                    (session_id, episode_index, frames, fps, duration_s, saved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, episode_index) DO NOTHING
                """,
                (session_id, episode_index, frames, fps, duration_s, _utc_now()),
            )
            if data.get("repo_id"):
                connection.execute(
                    "UPDATE collection_sessions SET repo_id = ? WHERE id = ?",
                    (str(data["repo_id"]), session_id),
                )

    def finish_session(self, session_id: str, *, failed: bool, error: str = "") -> None:
        status = "failed" if failed else "completed"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE collection_sessions
                SET status = ?, ended_at = ?, error = ?
                WHERE id = ? AND status = 'running'
                """,
                (status, _utc_now(), error, session_id),
            )

    def active_session(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.*, t.name AS task_name, t.description AS task_description,
                       t.work_date, COALESCE(SUM(e.duration_s), 0) AS saved_duration_s,
                       COUNT(e.episode_index) AS saved_episodes
                FROM collection_sessions s
                JOIN daily_tasks t ON t.id = s.task_id
                LEFT JOIN collection_episodes e ON e.session_id = s.id
                WHERE s.status = 'running'
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row is not None else None

    def progress(self, work_date: date, window: int) -> dict[str, Any]:
        tasks = self.list_tasks(work_date)
        target = sum(float(task["target_duration_s"]) for task in tasks)
        actual = sum(float(task["actual_duration_s"]) for task in tasks)
        episodes = sum(int(task["episode_count"]) for task in tasks)
        completed = sum(1 for task in tasks if task["completed"])

        first_date = work_date - timedelta(days=window - 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH task_totals AS (
                    SELECT t.id, t.work_date, t.target_duration_s,
                           COALESCE(SUM(e.duration_s), 0) AS actual_duration_s,
                           COUNT(e.episode_index) AS episode_count
                    FROM daily_tasks t
                    LEFT JOIN collection_sessions s ON s.task_id = t.id
                    LEFT JOIN collection_episodes e ON e.session_id = s.id
                    WHERE t.work_date BETWEEN ? AND ?
                    GROUP BY t.id
                )
                SELECT work_date,
                       SUM(target_duration_s) AS target_duration_s,
                       SUM(actual_duration_s) AS actual_duration_s,
                       SUM(episode_count) AS episode_count
                FROM task_totals
                GROUP BY work_date
                """,
                (first_date.isoformat(), work_date.isoformat()),
            ).fetchall()
        by_date = {row["work_date"]: dict(row) for row in rows}
        trend = []
        for offset in range(window):
            item_date = first_date + timedelta(days=offset)
            values = by_date.get(item_date.isoformat(), {})
            trend.append(
                {
                    "date": item_date.isoformat(),
                    "target_duration_s": float(values.get("target_duration_s", 0)),
                    "actual_duration_s": float(values.get("actual_duration_s", 0)),
                    "episode_count": int(values.get("episode_count", 0)),
                }
            )
        return {
            "date": work_date.isoformat(),
            "tasks": tasks,
            "summary": {
                "target_duration_s": target,
                "actual_duration_s": actual,
                "progress_percent": actual / target * 100 if target > 0 else 0,
                "episode_count": episodes,
                "completed_tasks": completed,
                "total_tasks": len(tasks),
            },
            "trend": trend,
            "active_session": self.active_session(),
        }


__all__ = [
    "CollectionStore",
    "CollectionStoreError",
    "CollectionTaskConflictError",
    "CollectionTaskNotFoundError",
    "local_today",
]
