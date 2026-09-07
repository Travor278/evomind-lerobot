import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

import evomind_lerobot.runtime_service as runtime_module
from evomind_lerobot.collection_store import CollectionStore, CollectionTaskConflictError, local_today
from evomind_lerobot.events import EventBroker, Operation
from evomind_lerobot.jobs import JobManager
from evomind_lerobot.runtime_service import CollectionStartRequest, RolloutStartRequest, RuntimeService


def test_legacy_tasks_migrate_to_manual_collection(tmp_path) -> None:
    database = tmp_path / "state.sqlite3"
    now = datetime.now().isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE daily_tasks (
                id TEXT PRIMARY KEY,
                work_date TEXT NOT NULL,
                name TEXT NOT NULL COLLATE NOCASE,
                description TEXT NOT NULL,
                target_duration_s REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (work_date, name)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO daily_tasks
                (id, work_date, name, description, target_duration_s, created_at, updated_at)
            VALUES ('legacy', ?, '旧任务', '人工示教', 600, ?, ?)
            """,
            (local_today().isoformat(), now, now),
        )

    task = CollectionStore(database).get_task("legacy")

    assert task["collection_method"] == "manual"
    assert task["policy_path"] == ""
    assert task["rollout_strategy"] == "episodic_dagger"
    assert task["session_count"] == 0


def test_policy_episode_saved_updates_shared_progress(tmp_path) -> None:
    store = CollectionStore(tmp_path / "state.sqlite3")
    task = store.create_task(
        work_date=local_today(),
        name="Policy 采集",
        description="pick block",
        target_duration_s=10,
        num_episodes=4,
        episode_time_s=5,
        reset_time_s=1,
        fps=20,
        collection_method="policy",
        policy_path="/models/policy",
        rollout_strategy="highlight",
        inference="rtc",
        duration_s=120,
        ring_buffer_seconds=8,
    )
    request = SimpleNamespace(fps=20, num_episodes=4, episode_time_s=5, reset_time_s=1)

    store.start_session("session", task["id"], "policy-run", request)
    store.save_episode(
        "session",
        {"episode_index": 0, "frames": 80, "fps": 20, "repo_id": "rollout_policy-run_1"},
    )

    progress = store.progress(local_today(), 7)
    active = progress["active_session"]
    assert active is not None
    assert active["collection_method"] == "policy"
    assert active["rollout_strategy"] == "highlight"
    assert active["policy_path"] == "/models/policy"
    assert active["saved_duration_s"] == 4
    assert active["saved_episodes"] == 1
    assert progress["tasks"][0]["actual_duration_s"] == 4

    store.finish_session("session", failed=False)
    with pytest.raises(CollectionTaskConflictError, match="不能修改采集方式"):
        store.update_task(
            task["id"],
            name=task["name"],
            description=task["description"],
            target_duration_s=task["target_duration_s"],
            num_episodes=task["num_episodes"],
            episode_time_s=task["episode_time_s"],
            reset_time_s=task["reset_time_s"],
            fps=task["fps"],
            collection_method="manual",
            policy_path="",
            rollout_strategy=task["rollout_strategy"],
            inference=task["inference"],
            duration_s=task["duration_s"],
            ring_buffer_seconds=task["ring_buffer_seconds"],
        )


def test_policy_collection_builds_rollout_from_task(tmp_path, monkeypatch) -> None:
    store = CollectionStore(tmp_path / "state.sqlite3")
    task = store.create_task(
        work_date=local_today(),
        name="Episodic DAgger",
        description="insert screw",
        target_duration_s=60,
        num_episodes=3,
        episode_time_s=20,
        reset_time_s=4,
        fps=30,
        collection_method="policy",
        policy_path="/models/pi05",
        rollout_strategy="episodic_dagger",
        inference="rtc",
        duration_s=90,
        ring_buffer_seconds=12,
    )
    monkeypatch.setattr(
        runtime_module,
        "policies_inventory",
        lambda: [{"id": "pi05", "path": "/models/pi05", "type": "pi05"}],
    )
    monkeypatch.setattr(
        runtime_module,
        "inspect_policy_compatibility",
        lambda request: {"compatible": True, "issues": []},
    )
    service = RuntimeService(EventBroker(), JobManager(EventBroker()), store)
    captured = {}

    def fake_start(operation, request, collection_task):
        captured.update(operation=operation, request=request, task=collection_task)
        return {"running": True}

    monkeypatch.setattr(service, "_start", fake_start)

    result = service.start_collection(CollectionStartRequest(task_id=task["id"]))

    assert result == {"running": True}
    assert captured["operation"] is Operation.ROLLOUT
    assert isinstance(captured["request"], RolloutStartRequest)
    assert captured["request"].strategy == "episodic_dagger"
    assert captured["request"].inference == "rtc"
    assert captured["request"].policy_path == "/models/pi05"
    assert captured["request"].dataset_name.startswith("Episodic-DAgger_")


def test_policy_collection_rejects_missing_local_model(tmp_path, monkeypatch) -> None:
    store = CollectionStore(tmp_path / "state.sqlite3")
    task = store.create_task(
        work_date=local_today(),
        name="Missing policy",
        description="pick block",
        target_duration_s=60,
        num_episodes=2,
        episode_time_s=20,
        reset_time_s=2,
        fps=30,
        collection_method="policy",
        policy_path="/models/missing",
    )
    monkeypatch.setattr(runtime_module, "policies_inventory", lambda: [])
    service = RuntimeService(EventBroker(), JobManager(EventBroker()), store)

    with pytest.raises(ValueError, match="不在本机模型目录"):
        service.start_collection(CollectionStartRequest(task_id=task["id"]))


def test_policy_collection_rejects_incompatible_model(tmp_path, monkeypatch) -> None:
    store = CollectionStore(tmp_path / "state.sqlite3")
    task = store.create_task(
        work_date=local_today(),
        name="Incompatible policy",
        description="pick block",
        target_duration_s=60,
        num_episodes=2,
        episode_time_s=20,
        reset_time_s=2,
        fps=30,
        collection_method="policy",
        policy_path="/models/pi05",
    )
    monkeypatch.setattr(
        runtime_module,
        "policies_inventory",
        lambda: [{"id": "pi05", "path": "/models/pi05", "type": "pi05"}],
    )
    monkeypatch.setattr(
        runtime_module,
        "inspect_policy_compatibility",
        lambda request: {"compatible": False, "issues": ["动作维度不匹配"]},
    )
    service = RuntimeService(EventBroker(), JobManager(EventBroker()), store)

    with pytest.raises(ValueError, match="动作维度不匹配"):
        service.start_collection(CollectionStartRequest(task_id=task["id"]))


def test_rollout_save_event_is_emitted_only_after_success(monkeypatch) -> None:
    pytest.importorskip("datasets")
    from lerobot.rollout.strategies import core

    emitted = []

    class Dataset:
        num_episodes = 0
        writer = SimpleNamespace(episode_buffer={"size": 75})

        def save_episode(self):
            self.num_episodes += 1

    dataset = Dataset()
    dataset_cfg = SimpleNamespace(repo_id="rollout_example_1", fps=25, num_episodes=5)
    context = SimpleNamespace(runtime=SimpleNamespace(cfg=SimpleNamespace(dataset=dataset_cfg)))
    monkeypatch.setattr(core, "emit_runtime_event", lambda *args, **kwargs: emitted.append((args, kwargs)))

    core.save_episode_and_emit(dataset, context, strategy="sentry")

    assert dataset.num_episodes == 1
    assert emitted == [
        (
            ("rollout", "running"),
            {
                "stage": "episode_saved",
                "strategy": "sentry",
                "repo_id": "rollout_example_1",
                "episode_index": 0,
                "frames": 75,
                "fps": 25,
                "duration_s": 3,
                "saved_episodes": 1,
                "target_episodes": 5,
            },
        )
    ]
