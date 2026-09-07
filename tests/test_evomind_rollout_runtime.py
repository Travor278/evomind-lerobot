import queue
import threading
from types import SimpleNamespace

from evomind_lerobot.device_config import (
    CameraBinding,
    CameraSlot,
    CanBinding,
    DeviceConfiguration,
    SerialBinding,
)
from evomind_lerobot.events import EventBroker, Operation
from evomind_lerobot.jobs import JobManager
from evomind_lerobot.runtime_service import (
    RolloutStartRequest,
    RuntimeService,
    _camera_rename_map,
    _configured_vector_dimensions,
    _configured_visual_features,
    _normalizer_feature_dim,
    _policy_path,
    _robot_payload,
    _rollout_inference_config,
    _rollout_repo_id,
    _run_policy_resident,
)


def _bi_so_configuration() -> DeviceConfiguration:
    return DeviceConfiguration(
        profile_id="bi_so",
        robot_type="bi_so_follower",
        teleoperator_type="bi_so_leader",
        camera_slots=[
            CameraSlot(alias="left_wrist", kind="wrist", side="left"),
            CameraSlot(alias="right_wrist", kind="wrist", side="right"),
            CameraSlot(alias="environment_1", kind="environment", side="single"),
        ],
        serial_bindings=[
            SerialBinding(id="left", port="/dev/left", alias="left_follower", kind="robot", side="left"),
            SerialBinding(id="right", port="/dev/right", alias="right_follower", kind="robot", side="right"),
        ],
        camera_bindings=[
            CameraBinding(
                id="left_cam",
                port="/dev/video-left",
                alias="left_wrist",
                side="left",
                driver="opencv",
                serial_number=None,
            ),
            CameraBinding(
                id="right_cam",
                port="/dev/video-right",
                alias="right_wrist",
                side="right",
                driver="opencv",
                serial_number=None,
            ),
            CameraBinding(
                id="front_cam",
                port="/dev/video-front",
                alias="environment_1",
                side="single",
                driver="opencv",
                serial_number=None,
            ),
        ],
    )


def _bi_piperx_configuration() -> DeviceConfiguration:
    return DeviceConfiguration(
        profile_id="bi_piperx",
        robot_type="bi_piperx_follower",
        camera_slots=[
            CameraSlot(alias="left_wrist", kind="wrist", side="left"),
            CameraSlot(alias="right_wrist", kind="wrist", side="right"),
            CameraSlot(alias="environment_1", kind="environment", side="single"),
        ],
        can_bindings=[
            CanBinding(id="can-left", alias="left_follower", kind="robot", side="left"),
            CanBinding(id="can-right", alias="right_follower", kind="robot", side="right"),
        ],
        camera_bindings=[
            CameraBinding(
                id="left_cam",
                port="/dev/video-left",
                alias="left_wrist",
                side="left",
                driver="opencv",
                serial_number=None,
            ),
            CameraBinding(
                id="right_cam",
                port="/dev/video-right",
                alias="right_wrist",
                side="right",
                driver="opencv",
                serial_number=None,
            ),
            CameraBinding(
                id="front_cam",
                port="/dev/video-front",
                alias="environment_1",
                side="single",
                driver="opencv",
                serial_number=None,
            ),
        ],
    )


def test_policy_path_accepts_hugging_face_url() -> None:
    assert (
        _policy_path("https://huggingface.co/Elvinky/pi05_full_mix_562ep_sft_fp32")
        == "Elvinky/pi05_full_mix_562ep_sft_fp32"
    )


def test_rollout_repo_id_adds_required_prefix() -> None:
    assert _rollout_repo_id("example") == "rollout_example"
    assert _rollout_repo_id("owner/example") == "owner/rollout_example"
    assert _rollout_repo_id("owner/rollout_example") == "owner/rollout_example"


def test_dual_arm_environment_camera_is_top_level() -> None:
    payload = _robot_payload(_bi_so_configuration(), 30, cameras=True)

    assert set(payload["left_arm_config"]["cameras"]) == {"wrist"}
    assert set(payload["right_arm_config"]["cameras"]) == {"wrist"}
    assert set(payload["cameras"]) == {"environment_1"}


def test_piper_environment_camera_stays_on_right_arm() -> None:
    configuration = _bi_piperx_configuration()

    payload = _robot_payload(configuration, 30, cameras=True)

    assert "cameras" not in payload
    assert set(payload["right_arm_config"]["cameras"]) == {"wrist", "environment_1"}


def test_piper_openpi_camera_mapping_uses_runtime_environment_name() -> None:
    configuration = _bi_piperx_configuration()
    provided = _configured_visual_features(configuration)
    expected = {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }

    assert provided == {
        "observation.images.left_wrist",
        "observation.images.right_environment_1",
        "observation.images.right_wrist",
    }
    assert _camera_rename_map(configuration, expected, provided) == {
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_environment_1": "observation.images.base_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
    }


def test_pi05_camera_mapping_and_vector_dimensions() -> None:
    configuration = _bi_so_configuration()
    provided = _configured_visual_features(configuration)
    expected = {
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.right_front",
    }

    assert _configured_vector_dimensions(configuration) == (12, 12)
    assert _camera_rename_map(configuration, expected, provided) == {
        "observation.images.environment_1": "observation.images.right_front"
    }


def test_pi05_openpi_camera_mapping() -> None:
    configuration = _bi_so_configuration()
    provided = _configured_visual_features(configuration)
    expected = {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }

    assert _camera_rename_map(configuration, expected, provided) == {
        "observation.images.environment_1": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
    }


def test_normalizer_feature_dim_reads_effective_unpadded_size(tmp_path) -> None:
    import torch
    from safetensors.torch import save_file

    save_file(
        {"observation.state.q01": torch.zeros(12)},
        tmp_path / "policy_preprocessor_step_3_normalizer_processor.safetensors",
    )

    assert _normalizer_feature_dim(str(tmp_path), "observation.state") == 12


def test_rollout_request_exposes_all_web_modes() -> None:
    for strategy in (
        "base",
        "episodic",
        "sentry",
        "highlight",
        "dagger_corrections",
        "dagger_continuous",
        "episodic_dagger",
    ):
        request = RolloutStartRequest(policy_path="model", strategy=strategy, task="task")
        assert request.strategy == strategy


def test_web_rtc_uses_evostudio_continuity_settings() -> None:
    config = _rollout_inference_config("rtc", SimpleNamespace(chunk_size=50))

    assert config.rtc.execution_horizon == 20
    assert config.rtc.max_guidance_weight == 5.0
    assert config.queue_threshold == 30


def test_web_rtc_continuity_settings_fit_short_chunks() -> None:
    config = _rollout_inference_config("rtc", SimpleNamespace(chunk_size=12))

    assert config.rtc.execution_horizon == 12
    assert config.queue_threshold == 11


def test_direct_rollout_resolves_policy_from_local_inventory(monkeypatch) -> None:
    monkeypatch.setattr(
        "evomind_lerobot.runtime_service.policies_inventory",
        lambda: [{"id": "pi05", "path": "/models/pi05", "type": "pi05"}],
    )
    events = EventBroker()
    service = RuntimeService(events, JobManager(events))
    captured = {}

    def fake_start(operation, request, collection_task):
        captured.update(operation=operation, request=request, task=collection_task)
        return {"running": True}

    monkeypatch.setattr(service, "_start", fake_start)

    result = service.start(
        Operation.ROLLOUT,
        RolloutStartRequest(policy_path="pi05", task="insert screw"),
    )

    assert result == {"running": True}
    assert captured["operation"] is Operation.ROLLOUT
    assert captured["request"].policy_path == "/models/pi05"


def test_resident_worker_reuses_one_loaded_policy_for_multiple_rollouts(monkeypatch) -> None:
    resident_policy = object()
    executions = []
    event_queue = queue.Queue()
    control_queue = queue.Queue()
    command_queue = queue.Queue()
    monkeypatch.setattr(
        "evomind_lerobot.runtime_service._load_resident_policy",
        lambda path: (resident_policy, {"policy_path": path, "device": "cuda"}),
    )
    monkeypatch.setattr(
        "evomind_lerobot.runtime_service._execute_rollout",
        lambda payload, *, preloaded_policy: executions.append((payload, preloaded_policy)),
    )
    worker = threading.Thread(
        target=_run_policy_resident,
        args=("/models/pi05", event_queue, control_queue, command_queue),
    )
    worker.start()

    assert event_queue.get(timeout=1)["kind"] == "resident_ready"
    for task in ("first", "second"):
        control_queue.put({"kind": "rollout", "payload": {"task": task}})
        assert event_queue.get(timeout=1) == {"kind": "job_exit", "error": ""}
    control_queue.put({"kind": "shutdown"})
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert [payload["task"] for payload, _policy in executions] == ["first", "second"]
    assert all(policy is resident_policy for _payload, policy in executions)
