import json
from pathlib import Path

from lerobot.rollout.context import _resolve_local_tokenizer


def _write_processor_config(policy_dir: Path, tokenizer_name: str) -> None:
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": tokenizer_name},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_tokenizer(tokenizer_dir: Path) -> Path:
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return tokenizer_dir.resolve()


def test_resolves_hub_tokenizer_from_shared_registry(tmp_path, monkeypatch):
    lerobot_home = tmp_path / "lerobot"
    tokenizer_dir = _write_tokenizer(lerobot_home / "tokenizers" / "google--paligemma-3b-pt-224")
    policy_dir = tmp_path / "policy"
    _write_processor_config(policy_dir, "google/paligemma-3b-pt-224")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(lerobot_home))

    assert _resolve_local_tokenizer(str(policy_dir)) == str(tokenizer_dir)


def test_resolves_stale_training_path_by_tokenizer_directory_name(tmp_path, monkeypatch):
    lerobot_home = tmp_path / "lerobot"
    tokenizer_dir = _write_tokenizer(lerobot_home / "tokenizers" / "google--paligemma-3b-pt-224")
    policy_dir = tmp_path / "policy"
    _write_processor_config(
        policy_dir,
        "/training/host/checkpoint/tokenizers/google--paligemma-3b-pt-224",
    )
    monkeypatch.setenv("HF_LEROBOT_HOME", str(lerobot_home))

    assert _resolve_local_tokenizer(str(policy_dir)) == str(tokenizer_dir)


def test_existing_configured_tokenizer_does_not_need_override(tmp_path):
    tokenizer_dir = _write_tokenizer(tmp_path / "existing-tokenizer")
    policy_dir = tmp_path / "policy"
    _write_processor_config(policy_dir, str(tokenizer_dir))

    assert _resolve_local_tokenizer(str(policy_dir)) is None
