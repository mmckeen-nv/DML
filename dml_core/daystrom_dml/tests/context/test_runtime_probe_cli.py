from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.dcm_kv_probe as dcm_kv_probe


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _argv(checkpoint_dir: Path, artifact: Path) -> list[str]:
    return [
        "--runtime-id",
        "runtime-a",
        "--runtime-version",
        "1.2.3",
        "--checkpoint-directory",
        str(checkpoint_dir),
        "--model-id",
        "model-a",
        "--model-digest",
        _digest("model"),
        "--tokenizer-digest",
        _digest("tokenizer"),
        "--positional-config-json",
        '{"n_ctx":4096,"rope":"default"}',
        "--model-limit-tokens",
        "4096",
        "--artifact",
        str(artifact),
    ]


def test_cli_uses_temporary_registry_and_writes_success_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact = tmp_path / "artifact.json"
    captured: dict[str, Any] = {}

    def fake_run_probe(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        assert kwargs["registry_directory"].is_dir()
        return {"artifact_version": "dcm-runtime-execution-probe-v2", "pass": True}

    monkeypatch.setattr(dcm_kv_probe, "run_probe", fake_run_probe)

    assert dcm_kv_probe.main(_argv(checkpoint_dir, artifact)) == 0
    assert json.loads(artifact.read_text())["pass"] is True
    assert not captured["registry_directory"].exists()
    assert captured["session_id"].startswith("probe-")
    assert captured["checkpoint_id"].startswith("probe-")


def test_cli_failure_artifact_digests_reason_without_leaking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact = tmp_path / "artifact.json"

    def fail_probe(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("private checkpoint path /secret/runtime/cache.bin")

    monkeypatch.setattr(dcm_kv_probe, "run_probe", fail_probe)

    assert dcm_kv_probe.main(_argv(checkpoint_dir, artifact)) == 1
    payload = json.loads(artifact.read_text())
    assert payload["pass"] is False
    assert payload["error_type"] == "RuntimeError"
    assert len(payload["reason_digest"]) == 64
    assert "secret" not in json.dumps(payload)
    assert "cache.bin" not in json.dumps(payload)


def test_cli_rejects_empty_positional_configuration_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact = tmp_path / "artifact.json"
    argv = _argv(checkpoint_dir, artifact)
    argv[argv.index("--positional-config-json") + 1] = "{}"
    called = False

    def fake_run_probe(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"pass": True}

    monkeypatch.setattr(dcm_kv_probe, "run_probe", fake_run_probe)
    with pytest.raises(SystemExit) as exc:
        dcm_kv_probe.main(argv)
    assert exc.value.code == 2
    assert called is False
