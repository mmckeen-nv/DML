from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "scripts" / "dcm_native_context_profile.py"
_SPEC = importlib.util.spec_from_file_location("dcm_native_context_profile", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def payload() -> dict:
    return {
        "config": {
            "model_id": "nvidia/nemotron-super",
            "runtime_id": "vllm-0.20",
            "model_native_limit": 262144,
            "served_limit": 65536,
            "target_hot_tokens": 60000,
            "stale_after_turns": 4,
            "freeze_after_turns": 10,
            "runtime_state_bytes_per_token": 1024,
        },
        "requested_span_ids": ["frozen-evidence"],
        "spans": [
            {
                "span_id": "system",
                "content_digest": digest("system"),
                "start_token": 0,
                "token_count": 4096,
                "authority": "immutable",
                "priority": "critical",
                "exact_required": True,
            },
            {
                "span_id": "working",
                "content_digest": digest("working"),
                "start_token": 4096,
                "token_count": 32768,
                "priority": "working",
                "age_turns": 1,
            },
            {
                "span_id": "stale",
                "content_digest": digest("stale"),
                "start_token": 36864,
                "token_count": 60000,
                "age_turns": 6,
                "summary_digest": digest("summary-stale"),
                "summary_tokens": 6000,
            },
            {
                "span_id": "cold",
                "content_digest": digest("cold"),
                "start_token": 96864,
                "token_count": 80000,
                "age_turns": 14,
            },
            {
                "span_id": "frozen-evidence",
                "content_digest": digest("frozen-evidence"),
                "start_token": 176864,
                "token_count": 16000,
                "resident": False,
                "age_turns": 20,
            },
        ],
    }


def test_cli_writes_successful_digest_only_profile(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    artifact = tmp_path / "artifact.json"
    source.write_text(json.dumps(payload()))

    assert cli.main(["--input", str(source), "--artifact", str(artifact)]) == 0

    report = json.loads(artifact.read_text())
    assert report["pass"] is True
    assert report["model_id"] == "nvidia/nemotron-super"
    assert report["model_native_limit"] == 262144
    assert report["served_limit"] == 65536
    assert report["served_limit_shortfall"] == 196608
    assert report["resident_tokens_after"] <= 60000
    assert report["hot_swap_in"] == ["frozen-evidence"]
    assert "spans" not in report
    assert "summary_digest" not in report
    assert len(report["profile_digest"]) == 64


def test_cli_failure_digests_reason_without_leaking_payload(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    artifact = tmp_path / "artifact.json"
    bad = payload()
    bad["spans"][0]["content_digest"] = "PRIVATE CONTEXT MUST NOT LEAK"
    source.write_text(json.dumps(bad))

    assert cli.main(["--input", str(source), "--artifact", str(artifact)]) == 1

    report = json.loads(artifact.read_text())
    assert report["pass"] is False
    assert report["error_type"] == "ContractError"
    assert len(report["reason_digest"]) == 64
    serialized = json.dumps(report)
    assert "PRIVATE" not in serialized
    assert "CONTEXT" not in serialized


def test_cli_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    artifact = tmp_path / "artifact.json"
    value = payload()
    value["raw_prompt"] = "must not be accepted"
    source.write_text(json.dumps(value))

    assert cli.main(["--input", str(source), "--artifact", str(artifact)]) == 1
    report = json.loads(artifact.read_text())
    assert report["pass"] is False
    assert report["error_type"] == "ContractError"


def test_cli_executes_directly_outside_repository(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    artifact = tmp_path / "artifact.json"
    source.write_text(json.dumps(payload()))

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(source),
            "--artifact",
            str(artifact),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(artifact.read_text())["pass"] is True
