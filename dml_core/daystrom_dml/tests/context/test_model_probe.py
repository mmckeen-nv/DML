from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from daystrom_dml.context.probe import (
    EvaluationSpec,
    FakeModelClient,
    ModelProbeRequest,
    OpenAICompatibleModelClient,
    ProbeSettings,
    ProbeTransportError,
    atomic_write_json,
    build_probe_request,
    manifest_messages,
    redact_endpoint_url,
    run_ab_probe,
)
from scripts.dcm_model_probe import main as cli_main


BASELINE_MESSAGES = [{"role": "user", "content": "Keep fact alpha. Never say LEAK."}]
MANAGED_MESSAGES = [
    {"role": "system", "content": "Instruction marker: SURVIVE."},
    {"role": "user", "content": "Keep fact alpha. Never say LEAK."},
]


def test_manifest_messages_uses_digest_without_prompt_text() -> None:
    manifest = manifest_messages(BASELINE_MESSAGES, label="baseline")

    payload = json.dumps(manifest.to_dict(), sort_keys=True)
    assert manifest.message_count == 1
    assert len(manifest.content_digest) == 64
    assert "alpha" not in payload
    assert "LEAK" not in payload


def test_json_contracts_roundtrip_without_prompt_or_completion_text() -> None:
    request = build_probe_request(
        run_id="run-1",
        endpoint_url="https://user:pass@example.test/v1/chat/completions?api_key=secret",
        model_id="gpt-test",
        baseline_messages=BASELINE_MESSAGES,
        managed_messages=MANAGED_MESSAGES,
        managed_authority_manifest={"authority": ["system", "user"]},
    )
    result = run_ab_probe(
        request,
        FakeModelClient({"baseline": "fact alpha SURVIVE", "managed": "fact alpha SURVIVE"}),
        EvaluationSpec(required_facts=["alpha"], forbidden_markers=["LEAK"], instruction_survival_marker="SURVIVE"),
    )

    request_roundtrip = ModelProbeRequest.from_dict(json.loads(json.dumps(request.to_dict(), sort_keys=True)))
    result_roundtrip = type(result).from_dict(json.loads(json.dumps(result.to_dict(), sort_keys=True)))

    assert request_roundtrip == request
    assert result_roundtrip == result
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "user:pass" not in serialized
    assert "api_key=secret" not in serialized
    assert "Keep fact alpha" not in serialized
    assert "fact alpha SURVIVE" not in serialized


def test_fake_ab_run_evaluates_facts_instruction_leaks_latency_and_tokens() -> None:
    request = build_probe_request(
        run_id="run-2",
        endpoint_url="https://example.test/v1/chat/completions",
        model_id="gpt-test",
        baseline_messages=BASELINE_MESSAGES,
        managed_messages=MANAGED_MESSAGES,
        managed_authority_manifest={"digest": "authority"},
        settings=ProbeSettings(temperature=0, max_output_tokens=24, timeout_seconds=3),
    )
    client = FakeModelClient(
        {
            "baseline": "fact alpha SURVIVE",
            "managed": "fact alpha SURVIVE LEAK",
        },
        usage={"baseline": {"total_tokens": 11}, "managed": {"total_tokens": 13}},
        latencies={"baseline": 10, "managed": 30},
    )

    result = run_ab_probe(
        request,
        client,
        EvaluationSpec(required_facts=["alpha"], forbidden_markers=["LEAK"], instruction_survival_marker="SURVIVE"),
    )

    assert result.status == "completed"
    assert result.baseline.status == "ok"
    assert result.managed.status == "ok"
    assert result.evaluator_outcomes["responses_equal"] is False
    assert result.evaluator_outcomes["required_fact_retention"]["alpha"] == {"baseline": True, "managed": True}
    assert result.evaluator_outcomes["forbidden_marker_leakage"]["LEAK"] == {"baseline": False, "managed": True}
    assert result.evaluator_outcomes["instruction_survival_marker"] == {"baseline": True, "managed": True}
    assert result.evaluator_outcomes["latency_delta_ms"] == pytest.approx(20.0)
    assert result.evaluator_outcomes["token_use_delta"]["total_tokens"] == 2


def test_runner_sends_identical_settings_and_requires_managed_authority_manifest() -> None:
    request = build_probe_request(
        run_id="run-3",
        endpoint_url="https://example.test/v1/chat/completions",
        model_id="gpt-test",
        baseline_messages=BASELINE_MESSAGES,
        managed_messages=MANAGED_MESSAGES,
        managed_authority_manifest={"digest": "authority"},
        settings=ProbeSettings(temperature=0.2, max_output_tokens=9, timeout_seconds=4),
    )
    client = FakeModelClient({"baseline": "ok", "managed": "ok"})

    run_ab_probe(request, client, EvaluationSpec())

    assert [call["label"] for call in client.calls] == ["baseline", "managed"]
    assert client.calls[0]["settings"] == client.calls[1]["settings"] == request.settings.to_dict()

    missing_authority = ModelProbeRequest.from_dict({**request.to_dict(), "managed_authority_manifest_digest": ""})
    with pytest.raises(ValueError, match="managed_authority_manifest_digest"):
        run_ab_probe(missing_authority, client, EvaluationSpec())


def test_endpoint_redaction_removes_userinfo_and_secret_query_values() -> None:
    redacted = redact_endpoint_url("https://user:pass@example.test/v1/chat/completions?api_key=secret&safe=1")

    assert redacted == "https://example.test/v1/chat/completions?api_key=REDACTED&safe=1"
    assert "secret" not in redacted
    assert "user:pass" not in redacted


def test_cli_dry_run_prints_sanitized_request_and_uses_no_network(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(
        [
            "--dry-run",
            "--endpoint-url",
            "https://user:pass@example.test/v1/chat/completions?api_key=secret",
            "--model",
            "gpt-test",
            "--baseline-messages-json",
            json.dumps(BASELINE_MESSAGES),
            "--managed-messages-json",
            json.dumps(MANAGED_MESSAGES),
            "--managed-authority-manifest-json",
            json.dumps({"authority": "ok"}),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"dry_run": true' in captured.out
    assert "example.test" in captured.out
    assert "secret" not in captured.out
    assert "user:pass" not in captured.out
    assert "Keep fact alpha" not in captured.out


def test_cli_live_mode_requires_allow_network(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(
        [
            "--endpoint-url",
            "https://example.test/v1/chat/completions",
            "--model",
            "gpt-test",
            "--baseline-messages-json",
            json.dumps(BASELINE_MESSAGES),
            "--managed-messages-json",
            json.dumps(MANAGED_MESSAGES),
            "--managed-authority-manifest-json",
            json.dumps({"authority": "ok"}),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--allow-network" in captured.err


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, limit: int = -1) -> bytes:
        if limit < 0:
            return self.body
        return self.body[:limit]


class _FakeOpener:
    def __init__(self, body: bytes, status: int = 200, raise_http: bool = False) -> None:
        self.body = body
        self.status = status
        self.raise_http = raise_http
        self.calls: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float) -> _FakeHTTPResponse:
        self.calls.append(
            {
                "url": req.full_url,
                "timeout": timeout,
                "authorization": req.headers.get("Authorization"),
                "body": json.loads(req.data.decode("utf-8")),
            }
        )
        if self.raise_http:
            raise HTTPError(req.full_url, self.status, "bad", hdrs=None, fp=None)
        return _FakeHTTPResponse(self.body, self.status)


def test_openai_compatible_client_http_success_and_secret_header() -> None:
    opener = _FakeOpener(
        json.dumps({"choices": [{"message": {"content": "hello alpha"}}], "usage": {"total_tokens": 7}}).encode(
            "utf-8"
        )
    )
    client = OpenAICompatibleModelClient(api_key="sk-secret", max_response_bytes=2048, opener=opener)
    response = client.complete(
        endpoint_url="http://127.0.0.1/v1/chat/completions",
        model_id="gpt-test",
        messages=BASELINE_MESSAGES,
        settings=ProbeSettings(max_output_tokens=5),
    )

    assert response.content == "hello alpha"
    assert response.usage == {"total_tokens": 7}
    assert opener.calls[0]["authorization"] == "Bearer sk-secret"
    assert opener.calls[0]["body"]["model"] == "gpt-test"
    assert opener.calls[0]["body"]["temperature"] == 0
    assert opener.calls[0]["body"]["max_tokens"] == 5


def test_openai_compatible_client_http_error_non_json_missing_content_and_oversize() -> None:
    endpoint_url = "http://127.0.0.1/v1/chat/completions"
    client = OpenAICompatibleModelClient(
        max_response_bytes=80,
        opener=_FakeOpener(b'{"error":"bad"}', status=500, raise_http=True),
    )

    with pytest.raises(ProbeTransportError, match="http_error"):
        client.complete(endpoint_url, "gpt-test", BASELINE_MESSAGES, ProbeSettings())

    client = OpenAICompatibleModelClient(max_response_bytes=80, opener=_FakeOpener(b"not-json"))
    with pytest.raises(ProbeTransportError, match="non_json"):
        client.complete(endpoint_url, "gpt-test", BASELINE_MESSAGES, ProbeSettings())

    client = OpenAICompatibleModelClient(max_response_bytes=80, opener=_FakeOpener(b'{"choices":[{"message":{}}]}'))
    with pytest.raises(ProbeTransportError, match="missing_content"):
        client.complete(endpoint_url, "gpt-test", BASELINE_MESSAGES, ProbeSettings())

    client = OpenAICompatibleModelClient(
        max_response_bytes=80,
        opener=_FakeOpener(json.dumps({"choices": [{"message": {"content": "x" * 200}}]}).encode("utf-8")),
    )
    with pytest.raises(ProbeTransportError, match="oversized_response"):
        client.complete(endpoint_url, "gpt-test", BASELINE_MESSAGES, ProbeSettings())


def test_atomic_write_json_replaces_target_without_temp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    atomic_write_json(target, {"old": True})
    atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_console_help_and_cli_dry_run_smoke() -> None:
    help_run = subprocess.run(
        [sys.executable, "-m", "scripts.dcm_model_probe", "--help"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "--allow-network" in help_run.stdout

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dcm_model_probe",
            "--dry-run",
            "--endpoint-url",
            "https://example.test/v1/chat/completions",
            "--model",
            "gpt-test",
            "--baseline-messages-json",
            json.dumps(BASELINE_MESSAGES),
            "--managed-messages-json",
            json.dumps(MANAGED_MESSAGES),
            "--managed-authority-manifest-json",
            json.dumps({"authority": "ok"}),
        ],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0
    assert '"dry_run": true' in dry_run.stdout
