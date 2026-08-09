from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daystrom_dml.context.adapters.vllm import VLLMCooperativeExecutionAdapter
from daystrom_dml.context.execution import RuntimeExecutionError


class Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]


class Opener:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float) -> Response:
        self.calls.append(
            {"url": req.full_url, "body": json.loads(req.data), "timeout": timeout}
        )
        return Response(self.payload)


class SequenceOpener(Opener):
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        super().__init__(payloads[0])
        self.payloads = list(payloads)

    def __call__(self, req: Any, timeout: float) -> Response:
        self.calls.append(
            {"url": req.full_url, "body": json.loads(req.data), "timeout": timeout}
        )
        return Response(self.payloads.pop(0))


def _secret(tmp_path: Path) -> Path:
    path = tmp_path / "control.key"
    path.write_text("bounded-test-secret\n")
    return path


def _digest() -> str:
    return "sha256:" + "a" * 64


def _response(operation: str = "save", digest: str | None = None) -> dict[str, Any]:
    return {
        "choices": [{"text": "OK"}],
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 96},
        },
        "kv_transfer_params": {
            "daystrom": {
                "schema_version": "daystrom-vllm-kv-v1",
                "operation": operation,
                "checkpoint_digest": digest or _digest(),
                "reason_code": (
                    "purge_complete" if operation == "purge" else f"{operation}_authorized"
                ),
                "matched_tokens": 96 if operation == "restore" else 0,
                "gpu_apc_matched_tokens": 32 if operation == "restore" else 0,
                "cpu_offload_matched_tokens": 96 if operation == "restore" else 0,
                "cache_route": (
                    "gpu_apc_and_cpu" if operation == "restore" else "not_applicable"
                ),
                "saved_tokens": 128 if operation == "save" else 0,
                "purged_blocks": 2 if operation == "purge" else 0,
                "purged_bytes": 4096 if operation == "purge" else 0,
                "shared_blocks": 1 if operation == "purge" else 0,
            }
        },
    }


def _status_response(
    reason_code: str,
    *,
    ready: bool = False,
    stored: int = 0,
    expected: int = 0,
) -> dict[str, Any]:
    payload = _response("status")
    payload["kv_transfer_params"]["daystrom"].update(
        {
            "reason_code": reason_code,
            "checkpoint_ready": ready,
            "stored_blocks": stored,
            "expected_blocks": expected,
        }
    )
    return payload


def test_request_is_signed_and_native_evidence_is_validated(tmp_path: Path) -> None:
    opener = Opener(_response())
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=opener,
    )

    trace = adapter.complete_with_checkpoint(
        "PRIVATE PROMPT",
        operation="save",
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce="nonce-1",
    )

    assert trace.saved_tokens == 128
    assert trace.cached_tokens == 96
    assert opener.calls[0]["url"].endswith("/v1/completions")
    envelope = opener.calls[0]["body"]["kv_transfer_params"]["daystrom"]
    assert envelope["operation"] == "save"
    assert len(envelope["authorization"]) == 64
    telemetry = json.dumps(trace.to_telemetry())
    assert "PRIVATE PROMPT" not in telemetry
    assert "OK" not in telemetry
    assert "nonce-1" not in telemetry
    assert "authorization" not in telemetry


def test_restore_uses_native_matched_token_count(tmp_path: Path) -> None:
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(_response("restore")),
    )

    trace = adapter.complete_with_checkpoint(
        "same prefix plus continuation",
        operation="restore",
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce="nonce-2",
    )

    assert trace.operation == "restore"
    assert trace.matched_tokens == 96
    assert trace.gpu_apc_matched_tokens == 32
    assert trace.cpu_offload_matched_tokens == 96
    assert trace.cache_route == "gpu_apc_and_cpu"
    assert trace.reason_code == "restore_authorized"


def test_checkpoint_status_returns_typed_payload_free_readiness(tmp_path: Path) -> None:
    opener = Opener(
        _status_response("checkpoint_ready", ready=True, stored=4, expected=4)
    )
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=opener,
    )

    status = adapter.checkpoint_status(
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce="status-nonce",
    )

    assert status.operation == "status"
    assert status.reason_code == "checkpoint_ready"
    assert status.checkpoint_ready is True
    assert status.stored_blocks == 4
    assert status.expected_blocks == 4
    envelope = opener.calls[0]["body"]["kv_transfer_params"]["daystrom"]
    assert envelope["operation"] == "status"
    telemetry = json.dumps(status.to_telemetry())
    assert "status-nonce" not in telemetry
    assert "authorization" not in telemetry


@pytest.mark.parametrize(
    ("reason_code", "stored", "expected"),
    [
        ("checkpoint_pending", 0, 4),
        ("checkpoint_partial", 2, 4),
        ("checkpoint_evicted", 0, 4),
        ("checkpoint_below_granularity", 0, 0),
        ("record_not_found", 0, 0),
        ("record_expired", 0, 0),
        ("purge_pending", 0, 0),
        ("purge_complete", 0, 0),
        ("purge_shared_ownership_changed", 0, 0),
    ],
)
def test_checkpoint_status_accepts_fail_closed_nonready_states(
    tmp_path: Path, reason_code: str, stored: int, expected: int
) -> None:
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(
            _status_response(reason_code, stored=stored, expected=expected)
        ),
    )

    status = adapter.checkpoint_status(
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce=f"status-{reason_code}",
    )

    assert status.reason_code == reason_code
    assert status.checkpoint_ready is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda daystrom: daystrom.update({"checkpoint_ready": "yes"}),
        lambda daystrom: daystrom.update({"stored_blocks": -1}),
        lambda daystrom: daystrom.update({"expected_blocks": True}),
        lambda daystrom: daystrom.update(
            {
                "reason_code": "checkpoint_ready",
                "checkpoint_ready": True,
                "stored_blocks": 3,
                "expected_blocks": 4,
            }
        ),
        lambda daystrom: daystrom.update(
            {
                "reason_code": "checkpoint_partial",
                "checkpoint_ready": True,
                "stored_blocks": 3,
                "expected_blocks": 4,
            }
        ),
    ],
)
def test_checkpoint_status_rejects_inconsistent_readiness(
    tmp_path: Path, mutator: Any
) -> None:
    payload = _status_response("checkpoint_ready", ready=True, stored=4, expected=4)
    mutator(payload["kv_transfer_params"]["daystrom"])
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(payload),
    )

    with pytest.raises(RuntimeExecutionError, match="readiness"):
        adapter.checkpoint_status(
            checkpoint_digest=_digest(),
            expires_at=4_000_000_000.0,
            nonce="status-invalid",
        )


def test_wait_for_checkpoint_ready_uses_fresh_signed_status_requests(
    tmp_path: Path,
) -> None:
    opener = SequenceOpener(
        [
            _status_response("checkpoint_pending", stored=0, expected=4),
            _status_response(
                "checkpoint_ready", ready=True, stored=4, expected=4
            ),
        ]
    )
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=opener,
    )

    status = adapter.wait_for_checkpoint_ready(
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce="ready-base",
        max_attempts=2,
        poll_interval_seconds=0,
    )

    assert status.checkpoint_ready is True
    nonces = [
        call["body"]["kv_transfer_params"]["daystrom"]["nonce"]
        for call in opener.calls
    ]
    assert len(set(nonces)) == 2
    assert all(
        call["body"]["kv_transfer_params"]["daystrom"]["operation"]
        == "status"
        for call in opener.calls
    )


def test_wait_for_checkpoint_ready_aborts_on_nonpending_state(tmp_path: Path) -> None:
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(
            _status_response("checkpoint_partial", stored=2, expected=4)
        ),
    )

    with pytest.raises(RuntimeExecutionError, match="checkpoint_partial"):
        adapter.wait_for_checkpoint_ready(
            checkpoint_digest=_digest(),
            expires_at=4_000_000_000.0,
            nonce="ready-partial",
            max_attempts=2,
            poll_interval_seconds=0,
        )


def test_purge_requires_completed_physical_counters(tmp_path: Path) -> None:
    pending = _response("purge")
    pending["kv_transfer_params"]["daystrom"].update(
        {
            "reason_code": "purge_pending",
            "purged_blocks": 0,
            "purged_bytes": 0,
            "shared_blocks": 1,
        }
    )
    complete = _status_response("purge_complete")
    complete["kv_transfer_params"]["daystrom"].update(
        {"purged_blocks": 2, "purged_bytes": 4096, "shared_blocks": 1}
    )
    opener = SequenceOpener([pending, complete])
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=opener,
    )

    trace = adapter.purge_checkpoint(
        checkpoint_digest=_digest(),
        expires_at=4_000_000_000.0,
        nonce="nonce-purge",
    )

    assert trace.reason_code == "purge_complete"
    assert trace.purged_blocks == 2
    assert trace.purged_bytes == 4096
    assert trace.shared_blocks == 1
    assert len(opener.calls) == 2
    first_envelope = opener.calls[0]["body"]["kv_transfer_params"]["daystrom"]
    second_envelope = opener.calls[1]["body"]["kv_transfer_params"]["daystrom"]
    assert first_envelope["operation"] == "purge"
    assert second_envelope["operation"] == "status"
    assert first_envelope["nonce"] != second_envelope["nonce"]


def test_capabilities_fail_closed_for_unimplemented_lifecycle(tmp_path: Path) -> None:
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(_response()),
    )

    caps = adapter.capabilities()
    assert caps.supports_kv_checkpoint is True
    assert caps.supports_kv_restore is True
    assert caps.supports_kv_erase is False
    assert caps.supports_kv_checkpoint_delete is False
    assert caps.supports_slot_affinity is False
    assert caps.metadata["physical_purge"] is False
    assert caps.metadata["cache_hierarchy"] == ["gpu_apc", "cpu_offload"]
    assert caps.metadata["gpu_apc_controller_scoped"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["kv_transfer_params"]["daystrom"].update(
                {"checkpoint_digest": "sha256:" + "b" * 64}
            ),
            "checkpoint mismatch",
        ),
        (
            lambda payload: payload["kv_transfer_params"]["daystrom"].update(
                {"reason_code": "record_not_found"}
            ),
            "not authorized",
        ),
        (
            lambda payload: payload.pop("kv_transfer_params"),
            "lacks cooperative KV evidence",
        ),
        (
            lambda payload: payload["kv_transfer_params"]["daystrom"].update(
                {"cache_route": ["gpu_apc"]}
            ),
            "invalid runtime cache route",
        ),
    ],
)
def test_malformed_or_denied_connector_evidence_fails_closed(
    tmp_path: Path, mutator: Any, message: str
) -> None:
    payload = _response()
    mutator(payload)
    adapter = VLLMCooperativeExecutionAdapter(
        "http://127.0.0.1:8000",
        model_id="model",
        runtime_id="vllm-test",
        runtime_version="0.20.0",
        secret_path=_secret(tmp_path),
        opener=Opener(payload),
    )

    with pytest.raises(RuntimeExecutionError, match=message):
        adapter.complete_with_checkpoint(
            "prompt",
            operation="save",
            checkpoint_digest=_digest(),
            expires_at=4_000_000_000.0,
            nonce="nonce-3",
        )


def test_exact_runtime_version_and_secret_file_are_required(tmp_path: Path) -> None:
    with pytest.raises(RuntimeExecutionError, match="runtime_version"):
        VLLMCooperativeExecutionAdapter(
            "http://127.0.0.1:8000",
            model_id="model",
            runtime_id="vllm-test",
            runtime_version="unknown",
            secret_path=_secret(tmp_path),
        )
    with pytest.raises(RuntimeExecutionError, match="secret_path"):
        VLLMCooperativeExecutionAdapter(
            "http://127.0.0.1:8000",
            model_id="model",
            runtime_id="vllm-test",
            runtime_version="0.20.0",
            secret_path=tmp_path / "missing.key",
        )
