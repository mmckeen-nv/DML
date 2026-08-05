from __future__ import annotations

import json
from typing import Any

import pytest

from daystrom_dml.context.adapters.llama_cpp import LlamaCppExecutionAdapter
from daystrom_dml.context.execution import RuntimeCacheOperation, RuntimeExecutionError
from daystrom_dml.context.probe import ProbeSettings


class Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json.dumps(payload).encode()
        self.status = 200

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]

    def getcode(self) -> int:
        return self.status


class Opener:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float) -> Response:
        self.calls.append({"url": req.full_url, "body": json.loads(req.data or b"{}"), "timeout": timeout})
        return Response(self.responses.pop(0))


def test_completion_trace_measures_materialized_kv_reuse_without_persisting_text() -> None:
    opener = Opener([
        {
            "content": "ORBIT",
            "tokens": [1],
            "tokens_cached": 100,
            "tokens_evaluated": 100,
            "timings": {"prompt_n": 4, "prompt_ms": 2.5, "predicted_n": 1},
            "truncated": False,
        }
    ])
    adapter = LlamaCppExecutionAdapter("http://127.0.0.1:18080", runtime_id="llama-test", opener=opener)

    trace = adapter.complete("SECRET PROMPT", slot_id=0, n_predict=1)

    assert trace.prompt_tokens_total == 100
    assert trace.prompt_tokens_processed == 4
    assert trace.prompt_tokens_reused == 96
    assert trace.output_text == "ORBIT"
    assert trace.output_token_ids == [1]
    assert "SECRET PROMPT" not in json.dumps(trace.to_telemetry())
    assert "ORBIT" not in json.dumps(trace.to_telemetry())
    assert opener.calls[0]["body"]["cache_prompt"] is True
    assert opener.calls[0]["body"]["id_slot"] == 0


def test_slot_checkpoint_restore_and_erase_use_narrow_operations() -> None:
    opener = Opener([
        {"id_slot": 0, "filename": "prefix.bin", "n_saved": 100, "n_written": 2000, "timings": {"save_ms": 3}},
        {"id_slot": 0, "filename": "prefix.bin", "n_restored": 100, "n_read": 2000, "timings": {"restore_ms": 2}},
        {"id_slot": 0, "n_erased": 100},
    ])
    adapter = LlamaCppExecutionAdapter("http://127.0.0.1:18080", runtime_id="llama-test", opener=opener)

    saved = adapter.save_slot(0, "prefix.bin")
    restored = adapter.restore_slot(0, "prefix.bin")
    erased = adapter.erase_slot(0)

    assert saved.operation is RuntimeCacheOperation.SAVE and saved.tokens_affected == 100
    assert restored.operation is RuntimeCacheOperation.RESTORE and restored.tokens_affected == 100
    assert erased.operation is RuntimeCacheOperation.ERASE
    assert [call["url"].split("?")[-1] for call in opener.calls] == ["action=save", "action=restore", "action=erase"]


def test_checkpoint_filename_and_runtime_payloads_fail_closed() -> None:
    adapter = LlamaCppExecutionAdapter("http://127.0.0.1:18080", runtime_id="llama-test", opener=Opener([]))
    with pytest.raises(RuntimeExecutionError, match="filename"):
        adapter.save_slot(0, "../escape.bin")

    malformed = LlamaCppExecutionAdapter(
        "http://127.0.0.1:18080",
        runtime_id="llama-test",
        opener=Opener([{"content": "x", "tokens_evaluated": 10, "timings": {"prompt_n": 20}}]),
    )
    with pytest.raises(RuntimeExecutionError, match="prompt token counters"):
        malformed.complete("prompt", slot_id=0)


def test_local_checkpoint_delete_is_confined_to_configured_directory(tmp_path: Any) -> None:
    checkpoint = tmp_path / "prefix.bin"
    checkpoint.write_bytes(b"sensitive-kv")
    adapter = LlamaCppExecutionAdapter(
        "http://127.0.0.1:18080",
        runtime_id="llama-test",
        runtime_version="10250",
        checkpoint_directory=tmp_path,
        opener=Opener([]),
    )

    result = adapter.delete_checkpoint("prefix.bin")

    assert result.existed is True
    assert result.bytes_deleted == len(b"sensitive-kv")
    assert not checkpoint.exists()
    assert adapter.capabilities().supports_kv_checkpoint_delete is True
    with pytest.raises(RuntimeExecutionError, match="filename"):
        adapter.delete_checkpoint("../outside.bin")


def test_checkpoint_delete_rejects_symlink_and_unconfigured_remote_runtime(tmp_path: Any) -> None:
    target = tmp_path / "outside.bin"
    target.write_bytes(b"keep")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    configured = LlamaCppExecutionAdapter(
        "http://127.0.0.1:18080",
        runtime_id="llama-test",
        runtime_version="10250",
        checkpoint_directory=tmp_path,
        opener=Opener([]),
    )
    with pytest.raises(RuntimeExecutionError, match="regular file"):
        configured.delete_checkpoint("link.bin")
    assert target.exists()

    remote = LlamaCppExecutionAdapter(
        "http://127.0.0.1:18080", runtime_id="llama-test", runtime_version="10250", opener=Opener([])
    )
    assert remote.capabilities().supports_kv_checkpoint_delete is False
    with pytest.raises(RuntimeExecutionError, match="not configured"):
        remote.delete_checkpoint("prefix.bin")


def test_slot_bound_chat_completion_uses_restored_slot_and_reports_native_reuse() -> None:
    opener = Opener(
        [
            {
                "choices": [{"message": {"role": "assistant", "content": "ORBIT-9"}}],
                "usage": {
                    "prompt_tokens": 130,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 128},
                },
                "tokens_evaluated": 130,
                "timings": {"prompt_n": 2, "prompt_ms": 4.5},
            }
        ]
    )
    adapter = LlamaCppExecutionAdapter(
        "http://127.0.0.1:18080",
        runtime_id="llama-test",
        runtime_version="10250",
        opener=opener,
    )

    response = adapter.complete_on_slot(
        "http://127.0.0.1:18080/v1/chat/completions",
        "llama3:8b",
        [{"role": "user", "content": "Use restored context.", "tool_calls": [{"secret": "drop"}]}],
        ProbeSettings(max_output_tokens=8),
        slot_id=4,
        label="fault-retry-1",
    )

    assert response.content == "ORBIT-9"
    assert response.usage["prompt_tokens_processed"] == 2
    assert response.usage["prompt_tokens_reused"] == 128
    assert opener.calls[0]["url"].endswith("/v1/chat/completions")
    assert opener.calls[0]["body"]["id_slot"] == 4
    assert opener.calls[0]["body"]["cache_prompt"] is True
    assert opener.calls[0]["body"]["messages"] == [
        {"role": "user", "content": "Use restored context."}
    ]
    with pytest.raises(RuntimeExecutionError, match="origin"):
        adapter.complete_on_slot(
            "http://127.0.0.1:19999/v1/chat/completions",
            "llama3:8b",
            [{"role": "user", "content": "x"}],
            ProbeSettings(),
            slot_id=4,
        )


def test_capabilities_are_explicit_and_provider_neutral() -> None:
    adapter = LlamaCppExecutionAdapter("http://127.0.0.1:18080", runtime_id="llama-test", runtime_version="10250")
    caps = adapter.capabilities()
    assert caps.runtime_id == "llama-test"
    assert caps.adapter_id == "llama_cpp_server"
    assert caps.supports_prompt_cache is True
    assert caps.supports_kv_checkpoint is True
    assert caps.supports_kv_restore is True
    assert caps.supports_kv_erase is True
    assert caps.runtime_version == "10250"
