from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.checkpoints import ExecutionCheckpointIdentity, FileExecutionCheckpointRegistry
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeCompletionTrace,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)
from daystrom_dml.context.probe import endpoint_origin_identity_digest
from daystrom_dml.context.runtime_probe import run_runtime_execution_probe

ENDPOINT = "http://127.0.0.1:18080"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _identity() -> ExecutionCheckpointIdentity:
    return ExecutionCheckpointIdentity(
        scope=DaystromScope(tenant_id="tenant-secret", session_id="session-secret"),
        model_id="model-a",
        model_digest=_digest("model"),
        tokenizer_digest=_digest("tokenizer"),
        positional_config_digest=_digest("position"),
        immutable_prefix_digest=_digest("private immutable prefix"),
        packet_digest=_digest("packet"),
        manifest_digest=_digest("manifest"),
        runtime_id="runtime-a",
        runtime_version="1.2.3",
        adapter_id="fake-runtime",
        runtime_endpoint_digest="sha256:" + endpoint_origin_identity_digest(ENDPOINT),
    )


class FakeProbeAdapter:
    def __init__(
        self,
        root: Path,
        *,
        restore_error: bool = False,
        bad_trace: bool = False,
        bad_first_erase: bool = False,
        delete_error: bool = False,
        final_erase_error: bool = False,
    ) -> None:
        self.root = root
        self.restore_error = restore_error
        self.bad_trace = bad_trace
        self.bad_first_erase = bad_first_erase
        self.delete_error = delete_error
        self.final_erase_error = final_erase_error
        self.erase_count = 0
        self.hot = False
        self.calls: list[tuple[Any, ...]] = []
        self._caps = RuntimeExecutionCapabilities(
            runtime_id="runtime-a",
            adapter_id="fake-runtime",
            runtime_version="1.2.3",
            supports_prompt_cache=True,
            supports_kv_checkpoint=True,
            supports_kv_restore=True,
            supports_kv_erase=True,
            supports_kv_checkpoint_delete=True,
            supports_slot_affinity=True,
            supports_metrics=True,
            metadata={
                "endpoint_origin_digest": "sha256:" + endpoint_origin_identity_digest(ENDPOINT),
                "private_runtime_note": "must-not-leak",
            },
        )

    def capabilities(self) -> RuntimeExecutionCapabilities:
        return self._caps

    def complete(
        self,
        prompt: str,
        *,
        slot_id: int,
        n_predict: int = 1,
        cache_prompt: bool = True,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> RuntimeCompletionTrace:
        self.calls.append(("complete", slot_id, n_predict, cache_prompt, temperature, seed))
        processed = 4 if self.hot else 100
        self.hot = True
        return RuntimeCompletionTrace(
            runtime_id="runtime-other" if self.bad_trace and len(self.calls) == 2 else "runtime-a",
            slot_id=slot_id,
            prompt_tokens_total=100,
            prompt_tokens_processed=processed,
            prompt_tokens_reused=100 - processed,
            prompt_ms=float(processed),
            predicted_tokens=1,
            output_text="ORBIT-SECRET",
            output_token_ids=[31415],
        )

    def erase_slot(self, slot_id: int) -> RuntimeCacheOperationResult:
        self.calls.append(("erase", slot_id))
        self.erase_count += 1
        if self.final_erase_error and self.erase_count >= 3:
            raise RuntimeExecutionError("injected final erase failure")
        self.hot = False
        result_slot = slot_id + 1 if self.bad_first_erase and self.erase_count == 1 else slot_id
        return RuntimeCacheOperationResult(
            runtime_id="runtime-a",
            slot_id=result_slot,
            operation=RuntimeCacheOperation.ERASE,
            tokens_affected=100,
        )

    def save_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("save", slot_id, filename))
        (self.root / filename).write_bytes(b"x" * 4096)
        return RuntimeCacheOperationResult(
            runtime_id="runtime-a",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.SAVE,
            tokens_affected=100,
            bytes_affected=4096,
            checkpoint_name=filename,
        )

    def restore_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("restore", slot_id, filename))
        if self.restore_error:
            raise RuntimeExecutionError("injected restore failure")
        assert (self.root / filename).exists()
        self.hot = True
        return RuntimeCacheOperationResult(
            runtime_id="runtime-a",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.RESTORE,
            tokens_affected=100,
            bytes_affected=4096,
            checkpoint_name=filename,
        )

    def delete_checkpoint(self, filename: str) -> RuntimeCheckpointDeleteResult:
        self.calls.append(("delete", filename))
        if self.delete_error:
            raise RuntimeExecutionError("injected delete failure")
        path = self.root / filename
        existed = path.exists()
        size = path.stat().st_size if existed else 0
        if existed:
            path.unlink()
        return RuntimeCheckpointDeleteResult(
            runtime_id="runtime-a",
            checkpoint_name=filename,
            bytes_deleted=size,
            existed=existed,
        )


def test_runtime_probe_proves_native_reuse_and_purges_payloads(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    registry_root = tmp_path / "registry"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root)

    result = run_runtime_execution_probe(
        adapter=adapter,
        registry=FileExecutionCheckpointRegistry(registry_root),
        identity=_identity(),
        prompt="PRIVATE PREFIX\nQuestion: marker?",
        slot_id=2,
        checkpoint_id="probe-a",
    )

    assert result["pass"] is True
    assert result["output_token_equivalent"] is True
    assert result["cold"]["prompt_tokens_processed"] == 100
    assert result["hot"]["prompt_tokens_reused"] == 96
    assert result["restored_run"]["prompt_tokens_reused"] == 96
    assert result["purge"]["registry_removed"] is True
    assert not list(checkpoint_root.iterdir())
    assert not list(registry_root.glob("*.json"))
    payload = json.dumps(result, sort_keys=True)
    for forbidden in (
        "PRIVATE PREFIX",
        "ORBIT-SECRET",
        "31415",
        "tenant-secret",
        "session-secret",
        "private_runtime_note",
        ".bin",
    ):
        assert forbidden not in payload
    assert [call[0] for call in adapter.calls] == [
        "erase",
        "complete",
        "complete",
        "save",
        "erase",
        "restore",
        "complete",
        "delete",
        "erase",
    ]


def test_runtime_probe_failure_purges_checkpoint_and_erases_slot(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    registry_root = tmp_path / "registry"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root, restore_error=True)

    with pytest.raises(RuntimeExecutionError, match="injected restore failure"):
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(registry_root),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-failure",
        )

    assert not list(checkpoint_root.iterdir())
    assert not list(registry_root.glob("*.json"))
    assert [call[0] for call in adapter.calls][-2:] == ["delete", "erase"]


def test_runtime_probe_cleanup_failure_after_success_is_fatal(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root, final_erase_error=True)

    with pytest.raises(RuntimeExecutionError, match="runtime probe cleanup failed") as caught:
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(tmp_path / "registry"),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-cleanup-failure",
        )

    assert isinstance(caught.value.__cause__, RuntimeExecutionError)
    assert "final erase failure" in str(caught.value.__cause__)
    cleanup_errors = getattr(caught.value, "cleanup_errors")
    assert len(cleanup_errors) == 1


def test_runtime_probe_preserves_primary_and_all_cleanup_failures(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(
        checkpoint_root,
        restore_error=True,
        delete_error=True,
        final_erase_error=True,
    )

    with pytest.raises(RuntimeExecutionError, match="runtime probe failed and cleanup failed") as caught:
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(tmp_path / "registry"),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-primary-and-cleanup-failure",
        )

    assert isinstance(caught.value.__cause__, RuntimeExecutionError)
    assert "restore failure" in str(caught.value.__cause__)
    cleanup_errors = getattr(caught.value, "cleanup_errors")
    assert len(cleanup_errors) == 2
    assert "delete failure" in str(cleanup_errors[0])
    assert "final erase failure" in str(cleanup_errors[1])


def test_runtime_probe_preflight_fails_before_slot_mutation(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root)
    adapter._caps = replace(adapter._caps, supports_kv_checkpoint_delete=False)

    with pytest.raises(RuntimeExecutionError, match="checkpoint deletion"):
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(tmp_path / "registry"),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-unsupported",
        )

    assert adapter.calls == []


def test_runtime_probe_rejects_trace_identity_drift_before_checkpoint_save(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root, bad_trace=True)

    with pytest.raises(RuntimeExecutionError, match="trace identity mismatch"):
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(tmp_path / "registry"),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-bad-trace",
        )

    assert [call[0] for call in adapter.calls] == ["erase", "complete", "erase"]
    assert not list(checkpoint_root.iterdir())


def test_runtime_probe_rejects_erase_identity_drift_before_completion(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    adapter = FakeProbeAdapter(checkpoint_root, bad_first_erase=True)

    with pytest.raises(RuntimeExecutionError, match="erase result identity mismatch"):
        run_runtime_execution_probe(
            adapter=adapter,
            registry=FileExecutionCheckpointRegistry(tmp_path / "registry"),
            identity=_identity(),
            prompt="private prompt",
            slot_id=0,
            checkpoint_id="probe-bad-erase",
        )

    assert [call[0] for call in adapter.calls] == ["erase", "erase"]
    assert not list(checkpoint_root.iterdir())
