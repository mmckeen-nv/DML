from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.checkpoints import (
    ExecutionCheckpointController,
    ExecutionCheckpointIdentity,
    ExecutionCheckpointRecord,
    FileExecutionCheckpointRegistry,
)
from daystrom_dml.context.admission import admit_context_segments
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)
from daystrom_dml.context.schema import ContextAuthority, ContextPriority, ContextSegment


def _digest(label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _identity(**overrides: Any) -> ExecutionCheckpointIdentity:
    values: dict[str, Any] = {
        "scope": DaystromScope(
            tenant_id="tenant-a",
            client_id="client-a",
            session_id="session-a",
            instance_id="instance-a",
            thread_id="thread-a",
            project_id="project-a",
            relationship_id="relationship-a",
        ),
        "model_id": "llama3:8b",
        "model_digest": _digest("model"),
        "tokenizer_digest": _digest("tokenizer"),
        "positional_config_digest": _digest("rope-and-context"),
        "immutable_prefix_digest": _digest("private immutable prefix"),
        "packet_digest": _digest("packet"),
        "manifest_digest": _digest("manifest"),
        "runtime_id": "llama-local",
        "runtime_version": "10250",
        "adapter_id": "llama_cpp_server",
    }
    values.update(overrides)
    return ExecutionCheckpointIdentity(**values)


def _record(**overrides: Any) -> ExecutionCheckpointRecord:
    values: dict[str, Any] = {
        "checkpoint_id": "checkpoint-a",
        "checkpoint_name": "dcm-kv-deadbeef.bin",
        "identity": _identity(),
        "tokens_saved": 128,
        "bytes_saved": 4096,
        "created_at": 100.0,
        "expires_at": 200.0,
    }
    values.update(overrides)
    return ExecutionCheckpointRecord(**values)


class FakeExecutionAdapter:
    def __init__(self, *, capabilities: RuntimeExecutionCapabilities | None = None) -> None:
        self._capabilities = capabilities or RuntimeExecutionCapabilities(
            runtime_id="llama-local",
            adapter_id="llama_cpp_server",
            runtime_version="10250",
            supports_prompt_cache=True,
            supports_kv_checkpoint=True,
            supports_kv_restore=True,
            supports_kv_erase=True,
            supports_kv_checkpoint_delete=True,
            supports_slot_affinity=True,
            supports_metrics=True,
        )
        self.calls: list[tuple[Any, ...]] = []
        self.save_result_override: RuntimeCacheOperationResult | None = None
        self.delete_error: Exception | None = None

    def capabilities(self) -> RuntimeExecutionCapabilities:
        return self._capabilities

    def save_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("save", slot_id, filename))
        if self.save_result_override is not None:
            return self.save_result_override
        return RuntimeCacheOperationResult(
            runtime_id="llama-local",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.SAVE,
            tokens_affected=128,
            bytes_affected=4096,
            elapsed_ms=3.0,
            checkpoint_name=filename,
        )

    def restore_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("restore", slot_id, filename))
        return RuntimeCacheOperationResult(
            runtime_id="llama-local",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.RESTORE,
            tokens_affected=128,
            bytes_affected=4096,
            elapsed_ms=2.0,
            checkpoint_name=filename,
        )
    def delete_checkpoint(self, filename: str) -> RuntimeCheckpointDeleteResult:
        self.calls.append(("delete", filename))
        if self.delete_error is not None:
            raise self.delete_error
        return RuntimeCheckpointDeleteResult(
            runtime_id="llama-local",
            checkpoint_name=filename,
            bytes_deleted=4096,
            existed=True,
            elapsed_ms=1.0,
        )


def test_registry_persists_digest_bound_metadata_without_context_payload(tmp_path: Path) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    record = _record()

    registry.put(record)
    restored = registry.require("checkpoint-a", record.identity)

    assert restored == record
    raw = (tmp_path / "checkpoint-a.json").read_text()
    assert "private immutable prefix" not in raw
    assert record.identity.immutable_prefix_digest in raw
    assert json.loads(raw)["record_digest"].startswith("sha256:")
    if os.name != "nt":
        assert (tmp_path / "checkpoint-a.json").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "mutated",
    [
        _identity(scope=DaystromScope(tenant_id="tenant-b", session_id="session-a")),
        _identity(scope=DaystromScope(tenant_id="tenant-a", session_id="session-b")),
        _identity(model_digest=_digest("other-model")),
        _identity(tokenizer_digest=_digest("other-tokenizer")),
        _identity(positional_config_digest=_digest("other-rope")),
        _identity(immutable_prefix_digest=_digest("other-prefix")),
        _identity(packet_digest=_digest("other-packet")),
        _identity(manifest_digest=_digest("other-manifest")),
        _identity(runtime_version="10251"),
        _identity(adapter_id="other-adapter"),
    ],
)
def test_registry_rejects_every_identity_drift_dimension_before_restore(
    tmp_path: Path, mutated: ExecutionCheckpointIdentity
) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry, clock=lambda: 100.0)
    registry.put(_record())

    with pytest.raises(RuntimeExecutionError, match="identity mismatch"):
        controller.restore("checkpoint-a", mutated, slot_id=1)

    assert adapter.calls == []


def test_registry_rejects_tampered_and_expired_records(tmp_path: Path) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    registry.put(_record())
    path = tmp_path / "checkpoint-a.json"
    payload = json.loads(path.read_text())
    payload["tokens_saved"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeExecutionError, match="integrity"):
        registry.require("checkpoint-a", _identity())

    expired_root = tmp_path / "expired"
    writer = FileExecutionCheckpointRegistry(expired_root, clock=lambda: 150.0)
    writer.put(_record())
    expired_registry = FileExecutionCheckpointRegistry(expired_root, clock=lambda: 201.0)
    with pytest.raises(RuntimeExecutionError, match="expired"):
        expired_registry.require("checkpoint-a", _identity())


def test_controller_saves_then_restores_only_exact_registered_checkpoint(tmp_path: Path) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry, clock=lambda: 100.0)

    saved = controller.save("checkpoint-a", _identity(), slot_id=3, ttl_seconds=60)
    restored = controller.restore("checkpoint-a", _identity(), slot_id=7)

    assert saved.tokens_saved == 128 and saved.expires_at == 160.0
    assert restored.operation is RuntimeCacheOperation.RESTORE
    assert adapter.calls == [
        ("save", 3, saved.checkpoint_name),
        ("restore", 7, saved.checkpoint_name),
    ]
    telemetry = saved.to_telemetry()
    assert telemetry["binding_digest"] == saved.identity.binding_digest
    assert "tenant-a" not in json.dumps(telemetry)
    assert "session-a" not in json.dumps(telemetry)


def test_duplicate_checkpoint_is_rejected_before_second_runtime_save(tmp_path: Path) -> None:
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(
        adapter=adapter,
        registry=FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 100.0),
        clock=lambda: 100.0,
    )
    controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)

    with pytest.raises(RuntimeExecutionError, match="already exists"):
        controller.save("checkpoint-a", _identity(), slot_id=1, ttl_seconds=60)

    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == "save"


def test_registry_write_failure_compensates_by_deleting_saved_checkpoint(tmp_path: Path) -> None:
    class FailingRegistry(FileExecutionCheckpointRegistry):
        def create(self, checkpoint_id: str, build_record: Any) -> Any:
            build_record()
            raise OSError("disk full")

    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(
        adapter=adapter,
        registry=FailingRegistry(tmp_path),
        clock=lambda: 100.0,
    )

    with pytest.raises(OSError, match="disk full"):
        controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)

    assert [call[0] for call in adapter.calls] == ["save", "delete"]


def test_controller_fails_closed_when_capability_or_runtime_result_is_invalid(tmp_path: Path) -> None:
    unsupported = RuntimeExecutionCapabilities(
        runtime_id="llama-local",
        adapter_id="llama_cpp_server",
        runtime_version="10250",
    )
    unsupported_adapter = FakeExecutionAdapter(capabilities=unsupported)
    controller = ExecutionCheckpointController(
        adapter=unsupported_adapter,
        registry=FileExecutionCheckpointRegistry(tmp_path / "unsupported"),
    )
    with pytest.raises(RuntimeExecutionError, match="checkpoint save"):
        controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)
    assert unsupported_adapter.calls == []

    malformed_adapter = FakeExecutionAdapter()
    malformed_adapter.save_result_override = RuntimeCacheOperationResult(
        runtime_id="other-runtime",
        slot_id=0,
        operation=RuntimeCacheOperation.SAVE,
        tokens_affected=128,
        bytes_affected=4096,
        checkpoint_name="wrong.bin",
    )
    malformed_registry = FileExecutionCheckpointRegistry(tmp_path / "malformed")
    malformed_controller = ExecutionCheckpointController(adapter=malformed_adapter, registry=malformed_registry)
    with pytest.raises(RuntimeExecutionError, match="save result"):
        malformed_controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)
    with pytest.raises(RuntimeExecutionError, match="not found"):
        malformed_registry.require("checkpoint-a", _identity())


def test_controller_purge_deletes_runtime_bytes_before_registry_metadata(tmp_path: Path) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry, clock=lambda: 100.0)
    record = controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)

    deleted = controller.purge("checkpoint-a", _identity())

    assert deleted.checkpoint_name_digest == _digest(record.checkpoint_name)
    assert adapter.calls[-1] == ("delete", record.checkpoint_name)
    with pytest.raises(RuntimeExecutionError, match="not found"):
        registry.require("checkpoint-a", _identity())


def test_failed_runtime_delete_retains_registry_authorization_record(tmp_path: Path) -> None:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry, clock=lambda: 100.0)
    controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)
    adapter.delete_error = RuntimeExecutionError("delete unavailable")

    with pytest.raises(RuntimeExecutionError, match="delete unavailable"):
        controller.purge("checkpoint-a", _identity())

    assert registry.require("checkpoint-a", _identity()).checkpoint_id == "checkpoint-a"


def test_expired_gc_physically_purges_checkpoint_and_metadata(tmp_path: Path) -> None:
    now = [100.0]
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: now[0])
    adapter = FakeExecutionAdapter()
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry, clock=lambda: now[0])
    controller.save("expired-a", _identity(), slot_id=0, ttl_seconds=5)
    controller.save("live-b", _identity(), slot_id=0, ttl_seconds=50)
    now[0] = 110.0

    results = controller.purge_expired(max_records=10)

    assert [item.checkpoint_id_digest for item in results] == [_digest("expired-a")]
    with pytest.raises(RuntimeExecutionError, match="not found"):
        registry.require("expired-a", _identity())
    assert registry.require("live-b", _identity()).checkpoint_id == "live-b"


def test_save_refuses_runtime_without_physical_delete_capability(tmp_path: Path) -> None:
    caps = RuntimeExecutionCapabilities(
        runtime_id="llama-local",
        adapter_id="llama_cpp_server",
        runtime_version="10250",
        supports_kv_checkpoint=True,
        supports_kv_restore=True,
    )
    adapter = FakeExecutionAdapter(capabilities=caps)
    controller = ExecutionCheckpointController(
        adapter=adapter,
        registry=FileExecutionCheckpointRegistry(tmp_path),
    )

    with pytest.raises(RuntimeExecutionError, match="checkpoint deletion"):
        controller.save("checkpoint-a", _identity(), slot_id=0, ttl_seconds=60)
    assert adapter.calls == []


def test_identity_is_derived_from_revalidated_packet_and_hashes_raw_runtime_inputs(tmp_path: Path) -> None:
    scope = _identity().scope
    packet = admit_context_segments(
        scope=scope,
        segments=[
            ContextSegment(
                segment_id="policy",
                kind="policy",
                content="Do not reveal private context.",
                authority=ContextAuthority.IMMUTABLE,
                priority=ContextPriority.CRITICAL,
                scope=scope,
                estimated_tokens=8,
            )
        ],
        model_id="llama3:8b",
        runtime_id="llama-local",
        model_limit_tokens=128,
    )
    caps = FakeExecutionAdapter().capabilities()
    raw_prefix = "private rendered prefix"
    raw_positional = {"n_ctx": 4096, "rope": "default"}

    identity = ExecutionCheckpointIdentity.from_packet(
        packet,
        caps,
        model_digest=_digest("model"),
        tokenizer_digest=_digest("tokenizer"),
        positional_config=raw_positional,
        immutable_prefix=raw_prefix,
    )
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 150.0)
    registry.put(_record(identity=identity))
    persisted = (tmp_path / "checkpoint-a.json").read_text()

    assert identity.packet_digest == "sha256:" + packet.packet_content_digest
    assert identity.manifest_digest == "sha256:" + packet.manifest.content_digest
    assert raw_prefix not in persisted
    assert "default" not in persisted

    packet.rendered_messages[0]["content"] = "tampered"
    with pytest.raises(RuntimeExecutionError, match="packet integrity"):
        ExecutionCheckpointIdentity.from_packet(
            packet,
            caps,
            model_digest=_digest("model"),
            tokenizer_digest=_digest("tokenizer"),
            positional_config=raw_positional,
            immutable_prefix=raw_prefix,
        )


def test_records_require_complete_strong_digests_and_session_scope() -> None:
    with pytest.raises(RuntimeExecutionError, match="session_id"):
        _identity(scope=DaystromScope(tenant_id="tenant-a"))
    with pytest.raises(RuntimeExecutionError, match="model_digest"):
        _identity(model_digest="model-latest")
    with pytest.raises(RuntimeExecutionError, match="checkpoint_id"):
        replace(_record(), checkpoint_id="../escape")
