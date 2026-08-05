from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.adapters.memory import DML2ExactPageFaultAdapter
from daystrom_dml.context.admission import admit_context_segments
from daystrom_dml.context.checkpoints import (
    CheckpointRestoreResult,
    ExecutionCheckpointController,
    ExecutionCheckpointIdentity,
    FileExecutionCheckpointRegistry,
)
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)
from daystrom_dml.context.faults import MemoryFaultResolver
from daystrom_dml.context.paging import MemoryContextPageCache
from daystrom_dml.context.probe import ModelClientResponse, ProbeSettings, endpoint_origin_identity_digest
from daystrom_dml.context.recovery import (
    AutonomousFaultRetryRunner,
    CheckpointRecoveryPlan,
    RecoveryStatus,
)
from daystrom_dml.context.schema import ContextAuthority, ContextSegment


ENDPOINT = "http://127.0.0.1:18082/v1/chat/completions"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _scope(**overrides: str) -> DaystromScope:
    values = {
        "tenant_id": "tenant-a",
        "client_id": "client-a",
        "session_id": "session-a",
        "instance_id": "instance-a",
        "thread_id": "thread-a",
        "project_id": "project-a",
        "relationship_id": "relationship-a",
    }
    values.update(overrides)
    return DaystromScope(**values)


def _packet_and_resolver(*, scope: DaystromScope | None = None) -> tuple[Any, MemoryFaultResolver]:
    scope = scope or _scope()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=4096)

    def page_out(item_scope: DaystromScope, segment: ContextSegment) -> dict[str, Any]:
        page = cache.put_text(
            item_scope,
            str(segment.content),
            ttl_seconds=60,
            source_segment_id=segment.segment_id,
        )
        return {"page_id": page.page_id, "digest": page.content_digest}

    packet = admit_context_segments(
        scope=scope,
        segments=[
            ContextSegment(
                segment_id="policy",
                kind="policy",
                content="Answer only from supplied evidence.",
                authority=ContextAuthority.IMMUTABLE,
                scope=scope,
                estimated_tokens=4,
            ),
            ContextSegment(
                segment_id="question",
                kind="question",
                content="What is the launch code?",
                authority=ContextAuthority.CURRENT_INSTRUCTION,
                scope=scope,
                estimated_tokens=4,
            ),
            ContextSegment(
                segment_id="distractor",
                kind="reference",
                content="Low-value context that occupies the normal working set.",
                authority=ContextAuthority.REFERENCE,
                scope=scope,
                estimated_tokens=12,
            ),
            ContextSegment(
                segment_id="fact",
                kind="fact",
                content="The launch code is ORBIT-9.",
                authority=ContextAuthority.UNTRUSTED_DATA,
                scope=scope,
                estimated_tokens=9,
            ),
        ],
        model_id="llama3:8b",
        runtime_id="llama-local",
        endpoint_url=ENDPOINT,
        model_limit_tokens=28,
        page_out=page_out,
    )
    return packet, MemoryFaultResolver(dml2_exact=DML2ExactPageFaultAdapter(cache))


@dataclass
class InitialClient:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        endpoint_url: str,
        model_id: str,
        messages: list[dict[str, Any]],
        settings: ProbeSettings,
        label: str = "",
    ) -> ModelClientResponse:
        self.calls.append({"endpoint": endpoint_url, "model": model_id, "messages": messages, "label": label})
        return ModelClientResponse(content="UNKNOWN", latency_ms=1.0)


class CombinedRuntime:
    def __init__(
        self,
        *,
        fail_restore: bool = False,
        fail_complete: bool = False,
        endpoint_url: str = ENDPOINT,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail_restore = fail_restore
        self.fail_complete = fail_complete
        self.endpoint_url = endpoint_url

    def capabilities(self) -> RuntimeExecutionCapabilities:
        return RuntimeExecutionCapabilities(
            runtime_id="llama-local",
            adapter_id="llama_cpp_server",
            runtime_version="10250",
            supports_kv_checkpoint=True,
            supports_kv_restore=True,
            supports_kv_checkpoint_delete=True,
            supports_slot_affinity=True,
            metadata={"endpoint_origin_digest": "sha256:" + endpoint_origin_identity_digest(self.endpoint_url)},
        )

    def save_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("save", slot_id, filename))
        return RuntimeCacheOperationResult(
            runtime_id="llama-local",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.SAVE,
            tokens_affected=128,
            bytes_affected=4096,
            checkpoint_name=filename,
        )

    def restore_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        self.calls.append(("restore", slot_id, filename))
        if self.fail_restore:
            raise RuntimeExecutionError("private runtime failure")
        return RuntimeCacheOperationResult(
            runtime_id="llama-local",
            slot_id=slot_id,
            operation=RuntimeCacheOperation.RESTORE,
            tokens_affected=128,
            bytes_affected=4096,
            checkpoint_name=filename,
        )

    def delete_checkpoint(self, filename: str) -> RuntimeCheckpointDeleteResult:
        self.calls.append(("delete", filename))
        return RuntimeCheckpointDeleteResult(
            runtime_id="llama-local",
            checkpoint_name=filename,
            bytes_deleted=4096,
            existed=True,
        )

    def complete_on_slot(
        self,
        endpoint_url: str,
        model_id: str,
        messages: list[dict[str, Any]],
        settings: ProbeSettings,
        *,
        slot_id: int,
        label: str = "",
    ) -> ModelClientResponse:
        self.calls.append(("complete_on_slot", slot_id, endpoint_url, model_id, label, messages))
        if self.fail_complete:
            raise RuntimeExecutionError("private continuation failure")
        return ModelClientResponse(
            content="ORBIT-9",
            latency_ms=2.0,
            usage={"prompt_tokens_reused": 128},
        )


def _identity(packet: Any, runtime: CombinedRuntime, **overrides: Any) -> ExecutionCheckpointIdentity:
    values: dict[str, Any] = {
        "packet": packet,
        "runtime_capabilities": runtime.capabilities(),
        "model_digest": _digest("model"),
        "tokenizer_digest": _digest("tokenizer"),
        "positional_config": {"n_ctx": 4096, "rope": "default"},
        "immutable_prefix": "exact templated immutable prefix",
        "runtime_endpoint_url": runtime.endpoint_url,
    }
    values.update(overrides)
    return ExecutionCheckpointIdentity.from_packet(**values)


def _runner(
    tmp_path: Path,
    *,
    packet: Any,
    resolver: MemoryFaultResolver,
    runtime: CombinedRuntime,
    initial: InitialClient,
    checkpoint_ids: tuple[str, ...] = ("checkpoint-a",),
    identity: ExecutionCheckpointIdentity | None = None,
    max_selection_records: int = 1024,
) -> AutonomousFaultRetryRunner:
    registry = FileExecutionCheckpointRegistry(tmp_path, clock=lambda: 100.0)
    controller = ExecutionCheckpointController(adapter=runtime, registry=registry, clock=lambda: 100.0)
    bound = identity or _identity(packet, runtime)
    for checkpoint_id in checkpoint_ids:
        controller.save(checkpoint_id, bound, slot_id=0, ttl_seconds=60)
    return AutonomousFaultRetryRunner(
        resolver=resolver,
        client=initial,
        endpoint_url=ENDPOINT,
        model_id="llama3:8b",
        runtime_id="llama-local",
        settings=ProbeSettings(max_output_tokens=16),
        checkpoint_plan=CheckpointRecoveryPlan(
            controller=controller,
            identity=bound,
            slot_id=3,
            runtime=runtime,
            max_selection_records=max_selection_records,
        ),
    )


def test_autonomous_recovery_selects_restores_and_continues_on_bound_slot(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    initial = InitialClient()

    result = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
    ).run(packet)

    assert result.status is RecoveryStatus.RECOVERED
    assert result.response.content == "ORBIT-9"
    assert result.reason_code == "bounded_checkpoint_page_in_retry_completed"
    assert isinstance(result.checkpoint_restore, CheckpointRestoreResult)
    assert result.checkpoint_restore.tokens_restored == 128
    assert [call[0] for call in runtime.calls] == ["save", "restore", "complete_on_slot"]
    assert runtime.calls[-1][1] == 3
    assert "ORBIT-9" in str(runtime.calls[-1][-1])
    telemetry = result.to_telemetry()
    assert telemetry["checkpoint_restore"]["binding_digest"] == _identity(packet, runtime).binding_digest
    assert "checkpoint-a" not in str(telemetry)
    assert "exact templated immutable prefix" not in str(telemetry)


def test_ambiguous_authorized_checkpoints_fail_closed_without_restore_or_retry(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    initial = InitialClient()
    runner = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
        checkpoint_ids=("checkpoint-a", "checkpoint-b"),
    )

    result = runner.run(packet)

    assert result.status is RecoveryStatus.FAULT_UNRESOLVED
    assert result.reason_code == "checkpoint_selection_ambiguous"
    assert [call[0] for call in runtime.calls] == ["save", "save"]
    assert len(initial.calls) == 1


def test_checkpoint_selection_scan_is_bounded_before_restore(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    initial = InitialClient()
    runner = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
        checkpoint_ids=("checkpoint-a", "checkpoint-b"),
        max_selection_records=1,
    )

    result = runner.run(packet)

    assert result.status is RecoveryStatus.FAULT_UNRESOLVED
    assert result.reason_code == "checkpoint_selection_scan_limit"
    assert [call[0] for call in runtime.calls] == ["save", "save"]


def test_missing_authorized_checkpoint_fails_closed_without_runtime_mutation(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    initial = InitialClient()
    runner = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
        checkpoint_ids=(),
    )

    result = runner.run(packet)

    assert result.status is RecoveryStatus.FAULT_UNRESOLVED
    assert result.reason_code == "checkpoint_selection_not_found"
    assert runtime.calls == []
    assert len(initial.calls) == 1


def test_checkpoint_restore_failure_never_falls_back_to_unbound_text_retry(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime(fail_restore=True)
    initial = InitialClient()

    result = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
    ).run(packet)

    assert result.status is RecoveryStatus.FAULT_UNRESOLVED
    assert result.reason_code == "checkpoint_restore_failed"
    assert [call[0] for call in runtime.calls] == ["save", "restore"]
    assert "private runtime failure" not in str(result.to_telemetry())


def test_checkpoint_continuation_failure_reports_restore_without_leaking_error(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime(fail_complete=True)
    initial = InitialClient()

    result = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
    ).run(packet)

    assert result.status is RecoveryStatus.MODEL_ERROR
    assert result.reason_code == "checkpoint_retry_model_error"
    assert result.checkpoint_restore is not None
    assert [call[0] for call in runtime.calls] == ["save", "restore", "complete_on_slot"]
    assert "private continuation failure" not in str(result.to_telemetry())


def test_checkpoint_identity_scope_drift_is_rejected_before_initial_model_call(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    initial = InitialClient()
    other_packet, _ = _packet_and_resolver(scope=_scope(tenant_id="tenant-b"))
    identity = _identity(other_packet, runtime)
    runner = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
        identity=identity,
    )

    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.run(packet)
    assert initial.calls == []
    assert [call[0] for call in runtime.calls] == ["save"]


def test_checkpoint_runtime_and_continuation_client_must_be_same_object(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime()
    other_runtime = CombinedRuntime()
    identity = _identity(packet, runtime)
    controller = ExecutionCheckpointController(
        adapter=runtime,
        registry=FileExecutionCheckpointRegistry(tmp_path),
    )

    with pytest.raises(ValueError, match="same runtime"):
        CheckpointRecoveryPlan(
            controller=controller,
            identity=identity,
            slot_id=0,
            runtime=other_runtime,
        )


def test_checkpoint_endpoint_origin_drift_is_rejected_before_initial_call(tmp_path: Path) -> None:
    packet, resolver = _packet_and_resolver()
    runtime = CombinedRuntime(endpoint_url="http://127.0.0.1:19999/v1/chat/completions")
    initial = InitialClient()
    identity = _identity(packet, runtime)
    runner = _runner(
        tmp_path,
        packet=packet,
        resolver=resolver,
        runtime=runtime,
        initial=initial,
        identity=identity,
    )

    with pytest.raises(ValueError, match="endpoint"):
        runner.run(packet)
    assert initial.calls == []
