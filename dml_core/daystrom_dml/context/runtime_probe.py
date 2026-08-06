"""Fail-closed runtime-native KV execution-state probe orchestration."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

from daystrom_dml.context.checkpoints import (
    ExecutionCheckpointController,
    ExecutionCheckpointIdentity,
    FileExecutionCheckpointRegistry,
)
from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCompletionTrace,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)

RUNTIME_EXECUTION_PROBE_V2 = "dcm-runtime-execution-probe-v2"


class RuntimeProbeAdapter(Protocol):
    """Narrow runtime surface required by the execution-state probe."""

    def capabilities(self) -> RuntimeExecutionCapabilities: ...

    def complete(
        self,
        prompt: str,
        *,
        slot_id: int,
        n_predict: int = 1,
        cache_prompt: bool = True,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> RuntimeCompletionTrace: ...

    def erase_slot(self, slot_id: int) -> Any: ...

    def save_slot(self, slot_id: int, filename: str) -> Any: ...

    def restore_slot(self, slot_id: int, filename: str) -> Any: ...

    def delete_checkpoint(self, filename: str) -> Any: ...


def run_runtime_execution_probe(
    *,
    adapter: RuntimeProbeAdapter,
    registry: FileExecutionCheckpointRegistry,
    identity: ExecutionCheckpointIdentity,
    prompt: str,
    slot_id: int,
    checkpoint_id: str,
    ttl_seconds: float = 300.0,
    n_predict: int = 1,
    seed: int = 7,
) -> dict[str, Any]:
    """Prove cold/hot/save/erase/restore/reuse/purge with payload-free telemetry."""

    if not isinstance(prompt, str) or not prompt:
        raise RuntimeExecutionError("runtime probe prompt must be non-empty")
    if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
        raise RuntimeExecutionError("runtime probe slot_id must be a non-negative integer")
    if not isinstance(n_predict, int) or isinstance(n_predict, bool) or n_predict <= 0:
        raise RuntimeExecutionError("runtime probe n_predict must be a positive integer")

    caps = _preflight(adapter, identity)
    controller = ExecutionCheckpointController(adapter=adapter, registry=registry)
    saved = False
    purged = False
    primary_error: Exception | None = None

    try:
        _validate_erase(adapter.erase_slot(slot_id), caps, slot_id)
        cold = adapter.complete(prompt, slot_id=slot_id, n_predict=n_predict, seed=seed)
        _validate_trace(cold, caps, slot_id)
        hot = adapter.complete(prompt, slot_id=slot_id, n_predict=n_predict, seed=seed)
        _validate_trace(hot, caps, slot_id)
        record = controller.save(checkpoint_id, identity, slot_id=slot_id, ttl_seconds=ttl_seconds)
        saved = True
        _validate_erase(adapter.erase_slot(slot_id), caps, slot_id)
        restored = controller.restore(checkpoint_id, identity, slot_id=slot_id)
        restored_run = adapter.complete(prompt, slot_id=slot_id, n_predict=n_predict, seed=seed)
        _validate_trace(restored_run, caps, slot_id)
        purge = controller.purge(checkpoint_id, identity)
        purged = True

        equivalent = (
            bool(cold.output_token_ids)
            and cold.output_token_ids == hot.output_token_ids == restored_run.output_token_ids
        )
        registry_removed = _registry_record_removed(registry, checkpoint_id, identity)
        passed = all(
            (
                cold.prompt_tokens_processed > hot.prompt_tokens_processed,
                cold.prompt_tokens_processed > restored_run.prompt_tokens_processed,
                hot.prompt_tokens_reused > 0,
                restored_run.prompt_tokens_reused > 0,
                restored.tokens_affected == record.tokens_saved,
                restored.bytes_affected == record.bytes_saved,
                purge.existed,
                purge.bytes_deleted == record.bytes_saved,
                registry_removed,
                equivalent,
            )
        )
        return {
            "artifact_version": RUNTIME_EXECUTION_PROBE_V2,
            "runtime": _capability_telemetry(caps),
            "binding_digest": identity.binding_digest,
            "prefix_digest": identity.immutable_prefix_digest,
            "cold": _trace_telemetry(cold),
            "hot": _trace_telemetry(hot),
            "checkpoint": record.to_telemetry(),
            "restore": {
                "binding_digest": identity.binding_digest,
                "tokens_restored": restored.tokens_affected,
                "bytes_restored": restored.bytes_affected,
                "elapsed_ms": restored.elapsed_ms,
                "slot_id": restored.slot_id,
            },
            "restored_run": _trace_telemetry(restored_run),
            "purge": {
                "binding_digest": purge.binding_digest,
                "checkpoint_id_digest": purge.checkpoint_id_digest,
                "checkpoint_name_digest": purge.checkpoint_name_digest,
                "bytes_deleted": purge.bytes_deleted,
                "existed": purge.existed,
                "elapsed_ms": purge.elapsed_ms,
                "registry_removed": registry_removed,
            },
            "output_token_equivalent": equivalent,
            "pass": passed,
        }
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[Exception] = []
        if saved and not purged:
            try:
                controller.purge(checkpoint_id, identity)
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            _validate_erase(adapter.erase_slot(slot_id), caps, slot_id)
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is not None:
                failure = RuntimeExecutionError("runtime probe failed and cleanup failed")
                setattr(failure, "cleanup_errors", tuple(cleanup_errors))
                raise failure from primary_error
            failure = RuntimeExecutionError("runtime probe cleanup failed")
            setattr(failure, "cleanup_errors", tuple(cleanup_errors))
            raise failure from cleanup_errors[0]


def _preflight(adapter: RuntimeProbeAdapter, identity: ExecutionCheckpointIdentity) -> RuntimeExecutionCapabilities:
    if not isinstance(identity, ExecutionCheckpointIdentity):
        raise RuntimeExecutionError("runtime probe identity is required")
    identity.to_dict()
    caps = adapter.capabilities()
    if not isinstance(caps, RuntimeExecutionCapabilities):
        raise RuntimeExecutionError("runtime probe capabilities are invalid")
    required = {
        "prompt cache": caps.supports_prompt_cache,
        "checkpoint save": caps.supports_kv_checkpoint,
        "checkpoint restore": caps.supports_kv_restore,
        "slot erase": caps.supports_kv_erase,
        "checkpoint deletion": caps.supports_kv_checkpoint_delete,
        "slot affinity": caps.supports_slot_affinity,
        "runtime metrics": caps.supports_metrics,
    }
    missing = [name for name, supported in required.items() if not supported]
    if missing:
        raise RuntimeExecutionError("runtime probe requires: " + ", ".join(missing))
    if caps.runtime_version.casefold() == "unknown":
        raise RuntimeExecutionError("runtime probe requires an exact runtime version")
    if (
        caps.runtime_id != identity.runtime_id
        or caps.runtime_version != identity.runtime_version
        or caps.adapter_id != identity.adapter_id
    ):
        raise RuntimeExecutionError("runtime probe identity does not match adapter capabilities")
    expected_endpoint = caps.metadata.get("endpoint_origin_digest")
    if expected_endpoint != identity.runtime_endpoint_digest:
        raise RuntimeExecutionError("runtime probe endpoint identity mismatch")
    return caps


def _trace_telemetry(trace: RuntimeCompletionTrace) -> dict[str, Any]:
    if not isinstance(trace, RuntimeCompletionTrace):
        raise RuntimeExecutionError("runtime probe completion trace is invalid")
    token_payload = ",".join(str(token) for token in trace.output_token_ids)
    return {
        "runtime_id": trace.runtime_id,
        "slot_id": trace.slot_id,
        "prompt_tokens_total": trace.prompt_tokens_total,
        "prompt_tokens_processed": trace.prompt_tokens_processed,
        "prompt_tokens_reused": trace.prompt_tokens_reused,
        "prompt_ms": trace.prompt_ms,
        "predicted_tokens": trace.predicted_tokens,
        "output_digest": hashlib.sha256(trace.output_text.encode("utf-8")).hexdigest(),
        "output_token_ids_digest": hashlib.sha256(token_payload.encode("ascii")).hexdigest(),
        "output_token_count": len(trace.output_token_ids),
        "truncated": trace.truncated,
    }


def _validate_trace(
    trace: RuntimeCompletionTrace,
    caps: RuntimeExecutionCapabilities,
    slot_id: int,
) -> None:
    if not isinstance(trace, RuntimeCompletionTrace):
        raise RuntimeExecutionError("runtime probe completion trace is invalid")
    if trace.runtime_id != caps.runtime_id or trace.slot_id != slot_id:
        raise RuntimeExecutionError("runtime probe completion trace identity mismatch")
    counters = (
        trace.prompt_tokens_total,
        trace.prompt_tokens_processed,
        trace.prompt_tokens_reused,
        trace.predicted_tokens,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters):
        raise RuntimeExecutionError("runtime probe completion counters are invalid")
    if trace.prompt_tokens_processed + trace.prompt_tokens_reused != trace.prompt_tokens_total:
        raise RuntimeExecutionError("runtime probe completion counters are inconsistent")
    if not isinstance(trace.prompt_ms, (int, float)) or isinstance(trace.prompt_ms, bool):
        raise RuntimeExecutionError("runtime probe prompt timing is invalid")
    if not math.isfinite(float(trace.prompt_ms)) or trace.prompt_ms < 0:
        raise RuntimeExecutionError("runtime probe prompt timing is invalid")
    if not isinstance(trace.output_text, str):
        raise RuntimeExecutionError("runtime probe output text is invalid")
    if not isinstance(trace.output_token_ids, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in trace.output_token_ids
    ):
        raise RuntimeExecutionError("runtime probe output token IDs are invalid")


def _validate_erase(
    result: Any,
    caps: RuntimeExecutionCapabilities,
    slot_id: int,
) -> None:
    if not isinstance(result, RuntimeCacheOperationResult):
        raise RuntimeExecutionError("runtime probe erase result is invalid")
    if (
        result.runtime_id != caps.runtime_id
        or result.slot_id != slot_id
        or result.operation is not RuntimeCacheOperation.ERASE
    ):
        raise RuntimeExecutionError("runtime probe erase result identity mismatch")
    if (
        not isinstance(result.tokens_affected, int)
        or isinstance(result.tokens_affected, bool)
        or result.tokens_affected < 0
    ):
        raise RuntimeExecutionError("runtime probe erase token count is invalid")


def _capability_telemetry(caps: RuntimeExecutionCapabilities) -> dict[str, Any]:
    return {
        "runtime_id": caps.runtime_id,
        "adapter_id": caps.adapter_id,
        "runtime_version": caps.runtime_version,
        "supports_prompt_cache": caps.supports_prompt_cache,
        "supports_kv_checkpoint": caps.supports_kv_checkpoint,
        "supports_kv_restore": caps.supports_kv_restore,
        "supports_kv_erase": caps.supports_kv_erase,
        "supports_kv_checkpoint_delete": caps.supports_kv_checkpoint_delete,
        "supports_slot_affinity": caps.supports_slot_affinity,
        "supports_metrics": caps.supports_metrics,
        "endpoint_origin_digest": caps.metadata.get("endpoint_origin_digest"),
    }


def _registry_record_removed(
    registry: FileExecutionCheckpointRegistry,
    checkpoint_id: str,
    identity: ExecutionCheckpointIdentity,
) -> bool:
    try:
        registry.require(checkpoint_id, identity)
    except RuntimeExecutionError as exc:
        return str(exc) == "checkpoint record not found"
    return False
