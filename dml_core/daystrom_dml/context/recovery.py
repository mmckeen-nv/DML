"""Bounded autonomous memory-fault detection, page-in, and model retry.

This module deliberately recognizes only explicit, configured miss markers. It
selects exactly one page handle emitted by admission, resolves it through the
scoped memory-fault hierarchy, reinjects bounded evidence as untrusted user
data, and permits at most one retry by default.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.checkpoints import (
    CheckpointRestoreResult,
    CheckpointSelectionError,
    ExecutionCheckpointController,
    ExecutionCheckpointIdentity,
)
from daystrom_dml.context.execution import RuntimeExecutionError
from daystrom_dml.context.faults import (
    EvidenceHandle,
    MemoryFaultBudget,
    MemoryFaultReason,
    MemoryFaultRequest,
    MemoryFaultResolver,
    MemoryFaultStatus,
)
from daystrom_dml.context.manifest import ContextPacket
from daystrom_dml.context.probe import (
    ModelClient,
    ModelClientResponse,
    ProbeSettings,
    endpoint_identity_digest,
    endpoint_origin_identity_digest,
)
from daystrom_dml.context.schema import ContextAuthority, ContextSegment


class RecoveryStatus(str, Enum):
    COMPLETED = "completed"
    RECOVERED = "recovered"
    FAULT_UNRESOLVED = "fault_unresolved"
    RETRY_EXHAUSTED = "retry_exhausted"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class FaultRetryPolicy:
    """Hard bounds and explicit miss markers for one autonomous run."""

    miss_markers: tuple[str, ...] = ("UNKNOWN",)
    max_retries: int = 1
    max_evidence_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.max_retries < 0 or self.max_retries > 1:
            raise ValueError("max_retries must be zero or one")
        if self.max_evidence_bytes <= 0:
            raise ValueError("max_evidence_bytes must be positive")
        if not self.miss_markers or any(not marker.strip() for marker in self.miss_markers):
            raise ValueError("miss_markers must contain non-empty strings")

    def is_explicit_miss(self, content: str) -> bool:
        normalized = content.strip().casefold()
        return any(normalized == marker.strip().casefold() for marker in self.miss_markers)


@dataclass
class FaultRetryResult:
    """Runtime result. Persist only :meth:`to_telemetry`, which is payload-free."""

    status: RecoveryStatus
    response: ModelClientResponse
    retry_count: int = 0
    reason_code: str = "completed"
    attempted_tiers: List[str] = field(default_factory=list)
    evidence: List[EvidenceHandle] = field(default_factory=list)
    checkpoint_restore: Optional[CheckpointRestoreResult] = None

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "retry_count": self.retry_count,
            "reason_code": self.reason_code,
            "attempted_tiers": list(self.attempted_tiers),
            "output_digest": hashlib.sha256(self.response.content.encode("utf-8")).hexdigest(),
            "checkpoint_restore": (
                self.checkpoint_restore.to_telemetry() if self.checkpoint_restore is not None else None
            ),
            "evidence": [
                {
                    "handle_id": item.handle_id,
                    "tier": item.tier.value,
                    "digest": item.digest,
                    "authority": item.authority.value,
                    "size_bytes": item.size_bytes,
                }
                for item in self.evidence
            ],
        }


@dataclass(frozen=True)
class _PageCandidate:
    segment_id: str
    key: str
    digest: str


class SlotBoundRuntime(Protocol):
    def complete_on_slot(
        self,
        endpoint_url: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        settings: ProbeSettings,
        *,
        slot_id: int,
        label: str = "",
    ) -> ModelClientResponse: ...


@dataclass(frozen=True)
class CheckpointRecoveryPlan:
    """Explicitly bind selection, restore, and retry to one runtime object and slot."""

    controller: ExecutionCheckpointController
    identity: ExecutionCheckpointIdentity
    slot_id: int
    runtime: SlotBoundRuntime
    max_selection_records: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.controller, ExecutionCheckpointController):
            raise ValueError("checkpoint controller is required")
        if not isinstance(self.identity, ExecutionCheckpointIdentity):
            raise ValueError("checkpoint identity is required")
        if self.runtime is not self.controller.adapter:
            raise ValueError("checkpoint control and continuation must use the same runtime")
        if not isinstance(self.slot_id, int) or isinstance(self.slot_id, bool) or self.slot_id < 0:
            raise ValueError("checkpoint slot_id must be a non-negative integer")
        if (
            not isinstance(self.max_selection_records, int)
            or isinstance(self.max_selection_records, bool)
            or self.max_selection_records <= 0
        ):
            raise ValueError("max_selection_records must be a positive integer")
        if not callable(getattr(self.runtime, "complete_on_slot", None)):
            raise ValueError("checkpoint runtime must support slot-bound completion")


class AutonomousFaultRetryRunner:
    """Provider-neutral, fail-closed model fault/retry orchestrator."""

    def __init__(
        self,
        *,
        resolver: MemoryFaultResolver,
        client: ModelClient,
        endpoint_url: str,
        model_id: str,
        runtime_id: str,
        settings: ProbeSettings,
        policy: Optional[FaultRetryPolicy] = None,
        runtime_adapter: Optional[APIMessageAdapter] = None,
        checkpoint_plan: Optional[CheckpointRecoveryPlan] = None,
    ) -> None:
        if not endpoint_url:
            raise ValueError("endpoint_url must be non-empty")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        if not runtime_id:
            raise ValueError("runtime_id must be non-empty")
        self.resolver = resolver
        self.client = client
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.runtime_id = runtime_id
        self.settings = settings
        self.policy = policy or FaultRetryPolicy()
        self.runtime_adapter = runtime_adapter or APIMessageAdapter()
        self.checkpoint_plan = checkpoint_plan

    def run(self, packet: ContextPacket) -> FaultRetryResult:
        # Reconstruct from the wire contract before using mutable caller-owned
        # packet state. This revalidates scope, prompt, manifest, and digest.
        validated = ContextPacket.from_dict(packet.to_dict())
        if self.model_id != validated.capabilities.model_id:
            raise ValueError("model_id must match packet capabilities")
        if self.runtime_id != validated.capabilities.backend_id:
            raise ValueError("runtime_id must match packet capabilities")
        expected_endpoint_digest = validated.capabilities.metadata.get("endpoint_url_digest")
        actual_endpoint_digest = endpoint_identity_digest(self.endpoint_url)
        if not isinstance(expected_endpoint_digest, str) or expected_endpoint_digest != actual_endpoint_digest:
            raise ValueError("endpoint_url must match packet capabilities")
        if self.checkpoint_plan is not None:
            self._validate_checkpoint_plan(validated, self.checkpoint_plan)

        initial = self._call(validated.rendered_messages, "initial")
        if initial is None:
            return FaultRetryResult(
                status=RecoveryStatus.MODEL_ERROR,
                response=_empty_response(),
                reason_code="initial_model_error",
            )
        if not self.policy.is_explicit_miss(initial.content):
            return FaultRetryResult(status=RecoveryStatus.COMPLETED, response=initial)
        if self.policy.max_retries == 0:
            return FaultRetryResult(
                status=RecoveryStatus.RETRY_EXHAUSTED,
                response=initial,
                reason_code="retry_disabled",
            )

        candidates = _page_candidates(validated.decisions)
        if len(candidates) != 1:
            reason = "missing_page_handle" if not candidates else "ambiguous_page_handle"
            return FaultRetryResult(status=RecoveryStatus.FAULT_UNRESOLVED, response=initial, reason_code=reason)
        candidate = candidates[0]

        request = MemoryFaultRequest(
            request_id=_request_id(validated.packet_content_digest, candidate.key),
            scope=validated.scope,
            key=candidate.key,
            expected_digest=candidate.digest,
            expected_source_id=candidate.segment_id,
            budget=MemoryFaultBudget(
                max_items=1,
                max_payload_bytes=self.policy.max_evidence_bytes,
                max_payload_tokens=validated.budget.available_input_tokens,
            ),
            include_payload=True,
            allow_durable=False,
        )
        resolved = self.resolver.resolve(request)
        attempted_tiers = [
            str(attempt.get("tier"))
            for attempt in resolved.telemetry.get("attempts", [])
            if isinstance(attempt, Mapping) and attempt.get("tier")
        ]
        if resolved.status is not MemoryFaultStatus.HIT or len(resolved.evidence) != 1:
            binding_mismatch = MemoryFaultReason.BINDING_MISMATCH in resolved.reason_codes or any(
                MemoryFaultReason.BINDING_MISMATCH.value in attempt.get("reason_codes", [])
                for attempt in resolved.telemetry.get("attempts", [])
                if isinstance(attempt, Mapping)
            )
            reason_code = (
                "evidence_binding_mismatch"
                if binding_mismatch
                else f"memory_fault_{resolved.status.value}"
            )
            return FaultRetryResult(
                status=RecoveryStatus.FAULT_UNRESOLVED,
                response=initial,
                reason_code=reason_code,
                attempted_tiers=attempted_tiers,
            )

        item = resolved.evidence[0]
        if item.digest != candidate.digest:
            return FaultRetryResult(
                status=RecoveryStatus.FAULT_UNRESOLVED,
                response=initial,
                reason_code="evidence_digest_mismatch",
                attempted_tiers=attempted_tiers,
            )
        if item.provenance.source_id and item.provenance.source_id != candidate.segment_id:
            return FaultRetryResult(
                status=RecoveryStatus.FAULT_UNRESOLVED,
                response=initial,
                reason_code="evidence_source_mismatch",
                attempted_tiers=attempted_tiers,
            )
        payload = item.payload_text
        if payload is None:
            return FaultRetryResult(
                status=RecoveryStatus.FAULT_UNRESOLVED,
                response=initial,
                reason_code="text_evidence_required",
                attempted_tiers=attempted_tiers,
            )

        retry_messages = self._retry_messages(validated, payload, candidate.segment_id)
        if retry_messages is None:
            return FaultRetryResult(
                status=RecoveryStatus.FAULT_UNRESOLVED,
                response=initial,
                reason_code="recovered_evidence_over_budget",
                attempted_tiers=attempted_tiers,
            )
        safe_evidence = [item.without_payload()]
        checkpoint_restore: Optional[CheckpointRestoreResult] = None
        success_reason = "bounded_page_in_retry_completed"
        retry_error_reason = "retry_model_error"
        if self.checkpoint_plan is not None:
            try:
                checkpoint_restore = self.checkpoint_plan.controller.restore_matching(
                    self.checkpoint_plan.identity,
                    slot_id=self.checkpoint_plan.slot_id,
                    max_records=self.checkpoint_plan.max_selection_records,
                )
            except CheckpointSelectionError as exc:
                return FaultRetryResult(
                    status=RecoveryStatus.FAULT_UNRESOLVED,
                    response=initial,
                    reason_code=exc.reason_code,
                    attempted_tiers=attempted_tiers,
                    evidence=safe_evidence,
                )
            except RuntimeExecutionError:
                return FaultRetryResult(
                    status=RecoveryStatus.FAULT_UNRESOLVED,
                    response=initial,
                    reason_code="checkpoint_restore_failed",
                    attempted_tiers=attempted_tiers,
                    evidence=safe_evidence,
                )
            retried = self._call_on_slot(retry_messages, "fault-retry-1", self.checkpoint_plan)
            success_reason = "bounded_checkpoint_page_in_retry_completed"
            retry_error_reason = "checkpoint_retry_model_error"
        else:
            retried = self._call(retry_messages, "fault-retry-1")
        if retried is None:
            return FaultRetryResult(
                status=RecoveryStatus.MODEL_ERROR,
                response=_empty_response(),
                retry_count=1,
                reason_code=retry_error_reason,
                attempted_tiers=attempted_tiers,
                evidence=safe_evidence,
                checkpoint_restore=checkpoint_restore,
            )
        if self.policy.is_explicit_miss(retried.content):
            return FaultRetryResult(
                status=RecoveryStatus.RETRY_EXHAUSTED,
                response=retried,
                retry_count=1,
                reason_code="explicit_miss_after_retry",
                attempted_tiers=attempted_tiers,
                evidence=safe_evidence,
                checkpoint_restore=checkpoint_restore,
            )
        return FaultRetryResult(
            status=RecoveryStatus.RECOVERED,
            response=retried,
            retry_count=1,
            reason_code=success_reason,
            attempted_tiers=attempted_tiers,
            evidence=safe_evidence,
            checkpoint_restore=checkpoint_restore,
        )

    def _validate_checkpoint_plan(self, packet: ContextPacket, plan: CheckpointRecoveryPlan) -> None:
        identity = plan.identity
        try:
            identity.to_dict()
            capabilities = plan.controller.adapter.capabilities()
        except Exception as exc:
            raise ValueError("checkpoint identity preflight failed") from exc
        if (
            identity.scope != packet.scope
            or identity.model_id != self.model_id
            or identity.runtime_id != self.runtime_id
            or identity.packet_digest != "sha256:" + packet.packet_content_digest
            or identity.manifest_digest != "sha256:" + packet.manifest.content_digest
        ):
            raise ValueError("checkpoint identity must match the context packet")
        endpoint_digest = "sha256:" + endpoint_origin_identity_digest(self.endpoint_url)
        if identity.runtime_endpoint_digest != endpoint_digest:
            raise ValueError("checkpoint endpoint must match the model endpoint")
        if (
            capabilities.runtime_id != identity.runtime_id
            or capabilities.runtime_version != identity.runtime_version
            or capabilities.adapter_id != identity.adapter_id
            or capabilities.metadata.get("endpoint_origin_digest") != endpoint_digest
            or not capabilities.supports_kv_restore
            or not capabilities.supports_slot_affinity
        ):
            raise ValueError("checkpoint runtime capabilities do not match the bound identity")

    def _call_on_slot(
        self,
        messages: Sequence[Mapping[str, Any]],
        label: str,
        plan: CheckpointRecoveryPlan,
    ) -> Optional[ModelClientResponse]:
        copied = [dict(message) for message in messages]
        try:
            return plan.runtime.complete_on_slot(
                self.endpoint_url,
                self.model_id,
                copied,
                self.settings,
                slot_id=plan.slot_id,
                label=label,
            )
        except Exception:
            return None

    def _retry_messages(
        self,
        packet: ContextPacket,
        payload: str,
        source_segment_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        essential = [
            segment
            for segment in packet.segments
            if segment.authority
            in {
                ContextAuthority.IMMUTABLE,
                ContextAuthority.CURRENT_INSTRUCTION,
                ContextAuthority.TRUSTED_CONTROL,
            }
        ]
        wrapped = f"[UNTRUSTED DATA FROM MEMORY]\n{payload}"
        evidence_tokens = max(1, (len(wrapped) + 3) // 4)
        if sum(segment.effective_tokens for segment in essential) + evidence_tokens > packet.budget.available_input_tokens:
            return None

        renderable = [_renderable(segment) for segment in essential]
        renderable.append(
            {
                "id": f"recovered:{source_segment_id}",
                "kind": "retrieved",
                "role": "user",
                "content": wrapped,
                "source": "memory",
                "metadata": {"authority": ContextAuthority.UNTRUSTED_DATA.value},
            }
        )
        rendered = self.runtime_adapter.render_messages(renderable)
        return [dict(message) for message in rendered["messages"]]

    def _call(self, messages: Sequence[Mapping[str, Any]], label: str) -> Optional[ModelClientResponse]:
        copied = [dict(message) for message in messages]
        try:
            return self.client.complete(
                self.endpoint_url,
                self.model_id,
                copied,
                self.settings,
                label=label,
            )
        except Exception:
            return None


def _page_candidates(decisions: Mapping[str, Any]) -> List[_PageCandidate]:
    by_segment = decisions.get("by_segment")
    omitted = decisions.get("omitted")
    if not isinstance(by_segment, Mapping) or not isinstance(omitted, list):
        return []
    candidates: List[_PageCandidate] = []
    for segment_id in omitted:
        decision = by_segment.get(segment_id)
        if not isinstance(decision, Mapping):
            continue
        page_out = decision.get("page_out")
        if not isinstance(page_out, Mapping) or page_out.get("status") != "stored":
            continue
        handle = page_out.get("handle")
        if not isinstance(handle, Mapping):
            continue
        key = handle.get("page_id") or handle.get("handle_id")
        digest = handle.get("digest")
        if not isinstance(key, str) or not key or not isinstance(digest, str) or not digest:
            continue
        candidates.append(_PageCandidate(segment_id=str(segment_id), key=key, digest=digest))
    return candidates


def _renderable(segment: ContextSegment) -> Dict[str, Any]:
    return {
        "id": segment.segment_id,
        "kind": segment.kind,
        "role": "system" if segment.authority is ContextAuthority.IMMUTABLE else "user",
        "content": segment.content,
        "source": segment.source,
        "metadata": {"authority": segment.authority.value, "priority": segment.priority.value},
    }


def _request_id(packet_digest: str, key: str) -> str:
    return "fault-" + hashlib.sha256(f"{packet_digest}:{key}".encode("utf-8")).hexdigest()[:24]


def _empty_response() -> ModelClientResponse:
    return ModelClientResponse(content="", latency_ms=0.0, usage={})
