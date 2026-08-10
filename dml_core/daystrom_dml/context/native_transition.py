"""Digest-bound execution plans joining semantic working sets to native runtime state."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.context.checkpoints import ExecutionCheckpointRecord
from daystrom_dml.context.manifest import ContextPacket
from daystrom_dml.context.working_set import WORKING_SET_TRANSITION_V1

NATIVE_CONTEXT_TRANSITION_V1 = "daystrom-native-context-transition-v1"
NATIVE_CONTEXT_CHECKPOINT_BINDING_V1 = (
    "daystrom-native-context-checkpoint-binding-v1"
)
_PAGE_ACTIONS = {"page_out_exact", "page_in_exact"}
_STEP_OPERATIONS = {
    "restore_parent_prefix",
    "prefill_suffix",
    "prefill_full",
    "checkpoint_current_generation",
}


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{name} must be {qualifier}")
    return value


def _sha256(value: Any, *, prefixed: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if prefixed and not value.startswith("sha256:"):
        return False
    candidate = value.removeprefix("sha256:") if prefixed else value
    return len(candidate) == 64 and all(character in "0123456789abcdef" for character in candidate)


def _json_digest(value: Any, *, prefixed: bool = False) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return "sha256:" + digest if prefixed else digest


def _copy_packet(value: ContextPacket, label: str) -> ContextPacket:
    if not isinstance(value, ContextPacket):
        raise ContractError(f"{label} must be a ContextPacket")
    try:
        return ContextPacket.from_dict(value.to_dict())
    except Exception as exc:
        raise ContractError(f"{label} integrity check failed") from exc


@dataclass(frozen=True)
class NativeContextPage:
    """One payload-free exact page movement across the active native window."""

    span_id: str
    span_digest: str
    token_start: int
    token_count: int
    action: str

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ContractError("span_id must be non-empty")
        if not _sha256(self.span_digest):
            raise ContractError("span_digest must be a lowercase SHA-256 digest")
        _positive_int("token_start", self.token_start, allow_zero=True)
        _positive_int("token_count", self.token_count)
        if self.action not in _PAGE_ACTIONS:
            raise ContractError("native context page action is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_digest": self.span_digest,
            "token_start": self.token_start,
            "token_count": self.token_count,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "NativeContextPage":
        if not isinstance(value, dict) or set(value) != {
            "span_id",
            "span_digest",
            "token_start",
            "token_count",
            "action",
        }:
            raise ContractError("native context page is malformed")
        return cls(**value)


@dataclass(frozen=True)
class NativeContextRuntimeStep:
    """Adapter-neutral runtime work required for one generation transition."""

    operation: str
    span_ids: list[str]
    token_start: int
    token_count: int
    reason_code: str
    checkpoint_id: Optional[str] = None
    checkpoint_digest: Optional[str] = None
    binding_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if self.operation not in _STEP_OPERATIONS:
            raise ContractError("native runtime step operation is invalid")
        if not isinstance(self.span_ids, list) or any(
            not isinstance(item, str) or not item for item in self.span_ids
        ):
            raise ContractError("runtime step span_ids must be non-empty strings")
        if len(set(self.span_ids)) != len(self.span_ids):
            raise ContractError("runtime step span_ids must be unique")
        _positive_int("token_start", self.token_start, allow_zero=True)
        _positive_int("token_count", self.token_count, allow_zero=True)
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ContractError("runtime step reason_code must be non-empty")
        if self.checkpoint_digest is not None and not _sha256(
            self.checkpoint_digest, prefixed=True
        ):
            raise ContractError("checkpoint_digest must be a SHA-256 identity")
        if self.binding_digest is not None and not _sha256(
            self.binding_digest, prefixed=True
        ):
            raise ContractError("binding_digest must be a SHA-256 identity")
        if self.operation == "restore_parent_prefix":
            if not self.checkpoint_id or not self.binding_digest:
                raise ContractError("restore step requires checkpoint identity binding")
        if self.operation == "checkpoint_current_generation" and not self.checkpoint_digest:
            raise ContractError("checkpoint step requires checkpoint_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "span_ids": list(self.span_ids),
            "token_start": self.token_start,
            "token_count": self.token_count,
            "reason_code": self.reason_code,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_digest": self.checkpoint_digest,
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "NativeContextRuntimeStep":
        expected = {
            "operation",
            "span_ids",
            "token_start",
            "token_count",
            "reason_code",
            "checkpoint_id",
            "checkpoint_digest",
            "binding_digest",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ContractError("native runtime step is malformed")
        return cls(**value)


@dataclass(frozen=True)
class NativeContextCheckpointBinding:
    """Bind controller checkpoint authorization to durable registry metadata."""

    record: ExecutionCheckpointRecord
    runtime_checkpoint_digest: str
    binding_version: str = NATIVE_CONTEXT_CHECKPOINT_BINDING_V1
    binding_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.record, ExecutionCheckpointRecord):
            raise ContractError("checkpoint binding record is invalid")
        if not _sha256(self.runtime_checkpoint_digest, prefixed=True):
            raise ContractError("runtime_checkpoint_digest must be a SHA-256 identity")
        if self.binding_version != NATIVE_CONTEXT_CHECKPOINT_BINDING_V1:
            raise ContractError("unsupported native checkpoint binding version")
        computed = _json_digest(
            {
                "binding_version": self.binding_version,
                "record_digest": self.record.record_digest,
                "runtime_checkpoint_digest": self.runtime_checkpoint_digest,
            }
        )
        if self.binding_digest and self.binding_digest != computed:
            raise ContractError("checkpoint binding integrity check failed")
        object.__setattr__(self, "binding_digest", computed)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "record": self.record.to_dict(),
            "runtime_checkpoint_digest": self.runtime_checkpoint_digest,
            "binding_version": self.binding_version,
            "binding_digest": self.binding_digest,
        }
        expected = _json_digest(
            {
                "binding_version": self.binding_version,
                "record_digest": self.record.record_digest,
                "runtime_checkpoint_digest": self.runtime_checkpoint_digest,
            }
        )
        if self.binding_digest != expected:
            raise ContractError("checkpoint binding integrity check failed")
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "NativeContextCheckpointBinding":
        if not isinstance(value, dict) or set(value) != {
            "record",
            "runtime_checkpoint_digest",
            "binding_version",
            "binding_digest",
        }:
            raise ContractError("native checkpoint binding is malformed")
        if not isinstance(value["record"], dict) or not value.get("binding_digest"):
            raise ContractError("native checkpoint binding integrity fields are required")
        return cls(
            record=ExecutionCheckpointRecord.from_dict(value["record"]),
            runtime_checkpoint_digest=value["runtime_checkpoint_digest"],
            binding_version=value["binding_version"],
            binding_digest=value["binding_digest"],
        )


@dataclass(frozen=True)
class NativeContextTransitionPlan:
    """Deterministic transition from one exact ContextPacket to the next."""

    model_id: str
    runtime_id: str
    parent_packet_digest: str
    current_packet_digest: str
    parent_manifest_digest: str
    current_manifest_digest: str
    model_native_limit: int
    served_limit: int
    current_tokens: int
    stable_prefix_span_ids: list[str]
    stable_prefix_tokens: int
    suffix_span_ids: list[str]
    suffix_tokens: int
    page_out: list[NativeContextPage]
    page_in: list[NativeContextPage]
    steps: list[NativeContextRuntimeStep]
    current_checkpoint_digest: str
    served_overflow_tokens: int
    served_limit_shortfall: int
    feasible: bool
    reason_codes: list[str] = field(default_factory=list)
    schema_version: str = NATIVE_CONTEXT_TRANSITION_V1
    plan_digest: str = ""

    def __post_init__(self) -> None:
        for name in ("model_id", "runtime_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ContractError(f"{name} must be non-empty")
        for name in (
            "parent_packet_digest",
            "current_packet_digest",
            "parent_manifest_digest",
            "current_manifest_digest",
        ):
            if not _sha256(getattr(self, name)):
                raise ContractError(f"{name} must be a lowercase SHA-256 digest")
        if not _sha256(self.current_checkpoint_digest, prefixed=True):
            raise ContractError("current_checkpoint_digest must be a SHA-256 identity")
        for name in (
            "model_native_limit",
            "served_limit",
            "current_tokens",
            "stable_prefix_tokens",
            "suffix_tokens",
            "served_overflow_tokens",
            "served_limit_shortfall",
        ):
            _positive_int(name, getattr(self, name), allow_zero=name not in {"model_native_limit", "served_limit"})
        if self.served_limit > self.model_native_limit:
            raise ContractError("served_limit cannot exceed model_native_limit")
        if self.current_tokens > self.model_native_limit:
            raise ContractError("current generation exceeds model_native_limit")
        if self.stable_prefix_tokens + self.suffix_tokens != self.current_tokens:
            raise ContractError("prefix and suffix tokens must cover the current generation")
        if len(set(self.stable_prefix_span_ids + self.suffix_span_ids)) != len(
            self.stable_prefix_span_ids + self.suffix_span_ids
        ):
            raise ContractError("transition span ids must be unique")
        if not all(isinstance(item, NativeContextPage) for item in self.page_out + self.page_in):
            raise ContractError("page movements must be NativeContextPage values")
        if not all(isinstance(item, NativeContextRuntimeStep) for item in self.steps):
            raise ContractError("steps must be NativeContextRuntimeStep values")
        if not isinstance(self.feasible, bool):
            raise ContractError("feasible must be boolean")
        if self.schema_version != NATIVE_CONTEXT_TRANSITION_V1:
            raise ContractError("unsupported native context transition schema")
        payload = self._payload()
        computed = _json_digest(payload)
        if self.plan_digest and self.plan_digest != computed:
            raise ContractError("plan_digest does not match native context transition")
        object.__setattr__(self, "plan_digest", computed)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "runtime_id": self.runtime_id,
            "parent_packet_digest": self.parent_packet_digest,
            "current_packet_digest": self.current_packet_digest,
            "parent_manifest_digest": self.parent_manifest_digest,
            "current_manifest_digest": self.current_manifest_digest,
            "model_native_limit": self.model_native_limit,
            "served_limit": self.served_limit,
            "current_tokens": self.current_tokens,
            "stable_prefix_span_ids": list(self.stable_prefix_span_ids),
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "suffix_span_ids": list(self.suffix_span_ids),
            "suffix_tokens": self.suffix_tokens,
            "page_out": [item.to_dict() for item in self.page_out],
            "page_in": [item.to_dict() for item in self.page_in],
            "steps": [item.to_dict() for item in self.steps],
            "current_checkpoint_digest": self.current_checkpoint_digest,
            "served_overflow_tokens": self.served_overflow_tokens,
            "served_limit_shortfall": self.served_limit_shortfall,
            "feasible": self.feasible,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        if self.plan_digest != _json_digest(payload):
            raise ContractError("plan_digest no longer matches native context transition")
        return {**payload, "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Any) -> "NativeContextTransitionPlan":
        if not isinstance(value, dict) or not value.get("plan_digest"):
            raise ContractError("native context transition plan_digest is required")
        allowed = set(cls.__dataclass_fields__)
        if set(value) != allowed:
            raise ContractError("native context transition is malformed")
        payload = dict(value)
        payload["page_out"] = [NativeContextPage.from_dict(item) for item in payload["page_out"]]
        payload["page_in"] = [NativeContextPage.from_dict(item) for item in payload["page_in"]]
        payload["steps"] = [NativeContextRuntimeStep.from_dict(item) for item in payload["steps"]]
        return cls(**payload)


class NativeContextTransitionCompiler:
    """Compile integrity-bound semantic lineage into adapter-neutral runtime work."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock

    def compile(
        self,
        *,
        parent_packet: ContextPacket,
        current_packet: ContextPacket,
        parent_checkpoint: Optional[NativeContextCheckpointBinding],
        model_native_limit: int,
        served_limit: int,
    ) -> NativeContextTransitionPlan:
        native_limit = _positive_int("model_native_limit", model_native_limit)
        runtime_limit = _positive_int("served_limit", served_limit)
        if runtime_limit > native_limit:
            raise ContractError("served_limit cannot exceed model_native_limit")
        parent = _copy_packet(parent_packet, "parent_packet")
        current = _copy_packet(current_packet, "current_packet")
        self._validate_lineage(parent, current)

        parent_ids = list(parent.manifest.segment_ids)
        current_ids = list(current.manifest.segment_ids)
        parent_digests = dict(parent.manifest.segment_digests)
        current_digests = dict(current.manifest.segment_digests)
        semantic_prefix = []
        for previous_id, current_id in zip(parent_ids, current_ids):
            if previous_id != current_id or parent_digests[previous_id] != current_digests[current_id]:
                break
            semantic_prefix.append(current_id)
        self._validate_transition_decision(current, semantic_prefix)

        parent_positions = self._positions(parent)
        current_positions = self._positions(current)
        page_out = [
            NativeContextPage(
                span_id=span_id,
                span_digest=parent_digests[span_id],
                token_start=parent_positions[span_id][0],
                token_count=parent_positions[span_id][1],
                action="page_out_exact",
            )
            for span_id in parent_ids
            if span_id not in current_digests or current_digests[span_id] != parent_digests[span_id]
        ]
        page_in = [
            NativeContextPage(
                span_id=span_id,
                span_digest=current_digests[span_id],
                token_start=current_positions[span_id][0],
                token_count=current_positions[span_id][1],
                action="page_in_exact",
            )
            for span_id in current_ids
            if span_id not in parent_digests or parent_digests[span_id] != current_digests[span_id]
        ]

        semantic_prefix_tokens = sum(current_positions[item][1] for item in semantic_prefix)
        checkpoint = self._copy_checkpoint(parent_checkpoint)
        reason_codes: list[str] = []
        reusable_prefix = list(semantic_prefix)
        reusable_tokens = semantic_prefix_tokens
        if checkpoint is None:
            reusable_prefix = []
            reusable_tokens = 0
            reason_codes.append("parent_checkpoint_unavailable")
        elif not semantic_prefix:
            reusable_prefix = []
            reusable_tokens = 0
            reason_codes.append("no_stable_prefix_to_restore")
        else:
            self._validate_checkpoint(checkpoint.record, parent, semantic_prefix_tokens)
            reason_codes.append("parent_checkpoint_bound_to_stable_prefix")

        suffix_ids = current_ids[len(reusable_prefix) :]
        suffix_tokens = sum(current_positions[item][1] for item in suffix_ids)
        current_tokens = sum(item.effective_tokens for item in current.segments)
        checkpoint_digest = self._current_checkpoint_digest(
            current,
            model_native_limit=native_limit,
            served_limit=runtime_limit,
        )
        steps: list[NativeContextRuntimeStep] = []
        if checkpoint is not None and reusable_tokens:
            steps.append(
                NativeContextRuntimeStep(
                    operation="restore_parent_prefix",
                    span_ids=reusable_prefix,
                    token_start=0,
                    token_count=reusable_tokens,
                    reason_code="verified_parent_checkpoint_reuses_stable_prefix",
                    checkpoint_id=checkpoint.record.checkpoint_id,
                    checkpoint_digest=checkpoint.runtime_checkpoint_digest,
                    binding_digest=checkpoint.record.identity.binding_digest,
                )
            )
            steps.append(
                NativeContextRuntimeStep(
                    operation="prefill_suffix",
                    span_ids=suffix_ids,
                    token_start=reusable_tokens,
                    token_count=suffix_tokens,
                    reason_code="prefill_begins_at_first_digest_or_position_divergence",
                )
            )
        else:
            steps.append(
                NativeContextRuntimeStep(
                    operation="prefill_full",
                    span_ids=current_ids,
                    token_start=0,
                    token_count=current_tokens,
                    reason_code="no_verified_runtime_prefix_available",
                )
            )
        steps.append(
            NativeContextRuntimeStep(
                operation="checkpoint_current_generation",
                span_ids=current_ids,
                token_start=0,
                token_count=current_tokens,
                reason_code="bind_completed_generation_for_future_prefix_restore",
                checkpoint_digest=checkpoint_digest,
            )
        )

        overflow = max(0, current_tokens - runtime_limit)
        if overflow:
            reason_codes.append("current_generation_exceeds_served_limit")
        if runtime_limit < native_limit:
            reason_codes.append("runtime_serves_less_than_model_native_limit")
        if page_out:
            reason_codes.append("exact_pages_leave_active_window")
        if page_in:
            reason_codes.append("exact_pages_enter_active_window")
        return NativeContextTransitionPlan(
            model_id=current.capabilities.model_id,
            runtime_id=current.capabilities.backend_id,
            parent_packet_digest=parent.packet_content_digest,
            current_packet_digest=current.packet_content_digest,
            parent_manifest_digest=parent.manifest.content_digest,
            current_manifest_digest=current.manifest.content_digest,
            model_native_limit=native_limit,
            served_limit=runtime_limit,
            current_tokens=current_tokens,
            stable_prefix_span_ids=reusable_prefix,
            stable_prefix_tokens=reusable_tokens,
            suffix_span_ids=suffix_ids,
            suffix_tokens=suffix_tokens,
            page_out=page_out,
            page_in=page_in,
            steps=steps,
            current_checkpoint_digest=checkpoint_digest,
            served_overflow_tokens=overflow,
            served_limit_shortfall=native_limit - runtime_limit,
            feasible=overflow == 0,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _positions(packet: ContextPacket) -> dict[str, tuple[int, int]]:
        positions: dict[str, tuple[int, int]] = {}
        start = 0
        for segment in packet.segments:
            positions[segment.segment_id] = (start, segment.effective_tokens)
            start += segment.effective_tokens
        return positions

    @staticmethod
    def _validate_lineage(parent: ContextPacket, current: ContextPacket) -> None:
        if parent.scope != current.scope:
            raise ContractError("parent packet scope does not match current packet")
        if parent.capabilities.model_id != current.capabilities.model_id:
            raise ContractError("parent packet model does not match current packet")
        if parent.capabilities.backend_id != current.capabilities.backend_id:
            raise ContractError("parent packet runtime does not match current packet")
        if current.manifest.parent_manifest_id != parent.manifest.content_digest:
            raise ContractError("current packet parent manifest lineage does not match")
        if not parent.manifest.segment_digests or not current.manifest.segment_digests:
            raise ContractError("native transition requires exact segment digest bindings")

    @staticmethod
    def _validate_transition_decision(current: ContextPacket, stable_prefix: list[str]) -> None:
        decision = current.decisions.get("working_set")
        if not isinstance(decision, dict) or decision.get("version") != WORKING_SET_TRANSITION_V1:
            raise ContractError("current packet lacks a working-set transition")
        if decision.get("parent_manifest_id") != current.manifest.parent_manifest_id:
            raise ContractError("working-set transition parent manifest drifted")
        if decision.get("stable_prefix") != stable_prefix:
            raise ContractError("working-set transition stable prefix drifted")
        if decision.get("prefill_from_index") != len(stable_prefix):
            raise ContractError("working-set transition prefill boundary drifted")
        if decision.get("prefill_segment_ids") != current.manifest.segment_ids[len(stable_prefix) :]:
            raise ContractError("working-set transition prefill suffix drifted")

    @staticmethod
    def _copy_checkpoint(
        value: Optional[NativeContextCheckpointBinding],
    ) -> Optional[NativeContextCheckpointBinding]:
        if value is None:
            return None
        if not isinstance(value, NativeContextCheckpointBinding):
            raise ContractError(
                "parent_checkpoint must be a NativeContextCheckpointBinding"
            )
        try:
            return NativeContextCheckpointBinding.from_dict(value.to_dict())
        except Exception as exc:
            raise ContractError("parent checkpoint integrity check failed") from exc

    def _validate_checkpoint(
        self,
        record: ExecutionCheckpointRecord,
        parent: ContextPacket,
        stable_prefix_tokens: int,
    ) -> None:
        identity = record.identity
        if record.expires_at <= float(self.clock()):
            raise ContractError("parent checkpoint is expired")
        if identity.scope != parent.scope:
            raise ContractError("parent checkpoint scope does not match parent packet")
        if identity.model_id != parent.capabilities.model_id:
            raise ContractError("parent checkpoint model does not match parent packet")
        if identity.runtime_id != parent.capabilities.backend_id:
            raise ContractError("parent checkpoint runtime does not match parent packet")
        if identity.packet_digest != "sha256:" + parent.packet_content_digest:
            raise ContractError("parent checkpoint packet digest does not match")
        if identity.manifest_digest != "sha256:" + parent.manifest.content_digest:
            raise ContractError("parent checkpoint manifest digest does not match")
        if record.tokens_saved < stable_prefix_tokens:
            raise ContractError("parent checkpoint does not cover the stable prefix")

    @staticmethod
    def _current_checkpoint_digest(
        current: ContextPacket,
        *,
        model_native_limit: int,
        served_limit: int,
    ) -> str:
        return _json_digest(
            {
                "schema_version": NATIVE_CONTEXT_TRANSITION_V1,
                "scope": current.scope.to_dict(),
                "model_id": current.capabilities.model_id,
                "runtime_id": current.capabilities.backend_id,
                "packet_digest": current.packet_content_digest,
                "manifest_digest": current.manifest.content_digest,
                "model_native_limit": model_native_limit,
                "served_limit": served_limit,
            },
            prefixed=True,
        )
