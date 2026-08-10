"""Payload-free profiler for managed model-native context generations.

The profiler does not mutate a runtime or summarize payloads.  It consumes exact
position/token/digest metadata plus optional precomputed summary candidates and
emits a deterministic retain/compress/freeze/thaw/hot-swap plan.  Any change to
an ordered native span invalidates KV/state reuse from the first divergent token
position; the report therefore exposes the stable-prefix and suffix-recompute
boundary explicitly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.context.schema import ContextAuthority, ContextPriority

NATIVE_CONTEXT_PROFILE_V1 = "daystrom-native-context-profile-v1"
_ACTIONS = {"retain", "compress", "freeze", "thaw", "hot_swap_out"}
_PROTECTED_AUTHORITIES = {
    ContextAuthority.IMMUTABLE.value,
    ContextAuthority.CURRENT_INSTRUCTION.value,
    ContextAuthority.TRUSTED_CONTROL.value,
}
_PRIORITY_COLDNESS = {
    ContextPriority.DISPOSABLE.value: 0,
    ContextPriority.REFERENCE.value: 1,
    ContextPriority.WORKING.value: 2,
    ContextPriority.CRITICAL.value: 3,
}


def _sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ContractError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True)
class NativeContextSpan:
    """One exact logical span in the model-native sequence address space."""

    span_id: str
    content_digest: str
    start_token: int
    token_count: int
    resident: bool = True
    age_turns: int = 0
    reference_count: int = 0
    priority: str = ContextPriority.REFERENCE.value
    authority: str = ContextAuthority.REFERENCE.value
    exact_required: bool = False
    summary_digest: Optional[str] = None
    summary_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ContractError("span_id must be non-empty")
        if not _sha256(self.content_digest):
            raise ContractError("content_digest must be a lowercase SHA-256 digest")
        _positive_int("start_token", self.start_token, allow_zero=True)
        _positive_int("token_count", self.token_count)
        _positive_int("age_turns", self.age_turns, allow_zero=True)
        _positive_int("reference_count", self.reference_count, allow_zero=True)
        if not isinstance(self.resident, bool) or not isinstance(self.exact_required, bool):
            raise ContractError("resident and exact_required must be booleans")
        if self.priority not in _PRIORITY_COLDNESS:
            raise ContractError("priority is invalid")
        if self.authority not in {item.value for item in ContextAuthority}:
            raise ContractError("authority is invalid")
        if (self.summary_digest is None) != (self.summary_tokens is None):
            raise ContractError("summary_digest and summary_tokens must be configured together")
        if self.summary_digest is not None:
            if not _sha256(self.summary_digest):
                raise ContractError("summary_digest must be a lowercase SHA-256 digest")
            summary_tokens = _positive_int("summary_tokens", self.summary_tokens)
            if summary_tokens >= self.token_count:
                raise ContractError("summary_tokens must be smaller than token_count")

    @property
    def end_token(self) -> int:
        return self.start_token + self.token_count

    @property
    def protected(self) -> bool:
        return self.exact_required or self.authority in _PROTECTED_AUTHORITIES


@dataclass(frozen=True)
class NativeContextProfileConfig:
    model_id: str
    runtime_id: str
    model_native_limit: int
    served_limit: int
    target_hot_tokens: int
    stale_after_turns: int = 4
    freeze_after_turns: int = 12
    runtime_state_bytes_per_token: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ContractError("model_id must be non-empty")
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ContractError("runtime_id must be non-empty")
        _positive_int("model_native_limit", self.model_native_limit)
        _positive_int("served_limit", self.served_limit)
        _positive_int("target_hot_tokens", self.target_hot_tokens)
        _positive_int("stale_after_turns", self.stale_after_turns, allow_zero=True)
        _positive_int("freeze_after_turns", self.freeze_after_turns, allow_zero=True)
        _positive_int(
            "runtime_state_bytes_per_token",
            self.runtime_state_bytes_per_token,
            allow_zero=True,
        )
        if self.served_limit > self.model_native_limit:
            raise ContractError("served_limit cannot exceed model_native_limit")
        if self.target_hot_tokens > self.model_native_limit:
            raise ContractError("target_hot_tokens cannot exceed model_native_limit")
        if self.freeze_after_turns < self.stale_after_turns:
            raise ContractError("freeze_after_turns cannot precede stale_after_turns")


@dataclass(frozen=True)
class NativeContextAction:
    span_id: str
    span_digest: str
    action: str
    tokens_before: int
    tokens_after: int
    reason_code: str

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ContractError("native context action is invalid")
        if not _sha256(self.span_digest):
            raise ContractError("span_digest must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "span_digest": self.span_digest,
            "action": self.action,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class NativeContextProfile:
    model_id: str
    runtime_id: str
    model_native_limit: int
    served_limit: int
    target_hot_tokens: int
    logical_tokens: int
    resident_tokens_before: int
    resident_tokens_after: int
    compressed_tokens_saved: int
    frozen_exact_tokens: int
    thawed_exact_tokens: int
    stable_prefix_tokens: int
    recompute_from_token: int
    recompute_tokens: int
    tier_bytes_out: int
    tier_bytes_in: int
    hot_overflow_tokens: int
    served_overflow_tokens: int
    served_limit_shortfall: int
    native_window_utilization: float
    served_window_utilization: float
    feasible: bool
    hot_swap_in: list[str] = field(default_factory=list)
    hot_swap_out: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    actions: list[NativeContextAction] = field(default_factory=list)
    schema_version: str = NATIVE_CONTEXT_PROFILE_V1
    profile_digest: str = ""

    def __post_init__(self) -> None:
        payload = self._payload()
        computed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.profile_digest and self.profile_digest != computed:
            raise ContractError("profile_digest does not match native context profile")
        object.__setattr__(self, "profile_digest", computed)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "runtime_id": self.runtime_id,
            "model_native_limit": self.model_native_limit,
            "served_limit": self.served_limit,
            "target_hot_tokens": self.target_hot_tokens,
            "logical_tokens": self.logical_tokens,
            "resident_tokens_before": self.resident_tokens_before,
            "resident_tokens_after": self.resident_tokens_after,
            "compressed_tokens_saved": self.compressed_tokens_saved,
            "frozen_exact_tokens": self.frozen_exact_tokens,
            "thawed_exact_tokens": self.thawed_exact_tokens,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "recompute_from_token": self.recompute_from_token,
            "recompute_tokens": self.recompute_tokens,
            "tier_bytes_out": self.tier_bytes_out,
            "tier_bytes_in": self.tier_bytes_in,
            "hot_overflow_tokens": self.hot_overflow_tokens,
            "served_overflow_tokens": self.served_overflow_tokens,
            "served_limit_shortfall": self.served_limit_shortfall,
            "native_window_utilization": self.native_window_utilization,
            "served_window_utilization": self.served_window_utilization,
            "feasible": self.feasible,
            "hot_swap_in": list(self.hot_swap_in),
            "hot_swap_out": list(self.hot_swap_out),
            "reason_codes": list(self.reason_codes),
            "actions": [item.to_dict() for item in self.actions],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "profile_digest": self.profile_digest}


class NativeContextProfiler:
    """Plan and measure a model-native context generation without payload text."""

    def __init__(self, config: NativeContextProfileConfig) -> None:
        if not isinstance(config, NativeContextProfileConfig):
            raise ContractError("config must be a NativeContextProfileConfig")
        self.config = config

    def profile(
        self,
        spans: Iterable[NativeContextSpan],
        *,
        requested_span_ids: Sequence[str] = (),
    ) -> NativeContextProfile:
        copied = [self._copy_span(item) for item in spans]
        self._validate_spans(copied)
        requested = list(requested_span_ids)
        if len(set(requested)) != len(requested) or any(
            not isinstance(item, str) or not item for item in requested
        ):
            raise ContractError("requested_span_ids must be unique non-empty strings")
        known = {item.span_id for item in copied}
        unknown = [item for item in requested if item not in known]
        if unknown:
            raise ContractError("unknown requested span id")
        requested_set = set(requested)

        planned: dict[str, tuple[str, int, str, str]] = {}
        for item in copied:
            if not item.resident:
                if item.span_id in requested_set:
                    planned[item.span_id] = (
                        "thaw",
                        item.token_count,
                        item.content_digest,
                        "requested_exact_span_thawed",
                    )
                else:
                    planned[item.span_id] = (
                        "freeze",
                        0,
                        item.content_digest,
                        "exact_span_remains_frozen",
                    )
            elif item.protected or item.span_id in requested_set:
                planned[item.span_id] = (
                    "retain",
                    item.token_count,
                    item.content_digest,
                    "protected_or_requested_span_retained",
                )
            elif (
                item.summary_tokens is not None
                and item.age_turns >= self.config.stale_after_turns
            ):
                planned[item.span_id] = (
                    "compress",
                    item.summary_tokens,
                    item.summary_digest or item.content_digest,
                    "stale_span_replaced_by_summary",
                )
            elif item.age_turns >= self.config.freeze_after_turns:
                planned[item.span_id] = (
                    "freeze",
                    0,
                    item.content_digest,
                    "cold_exact_span_frozen",
                )
            else:
                planned[item.span_id] = (
                    "retain",
                    item.token_count,
                    item.content_digest,
                    "span_remains_hot",
                )

        active_budget = min(self.config.target_hot_tokens, self.config.served_limit)
        resident_after = sum(value[1] for value in planned.values())
        hot_swap_out: list[str] = []
        if resident_after > active_budget:
            candidates = [
                item
                for item in copied
                if planned[item.span_id][0] == "retain"
                and not item.protected
                and item.span_id not in requested_set
            ]
            candidates.sort(
                key=lambda item: (
                    _PRIORITY_COLDNESS[item.priority],
                    -item.age_turns,
                    item.reference_count,
                    item.start_token,
                    item.span_id,
                )
            )
            for item in candidates:
                if resident_after <= active_budget:
                    break
                resident_after -= item.token_count
                planned[item.span_id] = (
                    "hot_swap_out",
                    0,
                    item.content_digest,
                    "cold_resident_span_swapped_for_pressure",
                )
                hot_swap_out.append(item.span_id)

        previous_active = [
            (item.span_id, item.content_digest, item.token_count)
            for item in copied
            if item.resident
        ]
        next_active = [
            (item.span_id, planned[item.span_id][2], planned[item.span_id][1])
            for item in copied
            if planned[item.span_id][0] not in {"freeze", "hot_swap_out"}
        ]
        stable_prefix_tokens = 0
        for before, after in zip(previous_active, next_active):
            if before != after:
                break
            stable_prefix_tokens += before[2]

        actions = [
            NativeContextAction(
                span_id=item.span_id,
                span_digest=item.content_digest,
                action=planned[item.span_id][0],
                tokens_before=item.token_count if item.resident else 0,
                tokens_after=planned[item.span_id][1],
                reason_code=planned[item.span_id][3],
            )
            for item in copied
        ]
        logical_tokens = copied[-1].end_token if copied else 0
        resident_before = sum(item.token_count for item in copied if item.resident)
        compressed_saved = sum(
            item.token_count - planned[item.span_id][1]
            for item in copied
            if planned[item.span_id][0] == "compress"
        )
        frozen_tokens = sum(
            item.token_count
            for item in copied
            if item.resident and planned[item.span_id][0] in {"freeze", "hot_swap_out"}
        )
        thawed_tokens = sum(
            item.token_count for item in copied if planned[item.span_id][0] == "thaw"
        )
        hot_overflow = max(0, resident_after - self.config.target_hot_tokens)
        served_overflow = max(0, resident_after - self.config.served_limit)
        reason_codes: list[str] = []
        if self.config.served_limit < self.config.model_native_limit:
            reason_codes.append("runtime_serves_less_than_model_native_limit")
        if hot_overflow:
            reason_codes.append("protected_context_exceeds_hot_budget")
        if served_overflow:
            reason_codes.append("planned_context_exceeds_served_limit")
        if compressed_saved:
            reason_codes.append("summary_reduces_native_residency")
        if frozen_tokens:
            reason_codes.append("exact_context_preserved_outside_native_window")
        if thawed_tokens:
            reason_codes.append("exact_context_thawed_into_native_window")
        if hot_swap_out and requested:
            reason_codes.append("native_context_hot_swap_planned")

        bytes_per_token = self.config.runtime_state_bytes_per_token
        return NativeContextProfile(
            model_id=self.config.model_id,
            runtime_id=self.config.runtime_id,
            model_native_limit=self.config.model_native_limit,
            served_limit=self.config.served_limit,
            target_hot_tokens=self.config.target_hot_tokens,
            logical_tokens=logical_tokens,
            resident_tokens_before=resident_before,
            resident_tokens_after=resident_after,
            compressed_tokens_saved=compressed_saved,
            frozen_exact_tokens=frozen_tokens,
            thawed_exact_tokens=thawed_tokens,
            stable_prefix_tokens=stable_prefix_tokens,
            recompute_from_token=stable_prefix_tokens,
            recompute_tokens=max(0, resident_after - stable_prefix_tokens),
            tier_bytes_out=frozen_tokens * bytes_per_token,
            tier_bytes_in=thawed_tokens * bytes_per_token,
            hot_overflow_tokens=hot_overflow,
            served_overflow_tokens=served_overflow,
            served_limit_shortfall=self.config.model_native_limit - self.config.served_limit,
            native_window_utilization=(
                resident_after / self.config.model_native_limit
            ),
            served_window_utilization=(resident_after / self.config.served_limit),
            feasible=hot_overflow == 0 and served_overflow == 0,
            hot_swap_in=[item for item in requested if not next(span.resident for span in copied if span.span_id == item)],
            hot_swap_out=hot_swap_out,
            reason_codes=reason_codes,
            actions=actions,
        )

    @staticmethod
    def _copy_span(value: NativeContextSpan) -> NativeContextSpan:
        if not isinstance(value, NativeContextSpan):
            raise ContractError("spans must contain NativeContextSpan values")
        return NativeContextSpan(**value.__dict__)

    def _validate_spans(self, spans: list[NativeContextSpan]) -> None:
        seen: set[str] = set()
        expected_start = 0
        for item in spans:
            if item.span_id in seen:
                raise ContractError("native context span ids must be unique")
            if item.start_token != expected_start:
                raise ContractError("native context spans must be contiguous and ordered")
            expected_start = item.end_token
            seen.add(item.span_id)
        if expected_start > self.config.model_native_limit:
            raise ContractError("logical context exceeds model-native limit")
