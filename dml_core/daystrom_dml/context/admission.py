"""Deterministic active admission for context-window mutation."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from daystrom_dml.api_contracts import AuditInfo, ContractError, DaystromScope
from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.adapters.base import RuntimeContextAdapter
from daystrom_dml.context.budget import ContextBudget
from daystrom_dml.context.capabilities import RuntimeCapabilities
from daystrom_dml.context.manifest import ContextManifest, ContextPacket
from daystrom_dml.context.probe import endpoint_identity_digest
from daystrom_dml.context.schema import ContextAuthority, ContextPriority, ContextSegment, context_segment_digest

OBSERVE_ONLY_MODE = "observe_only"
ACTIVE_ADMISSION_MODE = "active_admission"
ADMISSION_MODES = {OBSERVE_ONLY_MODE, ACTIVE_ADMISSION_MODE}

PageOutCallback = Callable[[DaystromScope, ContextSegment], Any]

_AUTHORITY_RANK = {
    ContextAuthority.TRUSTED_CONTROL: 0,
    ContextAuthority.REFERENCE: 1,
    ContextAuthority.UNTRUSTED_DATA: 2,
}
_PRIORITY_RANK = {
    ContextPriority.CRITICAL: 0,
    ContextPriority.WORKING: 1,
    ContextPriority.REFERENCE: 2,
    ContextPriority.DISPOSABLE: 3,
}
_PINNED_REASONS = {
    ContextAuthority.IMMUTABLE: "pinned_immutable",
    ContextAuthority.CURRENT_INSTRUCTION: "pinned_current_instruction",
}


def validate_admission_mode(mode: str) -> str:
    """Return a known admission mode or fail closed."""

    if mode not in ADMISSION_MODES:
        raise ContractError(f"mode must be one of {sorted(ADMISSION_MODES)}")
    return mode


def admit_context_segments(
    *,
    scope: DaystromScope | Dict[str, Any] | None = None,
    segments: Iterable[ContextSegment | Dict[str, Any]],
    model_id: str,
    runtime_id: str,
    endpoint_url: Optional[str] = None,
    model_limit_tokens: int,
    output_reserved_tokens: int = 0,
    runtime_reserved_tokens: int = 0,
    runtime_adapter: Optional[RuntimeContextAdapter] = None,
    capabilities: Optional[RuntimeCapabilities] = None,
    page_out: Optional[PageOutCallback] = None,
) -> ContextPacket:
    """Build an active context packet by deterministic bounded admission.

    The caller must explicitly invoke this function or use a controller in
    active_admission mode. It never retrieves, writes back, or increases the
    supplied model limit.
    """

    scope_obj = scope if isinstance(scope, DaystromScope) else DaystromScope.from_dict(scope)
    if endpoint_url is not None and (not isinstance(endpoint_url, str) or not endpoint_url):
        raise ContractError("endpoint_url must be a non-empty string when provided")
    adapter = runtime_adapter or APIMessageAdapter()
    normalized = [
        ContextSegment.from_dict(json.loads(json.dumps(segment.to_dict(), sort_keys=True)))
        if isinstance(segment, ContextSegment)
        else ContextSegment.from_dict(json.loads(json.dumps(segment, sort_keys=True)))
        for segment in segments
    ]
    _validate_unique_segment_ids(normalized)
    _validate_segment_scopes(scope_obj, normalized)

    budget = ContextBudget(
        model_limit_tokens=model_limit_tokens,
        output_reserved_tokens=output_reserved_tokens,
        runtime_reserved_tokens=runtime_reserved_tokens,
    )
    indexed = list(enumerate(normalized))
    pinned = [(index, segment) for index, segment in indexed if segment.authority in _PINNED_REASONS]
    pinned_tokens = sum(segment.effective_tokens for _, segment in pinned)
    if pinned_tokens > budget.available_input_tokens:
        raise ContractError("pinned segments exceed available input budget")

    decisions: Dict[str, Any] = {"by_segment": {}, "admitted": [], "omitted": []}
    admitted: List[Tuple[int, ContextSegment]] = []
    for index, segment in pinned:
        _admit_or_raise(budget, segment)
        admitted.append((index, segment))
        _record_decision(
            decisions,
            segment,
            original_index=index,
            admitted=True,
            reason_code=_PINNED_REASONS[segment.authority],
        )

    pinned_indexes = {index for index, _ in pinned}
    optional = [(index, segment) for index, segment in indexed if index not in pinned_indexes]
    for index, segment in sorted(optional, key=_optional_rank):
        if budget.admit(segment.effective_tokens):
            admitted.append((index, segment))
            _record_decision(
                decisions,
                segment,
                original_index=index,
                admitted=True,
                reason_code="admitted_optional",
            )
        else:
            _record_decision(
                decisions,
                segment,
                original_index=index,
                admitted=False,
                reason_code="omitted_budget_exhausted",
            )
            _page_out_omitted(decisions, scope_obj, segment, page_out)

    admitted_segments = [segment for _, segment in sorted(admitted, key=lambda item: item[0])]
    rendered = adapter.render_messages([_renderable_segment(segment) for segment in admitted_segments])
    if capabilities is not None:
        capability_payload = capabilities.to_dict()
        metadata = dict(capability_payload.get("metadata") or {})
        if endpoint_url is not None:
            metadata["endpoint_url_digest"] = endpoint_identity_digest(endpoint_url)
        capability_payload["metadata"] = metadata
        capability_obj = RuntimeCapabilities.from_dict(capability_payload)
    else:
        capability_obj = _capabilities_from_adapter(
            adapter,
            model_id=model_id,
            runtime_id=runtime_id,
            model_limit_tokens=model_limit_tokens,
            output_reserved_tokens=output_reserved_tokens,
            endpoint_url=endpoint_url,
        )
    manifest = ContextManifest(
        scope=scope_obj,
        model_id=capability_obj.model_id,
        runtime_id=capability_obj.backend_id,
        segment_ids=[segment.segment_id for segment in admitted_segments],
        segment_digests={segment.segment_id: context_segment_digest(segment) for segment in admitted_segments},
        estimated_input_tokens=budget.admitted_input_tokens,
        exact_input_tokens=_exact_input_tokens(admitted_segments),
        decisions=decisions,
        audit={"reason": "active deterministic context admission", "reason_codes": _reason_codes(decisions)},
        created_at=0.0,
    )
    return ContextPacket(
        scope=scope_obj,
        capabilities=capability_obj,
        budget=budget,
        segments=admitted_segments,
        manifest=manifest,
        rendered_messages=rendered["messages"],
        decisions=decisions,
        audit=AuditInfo(
            reason="active deterministic context admission",
            policy=ACTIVE_ADMISSION_MODE,
            reason_codes=_reason_codes(decisions),
        ),
    )


def _validate_segment_scopes(scope: DaystromScope, segments: List[ContextSegment]) -> None:
    for segment in segments:
        if segment.scope != scope:
            raise ContractError("segment scopes must match admission scope")


def _validate_unique_segment_ids(segments: List[ContextSegment]) -> None:
    seen = set()
    for segment in segments:
        if segment.segment_id in seen:
            raise ContractError("segment_id values must be unique for admission")
        seen.add(segment.segment_id)


def _optional_rank(item: Tuple[int, ContextSegment]) -> Tuple[int, int, int]:
    index, segment = item
    return (_AUTHORITY_RANK[segment.authority], _PRIORITY_RANK[segment.priority], index)


def _admit_or_raise(budget: ContextBudget, segment: ContextSegment) -> None:
    if not budget.admit(segment.effective_tokens):  # pragma: no cover - guarded by pinned preflight
        raise ContractError("pinned segments exceed available input budget")


def _record_decision(
    decisions: Dict[str, Any],
    segment: ContextSegment,
    *,
    original_index: int,
    admitted: bool,
    reason_code: str,
) -> None:
    decisions["by_segment"][segment.segment_id] = {
        "segment_id": segment.segment_id,
        "original_index": original_index,
        "authority": segment.authority.value,
        "priority": segment.priority.value,
        "tokens": segment.effective_tokens,
        "admitted": admitted,
        "reason_code": reason_code,
    }
    decisions["admitted" if admitted else "omitted"].append(segment.segment_id)


def _page_out_omitted(
    decisions: Dict[str, Any],
    scope: DaystromScope,
    segment: ContextSegment,
    page_out: Optional[PageOutCallback],
) -> None:
    if page_out is None:
        return
    decision = decisions["by_segment"][segment.segment_id]
    try:
        handle = page_out(scope, segment)
        decision["page_out"] = {"status": "stored", "handle": _sanitize_page_handle(handle)}
    except Exception as exc:  # pragma: no cover - exact exception type is callback-defined
        decision["page_out"] = {"status": "failed", "error_type": type(exc).__name__}


def _sanitize_page_handle(handle: Any) -> Dict[str, Any]:
    if not isinstance(handle, Mapping):
        raise ContractError("page_out handle must be an object")
    allowed = ("page_id", "handle_id", "digest", "tier", "source_segment_id", "expires_at", "size_bytes")
    safe: Dict[str, Any] = {}
    for key in allowed:
        value = handle.get(key)
        if value is None:
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ContractError("page_out handle fields must be scalar")
        if isinstance(value, str) and len(value) > 512:
            raise ContractError("page_out handle fields are too long")
        safe[key] = value
    if not safe.get("page_id") and not safe.get("handle_id"):
        raise ContractError("page_out handle requires page_id or handle_id")
    return safe


def _renderable_segment(segment: ContextSegment) -> Dict[str, Any]:
    return {
        "id": segment.segment_id,
        "kind": segment.kind,
        "role": "system" if segment.authority == ContextAuthority.IMMUTABLE else "user",
        "content": segment.content,
        "source": segment.source,
        "metadata": {
            "authority": segment.authority.value,
            "priority": segment.priority.value,
        },
    }


def _capabilities_from_adapter(
    adapter: RuntimeContextAdapter,
    *,
    model_id: str,
    runtime_id: str,
    model_limit_tokens: int,
    output_reserved_tokens: int,
    endpoint_url: Optional[str],
) -> RuntimeCapabilities:
    metadata: Dict[str, Any] = {"adapter": adapter.capabilities()}
    if endpoint_url is not None:
        metadata["endpoint_url_digest"] = endpoint_identity_digest(endpoint_url)
    return RuntimeCapabilities.api_only(
        model_id=model_id,
        backend_id=runtime_id,
        model_context_window=model_limit_tokens,
        max_output_tokens=output_reserved_tokens,
        metadata=metadata,
    )


def _exact_input_tokens(segments: List[ContextSegment]) -> Optional[int]:
    if not segments or any(segment.exact_tokens is None for segment in segments):
        return None
    return sum(segment.exact_tokens or 0 for segment in segments)


def _reason_codes(decisions: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    for segment_id in decisions.get("admitted", []) + decisions.get("omitted", []):
        code = decisions["by_segment"][segment_id]["reason_code"]
        if code not in codes:
            codes.append(code)
        page_out = decisions["by_segment"][segment_id].get("page_out")
        if page_out and page_out["status"] == "failed" and "page_out_failed" not in codes:
            codes.append("page_out_failed")
    return codes
