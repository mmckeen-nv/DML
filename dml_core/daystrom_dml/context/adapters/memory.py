"""Narrow memory-fault adapters over existing Daystrom memory components."""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.adapters.stm import STMHotContextAdapter
from daystrom_dml.context.faults import (
    EvidenceAuthority,
    EvidenceHandle,
    MemoryFaultReason,
    MemoryFaultProvenance,
    MemoryFaultRequest,
    MemoryFaultResult,
    MemoryFaultStatus,
    MemorySourceTier,
)
from daystrom_dml.context.paging import ContextPageError, MemoryContextPageCache
from daystrom_dml.stm.schema import STMState


STMStateProvider = Callable[[DaystromScope], Optional[STMState]]


@dataclass
class STMHotMemoryFaultAdapter:
    """DML1 semantic evidence lookup backed by the existing STM hot adapter."""

    state_provider: STMStateProvider
    hot_adapter: STMHotContextAdapter = field(default_factory=STMHotContextAdapter)

    def lookup(self, request: MemoryFaultRequest) -> MemoryFaultResult:
        state = self.state_provider(request.scope)
        if state is None:
            return MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.MISS,
                MemorySourceTier.DML1_HOT,
                [MemoryFaultReason.NOT_FOUND],
            )
        token_budget = request.budget.max_payload_tokens if request.budget.max_payload_tokens > 0 else 256
        segments = self.hot_adapter.render(request.scope, state, budget_tokens=token_budget)
        selected = _first_matching_segment(segments, request.query)
        if selected is None:
            return MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.MISS,
                MemorySourceTier.DML1_HOT,
                [MemoryFaultReason.NOT_FOUND],
            )

        text = str(selected["text"])
        rank_value = selected.get("rank", 0)
        rank = int(rank_value) if isinstance(rank_value, (int, str)) and not isinstance(rank_value, bool) else 0
        metadata_value = selected.get("metadata")
        segment_metadata = metadata_value if isinstance(metadata_value, dict) else {}
        evidence = EvidenceHandle(
            handle_id=str(selected["id"]),
            tier=MemorySourceTier.DML1_HOT,
            scope=request.scope,
            digest=_digest(text.encode("utf-8")),
            authority=EvidenceAuthority.REFERENCE,
            size_bytes=len(text.encode("utf-8")),
            payload_text=text,
            provenance=MemoryFaultProvenance(
                adapter_id="stm_hot_context",
                source_id=str(selected["id"]),
                rank=rank,
                metadata={
                    "checkpoint_digest": segment_metadata.get("checkpoint_digest"),
                    "kind": selected.get("kind"),
                },
            ),
        )
        return MemoryFaultResult.hit(request, MemorySourceTier.DML1_HOT, [evidence])


@dataclass
class DML2ExactPageFaultAdapter:
    """DML2 exact evidence lookup backed by MemoryContextPageCache."""

    page_cache: MemoryContextPageCache

    def lookup(self, request: MemoryFaultRequest) -> MemoryFaultResult:
        if not request.key:
            return MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.MISS,
                MemorySourceTier.DML2_EXACT,
                [MemoryFaultReason.NOT_FOUND],
            )
        try:
            page = self.page_cache.get(request.scope, request.key)
            if page is None:
                return MemoryFaultResult.empty(
                    request,
                    MemoryFaultStatus.MISS,
                    MemorySourceTier.DML2_EXACT,
                    [MemoryFaultReason.NOT_FOUND],
                )
            payload = page.bytes()
        except ContextPageError as exc:
            reason = MemoryFaultReason.DIGEST_MISMATCH if "digest" in str(exc).lower() else MemoryFaultReason.MALFORMED_ADAPTER_RESULT
            return MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.CORRUPT,
                MemorySourceTier.DML2_EXACT,
                [reason],
            )

        if page.content_digest != _digest(payload):
            return MemoryFaultResult.empty(
                request,
                MemoryFaultStatus.CORRUPT,
                MemorySourceTier.DML2_EXACT,
                [MemoryFaultReason.DIGEST_MISMATCH],
            )

        evidence = EvidenceHandle(
            handle_id=page.page_id,
            tier=MemorySourceTier.DML2_EXACT,
            scope=page.scope,
            digest=page.content_digest,
            authority=EvidenceAuthority.UNTRUSTED_DATA,
            size_bytes=len(payload),
            media_type=page.media_type,
            payload_text=page.text(),
            payload_bytes_b64=None if page.text() is not None else base64.b64encode(payload).decode("ascii"),
            provenance=MemoryFaultProvenance(
                adapter_id="memory_context_page_cache",
                source_id=page.source_segment_id,
                page_id=page.page_id,
                metadata={
                    "content_type": page.content_type,
                    "sensitivity_label": page.sensitivity_label,
                },
            ),
        )
        return MemoryFaultResult.hit(request, MemorySourceTier.DML2_EXACT, [evidence])


def _first_matching_segment(segments: list[dict[str, object]], query: Optional[str]) -> Optional[dict[str, object]]:
    if not segments:
        return None
    if not query:
        return segments[0]
    terms = {term.lower() for term in query.split() if term.strip()}
    if not terms:
        return segments[0]
    for segment in segments:
        text = str(segment.get("text", "")).lower()
        if all(term in text for term in terms):
            return segment
    return None


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()
