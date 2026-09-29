"""Context selection within caller-owned retrieval state.

The caller pins store ownership and the effective time before entering these
helpers. Collection membership is copied, but ``MemoryItem`` records remain
borrowed: the caller must keep them stable until context assembly is complete.
Semantic ranking stays with the public store capability, including its score
weights, similarity threshold, deterministic ties, and optional retrieval modes.
This module performs no embedding, storage I/O, clock reads, or locking.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from ..memory_store import MemoryItem
from .lifecycle import suppression_reason


Scope = tuple[str | None, str | None, str | None, str | None]
SCOPE_KEYS = ("tenant_id", "client_id", "session_id", "instance_id")


@dataclass(frozen=True)
class ScopedRetrievalRequest:
    """Resolved adapter policy, with collection inputs defensively copied.

    The adapter owns phase coercion, finite-time validation, and top-k resolution.
    Freezing this request does not freeze the borrowed memory records.
    """

    scope: Scope
    kinds: tuple[str, ...] | None
    phase: str | None
    top_k: int
    as_of: float
    include_quarantined: bool = False

    def __post_init__(self) -> None:
        scope = tuple(self.scope)
        if len(scope) != len(SCOPE_KEYS):
            raise ValueError("Retrieval scope must contain four fields")
        object.__setattr__(self, "scope", scope)
        if self.kinds is not None:
            object.__setattr__(self, "kinds", tuple(self.kinds))


class FilteredRetriever(Protocol):
    """The public ``MemoryStore.retrieve_filtered`` capability."""

    def __call__(
        self,
        query_embedding: np.ndarray,
        *,
        tenant_id: str | None,
        client_id: str | None,
        session_id: str | None,
        instance_id: str | None,
        kinds: tuple[str, ...] | None,
        top_k: int,
        strict_scope: bool,
        as_of: float,
        eligible: Callable[[MemoryItem], bool],
    ) -> list[MemoryItem]: ...


def _matches_scope(meta: Mapping, scope: Scope) -> bool:
    return all(meta.get(key) == value for key, value in zip(SCOPE_KEYS, scope))


def select_scoped_context(
    *,
    retrieve_filtered: FilteredRetriever,
    recent_candidates: Callable[[], Sequence[MemoryItem]],
    query_embedding: np.ndarray,
    request: ScopedRetrievalRequest,
) -> list[MemoryItem]:
    """Select strict-scope context, with recent fallback on an empty success.

    Both read capabilities must refer to the same owned store state, with the
    caller retaining ownership across both calls. The recent candidate read is
    performed only after an empty successful ranked read. Suppression is supplied
    as candidate eligibility before top-k.
    Preserve the existing distinction: semantic selection uses resolved kinds,
    while the recent fallback additionally filters the requested phase. Reader
    failures propagate; they never become successful recent-memory responses.
    """
    tenant_id, client_id, session_id, instance_id = request.scope
    items = retrieve_filtered(
        query_embedding,
        tenant_id=tenant_id,
        client_id=client_id,
        session_id=session_id,
        instance_id=instance_id,
        kinds=request.kinds,
        top_k=request.top_k,
        strict_scope=True,
        as_of=request.as_of,
        eligible=lambda item: suppression_reason(
            item.meta or {}, now=request.as_of,
            include_quarantined=request.include_quarantined,
        ) is None,
    )
    if items:
        return list(items)
    return recent_context_items(recent_candidates(), request=request)


def recent_context_items(
    candidates: Sequence[MemoryItem],
    *,
    request: ScopedRetrievalRequest,
    require_unscoped: bool = False,
) -> list[MemoryItem]:
    """Select recent eligible context with timestamp-descending, ID-ascending ties.

    An entirely omitted scope retains legacy global fallback behavior unless
    ``require_unscoped`` is requested. Empty kinds retain their existing fallback
    meaning: no kind restriction, except for execute/debug phase defaults.
    """
    allowed_kinds = set(request.kinds or ())
    if not allowed_kinds and request.phase in {"execute", "debug"}:
        allowed_kinds = {"action", "observation", "error"}
    scoped = any(value is not None for value in request.scope)
    selected = []
    for item in tuple(candidates):
        meta = item.meta or {}
        if suppression_reason(
            meta, now=request.as_of,
            include_quarantined=request.include_quarantined,
        ):
            continue
        if require_unscoped and any(meta.get(key) is not None for key in SCOPE_KEYS):
            continue
        if scoped and not _matches_scope(meta, request.scope):
            continue
        item_phase = meta.get("phase")
        if (request.phase is not None and item_phase is not None
                and str(item_phase).strip().lower() != request.phase):
            continue
        item_kind = str(meta.get("kind") or "memory").lower()
        if allowed_kinds and item_kind not in allowed_kinds:
            continue
        selected.append(item)
    return heapq.nlargest(
        max(1, request.top_k), selected,
        key=lambda item: (item.timestamp, -item.id),
    )


def suppressed_context_items(
    candidates: Sequence[MemoryItem], *, request: ScopedRetrievalRequest,
) -> list[dict]:
    """Return payload-free suppression evidence in original candidate order.

    Evidence covers the requested scope regardless of kind, phase, or rank.
    An entirely omitted scope retains the legacy global evidence behavior.
    """
    scoped = any(value is not None for value in request.scope)
    suppressed = []
    for item in tuple(candidates):
        meta = item.meta or {}
        if scoped and not _matches_scope(meta, request.scope):
            continue
        reason = suppression_reason(
            meta, now=request.as_of,
            include_quarantined=request.include_quarantined,
        )
        if reason:
            suppressed.append({"id": str(item.id), "reason": reason})
    return suppressed


def survival_ledger_for_scope(
    candidates: Sequence[MemoryItem], *, scope: Scope, enabled: bool,
) -> MemoryItem | None:
    """Return the latest ledger; equal timestamps retain source-order priority.

    Ledger scope comparison deliberately preserves string conversion of present
    fields. The caller applies lifecycle suppression after selecting this latest
    ledger; suppression does not trigger an older-ledger fallback in this lookup.
    """
    scope_values = tuple(scope)
    if len(scope_values) != len(SCOPE_KEYS):
        raise ValueError("Retrieval scope must contain four fields")
    if not enabled or not scope_values[0] or not scope_values[2]:
        return None
    target = tuple(str(value) if value is not None else None for value in scope_values)
    matching = []
    for item in tuple(candidates):
        meta = item.meta or {}
        if meta.get("kind") != "survival_ledger":
            continue
        item_scope = tuple(
            str(meta[key]) if meta.get(key) is not None else None
            for key in SCOPE_KEYS
        )
        if item_scope == target:
            matching.append(item)
    return max(matching, key=lambda item: item.timestamp, default=None)
