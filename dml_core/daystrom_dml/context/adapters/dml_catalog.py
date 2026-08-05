"""Strict DML-lattice adapter for bounded semantic context-page retrieval."""
from __future__ import annotations

import hashlib
import math
import time
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from daystrom_dml import utils
from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context.catalog import PageCatalogQuery, PageCatalogResult
from daystrom_dml.context.schema import ContextAuthority, ContextPriority, ContextSegment


class DMLSemanticPageCatalog:
    """Read context pages directly from a DML lattice without scope broadening.

    The supplied adapter is intentionally duck typed: it must expose ``embedder``
    and ``store`` surfaces compatible with :class:`DMLAdapter`. This keeps the
    page-catalog contract provider neutral while reusing DML's real embedding and
    lattice ranking machinery.
    """

    def __init__(self, dml_adapter: Any, *, clock: Any = None) -> None:
        if not hasattr(dml_adapter, "embedder") or not hasattr(dml_adapter, "store"):
            raise ContractError("dml_adapter must expose embedder and store")
        self._adapter = dml_adapter
        self._clock = clock or time.time

    def lookup(self, query: PageCatalogQuery) -> PageCatalogResult:
        if not isinstance(query, PageCatalogQuery):
            raise ContractError("query must be a PageCatalogQuery")
        query_embedding: Optional[np.ndarray] = None
        if query.query.strip():
            query_embedding = np.asarray(self._adapter.embedder.embed(query.query), dtype=np.float32)
            if query_embedding.ndim != 1 or query_embedding.size < 1 or not np.all(np.isfinite(query_embedding)):
                raise ContractError("DML page-catalog query embedding is invalid")

        exact_items = self._find_exact(query)
        exact_candidates, exact_scope_rejected, exact_payload_rejected = self._normalize_items(
            query, exact_items, query_embedding, exact_handles=set(query.exact_handles)
        )
        remaining = max(0, query.max_candidates - len(exact_candidates))
        semantic_items: List[Any] = []
        if remaining and query_embedding is not None:
            semantic_items = self._retrieve_semantic(query, query_embedding, remaining)
        semantic_candidates, semantic_scope_rejected, semantic_payload_rejected = self._normalize_items(
            query, semantic_items, query_embedding, exact_handles=set(query.exact_handles)
        )

        deduplicated: Dict[str, Dict[str, Any]] = {}
        for candidate in [*exact_candidates, *semantic_candidates]:
            current = deduplicated.get(candidate["page_id"])
            if current is None or candidate["rank_key"] < current["rank_key"]:
                deduplicated[candidate["page_id"]] = candidate
        ranked = sorted(deduplicated.values(), key=lambda item: item["rank_key"])
        if len(ranked) > query.max_candidates:
            ranked = ranked[: query.max_candidates]

        segments: List[ContextSegment] = []
        used_bytes = 0
        used_tokens = 0
        payload_omitted = exact_payload_rejected + semantic_payload_rejected
        for candidate in ranked:
            payload_bytes = len(candidate["text"].encode("utf-8"))
            payload_tokens = int(candidate["tokens"])
            if used_bytes + payload_bytes > query.max_payload_bytes or used_tokens + payload_tokens > query.max_payload_tokens:
                if candidate["exact"]:
                    raise ContractError("exact DML page exceeds the catalog payload budget")
                payload_omitted += 1
                continue
            segments.append(self._to_segment(query.scope, candidate))
            used_bytes += payload_bytes
            used_tokens += payload_tokens

        return PageCatalogResult(
            scope=query.scope,
            segments=segments,
            telemetry={
                "version": "dml-lattice-page-catalog-v1",
                "requested_candidates": query.max_candidates,
                "exact_handles": len(query.exact_handles),
                "materialized_candidates": len(exact_items) + len(semantic_items),
                "returned_candidates": len(segments),
                "scope_rejected": exact_scope_rejected + semantic_scope_rejected,
                "payload_bytes": used_bytes,
                "payload_tokens": used_tokens,
                "payload_omitted": payload_omitted,
            },
        )

    def _find_exact(self, query: PageCatalogQuery) -> List[Any]:
        if not query.exact_handles:
            return []
        finder = getattr(self._adapter.store, "find_filtered_by_handles", None)
        if not callable(finder):
            raise ContractError("DML store lacks bounded exact-handle lookup")
        raw = finder(
            handles=list(query.exact_handles),
            tenant_id=query.scope.tenant_id,
            client_id=query.scope.client_id,
            session_id=query.scope.session_id,
            instance_id=query.scope.instance_id,
            thread_id=query.scope.thread_id,
            project_id=query.scope.project_id,
            relationship_id=query.scope.relationship_id,
            limit=len(query.exact_handles),
        )
        return self._bounded_items(raw, len(query.exact_handles), "exact")

    def _retrieve_semantic(self, query: PageCatalogQuery, embedding: np.ndarray, limit: int) -> List[Any]:
        retriever = getattr(self._adapter.store, "retrieve_filtered_for_catalog", None)
        if not callable(retriever):
            raise ContractError("DML store lacks bounded read-only semantic lookup")
        raw = retriever(
            embedding,
            tenant_id=query.scope.tenant_id,
            client_id=query.scope.client_id,
            session_id=query.scope.session_id,
            instance_id=query.scope.instance_id,
            thread_id=query.scope.thread_id,
            project_id=query.scope.project_id,
            relationship_id=query.scope.relationship_id,
            top_k=limit,
        )
        return self._bounded_items(raw, limit, "semantic")

    @staticmethod
    def _bounded_items(raw: Any, limit: int, label: str) -> List[Any]:
        try:
            items = list(islice(iter(raw), limit + 1))
        except TypeError as exc:
            raise ContractError(f"DML {label} page lookup returned a non-iterable result") from exc
        if len(items) > limit:
            raise ContractError(f"DML {label} page lookup exceeded its candidate bound")
        return items

    def _normalize_items(
        self,
        query: PageCatalogQuery,
        items: Iterable[Any],
        query_embedding: Optional[np.ndarray],
        *,
        exact_handles: set[str],
    ) -> tuple[List[Dict[str, Any]], int, int]:
        normalized: List[Dict[str, Any]] = []
        scope_rejected = 0
        payload_rejected = 0
        now = float(self._clock())
        if not math.isfinite(now) or now < 0:
            raise ContractError("DML page-catalog clock returned an invalid timestamp")
        exact_order = {value: index for index, value in enumerate(query.exact_handles)}
        causal_anchors = set(query.causal_anchor_ids)
        for item in items:
            raw_meta = getattr(item, "meta", None) or {}
            if not isinstance(raw_meta, dict):
                raise ContractError("DML page metadata must be a dictionary")
            meta = raw_meta
            if not _matches_full_scope(query.scope, meta):
                scope_rejected += 1
                continue
            text = str(getattr(item, "text", ""))
            if not text:
                continue
            page_id = _page_id(item, meta)
            item_handles = [page_id, f"dml:{getattr(item, 'id', 'unknown')}"]
            item_handles.extend(
                str(meta[key]).strip()
                for key in ("context_page_id", "page_id", "handle_id")
                if meta.get(key) is not None and str(meta.get(key)).strip()
            )
            matching_exact = exact_handles.intersection(item_handles)
            matched_exact = next((handle for handle in query.exact_handles if handle in matching_exact), None)
            exact = matched_exact is not None
            payload = text.encode("utf-8")
            tokens = max(1, int(utils.estimate_tokens(text)))
            if len(payload) > query.max_payload_bytes or tokens > query.max_payload_tokens:
                if exact:
                    raise ContractError("exact DML page exceeds the catalog payload budget")
                payload_rejected += 1
                continue
            digest = _payload_digest(payload)
            declared_digest = meta.get("payload_digest") or meta.get("content_digest")
            if declared_digest is not None and str(declared_digest).startswith("sha256:") and declared_digest != digest:
                raise ContractError("DML page payload digest mismatch")
            semantic_score = 0.0
            if query_embedding is not None:
                embedding = np.asarray(getattr(item, "embedding", []), dtype=np.float32)
                if embedding.ndim != 1 or embedding.size != query_embedding.size or not np.all(np.isfinite(embedding)):
                    raise ContractError("DML page embedding is invalid")
                semantic_score = float(max(-1.0, min(1.0, utils.cosine_similarity(embedding, query_embedding))))
            try:
                timestamp = float(getattr(item, "timestamp", 0.0))
            except (TypeError, ValueError) as exc:
                raise ContractError("DML page timestamp is invalid") from exc
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ContractError("DML page timestamp is invalid")
            age_hours = max(0.0, (now - timestamp) / 3600.0)
            recency_score = 1.0 / (1.0 + age_hours)
            raw_neighbors = meta.get("lattice_neighbors") or []
            neighbor_ids = (
                [str(value)[:128] for value in raw_neighbors[:64]]
                if isinstance(raw_neighbors, (list, tuple))
                else []
            )
            causal_score = (
                len(causal_anchors.intersection(neighbor_ids)) / max(1, len(causal_anchors)) if causal_anchors else 0.0
            )
            fidelity = _unit_float(getattr(item, "fidelity", 0.0))
            salience = _unit_float(getattr(item, "salience", 0.0))
            combined_score = (
                (2.0 if exact else 0.0)
                + 0.55 * max(0.0, semantic_score)
                + 0.20 * causal_score
                + 0.10 * recency_score
                + 0.10 * fidelity
                + 0.05 * salience
            )
            exact_rank = exact_order[matched_exact] if matched_exact is not None else len(exact_order)
            normalized.append(
                {
                    "page_id": page_id,
                    "source_id": _bounded_text(getattr(item, "id", page_id), 256),
                    "text": text,
                    "digest": digest,
                    "tokens": tokens,
                    "exact": exact,
                    "semantic_score": semantic_score,
                    "causal_score": causal_score,
                    "recency_score": recency_score,
                    "combined_score": combined_score,
                    "rank_key": (0 if exact else 1, exact_rank if exact else 0, -combined_score, page_id),
                    "dml": _safe_dml_provenance(item, meta, neighbor_ids),
                }
            )
        return normalized, scope_rejected, payload_rejected

    @staticmethod
    def _to_segment(scope: DaystromScope, candidate: Dict[str, Any]) -> ContextSegment:
        return ContextSegment(
            segment_id=f"dml-page:{candidate['page_id']}",
            kind="dml-context-page",
            content=candidate["text"],
            authority=ContextAuthority.UNTRUSTED_DATA,
            priority=ContextPriority.REFERENCE,
            scope=DaystromScope.from_dict(scope.to_dict()),
            source={
                "adapter_id": "dml_semantic_page_catalog",
                "page_id": candidate["page_id"],
                "source_id": candidate["source_id"],
                "payload_digest": candidate["digest"],
            },
            provenance={
                "catalog": {
                    "exact_handle": candidate["exact"],
                    "semantic_score": candidate["semantic_score"],
                    "causal_score": candidate["causal_score"],
                    "recency_score": candidate["recency_score"],
                    "combined_score": candidate["combined_score"],
                },
                "dml": candidate["dml"],
            },
            cache={"payload_digest": candidate["digest"], "reusable": False},
            retention={"source": "durable_semantic_page"},
            estimated_tokens=candidate["tokens"],
        )


def _matches_full_scope(scope: DaystromScope, meta: Dict[str, Any]) -> bool:
    for name in (
        "tenant_id",
        "client_id",
        "session_id",
        "instance_id",
        "thread_id",
        "project_id",
        "relationship_id",
    ):
        expected = getattr(scope, name)
        if expected is not None and meta.get(name) != expected:
            return False
    return True


def _page_id(item: Any, meta: Dict[str, Any]) -> str:
    for key in ("context_page_id", "page_id", "handle_id"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            result = str(value).strip()
            if len(result) > 1024:
                raise ContractError("DML page identifier exceeds its bound")
            return result
    result = f"dml:{getattr(item, 'id', 'unknown')}"
    if len(result) > 1024:
        raise ContractError("DML page identifier exceeds its bound")
    return result


def _payload_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _unit_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit]


def _safe_dml_provenance(item: Any, meta: Dict[str, Any], neighbor_ids: List[str]) -> Dict[str, Any]:
    try:
        level = int(getattr(item, "level", 0))
    except (TypeError, ValueError):
        level = 0
    result: Dict[str, Any] = {
        "item_id": _bounded_text(getattr(item, "id", "unknown"), 256),
        "kind": _bounded_text(meta.get("kind") or "memory", 128),
        "level": level,
        "fidelity": _unit_float(getattr(item, "fidelity", 0.0)),
        "salience": _unit_float(getattr(item, "salience", 0.0)),
        "lattice_neighbors": neighbor_ids[:8],
    }
    for key in ("source", "lattice_policy"):
        value = meta.get(key)
        if value is not None:
            result[key] = _bounded_text(value, 512 if key == "source" else 128)
    for key in ("lattice_row", "lattice_col", "lattice_layer"):
        value = meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    return result
