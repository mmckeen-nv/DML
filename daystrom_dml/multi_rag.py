"""Aggregation helpers for comparing multiple RAG backends."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .rag_store import SimpleRAGStore


def _identity(score: float, _: Dict[str, Any]) -> float:
    return score


def _favor_short(score: float, doc: Dict[str, Any]) -> float:
    tokens = max(1, int(doc.get("tokens") or 1))
    return score * (1.0 + min(0.15, 12.0 / math.sqrt(tokens)))


def _favor_long(score: float, doc: Dict[str, Any]) -> float:
    tokens = max(1, int(doc.get("tokens") or 1))
    boost = min(0.2, math.log1p(tokens) / 25.0)
    return score * (1.0 + boost)


def _add_bias(score: float, _: Dict[str, Any], bias: float) -> float:
    return score + bias


def _bias_factory(bias: float):
    return lambda score, doc: _add_bias(score, doc, bias)


@dataclass(frozen=True)
class RAGBackendDescriptor:
    """Lightweight description of a simulated RAG backend."""

    identifier: str
    label: str
    score_transform: Any
    description: str = ""


DEFAULT_BACKENDS: List[RAGBackendDescriptor] = [
    RAGBackendDescriptor(
        "chroma",
        "Chroma (Local)",
        _identity,
        "Baseline cosine similarity with no scoring adjustments.",
    ),
    RAGBackendDescriptor(
        "milvus",
        "Milvus (HNSW)",
        _favor_long,
        "Boosts longer passages to highlight dense vector recall.",
    ),
    RAGBackendDescriptor(
        "pinecone",
        "Pinecone (Managed)",
        _bias_factory(0.015),
        "Applies a constant bias to simulate tuned production weights.",
    ),
    RAGBackendDescriptor(
        "weaviate",
        "Weaviate (Hybrid)",
        _favor_short,
        "Rewards concise chunks similar to BM25 hybrid recall.",
    ),
    RAGBackendDescriptor(
        "qdrant",
        "Qdrant (Filtered)",
        _bias_factory(0.005),
        "Adds a slight boost for matches passing metadata filters.",
    ),
    RAGBackendDescriptor(
        "faiss",
        "Faiss (Library)",
        _favor_long,
        "Prefers longer vectors to emulate IVFPQ recall heuristics.",
    ),
    RAGBackendDescriptor(
        "annoy",
        "Annoy (In-memory)",
        _bias_factory(-0.01),
        "Downranks lower-scoring hits to mimic approximate neighbours.",
    ),
]


class MultiRAGStore:
    """Fan out incoming documents and retrievals to multiple backends."""

    def __init__(
        self,
        embedder,
        backends: Optional[Iterable[RAGBackendDescriptor]] = None,
    ) -> None:
        self.embedder = embedder
        self._descriptors: List[RAGBackendDescriptor] = list(backends or DEFAULT_BACKENDS)
        self._backends: List[Dict[str, Any]] = []
        for descriptor in self._descriptors:
            store = SimpleRAGStore(self.embedder, score_transform=descriptor.score_transform)
            self._backends.append(
                {
                    "id": descriptor.identifier,
                    "label": descriptor.label,
                    "store": store,
                    "descriptor": descriptor,
                }
            )
        self._raw_documents: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return
        payload = {"text": text, "meta": meta or {}}
        self._raw_documents.append(payload)
        for backend in self._backends:
            backend["store"].add_document(text, meta=meta)

    def clear(self) -> None:
        self._raw_documents.clear()
        for backend in self._backends:
            backend["store"].clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report_all(self, prompt: str, *, top_k: int = 4) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        for backend in self._backends:
            store: SimpleRAGStore = backend["store"]
            report = store.report(prompt, top_k=top_k)
            reports.append(
                {
                    "id": backend["id"],
                    "label": backend["label"],
                    "strategy": backend["descriptor"].description,
                    **report,
                }
            )
        return reports

    def primary_report(self, prompt: str, *, top_k: int = 4) -> Optional[Dict[str, Any]]:
        reports = self.report_all(prompt, top_k=top_k)
        return reports[0] if reports else None

    def catalog_summary(self) -> Dict[str, Any]:
        if not self._backends:
            return {"documents": [], "total_tokens": 0, "count": 0, "backends": []}
        # All backends share the same source documents; use the first as canonical.
        primary_store: SimpleRAGStore = self._backends[0]["store"]
        catalog = primary_store.catalog()
        catalog["backends"] = [
            {
                "id": backend["id"],
                "label": backend["label"],
                "strategy": backend["descriptor"].description,
            }
            for backend in self._backends
        ]
        return catalog

    def export_state(self) -> Dict[str, Any]:
        return {"documents": list(self._raw_documents)}

    def import_state(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        documents = payload.get("documents") or []
        self.clear()
        for entry in documents:
            text = entry.get("text")
            if not text:
                continue
            meta = entry.get("meta") or {}
            self.add_document(str(text), meta=meta)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def descriptors(self) -> List[Dict[str, str]]:
        return [
            {
                "id": backend["id"],
                "label": backend["label"],
                "strategy": backend["descriptor"].description,
            }
            for backend in self._backends
        ]

