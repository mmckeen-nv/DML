"""Disk-backed vector index for lightweight RAG persistence."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import utils
from .atomic_io import atomic_write_text
from .multi_rag import RAGBackendProtocol
from .vector_backend import get_vector_backend


class PersistentVectorIndex:
    """Store embeddings and metadata on disk for reproducible retrieval."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._vectors: List[np.ndarray] = []
        self._payloads: List[Dict[str, Any]] = []
        self._backend = get_vector_backend()
        self._matrix: Optional[np.ndarray] = None
        self._load()

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return
            embeddings = raw.get("embeddings") or []
            payloads = raw.get("payloads") or []
            if len(embeddings) != len(payloads):
                return
            self._vectors = [
                np.asarray(vector, dtype=np.float32)
                for vector in embeddings
            ]
            self._payloads = [dict(entry) for entry in payloads]

    def _flush(self) -> None:
        data = {
            "embeddings": [vector.tolist() for vector in self._vectors],
            "payloads": self._payloads,
        }
        atomic_write_text(self.path, json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, embedding: np.ndarray, payload: Dict[str, Any]) -> None:
        self.extend([embedding], [payload])

    def extend(self, embeddings: Iterable[np.ndarray], payloads: Iterable[Dict[str, Any]]) -> None:
        vectors, entries = [], []
        from itertools import zip_longest
        missing = object()
        for embedding, payload in zip_longest(embeddings, payloads, fillvalue=missing):
            if embedding is missing or payload is missing:
                raise ValueError("embeddings and payloads must have equal lengths")
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.size == 0:
                continue
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError("embedding must be a finite vector")
            entry = {
                "text": payload.get("text", ""),
                "tokens": int(payload.get("tokens") or utils.estimate_tokens(payload.get("text", ""))),
                "meta": payload.get("meta") or {},
            }
            vectors.append(vector.copy())
            entries.append(entry)
        if not vectors:
            return
        with self._lock:
            dim = self._vectors[0].size if self._vectors else vectors[0].size
            if any(vector.size != dim for vector in vectors):
                raise ValueError("embedding dimension mismatch")
            previous = len(self._vectors)
            self._vectors.extend(vectors)
            self._payloads.extend(entries)
            try:
                self._flush()
            except BaseException:
                del self._vectors[previous:]
                del self._payloads[previous:]
                raise
            self._matrix = None

    def clear(self) -> None:
        with self._lock:
            # Commit the empty index before dropping the in-memory copy.
            atomic_write_text(self.path, '{"embeddings":[],"payloads":[]}')
            self._vectors.clear()
            self._payloads.clear()
            self._matrix = None

    def search(self, query: np.ndarray, *, top_k: int = 4) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._vectors:
                return []
            query_vec = np.asarray(query, dtype=np.float32)
            if query_vec.size == 0:
                return []
            if self._matrix is None:
                self._matrix = np.stack(self._vectors).astype(np.float32)
            matrix = self._matrix
            scores_matrix = self._backend.cosine_sim_matrix(query_vec, matrix)
            scores = scores_matrix[0]
            top_indices, _ = self._backend.top_k(scores, max(1, int(top_k)))
            ranked = [
                (scores[idx], self._payloads[idx]) for idx in top_indices[0]
            ]
            results: List[Dict[str, Any]] = []
            for score, payload in ranked:
                entry = dict(payload)
                entry["score"] = float(score)
                entry.setdefault("tokens", utils.estimate_tokens(entry.get("text", "")))
                results.append(entry)
            return results


class PersistentVectorBackend(RAGBackendProtocol):
    """Adapter exposing :class:`PersistentVectorIndex` as a RAG backend."""

    def __init__(self, path: Path) -> None:
        self.identifier = "persistent"
        self.label = "Persistent Index"
        self.description = "Disk-backed cosine similarity index"
        self._index = PersistentVectorIndex(path)

    def add_document(
        self,
        text: str,
        embedding: np.ndarray,
        tokens: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "text": text,
            "tokens": tokens,
            "meta": dict(meta or {}),
        }
        self._index.add(embedding, payload)

    def clear(self) -> None:
        self._index.clear()

    def retrieve(self, query_embedding: np.ndarray, *, top_k: int) -> List[Dict[str, Any]]:
        results = self._index.search(query_embedding, top_k=top_k)
        for result in results:
            result.setdefault("meta", {})
        return results

    # Helpers for diagnostics -------------------------------------------------
    def export(self) -> Dict[str, Any]:
        return {
            "count": len(self._index._vectors),
            "path": str(self._index.path),
        }
