"""Lightweight persistent FAISS-backed retrieval store."""
from __future__ import annotations

import json
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .atomic_io import atomic_write_text, atomic_write_via

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import]
except Exception:  # pragma: no cover - handled gracefully when unavailable
    faiss = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class RAGRecord:
    """Single document stored in the persistent RAG index."""

    id: int
    text: str
    meta: Dict[str, Any]


class PersistentRAGStore:
    """Minimal persistent vector store backed by FAISS."""

    def __init__(
        self,
        *,
        enable: bool,
        index_path: Path,
        meta_path: Path,
        dim: int,
        backend: str = "faiss",
    ) -> None:
        self.enable = enable
        self.backend = backend
        self.index_path = index_path
        self.meta_path = meta_path
        self.manifest_path = meta_path.with_name(meta_path.name + ".manifest.json")
        self._records: List[RAGRecord] = []
        self._id_lookup: Dict[int, int] = {}
        self._index: Any = None
        self._next_id = 0
        self._dim = int(dim)
        if self.backend != "faiss":
            raise ValueError(f"Unsupported backend: {backend}")
        if self.enable and faiss is None:
            raise RuntimeError("faiss is required for the persistent RAG store")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load an existing index from disk and report whether it was restored."""

        if not self.enable:
            return False
        self._clear_loaded_state()
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.exception("Failed to read persistent RAG generation manifest")
            else:
                for entry in (manifest.get("current"), manifest.get("previous")):
                    if isinstance(entry, dict) and self._load_generation(entry):
                        return True
                LOGGER.error("No complete persistent RAG generation could be restored")
                return False
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        return self._load_pair(self.index_path, self.meta_path)

    def _load_generation(self, entry: Dict[str, Any]) -> bool:
        index_path = self.manifest_path.parent / str(entry.get("index") or "")
        meta_path = self.manifest_path.parent / str(entry.get("metadata") or "")
        if not index_path.is_file() or not meta_path.is_file():
            return False
        if entry.get("index_sha256") != self._sha256(index_path):
            return False
        if entry.get("metadata_sha256") != self._sha256(meta_path):
            return False
        return self._load_pair(index_path, meta_path, expected_generation=str(entry.get("generation") or ""))

    def _load_pair(self, index_path: Path, meta_path: Path, *, expected_generation: str = "") -> bool:
        assert faiss is not None
        try:
            self._index = faiss.read_index(str(index_path))
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to load persistent RAG index from disk")
            return False
        if expected_generation and data.get("generation") != expected_generation:
            self._clear_loaded_state()
            return False
        records = data.get("records", [])
        next_id = int(data.get("next_id", len(records)))
        dim = int(data.get("dim", getattr(self._index, "d", self._dim)))
        self._records = [
            RAGRecord(id=int(entry.get("id", idx)), text=entry.get("text", ""), meta=dict(entry.get("meta") or {}))
            for idx, entry in enumerate(records)
        ]
        self._id_lookup = {record.id: pos for pos, record in enumerate(self._records)}
        self._next_id = max(next_id, (max(self._id_lookup.keys()) + 1) if self._id_lookup else 0)
        self._dim = dim
        if self._index is not None and getattr(self._index, "d", dim) != dim:
            LOGGER.warning(
                "FAISS index dimension (%s) mismatches metadata (%s); using index dimension.",
                getattr(self._index, "d", "unknown"),
                dim,
            )
            self._dim = int(getattr(self._index, "d", dim))
        if self._index is not None and self._index.ntotal != len(self._records):
            LOGGER.error(
                "FAISS index document count (%s) mismatches metadata (%s); rejecting torn durable state.",
                self._index.ntotal,
                len(self._records),
            )
            self._clear_loaded_state()
            return False
        return True

    def persist(self) -> None:
        """Persist the FAISS index and metadata to disk."""

        if not self.enable or self._index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        payload = {
            "generation": generation,
            "dim": self._dim,
            "next_id": self._next_id,
            "records": [record.__dict__ for record in self._records],
        }
        index_gen = self.index_path.with_name(f"{self.index_path.name}.{generation}")
        meta_gen = self.meta_path.with_name(f"{self.meta_path.name}.{generation}")
        assert faiss is not None
        faiss_module = faiss
        atomic_write_via(index_gen, lambda path: faiss_module.write_index(self._index, str(path)))
        atomic_write_text(meta_gen, json.dumps(payload, ensure_ascii=False, indent=2))
        previous = None
        if self.manifest_path.exists():
            try:
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8")).get("current")
            except Exception:
                LOGGER.warning("Ignoring unreadable prior RAG manifest", exc_info=True)
        current = {
            "generation": generation,
            "index": index_gen.name,
            "metadata": meta_gen.name,
            "index_sha256": self._sha256(index_gen),
            "metadata_sha256": self._sha256(meta_gen),
        }
        atomic_write_text(
            self.manifest_path,
            json.dumps({"version": 1, "current": current, "previous": previous}, indent=2),
        )

    def _clear_loaded_state(self) -> None:
        self._index = None
        self._records = []
        self._id_lookup = {}
        self._next_id = 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # core operations
    # ------------------------------------------------------------------
    def add(self, text: str, embedding: Iterable[float], meta: Optional[Dict[str, Any]] = None) -> int:
        """Add a new document to the persistent index."""

        if not text or not self.enable:
            return -1
        vector = self._prepare_vector(embedding)
        if vector is None:
            return -1
        if self._index is None:
            self._index = faiss.IndexFlatIP(vector.shape[1])
            self._dim = vector.shape[1]
        if vector.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, received {vector.shape[1]}"
            )
        faiss.normalize_L2(vector)
        self._index.add(vector)
        record_id = self._next_id
        self._next_id += 1
        record = RAGRecord(id=record_id, text=text, meta=dict(meta or {}))
        self._records.append(record)
        self._id_lookup[record_id] = len(self._records) - 1
        return record_id

    def search(self, embedding: Iterable[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve the closest matches for the supplied embedding."""

        if not self.enable or self._index is None or not self._records:
            return []
        vector = self._prepare_vector(embedding)
        if vector is None or vector.shape[1] != self._dim:
            return []
        faiss.normalize_L2(vector)
        top_k = max(1, min(int(top_k), len(self._records)))
        scores, indices = self._index.search(vector, top_k)
        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            record = self._records[idx]
            results.append(
                {
                    "id": record.id,
                    "text": record.text,
                    "meta": dict(record.meta),
                    "score": float(score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_vector(self, embedding: Iterable[float]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        vector = np.asarray(list(embedding), dtype=np.float32)
        if vector.size == 0:
            return None
        return vector.reshape(1, -1)
