"""Lightweight in-memory RAG store for the Daystrom demo UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import utils
from .embeddings import Embedder


@dataclass
class SimpleRAGStore:
    """Extremely small RAG index backed by the configured embedder.

    The implementation is intentionally naive – it keeps all documents in
    memory, scores them using cosine similarity, and exposes the top ``k``
    matches together with a human readable context block.  The goal is to
    provide a transparent baseline for the frontend where users can inspect
    exactly which snippets are forwarded to the language model.
    """

    embedder: Embedder
    _documents: List[Dict[str, Any]] = field(default_factory=list)

    def add_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add ``text`` to the store along with optional metadata."""

        if not text:
            return
        embedding = self.embedder.embed(text)
        payload = {
            "text": text,
            "embedding": embedding,
            "meta": meta or {},
            "tokens": utils.estimate_tokens(text),
        }
        self._documents.append(payload)

    def clear(self) -> None:
        """Reset the in-memory collection."""

        self._documents.clear()

    def _score_documents(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if query_embedding.size == 0:
            return results
        for doc in self._documents:
            score = utils.cosine_similarity(doc["embedding"], query_embedding)
            results.append(
                {
                    "text": doc["text"],
                    "meta": doc["meta"],
                    "tokens": doc["tokens"],
                    "score": float(score),
                }
            )
        results.sort(key=lambda entry: entry["score"], reverse=True)
        return results

    def retrieve(self, prompt: str, *, top_k: int = 4) -> List[Dict[str, Any]]:
        """Return the highest scoring documents for ``prompt``."""

        if not self._documents:
            return []
        query_embedding = self.embedder.embed(prompt)
        scored = self._score_documents(query_embedding)
        return scored[: max(0, int(top_k))]

    def report(self, prompt: str, *, top_k: int = 4) -> Dict[str, Any]:
        """Return retrieval matches together with a formatted context block."""

        matches = self.retrieve(prompt, top_k=top_k)
        context_lines: List[str] = []
        for idx, match in enumerate(matches, start=1):
            source = match["meta"].get("doc_path") if match.get("meta") else None
            source = source or match.get("meta", {}).get("source") or "uploaded document"
            header = f"Document {idx} (score={match['score']:.3f})"
            context_lines.extend([header, f"Source: {source}", match["text"].strip(), ""])
        context = "\n".join(context_lines).strip()
        tokens = utils.estimate_tokens(context)
        return {
            "documents": matches,
            "context": context,
            "tokens": tokens,
        }

    def count(self) -> int:
        """Expose the number of stored documents for diagnostics."""

        return len(self._documents)
