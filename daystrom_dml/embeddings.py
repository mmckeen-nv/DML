"""Embedding backends for the Daystrom Memory Lattice."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import utils

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    """Simple embedder protocol."""

    def embed(self, text: str) -> np.ndarray:
        ...


@dataclass
class SentenceTransformerEmbedder:
    """Embed text using a SentenceTransformer model.

    The class degrades gracefully when the ``sentence_transformers`` package is
    unavailable by falling back to :class:`RandomEmbedder`.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
            LOGGER.info("Loaded SentenceTransformer model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - exercised when dependency missing
            LOGGER.warning("Falling back to RandomEmbedder: %s", exc)
            self._model = None
            self._dim = 384

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            return RandomEmbedder(self._dim).embed(text)
        if not text:
            return np.zeros(self._dim, dtype=np.float32)
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)


@dataclass
class RandomEmbedder:
    """Deterministic pseudo-random embeddings used in tests."""

    dim: int = 384

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        return utils.seeded_random_vector(self.dim, seed)


def create_embedder(model_name: str | None) -> Embedder:
    """Factory returning the best available embedder."""

    if model_name:
        return SentenceTransformerEmbedder(model_name)
    return RandomEmbedder()

