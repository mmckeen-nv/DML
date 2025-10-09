"""Utility helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import math
import time
from typing import Iterable, List

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    The implementation is robust to zero vectors and returns ``0`` when either
    vector does not contain any magnitude.  The function accepts any
    ``numpy.ndarray``-like inputs and converts them into ``float32`` arrays for
    deterministic behaviour.
    """

    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(values: Iterable[float]) -> List[float]:
    """Return softmax of the input iterable."""

    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return []
    vals = vals - np.max(vals)
    exps = np.exp(vals)
    denom = np.sum(exps)
    if denom == 0:
        return [0.0 for _ in exps]
    return list(exps / denom)


def estimate_tokens(text: str) -> int:
    """Rough token estimation using a GPT-2 style heuristic."""

    if not text:
        return 0
    # Empirical: 1 token ~ 4 characters for english-like text.
    return max(1, int(len(text) / 4))


def age_in_hours(timestamp: float, now: float | None = None) -> float:
    """Return age in hours from ``timestamp`` to ``now``."""

    now = time.time() if now is None else now
    return max(0.0, (now - timestamp) / 3600.0)


def ensure_serializable(array: np.ndarray) -> List[float]:
    """Convert numpy array to plain python list for JSON serialization."""

    return np.asarray(array, dtype=np.float32).tolist()


def seeded_random_vector(dim: int, seed: int) -> np.ndarray:
    """Return a deterministic pseudo-random vector for offline embeddings."""

    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)
