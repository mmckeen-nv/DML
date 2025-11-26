"""Core Daystrom Memory Lattice implementation."""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import utils
from .summarizer import Summarizer


@dataclass
class MemoryItem:
    """Container for a single memory element."""

    id: int
    text: str
    embedding: np.ndarray
    timestamp: float
    salience: float
    fidelity: float
    level: int
    meta: Optional[Dict] = field(default_factory=dict)
    summary_of: List[int] = field(default_factory=list)

    @property
    def children(self) -> List[int]:
        if self.summary_of:
            return list(self.summary_of)
        return [self.id]

    def cached_summary(self, max_len: int = 256) -> str:
        summary = ""
        if self.meta is not None:
            summary = str(self.meta.get("summary") or "").strip()
        if summary:
            if len(summary) > max_len:
                return summary[: max_len - 3] + "..."
            return summary
        text = (self.text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
            "salience": self.salience,
            "fidelity": self.fidelity,
            "level": self.level,
            "meta": self.meta,
            "summary_of": self.summary_of,
            "embedding": utils.ensure_serializable(self.embedding),
            "children": self.children,
        }


class MemoryStore:
    """Implements storage, retrieval and ageing for the DML."""

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        beta_a: float,
        beta_r: float,
        eta: float,
        gamma: float,
        kappa: float,
        tau_s: float,
        theta_merge: float,
        K: int,
        capacity: int,
        start_aging_loop: bool = True,
    ) -> None:
        self.summarizer = summarizer
        self.beta_a = beta_a
        self.beta_r = beta_r
        self.eta = eta
        self.gamma = gamma
        self.kappa = kappa
        self.tau_s = tau_s
        self.theta_merge = theta_merge
        self.K = max(1, K)
        self.capacity = max(1, capacity)
        self._items: List[MemoryItem] = []
        self._id = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._lineage: Dict[int, MemoryItem] = {}
        self._repair_queue: List[int] = []
        self.quality_threshold = -0.1
        self._aging_thread: Optional[threading.Thread] = None
        if start_aging_loop:
            self._aging_thread = threading.Thread(
                target=self._aging_loop, name="dml-aging", daemon=True
            )
            self._aging_thread.start()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._stop_event.set()
        if self._aging_thread and self._aging_thread.is_alive():
            self._aging_thread.join(timeout=1.0)

    def ingest(
        self,
        text: str,
        embedding: np.ndarray,
        *,
        salience: float = 1.0,
        fidelity: float = 1.0,
        level: int = 0,
        meta: Optional[Dict] = None,
    ) -> Tuple[MemoryItem, bool]:
        now = time.time()
        enriched_meta = dict(meta or {})

        with self._lock:
            best_match, best_sim = self._best_match(embedding)

        if best_match and best_sim >= self.theta_merge:
            combined_text = f"{best_match.text}\n{text}".strip()
            summary = self._generate_summary(combined_text, max_len=256)
        else:
            summary = enriched_meta.get("summary") or self._generate_summary(
                text, max_len=256
            )

        with self._lock:
            merged = self._try_merge(text, embedding, salience, meta=meta)
            if merged:
                return merged, True
            item = MemoryItem(
                id=self._next_id(),
                text=text,
                embedding=np.asarray(embedding, dtype=np.float32),
                timestamp=now,
                salience=float(salience),
                fidelity=float(max(0.0, min(1.0, fidelity))),
                level=int(level),
                meta=enriched_meta,
            )
            self._cache_summary(item, text)
            item.summary_of = [item.id]
            self._register_lineage(item)
            self._items.append(item)
            self._enforce_capacity()
            return item, False

    def retrieve(
        self, query_embedding: np.ndarray, top_k: Optional[int] = 6
    ) -> List[MemoryItem]:
        with self._lock:
            scored = []
            now = time.time()
            for item in self._items:
                score = self._score_item(item, query_embedding, now)
                self._assess_quality(item, query_embedding, now)
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            if top_k is None:
                limit = len(scored)
            else:
                try:
                    limit = int(top_k)
                except (TypeError, ValueError):
                    limit = 0
                if limit <= 0 or limit >= len(scored):
                    limit = len(scored)
            return [item for _, item in scored[:limit]]

    def items(self) -> Sequence[MemoryItem]:
        with self._lock:
            return list(self._items)

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON serialisable snapshot of the memory lattice."""

        with self._lock:
            return {
                "items": [item.to_dict() for item in self._items],
                "lineage": [item.to_dict() for item in self._lineage.values()],
                "repair_queue": list(self._repair_queue),
                "next_id": self._id,
            }

    def import_state(self, payload: Optional[Dict[str, Any]]) -> None:
        """Restore the lattice from ``payload`` if provided."""

        if not payload:
            return

        items_data = payload.get("items") or []
        reconstructed: List[MemoryItem] = []
        lineage_entries = payload.get("lineage") or []
        lineage_map: Dict[int, MemoryItem] = {}
        for entry in lineage_entries:
            item = self._reconstruct_item(entry)
            if item is not None:
                lineage_map[item.id] = item
        for entry in items_data:
            item = self._reconstruct_item(entry)
            if item is None:
                continue
            reconstructed.append(item)
            lineage_map[item.id] = item

        with self._lock:
            self._items = reconstructed
            self._lineage = lineage_map
            if self._items:
                self._id = max(item.id for item in self._items) + 1
            else:
                self._id = int(payload.get("next_id") or 0)
            existing_queue = payload.get("repair_queue") or []
            self._repair_queue = [
                int(val) for val in existing_queue if int(val) in self._lineage
            ]

    def decay_step(self, now: Optional[float] = None) -> None:
        """Public hook used in tests to simulate ageing."""

        with self._lock:
            self._apply_decay(now=now)
            self._abstract_low_fidelity()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        value = self._id
        self._id += 1
        return value

    def _try_merge(
        self, text: str, embedding: np.ndarray, salience: float, meta: Optional[Dict]
    ) -> Optional[MemoryItem]:
        if not self._items:
            return None
        best: Optional[MemoryItem] = None
        best_sim = 0.0
        for item in self._items:
            sim = utils.cosine_similarity(item.embedding, embedding)
            if sim > best_sim:
                best_sim = sim
                best = item
        if best and best_sim >= self.theta_merge:
            combined_text = f"{best.text}\n{text}".strip()
            summary = self.summarizer.summarize(combined_text, max_len=256)
            summary = summary or combined_text[:253] + "..."
            best.meta["summary"] = summary
            best.text = combined_text
            best.embedding = (best.embedding + embedding) / 2.0
            best.timestamp = time.time()
            best.salience = max(best.salience, salience)
            best.meta.setdefault("merges", 0)
            best.meta["merges"] += 1
            if not best.summary_of:
                best.summary_of = [best.id]
            child = MemoryItem(
                id=self._next_id(),
                text=text,
                embedding=np.asarray(embedding, dtype=np.float32),
                timestamp=time.time(),
                salience=float(salience),
                fidelity=1.0,
                level=0,
                meta=meta or {},
            )
            self._cache_summary(child, text)
            child.summary_of = [child.id]
            self._register_lineage(child)
            for child_id in child.summary_of:
                if child_id not in best.summary_of:
                    best.summary_of.append(child_id)
            self._register_lineage(best)
            return best
        return None

    def _score_item(
        self, item: MemoryItem, query_embedding: np.ndarray, now: float
    ) -> float:
        similarity = utils.cosine_similarity(item.embedding, query_embedding)
        age = utils.age_in_hours(item.timestamp, now)
        recency = 1.0 / (1.0 + age)
        return (
            similarity
            + self.eta * recency
            + self.gamma * item.salience
            + self.kappa * item.fidelity
        )

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        now = time.time()
        self._items.sort(
            key=lambda item: (
                item.fidelity
                + 0.1 * item.salience
                - 0.01 * utils.age_in_hours(item.timestamp, now)
            )
        )
        while len(self._items) > self.capacity:
            self._items.pop(0)

    def _aging_loop(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            with self._lock:
                self._apply_decay()
                self._abstract_low_fidelity()
            self._stop_event.wait(5.0)

    def _apply_decay(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for item in self._items:
            age = utils.age_in_hours(item.timestamp, now)
            recency = 1.0 / (1.0 + age)
            lambda_star = utils.sigmoid(self.beta_r * recency - self.beta_a * age)
            item.fidelity = float(max(0.0, min(1.0, lambda_star)))
            item.level = min(self.K, max(0, int((1.0 - item.fidelity) * self.K)))

    def _abstract_low_fidelity(self) -> None:
        new_items: List[MemoryItem] = []
        for item in list(self._items):
            if item.fidelity < self.tau_s and item.level < self.K:
                summary_text = self._generate_summary(item.text, max_len=256)
                if not summary_text:
                    summary_text = item.text[:253] + "..."
                summary_embedding = item.embedding.copy()
                new_item = MemoryItem(
                    id=self._next_id(),
                    text=summary_text,
                    embedding=summary_embedding,
                    timestamp=time.time(),
                    salience=item.salience * 0.9,
                    fidelity=min(1.0, item.fidelity + 0.5),
                    level=item.level + 1,
                    meta={"abstracted_from": item.id, "summary": summary_text},
                    summary_of=list({item.id, *item.summary_of}),
                )
                self._register_lineage(new_item)
                item.meta.setdefault("abstracted", True)
                item.meta.setdefault("summary", self._generate_summary(item.text, max_len=256))
                item.fidelity *= 0.5
                new_items.append(new_item)
        self._items.extend(new_items)
        self._enforce_capacity()

    # ------------------------------------------------------------------
    # lineage and quality helpers
    # ------------------------------------------------------------------
    def _register_lineage(self, item: MemoryItem) -> None:
        self._lineage[item.id] = item

    def _reconstruct_item(self, entry: Dict[str, Any]) -> Optional[MemoryItem]:
        try:
            embedding = np.asarray(entry.get("embedding") or [], dtype=np.float32)
            item = MemoryItem(
                id=int(entry.get("id", 0)),
                text=str(entry.get("text") or ""),
                embedding=embedding,
                timestamp=float(entry.get("timestamp") or 0.0),
                salience=float(entry.get("salience") or 0.0),
                fidelity=float(entry.get("fidelity") or 0.0),
                level=int(entry.get("level") or 0),
                meta=entry.get("meta") or {},
                summary_of=list(entry.get("summary_of") or []),
            )
            return item
        except Exception:
            return None

    def _cache_summary(self, item: MemoryItem, text: str) -> None:
        summary = self.summarizer.summarize(text, max_len=256)
        if not summary:
            summary = text[:253] + "..."
        item.meta["summary"] = summary

    def _assess_quality(self, item: MemoryItem, query_embedding: np.ndarray, now: float) -> None:
        similarity = utils.cosine_similarity(item.embedding, query_embedding)
        child_embeddings = self._child_embeddings(item.children)
        variance = float(np.var(child_embeddings)) if child_embeddings else 0.0
        age_hours = utils.age_in_hours(item.timestamp, now)
        aging_penalty = age_hours * 0.01
        quality = similarity - variance - aging_penalty
        if quality < self.quality_threshold:
            self._enqueue_repair(item.id)

    def _child_embeddings(self, child_ids: Iterable[int]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for child_id in child_ids:
            child = self._lineage.get(child_id)
            if child is None:
                continue
            vectors.append(child.embedding)
        return vectors

    def _enqueue_repair(self, item_id: int) -> None:
        if item_id in self._repair_queue:
            return
        self._repair_queue.append(item_id)

    # ------------------------------------------------------------------
    # public maintenance hooks
    # ------------------------------------------------------------------
    def dequeue_repair_batch(self, limit: int = 5) -> List[MemoryItem]:
        with self._lock:
            batch_ids = self._repair_queue[:limit]
            self._repair_queue = self._repair_queue[limit:]
            return [self._lineage.get(idx) for idx in batch_ids if self._lineage.get(idx)]

    def lineage_items(self, ids: Iterable[int]) -> List[MemoryItem]:
        with self._lock:
            return [self._lineage[it] for it in ids if it in self._lineage]

    def update_node(self, item_id: int, *, summary: str, embedding: np.ndarray) -> None:
        with self._lock:
            target = next((it for it in self._items if it.id == item_id), None)
            if target is None:
                return
            target.meta["summary"] = summary
            target.embedding = np.asarray(embedding, dtype=np.float32)
            target.timestamp = time.time()
            self._register_lineage(target)

    def repair_queue(self) -> List[int]:
        with self._lock:
            return list(self._repair_queue)

