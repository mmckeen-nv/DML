"""Independent M8 memory arms; no model, DML lifecycle, or answer-key access.

The existing SQLiteBaseline owns durable storage. Its cosine/recency ranking is
preserved below, while eligibility and exact-token presentation precede cutoffs.
Authorization is the caller's responsibility; scopes are matched exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.production_baseline import SQLiteBaseline


@dataclass(frozen=True)
class MemoryEvent:
    record_id: int
    text: str
    scope: dict
    source_id: str
    fact_key: str
    version: int
    timestamp: float
    corrects: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PresentedEvidence:
    records: tuple[dict, ...]
    context: str
    token_count: int


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"invalid {name}")


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid timestamp")


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {"tenant_id", "client_id", "session_id", "instance_id"}:
        raise ValueError("nonempty exact scope required")
    if scope["tenant_id"] is None:
        raise ValueError("tenant scope required")
    if any(v is not None and (not isinstance(v, str) or not v.strip() or len(v.encode("utf-8")) > 256)
           for v in scope.values()):
        raise ValueError("invalid scope")
    return json.dumps(scope, sort_keys=True)


def validate_event(event):
    _integer(event.record_id, "record ID")
    _integer(event.version, "version", 1)
    _time(event.timestamp)
    validate_scope(event.scope)
    for value in (event.text, event.source_id, event.fact_key):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("nonempty text/source/fact required")
    if event.corrects is not None:
        _integer(event.corrects, "correction ID")
        if event.corrects == event.record_id:
            raise ValueError("self correction")
    if not isinstance(event.metadata, dict):
        raise ValueError("invalid metadata")
    # Snapshot caller-owned data and reject non-JSON or nonfinite metadata.
    return json.loads(json.dumps(event.__dict__, allow_nan=False, sort_keys=True))


def present_records(records, budget, count_tokens, render_records):
    _integer(budget, "token budget", 1)
    records = json.loads(json.dumps(records, allow_nan=False))
    selected = []
    context = render_records(selected)
    tokens = count_tokens(context)
    _integer(tokens, "token count")
    if tokens > budget:
        raise ValueError("empty evidence envelope exceeds budget")
    for record in records:
        candidate = selected + [record]
        rendered = render_records(candidate)
        used = count_tokens(rendered)
        _integer(used, "token count")
        if used > budget:
            break
        selected, context, tokens = candidate, rendered, used
    return PresentedEvidence(tuple(selected), context, tokens)


class PersistentRAGBaseline:
    """Append-only evidence plus an explicit same-source current view."""

    def __init__(self, path: Path, *, embedding_identity: str, dimensions: int):
        _integer(dimensions, "dimensions", 1)
        if not isinstance(embedding_identity, str) or not embedding_identity:
            raise ValueError("embedding identity required")
        self.dimensions = dimensions
        self.embedding_identity = embedding_identity
        self.store = SQLiteBaseline(Path(path))
        connection = self.store.connection
        try:
            with connection:
                connection.execute("CREATE TABLE IF NOT EXISTS m8_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), identity TEXT NOT NULL, dimensions INTEGER NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS m8_events (id INTEGER PRIMARY KEY, envelope TEXT NOT NULL, predecessor INTEGER UNIQUE)")
                identity = connection.execute("SELECT identity,dimensions FROM m8_identity").fetchone()
                if identity is None:
                    if connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0]:
                        raise ValueError("unattested legacy rows cannot be adopted")
                    connection.execute("INSERT INTO m8_identity VALUES (1,?,?)", (embedding_identity, dimensions))
                elif identity != (embedding_identity, dimensions):
                    raise ValueError("embedding identity/dimension mismatch")
                if connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0] != connection.execute("SELECT COUNT(*) FROM m8_events").fetchone()[0]:
                    raise ValueError("metadata history mismatch")
        except BaseException:
            self.close()
            raise

    def _vector(self, vector):
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (self.dimensions,) or not np.all(np.isfinite(values)):
            raise ValueError("invalid embedding")
        if not np.isclose(np.linalg.norm(values), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("embedding must be nonzero L2 normalized")
        return values

    def ingest(self, event: MemoryEvent, vector):
        envelope = validate_event(event)
        values = self._vector(vector)
        connection = self.store.connection
        # Both legacy memory and provenance commit together; remember() commits
        # internally, so use its exact insert within this encompassing transaction.
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute("SELECT MAX(timestamp) FROM memory WHERE scope=?", (validate_scope(event.scope),)).fetchone()[0]
            if latest is not None and event.timestamp < latest:
                raise ValueError("nonchronological input")
            if event.corrects is None:
                existing = connection.execute("SELECT e.envelope FROM memory m JOIN m8_events e ON m.id=e.id WHERE m.scope=?", (validate_scope(event.scope),)).fetchall()
                if any((prior := json.loads(row[0]))["source_id"] == event.source_id and prior["fact_key"] == event.fact_key for row in existing):
                    raise ValueError("existing source/fact requires explicit correction")
            if event.corrects is not None:
                row = connection.execute("SELECT envelope FROM m8_events WHERE id=?", (event.corrects,)).fetchone()
                if row is None:
                    raise ValueError("unknown correction predecessor")
                previous = json.loads(row[0])
                if any(previous[key] != envelope[key] for key in ("scope", "source_id", "fact_key")):
                    raise ValueError("correction must retain scope/source/fact")
                if event.version <= previous["version"] or event.timestamp < previous["timestamp"]:
                    raise ValueError("nonmonotonic correction")
                if connection.execute("SELECT 1 FROM m8_events WHERE predecessor=?", (event.corrects,)).fetchone():
                    raise ValueError("stale correction predecessor")
            connection.execute("INSERT INTO memory VALUES (?,?,?,?,?)", (event.record_id, event.text, values.tobytes(), validate_scope(event.scope), event.timestamp))
            connection.execute("INSERT INTO m8_events VALUES (?,?,?)", (event.record_id, json.dumps(envelope, allow_nan=False, sort_keys=True), event.corrects))

    def history(self, *, scope):
        rows = self.store.connection.execute("SELECT e.envelope FROM memory m JOIN m8_events e ON m.id=e.id WHERE m.scope=? ORDER BY m.id", (validate_scope(scope),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def retrieve(self, vector, *, scope, now, top_k=8, token_budget=2048,
                 count_tokens: Callable[[str], int], render_records: Callable[[list[dict]], str]):
        query = self._vector(vector)
        _time(now)
        _integer(top_k, "top_k", 1)
        if top_k > 8:
            raise ValueError("top_k exceeds frozen limit")
        rows = self.store.connection.execute(
            "SELECT m.id,m.vector,m.timestamp,e.envelope FROM memory m JOIN m8_events e ON e.id=m.id "
            "WHERE m.scope=? AND NOT EXISTS (SELECT 1 FROM m8_events newer WHERE newer.predecessor=m.id) ORDER BY m.id",
            (validate_scope(scope),)).fetchall()
        ranked = []
        for record_id, raw, timestamp, envelope in rows:
            stored = self._vector(np.frombuffer(raw, dtype=np.float32))
            _time(timestamp)
            if timestamp > now:
                raise ValueError("future evidence at query time")
            denominator = np.linalg.norm(query) * np.linalg.norm(stored)
            cosine = float(np.dot(query, stored) / denominator) if denominator else 0.0
            recency = 1 / (1 + max(0., now - timestamp) / 3600)
            score = cosine + .15 * recency
            if not math.isfinite(score):
                raise ValueError("nonfinite ranking score")
            ranked.append((score, record_id, json.loads(envelope)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return present_records([row[2] for row in ranked[:top_k]], token_budget, count_tokens, render_records)

    def close(self):
        self.store.close()


class ContextBuffer:
    """Newest whole chronological records, without retrieval or hidden history."""

    def __init__(self, *, scope, token_budget, count_tokens, render_records):
        self.scope = json.loads(validate_scope(scope))
        _integer(token_budget, "token budget", 1)
        self.budget, self.count_tokens, self.render_records = token_budget, count_tokens, render_records
        self._records = []
        self._last_timestamp = -1
        self._ids = set()

    def ingest(self, event: MemoryEvent):
        record = validate_event(event)
        if record["scope"] != self.scope:
            raise ValueError("scope mismatch")
        if event.record_id in self._ids or event.timestamp < self._last_timestamp:
            raise ValueError("duplicate or nonchronological input")
        candidate = self._records + [record]
        while candidate:
            used = self.count_tokens(self.render_records(candidate))
            _integer(used, "token count")
            if used <= self.budget:
                break
            candidate.pop(0)
        present_records([], self.budget, self.count_tokens, self.render_records)
        self._records = candidate
        self._last_timestamp = event.timestamp
        self._ids.add(event.record_id)

    def present(self):
        return present_records(self._records, self.budget, self.count_tokens, self.render_records)
