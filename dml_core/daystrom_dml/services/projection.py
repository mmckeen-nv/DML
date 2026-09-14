"""Disposable, transactionally replaced SQLite vector snapshots.

The receipt journal remains the sole authority. Publication is replayable and
queries pin a verified source revision; this is not a distributed transaction,
an incremental outbox, or a receipt backup. No source lock crosses backend I/O.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from ..journal import JournalStateStore, RevisionConflict
from ..persistence import validate_snapshot
from ..store_lock import store_write_lock
from .lifecycle import suppression_reason
from .receipt_ingestion import SCOPE_KEYS, _strict_json, validate_embedding_contract

FORMAT = "dml-sqlite-projection-v1"
FAULT_POINTS = ("before_reconcile", "after_prepare", "before_publish", "after_publish", "after_reconcile")


class ProjectionError(ValueError):
    """Projection state or request cannot safely be used."""


class ProjectionStale(ProjectionError):
    """The projection does not match the pinned authority snapshot."""


def _encode(value) -> str:
    _strict_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_encode(value).encode("utf-8")).hexdigest()


def _scope(value) -> dict:
    if type(value) is not dict or set(value) != set(SCOPE_KEYS):
        raise ProjectionError("An exact four-member scope is required")
    for name, member in value.items():
        if member is None and name != "tenant_id":
            continue
        if type(member) is not str or not member.strip() or len(member.encode("utf-8")) > 256:
            raise ProjectionError("Invalid projection scope")
    return value


def _state_contract(state: dict) -> dict | None:
    validate_snapshot(state)
    contract = state.get("embedding_contract")
    if contract is None:
        validate_embedding_contract(state, {})
        return None
    identity = contract.get("identity") if type(contract) is dict else None
    if type(identity) is not dict or set(identity) != {"backend", "revision", "model", "mode"}:
        raise ProjectionError("Invalid embedding identity")
    for key in ("backend", "revision", "mode"):
        if type(identity[key]) is not str or not identity[key].strip():
            raise ProjectionError("Invalid embedding identity")
    if identity["model"] is not None and type(identity["model"]) is not str:
        raise ProjectionError("Invalid embedding model")
    validate_embedding_contract(state, identity)
    for record in state["items"]:
        meta = record.get("meta") or {}
        _scope({name: meta.get(name) for name in SCOPE_KEYS})
    return contract


def prepare_source(source: JournalStateStore) -> dict:
    if not source.path.is_file() or source.schema_version != 2:
        raise ProjectionError("Projection source must be an existing schema-2 receipt journal")
    store_id, revision, state = source.verified_snapshot()
    _state_contract(state)
    return {"source_store_id": store_id, "source_revision": revision,
            "source_digest": _digest(state), "source_state": state,
            "source_path": str(source.path)}


def _prepared(snapshot: dict) -> tuple[dict, Path]:
    snapshot = json.loads(_encode(snapshot))  # freeze before any backend hook/I/O
    keys = {"source_store_id", "source_revision", "source_digest", "source_state", "source_path"}
    if type(snapshot) is not dict or set(snapshot) != keys:
        raise ProjectionError("Invalid prepared snapshot")
    _cursor(snapshot)
    if type(snapshot["source_path"]) is not str or not Path(snapshot["source_path"]).is_absolute():
        raise ProjectionError("Invalid source path")
    state = snapshot["source_state"]
    contract = _state_contract(state)
    if _digest(state) != snapshot["source_digest"]:
        raise ProjectionError("Prepared source digest mismatch")
    return {"schema_version": 1, "projection_format": FORMAT,
            "cursor": {key: snapshot[key] for key in ("source_store_id", "source_revision", "source_digest")},
            "embedding_contract": contract, "items": state["items"], "lineage": []}, Path(snapshot["source_path"]).resolve()


def _cursor(value: dict) -> None:
    sid, revision, digest = (value.get(key) for key in ("source_store_id", "source_revision", "source_digest"))
    if type(sid) is not str or len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
        raise ProjectionError("Invalid projection authority identity")
    if type(revision) is not int or revision < 0:
        raise ProjectionError("Invalid source revision")
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ProjectionError("Invalid source digest")


def _validate(envelope: dict) -> dict:
    if set(envelope) != {"schema_version", "projection_format", "cursor", "embedding_contract", "items", "lineage"} or envelope.get("projection_format") != FORMAT:
        raise ProjectionError("Target is not a recognized projection")
    if envelope["lineage"] != []:
        raise ProjectionError("Projection must not contain lineage")
    cursor = envelope["cursor"]
    if cursor is None:
        if envelope["items"] != [] or envelope["embedding_contract"] is not None or type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
            raise ProjectionError("Invalid uninitialized projection")
    else:
        if type(cursor) is not dict or set(cursor) != {"source_store_id", "source_revision", "source_digest"}:
            raise ProjectionError("Invalid projection cursor")
        _cursor(cursor)
        _state_contract(envelope)
    return envelope


class SQLiteProjection:
    def __init__(self, path: str | Path, *, fault_hook: Callable[[str], None] | None = None):
        self.path = Path(path).resolve()
        self.fault_hook = fault_hook or (lambda _point: None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with store_write_lock(self.path.parent, operation="projection-initialize"):
            existed = self.path.exists()
            self.journal = JournalStateStore(self.path, fault_hook=self.fault_hook)
            if self.journal.schema_version != 1:
                raise ProjectionError("Unsupported target journal schema")
            revision, envelope = self.journal.read_snapshot()
            if not existed:
                self.journal.save({"schema_version": 1, "projection_format": FORMAT,
                                   "cursor": None, "embedding_contract": None,
                                   "items": [], "lineage": []}, expected_revision=revision,
                                  operation="projection-initialize")
            else:
                _validate(envelope)  # even an unrelated empty journal is rejected

    def read(self) -> dict:
        if self.journal.schema_version != 1:
            raise ProjectionError("Unsupported target journal schema")
        return _validate(self.journal.read_snapshot()[1])

    def apply_delta(self, packet: dict) -> dict:
        from .projection_delta import apply_sqlite_delta

        return apply_sqlite_delta(self, packet)

    def publish(self, snapshot: dict) -> dict:
        proposed, source_path = _prepared(snapshot)
        if source_path.parent == self.path.parent:
            raise ProjectionError("Projection requires a dedicated directory outside the source directory")
        self.fault_hook("before_publish")
        for _ in range(8):
            revision, current = self.journal.read_snapshot()
            _validate(current)
            cursor = current["cursor"]
            desired = proposed["cursor"]
            if cursor is not None:
                if cursor["source_store_id"] != desired["source_store_id"]:
                    raise ProjectionError("Projection is permanently bound to another authority")
                if cursor["source_revision"] > desired["source_revision"]:
                    raise ProjectionStale("Publication would roll back the projection")
                if cursor["source_revision"] == desired["source_revision"]:
                    if _encode(current) != _encode(proposed):
                        raise ProjectionError("Different contents claim the same source revision")
                    return dict(cursor)
            try:
                self.journal.save(proposed, expected_revision=revision, operation="projection-replace")
                self.fault_hook("after_publish")
                return dict(desired)
            except RevisionConflict:
                continue
            except Exception:
                # A lost acknowledgement cannot undo a committed publication.
                if _encode(self.read()) == _encode(proposed):
                    return dict(desired)
                raise
        raise RevisionConflict("Projection publication remained contended")


def _status(snapshot: dict, projection: SQLiteProjection) -> dict:
    expected, source_path = _prepared(snapshot)
    if source_path.parent == projection.path.parent:
        raise ProjectionError("Projection and source directories must differ")
    actual = projection.read()
    return {"matches_pinned_source": _encode(actual) == _encode(expected),
            "source": expected["cursor"], "projection": actual["cursor"],
            "record_count": len(actual["items"])}


def projection_status(source: JournalStateStore, projection: SQLiteProjection) -> dict:
    return _status(prepare_source(source), projection)


def reconcile(source: JournalStateStore, projection: SQLiteProjection) -> dict:
    projection.fault_hook("before_reconcile")
    snapshot = prepare_source(source)
    projection.fault_hook("after_prepare")
    published = projection.publish(snapshot)
    projection.fault_hook("after_reconcile")
    status = projection_status(source, projection)
    return {**status, "published": published}


def query_projection(source: JournalStateStore, projection: SQLiteProjection, *,
                     vector, embedding_identity: dict, scope: dict, top_k: int = 10,
                     as_of: float) -> dict:
    scope = json.loads(_encode(_scope(scope)))
    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise ProjectionError("top_k must be between 1 and 1000")
    if type(as_of) not in (float, int) or not math.isfinite(as_of) or as_of < 0:
        raise ProjectionError("as_of must be a finite nonnegative timestamp")
    embedding_identity = json.loads(_encode(embedding_identity))
    query = np.array(vector, copy=True)
    if query.dtype.kind not in "ifu":
        raise ProjectionError("Query requires numeric vector members")
    query = query.astype(np.float64)
    if query.ndim != 1 or not query.size or not np.isfinite(query).all():
        raise ProjectionError("Query requires a finite nonempty vector")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm == 0:
        raise ProjectionError("Query vector must have a finite nonzero norm")
    snapshot = prepare_source(source)  # query linearizes at this authority read
    expected, source_path = _prepared(snapshot)
    if source_path.parent == projection.path.parent:
        raise ProjectionError("Projection and source directories must differ")
    actual = projection.read()
    if actual["cursor"] is None or _encode(actual) != _encode(expected):
        raise ProjectionStale("Projection does not match the pinned authority snapshot")
    contract = actual["embedding_contract"]
    if contract is None:
        raise ProjectionError("Empty source has no embedding contract")
    validate_embedding_contract(actual, embedding_identity, query)
    ranked = []
    suppressed = []
    for record in actual["items"]:
        meta = record.get("meta") or {}
        if any(meta.get(key) != scope[key] for key in SCOPE_KEYS):
            continue
        reason = suppression_reason(meta, now=as_of)
        if reason is not None:
            suppressed.append({"id": record["id"], "reason": reason})
            continue
        stored = np.asarray(record["embedding"], dtype=np.float64)
        length = float(np.linalg.norm(stored))
        score = float(np.dot(stored / length, query / norm)) if length else 0.0
        ranked.append({"memory": record, "score": max(-1.0, min(1.0, score))})
    ranked.sort(key=lambda hit: (-hit["score"], hit["memory"]["id"]))
    suppressed.sort(key=lambda item: item["id"])
    return {"source": actual["cursor"], "as_of": as_of, "scope": scope,
            "results": ranked[:top_k], "suppressed": suppressed}
