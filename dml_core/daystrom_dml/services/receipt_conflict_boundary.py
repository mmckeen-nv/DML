"""Invocation-owned evidence for a narrowly precommit stale-record rejection.

This is not an exception-name classifier. Only scoped, validated snapshots at
specific receipt-service sites can mint a conflict; entering any save attempt
permanently disables minting for the entire invocation, including later rebases.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json

from .receipt_lifecycle import ReceiptLifecycleConflict, memory_digest

_ACTIVE: ContextVar[ReceiptConflictBoundary | None] = ContextVar("receipt_conflict_boundary", default=None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class ReceiptPreconditionConflict(ReceiptLifecycleConflict):
    """A diagnostic exception; only its owning boundary may authenticate it."""


class ReceiptConflictBoundary:
    def __init__(self, *, journal, operation, request_digest, key, scope):
        if operation not in ("retire", "supersede"):
            raise ValueError("Unsupported precommit recovery operation")
        self._journal = journal
        self._operation = operation
        self._request_digest = request_digest
        self._key = key
        self._scope = _canonical(scope)
        self._owner = object()
        self._save_entered = False
        self._error = None
        self._proof = None
        self._token = None
        self._used = False
        self._consumed = False

    def __enter__(self):
        if self._used or _ACTIVE.get() is not None:
            raise RuntimeError("Receipt conflict boundary is one-shot and cannot nest")
        self._used = True
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *args):
        _ACTIVE.reset(self._token)

    def _matches(self, journal, operation, request_digest, key, scope):
        return (journal is self._journal and operation == self._operation
                and request_digest == self._request_digest and key == self._key
                and _canonical(scope) == self._scope)

    def authenticate(self, error):
        if (type(error) is not ReceiptPreconditionConflict or error is not self._error
                or getattr(error, "_owner", None) is not self._owner or self._save_entered
                or self._proof is None or self._consumed):
            return None
        self._consumed = True
        return json.loads(self._proof)


def note_journal_save_entered(journal):
    """Every authoritative journal save entry calls this; inactive is a no-op."""
    boundary = _ACTIVE.get()
    if boundary is not None:
        boundary._save_entered = True


def note_save_entered(journal, *, operation, request_digest, key, scope):
    """Call immediately before save_with_receipt; never reset after CAS failure."""
    boundary = _ACTIVE.get()
    if boundary is not None:
        # Even an unexpected nested write invalidates no-write attribution.
        boundary._save_entered = True


def reject_stale_record(journal, *, operation, request_digest, key, scope,
                        revision, snapshot, record, expected_digest, record_role, message):
    """Called only after service validation/scope checks while owning write lock."""
    boundary = _ACTIVE.get()
    if (boundary is None or boundary._save_entered
            or not boundary._matches(journal, operation, request_digest, key, scope)):
        raise ReceiptLifecycleConflict(message)
    observed_digest = memory_digest(record)
    if observed_digest == expected_digest:
        raise RuntimeError("Precommit conflict must prove different record digests")
    if any(type(record['meta'].get(k)) is not type(v) or record['meta'].get(k) != v for k, v in scope.items()):
        raise RuntimeError("Precommit conflict scope is not authenticated")
    if not journal.observe_precommit_receipt_absence(
            scope=scope, key=key, request_digest=request_digest,
            revision=revision, snapshot=snapshot):
        raise ReceiptLifecycleConflict(message)
    proof = {"schema_version": "dml-precommit-conflict-v1", "operation": operation,
             "phase": "validated_scoped_snapshot_before_first_save", "request_digest": request_digest,
             "key": key, "scope": deepcopy(scope), "snapshot_revision": revision,
             "snapshot_digest": hashlib.sha256(_canonical(snapshot)).hexdigest(),
             "snapshot": deepcopy(snapshot),
             "record_role": record_role, "record_id": record['id'],
             "expected_memory_digest": expected_digest, "observed_memory_digest": observed_digest,
             "observed_record": deepcopy(record), "save_entered": False, "receipt_absent_at_snapshot": True}
    error = ReceiptPreconditionConflict(message)
    error._owner = boundary._owner
    boundary._error = error
    boundary._proof = _canonical(proof)
    raise error
