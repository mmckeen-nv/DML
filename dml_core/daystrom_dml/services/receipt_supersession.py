"""Explicit supersession links with two-record CAS and historical receipts.

Only the source's lifecycle metadata changes. The replacement remains an
independent memory; a link neither promotes its authority nor promises that it
will remain eligible. Historical receipt replay does not resolve link chains.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable

from ..journal import IdempotencyConflict, JournalStateStore, RevisionConflict
from ..persistence import validate_snapshot
from ..store_lock import store_write_lock
from .lifecycle import suppression_reason
from .receipt_ingestion import ReceiptCommitRejected, ReceiptCommitUncertain, SCOPE_KEYS
from .receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, _canonical_json, _valid_digest,
    canonical_retirement_request, memory_digest,
)


def canonical_supersession_request(memory_id: int, *, replacement_memory_id: int,
                                   expected_memory_digest: str,
                                   expected_replacement_digest: str, reason: str,
                                   tenant_id: str, client_id=None, session_id=None,
                                   instance_id=None) -> tuple[dict, str]:
    """Freeze both full-record preconditions and the exact scoped intent."""
    base, _ = canonical_retirement_request(memory_id,
        expected_memory_digest=expected_memory_digest, reason=reason,
        tenant_id=tenant_id, client_id=client_id, session_id=session_id,
        instance_id=instance_id)
    if type(replacement_memory_id) is not int or replacement_memory_id < 0:
        raise ValueError("Supersession replacement_memory_id must be a nonnegative integer")
    if replacement_memory_id == memory_id:
        raise ValueError("Supersession requires distinct source and replacement memories")
    if not _valid_digest(expected_replacement_digest):
        raise ValueError("expected_replacement_digest must be a canonical SHA-256 digest")
    request = {**base, "schema_version": "dml-supersede-request-v1",
               "replacement_memory_id": replacement_memory_id,
               "expected_replacement_digest": expected_replacement_digest}
    raw = _canonical_json(request)
    return json.loads(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_request(request: dict, request_digest: str) -> dict:
    if type(request) is not dict:
        raise ValueError("Supersession request must be an object")
    raw = _canonical_json(request)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Supersession request exceeds 1 MiB")
    frozen = json.loads(raw)
    if set(frozen) != {"schema_version", "memory_id", "replacement_memory_id",
                       "expected_memory_digest", "expected_replacement_digest", "reason", "scope"}:
        raise ValueError("Invalid supersession request fields")
    if type(frozen["scope"]) is not dict or set(frozen["scope"]) != set(SCOPE_KEYS):
        raise ValueError("Supersession requires the complete receipt scope")
    canonical, digest = canonical_supersession_request(frozen["memory_id"],
        replacement_memory_id=frozen["replacement_memory_id"],
        expected_memory_digest=frozen["expected_memory_digest"],
        expected_replacement_digest=frozen["expected_replacement_digest"],
        reason=frozen["reason"], **frozen["scope"])
    if raw != _canonical_json(canonical) or not _valid_digest(request_digest) or digest != request_digest:
        raise ValueError("Supersession request does not match its canonical digest")
    return canonical


def _decision(request: dict) -> dict:
    return {"schema_version": "dml-supersession-decision-v1",
            "prior_memory_digest": request["expected_memory_digest"],
            "replacement_memory_id": request["replacement_memory_id"],
            "replacement_memory_digest": request["expected_replacement_digest"],
            "reason": request["reason"]}


def _supersession_receipt(receipt: dict, request: dict) -> dict:
    record = receipt["result"]["memory"]
    meta = record.get("meta") or {}
    if (type(record.get("id")) is not int or record["id"] != request["memory_id"]
            or meta.get("memory_state") != "superseded"
            or type(meta.get("superseded_by")) is not int
            or meta["superseded_by"] != request["replacement_memory_id"]
            or any(type(meta.get(name)) is not type(value) or meta.get(name) != value
                   for name, value in request["scope"].items())
            or _canonical_json(meta.get("supersession_decision")) != _canonical_json(_decision(request))):
        raise ReceiptLifecycleConflict("Existing receipt does not acknowledge the requested supersession")
    return receipt


def _has_decision(meta: dict) -> bool:
    return ("retirement_decision" in meta or "supersession_decision" in meta
            or meta.get("superseded_by") is not None)


def _states(meta: dict) -> set[str]:
    # Treat aliases independently: an active primary field cannot hide a
    # terminal legacy alias and allow an old decision to be overwritten.
    return {str(meta.get(name) or "").strip().lower()
            for name in ("memory_state", "lifecycle_state")}


def supersede_receipted(journal: JournalStateStore, *, request: dict,
                       request_digest: str, key: str,
                       hydrate: Callable[[int, dict], None],
                       degraded: Callable[[Exception], None]) -> dict:
    """Link one source to an eligible replacement, or replay its exact receipt.

    Both records are checked in each snapshot and again after revision races.
    Other memories may change during bounded rebases. The full scoped key
    namespace is shared with ingestion and retirement.
    """
    request = _validated_request(request, request_digest)
    scope = request["scope"]
    existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
    if existing is not None:
        return _supersession_receipt(existing, request)
    with store_write_lock(journal.path.parent, operation="receipt-supersede", timeout_ms=30000):
        for _ in range(3):
            existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
            if existing is not None:
                return _supersession_receipt(existing, request)
            revision, payload = journal.read_snapshot()
            validate_snapshot(payload)
            records = {item["id"]: item for item in payload["items"]}
            source = records.get(request["memory_id"])
            replacement = records.get(request["replacement_memory_id"])
            for record in (source, replacement):
                meta = record.get("meta") if record is not None else None
                if type(meta) is not dict or any(
                    type(meta.get(name)) is not type(value) or meta.get(name) != value
                    for name, value in scope.items()
                ):
                    raise ReceiptMemoryNotFound("Memory is not available in the requested scope")
            assert source is not None and replacement is not None
            if memory_digest(source) != request["expected_memory_digest"]:
                raise ReceiptLifecycleConflict("Source memory changed since the supersession decision")
            if memory_digest(replacement) != request["expected_replacement_digest"]:
                raise ReceiptLifecycleConflict("Replacement memory changed since the supersession decision")
            source_meta, replacement_meta = source["meta"], replacement["meta"]
            if _has_decision(source_meta) or _states(source_meta) & {"deleted", "retired", "superseded"}:
                raise ReceiptLifecycleConflict("Source memory already has a retirement or supersession state or decision")
            if (_has_decision(replacement_meta)
                    or _states(replacement_meta) & {"quarantine", "quarantined", "suppressed", "deleted", "retired", "superseded", "expired"}
                    or suppression_reason(replacement_meta, now=time.time()) is not None):
                raise ReceiptLifecycleConflict("Replacement memory is not eligible for normal retrieval")
            source_meta["memory_state"] = "superseded"
            source_meta["superseded_by"] = request["replacement_memory_id"]
            source_meta["supersession_decision"] = _decision(request)
            validate_snapshot(payload)
            try:
                receipt = journal.save_with_receipt(payload, scope=scope, key=key,
                    request_digest=request_digest, result={"memory": source},
                    expected_revision=revision, operation="supersede-receipt-v1")
            except RevisionConflict:
                continue
            except IdempotencyConflict:
                raise
            except Exception as exc:
                # Reconcile lost acknowledgements without compensating a commit.
                try:
                    resolved = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
                except IdempotencyConflict:
                    raise
                except Exception as lookup_error:
                    raise ReceiptCommitUncertain("Receipt outcome unavailable; retry identical request and key") from lookup_error
                if resolved is None:
                    raise ReceiptCommitRejected("Receipt commit failed and no receipt is present") from exc
                receipt = resolved
            try:
                committed_revision, committed_payload = journal.read_snapshot()
                hydrate(committed_revision, committed_payload)
            except Exception as exc:
                try:
                    degraded(exc)
                except Exception:
                    pass  # Observability cannot revoke a durable success.
            return _supersession_receipt(receipt, request)
    raise RevisionConflict("Receipt supersession exhausted three revision retries")
