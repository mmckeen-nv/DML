"""Explicit, compare-and-set retirement with durable historical receipts.

Retirement marks an existing item deleted without removing its text, vector,
provenance, or capacity charge. It neither calls a backend nor changes authority
formats. A receipt acknowledges its historical commit, not the current state.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from ..journal import IdempotencyConflict, JournalStateStore, RevisionConflict
from ..persistence import validate_record, validate_snapshot
from ..store_lock import store_write_lock
from .receipt_ingestion import (
    ReceiptCommitRejected, ReceiptCommitUncertain, SCOPE_KEYS, _strict_json,
)


class ReceiptMemoryNotFound(LookupError):
    """No item is visible under the exact requested receipt scope."""


class ReceiptLifecycleConflict(RuntimeError):
    """The target changed or cannot accept a new retirement decision."""


def _canonical_json(value) -> str:
    try:
        _strict_json(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, RecursionError, OverflowError) as exc:
        raise ValueError("Retirement values must be finite JSON") from exc


def _valid_digest(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def memory_digest(record: dict) -> str:
    """Hash the complete finite JSON record without coercing metadata or vectors.

    Pass the exact persisted record (or its unchanged ``MemoryItem.to_dict()``
    representation). Every field participates, including unknown provenance
    fields; integers, floats, and booleans remain distinguishable.
    """
    if type(record) is not dict:
        raise ValueError("A memory digest requires a complete memory record")
    raw = _canonical_json(record)
    validate_record(json.loads(raw))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_retirement_request(memory_id: int, *, expected_memory_digest: str,
                                 reason: str, tenant_id: str, client_id=None,
                                 session_id=None, instance_id=None) -> tuple[dict, str]:
    """Freeze a versioned retirement intent and its canonical SHA-256 digest."""
    if type(memory_id) is not int or memory_id < 0:
        raise ValueError("Retirement memory_id must be a nonnegative integer")
    if not _valid_digest(expected_memory_digest):
        raise ValueError("expected_memory_digest must be a canonical SHA-256 digest")
    if type(reason) is not str or not reason.strip() or len(reason.encode("utf-8")) > 1024:
        raise ValueError("Retirement reason must be nonempty and at most 1024 UTF-8 bytes")
    scope = dict(zip(SCOPE_KEYS, (tenant_id, client_id, session_id, instance_id)))
    for name, value in scope.items():
        if value is None and name != "tenant_id":
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise ValueError("Retirement scope members must be nonempty strings of at most 256 UTF-8 bytes")
    request = {"schema_version": "dml-retire-request-v1", "memory_id": memory_id,
               "expected_memory_digest": expected_memory_digest, "reason": reason,
               "scope": scope}
    raw = _canonical_json(request)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Retirement request exceeds 1 MiB")
    return json.loads(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_request(request: dict, request_digest: str) -> dict:
    # Freeze first, then validate that exact frozen object. A caller retaining
    # references cannot switch the target, scope, or reason during storage I/O.
    if type(request) is not dict:
        raise ValueError("Retirement request must be an object")
    raw = _canonical_json(request)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Retirement request exceeds 1 MiB")
    frozen = json.loads(raw)
    if set(frozen) != {"schema_version", "memory_id", "expected_memory_digest", "reason", "scope"}:
        raise ValueError("Invalid retirement request fields")
    if type(frozen["scope"]) is not dict or set(frozen["scope"]) != set(SCOPE_KEYS):
        raise ValueError("Retirement requires the complete receipt scope")
    canonical, digest = canonical_retirement_request(
        frozen["memory_id"], expected_memory_digest=frozen["expected_memory_digest"],
        reason=frozen["reason"], **frozen["scope"])
    if raw != _canonical_json(canonical) or not _valid_digest(request_digest) or digest != request_digest:
        raise ValueError("Retirement request does not match its canonical digest")
    return canonical


def _retirement_receipt(receipt: dict, request: dict) -> dict:
    """A generic journal receipt must actually acknowledge this operation."""
    record = receipt["result"]["memory"]
    meta = record.get("meta") or {}
    decision = {"schema_version": "dml-retirement-decision-v1",
                "prior_memory_digest": request["expected_memory_digest"],
                "reason": request["reason"]}
    if (type(record.get("id")) is not int or record["id"] != request["memory_id"]
            or meta.get("memory_state") != "deleted"
            or _canonical_json(meta.get("retirement_decision")) != _canonical_json(decision)):
        raise ReceiptLifecycleConflict("Existing receipt does not acknowledge the requested retirement")
    return receipt


def retire_receipted(journal: JournalStateStore, *, request: dict, request_digest: str,
                    key: str, hydrate: Callable[[int, dict], None],
                    degraded: Callable[[Exception], None]) -> dict:
    """Retire one exact visible item, or replay the identical historical receipt.

    All three CAS attempts rebuild from the current authority; unrelated writes
    may be retried, but a changed target always requires a new caller decision.
    The ingestion and retirement APIs share the same scoped key namespace.
    """
    request = _validated_request(request, request_digest)
    scope = request["scope"]
    existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
    if existing is not None:
        return _retirement_receipt(existing, request)
    with store_write_lock(journal.path.parent, operation="receipt-retire", timeout_ms=30000):
        for _ in range(3):
            existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
            if existing is not None:
                return _retirement_receipt(existing, request)
            revision, payload = journal.read_snapshot()
            validate_snapshot(payload)
            record = next((item for item in payload["items"] if item["id"] == request["memory_id"]), None)
            meta = record.get("meta") if record is not None else None
            if type(meta) is not dict or any(
                type(meta.get(name)) is not type(value) or meta.get(name) != value
                for name, value in scope.items()
            ):
                # Do not distinguish missing, lineage-only, and other-scope IDs.
                raise ReceiptMemoryNotFound("Memory is not available in the requested scope")
            assert record is not None
            prior_digest = memory_digest(record)
            if prior_digest != request["expected_memory_digest"]:
                raise ReceiptLifecycleConflict("Memory changed since the retirement decision")
            state = str(meta.get("memory_state") or meta.get("lifecycle_state") or "").strip().lower()
            if state == "deleted" or "retirement_decision" in meta:
                raise ReceiptLifecycleConflict("Memory already has a retirement state or decision")
            meta["memory_state"] = "deleted"
            meta["retirement_decision"] = {"schema_version": "dml-retirement-decision-v1",
                                           "prior_memory_digest": prior_digest,
                                           "reason": request["reason"]}
            validate_snapshot(payload)
            try:
                receipt = journal.save_with_receipt(payload, scope=scope, key=key,
                    request_digest=request_digest, result={"memory": record},
                    expected_revision=revision, operation="retire-receipt-v1")
            except RevisionConflict:
                continue
            except IdempotencyConflict:
                raise
            except Exception as exc:
                # A failed acknowledgement may follow a successful transaction.
                # Never compensate or delete: reconcile against durable receipts.
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
            return _retirement_receipt(receipt, request)
    raise RevisionConflict("Receipt retirement exhausted three revision retries")
