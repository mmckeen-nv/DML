"""Content-only updates with pinned embedding space and historical receipts.

Text, vector, and the latest content-update decision change atomically. Scope,
authority, lifecycle, identity, capacity, and all other provenance are preserved.
A receipt describes its historical commit, even after later lifecycle changes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from ..journal import IdempotencyConflict, JournalStateStore, RevisionConflict
from ..persistence import validate_record, validate_snapshot
from ..store_lock import store_write_lock
from .receipt_ingestion import (
    ReceiptCommitRejected, ReceiptCommitUncertain, ReceiptEmbeddingError,
    ReceiptEmbeddingCompatibilityError, SCOPE_KEYS, validate_embedding_contract,
)
from .receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, _canonical_json, _valid_digest,
    canonical_retirement_request, memory_digest,
)


def canonical_update_request(memory_id: int, *, text: str,
                             expected_memory_digest: str, reason: str,
                             tenant_id: str, client_id=None, session_id=None,
                             instance_id=None) -> tuple[dict, str]:
    """Freeze the complete record precondition, replacement text, and scope."""
    base, _ = canonical_retirement_request(memory_id,
        expected_memory_digest=expected_memory_digest, reason=reason,
        tenant_id=tenant_id, client_id=client_id, session_id=session_id,
        instance_id=instance_id)
    if type(text) is not str or not text.strip():
        raise ValueError("Content updates require nonempty text")
    request = {**base, "schema_version": "dml-update-request-v1", "text": text}
    raw = _canonical_json(request)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Content update request exceeds 1 MiB")
    return json.loads(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_request(request: dict, request_digest: str) -> dict:
    if type(request) is not dict:
        raise ValueError("Content update request must be an object")
    raw = _canonical_json(request)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Content update request exceeds 1 MiB")
    frozen = json.loads(raw)
    if set(frozen) != {"schema_version", "memory_id", "text",
                       "expected_memory_digest", "reason", "scope"}:
        raise ValueError("Invalid content update request fields")
    if type(frozen["scope"]) is not dict or set(frozen["scope"]) != set(SCOPE_KEYS):
        raise ValueError("Content updates require the complete receipt scope")
    canonical, digest = canonical_update_request(frozen["memory_id"],
        text=frozen["text"], expected_memory_digest=frozen["expected_memory_digest"],
        reason=frozen["reason"], **frozen["scope"])
    if raw != _canonical_json(canonical) or not _valid_digest(request_digest) or digest != request_digest:
        raise ValueError("Content update request does not match its canonical digest")
    return canonical


def _decision(request: dict) -> dict:
    return {"schema_version": "dml-content-update-decision-v1",
            "prior_memory_digest": request["expected_memory_digest"],
            "reason": request["reason"]}


def _recognized_decision(decision) -> bool:
    if not (type(decision) is dict
            and set(decision) == {"schema_version", "prior_memory_digest", "reason"}
            and decision["schema_version"] == "dml-content-update-decision-v1"
            and _valid_digest(decision["prior_memory_digest"])
            and type(decision["reason"]) is str and bool(decision["reason"].strip())):
        return False
    try:
        return len(decision["reason"].encode("utf-8")) <= 1024
    except UnicodeError:
        return False


def _terminal(meta: dict) -> bool:
    # Check both aliases independently so an active primary field cannot hide
    # a terminal legacy state or an existing retirement/supersession decision.
    return ("retirement_decision" in meta or "supersession_decision" in meta
            or meta.get("superseded_by") is not None
            or any(str(meta.get(name) or "").strip().lower() in {"deleted", "retired", "superseded"}
                   for name in ("memory_state", "lifecycle_state")))


def _scoped(meta, scope: dict) -> bool:
    return type(meta) is dict and all(
        name in meta and type(meta[name]) is type(value) and meta[name] == value
        for name, value in scope.items())


def _target(payload: dict, request: dict) -> dict:
    record = next((item for item in payload["items"] if item["id"] == request["memory_id"]), None)
    if record is None or not _scoped(record.get("meta"), request["scope"]):
        raise ReceiptMemoryNotFound("Memory is not available in the requested scope")
    if memory_digest(record) != request["expected_memory_digest"]:
        raise ReceiptLifecycleConflict("Memory changed since the content update decision")
    meta = record["meta"]
    if _terminal(meta):
        raise ReceiptLifecycleConflict("Memory already has a retirement or supersession state or decision")
    if "content_update_decision" in meta and not _recognized_decision(meta["content_update_decision"]):
        raise ReceiptLifecycleConflict("Memory has an unrecognized content update decision")
    if record["text"] == request["text"]:
        raise ReceiptLifecycleConflict("Content update text is unchanged")
    return record


def _update_receipt(receipt: dict, request: dict) -> dict:
    """A generic receipt must actually acknowledge the requested text update."""
    try:
        record = receipt["result"]["memory"]
        validate_record(record)
        meta = record.get("meta")
        valid = (type(record["id"]) is int and record["id"] == request["memory_id"]
                 and record["text"] == request["text"] and bool(record["embedding"])
                 and _scoped(meta, request["scope"]) and not _terminal(meta)
                 and _canonical_json(meta.get("content_update_decision")) == _canonical_json(_decision(request)))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ReceiptLifecycleConflict("Existing receipt does not acknowledge the requested content update")
    return receipt


def _frozen_identity(embedding_space: Callable[[], dict]) -> dict:
    try:
        identity = embedding_space()
        if type(identity) is not dict:
            raise ValueError("Embedding identity must be an object")
        return json.loads(_canonical_json(identity))
    except ReceiptEmbeddingCompatibilityError:
        raise
    except Exception as exc:
        raise ReceiptEmbeddingCompatibilityError("Embedding identity is unavailable or invalid") from exc


def _contract(payload: dict, identity: dict, vector=None) -> None:
    contract = validate_embedding_contract(payload, identity, vector)
    # Canonical JSON equality distinguishes nested bool/int/float identities;
    # ordinary Python dictionary equality would consider True equal to 1.
    if contract is None or _canonical_json(contract["identity"]) != _canonical_json(identity):
        raise ReceiptEmbeddingCompatibilityError("Configured embedding identity differs from the committed embedding space")


def _check_identity(embedding_space: Callable[[], dict], identity: dict) -> None:
    if _canonical_json(_frozen_identity(embedding_space)) != _canonical_json(identity):
        raise ReceiptEmbeddingCompatibilityError("Embedding identity changed while preparing the content update")


def update_receipted(journal: JournalStateStore, *, request: dict,
                    request_digest: str, key: str, embed: Callable,
                    embedding_space: Callable[[], dict],
                    hydrate: Callable[[int, dict], None],
                    degraded: Callable[[Exception], None]) -> dict:
    """Replace one exact scoped record's content, or replay its durable receipt.

    Models run outside store ownership. The full target and embedding contract
    are checked before preparation and on each of three bounded CAS attempts.
    Retries acknowledge history before consulting current records or models.
    """
    request = _validated_request(request, request_digest)
    scope = request["scope"]

    def replay(*, uncertain: bool = False) -> dict | None:
        try:
            existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
        except IdempotencyConflict:
            raise
        except Exception as lookup_error:
            if uncertain:
                raise ReceiptCommitUncertain("Receipt outcome unavailable; retry identical request and key") from lookup_error
            raise
        return _update_receipt(existing, request) if existing is not None else None

    existing = replay()
    if existing is not None:
        return existing
    try:
        _, preparation_state = journal.read_snapshot()
        validate_snapshot(preparation_state)
        _target(preparation_state, request)
        identity = _frozen_identity(embedding_space)
        _contract(preparation_state, identity)
    except Exception:
        # A raw journal writer can commit after our lookup and before this
        # snapshot, even while advisory ownership excludes service writers.
        existing = replay(uncertain=True)
        if existing is not None:
            return existing
        raise
    try:
        # Copy immediately: custom embedders may retain and mutate their result.
        prepared = np.array(embed(request["text"]), copy=True)
        if prepared.ndim != 1 or not prepared.size or prepared.dtype.kind not in "fiu":
            raise ValueError("Content update embedding must be a nonempty numeric vector")
        with np.errstate(over="raise", invalid="raise"):
            vector = prepared.astype(np.float32, copy=True)
        if not np.isfinite(vector).all():
            raise ValueError("Content update embedding must be finite")
    except Exception as exc:
        existing = replay(uncertain=True)
        if existing is not None:
            return existing
        raise ReceiptEmbeddingError("Receipt embedding unavailable; retry identical request and key") from exc
    existing = replay()
    if existing is not None:
        return existing
    try:
        _check_identity(embedding_space, identity)
        _contract(preparation_state, identity, vector)
    except Exception:
        existing = replay(uncertain=True)
        if existing is not None:
            return existing
        raise
    with store_write_lock(journal.path.parent, operation="receipt-update", timeout_ms=30000):
        for _ in range(3):
            existing = replay()
            if existing is not None:
                return existing
            try:
                revision, payload = journal.read_snapshot()
                validate_snapshot(payload)
                record = _target(payload, request)
                _check_identity(embedding_space, identity)
                _contract(payload, identity, vector)
            except Exception:
                existing = replay(uncertain=True)
                if existing is not None:
                    return existing
                raise
            record["text"] = request["text"]
            record["embedding"] = vector.tolist()
            record["meta"]["content_update_decision"] = _decision(request)
            validate_snapshot(payload)
            try:
                receipt = journal.save_with_receipt(payload, scope=scope, key=key,
                    request_digest=request_digest, result={"memory": record},
                    expected_revision=revision, operation="update-receipt-v1")
            except RevisionConflict:
                continue
            except IdempotencyConflict:
                raise
            except Exception as exc:
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
            return _update_receipt(receipt, request)
    raise RevisionConflict("Receipt content update exhausted three revision retries")
