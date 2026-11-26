"""Explicit snapshot promotion and merge with complete, immutable source proof.

A caller supplies the derived text. Eligible base sources remain untouched, and
one independent level-one memory is appended without increasing source authority,
recency, salience or fidelity. This operation never infers semantic truth.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable

import numpy as np

from ..journal import IdempotencyConflict, JournalStateStore, RevisionConflict
from ..memory_store import _merge_disabled
from ..persistence import validate_record, validate_snapshot
from ..store_lock import store_write_lock
from .lifecycle import suppression_reason
from .receipt_ingestion import (
    ReceiptCapacityError, ReceiptCommitRejected, ReceiptCommitUncertain,
    ReceiptEmbeddingError, SCOPE_KEYS,
)
from .receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, _canonical_json,
    _valid_digest, canonical_retirement_request, memory_digest,
)
from .receipt_update import (
    _check_identity, _contract, _frozen_identity, _recognized_decision, _scoped,
)

_MAX_BYTES = 1024 * 1024
_SOURCE_METADATA = frozenset({
    "summary", "source", "source_id", "source_ids", "provenance",
    "content_update_decision",
})


def _bounded(value, description: str) -> str:
    raw = _canonical_json(value)
    if len(raw.encode("utf-8")) > _MAX_BYTES:
        raise ValueError(f"{description} exceeds 1 MiB")
    return raw


def canonical_promotion_request(sources, *, text: str, reason: str,
                                tenant_id: str, client_id=None, session_id=None,
                                instance_id=None) -> tuple[dict, str]:
    """Freeze sorted, unique source preconditions and caller-directed text."""
    base, _ = canonical_retirement_request(0, expected_memory_digest="0" * 64,
        reason=reason, tenant_id=tenant_id, client_id=client_id,
        session_id=session_id, instance_id=instance_id)
    if type(text) is not str or not text.strip():
        raise ValueError("Promotion requires nonempty text")
    if type(sources) is not list or not 1 <= len(sources) <= 32:
        raise ValueError("Promotion requires between one and 32 sources")
    frozen_sources = []
    for source in sources:
        if (type(source) is not dict
                or set(source) != {"memory_id", "expected_memory_digest"}
                or type(source["memory_id"]) is not int or source["memory_id"] < 0
                or not _valid_digest(source["expected_memory_digest"])):
            raise ValueError("Invalid promotion source precondition")
        frozen_sources.append(dict(source))
    if len({source["memory_id"] for source in frozen_sources}) != len(frozen_sources):
        raise ValueError("Promotion requires distinct source memories")
    frozen_sources.sort(key=lambda source: source["memory_id"])
    request = {"schema_version": "dml-promotion-request-v1", "sources": frozen_sources,
               "text": text, "reason": reason, "scope": base["scope"]}
    raw = _bounded(request, "Promotion request")
    return json.loads(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_request(request: dict, request_digest: str) -> dict:
    if type(request) is not dict:
        raise ValueError("Promotion request must be an object")
    raw = _bounded(request, "Promotion request")
    frozen = json.loads(raw)
    if set(frozen) != {"schema_version", "sources", "text", "reason", "scope"}:
        raise ValueError("Invalid promotion request fields")
    if type(frozen["scope"]) is not dict or set(frozen["scope"]) != set(SCOPE_KEYS):
        raise ValueError("Promotion requires the complete receipt scope")
    canonical, digest = canonical_promotion_request(frozen["sources"],
        text=frozen["text"], reason=frozen["reason"], **frozen["scope"])
    if raw != _canonical_json(canonical) or not _valid_digest(request_digest) or digest != request_digest:
        raise ValueError("Promotion request does not match its canonical digest")
    return canonical


def _base_source(record: dict, *, merging: bool, now: float | None) -> None:
    meta = record["meta"]
    for field in ("summary_of", "children"):
        lineage = record.get(field, [])
        if (type(lineage) is not list
                or any(type(value) is not int for value in lineage)
                or lineage not in ([], [record["id"]])):
            raise ReceiptLifecycleConflict("Promotion requires base memories without indirect lineage")
    if record["level"] != 0 or "promotion_decision" in meta or "abstracted_from" in meta:
        raise ReceiptLifecycleConflict("Promotion requires base memories without prior abstraction")
    if ("retirement_decision" in meta or "supersession_decision" in meta
            or meta.get("superseded_by") is not None):
        raise ReceiptLifecycleConflict("Promotion source has a terminal lifecycle decision")
    for name in ("memory_state", "lifecycle_state"):
        state = meta.get(name)
        if state is not None and (type(state) is not str or state.strip().lower() not in {"", "active"}):
            raise ReceiptLifecycleConflict("Promotion source is not active")
    if type(meta.get("source_trust")) is not str or meta["source_trust"] not in {"trusted", "verified"}:
        raise ReceiptLifecycleConflict("Promotion requires explicitly trusted or verified sources")
    # Historical proof retains the original expiry but does not expire a receipt.
    if suppression_reason(meta, now=float("-inf") if now is None else now) is not None:
        raise ReceiptLifecycleConflict("Promotion source is not eligible for normal retrieval")
    if merging and (_merge_disabled(meta) or ("no_merge" in meta and meta["no_merge"] is not False)):
        raise ReceiptLifecycleConflict("Promotion source forbids merging or has an ambiguous merge policy")
    if "content_update_decision" in meta and not _recognized_decision(meta["content_update_decision"]):
        raise ReceiptLifecycleConflict("Promotion source has an unrecognized content update decision")


def _shared_metadata(records: list[dict]) -> dict:
    metadata = [{name: value for name, value in record["meta"].items()
                 if name not in _SOURCE_METADATA} for record in records]
    first = _canonical_json(metadata[0])
    if any(_canonical_json(candidate) != first for candidate in metadata[1:]):
        raise ReceiptLifecycleConflict("Promotion sources have incompatible authority or metadata")
    return json.loads(first)


def _proof(records: list[dict], request: dict) -> dict:
    proof = {"schema_version": "dml-promotion-decision-v1", "reason": request["reason"],
             "sources": [{"memory_digest": source["expected_memory_digest"], "memory": record}
                         for source, record in zip(request["sources"], records)]}
    return json.loads(_bounded(proof, "Promotion source proof"))


def _check_sources(records: list[dict], request: dict, *, now: float | None) -> None:
    # Every scope is checked before any digest or eligibility result is exposed.
    if len(records) != len(request["sources"]) or any(
            not _scoped(record.get("meta"), request["scope"]) for record in records):
        raise ReceiptMemoryNotFound("Memory is not available in the requested scope")
    for source, record in zip(request["sources"], records):
        if record["id"] != source["memory_id"] or memory_digest(record) != source["expected_memory_digest"]:
            raise ReceiptLifecycleConflict("Promotion source changed since the caller decision")
        _base_source(record, merging=len(records) > 1, now=now)
    _shared_metadata(records)
    _proof(records, request)


def _sources(payload: dict, request: dict, *, capacity: int) -> tuple[list[dict], int]:
    by_id = {record["id"]: record for record in payload["items"]}
    records = [by_id[source["memory_id"]] for source in request["sources"]
               if source["memory_id"] in by_id]
    _check_sources(records, request, now=time.time())
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Promotion capacity must be a positive integer")
    if len(payload["items"]) >= capacity:
        raise ReceiptCapacityError("Promotion capacity reached; explicit lifecycle action required")
    next_id = payload.get("next_id", 0)
    if type(next_id) is not int or next_id < 0:
        raise ValueError("Invalid persisted memory ID allocator")
    ids = [record["id"] for bucket in ("items", "lineage") for record in payload.get(bucket, [])]
    return records, max(next_id, max(ids, default=-1) + 1)


def _output(records: list[dict], request: dict, *, ident: int, vector: list) -> dict:
    meta = _shared_metadata(records)
    meta["no_merge"] = True
    meta["promotion_decision"] = _proof(records, request)
    source_ids = [source["memory_id"] for source in request["sources"]]
    record = {"schema_version": 1, "id": ident, "text": request["text"],
              "timestamp": min(source["timestamp"] for source in records),
              "salience": min(source["salience"] for source in records),
              "fidelity": min(source["fidelity"] for source in records), "level": 1,
              "meta": meta, "summary_of": source_ids, "children": list(source_ids),
              "embedding": vector}
    validate_record(record)
    return json.loads(_bounded(record, "Promoted memory"))


def _promotion_receipt(receipt: dict, request: dict) -> dict:
    """Prove the acknowledged result from immutable sources, never current state."""
    try:
        record = receipt["result"]["memory"]
        validate_record(record)
        proof = record["meta"]["promotion_decision"]
        if type(proof) is not dict or set(proof) != {"schema_version", "reason", "sources"}:
            raise ValueError("Invalid promotion proof")
        evidence = proof["sources"]
        if type(evidence) is not list or len(evidence) != len(request["sources"]):
            raise ValueError("Invalid promotion proof sources")
        records = []
        for source, binding in zip(request["sources"], evidence):
            if (type(binding) is not dict or set(binding) != {"memory_digest", "memory"}
                    or binding["memory_digest"] != source["expected_memory_digest"]):
                raise ValueError("Invalid promotion source binding")
            validate_record(binding["memory"])
            records.append(binding["memory"])
        _check_sources(records, request, now=None)
        dimensions = {len(source["embedding"]) for source in records}
        if (0 in dimensions or len(dimensions) != 1
                or len(record["embedding"]) not in dimensions
                or record["id"] <= max(source["id"] for source in records)):
            raise ValueError("Invalid promoted memory identity or historical vector dimension")
        expected = _output(records, request, ident=record["id"], vector=record["embedding"])
        if _canonical_json(record) != _canonical_json(expected):
            raise ValueError("Invalid promoted memory semantics")
    except (KeyError, TypeError, ValueError, ReceiptLifecycleConflict, ReceiptMemoryNotFound) as exc:
        raise ReceiptLifecycleConflict("Existing receipt does not acknowledge the requested promotion") from exc
    return receipt


def promote_receipted(journal: JournalStateStore, *, request: dict,
                     request_digest: str, key: str, embed: Callable,
                     embedding_space: Callable[[], dict], capacity: int,
                     hydrate: Callable[[int, dict], None],
                     degraded: Callable[[Exception], None]) -> dict:
    """Append one caller-directed derived memory, or replay its durable receipt."""
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
        return _promotion_receipt(existing, request) if existing is not None else None

    existing = replay()
    if existing is not None:
        return existing
    try:
        _, preparation_state = journal.read_snapshot()
        validate_snapshot(preparation_state)
        _sources(preparation_state, request, capacity=capacity)
        identity = _frozen_identity(embedding_space)
        _contract(preparation_state, identity)
    except Exception:
        existing = replay(uncertain=True)
        if existing is not None:
            return existing
        raise
    try:
        prepared = np.array(embed(request["text"]), copy=True)
        if prepared.ndim != 1 or not prepared.size or prepared.dtype.kind not in "fiu":
            raise ValueError("Promotion embedding must be a nonempty numeric vector")
        with np.errstate(over="raise", invalid="raise"):
            vector = prepared.astype(np.float32, copy=True)
        if not np.isfinite(vector).all():
            raise ValueError("Promotion embedding must be finite")
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
    with store_write_lock(journal.path.parent, operation="receipt-promotion", timeout_ms=30000):
        for _ in range(3):
            existing = replay()
            if existing is not None:
                return existing
            try:
                revision, payload = journal.read_snapshot()
                validate_snapshot(payload)
                sources, ident = _sources(payload, request, capacity=capacity)
                _check_identity(embedding_space, identity)
                _contract(payload, identity, vector)
                record = _output(sources, request, ident=ident, vector=vector.tolist())
                payload["items"].append(record)
                payload["next_id"] = ident + 1
                validate_snapshot(payload)
            except Exception:
                existing = replay(uncertain=True)
                if existing is not None:
                    return existing
                raise
            try:
                receipt = journal.save_with_receipt(payload, scope=scope, key=key,
                    request_digest=request_digest, result={"memory": record},
                    expected_revision=revision, operation="promote-receipt-v1")
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
            return _promotion_receipt(receipt, request)
    raise RevisionConflict("Receipt promotion exhausted three revision retries")
