"""Append-only journal ingestion: prepare outside ownership, commit once, retry.

This path does not mirror to RAG, merge, promote, or generate survival ledgers.
Receipts describe a historical commit, not a promise that a memory remains live.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable

import numpy as np

from ..embeddings import RandomEmbedder, SentenceTransformerEmbedder
from ..journal import IdempotencyConflict, JournalStateStore, RevisionConflict
from ..memory_store import MemoryItem
from ..persistence import validate_snapshot
from ..store_lock import store_write_lock

SCOPE_KEYS = ("tenant_id", "client_id", "session_id", "instance_id")


class ReceiptCapacityError(ValueError):
    """An append would exceed capacity; this API never evicts implicitly."""


class ReceiptCommitUncertain(RuntimeError):
    """Storage could not establish an outcome; retry the identical request/key."""


class ReceiptCommitRejected(RuntimeError):
    """The failed commit was checked and has no receipt in the authority."""


class ReceiptEmbeddingError(RuntimeError):
    """The embedding could not be prepared; no mutation was attempted here."""


class ReceiptEmbeddingCompatibilityError(ReceiptEmbeddingError):
    """The configured embedding space cannot safely read or extend this store."""


def embedding_identity(embedder, declared=None) -> dict:
    """Resolve a versioned embedding-space declaration without invoking a backend.

    Arbitrary models must supply an immutable revision declaration; a mutable
    model name or equal vector dimensions alone never establishes compatibility.
    The declaration is an operator/backend assertion, not weight attestation.
    """
    revision: str | None
    model: str | None
    backend = type(embedder).__module__ + "." + type(embedder).__qualname__
    if type(embedder) is RandomEmbedder:
        revision = "dml-sha256-seeded-random-v1"
        model = str(embedder.dim)
        mode = "deterministic-test"
    else:
        revision = declared if declared is not None else getattr(embedder, "receipt_embedding_identity", None)
        if type(revision) is not str or not revision.strip() or len(revision.encode("utf-8")) > 1024:
            raise ReceiptEmbeddingCompatibilityError("Model-backed and custom receipt embedders require an immutable receipt_embedding_identity")
        model = getattr(embedder, "model_name", None)
        if model is not None and type(model) is not str:
            raise ReceiptEmbeddingCompatibilityError("Embedding model_name must be a string")
        mode = "native"
        if isinstance(embedder, SentenceTransformerEmbedder) and getattr(embedder, "_model", None) is None:
            # A successful model load must never reuse a fallback vector space.
            mode = "random-fallback"
    return {"backend": backend, "revision": revision, "model": model, "mode": mode}


def validate_embedding_contract(payload: dict, identity: dict, vector=None) -> dict | None:
    """Validate every persisted vector before permitting retrieval or an append."""
    records = [record for bucket in ("items", "lineage") for record in payload.get(bucket, [])]
    contract = payload.get("embedding_contract")
    if contract is None:
        if records:
            raise ReceiptEmbeddingCompatibilityError("Existing vectors lack an embedding identity; explicit offline conversion is required")
        if vector is None:
            return None
        contract = {"schema_version": "dml-embedding-contract-v1", "identity": identity, "dimension": int(vector.size)}
    if not isinstance(contract, dict) or set(contract) != {"schema_version", "identity", "dimension"} or contract["schema_version"] != "dml-embedding-contract-v1" or type(contract["dimension"]) is not int or contract["dimension"] <= 0:
        raise ReceiptEmbeddingCompatibilityError("Invalid persisted embedding contract")
    if contract["identity"] != identity:
        raise ReceiptEmbeddingCompatibilityError("Configured embedding identity differs from the committed embedding space")
    for record in records:
        stored = record.get("embedding")
        if not isinstance(stored, list) or len(stored) != contract["dimension"]:
            raise ReceiptEmbeddingCompatibilityError("Persisted embedding dimensions differ from their contract")
    if vector is not None and (vector.ndim != 1 or vector.size != contract["dimension"] or not np.isfinite(vector).all()):
        raise ReceiptEmbeddingCompatibilityError("Prepared embedding dimensions differ from the committed embedding space")
    return contract


def _strict_json(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _strict_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _strict_json(item)
        return
    raise ValueError("Receipt requests require finite JSON values and string object keys")


def canonical_request(text: str, *, tenant_id: str, client_id=None, session_id=None,
                      instance_id=None, kind="memory", meta=None) -> tuple[dict, str]:
    scope = dict(zip(SCOPE_KEYS, (tenant_id, client_id, session_id, instance_id)))
    if type(text) is not str or not text.strip():
        raise ValueError("Receipt ingestion requires nonempty text")
    for name, value in scope.items():
        if value is None and name != "tenant_id":
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise ValueError("Receipt scope members must be nonempty strings of at most 256 UTF-8 bytes")
    if kind is None:
        kind = "memory"
    if type(kind) is not str or not kind.strip() or len(kind.encode("utf-8")) > 256:
        raise ValueError("Receipt kind must be a nonempty string of at most 256 UTF-8 bytes")
    if meta is None:
        meta = {}
    if type(meta) is not dict:
        raise ValueError("Receipt metadata must be an object")
    # Reject conflicting scope rather than silently overriding ambiguous input.
    for name, value in {**scope, "kind": kind, "no_merge": True}.items():
        if name in meta and (type(meta[name]) is not type(value) or meta[name] != value):
            raise ValueError("Receipt metadata conflicts with explicit scope or append-only policy")
    metadata = {**meta, **scope, "kind": kind, "no_merge": True}
    request = {"schema_version": "dml-append-request-v1", "text": text,
               "scope": scope, "kind": kind, "meta": metadata}
    _strict_json(request)
    raw = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Receipt request exceeds 1 MiB")
    return json.loads(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_receipted(journal: JournalStateStore, *, request: dict, request_digest: str,
                    key: str, embed: Callable, capacity: int, embedding_space: Callable[[], dict],
                    hydrate: Callable[[int, dict], None], degraded: Callable[[Exception], None]) -> dict:
    scope = request["scope"]
    existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
    if existing is not None:
        return existing
    identity = embedding_space()
    _, preparation_state = journal.read_snapshot()
    validate_embedding_contract(preparation_state, identity)
    try:
        vector = np.array(embed(request["text"]), dtype=np.float32, copy=True)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Receipt embedding must be a nonempty finite vector")
    except Exception as exc:
        # A competing writer may have committed while this backend failed.
        existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
        if existing is not None:
            return existing
        raise ReceiptEmbeddingError("Receipt embedding unavailable; retry identical request and key") from exc

    if embedding_space() != identity:
        raise ReceiptEmbeddingCompatibilityError("Embedding identity changed while preparing the vector")
    validate_embedding_contract(preparation_state, identity, vector)
    with store_write_lock(journal.path.parent, operation="receipt-ingest", timeout_ms=30000):
        for _ in range(3):
            existing = journal.lookup_receipt(scope=scope, key=key, request_digest=request_digest)
            if existing is not None:
                return existing
            revision, payload = journal.read_snapshot()
            validate_snapshot(payload)
            if embedding_space() != identity:
                raise ReceiptEmbeddingCompatibilityError("Embedding identity changed before commit")
            payload["embedding_contract"] = validate_embedding_contract(payload, identity, vector)
            if len(payload["items"]) >= capacity:
                raise ReceiptCapacityError("Receipt ingestion capacity reached; explicit lifecycle action required")
            ids = [item["id"] for bucket in ("items", "lineage") for item in payload.get(bucket, [])]
            next_id = payload.get("next_id", 0)
            if type(next_id) is not int or next_id < 0:
                raise ValueError("Invalid persisted memory ID allocator")
            ident = max(next_id, max(ids, default=-1) + 1)
            record = MemoryItem(id=ident, text=request["text"], embedding=vector,
                timestamp=time.time(), salience=1.0, fidelity=1.0, level=0, meta=request["meta"]).to_dict()
            payload["items"].append(record)
            payload["next_id"] = ident + 1
            validate_snapshot(payload)
            try:
                receipt = journal.save_with_receipt(payload, scope=scope, key=key,
                    request_digest=request_digest, result={"memory": record},
                    expected_revision=revision, operation="append-receipt-v1")
            except RevisionConflict:
                continue
            except IdempotencyConflict:
                raise
            except Exception as exc:
                # Never compensate an uncertain commit. Resolve against durable
                # receipt state, independently of the in-process projection.
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
                # A valid durable receipt remains success when cache refresh
                # fails. Mark runtime degradation and let a later read recover.
                try:
                    degraded(exc)
                except Exception:
                    # Observability callbacks cannot revoke a durable success.
                    pass
            return receipt
    raise RevisionConflict("Receipt ingest exhausted three revision retries")
