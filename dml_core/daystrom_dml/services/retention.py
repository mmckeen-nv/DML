"""Scoped inventory of known retained memory occurrences, without erasure.

Only explicit record containers and the versioned first-level promotion proof
are interpreted. This is an inspection of one verified journal transaction, not
a proof that every semantic copy, backing file, or external store was found.
"""
from __future__ import annotations

import json

from ..journal import JournalStateStore
from ..persistence import validate_record
from .receipt_ingestion import SCOPE_KEYS
from .receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, _canonical_json, _valid_digest,
    memory_digest,
)
from .receipt_promotion import _base_source, _shared_metadata

UNINSPECTED_SURFACES = (
    "projections", "outbox_consumers", "external_checkpoints",
    "backups_and_migration_sources", "sqlite_wal_and_free_pages",
    "filesystem_snapshots", "runtime_and_caller_copies",
    "arbitrary_metadata_and_semantic_copies",
)
_MAX_PROOF_BYTES = 1024 * 1024


class RetentionInspectionUnsupported(ValueError):
    """A scoped stored shape cannot be completely inspected by this contract."""


def canonical_retention_request(memory_id: int, *, tenant_id: str, client_id=None,
                                session_id=None, instance_id=None) -> dict:
    """Freeze a strict record identity and complete receipt scope."""
    if type(memory_id) is not int or memory_id < 0:
        raise ValueError("Retention memory_id must be a nonnegative integer")
    scope = dict(zip(SCOPE_KEYS, (tenant_id, client_id, session_id, instance_id)))
    for name, value in scope.items():
        if value is None and name != "tenant_id":
            continue
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 256:
            raise ValueError("Retention scope members must be nonempty strings of at most 256 UTF-8 bytes")
    request = {"schema_version": "dml-retention-request-v1", "memory_id": memory_id,
               "scope": scope}
    return json.loads(_canonical_json(request))


def _request(request: dict) -> dict:
    if type(request) is not dict:
        raise ValueError("Retention request must be an object")
    raw = _canonical_json(request)
    frozen = json.loads(raw)
    if set(frozen) != {"schema_version", "memory_id", "scope"}:
        raise ValueError("Invalid retention request fields")
    if type(frozen["scope"]) is not dict or set(frozen["scope"]) != set(SCOPE_KEYS):
        raise ValueError("Retention requires the complete receipt scope")
    canonical = canonical_retention_request(frozen["memory_id"], **frozen["scope"])
    if raw != _canonical_json(canonical):
        raise ValueError("Unsupported retention request version or values")
    return canonical


def _scoped(metadata, scope: dict) -> bool:
    return type(metadata) is dict and all(
        type(metadata.get(name)) is type(value) and metadata.get(name) == value
        for name, value in scope.items()
    )


def _promotion_sources(record: dict, scope: dict) -> list[dict]:
    """Validate bounded, immutable source evidence without rechecking its age.

    The container may since have been updated or retired. Source proof remains
    historical and must retain the first-level shape accepted by promotion.
    """
    proof = record["meta"]["promotion_decision"]
    if (type(proof) is not dict
            or set(proof) != {"schema_version", "reason", "sources"}
            or proof["schema_version"] != "dml-promotion-decision-v1"
            or type(proof["reason"]) is not str or not proof["reason"].strip()
            or len(proof["reason"].encode("utf-8")) > 1024
            or type(proof["sources"]) is not list
            or not 1 <= len(proof["sources"]) <= 32):
        raise ValueError("Unsupported promotion proof")
    if len(_canonical_json(proof).encode("utf-8")) > _MAX_PROOF_BYTES:
        raise ValueError("Promotion proof exceeds inspection budget")
    sources = []
    for binding in proof["sources"]:
        if (type(binding) is not dict or set(binding) != {"memory_digest", "memory"}
                or not _valid_digest(binding["memory_digest"])
                or type(binding["memory"]) is not dict
                or not _scoped(binding["memory"].get("meta"), scope)):
            raise ValueError("Unsupported promotion source binding")
        source = binding["memory"]
        validate_record(source)
        # _base_source rejects recursive proof, indirect lineage, and terminal or
        # untrusted sources without comparing historical expiry with wall time.
        _base_source(source, merging=len(proof["sources"]) > 1, now=None)
        if memory_digest(source) != binding["memory_digest"]:
            raise ValueError("Promotion source digest mismatch")
        sources.append(source)
    ids = [source["id"] for source in sources]
    dimensions = {len(source["embedding"]) for source in sources}
    children = record.get("children")
    if (ids != sorted(set(ids)) or record["id"] <= max(ids) or record["level"] != 1
            or record.get("summary_of") != ids or record.get("children") != ids
            or type(children) is not list or any(type(ident) is not int for ident in children)
            or 0 in dimensions or len(dimensions) != 1
            or len(record["embedding"]) not in dimensions):
        raise ValueError("Unsupported first-level promotion shape")
    _shared_metadata(sources)
    return sources


def _count_record(record: dict, request: dict, counts: dict) -> None:
    scope = request["scope"]
    # No foreign proof is interpreted, even when it is malformed or includes a
    # same-numbered record. Scope is an exact four-field identity, including null.
    if type(record) is not dict or not _scoped(record.get("meta"), scope):
        return
    try:
        validate_record(record)
        if record["id"] == request["memory_id"]:
            counts["direct_records"] += 1
        if "promotion_decision" in record["meta"]:
            for source in _promotion_sources(record, scope):
                if source["id"] == request["memory_id"]:
                    counts["embedded_source_records"] += 1
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError,
            ReceiptLifecycleConflict) as exc:
        raise RetentionInspectionUnsupported("Scoped memory has an unsupported retained shape") from exc


def _count_state(state: dict, request: dict, counts: dict) -> None:
    for bucket in ("items", "lineage"):
        for record in state[bucket]:
            _count_record(record, request, counts)


def inspect_memory_retention(journal: JournalStateStore, *, request: dict) -> dict:
    """Count literal serialized occurrences in one pinned, verified authority.

    Counts are neither distinct memories nor physical copies or byte estimates.
    The journal's historical snapshot, every receipt and every outbox state are
    inspected even when the target is no longer present in current live state.
    """
    request = _request(request)
    view = journal.read_retention_view()
    surfaces = {name: {"direct_records": 0, "embedded_source_records": 0} for name in (
        "current_items", "current_lineage", "journal_snapshot", "receipts", "outbox_states",
    )}
    for bucket, name in (("items", "current_items"), ("lineage", "current_lineage")):
        for record in view["state"][bucket]:
            _count_record(record, request, surfaces[name])
    if view["snapshot"] is not None:
        _count_state(view["snapshot"], request, surfaces["journal_snapshot"])
    for receipt in view["receipts"]:
        if _scoped(receipt["scope"], request["scope"]):
            _count_record(receipt["result"]["memory"], request, surfaces["receipts"])
    # Events contain full states; their receipt fields contain only bindings, so
    # following those bindings would incorrectly recount receipt payloads.
    for event in view["outbox"]:
        _count_state(event["state"], request, surfaces["outbox_states"])
    count = sum(value for surface in surfaces.values() for value in surface.values())
    if count == 0:
        raise ReceiptMemoryNotFound("Memory is not available in the requested scope")
    return {"schema_version": "dml-retention-report-v1", "request": request,
            "source": {"store_id": view["store_id"], "journal_schema_version": view["schema_version"],
                       "revision": view["revision"], "state_digest": view["state_digest"]},
            "coverage": "known_structured_references_in_one_journal", "surfaces": surfaces,
            "known_reference_count": count, "physical_erasure_supported": False,
            "erasure_proven": False, "retirement_is_erasure": False,
            "uninspected_surfaces": list(UNINSPECTED_SURFACES)}
