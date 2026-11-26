"""Verified, coalesced projection deltas over a trusted backend boundary.

Only changed record payloads are transported. Preparation and SQLite application
still verify full snapshots and journal history; this is not an event outbox.
Checksums detect damage, not a malicious party able to rewrite a packet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..journal import JournalStateStore, RevisionConflict
from .projection import (
    FORMAT, ProjectionError, ProjectionStale, SQLiteProjection, _cursor, _digest,
    _encode, _prepared, _state_contract, _validate, prepare_source,
)

DELTA_FORMAT = "dml-projection-delta-v1"
FAULT_POINTS = ("delta_before_apply", "delta_after_apply")


class ProjectionBackend(Protocol):
    path: Path

    def read(self) -> dict: ...

    def apply_delta(self, packet: dict) -> dict: ...


def _frozen(value):
    return json.loads(_encode(value))


def _envelope(value) -> dict:
    value = _frozen(value)
    if type(value) is not dict:
        raise ProjectionError("Projection envelope must be an object")
    return _validate(value)


def _ids(values) -> list[int]:
    if type(values) is not list or any(type(i) is not int or i < 0 for i in values):
        raise ProjectionError("Delta IDs must be nonnegative integers")
    if len(values) != len(set(values)):
        raise ProjectionError("Duplicate delta IDs")
    return values


def _hash(value) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ProjectionError("Invalid delta digest")


def _next(base, desired) -> None:
    _cursor(desired)
    if base is None:
        return
    _cursor(base)
    if base["source_store_id"] != desired["source_store_id"]:
        raise ProjectionError("Delta cannot change projection authority")
    if base["source_revision"] > desired["source_revision"]:
        raise ProjectionStale("Delta cannot roll back source revision")
    if base["source_revision"] == desired["source_revision"] and base != desired:
        raise ProjectionError("Different contents claim the same source revision")


def prepare_delta(source: JournalStateStore, backend: ProjectionBackend) -> dict:
    desired, source_path = _prepared(prepare_source(source))
    if source_path.parent == Path(backend.path).resolve().parent:
        raise ProjectionError("Projection and source directories must differ")
    base = _envelope(backend.read())
    _next(base["cursor"], desired["cursor"])
    if base["cursor"] == desired["cursor"] and _encode(base) != _encode(desired):
        raise ProjectionError("Different contents claim the same source revision")
    old = {record["id"]: record for record in base["items"]}
    new = {record["id"]: record for record in desired["items"]}
    packet = {
        "schema_version": 1, "delta_format": DELTA_FORMAT,
        "base_cursor": base["cursor"], "base_envelope_digest": _digest(base),
        "next_cursor": desired["cursor"], "target_envelope_digest": _digest(desired),
        "source_path": str(source_path),
        "upserts": [record for record in desired["items"] if _encode(old.get(record["id"])) != _encode(record)],
        "deletes": sorted(set(old) - set(new)), "ordered_ids": [record["id"] for record in desired["items"]],
        "embedding_contract": desired["embedding_contract"],
    }
    return _frozen({**packet, "checksum": _digest(packet)})


def _packet(value: dict) -> dict:
    packet = _frozen(value)  # freeze all nested inputs before callbacks or I/O
    keys = {"schema_version", "delta_format", "base_cursor", "base_envelope_digest", "next_cursor",
            "target_envelope_digest", "source_path", "upserts", "deletes", "ordered_ids",
            "embedding_contract", "checksum"}
    if type(packet) is not dict or set(packet) != keys:
        raise ProjectionError("Invalid delta packet fields")
    if type(packet["schema_version"]) is not int or packet["schema_version"] != 1 or packet["delta_format"] != DELTA_FORMAT:
        raise ProjectionError("Unknown delta format")
    for key in ("checksum", "base_envelope_digest", "target_envelope_digest"):
        _hash(packet[key])
    if _digest({key: value for key, value in packet.items() if key != "checksum"}) != packet["checksum"]:
        raise ProjectionError("Delta checksum mismatch")
    for key in ("base_cursor", "next_cursor"):
        cursor = packet[key]
        if cursor is None and key == "base_cursor":
            continue
        if type(cursor) is not dict or set(cursor) != {"source_store_id", "source_revision", "source_digest"}:
            raise ProjectionError("Invalid delta cursor")
    _next(packet["base_cursor"], packet["next_cursor"])
    if packet["base_cursor"] == packet["next_cursor"] and packet["base_envelope_digest"] != packet["target_envelope_digest"]:
        raise ProjectionError("Different envelopes claim the same source revision")
    if type(packet["source_path"]) is not str or not Path(packet["source_path"]).is_absolute():
        raise ProjectionError("Invalid delta source path")
    ordered = set(_ids(packet["ordered_ids"]))
    deletes = set(_ids(packet["deletes"]))
    records = packet["upserts"]
    if type(records) is not list or any(type(record) is not dict or "id" not in record for record in records):
        raise ProjectionError("Invalid delta upserts")
    upserts = set(_ids([record["id"] for record in records]))
    if deletes & (upserts | ordered) or not upserts <= ordered:
        raise ProjectionError("Inconsistent delta record manifests")
    _state_contract({"schema_version": 1, "items": records, "lineage": [],
                     "embedding_contract": packet["embedding_contract"]})
    if packet["base_cursor"] is None:
        empty = {"schema_version": 1, "projection_format": FORMAT, "cursor": None,
                 "embedding_contract": None, "items": [], "lineage": []}
        if packet["base_envelope_digest"] != _digest(empty) or deletes or upserts != ordered:
            raise ProjectionError("Invalid initial delta")
    return packet


def apply_sqlite_delta(backend: SQLiteProjection, value: dict) -> dict:
    packet = _packet(value)
    if Path(packet["source_path"]).resolve().parent == backend.path.parent:
        raise ProjectionError("Projection and source directories must differ")
    backend.fault_hook("delta_before_apply")
    for _ in range(8):
        revision, current = backend.journal.read_snapshot()
        current = _envelope(current)
        # A replay is acknowledged only after proving the complete target state.
        if current["cursor"] == packet["next_cursor"]:
            if _digest(current) != packet["target_envelope_digest"]:
                raise ProjectionError("Different envelopes claim the same source revision")
            records = {record["id"]: record for record in current["items"]}
            if (packet["ordered_ids"] != [record["id"] for record in current["items"]]
                    or _encode(packet["embedding_contract"]) != _encode(current["embedding_contract"])
                    or any(_encode(record) != _encode(records.get(record["id"])) for record in packet["upserts"])):
                raise ProjectionError("Replay delta does not describe the current target")
            return dict(current["cursor"])
        _next(current["cursor"], packet["next_cursor"])
        if current["cursor"] != packet["base_cursor"] or _digest(current) != packet["base_envelope_digest"]:
            raise ProjectionStale("Delta base no longer matches the projection")
        records = {record["id"]: record for record in current["items"]}
        if not set(packet["deletes"]) <= records.keys():
            raise ProjectionError("Delta deletes an unknown record")
        for memory_id in packet["deletes"]:
            del records[memory_id]
        records.update({record["id"]: record for record in packet["upserts"]})
        if set(records) != set(packet["ordered_ids"]):
            raise ProjectionError("Delta final ID manifest is incomplete")
        proposed = _envelope({"schema_version": 1, "projection_format": FORMAT,
                              "cursor": packet["next_cursor"], "embedding_contract": packet["embedding_contract"],
                              "items": [records[i] for i in packet["ordered_ids"]], "lineage": []})
        if _digest(proposed) != packet["target_envelope_digest"]:
            raise ProjectionError("Reconstructed delta target digest mismatch")
        try:
            backend.journal.save(proposed, expected_revision=revision, operation="projection-delta-v1")
            backend.fault_hook("delta_after_apply")
            return dict(proposed["cursor"])
        except RevisionConflict:
            continue
        except Exception:
            # Publication may have committed before its acknowledgement failed.
            if _encode(_envelope(backend.read())) == _encode(proposed):
                return dict(proposed["cursor"])
            raise
    raise RevisionConflict("Delta publication remained contended")


def reconcile_incremental(source: JournalStateStore, backend: ProjectionBackend) -> dict:
    packet = prepare_delta(source, backend)
    desired_cursor = _frozen(packet["next_cursor"])
    desired_digest = packet["target_envelope_digest"]
    upsert_count, delete_count = len(packet["upserts"]), len(packet["deletes"])
    published = _frozen(backend.apply_delta(packet))
    if type(published) is not dict or _encode(published) != _encode(desired_cursor):
        raise ProjectionError("Backend acknowledged a different delta cursor")
    expected, _ = _prepared(prepare_source(source))
    actual = _envelope(backend.read())
    cursor = actual["cursor"]
    if (cursor is None or cursor["source_store_id"] != desired_cursor["source_store_id"]
            or cursor["source_revision"] < desired_cursor["source_revision"]):
        raise ProjectionError("Backend did not publish the acknowledged delta")
    if cursor["source_revision"] == desired_cursor["source_revision"] and _digest(actual) != desired_digest:
        raise ProjectionError("Backend contents differ from the acknowledged delta")
    return {"matches_pinned_source": _encode(actual) == _encode(expected), "source": expected["cursor"],
            "projection": actual["cursor"], "record_count": len(actual["items"]),
            "published": published, "upsert_count": upsert_count,
            "delete_count": delete_count}
