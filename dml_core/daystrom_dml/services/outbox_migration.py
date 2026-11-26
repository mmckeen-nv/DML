"""Explicit, side-by-side schema-2 to schema-4 migration.

Legacy decisions retain their original bytes. Full-state delivery starts with an
honest baseline at the copied revision plus one; absent historical payloads are
never reconstructed. Stop every writer before migrating and switching paths.
The original remains a rollback source only until the first destination write.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import os
import sqlite3
from typing import Callable

from ..atomic_io import _sync_directory, atomic_write_text
from ..journal import (
    MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION, RECEIPT_JOURNAL_SCHEMA_VERSION,
    JournalIntegrityError, JournalSchemaError, JournalStateStore, RevisionConflict,
    _checked, _digest, _encode, _initialization_lock, _read_identity,
)
from ..store_lock import store_write_lock

OUTBOX_MIGRATION_FAULT_POINTS = (
    "migration_before_marker", "migration_after_marker", "migration_after_backup",
    "migration_before_commit", "migration_after_commit", "migration_after_identity",
    "migration_after_publish",
)
BASELINE_OPERATION = "outbox-migration-baseline-v1"


def upgrade_outbox_journal(source: Path, destination: Path, *,
                           fault_hook: Callable[[str], None] | None = None) -> dict:
    """Copy and qualify an offline authority; never overwrite or switch paths.

    A durable marker rejects interrupted destinations. Preserve failed output
    and retry a new location. A failure after publication can leave a complete,
    verified destination; opening that path distinguishes this outcome.
    """
    raw_destination = Path(destination)
    reserved_outputs = (raw_destination, raw_destination.with_name(raw_destination.name + ".identity.json"),
                        raw_destination.with_name(raw_destination.name + ".migration.json"),
                        raw_destination.with_name(raw_destination.name + ".init.lock"),
                        Path(str(raw_destination) + "-wal"), Path(str(raw_destination) + "-shm"))
    if any(os.path.lexists(path) for path in reserved_outputs):
        raise ValueError("outbox upgrade requires unused destination paths without symlinks")
    source, destination = Path(source).resolve(), raw_destination.resolve()
    if not source.is_file() or source.parent == destination.parent:
        raise ValueError("outbox upgrade requires an existing source and a separate destination directory")
    source_identity = source.with_name(source.name + ".identity.json")
    if not source_identity.is_file():
        raise JournalIntegrityError("Migration source requires its existing identity marker")
    # Source-owned files cannot be selected through symlink aliases either.
    if destination.exists():
        raise ValueError("outbox upgrade requires an unused destination")
    marker = destination.with_name(destination.name + ".migration.json")
    identity_path = destination.with_name(destination.name + ".identity.json")
    init_lock = destination.with_name(destination.name + ".init.lock")
    if init_lock.exists():
        raise ValueError("outbox upgrade requires an unused initialization lock")
    fault = fault_hook or (lambda _point: None)
    with store_write_lock(source.parent, operation="journal-outbox-upgrade", timeout_ms=30000):
        original = JournalStateStore(source, receipt_mode=True)
        if original.schema_version != RECEIPT_JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError("outbox upgrade accepts only a schema-2 source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _initialization_lock(init_lock):
            if any(os.path.lexists(path) for path in (destination, marker, identity_path, Path(str(destination) + "-wal"), Path(str(destination) + "-shm"))):
                raise ValueError("outbox upgrade requires an unused destination with no journal sidecars")
            with original._connect() as source_connection:
                source_connection.execute("BEGIN")
                identity_rows = source_connection.execute("SELECT id,store_id FROM identity").fetchall()
                if identity_rows != [(1, original._identity["store_id"])]:
                    raise JournalIntegrityError("Migration source identity changed")
                revision, payload, _ = original._read_snapshot(source_connection)
                decision_rows = source_connection.execute("SELECT revision,payload,checksum FROM decisions ORDER BY revision").fetchall()
                receipt_rows = source_connection.execute("SELECT scope,key,revision,payload,checksum FROM receipts ORDER BY scope,key").fetchall()
                origin = {
                    "source_schema_version": RECEIPT_JOURNAL_SCHEMA_VERSION,
                    "source_revision": revision, "source_digest": _digest(_encode(payload)),
                    "history_digest": _digest(_encode([_checked(raw, checksum) for _, raw, checksum in decision_rows])),
                }
                report = {"schema_version": MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION,
                          "source_store_id": original._identity["store_id"],
                          "source_revision": revision, "destination_revision": revision + 1,
                          "boundary_revision": revision + 1,
                          "preserved_decision_count": len(decision_rows),
                          "preserved_receipt_count": len(receipt_rows), "origin": origin}
                fault("migration_before_marker")
                atomic_write_text(marker, _encode(report))
                fault("migration_after_marker")
                destination.touch(exist_ok=False)
                with closing(sqlite3.connect(destination, timeout=30)) as target:
                    source_connection.backup(target)
                    fault("migration_after_backup")
                    target.execute("PRAGMA journal_mode=WAL")
                    target.execute("PRAGMA synchronous=FULL")
                    with target:
                        target.execute("BEGIN IMMEDIATE")
                        target.execute("CREATE TABLE outbox (revision INTEGER PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)")
                        target.execute("CREATE TABLE migration_origin (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, checksum TEXT NOT NULL)")
                        raw_origin = _encode(origin)
                        target.execute("INSERT INTO migration_origin VALUES (1,?,?)", (raw_origin, _digest(raw_origin)))
                        decision = {"schema_version": MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION,
                                    "revision": revision + 1, "operation": BASELINE_OPERATION,
                                    "changed": [], "deleted": [],
                                    "record_count": len(payload["items"]) + len(payload["lineage"]),
                                    "state_digest": origin["source_digest"], "receipt": None}
                        event = {"schema_version": 2, "event_format": "dml-journal-outbox-v2",
                                 "source_store_id": original._identity["store_id"],
                                 "source_revision": revision + 1, "source_digest": origin["source_digest"],
                                 "operation": BASELINE_OPERATION, "receipt": None, "state": payload,
                                 "decision_digest": _digest(_encode(decision)), "origin": origin}
                        event["checksum"] = _digest(_encode(event))
                        decision["outbox_digest"] = event["checksum"]
                        raw_decision, raw_event = _encode(decision), _encode(event)
                        target.execute("INSERT INTO decisions VALUES (?,?,?)", (revision + 1, raw_decision, _digest(raw_decision)))
                        target.execute("INSERT INTO outbox VALUES (?,?,?)", (revision + 1, raw_event, _digest(raw_event)))
                        target.execute("UPDATE state SET revision=? WHERE id=1", (revision + 1,))
                        raw_state = _encode(payload)
                        target.execute("INSERT OR REPLACE INTO snapshot VALUES (1,?,?,?)", (revision + 1, raw_state, _digest(raw_state)))
                        target.execute(f"PRAGMA user_version={MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION}")
                        fault("migration_before_commit")
                    fault("migration_after_commit")
                    verifier = JournalStateStore.__new__(JournalStateStore)
                    verifier._schema_version = MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION
                    verifier._identity = dict(original._identity)
                    with target:
                        target.execute("BEGIN")
                        if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                            raise JournalIntegrityError("Upgraded outbox journal integrity check failed")
                        verifier._validate_layout(target)
                        new_revision, new_payload, _ = verifier._read_snapshot(target)
                        preserved_decisions = target.execute("SELECT revision,payload,checksum FROM decisions WHERE revision<=? ORDER BY revision", (revision,)).fetchall()
                        preserved_receipts = target.execute("SELECT scope,key,revision,payload,checksum FROM receipts ORDER BY scope,key").fetchall()
                        if new_revision != revision + 1 or _encode(new_payload) != _encode(payload) or preserved_decisions != decision_rows or preserved_receipts != receipt_rows:
                            raise JournalIntegrityError("Upgraded journal differs from pinned source")
                    atomic_write_text(identity_path, _encode(original._identity))
                    fault("migration_after_identity")
                # A direct writer bypassing the advisory lock must not silently
                # turn an offline migration into a published stale snapshot.
                current_id, current_revision, current_payload = original.verified_snapshot()
                if current_id != original._identity["store_id"] or current_revision != revision or _encode(current_payload) != _encode(payload) or _read_identity(source_identity) != original._identity:
                    raise RevisionConflict("Migration source advanced; stop writers and retry a new destination")
                marker.unlink()
                _sync_directory(destination.parent)
                fault("migration_after_publish")
                return report
