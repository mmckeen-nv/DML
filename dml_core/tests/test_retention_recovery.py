"""Independent retained-copy oracles and interruption tests for inspection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pytest

from daystrom_dml.checkpoint import CheckpointManager
from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer, deliver_outbox
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.projection import SQLiteProjection, reconcile
from daystrom_dml.services.receipt_ingestion import append_receipted, canonical_request
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptMemoryNotFound, canonical_retirement_request, retire_receipted,
)
from daystrom_dml.services.receipt_promotion import canonical_promotion_request, promote_receipted
from daystrom_dml.services.receipt_update import canonical_update_request, update_receipted
from daystrom_dml.services.retention import canonical_retention_request, inspect_memory_retention


IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
SCOPE = {"tenant_id": "tenant", "client_id": "client", "session_id": None, "instance_id": None}
SENTINEL = "PRIVATE_RETAINED_CONTENT_63ab7c"
FOREIGN_SENTINEL = "PRIVATE_FOREIGN_CONTENT_d284e1"
SURFACES = ("current_items", "current_lineage", "journal_snapshot", "receipts", "outbox_states")
READ_HOOKS = ("retention_after_snapshot", "retention_after_history")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def append(source, key, *, text=SENTINEL, scope=None, meta=None):
    request, request_digest = canonical_request(text, **(scope or SCOPE),
        meta=meta or {"source_trust": "trusted"})
    return append_receipted(source, request=request, request_digest=request_digest, key=key,
        embed=lambda _: np.array([1., 0.]), embedding_space=lambda: IDENTITY,
        capacity=1000, hydrate=lambda *_: None, degraded=lambda _: None)


def prepare(tmp_path, schema, *, removed=True):
    source = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
        receipt_mode=True, outbox_mode=schema == 3, snapshot_interval=1)
    first = append(source, "private-original-key")
    original = first["result"]["memory"]
    # A foreign container's unsupported proof must not disclose its existence
    # through inspection in this scope, including through historical events.
    foreign = append(source, "private-foreign-key", text=FOREIGN_SENTINEL,
        scope={**SCOPE, "tenant_id": "foreign"},
        meta={"source_trust": "trusted", "promotion_decision": {"schema_version": "private-future-proof"}})
    request, request_digest = canonical_update_request(original["id"],
        text=SENTINEL + " corrected", expected_memory_digest=digest(original),
        reason="private update reason", **SCOPE)
    updated = update_receipted(source, request=request, request_digest=request_digest,
        key="private-update-key", embed=lambda _: np.array([0., 1.]),
        embedding_space=lambda: IDENTITY, hydrate=lambda *_: None, degraded=lambda _: None)
    request, request_digest = canonical_promotion_request([
        {"memory_id": original["id"], "expected_memory_digest": digest(updated["result"]["memory"])}],
        text="An independent derived interpretation", reason="private promotion reason", **SCOPE)
    promoted = promote_receipted(source, request=request, request_digest=request_digest,
        key="private-promotion-key", embed=lambda _: np.array([0., 1.]),
        embedding_space=lambda: IDENTITY, capacity=1000,
        hydrate=lambda *_: None, degraded=lambda _: None)
    request, request_digest = canonical_retirement_request(original["id"],
        expected_memory_digest=digest(updated["result"]["memory"]),
        reason="private retirement reason", **SCOPE)
    retired = retire_receipted(source, request=request, request_digest=request_digest,
        key="private-retirement-key", hydrate=lambda *_: None, degraded=lambda _: None)
    revision, state = source.read_snapshot()
    state["lineage"] = [copy.deepcopy(original)]
    source.save(state, expected_revision=revision, operation="test-retained-lineage")
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(source.path, destination)
        source = JournalStateStore(destination, snapshot_interval=1)
    if removed:
        remove_current(source, original["id"])
    return source, original["id"], [first, foreign, updated, promoted, retired]


def remove_current(source, memory_id):
    # Preserve the previous full snapshot, which is a distinct retained copy.
    source.snapshot_interval = 1000
    revision, state = source.read_snapshot()
    state["items"] = [record for record in state["items"] if record["id"] != memory_id]
    source.save(state, expected_revision=revision, operation="test-remove-live-only")


def logical_rows(path):
    """Independent logical database oracle; SQLite page layouts may change."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: sorted(connection.execute('SELECT * FROM "' + name + '"').fetchall(), key=repr)
                for name in tables}


def expected_observation(source, memory_id):
    """Count literal serialized occurrences from SQL, without service helpers."""
    with closing(sqlite3.connect(source.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        store_id = connection.execute("SELECT store_id FROM identity WHERE id=1").fetchone()[0]
        revision, metadata = connection.execute("SELECT revision,metadata FROM state WHERE id=1").fetchone()
        state = {**json.loads(metadata), "items": [], "lineage": []}
        for bucket, raw in connection.execute("SELECT bucket,payload FROM records ORDER BY bucket,position"):
            state[bucket].append(json.loads(raw))
        snapshots = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM snapshot")]
        receipts = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM receipts")]
        events = ([json.loads(row[0]) for row in connection.execute("SELECT payload FROM outbox")]
                  if source.schema_version in (3, 4) else [])

    groups = {
        "current_items": state["items"], "current_lineage": state["lineage"],
        "journal_snapshot": [record for snapshot in snapshots for bucket in ("items", "lineage")
                             for record in snapshot[bucket]],
        "receipts": [receipt["result"]["memory"] for receipt in receipts if receipt["scope"] == SCOPE],
        "outbox_states": [record for event in events for bucket in ("items", "lineage")
                          for record in event["state"][bucket]],
    }
    counts = {}
    for surface, records in groups.items():
        direct = embedded = 0
        for record in records:
            meta = record["meta"]
            if any(type(meta.get(name)) is not type(value) or meta.get(name) != value
                   for name, value in SCOPE.items()):
                continue
            direct += record["id"] == memory_id
            proof = meta.get("promotion_decision")
            if proof is not None:
                # All in-scope proofs in this independently constructed fixture
                # were acknowledged by the real first-level promotion API.
                assert proof["schema_version"] == "dml-promotion-decision-v1"
                for entry in proof["sources"]:
                    original = entry["memory"]
                    if original["id"] == memory_id and all(
                        type(original["meta"].get(name)) is type(value)
                        and original["meta"].get(name) == value for name, value in SCOPE.items()):
                        embedded += 1
        counts[surface] = {"direct_records": direct, "embedded_source_records": embedded}
    return {"store_id": store_id, "journal_schema_version": source.schema_version,
            "revision": revision, "state_digest": digest(state)}, counts


def inspect(source, memory_id):
    return inspect_memory_retention(source, request=canonical_retention_request(memory_id, **SCOPE))


def assert_report(report, source_info, counts):
    assert set(report) == {"schema_version", "source", "request", "coverage", "surfaces",
        "known_reference_count", "physical_erasure_supported", "erasure_proven",
        "retirement_is_erasure", "uninspected_surfaces"}
    assert report["schema_version"] == "dml-retention-report-v1"
    assert report["request"] == {"schema_version": "dml-retention-request-v1", "memory_id": 0, "scope": SCOPE}
    assert report["source"] == source_info
    assert report["coverage"] == "known_structured_references_in_one_journal"
    assert report["surfaces"] == counts
    assert all(type(value) is int and value >= 0 for surface in report["surfaces"].values()
               for value in surface.values())
    assert report["known_reference_count"] == sum(sum(value.values()) for value in counts.values())
    for flag in ("physical_erasure_supported", "erasure_proven", "retirement_is_erasure"):
        assert report[flag] is False
    assert report["uninspected_surfaces"] == ["projections", "outbox_consumers", "external_checkpoints",
        "backups_and_migration_sources", "sqlite_wal_and_free_pages", "filesystem_snapshots",
        "runtime_and_caller_copies", "arbitrary_metadata_and_semantic_copies"]
    raw = encoded(report)
    for private in (SENTINEL, FOREIGN_SENTINEL, "private-original-key", "private-foreign-key",
                    "private-update-key", "private-promotion-key", "private-retirement-key",
                    "private update reason", "private promotion reason", "private retirement reason",
                    "An independent derived interpretation", "private-future-proof"):
        assert private not in raw


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_history_and_proof_inventory_matches_independent_sql_oracle(tmp_path, schema):
    source, memory_id, receipts = prepare(tmp_path, schema)
    before = logical_rows(source.path)
    source_info, counts = expected_observation(source, memory_id)
    assert counts["current_items"] == {"direct_records": 0, "embedded_source_records": 1}
    assert counts["current_lineage"] == {"direct_records": 1, "embedded_source_records": 0}
    assert counts["journal_snapshot"] == {"direct_records": 2, "embedded_source_records": 1}
    assert counts["receipts"] == {"direct_records": 3, "embedded_source_records": 1}
    assert counts["outbox_states"]["direct_records"] > 0 if schema in (3, 4) else counts["outbox_states"]["direct_records"] == 0
    report = inspect(source, memory_id)
    assert_report(report, source_info, counts)
    assert logical_rows(source.path) == before
    reopened = JournalStateStore(source.path)
    assert inspect(reopened, memory_id) == report
    for receipt in receipts:
        assert reopened.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_history_only_target_is_not_mistaken_for_absent_memory(tmp_path, schema):
    source, memory_id, _ = prepare(tmp_path, schema)
    revision, state = source.read_snapshot()
    state["items"] = [record for record in state["items"] if record["meta"]["tenant_id"] == "foreign"]
    state["lineage"] = []
    source.snapshot_interval = 1
    source.save(state, expected_revision=revision, operation="test-remove-all-current-references")
    info, counts = expected_observation(source, memory_id)
    for surface in ("current_items", "current_lineage", "journal_snapshot"):
        assert counts[surface] == {"direct_records": 0, "embedded_source_records": 0}
    assert counts["receipts"]["direct_records"] == 3
    assert_report(inspect(source, memory_id), info, counts)
    with pytest.raises(ReceiptMemoryNotFound):
        inspect(source, memory_id + 1000)


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("point", READ_HOOKS)
def test_concurrent_commit_between_inspection_reads_cannot_mix_revisions(tmp_path, schema, point):
    source, memory_id, _ = prepare(tmp_path, schema)
    writer = JournalStateStore(source.path)
    expected = expected_observation(source, memory_id)
    entered, release = threading.Event(), threading.Event()

    def pause(here):
        if here == point:
            entered.set()
            assert release.wait(15), "inspection hook was not released"

    source._fault_hook = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(inspect, source, memory_id)
        try:
            assert entered.wait(10), "inspection hook was not reached"
            mutation = pool.submit(append, writer, "concurrent-commit", text="Unrelated acknowledged change")
            # An inspection may pin a read transaction, but it must not hold
            # store writer ownership while an operator receives the report.
            receipt = mutation.result(timeout=10)
            assert receipt["revision"] == expected[0]["revision"] + 1
        finally:
            release.set()
        report = reader.result(timeout=10)
    source._fault_hook = lambda _: None
    assert_report(report, *expected)
    assert_report(inspect(source, memory_id), *expected_observation(source, memory_id))
    assert source.read_snapshot()[0] == expected[0]["revision"] + 1


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("point", READ_HOOKS)
def test_process_death_during_inspection_preserves_every_acknowledged_row(tmp_path, schema, point):
    source, memory_id, receipts = prepare(tmp_path, schema)
    before, report = logical_rows(source.path), inspect(source, memory_id)
    code = '''
import json, os, signal, sys
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.retention import inspect_memory_retention
path, point, request = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
source = JournalStateStore(Path(path), fault_hook=crash)
inspect_memory_retention(source, request=json.loads(request))
raise AssertionError('inspection fault hook was not reached')
'''
    child = subprocess.run([sys.executable, "-c", code, str(source.path), point,
        encoded(canonical_retention_request(memory_id, **SCOPE))], capture_output=True, text=True, timeout=45)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    assert logical_rows(source.path) == before
    reopened = JournalStateStore(source.path)
    assert inspect(reopened, memory_id) == report
    for receipt in receipts:
        assert reopened.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt


COMMON_DAMAGE = ("records", "state", "snapshot", "decisions", "receipts", "missing-receipt", "missing-decision",
                 "identity", "identity-marker", "migration-marker", "schema-zero")
DAMAGE_CASES = [(schema, damage) for schema in (2, 3, 4)
                for damage in (*COMMON_DAMAGE, *(("outbox", "missing-outbox") if schema in (3, 4) else ()))]


@pytest.mark.parametrize("schema,damage", DAMAGE_CASES,
    ids=[f"schema-{schema}-{damage}" for schema, damage in DAMAGE_CASES])
def test_invalid_authority_never_produces_a_partial_retention_report(tmp_path, schema, damage):
    source, memory_id, _ = prepare(tmp_path, schema)
    if damage == "identity-marker":
        source.identity_path.write_text(encoded({"schema_version": 1, "store_id": "f" * 32}), encoding="utf-8")
    elif damage == "migration-marker":
        source.migration_path.write_text('{"incomplete":true}', encoding="utf-8")
    else:
        with closing(sqlite3.connect(source.path)) as connection, connection:
            if damage == "schema-zero":
                connection.execute("PRAGMA user_version=0")
            elif damage == "identity":
                connection.execute("UPDATE identity SET store_id=?", ("f" * 32,))
            elif damage.startswith("missing-"):
                table = {"missing-receipt": "receipts", "missing-decision": "decisions", "missing-outbox": "outbox"}[damage]
                connection.execute("DELETE FROM " + table + " WHERE rowid=(SELECT MIN(rowid) FROM " + table + ")")
            else:
                connection.execute("UPDATE " + damage + " SET checksum=?", ("0" * 64,))
    before = logical_rows(source.path)
    identity = source.identity_path.read_bytes()
    with pytest.raises(JournalIntegrityError) as failure:
        inspect(source, memory_id)
    assert SENTINEL not in str(failure.value) and FOREIGN_SENTINEL not in str(failure.value)
    assert logical_rows(source.path) == before
    assert source.identity_path.read_bytes() == identity
    if damage == "migration-marker":
        assert source.migration_path.read_text(encoding="utf-8") == '{"incomplete":true}'


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_external_retained_copies_cannot_be_mistaken_for_verified_erasure(tmp_path, schema):
    source, memory_id, _ = prepare(tmp_path, schema, removed=False)
    # These are known test-owned copies, intentionally outside inspection's
    # declared journal boundary. No operator filesystem is searched or changed.
    backup_path = tmp_path / "backup" / "journal.sqlite3"
    backup_path.parent.mkdir()
    with closing(sqlite3.connect(source.path)) as origin, closing(sqlite3.connect(backup_path)) as backup:
        origin.backup(backup)
    backup_path.with_name(backup_path.name + ".identity.json").write_bytes(source.identity_path.read_bytes())
    projection = SQLiteProjection(tmp_path / "projection" / "index.sqlite3")
    reconcile(source, projection)
    manager = CheckpointManager(tmp_path / "checkpoint", lambda: source.read_snapshot()[1], start=False)
    try:
        checkpoint = manager.checkpoint()
    finally:
        assert manager.close()
    consumer = None
    if schema in (3, 4):
        consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "events.sqlite3")
        deliver_outbox(source, consumer)

    remove_current(source, memory_id)
    report = inspect(source, memory_id)
    assert_report(report, *expected_observation(source, memory_id))
    assert SENTINEL in encoded(JournalStateStore(backup_path).read_snapshot()[1])
    assert SENTINEL in encoded(projection.read())
    assert SENTINEL in checkpoint.read_text(encoding="utf-8")
    if consumer is not None:
        assert SENTINEL in encoded(consumer.read())
    assert report["physical_erasure_supported"] is report["erasure_proven"] is False


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_simultaneous_retention_readers_leave_authority_unchanged(tmp_path, schema):
    source, memory_id, _ = prepare(tmp_path, schema)
    before, expected = logical_rows(source.path), inspect(source, memory_id)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "24"))
    barrier = threading.Barrier(clients)

    def observe(_):
        barrier.wait(timeout=30)
        return inspect(source, memory_id)

    with ThreadPoolExecutor(max_workers=clients) as pool:
        reports = list(pool.map(observe, range(clients)))
    assert reports == [expected] * clients
    assert logical_rows(source.path) == before
