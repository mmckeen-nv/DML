"""Migration boundary, preservation, corruption and failure-state oracles."""
from __future__ import annotations

from contextlib import closing
import errno
import json
from pathlib import Path
import sqlite3

import pytest

from daystrom_dml.journal import (
    JournalIntegrityError, JournalSchemaError, JournalStateStore, RevisionConflict,
    _digest, _encode,
)
from daystrom_dml.services.journal_outbox import validate_outbox_event
from daystrom_dml.services.outbox_migration import (
    OUTBOX_MIGRATION_FAULT_POINTS, upgrade_outbox_journal,
)
from test_journal_outbox import SCOPE, DIGEST, commit, memory


def rows(path, table):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()


@pytest.fixture
def old(tmp_path):
    source = JournalStateStore(tmp_path / "source" / "authority.db", receipt_mode=True)
    commit(source)
    commit(source, key="deleted-memory", ident=1)
    state = source.load()
    state["items"].pop()
    state["items"][0]["text"] = "updated after receipt"
    state["marker"] = True
    source.save(state, operation="update-delete")
    state["marker"] = 1
    source.save(state, operation="metadata-integer")
    return source


def migrated(old, tmp_path):
    target = tmp_path / "destination" / "authority.db"
    report = upgrade_outbox_journal(old.path, target)
    return JournalStateStore(target, receipt_mode=True, outbox_mode=True), report


def test_preserves_raw_history_receipts_identity_and_exact_current_state(old, tmp_path):
    original = old.verified_snapshot()
    decisions, receipts = rows(old.path, "decisions"), rows(old.path, "receipts")
    target, report = migrated(old, tmp_path)
    identity, revision, state = target.verified_snapshot()
    assert identity == original[0] and revision == original[1] + 1
    assert _encode(state) == _encode(original[2])
    assert rows(target.path, "decisions")[:-1] == decisions
    assert rows(target.path, "receipts") == receipts
    assert old.verified_snapshot() == original
    assert report["preserved_decision_count"] == original[1]
    assert report["preserved_receipt_count"] == 2
    assert target.lookup_receipt(scope=SCOPE, key="deleted-memory", request_digest=DIGEST)["result"]["memory"] == memory(1)
    baseline = target.outbox_events()["events"]
    assert len(baseline) == 1
    assert baseline[0]["source_revision"] == original[1] + 1
    assert baseline[0]["state"] == state
    assert baseline[0]["receipt"] is None
    assert target.decisions()[:-1] == old.decisions()
    assert target.decisions(after_revision=original[1] - 1, limit=2) == old.decisions(after_revision=original[1] - 1) + target.decisions(after_revision=original[1])


def test_empty_source_baseline_and_receipt_retry_after_cutover(tmp_path):
    source = JournalStateStore(tmp_path / "source" / "a.db", receipt_mode=True)
    target, report = migrated(source, tmp_path)
    assert report["source_revision"] == 0 and report["boundary_revision"] == 1
    assert target.outbox_events()["events"][0]["state"] == {"items": [], "lineage": []}
    receipt = commit(target)
    assert receipt["revision"] == 2
    assert target.save_with_receipt({}, scope=SCOPE, key="request", request_digest=DIGEST,
                                   result={"memory": memory()}, expected_revision=0) == receipt
    assert target.outbox_events()["head_revision"] == 2


def test_future_mutations_contiguous_preserve_origin_and_old_retry(old, tmp_path):
    target, report = migrated(old, tmp_path)
    before = target.outbox_events()
    receipt = target.lookup_receipt(scope=SCOPE, key="request", request_digest=DIGEST)
    assert target.save_with_receipt({}, scope=SCOPE, key="request", request_digest=DIGEST,
        result={"memory": memory()}, expected_revision=0) == receipt
    assert target.outbox_events() == before
    state = target.load()
    state["items"][0]["text"] = "new update"
    target.save(state, operation="update")
    state["lineage"] = state["items"]
    state["items"] = []
    target.save(state, operation="promotion")
    state["lineage"] = []
    target.save(state, operation="delete")
    page = target.outbox_events(limit=2)
    assert page["schema_version"] == 2 and page["has_more"]
    assert page["origin"] == report["origin"]
    assert [event["source_revision"] for event in page["events"]] == [5, 6]
    second = target.outbox_events(after_revision=6)
    assert [event["source_revision"] for event in second["events"]] == [7, 8]
    assert second["prefix_digest"] == _digest(_encode([event["checksum"] for event in page["events"]]))
    assert not second["has_more"]
    for boundary in (1, 2, 3, 4):
        with pytest.raises(ValueError, match="precedes"):
            target.outbox_events(after_revision=boundary)


@pytest.mark.parametrize("version", [1, 3, 4])
def test_only_schema2_source_supported(tmp_path, version):
    if version == 4:
        source2 = JournalStateStore(tmp_path / "original" / "a.db", receipt_mode=True)
        source, _ = migrated(source2, tmp_path)
    else:
        source = JournalStateStore(tmp_path / "source" / "a.db", receipt_mode=version >= 2, outbox_mode=version == 3)
    with pytest.raises(JournalSchemaError):
        upgrade_outbox_journal(source.path, tmp_path / "another" / "a.db")
    assert not (tmp_path / "another").exists()


@pytest.mark.parametrize("suffix", ["", ".identity.json", ".migration.json", ".init.lock", "-wal", "-shm"])
@pytest.mark.parametrize("dangling", [False, True])
def test_destination_and_all_sidecars_never_follow_symlinks(old, tmp_path, suffix, dangling):
    target = tmp_path / "dest" / "a.db"
    target.parent.mkdir()
    alias = Path(str(target) + suffix)
    unrelated = tmp_path / "unrelated"
    if not dangling:
        unrelated.write_bytes(b"preserve")
    try:
        alias.symlink_to(unrelated)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    before = old.verified_snapshot()
    with pytest.raises(ValueError):
        upgrade_outbox_journal(old.path, target)
    assert alias.is_symlink()
    assert unrelated.exists() is not dangling
    if not dangling:
        assert unrelated.read_bytes() == b"preserve"
    assert old.verified_snapshot() == before


@pytest.mark.parametrize("suffix", ["", ".identity.json", ".migration.json", ".init.lock", "-wal", "-shm"])
def test_existing_destination_artifacts_are_preserved(old, tmp_path, suffix):
    target = tmp_path / "dest" / "a.db"
    target.parent.mkdir()
    existing = Path(str(target) + suffix)
    existing.write_bytes(b"existing")
    with pytest.raises(ValueError):
        upgrade_outbox_journal(old.path, target)
    assert existing.read_bytes() == b"existing"


def test_missing_source_and_same_directory_fail_without_creation(old, tmp_path):
    missing = tmp_path / "absent" / "a.db"
    target = tmp_path / "dest" / "a.db"
    with pytest.raises(ValueError):
        upgrade_outbox_journal(missing, target)
    assert not missing.parent.exists() and not target.parent.exists()
    with pytest.raises(ValueError):
        upgrade_outbox_journal(old.path, old.path.with_name("other.db"))
    assert not old.path.with_name("other.db").exists()


@pytest.mark.parametrize("point", OUTBOX_MIGRATION_FAULT_POINTS)
def test_disk_full_at_migration_boundary_preserves_source_and_explicit_failure(old, tmp_path, point):
    original = old.verified_snapshot()
    target = tmp_path / "dest" / "a.db"
    def fault(here):
        if here == point:
            raise OSError(errno.ENOSPC, "injected")
    with pytest.raises(OSError):
        upgrade_outbox_journal(old.path, target, fault_hook=fault)
    assert old.verified_snapshot() == original
    if point == "migration_before_marker":
        assert not target.exists() and not Path(str(target) + ".migration.json").exists()
    elif point == "migration_after_publish":
        assert JournalStateStore(target).schema_version == 4
    else:
        assert Path(str(target) + ".migration.json").is_file()
        with pytest.raises(JournalIntegrityError):
            JournalStateStore(target)
    # Recovery deliberately uses a new location; failed artifacts remain.
    fresh = tmp_path / "retry" / "a.db"
    upgrade_outbox_journal(old.path, fresh)
    assert JournalStateStore(fresh).verified_snapshot()[2] == original[2]


def test_direct_source_writer_during_copy_refuses_stale_publication(old, tmp_path):
    target = tmp_path / "dest" / "a.db"
    def fault(here):
        if here == "migration_after_backup":
            writer = JournalStateStore(old.path, receipt_mode=True)
            state = writer.load()
            state["latest"] = "concurrent-write"
            writer.save(state)
    with pytest.raises(RevisionConflict):
        upgrade_outbox_journal(old.path, target, fault_hook=fault)
    assert old.load()["latest"] == "concurrent-write"
    with pytest.raises(JournalIntegrityError):
        JournalStateStore(target)


@pytest.mark.parametrize("table", ["outbox", "migration_origin", "receipts", "decisions"])
def test_missing_migration_or_authority_rows_fail_closed(old, tmp_path, table):
    target, _ = migrated(old, tmp_path)
    with closing(sqlite3.connect(target.path)) as connection, connection:
        connection.execute(f"DELETE FROM {table}")
    with pytest.raises(JournalIntegrityError):
        JournalStateStore(target.path)


@pytest.mark.parametrize("field,value", [("source_revision", True), ("source_revision", -1),
    ("source_schema_version", True), ("source_schema_version", 3),
    ("source_digest", "bad"), ("history_digest", "bad")])
def test_origin_wire_validation_rejects_invalid_fields_with_recomputed_checksum(old, tmp_path, field, value):
    target, _ = migrated(old, tmp_path)
    event = target.outbox_events()["events"][0]
    event["origin"][field] = value
    event["checksum"] = _digest(_encode({key: value for key, value in event.items() if key != "checksum"}))
    with pytest.raises(JournalIntegrityError):
        validate_outbox_event(event)


def test_history_digest_tamper_and_downgrade_reject(old, tmp_path):
    target, _ = migrated(old, tmp_path)
    with closing(sqlite3.connect(target.path)) as connection, connection:
        origin = json.loads(connection.execute("SELECT payload FROM migration_origin").fetchone()[0])
        origin["history_digest"] = "a" * 64
        raw = _encode(origin)
        connection.execute("UPDATE migration_origin SET payload=?,checksum=?", (raw, _digest(raw)))
    with pytest.raises(JournalIntegrityError, match="history"):
        target.load()
    with closing(sqlite3.connect(target.path)) as connection, connection:
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(JournalSchemaError):
        JournalStateStore(target.path)
