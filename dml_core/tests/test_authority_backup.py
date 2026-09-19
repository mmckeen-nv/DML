"""Full-authority backup oracles use submitted receipts and direct SQL history."""
from __future__ import annotations

from contextlib import closing
import errno
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.authority_backup import (
    BACKUP_FAULT_POINTS, DATABASE, IDENTITY, MANIFEST, MARKER,
    backup_authority, restore_backup, verify_backup,
)
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from scripts.dml_journal import main

SCOPE = {"tenant_id": "backup-owner", "client_id": None, "session_id": None, "instance_id": None}


def append(journal, text):
    revision, payload = journal.read_snapshot()
    record = {"id": len(payload["items"]), "text": text, "meta": SCOPE}
    payload["items"].append(record)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return journal.save_with_receipt(payload, scope=SCOPE, key=text, request_digest=digest,
                                     result={"memory": record}, expected_revision=revision)


def make(tmp_path, schema=2):
    source = tmp_path / "source"
    source.mkdir()
    journal = JournalStateStore(source / DATABASE, receipt_mode=True, outbox_mode=schema == 3)
    receipts = [append(journal, "first acknowledged"), append(journal, "second acknowledged")]
    if schema == 4:
        destination = tmp_path / "migrated" / DATABASE
        upgrade_outbox_journal(journal.path, destination)
        journal = JournalStateStore(destination, receipt_mode=True, outbox_mode=True)
    return journal, receipts


def rows(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: sorted(connection.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr) for name in names}


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_full_roundtrip_preserves_every_table_and_exact_historical_receipts(tmp_path, schema):
    journal, receipts = make(tmp_path, schema)
    original = rows(journal.path)
    manifest = backup_authority(journal.path, tmp_path / "backup")
    assert manifest["authority"]["receipt_count"] == 2
    assert manifest["authority"]["revision"] == (3 if schema == 4 else 2)
    assert manifest["receipt_comparison"]["coverage"] == "not_provided"
    verified = verify_backup(tmp_path / "backup", receipts=receipts)
    assert verified["receipt_comparison"] == {"provided": 2, "matched": 2, "coverage": "provided_receipts_only"}
    assert restore_backup(tmp_path / "backup", tmp_path / "restored", receipts=receipts) == verified
    assert rows(journal.path) == original == rows(tmp_path / "restored" / DATABASE)
    restored = JournalStateStore(tmp_path / "restored" / DATABASE, receipt_mode=True, outbox_mode=schema != 2)
    for receipt in receipts:
        assert restored.lookup_receipt(SCOPE, receipt["key"], receipt["request_digest"]) == receipt
    append(restored, "new after restore")
    assert rows(journal.path) == original


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_live_committed_wal_is_copied_into_self_contained_backup(tmp_path, schema):
    journal, receipts = make(tmp_path, schema)
    # Retain a SQLite connection so commits remain in the real WAL file.
    with closing(sqlite3.connect(journal.path)) as keepalive:
        keepalive.execute("PRAGMA wal_autocheckpoint=0")
        keepalive.execute("SELECT count(*) FROM records").fetchone()
        receipts.append(append(journal, "acknowledged only in WAL"))
        assert Path(str(journal.path) + "-wal").stat().st_size > 0
        manifest = backup_authority(journal.path, tmp_path / "backup")
        assert manifest["authority"]["revision"] == receipts[-1]["revision"]
    assert {p.name for p in (tmp_path / "backup").iterdir()} == {DATABASE, IDENTITY, MANIFEST}
    assert verify_backup(tmp_path / "backup", receipts=receipts)["receipt_comparison"]["matched"] == 3


@pytest.mark.parametrize("operation", ["backup", "restore"])
@pytest.mark.parametrize("point", BACKUP_FAULT_POINTS)
def test_process_kill_preserves_source_and_quarantines_incomplete_target(tmp_path, operation, point):
    journal, receipts = make(tmp_path)
    original = rows(journal.path)
    source = journal.path
    if operation == "restore":
        backup_authority(source, tmp_path / "backup")
        source = tmp_path / "backup"
    destination = tmp_path / "interrupted"
    script = '''
import os,sys
from pathlib import Path
from daystrom_dml.services.authority_backup import backup_authority,restore_backup
def die(point):
    if point==sys.argv[3]: os._exit(91)
action=backup_authority if sys.argv[4]=='backup' else restore_backup
action(Path(sys.argv[1]),Path(sys.argv[2]),fault_hook=die)
'''
    result = subprocess.run([sys.executable, "-c", script, str(source), str(destination), point, operation],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 91, result.stderr
    assert rows(journal.path) == original
    if point == "after_publish":
        assert verify_backup(destination, receipts=receipts)["receipt_comparison"]["matched"] == 2
    else:
        assert (destination / MARKER).is_file()
        with pytest.raises(JournalIntegrityError):
            verify_backup(destination)
        with pytest.raises(JournalIntegrityError, match="Incomplete journal migration"):
            JournalStateStore(destination / DATABASE, receipt_mode=True)


@pytest.mark.parametrize("operation", ["backup", "restore"])
@pytest.mark.parametrize("point", BACKUP_FAULT_POINTS)
def test_caught_disk_full_at_publication_boundaries_has_explicit_outcome(tmp_path, operation, point):
    journal, receipts = make(tmp_path)
    original = rows(journal.path)
    source = journal.path
    action = backup_authority
    if operation == "restore":
        backup_authority(source, tmp_path / "backup")
        source = tmp_path / "backup"
        action = restore_backup

    def full(here):
        if here == point:
            raise OSError(errno.ENOSPC, "synthetic full device")

    with pytest.raises(OSError):
        action(source, tmp_path / "failed", fault_hook=full)
    assert rows(journal.path) == original
    if point == "after_publish":
        verify_backup(tmp_path / "failed", receipts=receipts)
    else:
        with pytest.raises(JournalIntegrityError):
            verify_backup(tmp_path / "failed")


@pytest.mark.parametrize("damage", ["missing-identity", "wrong-identity", "migration", "corrupt-db", "nonwal", "legacy"])
def test_invalid_source_rejects_before_creating_destination(tmp_path, damage):
    journal, _ = make(tmp_path)
    if damage == "missing-identity":
        journal.identity_path.unlink()
    elif damage == "wrong-identity":
        journal.identity_path.write_text(json.dumps({"schema_version": 1, "store_id": "0" * 32}), encoding="utf-8")
    elif damage == "migration":
        journal.migration_path.write_text("{}", encoding="utf-8")
    elif damage == "corrupt-db":
        journal.path.write_bytes(b"not sqlite")
    elif damage == "nonwal":
        with closing(sqlite3.connect(journal.path)) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
    else:
        with closing(sqlite3.connect(journal.path)) as connection:
            connection.execute("PRAGMA user_version=1")
    before = {p.name: p.read_bytes() for p in journal.path.parent.iterdir() if p.is_file()}
    with pytest.raises((JournalIntegrityError, sqlite3.DatabaseError)):
        backup_authority(journal.path, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    assert {p.name: p.read_bytes() for p in journal.path.parent.iterdir() if p.is_file() and not p.name.startswith(".dml_store.lock") and not p.name.endswith(("-wal", "-shm"))} == {
        name: value for name, value in before.items() if not name.startswith(".dml_store.lock") and not name.endswith(("-wal", "-shm"))}


@pytest.mark.parametrize("damage", ["database", "identity", "manifest", "future", "missing", "extra", "wal", "marker", "head", "schema"])
def test_damaged_backup_never_restores_or_overwrites(tmp_path, damage):
    journal, _ = make(tmp_path)
    directory = tmp_path / "backup"
    backup_authority(journal.path, directory)
    if damage in {"database", "identity"}:
        path = directory / (DATABASE if damage == "database" else IDENTITY)
        path.write_bytes(path.read_bytes() + b"damaged")
    elif damage == "manifest":
        (directory / MANIFEST).write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    elif damage in {"future", "head", "schema"}:
        manifest = json.loads((directory / MANIFEST).read_text())
        if damage == "future":
            manifest["schema_version"] = "future"
        else:
            manifest["authority"]["revision" if damage == "head" else "journal_schema_version"] += 1
        (directory / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "missing":
        (directory / IDENTITY).unlink()
    else:
        (directory / {"extra": "unlisted", "wal": DATABASE + "-wal", "marker": MARKER}[damage]).write_bytes(b"x")
    with pytest.raises(JournalIntegrityError):
        restore_backup(directory, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_old_valid_backup_fails_external_later_acknowledgement_comparison(tmp_path):
    journal, receipts = make(tmp_path)
    backup_authority(journal.path, tmp_path / "backup")
    receipts.append(append(journal, "later acknowledged"))
    # Internal consistency alone cannot detect a valid rollback.
    assert verify_backup(tmp_path / "backup")["authority"]["revision"] == 2
    with pytest.raises(JournalIntegrityError, match="outside"):
        restore_backup(tmp_path / "backup", tmp_path / "refused", receipts=receipts)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("damage", ["store", "content", "key", "type"])
def test_retained_receipts_must_exactly_match_verified_history(tmp_path, damage):
    journal, receipts = make(tmp_path)
    backup_authority(journal.path, tmp_path / "backup")
    if damage == "store":
        receipts[0]["store_id"] = "0" * 32
    elif damage == "content":
        receipts[0]["result"]["memory"]["text"] = "not the original acknowledgement"
    elif damage == "key":
        receipts[0]["key"] = "absent"
    else:
        receipts = {"invalid": "array"}
    with pytest.raises((JournalIntegrityError, ValueError)):
        verify_backup(tmp_path / "backup", receipts=receipts)


@pytest.mark.parametrize("destination", ["existing", "source", "nested"])
def test_backup_destination_must_be_new_and_separate(tmp_path, destination):
    journal, _ = make(tmp_path)
    if destination == "existing":
        target = tmp_path / "existing"
        target.mkdir()
        (target / "sentinel").write_bytes(b"preserve")
    elif destination == "source":
        target = journal.path.parent
    else:
        target = journal.path.parent / "nested"
    before = rows(journal.path)
    with pytest.raises(ValueError):
        backup_authority(journal.path, target)
    assert rows(journal.path) == before
    if destination == "existing":
        assert (target / "sentinel").read_bytes() == b"preserve"


def test_cli_backup_verify_restore_and_sanitized_failure(tmp_path, capsys):
    journal, receipts = make(tmp_path)
    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(json.dumps(receipts), encoding="utf-8")
    assert main(["backup", str(journal.path), str(tmp_path / "backup")]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert main(["verify-backup", str(tmp_path / "backup"), "--receipts", str(receipt_path)]) == 0
    assert json.loads(capsys.readouterr().out)["receipt_comparison"]["matched"] == 2
    assert main(["restore", str(tmp_path / "backup"), str(tmp_path / "restored"), "--receipts", str(receipt_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert main(["restore", str(tmp_path / "backup"), str(tmp_path / "restored")]) == 2
    error = json.loads(capsys.readouterr().out)
    assert error == {"ok": False, "error": "ValueError"}


def test_source_commit_during_copy_refuses_stale_publication(tmp_path):
    journal, _ = make(tmp_path)

    def advance(point):
        if point == "after_copy":
            append(journal, "direct writer violated offline precondition")

    from daystrom_dml.journal import RevisionConflict
    with pytest.raises(RevisionConflict):
        backup_authority(journal.path, tmp_path / "stale", fault_hook=advance)
    assert journal.read_snapshot()[0] == 3
    assert (tmp_path / "stale" / MARKER).exists()
    with pytest.raises(JournalIntegrityError):
        verify_backup(tmp_path / "stale")


@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_reservation_marker_write_failure_never_exposes_empty_destination(tmp_path, monkeypatch, operation):
    import daystrom_dml.services.authority_backup as backup_module

    journal, _ = make(tmp_path)
    source = journal.path
    action = backup_authority
    if operation == "restore":
        backup_authority(source, tmp_path / "backup")
        source = tmp_path / "backup"
        action = restore_backup
    real_write = backup_module.atomic_write_text

    def fail_marker(path, text):
        if path.name == MARKER:
            raise OSError(errno.ENOSPC, "reservation marker unavailable")
        return real_write(path, text)

    monkeypatch.setattr(backup_module, "atomic_write_text", fail_marker)
    with pytest.raises(OSError):
        action(source, tmp_path / "unpublished")
    assert not (tmp_path / "unpublished").exists()
    assert list(tmp_path.glob(".dml-backup-staging-*"))


def test_real_sqlite_backup_failure_keeps_partial_destination_quarantined(tmp_path, monkeypatch):
    import daystrom_dml.services.authority_backup as backup_module

    journal, _ = make(tmp_path)
    original = rows(journal.path)
    real_open = backup_module._open

    class InterruptedConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def backup(self, target):
            target.execute("CREATE TABLE partial (value TEXT)")
            target.commit()
            raise sqlite3.OperationalError("database or disk is full")

    def interrupted(path, *, sealed=False):
        connection = real_open(path, sealed=sealed)
        return InterruptedConnection(connection) if path == journal.path else connection

    monkeypatch.setattr(backup_module, "_open", interrupted)
    with pytest.raises(sqlite3.OperationalError):
        backup_authority(journal.path, tmp_path / "partial")
    assert rows(journal.path) == original
    assert (tmp_path / "partial" / MARKER).is_file()
    with pytest.raises(JournalIntegrityError, match="Incomplete journal migration"):
        JournalStateStore(tmp_path / "partial" / DATABASE, receipt_mode=True)


def test_unpatched_sqlite_guard_runs_before_any_new_files(tmp_path, monkeypatch):
    import daystrom_dml.services.authority_backup as backup_module

    journal, _ = make(tmp_path)
    backup_authority(journal.path, tmp_path / "backup")
    before = set(tmp_path.iterdir())

    def reject():
        raise JournalIntegrityError("Unpatched SQLite is refused")

    monkeypatch.setattr(backup_module, "require_patched_sqlite", reject)
    for action in (
        lambda: backup_authority(journal.path, tmp_path / "refused"),
        lambda: verify_backup(tmp_path / "backup"),
        lambda: restore_backup(tmp_path / "backup", tmp_path / "refused"),
    ):
        with pytest.raises(JournalIntegrityError, match="Unpatched"):
            action()
        assert set(tmp_path.iterdir()) == before
