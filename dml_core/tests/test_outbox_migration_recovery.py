"""Pinned legacy reader/state fixtures and process-death migration oracles."""
from __future__ import annotations

from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.outbox_migration import OUTBOX_MIGRATION_FAULT_POINTS, upgrade_outbox_journal
from test_projection import run_child

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def legacy(tmp_path):
    sql = FIXTURES / "journal_schema2_8a08efa.sql"
    manifest = json.loads((FIXTURES / "journal_schema2_8a08efa.json").read_text())
    assert hashlib.sha256(sql.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    path = tmp_path / "legacy" / "source.sqlite3"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(sql.read_text())
    source = JournalStateStore(path)
    return source, manifest


def historical_rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return {
            "decisions": connection.execute("SELECT revision,payload,checksum FROM decisions ORDER BY revision").fetchall(),
            "receipts": connection.execute("SELECT scope,key,revision,payload,checksum FROM receipts ORDER BY scope,key").fetchall(),
        }


def assert_preserved(source, migrated, manifest, original_rows):
    assert migrated.schema_version == 4
    assert migrated.verified_snapshot()[0] == source.verified_snapshot()[0]
    assert migrated.read_snapshot()[0] == manifest["head_revision"] + 1
    assert migrated.load() == source.load()
    rows = historical_rows(migrated.path)
    assert rows["receipts"] == original_rows["receipts"]
    assert rows["decisions"][:-1] == original_rows["decisions"]
    assert migrated.decisions()[:-1] == source.decisions()
    page = migrated.outbox_events()
    assert len(page["events"]) == 1
    event = page["events"][0]
    assert event["source_revision"] == manifest["head_revision"] + 1
    assert event["operation"] == "outbox-migration-baseline-v1"
    assert event["origin"]["source_revision"] == manifest["head_revision"]
    assert event["state"] == source.load()
    for receipt in manifest["receipts"]:
        assert migrated.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt
    # The deleted original remains only a historical receipt, not live memory.
    assert [record["id"] for record in migrated.load()["items"]] == [1]


def test_pinned_schema2_fixture_preserves_bytes_receipts_and_explicit_baseline(legacy, tmp_path):
    source, manifest = legacy
    rows = historical_rows(source.path)
    before = source.path.read_bytes()
    destination = tmp_path / "migrated" / "target.sqlite3"
    upgrade_outbox_journal(source.path, destination)
    assert source.path.read_bytes() == before
    assert_preserved(source, JournalStateStore(destination), manifest, rows)


def test_actual_pinned_legacy_reader_rejects_migrated_schema_before_mutation(legacy, tmp_path):
    source, manifest = legacy
    reader_path = FIXTURES / "journal_schema2_reader_8a08efa.py"
    assert hashlib.sha256(reader_path.read_bytes()).hexdigest() == manifest["reader_sha256"]
    spec = importlib.util.spec_from_file_location("daystrom_dml._pinned_schema2_reader", reader_path)
    assert spec is not None and spec.loader is not None
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    assert reader.JournalStateStore(source.path).load() == source.load()
    destination = tmp_path / "migrated" / "target.sqlite3"
    upgrade_outbox_journal(source.path, destination)
    before = destination.read_bytes()
    with pytest.raises(reader.JournalSchemaError):
        reader.JournalStateStore(destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("point", OUTBOX_MIGRATION_FAULT_POINTS)
def test_migration_process_kill_never_exposes_an_incomplete_authority(legacy, tmp_path, point):
    source, manifest = legacy
    rows = historical_rows(source.path)
    before = source.path.read_bytes()
    destination = tmp_path / "migrated" / "target.sqlite3"
    child = run_child('''
import os, signal, sys, threading
from pathlib import Path
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
source, destination, point = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
            threading.Event().wait(5)
            os._exit(80)
        else:
            os._exit(79)
upgrade_outbox_journal(Path(source), Path(destination), fault_hook=crash)
raise AssertionError('migration hook not reached')
''', source.path, destination, point)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    assert source.path.read_bytes() == before
    assert historical_rows(source.path) == rows
    assert source.schema_version == 2
    if point == "migration_before_marker":
        assert not destination.exists()
        assert destination.with_name(destination.name + ".init.lock").exists()
        with pytest.raises(ValueError, match="unused destination"):
            upgrade_outbox_journal(source.path, destination)
        recovery_path = tmp_path / "recovered" / "target.sqlite3"
        upgrade_outbox_journal(source.path, recovery_path)
        assert_preserved(source, JournalStateStore(recovery_path), manifest, rows)
    elif point == "migration_after_publish":
        assert_preserved(source, JournalStateStore(destination), manifest, rows)
    else:
        marker = destination.with_name(destination.name + ".migration.json")
        assert marker.is_file()
        existed = destination.exists()
        with pytest.raises(JournalIntegrityError):
            JournalStateStore(destination)
        assert destination.exists() == existed
        recovery_path = tmp_path / "recovered" / "target.sqlite3"
        upgrade_outbox_journal(source.path, recovery_path)
        assert_preserved(source, JournalStateStore(recovery_path), manifest, rows)
