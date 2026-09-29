"""Independent regressions for migration authority and output preservation."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal


@pytest.mark.parametrize("suffix", ["", ".identity.json", ".migration.json", ".init.lock", "-wal", "-shm"])
def test_existing_dangling_output_symlinks_remain_untouched(tmp_path, suffix):
    source = JournalStateStore(tmp_path / "source" / "journal.db", receipt_mode=True)
    source.save({"items": [], "value": True}, expected_revision=0)
    before = source.verified_snapshot()
    destination = tmp_path / "destination" / "journal.db"
    destination.parent.mkdir()
    reserved = Path(str(destination) + suffix)
    unrelated = tmp_path / "unrelated-target"
    try:
        reserved.symlink_to(unrelated)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform")
    original_link = os.readlink(reserved)

    with pytest.raises(ValueError):
        upgrade_outbox_journal(source.path, destination)

    assert reserved.is_symlink()
    assert os.readlink(reserved) == original_link
    assert not unrelated.exists()
    assert not destination.exists()
    assert source.verified_snapshot() == before


def test_decisions_remain_readable_across_legacy_baseline_and_new_commit(tmp_path):
    source = JournalStateStore(tmp_path / "source" / "journal.db", receipt_mode=True)
    source.save({"items": [], "value": True}, expected_revision=0)
    source.save({"items": [], "value": 1}, expected_revision=1)
    legacy = source.decisions()
    destination = tmp_path / "destination" / "journal.db"
    upgrade_outbox_journal(source.path, destination)
    migrated = JournalStateStore(destination)
    migrated.save({"items": [], "value": 1.0}, expected_revision=3)

    decisions = migrated.decisions()
    assert decisions[:2] == legacy
    assert [decision["schema_version"] for decision in decisions] == [2, 2, 4, 4]
    assert [decision["revision"] for decision in migrated.decisions(after_revision=1, limit=2)] == [2, 3]
    assert len({decision["state_digest"] for decision in decisions}) == 3
    assert migrated.outbox_events()["events"][0]["state"]["value"] == 1
    assert type(migrated.load()["value"]) is float


def test_identity_change_between_connection_guard_and_snapshot_is_rejected(tmp_path, monkeypatch):
    source = JournalStateStore(tmp_path / "source" / "journal.db", receipt_mode=True)
    source.save({"items": [], "value": "pinned authority"}, expected_revision=0)
    destination = tmp_path / "destination" / "journal.db"
    original_connect = JournalStateStore._connect
    injected = False

    @contextmanager
    def race_identity(store):
        nonlocal injected
        with original_connect(store) as connection:
            if store.path == source.path and not injected:
                injected = True
                with sqlite3.connect(source.path) as writer:
                    writer.execute("UPDATE identity SET store_id=? WHERE id=1", ("f" * 32,))
            yield connection

    monkeypatch.setattr(JournalStateStore, "_connect", race_identity)
    with pytest.raises(JournalIntegrityError):
        upgrade_outbox_journal(source.path, destination)
    assert injected
    assert not destination.exists()
    assert not Path(str(destination) + ".migration.json").exists()
