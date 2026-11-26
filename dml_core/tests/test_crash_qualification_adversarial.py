"""Independent recovery oracles: retained acknowledgements and sealed evidence.

These cases use synthetic stores. They establish software failure handling, not
physical power-loss durability or certification of an underlying storage device.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import sqlite3

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
import daystrom_dml.services.authority_backup as backup_module
from daystrom_dml.services.authority_backup import (
    DATABASE, IDENTITY, MANIFEST, MARKER, backup_authority, restore_backup,
    verify_backup,
)
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import append_receipted, canonical_request


def _append(journal, key, text):
    request, digest = canonical_request(text, tenant_id="independent-tenant")
    return append_receipted(
        journal, request=request, request_digest=digest, key=key,
        embed=lambda _: [1.0, 0.0], capacity=20,
        embedding_space=lambda: {"backend": "independent", "revision": "v1"},
        hydrate=lambda *_: None, degraded=lambda _: None,
    )


@pytest.fixture(params=[2, 3, 4], ids=["schema2", "schema3", "schema4"])
def authority(tmp_path, request):
    schema = request.param
    folder = tmp_path / ("legacy" if schema == 4 else "source")
    journal = JournalStateStore(folder / DATABASE, receipt_mode=True, outbox_mode=schema == 3)
    receipt = _append(journal, "ack-one", "An independently acknowledged memory.")
    if schema == 4:
        path = tmp_path / "source" / DATABASE
        upgrade_outbox_journal(journal.path, path)
        journal = JournalStateStore(path, receipt_mode=True, outbox_mode=True)
    return journal, receipt


def _bundle(tmp_path, authority):
    journal, receipt = authority
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    return folder, journal, receipt


def _bytes(folder):
    return {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


def _rewrite_manifest(folder, mutate):
    path = folder / MANIFEST
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))


def test_old_valid_backup_cannot_explain_later_acknowledgement(tmp_path, authority):
    folder, journal, first = _bundle(tmp_path, authority)
    later = _append(journal, "ack-two", "Acknowledged after the backup snapshot.")
    assert verify_backup(folder, receipts=[first])["receipt_comparison"]["matched"] == 1
    destination = tmp_path / "must-not-exist"
    with pytest.raises(JournalIntegrityError):
        restore_backup(folder, destination, receipts=[first, later])
    assert not destination.exists()
    assert journal.lookup_receipt(later["scope"], later["key"], later["request_digest"]) == later


def test_independent_receipt_content_must_match_not_just_key_and_revision(tmp_path, authority):
    folder, journal, receipt = _bundle(tmp_path, authority)
    forged = copy.deepcopy(receipt)
    forged["result"]["memory"]["text"] = "A memory never acknowledged by this store."
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder, receipts=[forged])
    assert journal.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt


def test_valid_foreign_receipt_is_not_evidence_for_this_store(tmp_path, authority):
    folder, _, _ = _bundle(tmp_path, authority)
    foreign = JournalStateStore(tmp_path / "foreign" / DATABASE, receipt_mode=True)
    receipt = _append(foreign, "ack-one", "An independently acknowledged memory.")
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder, receipts=[receipt])


@pytest.mark.parametrize("field,value", [
    ("revision", 0), ("receipt_count", 0), ("store_id", "f" * 32),
    ("authority_digest", "0" * 64), ("state_digest", "0" * 64),
])
def test_manifest_claims_are_compared_to_actual_authority(tmp_path, field, value):
    journal = JournalStateStore(tmp_path / "source" / DATABASE, receipt_mode=True)
    _append(journal, "ack-one", "Acknowledged source.")
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    _rewrite_manifest(folder, lambda manifest: manifest["authority"].__setitem__(field, value))
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder)


@pytest.mark.parametrize("field", ["revision", "receipt_count"])
def test_manifest_integer_fields_do_not_accept_boolean_aliases(tmp_path, field):
    journal = JournalStateStore(tmp_path / "source" / DATABASE, receipt_mode=True)
    _append(journal, "ack-one", "Acknowledged source.")
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    _rewrite_manifest(folder, lambda manifest: manifest["authority"].__setitem__(field, True))
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder)


def test_recomputed_file_hash_cannot_hide_broken_receipt_history(tmp_path, authority):
    folder, _, receipt = _bundle(tmp_path, authority)
    database = folder / DATABASE
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM receipts WHERE key=?", (receipt["key"],))
    # This connection is deliberately closed before recomputing main-file hash.
    # A matching file checksum cannot replace journal history validation.
    connection.close()
    payload = database.read_bytes()
    _rewrite_manifest(folder, lambda manifest: manifest["files"].__setitem__(DATABASE, {
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
    }))
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder)


@pytest.mark.parametrize("extra", [DATABASE + "-wal", DATABASE + "-shm", MARKER, "unlisted.json"])
def test_sealed_bundle_rejects_unlisted_wal_or_incomplete_evidence(tmp_path, extra):
    journal = JournalStateStore(tmp_path / "source" / DATABASE, receipt_mode=True)
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    (folder / extra).write_bytes(b"must not be ignored")
    with pytest.raises(JournalIntegrityError):
        verify_backup(folder)
    with pytest.raises(JournalIntegrityError):
        restore_backup(folder, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


@pytest.mark.parametrize("operation", ["backup", "restore"])
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_destination_is_never_overwritten(tmp_path, operation, kind):
    journal = JournalStateStore(tmp_path / "source" / DATABASE, receipt_mode=True)
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    destination = tmp_path / "occupied"
    if kind == "file":
        destination.write_bytes(b"operator-owned file")
    else:
        destination.mkdir()
        (destination / "operator-file").write_bytes(b"operator-owned directory")
    before = destination.read_bytes() if kind == "file" else _bytes(destination)
    with pytest.raises(ValueError):
        if operation == "backup":
            backup_authority(journal.path, destination)
        else:
            restore_backup(folder, destination)
    after = destination.read_bytes() if kind == "file" else _bytes(destination)
    assert after == before


@pytest.mark.parametrize("operation", ["backup", "restore"])
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EIO])
def test_marker_write_failure_never_exposes_empty_requested_destination(tmp_path, monkeypatch, operation, error_number):
    journal = JournalStateStore(tmp_path / "source" / DATABASE, receipt_mode=True)
    receipt = _append(journal, "ack-one", "Acknowledged before reservation failure.")
    folder = tmp_path / "backup"
    backup_authority(journal.path, folder)
    destination = tmp_path / "restore-or-backup"
    original = backup_module.atomic_write_text

    def broken_marker(path, text):
        if path.name == MARKER:
            raise OSError(error_number, "Injected marker write failure")
        return original(path, text)

    with monkeypatch.context() as fault:
        fault.setattr(backup_module, "atomic_write_text", broken_marker)
        with pytest.raises(OSError, match="Injected marker write failure"):
            if operation == "backup":
                backup_authority(journal.path, destination)
            else:
                restore_backup(folder, destination, receipts=[receipt])
    assert not destination.exists()
    assert journal.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt
    # Recovery uses a new location, not a failed private reservation artifact.
    recovered = tmp_path / "new-attempt"
    restore_backup(folder, recovered, receipts=[receipt])
    assert verify_backup(recovered, receipts=[receipt])["receipt_comparison"]["matched"] == 1


@pytest.mark.parametrize("point,published", [("before_publish", False), ("after_publish", True)])
def test_interrupted_restore_has_explicit_outcome_and_keeps_original(tmp_path, authority, point, published):
    folder, journal, receipt = _bundle(tmp_path, authority)
    before = _bytes(folder)
    destination = tmp_path / "restore"

    def interrupt(current):
        if current == point:
            raise OSError("Injected publication failure")

    with pytest.raises(OSError, match="Injected publication failure"):
        restore_backup(folder, destination, receipts=[receipt], fault_hook=interrupt)
    assert _bytes(folder) == before
    assert (destination / MARKER).exists() is not published
    if published:
        assert verify_backup(destination, receipts=[receipt])["receipt_comparison"]["matched"] == 1
    else:
        with pytest.raises(JournalIntegrityError):
            verify_backup(destination, receipts=[receipt])
        with pytest.raises(JournalIntegrityError):
            JournalStateStore(destination / DATABASE, receipt_mode=True)
    assert journal.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt


def test_postbackup_source_damage_does_not_invalidate_separate_verified_copy(tmp_path, authority):
    folder, journal, receipt = _bundle(tmp_path, authority)
    journal.path.write_bytes(b"corrupted authority must be preserved for investigation")
    destination = tmp_path / "restore"
    restore_backup(folder, destination, receipts=[receipt])
    reopened = JournalStateStore(destination / DATABASE, receipt_mode=True)
    assert reopened.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt
    assert journal.path.read_bytes() == b"corrupted authority must be preserved for investigation"
    assert (destination / IDENTITY).read_bytes() == (folder / IDENTITY).read_bytes()
