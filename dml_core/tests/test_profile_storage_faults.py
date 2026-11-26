"""Failure oracles for profile storage barriers and ownership.

All faults target isolated synthetic stores. Injected POSIX I/O errors exercise
error propagation, not device power-loss behavior or real filesystem exhaustion.
"""
from __future__ import annotations

from contextlib import closing
import errno
import os
from pathlib import Path
import sqlite3
import stat

import numpy as np
import pytest

from daystrom_dml import atomic_io, store_lock
from daystrom_dml.contracts.profile import PROFILE_ID, ProductionProfileError
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.profile_runtime import preflight_profile_authority


def publish(kind, target, payload):
    if kind == "bytes":
        return atomic_io.atomic_write_bytes(target, payload)
    if kind == "text":
        return atomic_io.atomic_write_text(target, payload.decode("utf-8"))
    return atomic_io.atomic_write_via(target, lambda path: path.write_bytes(payload))


@pytest.mark.skipif(os.name == "nt", reason="Windows helper has no directory fsync barrier")
@pytest.mark.parametrize("failure_errno", [errno.EIO, errno.ENOSPC, errno.EACCES, errno.EINVAL])
@pytest.mark.parametrize("kind", ["bytes", "text", "via"])
def test_directory_open_failure_reports_visible_but_unconfirmed_publication(
    tmp_path, monkeypatch, failure_errno, kind,
):
    target = tmp_path / "authority-marker.json"
    target.write_bytes(b"old generation")
    original_open = os.open
    fault = OSError(failure_errno, "synthetic directory barrier failure")

    def fail_directory(path, *args, **kwargs):
        if Path(path) == tmp_path:
            raise fault
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(atomic_io.os, "open", fail_directory)
    with pytest.raises(OSError) as observed:
        publish(kind, target, b"new generation")
    assert observed.value is fault
    # A barrier failure cannot be advertised as proof of rollback.
    assert target.read_bytes() == b"new generation"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="Windows helper has no directory fsync barrier")
@pytest.mark.parametrize("kind", ["bytes", "text", "via"])
def test_directory_fsync_failure_closes_descriptor_and_preserves_publication(tmp_path, monkeypatch, kind):
    target = tmp_path / "marker"
    original_sync = os.fsync
    directory_fds = []

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fds.append(fd)
            raise OSError(errno.EIO, "synthetic directory sync failure")
        return original_sync(fd)

    monkeypatch.setattr(atomic_io.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory sync"):
        publish(kind, target, b"published")
    assert target.read_bytes() == b"published"
    assert len(directory_fds) == 1
    with pytest.raises(OSError) as observed:
        os.fstat(directory_fds[0])
    assert observed.value.errno == errno.EBADF


@pytest.mark.parametrize("failure_errno", [errno.EIO, errno.ENOSPC])
@pytest.mark.parametrize("kind", ["bytes", "text", "via"])
def test_file_sync_failure_preserves_prior_generation(tmp_path, monkeypatch, failure_errno, kind):
    target = tmp_path / "marker"
    target.write_bytes(b"acknowledged")

    def fail_sync(_fd):
        raise OSError(failure_errno, "synthetic file sync failure")

    monkeypatch.setattr(atomic_io.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="file sync"):
        publish(kind, target, b"unacknowledged")
    assert target.read_bytes() == b"acknowledged"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("cleanup_failure", ["unlink", "unlock", "both"])
@pytest.mark.parametrize("body_failure", [False, True])
def test_lock_cleanup_always_closes_and_preserves_original_error(
    tmp_path, monkeypatch, cleanup_failure, body_failure,
):
    original_open, original_unlink = Path.open, Path.unlink
    original_unlock = store_lock.release_file_lock
    handles, unlock_attempts = [], []
    body_error = RuntimeError("original operation failed")
    unlink_error = OSError(errno.EIO, "diagnostic unlink failed")
    unlock_error = OSError(errno.EIO, "explicit unlock failed")

    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == ".dml_store.lock":
            handles.append(handle)
        return handle

    def failed_unlink(path, *args, **kwargs):
        if path.name == ".dml_store.lock.json" and cleanup_failure in {"unlink", "both"}:
            raise unlink_error
        return original_unlink(path, *args, **kwargs)

    def failed_unlock(handle):
        unlock_attempts.append(handle)
        if cleanup_failure in {"unlock", "both"}:
            raise unlock_error
        return original_unlock(handle)

    with monkeypatch.context() as faults:
        faults.setattr(Path, "open", tracked_open)
        faults.setattr(Path, "unlink", failed_unlink)
        faults.setattr(store_lock, "release_file_lock", failed_unlock)
        expected = body_error if body_failure else (
            unlink_error if cleanup_failure in {"unlink", "both"} else unlock_error
        )
        with pytest.raises(type(expected)) as observed:
            with store_lock.store_write_lock(tmp_path, operation="fault", timeout_ms=0):
                if body_failure:
                    raise body_error
        assert observed.value is expected
    assert len(handles) == 1 and handles[0].closed
    assert unlock_attempts == handles
    with store_lock.store_write_lock(tmp_path, operation="recovered", timeout_ms=0):
        pass


def test_metadata_publication_failure_keeps_primary_error_and_releases_ownership(tmp_path, monkeypatch):
    original_open = Path.open
    handles = []
    primary = OSError(errno.ENOSPC, "metadata publication failed")

    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == ".dml_store.lock":
            handles.append(handle)
        return handle

    def failed_publish(*_args):
        raise primary

    def failed_cleanup(*_args, **_kwargs):
        raise OSError(errno.EACCES, "secondary cleanup failure")

    with monkeypatch.context() as faults:
        faults.setattr(Path, "open", tracked_open)
        faults.setattr(Path, "unlink", failed_cleanup)
        faults.setattr(store_lock, "atomic_write_text", failed_publish)
        with pytest.raises(OSError) as observed:
            with store_lock.store_write_lock(tmp_path, operation="unstarted", timeout_ms=0):
                pytest.fail("A failed acquisition must not execute the operation")
        assert observed.value is primary
    assert len(handles) == 1 and handles[0].closed
    with store_lock.store_write_lock(tmp_path, operation="recovered", timeout_ms=0):
        pass


def set_delete_mode(path):
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)


def assert_delete_mode(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_wal_admission_option_is_strict_before_storage_effects(tmp_path, value):
    path = tmp_path / "must-remain-absent" / "journal.sqlite3"
    with pytest.raises(ValueError, match="require_wal"):
        JournalStateStore(path, require_wal=value)
    assert not path.parent.exists()


@pytest.mark.parametrize("outbox", [False, True])
@pytest.mark.parametrize("missing_marker", [False, True])
def test_strict_startup_rejects_non_wal_without_rewriting_authority(tmp_path, outbox, missing_marker):
    path = tmp_path / "dml_state.sqlite3"
    journal = JournalStateStore(path, receipt_mode=True, outbox_mode=outbox)
    journal.save({"items": [{"id": 0, "text": "committed"}], "lineage": []})
    if missing_marker:
        journal.identity_path.unlink()
    set_delete_mode(path)
    before = path.read_bytes()
    with pytest.raises(JournalIntegrityError, match="requires SQLite WAL"):
        JournalStateStore(path, receipt_mode=True, outbox_mode=outbox, require_wal=True)
    assert path.read_bytes() == before
    assert journal.identity_path.exists() is not missing_marker
    assert_delete_mode(path)
    with pytest.raises(ProductionProfileError, match="requires SQLite WAL"):
        preflight_profile_authority(PROFILE_ID, tmp_path, outbox_enabled=outbox)
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["load", "read_snapshot", "verified_snapshot", "stamp", "decisions", "checkpoint", "lookup_receipt", "save"])
def test_live_strict_operations_reject_mode_drift_before_mutation(tmp_path, operation):
    path = tmp_path / "dml_state.sqlite3"
    journal = JournalStateStore(path, receipt_mode=True, require_wal=True)
    journal.save({"items": [{"id": 0, "text": "committed"}], "lineage": []})
    set_delete_mode(path)
    before = path.read_bytes()
    def call():
        if operation == "lookup_receipt":
            return journal.lookup_receipt(
                {"tenant_id": "owner", "client_id": None, "session_id": None, "instance_id": None},
                "key", "a" * 64,
            )
        if operation == "save":
            return journal.save({"items": [], "lineage": []}, expected_revision=1)
        return getattr(journal, operation)()
    with pytest.raises(JournalIntegrityError, match="requires SQLite WAL"):
        call()
    assert path.read_bytes() == before
    assert_delete_mode(path)


def test_legacy_default_retains_existing_non_wal_compatibility(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    JournalStateStore(path)
    set_delete_mode(path)
    journal = JournalStateStore(path)
    journal.save({"items": [{"id": 0, "text": "legacy"}], "lineage": []})
    assert journal.read_snapshot() == (1, {"items": [{"id": 0, "text": "legacy"}], "lineage": []})
    assert_delete_mode(path)


class FixedEmbedder:
    model_name = "synthetic-storage-fault-test"

    def embed(self, _text):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_selected_profile_requires_wal_on_startup_and_live_use(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("DML_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    config = {
        "production_profile": PROFILE_ID, "storage_dir": str(tmp_path / "authority"),
        "model_name": "dummy", "llm_backend": "dummy", "embedding_model": None,
        "strict_embedding_required": True,
        "persistence": {"enable": False, "interval_sec": 0, "journal": True, "receipts": True,
                        "receipt_embedding_identity": "synthetic-storage-fault-v1"},
        "rag_store": {"enable": False},
        "dpm": {"enable": False, "mode": "disabled", "include_in_context": False, "include_in_preamble": False},
        "skip_rag_state_import": True, "survival_ledger_enabled": False,
        "background_processing_enabled": False,
    }
    adapter = DMLAdapter(config_path=tmp_path / "absent.yaml", config_overrides=config,
                         embedder=FixedEmbedder(), start_aging_loop=False)
    try:
        receipt = adapter.ingest_memory_receipted("acknowledged", idempotency_key="first", tenant_id="owner")
        path = tmp_path / "authority" / "dml_state.sqlite3"
        set_delete_mode(path)
        before = path.read_bytes()
        with pytest.raises(JournalIntegrityError, match="requires SQLite WAL"):
            adapter.ingest_memory_receipted("acknowledged", idempotency_key="first", tenant_id="owner")
        with pytest.raises(ProductionProfileError, match="requires SQLite WAL"):
            DMLAdapter(config_path=tmp_path / "absent.yaml", config_overrides=config,
                       embedder=FixedEmbedder(), start_aging_loop=False)
        assert path.read_bytes() == before
        assert_delete_mode(path)
        # The valid receipt survives; admission failure does not alter its authority.
        journal = JournalStateStore(path, receipt_mode=True)
        assert journal.lookup_receipt(receipt["scope"], receipt["key"], receipt["request_digest"]) == receipt
    finally:
        adapter.close(persist=False)


@pytest.mark.parametrize("version, admitted", [
    ("3.7.0", False), ("3.43.999", False), ("3.44.5", False),
    ("3.44.6", True), ("3.44.7", True), ("3.45.99", False),
    ("3.49.99", False), ("3.50.6", False), ("3.50.7", True),
    ("3.50.8", True), ("3.51.0", False), ("3.51.2", False),
    ("3.51.3", True), ("3.51.4", True), ("3.52.0", True),
    ("3.53.1", True), ("4.0.0", True), ("3.051.3", False),
    ("3.51.3-custom", False), ("3.51", False), ("3.51.3.1", False),
    (None, False), (True, False), ("", False),
])
def test_wal_reset_version_admission_has_exact_release_boundaries(version, admitted):
    from daystrom_dml.journal import _patched_sqlite_version

    assert _patched_sqlite_version(version) is admitted


def test_runtime_guard_uses_sql_values_instead_of_mutable_module_metadata(monkeypatch):
    from daystrom_dml.journal import require_patched_sqlite

    with closing(sqlite3.connect(":memory:")) as connection:
        expected = connection.execute("SELECT sqlite_version(), sqlite_source_id()").fetchone()
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.7.0")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 7, 0))
    assert require_patched_sqlite() == {
        "sqlite_version": expected[0], "sqlite_source_id": expected[1],
    }


@pytest.mark.parametrize("entrypoint", ["journal", "preflight"])
def test_unpatched_runtime_rejects_before_profile_storage_access(tmp_path, monkeypatch, entrypoint):
    import daystrom_dml.journal as journal_module
    import daystrom_dml.services.profile_runtime as runtime_module

    fault = JournalIntegrityError("synthetic unpatched SQLite runtime")

    def rejected_runtime():
        raise fault

    def forbidden_connection(*_args, **_kwargs):
        pytest.fail("Rejected runtime accessed an authority connection")

    path = tmp_path / "must-remain-absent" / "dml_state.sqlite3"
    monkeypatch.setattr(journal_module, "require_patched_sqlite", rejected_runtime)
    monkeypatch.setattr(runtime_module, "require_patched_sqlite", rejected_runtime)
    monkeypatch.setattr(sqlite3, "connect", forbidden_connection)
    if entrypoint == "journal":
        with pytest.raises(JournalIntegrityError) as observed:
            JournalStateStore(path, receipt_mode=True, require_wal=True)
        assert observed.value is fault
    else:
        with pytest.raises(ProductionProfileError, match="unpatched SQLite"):
            preflight_profile_authority(PROFILE_ID, path.parent, outbox_enabled=False)
    assert not path.parent.exists()


def test_uninspectable_sql_runtime_fails_closed(monkeypatch):
    from daystrom_dml.journal import require_patched_sqlite

    connections = []

    def failed_connection(path):
        connections.append(path)
        raise sqlite3.OperationalError("runtime unavailable")

    monkeypatch.setattr(sqlite3, "connect", failed_connection)
    with pytest.raises(JournalIntegrityError, match="runtime cannot be inspected"):
        require_patched_sqlite()
    assert connections == [":memory:"]
