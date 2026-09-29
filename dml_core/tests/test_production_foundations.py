"""Crash/recovery and schema invariants, using real subprocesses and SQLite."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from daystrom_dml.checkpoint import CheckpointManager
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import FAULT_POINTS, JournalStateStore, JournalIntegrityError, JournalSchemaError, RevisionConflict
from daystrom_dml.persistence import load_state, save_state, PersistenceFormatError
from daystrom_dml.memory_store import MemoryItem
from scripts.dml_journal import upgrade_legacy


def state(text="acknowledged"):
    return {"items": [{"id": 1, "text": text}], "lineage": [], "next_id": 2}


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_abrupt_process_death_at_every_journal_boundary(tmp_path, point):
    path = tmp_path / "state.sqlite3"
    store = JournalStateStore(path, snapshot_interval=1)
    store.save(state())
    code = '''
import os, signal, sys
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
point, path = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
store = JournalStateStore(Path(path), snapshot_interval=1, fault_hook=crash)
store.save({'items':[{'id':1,'text':'uncertain'}],'lineage':[],'next_id':2}, expected_revision=1, operation='crash-test')
'''
    child = subprocess.run([sys.executable, "-c", code, point, str(path)], timeout=20)
    assert child.returncode != 0
    recovered = JournalStateStore(path)
    expected = state("uncertain") if point == "after_commit" else state()
    assert recovered.load() == expected
    assert recovered.stamp()[0] == (2 if point == "after_commit" else 1)
    assert len(recovered.decisions()) == recovered.stamp()[0]
    # A new committed write proves abandoned locks and partial rows are gone.
    recovered.save(state("recovered"), expected_revision=recovered.stamp()[0])
    assert recovered.load() == state("recovered")


@pytest.mark.parametrize("damage", ["payload", "deleted_record", "event", "metadata", "identity"])
def test_journal_corruption_is_explicit(tmp_path, damage):
    store = JournalStateStore(tmp_path / "state.sqlite3")
    store.save(state())
    statements = {
        "payload": "UPDATE records SET payload='{}'",
        "deleted_record": "DELETE FROM records",
        "event": "UPDATE decisions SET payload='{}'",
        "metadata": "UPDATE state SET metadata='{}'",
        "identity": "UPDATE identity SET store_id='wrong'",
    }
    with sqlite3.connect(store.path) as connection:
        connection.execute(statements[damage])
    with pytest.raises(JournalIntegrityError):
        store.load()


def test_missing_initialized_database_cannot_be_recreated(tmp_path):
    store = JournalStateStore(tmp_path / "state.sqlite3")
    store.save(state())
    store.path.unlink()
    with pytest.raises((sqlite3.OperationalError, JournalIntegrityError)):
        store.load()
    with pytest.raises(JournalIntegrityError, match="missing"):
        JournalStateStore(store.path)
    assert not store.path.exists()


def test_disk_quota_failure_rolls_back_state_and_audit(tmp_path, monkeypatch):
    store = JournalStateStore(tmp_path / "state.sqlite3")
    store.save(state())
    connect = store._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(store, "_connect", limited)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.save(state("x" * 1_000_000))
    reopened = JournalStateStore(store.path)
    assert reopened.load() == state()
    assert len(reopened.decisions()) == 1


def test_serialization_failure_does_not_commit(tmp_path):
    store = JournalStateStore(tmp_path / "state.sqlite3")
    store.save(state())
    invalid = state()
    invalid["items"][0]["embedding"] = [float("nan")]
    with pytest.raises(ValueError):
        store.save(invalid)
    assert store.load() == state()
    assert store.stamp()[0] == 1


def test_stale_writer_cannot_overwrite_related_changes(tmp_path):
    first = JournalStateStore(tmp_path / "state.sqlite3")
    first.save(state())
    second = JournalStateStore(first.path)
    stale = second.load()
    revision = second.stamp()[0]
    first.save(state("new evidence"), expected_revision=revision)
    with pytest.raises(RevisionConflict):
        second.save(stale, expected_revision=revision)
    assert first.load() == state("new evidence")


def test_concurrent_clients_use_compare_and_swap_without_lost_updates(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = JournalStateStore(path)
    first.save({"items": [], "lineage": []})
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def run(index):
        store = JournalStateStore(path)
        barrier.wait(timeout=30)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            payload = store.load()
            revision = store._revision
            payload["items"].append({"id": index, "text": f"client {index}"})
            time.sleep((index % 3) * .0001)
            try:
                store.save(payload, expected_revision=revision, operation="concurrent-client")
                return
            except RevisionConflict:
                continue
        raise AssertionError("writer starved")
    with ThreadPoolExecutor(max_workers=clients) as executor:
        list(executor.map(run, range(clients)))
    assert {r["id"] for r in first.load()["items"]} == set(range(clients))
    assert first.stamp()[0] == clients + 1
    assert len(first.decisions(limit=1000)) == clients + 1


def test_explicit_legacy_upgrade_preserves_source_and_refuses_downgrade(tmp_path):
    source, target = tmp_path / "old.sqlite3", tmp_path / "new.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.executescript((Path(__file__).parent / "fixtures/journal_schema0.sql").read_text())
    original = source.read_bytes()
    with pytest.raises(JournalSchemaError):
        JournalStateStore(source)
    upgrade_legacy(source, target)
    upgraded = JournalStateStore(target)
    assert upgraded.load()["items"][0]["text"] == "legacy evidence"
    assert upgraded.load()["items"][0]["meta"] == {"source": "fixture"}
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="new, separate"):
        upgrade_legacy(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(JournalSchemaError):
        JournalStateStore(target)


def write_jsonl(path, header_change=None, record_change=None):
    item = MemoryItem(id=1, text="evidence", embedding=np.ones(2), timestamp=1., salience=.5, fidelity=1., level=0)
    save_state([item], path)
    lines = path.read_text().splitlines()
    header, record = json.loads(lines[0]), json.loads(lines[1])
    if record_change:
        record_change(record)
        lines[1] = json.dumps(record)
        header["checksum"] = hashlib.sha256(lines[1].encode()).hexdigest()
    if header_change:
        header_change(header)
    path.write_text(json.dumps(header) + "\n" + lines[1])


@pytest.mark.parametrize("change", [lambda h: h.update(version=99), lambda h: h.pop("version"), lambda h: h.update(version=True), lambda h: h.pop("checksum"), lambda h: h.update(count=True)])
def test_jsonl_rejects_invalid_header_even_with_valid_payload(tmp_path, change):
    path = tmp_path / "state.jsonl"
    write_jsonl(path, header_change=change)
    with pytest.raises(PersistenceFormatError):
        load_state(path)


@pytest.mark.parametrize("change", [lambda r: r.update(schema_version=99), lambda r: r.update(id="1"), lambda r: r.update(embedding=[float("nan")]), lambda r: r.pop("text"), lambda r: r.update(embedding=[1e100])])
def test_jsonl_rejects_corrupt_records_even_with_recomputed_checksum(tmp_path, change):
    path = tmp_path / "state.jsonl"
    write_jsonl(path, record_change=change)
    with pytest.raises(PersistenceFormatError):
        load_state(path)


@pytest.mark.parametrize("filename", ["dml_state.jsonl", "dml_store.json"])
def test_adapter_startup_never_falls_back_after_corruption(tmp_path, filename):
    (tmp_path / filename).write_text("corrupt")
    before = (tmp_path / filename).read_bytes()
    with pytest.raises((ValueError, PersistenceFormatError)):
        DMLAdapter(config_overrides={"storage_dir": str(tmp_path), "model_name": "dummy", "embedding_model": None,
                                    "persistence": {"enable": filename.endswith("jsonl"), "path": str(tmp_path / filename)}}, start_aging_loop=False)
    assert (tmp_path / filename).read_bytes() == before


def test_checkpoint_collision_and_failure_diagnostics(tmp_path, monkeypatch):
    import daystrom_dml.checkpoint as module
    monkeypatch.setattr(module.time, "time_ns", lambda: 123)
    manager = CheckpointManager(tmp_path, lambda: {"generation": 1}, retention=3)
    first, second = manager.checkpoint(), manager.checkpoint()
    assert first != second and first.exists() and second.exists()
    def fail(*args, **kwargs):
        raise OSError("full")
    monkeypatch.setattr(module, "atomic_write_text", fail)
    with pytest.raises(OSError):
        manager.checkpoint()
    assert manager.status()["status"] == "degraded"
    assert manager.status()["last_checkpoint"] == second.name


def test_corruption_cannot_be_overwritten_by_a_write(tmp_path):
    store = JournalStateStore(tmp_path / "state.sqlite3")
    store.save(state())
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM records")
    with pytest.raises(JournalIntegrityError):
        store.save(state("replacement must not conceal loss"))
    with pytest.raises(JournalIntegrityError):
        store.stamp()
    # Inspect the damaged authority directly; normal readers must reject it.
    from contextlib import closing
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT revision FROM state").fetchone()[0] == 1


def test_provider_reports_degraded_durability_and_explicit_maturity(tmp_path):
    from fastapi.testclient import TestClient
    from daystrom_dml.provider_server import create_app
    instance = DMLAdapter(config_overrides={"storage_dir": str(tmp_path), "model_name": "dummy", "embedding_model": None,
                                          "persistence": {"enable": False}, "rag_store": {"enable": False}}, start_aging_loop=False)
    with TestClient(create_app(adapter_factory=lambda: instance)) as client:
        with instance._persist_lock:
            instance._record_durability_failure_locked("dml", OSError("disk unavailable"))
        response = client.get("/health").json()
        assert response["status"] == "degraded"
        assert response["durability"]["status"] == "degraded"
        contract = client.get("/api/contracts").json()
        assert contract["production_ready"] is False
        assert "native-kv-reuse" in contract["experimental"]
