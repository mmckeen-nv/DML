"""Independent process histories and negative cases for journal ownership.

The parent is the operation oracle; child clients communicate acknowledgements
only after save returns. All databases are isolated under pytest's temporary root.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import JournalIntegrityError, JournalSchemaError, JournalStateStore, RevisionConflict
from scripts.dml_journal import import_snapshot


def memory(text="acknowledged", ident=1):
    return {"items": [{"id": ident, "text": text}], "lineage": [], "next_id": ident + 1}


def full_memory(text="acknowledged"):
    return {"schema_version": 1, "items": [{"schema_version": 1, "id": 1,
        "text": text, "embedding": [1., 1., 1., 1.], "timestamp": 1.,
        "salience": 1., "fidelity": 1., "level": 0, "meta": {}}],
        "lineage": [], "next_id": 2}


def run_child(code, *args):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          capture_output=True, text=True, timeout=40)


@pytest.mark.parametrize("point, outcome", [
    ("init_before_connect", "new"),
    ("init_after_connect", "recovery-required"),
    ("init_after_begin", "recovery-required"),
    ("init_after_schema", "recovery-required"),
    ("init_after_commit", "valid"),
    ("init_before_identity", "valid"),
    ("init_after_identity", "valid"),
])
def test_process_death_during_initialization_has_explicit_outcome(tmp_path, point, outcome):
    path = tmp_path / "fresh.sqlite3"
    child = run_child('''
import os, signal, sys
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
point, filename = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
JournalStateStore(Path(filename), fault_hook=crash)
raise AssertionError('initialization hook was not reached')
''', point, path)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    if outcome == "recovery-required":
        assert path.exists()
        before = path.read_bytes()
        with pytest.raises(JournalSchemaError):
            JournalStateStore(path)
        assert path.read_bytes() == before, "failed startup must preserve incomplete authority"
    else:
        assert path.exists() == (outcome == "valid")
        reopened = JournalStateStore(path)
        assert reopened.read_snapshot() == (0, {"items": [], "lineage": []})
        reopened.save(memory(), expected_revision=0)
        assert JournalStateStore(path).load() == memory()


def test_fresh_multiprocess_clients_have_one_identity_and_complete_acknowledged_history(tmp_path):
    path, barrier = tmp_path / "fresh.sqlite3", tmp_path / "start"
    clients = 8
    code = '''
import json, sys, time
from pathlib import Path
from daystrom_dml.journal import JournalStateStore, RevisionConflict
filename, gate, raw_id = sys.argv[1:]
ident = int(raw_id)
deadline = time.monotonic() + 30
while not Path(gate).exists():
    if time.monotonic() > deadline:
        raise TimeoutError('parent did not release barrier')
    time.sleep(.005)
def delay(point):
    if point == 'after_records':
        time.sleep(.002 * (ident % 3))
store = JournalStateStore(Path(filename), fault_hook=delay)
while time.monotonic() < deadline:
    revision, payload = store.read_snapshot()
    payload['items'].append({'id': ident, 'text': 'client-' + str(ident)})
    try:
        store.save(payload, expected_revision=revision, operation='client-' + str(ident))
        print(json.dumps({'id': ident, 'revision': store.revision,
                          'identity': json.loads(store.identity_path.read_text())}))
        break
    except RevisionConflict:
        continue
else:
    raise TimeoutError('client starved')
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(path), str(barrier), str(i)],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(clients)]
    acknowledgements = []
    try:
        barrier.touch()
        for child in children:
            stdout, stderr = child.communicate(timeout=40)
            assert child.returncode == 0, stderr
            acknowledgements.append(json.loads(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
    assert len({json.dumps(ack["identity"], sort_keys=True) for ack in acknowledgements}) == 1
    assert sorted(ack["revision"] for ack in acknowledgements) == list(range(1, clients + 1))
    reopened = JournalStateStore(path)
    revision, payload = reopened.read_snapshot()
    assert revision == clients
    assert {item["id"]: item["text"] for item in payload["items"]} == {i: f"client-{i}" for i in range(clients)}
    decisions = reopened.decisions(limit=1000)
    assert [(d["revision"], d["operation"]) for d in decisions] == [
        (ack["revision"], f"client-{ack['id']}") for ack in sorted(acknowledgements, key=lambda ack: ack["revision"])]


def dump(path):
    with closing(sqlite3.connect(path)) as connection:
        return list(connection.iterdump())


def replace_json(connection, table, selector, payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(raw.encode()).hexdigest()
    connection.execute(f"UPDATE {table} SET payload=?,checksum=? WHERE {selector}", (raw, checksum))


@pytest.mark.parametrize("damage", [
    "negative-revision", "null-revision", "fractional-revision", "text-revision", "reset-revision",
    "reset-and-delete-records", "history-gap", "old-decision-corrupt", "decision-list",
    "decision-wrong-schema", "decision-wrong-revision", "snapshot-corrupt", "snapshot-forged",
    "snapshot-future", "snapshot-missing",
])
def test_structural_corruption_cannot_be_concealed_or_overwritten(tmp_path, damage):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    store.save(memory(), operation="first")
    store.save(memory("second"), operation="second")
    with closing(sqlite3.connect(store.path)) as connection, connection:
        revisions = {"negative-revision": -1, "null-revision": None, "fractional-revision": 1.5,
                     "text-revision": "invalid", "reset-revision": 0, "reset-and-delete-records": 0}
        if damage in revisions:
            connection.execute("UPDATE state SET revision=?", (revisions[damage],))
            if damage == "reset-and-delete-records":
                connection.execute("DELETE FROM records")
        elif damage == "history-gap":
            connection.execute("DELETE FROM decisions WHERE revision=1")
        elif damage == "old-decision-corrupt":
            connection.execute("UPDATE decisions SET checksum='wrong' WHERE revision=1")
        elif damage.startswith("decision-"):
            event = json.loads(connection.execute("SELECT payload FROM decisions WHERE revision=2").fetchone()[0])
            if damage == "decision-list":
                event = []
            elif damage == "decision-wrong-schema":
                event["schema_version"] = 99
            else:
                event["revision"] = 1
            replace_json(connection, "decisions", "revision=2", event)
        elif damage == "snapshot-corrupt":
            connection.execute("UPDATE snapshot SET checksum='wrong'")
        elif damage == "snapshot-forged":
            replace_json(connection, "snapshot", "id=1", memory("invented"))
        elif damage == "snapshot-future":
            connection.execute("UPDATE snapshot SET revision=999")
        else:
            connection.execute("DELETE FROM snapshot")
    before = dump(store.path)
    for action in (store.load, store.stamp, store.decisions, store.checkpoint,
                   lambda: store.save(memory("must never commit")), lambda: JournalStateStore(store.path)):
        with pytest.raises(JournalIntegrityError):
            action()
    assert dump(store.path) == before


@pytest.mark.parametrize("kwargs", [
    {"after_revision": -1}, {"after_revision": True}, {"after_revision": 1.5},
    {"limit": True}, {"limit": 1.5}, {"limit": 0}, {"limit": 1001},
])
def test_invalid_pagination_is_rejected_before_query(tmp_path, kwargs):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    with pytest.raises(ValueError):
        store.decisions(**kwargs)


def test_corrupt_existing_v1_without_marker_is_not_adopted(tmp_path):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    store.save(memory())
    store.identity_path.unlink()
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DELETE FROM records")
    with pytest.raises(JournalIntegrityError):
        JournalStateStore(store.path)
    assert not store.identity_path.exists()


def test_commit_pinned_schema1_remains_readable_and_writable(tmp_path):
    path = tmp_path / "legacy-v1.sqlite3"
    fixture = Path(__file__).parent / "fixtures/journal_schema1_b8626b7.sql"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(fixture.read_text())
    store = JournalStateStore(path)
    assert store.read_snapshot() == (2, memory("updated schema-1 evidence"))
    assert store.identity_path.exists()
    assert [event["operation"] for event in store.decisions()] == ["legacy-first", "legacy-second"]
    store.save(memory("current writer"), expected_revision=2)
    assert JournalStateStore(path).read_snapshot() == (3, memory("current writer"))


class FixedEmbedder:
    def embed(self, text):
        return np.ones(4, dtype=np.float32)


def test_adapter_startup_pins_loaded_revision_when_peer_commits_after_load(tmp_path, monkeypatch):
    store = JournalStateStore(tmp_path / "dml_state.sqlite3")
    store.save(full_memory())
    original = DMLAdapter._load_persisted_state
    def interleaved_load(instance):
        original(instance)
        store.save(full_memory("peer committed during startup"), expected_revision=1)
    monkeypatch.setattr(DMLAdapter, "_load_persisted_state", interleaved_load)
    instance = DMLAdapter(config_overrides={"storage_dir": str(tmp_path), "model_name": "dummy",
        "embedding_model": None, "persistence": {"journal": True, "enable": False},
        "rag_store": {"enable": False}, "dpm": {"enable": False, "include_in_context": False}},
        embedder=FixedEmbedder(), start_aging_loop=False)
    try:
        assert instance._last_observed_state[0] == 1
        instance.ingest_memory("later adapter write", tenant_id="owner", meta={"no_merge": True})
        assert {item["text"] for item in store.load()["items"]} == {
            "peer committed during startup", "later adapter write"}
    finally:
        instance.close(persist=False)


def test_snapshot_import_refuses_peer_commit_after_empty_check(tmp_path, monkeypatch):
    source, path = tmp_path / "incoming.json", tmp_path / "journal.sqlite3"
    source.write_text(json.dumps(full_memory("incoming")))
    original = JournalStateStore.stamp
    raced = False
    def interleaved_stamp(instance):
        nonlocal raced
        value = original(instance)
        if value[0] == 0 and not raced:
            raced = True
            peer = JournalStateStore(path)
            peer.save(full_memory("peer acknowledged"), expected_revision=0)
        return value
    monkeypatch.setattr(JournalStateStore, "stamp", interleaved_stamp)
    with pytest.raises(RevisionConflict):
        import_snapshot(source, path)
    assert JournalStateStore(path).load() == full_memory("peer acknowledged")


def test_existing_database_deleted_before_open_is_never_recreated(tmp_path, monkeypatch):
    path = tmp_path / "journal.sqlite3"
    JournalStateStore(path).save(memory())
    original = sqlite3.connect
    deleted = False
    def disappearing_connect(*args, **kwargs):
        nonlocal deleted
        if not deleted:
            deleted = True
            path.unlink()
        return original(*args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", disappearing_connect)
    with pytest.raises(JournalIntegrityError):
        JournalStateStore(path)
    assert not path.exists()


@pytest.mark.parametrize("name", ["journal.sqlite3", "journal.sqlite3-wal", "journal.sqlite3-shm",
    "journal.sqlite3.identity.json", "journal.sqlite3.init.lock", ".dml_store.lock", ".dml_store.lock.json"])
def test_export_cannot_replace_storage_or_coordination_files(tmp_path, name):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    store.save(memory())
    target = tmp_path / name
    before = target.read_bytes() if target.exists() else None
    with pytest.raises(ValueError):
        store.export_snapshot(target)
    assert (target.read_bytes() if target.exists() else None) == before


def test_save_freezes_caller_payload_before_any_fault_hook(tmp_path):
    payload = memory("original caller input")
    def mutate_after_begin(point):
        if point == "after_begin":
            payload["items"][0]["text"] = "mutated during transaction"
            payload["next_id"] = 999
    store = JournalStateStore(tmp_path / "journal.sqlite3", snapshot_interval=1, fault_hook=mutate_after_begin)
    store.save(payload, expected_revision=0)
    assert payload["next_id"] == 999, "the fault hook must exercise actual caller mutation"
    assert JournalStateStore(store.path).load() == memory("original caller input")


@pytest.mark.parametrize("raw", ['{"items":[],"items":[]}', '{"temperature":1e999}', '{"temperature":NaN}'])
def test_checksummed_metadata_with_ambiguous_or_nonfinite_json_is_rejected(tmp_path, raw):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    store.save(memory())
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("UPDATE state SET metadata=?,checksum=?", (raw, hashlib.sha256(raw.encode()).hexdigest()))
    with pytest.raises(JournalIntegrityError):
        store.load()


@pytest.mark.parametrize("sql", ["CREATE TABLE sqliteXextra (id INTEGER)",
    "CREATE TABLE extra (id INTEGER)", "ALTER TABLE records ADD COLUMN unknown TEXT"])
def test_unexpected_schema_objects_and_layout_cannot_hide_from_validation(tmp_path, sql):
    store = JournalStateStore(tmp_path / "journal.sqlite3")
    store.save(memory())
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute(sql)
    before = dump(store.path)
    for action in (store.load, lambda: store.save(memory("rejected")), lambda: JournalStateStore(store.path)):
        with pytest.raises(JournalSchemaError):
            action()
    assert dump(store.path) == before


def test_history_cost_harness_runs_and_reports_measured_revisions():
    from scripts.journal_history_benchmark import run_history_benchmark
    report = run_history_benchmark(milestones=(1, 3), samples=2)
    assert report["schema_version"] == "dml-journal-history-cost-v1"
    assert [row["revision"] for row in report["measurements"]] == [1, 3]
    assert report["sqlite"] == sqlite3.sqlite_version
    for row in report["measurements"]:
        assert row["live_records"] == 10
        assert set(row["operations"]) == {"read", "stamp", "no_change_save", "reopen"}
        assert all(len(value["samples_ms"]) == 2 and value["max_ms"] >= value["p50_ms"] >= 0
                   for value in row["operations"].values())
