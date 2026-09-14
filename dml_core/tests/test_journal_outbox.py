"""Transactional outbox authority, corruption and crash regression oracles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from daystrom_dml.journal import (
    OUTBOX_FAULT_POINTS, JournalIntegrityError, JournalSchemaError,
    JournalStateStore, RevisionConflict,
)
from daystrom_dml.services.journal_outbox import validate_outbox_event

SCOPE = dict(tenant_id="tenant", client_id=None, session_id=None, instance_id=None)
DIGEST = hashlib.sha256(b"request").hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def memory(ident=0, text="remember"):
    return {"schema_version": 1, "id": ident, "text": text, "meta": dict(SCOPE)}


def create(tmp_path):
    return JournalStateStore(tmp_path / "authority.sqlite", receipt_mode=True, outbox_mode=True)


def commit(store, key="request", ident=0):
    revision, state = store.read_snapshot()
    item = memory(ident)
    state["items"].append(item)
    return store.save_with_receipt(state, scope=SCOPE, key=key, request_digest=DIGEST,
                                   result={"memory": item}, expected_revision=revision)


def test_schema3_opt_in_and_existing_version_guards(tmp_path):
    for version in (1, 2):
        path = tmp_path / f"old{version}.sqlite"
        old = JournalStateStore(path, receipt_mode=version == 2)
        before = old.verified_snapshot()
        with pytest.raises(JournalSchemaError, match="future explicit migration"):
            JournalStateStore(path, receipt_mode=True, outbox_mode=True)
        assert JournalStateStore(path).schema_version == version
        assert old.verified_snapshot() == before
        with pytest.raises(JournalSchemaError):
            old.outbox_events()
    store = create(tmp_path)
    assert store.schema_version == 3
    assert JournalStateStore(store.path).schema_version == 3
    assert JournalStateStore(store.path, receipt_mode=True).schema_version == 3
    with pytest.raises(JournalSchemaError):
        store.export_snapshot(tmp_path / "export.json")


@pytest.mark.parametrize("receipt,outbox", [(False, True), (True, 1), (True, None), (True, "true")])
def test_invalid_constructor_flags_do_not_create_authority(tmp_path, receipt, outbox):
    with pytest.raises(ValueError):
        JournalStateStore(tmp_path / "absent.sqlite", receipt_mode=receipt, outbox_mode=outbox)
    assert not (tmp_path / "absent.sqlite").exists()


def test_every_committed_transition_retained_without_duplicate_noop_events(tmp_path):
    store = create(tmp_path)
    assert store.outbox_events()["events"] == []
    receipt = commit(store)
    original = store.outbox_events()["events"][0]
    assert original["state"]["items"] == [memory()]
    assert original["receipt"]["digest"] == digest(receipt)
    # Retry ignores stale proposal and does not allocate another event.
    assert store.save_with_receipt({}, scope=SCOPE, key="request", request_digest=DIGEST,
        result={"memory": memory()}, expected_revision=0) == receipt
    state = store.load()
    store.save(state, expected_revision=1)
    assert store.outbox_events()["head_revision"] == 1
    state["items"][0]["text"] = "changed"
    store.save(state, expected_revision=1, operation="update")
    state["lineage"] = state.pop("items")
    state["items"] = []
    store.save(state, expected_revision=2, operation="promote")
    state["lineage"] = []
    store.save(state, expected_revision=3, operation="delete")
    page = store.outbox_events(limit=2)
    assert page["head_revision"] == 4 and page["next_revision"] == 2 and page["has_more"]
    assert page["prefix_digest"] == digest([])
    assert page["events"][0] == original
    assert page["events"][1]["operation"] == "update"
    remainder = store.outbox_events(after_revision=2)
    assert [event["source_revision"] for event in remainder["events"]] == [3, 4]
    assert remainder["events"][-1]["state"] == {"items": [], "lineage": []}
    assert remainder["prefix_digest"] == digest([event["checksum"] for event in page["events"]])
    assert not remainder["has_more"]
    assert store.lookup_receipt(SCOPE, "request", DIGEST) == receipt
    # Returned objects are detached from both storage and other page reads.
    original["state"]["items"].clear()
    assert store.outbox_events()["events"][0]["state"]["items"] == [memory()]


@pytest.mark.parametrize("after,limit", [(True, 1), (-1, 1), (0.0, 1), (0, True), (0, 0), (0, 1001), (0, 1.0), (1, 1)])
def test_page_rejects_invalid_or_future_cursor(tmp_path, after, limit):
    with pytest.raises(ValueError):
        create(tmp_path).outbox_events(after_revision=after, limit=limit)


def test_metadata_types_and_order_are_preserved_exactly(tmp_path):
    store = create(tmp_path)
    store.save({"items": [memory(), memory(1)], "lineage": [], "flag": True}, expected_revision=0)
    for revision, value in enumerate([1, 1.0], 1):
        state = store.load()
        state["flag"] = value
        state["items"].reverse()
        store.save(state, expected_revision=revision)
    events = store.outbox_events()["events"]
    assert [type(event["state"]["flag"]) for event in events] == [bool, int, float]
    assert [[item["id"] for item in event["state"]["items"]] for event in events] == [[0, 1], [1, 0], [0, 1]]
    assert len({event["source_digest"] for event in events}) == 3


@pytest.mark.parametrize("damage", ["missing", "extra", "checksum", "unknown_format", "foreign", "operation", "receipt", "state", "decision", "historical_state"])
def test_corrupt_outbox_never_reads_or_extends_authority(tmp_path, damage):
    store = create(tmp_path)
    commit(store)
    commit(store, "second", 1)
    with closing(sqlite3.connect(store.path)) as db, db:
        raw = db.execute("SELECT payload FROM outbox WHERE revision=1").fetchone()[0]
        event = json.loads(raw)
        if damage == "missing":
            db.execute("DELETE FROM outbox WHERE revision=1")
        elif damage == "extra":
            db.execute("INSERT INTO outbox VALUES (3,?,?)", (raw, hashlib.sha256(raw.encode()).hexdigest()))
        elif damage == "checksum":
            db.execute("UPDATE outbox SET checksum=? WHERE revision=1", ("0" * 64,))
        else:
            if damage == "unknown_format":
                event["event_format"] = "future"
            elif damage == "foreign":
                event["source_store_id"] = "a" * 32
            elif damage == "operation":
                event["operation"] = "invented"
            elif damage == "receipt":
                event["receipt"] = None
            elif damage == "state":
                event["state"]["items"][0]["text"] = "invented"
            elif damage == "decision":
                event["decision_digest"] = "0" * 64
            elif damage == "historical_state":
                event["state"]["items"][0]["text"] = "invented"
                event["source_digest"] = digest(event["state"])
            event["checksum"] = digest({key: value for key, value in event.items() if key != "checksum"})
            if damage == "historical_state":
                decision = json.loads(db.execute("SELECT payload FROM decisions WHERE revision=1").fetchone()[0])
                decision["state_digest"] = event["source_digest"]
                event["decision_digest"] = digest({key: value for key, value in decision.items() if key != "outbox_digest"})
                event["checksum"] = digest({key: value for key, value in event.items() if key != "checksum"})
                decision["outbox_digest"] = event["checksum"]
                db.execute("UPDATE decisions SET payload=?,checksum=? WHERE revision=1", (encode(decision), digest(decision)))
                # Snapshot normally anchors revision1; move to genuine current
                # revision2 so rejection must come from historical event replay.
                current = json.loads(db.execute("SELECT payload FROM outbox WHERE revision=2").fetchone()[0])["state"]
                db.execute("UPDATE snapshot SET revision=2,payload=?,checksum=?", (encode(current), digest(current)))
            db.execute("UPDATE outbox SET payload=?,checksum=? WHERE revision=1", (encode(event), digest(event)))
    for action in (store.outbox_events, store.load, lambda: commit(store, "third", 2), lambda: JournalStateStore(store.path)):
        with pytest.raises(JournalIntegrityError):
            action()


@pytest.mark.parametrize("point", OUTBOX_FAULT_POINTS)
def test_process_death_keeps_receipt_state_and_event_atomic(tmp_path, point):
    store = create(tmp_path)
    acknowledged = commit(store)
    program = '''
import json, os, signal, sys, threading
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
point, path = sys.argv[1:3]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
            threading.Event().wait(5)
            os._exit(80)
        os._exit(79)
store = JournalStateStore(Path(path), fault_hook=crash)
revision, state = store.read_snapshot()
item, scope, request_digest = json.loads(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5]
state['items'].append(item)
store.save_with_receipt(state, scope=scope, key='interrupted', request_digest=request_digest,
                       result={'memory':item}, expected_revision=revision)
raise AssertionError('hook not reached')
'''
    result = subprocess.run([sys.executable, "-c", program, point, str(store.path), encode(memory(1)), encode(SCOPE), DIGEST],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    reopened = JournalStateStore(store.path)
    page = reopened.outbox_events()
    expected_head = 2 if point == "after_commit" else 1
    assert page["head_revision"] == expected_head
    assert len(page["events"]) == expected_head
    assert reopened.lookup_receipt(SCOPE, "request", DIGEST) == acknowledged
    assert bool(reopened.lookup_receipt(SCOPE, "interrupted", DIGEST)) == (expected_head == 2)
    assert len(reopened.load()["items"]) == expected_head
    if expected_head == 1:
        commit(reopened, "interrupted", 1)
    assert len(reopened.outbox_events()["events"]) == 2


def test_read_page_head_and_events_share_snapshot_during_writer_commit(tmp_path):
    store = create(tmp_path)
    commit(store)
    entered, release = threading.Event(), threading.Event()
    original = store._read_snapshot
    def paused(connection):
        result = original(connection)
        entered.set()
        assert release.wait(10)
        return result
    store._read_snapshot = paused
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.outbox_events)
        assert entered.wait(10)
        try:
            commit(JournalStateStore(store.path), "second", 1)
        finally:
            release.set()
        page = pending.result(timeout=10)
    assert page["head_revision"] == 1
    assert [event["source_revision"] for event in page["events"]] == [1]
    store._read_snapshot = original
    assert store.outbox_events()["head_revision"] == 2


def test_competing_cas_writers_publish_exactly_one_outbox_event(tmp_path):
    store = create(tmp_path)
    commit(store)
    barrier = threading.Barrier(16)
    def writer(index):
        local = JournalStateStore(store.path)
        revision, state = local.read_snapshot()
        state["items"][0]["text"] = str(index)
        barrier.wait(timeout=15)
        try:
            local.save(state, expected_revision=revision)
            return True
        except RevisionConflict:
            return False
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(writer, range(16)))
    assert sum(outcomes) == 1
    assert store.outbox_events()["head_revision"] == 2
    assert len(store.outbox_events()["events"]) == 2


@pytest.mark.parametrize("field,value", [("schema_version", True), ("source_revision", True), ("source_revision", 0), ("source_revision", 1.0), ("operation", ""), ("checksum", "BAD")])
def test_wire_event_rejects_invalid_types_even_with_recomputed_hash(tmp_path, field, value):
    store = create(tmp_path)
    commit(store)
    event = store.outbox_events()["events"][0]
    event[field] = value
    if field != "checksum":
        event["checksum"] = digest({key: value for key, value in event.items() if key != "checksum"})
    with pytest.raises(JournalIntegrityError):
        validate_outbox_event(event)


def test_event_serialization_failure_rolls_back_memory_receipt_and_decision(tmp_path):
    store = create(tmp_path)
    commit(store)
    before = store.outbox_events()
    original_encode = store._encode
    def interrupted(value):
        if isinstance(value, dict) and "event_format" in value:
            raise OSError(28, "injected disk full during event serialization")
        return original_encode(value)
    store._encode = interrupted
    with pytest.raises(OSError):
        commit(store, "interrupted", 1)
    store._encode = original_encode
    assert store.outbox_events() == before
    assert store.lookup_receipt(SCOPE, "interrupted", DIGEST) is None
    assert store.load()["items"] == [memory()]
    assert len(store.decisions()) == 1


@pytest.mark.parametrize("point", ["after_outbox", "before_commit", "after_commit"])
def test_disk_full_error_preserves_unambiguous_committed_event_state(tmp_path, point):
    store = create(tmp_path)
    commit(store)
    def fail(here):
        if here == point:
            raise OSError(28, "injected disk full")
    store._fault_hook = fail
    with pytest.raises(OSError):
        commit(store, "interrupted", 1)
    reopened = JournalStateStore(store.path)
    committed = point == "after_commit"
    assert reopened.outbox_events()["head_revision"] == (2 if committed else 1)
    assert bool(reopened.lookup_receipt(SCOPE, "interrupted", DIGEST)) == committed
    assert len(reopened.load()["items"]) == (2 if committed else 1)


def test_schema3_write_checks_identity_inside_transaction(tmp_path):
    store = create(tmp_path)
    # Simulate an identity change after connection preflight and before BEGIN.
    # The pinned write transaction must reject even an otherwise empty journal.
    original_read = store._read_snapshot
    def replaced_identity(connection):
        connection.execute("UPDATE identity SET store_id=?", ("a" * 32,))
        return original_read(connection)
    store._read_snapshot = replaced_identity
    with pytest.raises(JournalIntegrityError, match="transaction identity changed"):
        store.save({"items": [memory()], "lineage": []}, expected_revision=0)
    store._read_snapshot = original_read
    assert store.outbox_events()["head_revision"] == 0
