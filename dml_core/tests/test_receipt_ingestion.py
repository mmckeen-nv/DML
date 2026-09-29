"""Parent-oracle receipt and migration failures; no child acknowledgement inferred."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from daystrom_dml.journal import (
    IdempotencyConflict, JournalIntegrityError, JournalSchemaError, JournalStateStore,
    RECEIPT_FAULT_POINTS, RECEIPT_MIGRATION_FAULT_POINTS, upgrade_receipt_journal,
)

SCOPE = dict(tenant_id="tenant", client_id=None, session_id=None, instance_id=None)
DIGEST = hashlib.sha256(b"parent request").hexdigest()


def record(ident=0, text="parent request"):
    return dict(schema_version=1, id=ident, text=text, embedding=[1., 0., 0., 0.],
                timestamp=1., salience=1., fidelity=1., level=0, meta=dict(SCOPE))


def commit(store, key="key", *, ident=0, digest=DIGEST):
    revision, payload = store.read_snapshot()
    item = record(ident)
    payload["items"].append(item)
    payload["next_id"] = ident + 1
    return store.save_with_receipt(payload, scope=SCOPE, key=key, request_digest=digest,
                                   result={"memory": item}, expected_revision=revision)


def child(code, *args):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          capture_output=True, text=True, timeout=40)


CRASH_SETUP = '''
import os, signal, sys, json
from pathlib import Path
from daystrom_dml.journal import JournalStateStore, upgrade_receipt_journal
point, filename = sys.argv[1:3]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(79)
'''


@pytest.mark.parametrize("point", RECEIPT_FAULT_POINTS)
def test_killed_receipt_transaction_preserves_parent_acknowledgements(tmp_path, point):
    path = tmp_path / "journal.sqlite3"
    journal = JournalStateStore(path, receipt_mode=True, snapshot_interval=1)
    acknowledged = commit(journal, "acknowledged")
    proposed = record(1)
    result = child(CRASH_SETUP + '''
journal = JournalStateStore(Path(filename), fault_hook=crash, snapshot_interval=1)
revision, payload = journal.read_snapshot()
item, scope, digest = json.loads(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5]
payload['items'].append(item)
payload['next_id'] = 2
journal.save_with_receipt(payload, scope=scope, key='interrupted', request_digest=digest,
                         result={'memory':item}, expected_revision=revision)
raise AssertionError('mutation hook not reached')
''', point, path, json.dumps(proposed), json.dumps(SCOPE), DIGEST)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    reopened = JournalStateStore(path)
    assert reopened.lookup_receipt(SCOPE, "acknowledged", DIGEST) == acknowledged
    interrupted = reopened.lookup_receipt(SCOPE, "interrupted", DIGEST)
    assert (interrupted is not None) == (point == "after_commit")
    assert reopened.read_snapshot()[0] == (2 if interrupted else 1)
    if interrupted is None:
        interrupted = commit(reopened, "interrupted", ident=1)
    # Retry a stale proposal, independently of the current snapshot, must replay.
    assert reopened.save_with_receipt({"items": [proposed], "lineage": []}, scope=SCOPE,
        key="interrupted", request_digest=DIGEST, result={"memory": proposed},
        expected_revision=0) == interrupted
    assert reopened.read_snapshot()[0] == 2
    assert reopened.load()["items"] == [record(), proposed]


@pytest.mark.parametrize("point", RECEIPT_MIGRATION_FAULT_POINTS)
def test_killed_migration_never_publishes_partial_authority(tmp_path, point):
    source, target = tmp_path / "source.sqlite3", tmp_path / "target.sqlite3"
    original = JournalStateStore(source)
    original.save({"items": [record()], "lineage": [], "next_id": 1}, expected_revision=0)
    original.save({"items": [], "lineage": [record()], "next_id": 1}, expected_revision=1)
    before, decisions = original.read_snapshot(), original.decisions()
    identity = original.identity_path.read_bytes()
    result = child(CRASH_SETUP + '''
upgrade_receipt_journal(Path(filename), Path(sys.argv[3]), fault_hook=crash)
raise AssertionError('migration hook not reached')
''', point, source, target)
    assert result.returncode == (-9 if os.name == "posix" else 79), result.stderr
    assert JournalStateStore(source).read_snapshot() == before
    assert original.decisions() == decisions
    assert original.identity_path.read_bytes() == identity
    if point == "migration_after_publish":
        upgraded = JournalStateStore(target)
        assert upgraded.schema_version == 2 and upgraded.read_snapshot() == before
        assert upgraded.identity_path.read_bytes() == identity
        assert upgraded.decisions() == [{**event, "schema_version": 2, "receipt": None} for event in decisions]
    else:
        with pytest.raises(JournalIntegrityError):
            JournalStateStore(target)
        with pytest.raises(ValueError):
            upgrade_receipt_journal(source, target)
        # Recovery starts from the untouched source into another explicit target.
        recovered = tmp_path / "recovered.sqlite3"
        upgrade_receipt_journal(source, recovered)
        assert JournalStateStore(recovered).read_snapshot() == before


def test_schema_one_never_silently_upgrades_and_schema_two_cannot_export_without_ledger(tmp_path):
    source = tmp_path / "old.sqlite3"
    journal = JournalStateStore(source)
    journal.save({"items": [record()], "lineage": []}, expected_revision=0)
    with pytest.raises(JournalSchemaError):
        JournalStateStore(source, receipt_mode=True)
    assert JournalStateStore(source).schema_version == 1
    target = tmp_path / "new.sqlite3"
    upgrade_receipt_journal(source, target)
    upgraded = JournalStateStore(target)
    with pytest.raises(JournalSchemaError):
        upgraded.export_snapshot(tmp_path / "lossy.json")
    assert not (tmp_path / "lossy.json").exists()


def test_same_key_thread_clients_commit_one_receipt(tmp_path):
    path = tmp_path / "journal.sqlite3"
    journal = JournalStateStore(path, receipt_mode=True)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def submit(_):
        # All clients observed the same empty revision before the barrier.
        payload = {"items": [record()], "lineage": [], "next_id": 1}
        barrier.wait(timeout=30)
        return journal.save_with_receipt(payload, scope=SCOPE, key="shared", request_digest=DIGEST,
                                         result={"memory": record()}, expected_revision=0)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        receipts = list(pool.map(submit, range(clients)))
    assert receipts == [receipts[0]] * clients
    assert journal.read_snapshot()[0] == 1 and journal.load()["items"] == [record()]
    assert len(journal.decisions()) == 1


def test_same_key_process_clients_return_one_historical_receipt(tmp_path):
    path, gate = tmp_path / "journal.sqlite3", tmp_path / "start"
    journal = JournalStateStore(path, receipt_mode=True)
    code = '''
import json, sys, time
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
filename, gate, raw_item, raw_scope, digest = sys.argv[1:]
item, scope = json.loads(raw_item), json.loads(raw_scope)
journal = JournalStateStore(Path(filename))
deadline = time.monotonic() + 30
while not Path(gate).exists():
    if time.monotonic() > deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
def delay(point):
    if point == 'after_receipt': time.sleep(.01)
journal._fault_hook = delay
receipt = journal.save_with_receipt({'items':[item], 'lineage':[], 'next_id':1},
    scope=scope, key='shared', request_digest=digest, result={'memory':item}, expected_revision=0)
print(json.dumps(receipt, sort_keys=True))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(path), str(gate),
                json.dumps(record()), json.dumps(SCOPE), DIGEST], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True) for _ in range(8)]
    receipts = []
    try:
        gate.touch()
        for process in children:
            out, err = process.communicate(timeout=40)
            assert process.returncode == 0, err
            receipts.append(json.loads(out))
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
    assert receipts == [receipts[0]] * 8
    assert journal.read_snapshot()[0] == 1 and journal.load()["items"] == [record()]


def test_receipt_is_detached_historical_and_conflicts_do_not_mutate(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    receipt = commit(journal)
    frozen = json.loads(json.dumps(receipt))
    receipt["result"]["memory"]["text"] = "caller corruption"
    journal.save({"items": [], "lineage": [], "next_id": 1}, expected_revision=1)
    assert journal.lookup_receipt(SCOPE, "key", DIGEST) == frozen
    with pytest.raises(IdempotencyConflict):
        journal.lookup_receipt(SCOPE, "key", "b" * 64)
    assert journal.read_snapshot()[0] == 2 and journal.load()["items"] == []


@pytest.mark.parametrize("damage", ["delete", "checksum", "revision", "scope", "binding", "forged-result"])
def test_corrupt_receipt_fails_all_normal_entrypoints(tmp_path, damage):
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    commit(journal)
    with closing(sqlite3.connect(journal.path)) as connection:
        if damage == "delete":
            connection.execute("DELETE FROM receipts")
        elif damage == "checksum":
            connection.execute("UPDATE receipts SET checksum=?", ("0" * 64,))
        elif damage == "revision":
            connection.execute("UPDATE receipts SET revision=2")
        elif damage == "scope":
            connection.execute("UPDATE receipts SET scope='{}'")
        elif damage == "binding":
            raw = json.loads(connection.execute("SELECT payload FROM decisions").fetchone()[0])
            raw["receipt"] = None
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            connection.execute("UPDATE decisions SET payload=?,checksum=?", (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        else:
            raw = json.loads(connection.execute("SELECT payload FROM receipts").fetchone()[0])
            raw["result"]["memory"]["text"] = "invented"
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            connection.execute("UPDATE receipts SET payload=?,checksum=?", (encoded, digest))
            event = json.loads(connection.execute("SELECT payload FROM decisions").fetchone()[0])
            event["receipt"]["digest"] = digest
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
            connection.execute("UPDATE decisions SET payload=?,checksum=?", (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        connection.commit()
    for action in (journal.read_snapshot, journal.decisions,
                   lambda: journal.lookup_receipt(SCOPE, "key", DIGEST),
                   lambda: JournalStateStore(journal.path),
                   lambda: journal.save({"items": [], "lineage": []}, expected_revision=1)):
        with pytest.raises(JournalIntegrityError):
            action()


@pytest.mark.parametrize("bad", ["absent-record", "scope-mismatch"])
def test_receipt_cannot_acknowledge_memory_outside_committed_snapshot(tmp_path, bad):
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    item = record()
    if bad == "scope-mismatch":
        item["meta"]["tenant_id"] = "other"
    payload = {"items": [] if bad == "absent-record" else [item], "lineage": []}
    with pytest.raises((ValueError, JournalIntegrityError)):
        journal.save_with_receipt(payload, scope=SCOPE, key="key", request_digest=DIGEST,
                                  result={"memory": item}, expected_revision=0)
    assert journal.read_snapshot()[0] == 0


def test_database_full_rolls_back_receipt_memory_and_decision(tmp_path, monkeypatch):
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    acknowledged = commit(journal, "acknowledged")
    connect = journal._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(journal, "_connect", limited)
    item = record(1, "x" * 1_000_000)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        journal.save_with_receipt({"items": [record(), item], "lineage": [], "next_id": 2},
            scope=SCOPE, key="failed", request_digest=DIGEST, result={"memory": item}, expected_revision=1)
    reopened = JournalStateStore(journal.path)
    assert reopened.lookup_receipt(SCOPE, "failed", DIGEST) is None
    assert reopened.lookup_receipt(SCOPE, "acknowledged", DIGEST) == acknowledged
    assert reopened.read_snapshot()[0] == 1
    assert reopened.load()["items"] == [record()]


def test_receipt_serialization_failure_preserves_acknowledged_history(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    acknowledged = commit(journal, "acknowledged")
    item = record(1)
    item["meta"]["not_json"] = object()
    with pytest.raises((TypeError, ValueError)):
        journal.save_with_receipt({"items": [record(), item], "lineage": []}, scope=SCOPE,
            key="failed", request_digest=DIGEST, result={"memory": item}, expected_revision=1)
    assert journal.lookup_receipt(SCOPE, "failed", DIGEST) is None
    assert journal.lookup_receipt(SCOPE, "acknowledged", DIGEST) == acknowledged
    assert journal.read_snapshot()[0] == 1


@pytest.mark.parametrize("overrides", [
    {"text": "   "}, {"tenant_id": ""}, {"client_id": "é" * 129},
    {"kind": False}, {"meta": {"tenant_id": "other"}},
    {"meta": {"no_merge": 1}}, {"meta": {"x": float("nan")}},
    {"meta": {1: "integer key"}}, {"meta": {"tuple": (1, 2)}},
    {"text": "x" * (1024 * 1024)},
])
def test_noncanonical_requests_are_rejected_before_storage(overrides):
    from daystrom_dml.services.receipt_ingestion import canonical_request
    with pytest.raises(ValueError):
        canonical_request(**{"text": "memory", "tenant_id": "tenant", **overrides})


@pytest.mark.parametrize("vector", [[], [[1., 0.]], [float("nan")], [float("inf")]])
def test_invalid_prepared_vector_never_creates_receipt(tmp_path, vector):
    from daystrom_dml.services.receipt_ingestion import (
        ReceiptEmbeddingError, append_receipted, canonical_request,
    )
    journal = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    request, digest = canonical_request("memory", tenant_id="tenant")
    with pytest.raises(ReceiptEmbeddingError):
        append_receipted(journal, request=request, request_digest=digest, key="key",
            embed=lambda _: vector, embedding_space=lambda: {"declared": "test-v1"},
            capacity=10, hydrate=lambda *_: None, degraded=lambda _: None)
    assert journal.lookup_receipt(SCOPE, "key", digest) is None
    assert journal.read_snapshot() == (0, {"items": [], "lineage": []})
