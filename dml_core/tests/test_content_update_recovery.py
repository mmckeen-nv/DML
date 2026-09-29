"""Parent-held receipt oracles for content_update process death and competing clients."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from daystrom_dml.journal import JournalStateStore, OUTBOX_FAULT_POINTS, RECEIPT_FAULT_POINTS
from daystrom_dml.services.lifecycle import suppression_reason
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_lifecycle import memory_digest
from daystrom_dml.services.receipt_update import canonical_update_request, update_receipted
from test_projection import IDENTITY, SCOPE, run_child, write


def prepare(tmp_path, schema):
    source = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                               receipt_mode=True, outbox_mode=schema == 3, snapshot_interval=1)
    original = write(source, meta={"source_trust": "untrusted", "provenance": {"typed": [True, 1, 1.0]}})
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(source.path, destination)
        source = JournalStateStore(destination, snapshot_interval=1)
    request, digest = canonical_update_request(original["result"]["memory"]["id"],
        text="The corrected memory text",
        expected_memory_digest=memory_digest(original["result"]["memory"]),
        reason="Operator corrected the record", **SCOPE)
    return source, original, request, digest


def update(source, request, digest):
    return update_receipted(source, request=request, request_digest=digest, key="update-once",
                            embed=lambda _: np.array([0., 1.]), embedding_space=lambda: IDENTITY,
                            hydrate=lambda *_: None, degraded=lambda _: None)


def assert_complete(source, original, request, digest, initial_revision):
    receipt = source.lookup_receipt(SCOPE, "update-once", digest)
    assert receipt is not None
    assert source.lookup_receipt(SCOPE, "one", original["request_digest"]) == original
    revision, state = source.read_snapshot()
    assert revision == initial_revision + 1
    updated = receipt["result"]["memory"]
    expected = json.loads(json.dumps(original["result"]["memory"]))
    expected.update(text=request["text"], embedding=[0., 1.])
    expected["meta"]["content_update_decision"] = {
        "schema_version": "dml-content-update-decision-v1",
        "prior_memory_digest": request["expected_memory_digest"], "reason": request["reason"]}
    assert json.dumps(updated, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert state["items"] == [updated]
    assert state["next_id"] == 1
    assert state["embedding_contract"]["identity"] == IDENTITY
    assert state["embedding_contract"]["dimension"] == 2
    assert suppression_reason(updated["meta"], now=100) == "untrusted_source"
    def unavailable(*_):
        raise AssertionError("A historical retry must not access the embedding backend")
    assert update_receipted(source, request=request, request_digest=digest, key="update-once",
        embed=unavailable, embedding_space=unavailable, hydrate=unavailable, degraded=unavailable) == receipt
    assert source.read_snapshot()[0] == revision
    decision = source.decisions(after_revision=initial_revision)[0]
    assert decision["operation"] == "update-receipt-v1"
    assert decision["receipt"]["request_digest"] == digest
    assert len(decision["changed"]) == 1 and decision["deleted"] == []
    if source.schema_version in (3, 4):
        matching = [event for event in source.outbox_events()["events"] if event["operation"] == "update-receipt-v1"]
        assert len(matching) == 1
        assert matching[0]["source_revision"] == revision
        assert matching[0]["state"] == state
        assert matching[0]["receipt"] == decision["receipt"]
    return receipt


KILL_CASES = [(schema, point) for schema in (2, 3, 4)
              for point in ("during_embedding", *(RECEIPT_FAULT_POINTS if schema == 2 else OUTBOX_FAULT_POINTS))]


@pytest.mark.parametrize("schema,point", KILL_CASES)
def test_content_update_process_death_preserves_acknowledged_history(tmp_path, schema, point):
    source, original, request, digest = prepare(tmp_path, schema)
    initial_revision, before = source.read_snapshot()
    code = '''
import json, os, signal, sys, threading
import numpy as np
IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_update import update_receipted
path, point, encoded, digest = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
            threading.Event().wait(5)
            os._exit(80)
        else:
            os._exit(79)
source = JournalStateStore(Path(path), snapshot_interval=1, fault_hook=crash)
def embed(_):
    crash("during_embedding")
    return np.array([0., 1.])
update_receipted(source, request=json.loads(encoded), request_digest=digest,
                 key='update-once', embed=embed, embedding_space=lambda: IDENTITY,
                 hydrate=lambda *_: None, degraded=lambda _: None)
raise AssertionError('content_update hook was not reached')
'''
    child = run_child(code, source.path, point, json.dumps(request), digest)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    source = JournalStateStore(source.path)
    committed = source.lookup_receipt(SCOPE, "update-once", digest)
    assert (committed is not None) == (point == "after_commit")
    if committed is None:
        assert source.read_snapshot() == (initial_revision, before)
        update(source, request, digest)
    assert_complete(source, original, request, digest, initial_revision)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_simultaneous_content_update_clients_commit_one_transition(tmp_path, schema):
    source, original, request, digest = prepare(tmp_path, schema)
    revision = source.read_snapshot()[0]
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def submit(_):
        barrier.wait(timeout=30)
        return update(source, request, digest)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        receipts = list(pool.map(submit, range(clients)))
    assert receipts == [receipts[0]] * clients
    assert receipts[0] == assert_complete(source, original, request, digest, revision)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_competing_content_update_processes_replay_one_durable_result(tmp_path, schema):
    source, original, request, digest = prepare(tmp_path, schema)
    revision = source.read_snapshot()[0]
    gate = tmp_path / "start"
    code = '''
import json, sys, time
import numpy as np
IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_update import update_receipted
path, gate, encoded, digest = sys.argv[1:]
source = JournalStateStore(Path(path))
deadline = time.monotonic() + 30
while not Path(gate).exists():
    if time.monotonic() > deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
receipt = update_receipted(source, request=json.loads(encoded), request_digest=digest,
                           key='update-once', embed=lambda _: np.array([0., 1.]), embedding_space=lambda: IDENTITY,
                           hydrate=lambda *_: None, degraded=lambda _: None)
print(json.dumps(receipt, sort_keys=True))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(source.path), str(gate),
                json.dumps(request), digest], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True) for _ in range(8)]
    receipts = []
    try:
        gate.touch()
        for process in children:
            out, err = process.communicate(timeout=45)
            assert process.returncode == 0, err
            receipts.append(json.loads(out))
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
    assert receipts == [receipts[0]] * 8
    assert receipts[0] == assert_complete(source, original, request, digest, revision)
