"""Parent-held receipt oracles for retirement process death and competing clients."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from daystrom_dml.journal import JournalStateStore, OUTBOX_FAULT_POINTS, RECEIPT_FAULT_POINTS
from daystrom_dml.services.lifecycle import suppression_reason
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_lifecycle import canonical_retirement_request, memory_digest, retire_receipted
from test_projection import SCOPE, run_child, write


def prepare(tmp_path, schema):
    source = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                               receipt_mode=True, outbox_mode=schema == 3, snapshot_interval=1)
    original = write(source, meta={"source_trust": "untrusted", "provenance": {"typed": [True, 1, 1.0]}})
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(source.path, destination)
        source = JournalStateStore(destination, snapshot_interval=1)
    request, digest = canonical_retirement_request(original["result"]["memory"]["id"],
        expected_memory_digest=memory_digest(original["result"]["memory"]),
        reason="Operator retired this obsolete memory", **SCOPE)
    return source, original, request, digest


def retire(source, request, digest):
    return retire_receipted(source, request=request, request_digest=digest, key="retire-once",
                            hydrate=lambda *_: None, degraded=lambda _: None)


def assert_complete(source, original, request, digest, initial_revision):
    receipt = source.lookup_receipt(SCOPE, "retire-once", digest)
    assert receipt is not None
    assert source.lookup_receipt(SCOPE, "one", original["request_digest"]) == original
    revision, state = source.read_snapshot()
    assert revision == initial_revision + 1
    assert len(state["items"]) == 1
    retired = receipt["result"]["memory"]
    assert state["items"] == [retired]
    expected = json.loads(json.dumps(original["result"]["memory"]))
    expected["meta"].update(memory_state="deleted", retirement_decision={
        "schema_version": "dml-retirement-decision-v1",
        "prior_memory_digest": request["expected_memory_digest"], "reason": request["reason"],
    })
    assert json.dumps(retired, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert retired["meta"]["memory_state"] == "deleted"
    assert suppression_reason(retired["meta"], now=100) == "state_deleted"
    assert {k: v for k, v in retired.items() if k != "meta"} == {
        k: v for k, v in original["result"]["memory"].items() if k != "meta"}
    assert retired["meta"]["source_trust"] == "untrusted"
    assert retired["meta"]["provenance"] == {"typed": [True, 1, 1.0]}
    assert retire(source, request, digest) == receipt
    assert source.read_snapshot()[0] == revision
    decision = source.decisions(after_revision=initial_revision)[0]
    assert decision["operation"] == "retire-receipt-v1"
    assert decision["receipt"]["request_digest"] == digest
    if source.schema_version in (3, 4):
        events = source.outbox_events()["events"]
        retirement = [event for event in events if event["operation"] == "retire-receipt-v1"]
        assert len(retirement) == 1
        assert retirement[0]["source_revision"] == revision
        assert retirement[0]["state"] == state
        assert retirement[0]["receipt"] == decision["receipt"]
    return receipt


KILL_CASES = [(schema, point) for schema in (2, 3, 4)
              for point in (RECEIPT_FAULT_POINTS if schema == 2 else OUTBOX_FAULT_POINTS)]


@pytest.mark.parametrize("schema,point", KILL_CASES)
def test_retirement_process_death_preserves_acknowledged_history(tmp_path, schema, point):
    source, original, request, digest = prepare(tmp_path, schema)
    initial_revision, before = source.read_snapshot()
    code = '''
import json, os, signal, sys, threading
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_lifecycle import retire_receipted
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
retire_receipted(source, request=json.loads(encoded), request_digest=digest,
                 key='retire-once', hydrate=lambda *_: None, degraded=lambda _: None)
raise AssertionError('retirement hook was not reached')
'''
    child = run_child(code, source.path, point, json.dumps(request), digest)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    source = JournalStateStore(source.path)
    committed = source.lookup_receipt(SCOPE, "retire-once", digest)
    assert (committed is not None) == (point == "after_commit")
    if committed is None:
        assert source.read_snapshot() == (initial_revision, before)
        retire(source, request, digest)
    assert_complete(source, original, request, digest, initial_revision)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_simultaneous_retirement_clients_commit_one_transition(tmp_path, schema):
    source, original, request, digest = prepare(tmp_path, schema)
    revision = source.read_snapshot()[0]
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def submit(_):
        barrier.wait(timeout=30)
        return retire(source, request, digest)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        receipts = list(pool.map(submit, range(clients)))
    assert receipts == [receipts[0]] * clients
    assert receipts[0] == assert_complete(source, original, request, digest, revision)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_competing_retirement_processes_replay_one_durable_result(tmp_path, schema):
    source, original, request, digest = prepare(tmp_path, schema)
    revision = source.read_snapshot()[0]
    gate = tmp_path / "start"
    code = '''
import json, sys, time
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_lifecycle import retire_receipted
path, gate, encoded, digest = sys.argv[1:]
source = JournalStateStore(Path(path))
deadline = time.monotonic() + 30
while not Path(gate).exists():
    if time.monotonic() > deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
receipt = retire_receipted(source, request=json.loads(encoded), request_digest=digest,
                           key='retire-once', hydrate=lambda *_: None, degraded=lambda _: None)
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
