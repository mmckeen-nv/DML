"""Parent-held promotion oracles across process death and competing writers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from daystrom_dml.journal import JournalStateStore, OUTBOX_FAULT_POINTS, RECEIPT_FAULT_POINTS
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_promotion import canonical_promotion_request, promote_receipted
from test_projection import IDENTITY, SCOPE, run_child, write


def encoded(value):
    """Preserve JSON type distinctions in independent parent-held comparisons."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_digest(record):
    return hashlib.sha256(encoded(record).encode("utf-8")).hexdigest()


def prepare(tmp_path, schema):
    journal = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                               receipt_mode=True, outbox_mode=schema == 3, snapshot_interval=1)
    acknowledged = [
        write(journal, "first", text="The first source fact", meta={
            "source_trust": "trusted", "provenance": {"source": "first", "typed": [True, 1, 1.0]}}),
        write(journal, "second", text="The second source fact", meta={
            "source_trust": "trusted", "provenance": {"source": "second", "typed": [False, 0, 0.0]}}),
        write(journal, "unrelated", text="An unrelated record", meta={"source_trust": "untrusted"}),
    ]
    revision, state = journal.read_snapshot()
    # Public append records deliberately disallow merging. This trusted fixture
    # models eligible imported records; it does not change the ingestion policy.
    for record, timestamp, salience, fidelity in zip(state["items"][:2],
            (101.0, 202.0), (0.8, 0.2), (0.7, 0.9)):
        record["meta"]["no_merge"] = False
        record.update(timestamp=timestamp, salience=salience, fidelity=fidelity)
    journal.save(state, expected_revision=revision, operation="test-import-mergeable-records")
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(journal.path, destination)
        journal = JournalStateStore(destination, snapshot_interval=1)
    revision, before = journal.read_snapshot()
    request, digest = canonical_promotion_request([
        {"memory_id": record["id"], "expected_memory_digest": record_digest(record)}
        for record in before["items"][:2]], text="The explicit merged interpretation",
        reason="Operator selected these exact source records", **SCOPE)
    return journal, acknowledged, (revision, before), request, digest


def promote(journal, request, digest, *, key="promote-once", embed=None):
    return promote_receipted(journal, request=request, request_digest=digest, key=key,
        embed=embed or (lambda _: np.array([0., 1.])), embedding_space=lambda: IDENTITY,
        capacity=1000, hydrate=lambda *_: None, degraded=lambda _: None)


def assert_history(journal, acknowledged, before):
    for old in acknowledged:
        assert encoded(journal.lookup_receipt(SCOPE, old["key"], old["request_digest"])) == encoded(old)
    current = journal.read_snapshot()[1]
    assert encoded(current["items"][:len(before["items"])]) == encoded(before["items"])
    assert encoded(current["lineage"]) == encoded(before["lineage"])
    assert encoded(current["embedding_contract"]) == encoded(before["embedding_contract"])


def expected_record(before, request, ident):
    # Complete output oracle is defined from parent-held originals, never from
    # the service's result or any of its private record construction helpers.
    selected = {record["id"]: record for record in before["items"]}
    originals = [selected[source["memory_id"]] for source in request["sources"]]
    source_ids = [record["id"] for record in originals]
    return {
        "schema_version": 1, "id": ident, "text": request["text"],
        "embedding": [0., 1.], "timestamp": min(record["timestamp"] for record in originals),
        "salience": min(record["salience"] for record in originals),
        "fidelity": min(record["fidelity"] for record in originals),
        "level": 1, "summary_of": source_ids, "children": source_ids,
        "meta": {
            **SCOPE, "kind": "memory", "source_trust": "trusted", "no_merge": True,
            "promotion_decision": {
                "schema_version": "dml-promotion-decision-v1", "reason": request["reason"],
                "sources": [
                    {"memory_digest": record_digest(record), "memory": record}
                    for record in originals],
            },
        },
    }


def assert_complete(journal, acknowledged, original, request, digest):
    initial_revision, before = original
    receipt = journal.lookup_receipt(SCOPE, "promote-once", digest)
    assert receipt is not None
    assert_history(journal, acknowledged, before)
    revision, state = journal.read_snapshot()
    assert revision == initial_revision + 1
    promoted = expected_record(before, request, before["next_id"])
    assert encoded(receipt["result"]["memory"]) == encoded(promoted)
    expected = json.loads(encoded(before))
    expected["items"].append(promoted)
    expected["next_id"] += 1
    assert encoded(state) == encoded(expected)

    def unavailable(*_):
        raise AssertionError("A historical retry must not access a backend or hydrate")

    assert promote_receipted(journal, request=request, request_digest=digest, key="promote-once",
        embed=unavailable, embedding_space=unavailable, capacity=0,
        hydrate=unavailable, degraded=unavailable) == receipt
    assert journal.read_snapshot()[0] == revision
    decisions = journal.decisions(after_revision=initial_revision)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision["operation"] == "promote-receipt-v1"
    assert decision["receipt"]["request_digest"] == digest
    assert len(decision["changed"]) == 1 and decision["deleted"] == []
    assert decision["changed"][0]["id"] == str(promoted["id"])
    assert decision["changed"][0]["bucket"] == "items"
    if journal.schema_version in (3, 4):
        events = [event for event in journal.outbox_events()["events"]
                  if event["operation"] == "promote-receipt-v1"]
        assert len(events) == 1
        assert events[0]["source_revision"] == revision
        assert encoded(events[0]["state"]) == encoded(expected)
        assert events[0]["receipt"] == decision["receipt"]
    return receipt


KILL_CASES = [(schema, point) for schema in (2, 3, 4)
              for point in ("during_embedding", *(RECEIPT_FAULT_POINTS if schema == 2 else OUTBOX_FAULT_POINTS))]


@pytest.mark.parametrize("schema,point", KILL_CASES,
                         ids=[f"schema-{schema}-{point}" for schema, point in KILL_CASES])
def test_promotion_process_death_preserves_sources_and_acknowledged_history(tmp_path, schema, point):
    journal, acknowledged, original, request, digest = prepare(tmp_path, schema)
    code = '''
import json, os, signal, sys, threading
import numpy as np
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_promotion import promote_receipted
IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
path, point, encoded, digest = sys.argv[1:]
def crash(here):
    if here == point:
        if os.name == 'posix':
            os.kill(os.getpid(), signal.SIGKILL)
            threading.Event().wait(5)
            os._exit(80)
        else:
            os._exit(79)
journal = JournalStateStore(Path(path), snapshot_interval=1, fault_hook=crash)
def embed(_):
    crash("during_embedding")
    return np.array([0., 1.])
promote_receipted(journal, request=json.loads(encoded), request_digest=digest,
                 key='promote-once', embed=embed, embedding_space=lambda: IDENTITY,
                 capacity=1000, hydrate=lambda *_: None, degraded=lambda _: None)
raise AssertionError('promotion hook was not reached')
'''
    child = run_child(code, journal.path, point, json.dumps(request), digest)
    assert child.returncode == (-9 if os.name == "posix" else 79), child.stderr
    reopened = JournalStateStore(journal.path)
    committed = reopened.lookup_receipt(SCOPE, "promote-once", digest)
    assert (committed is not None) == (point == "after_commit")
    if committed is None:
        assert encoded(reopened.read_snapshot()) == encoded(original)
        assert_history(reopened, acknowledged, original[1])
        promote(reopened, request, digest)
    assert_complete(reopened, acknowledged, original, request, digest)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_simultaneous_promotion_clients_commit_one_transition(tmp_path, schema):
    journal, acknowledged, original, request, digest = prepare(tmp_path, schema)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)

    def submit(_):
        barrier.wait(timeout=30)
        return promote(journal, request, digest)

    with ThreadPoolExecutor(max_workers=clients) as pool:
        receipts = list(pool.map(submit, range(clients)))
    assert receipts == [receipts[0]] * clients
    assert receipts[0] == assert_complete(journal, acknowledged, original, request, digest)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_competing_promotion_processes_replay_one_durable_result(tmp_path, schema):
    journal, acknowledged, original, request, digest = prepare(tmp_path, schema)
    gate = tmp_path / "start"
    code = '''
import json, sys, time
import numpy as np
from pathlib import Path
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_promotion import promote_receipted
IDENTITY = {"backend": "test.reference", "revision": "v1", "model": None, "mode": "native"}
path, gate, encoded, digest = sys.argv[1:]
journal = JournalStateStore(Path(path))
deadline = time.monotonic() + 30
while not Path(gate).exists():
    if time.monotonic() > deadline: raise TimeoutError('parent barrier')
    time.sleep(.005)
receipt = promote_receipted(journal, request=json.loads(encoded), request_digest=digest,
                           key='promote-once', embed=lambda _: np.array([0., 1.]),
                           embedding_space=lambda: IDENTITY, capacity=1000,
                           hydrate=lambda *_: None, degraded=lambda _: None)
print(json.dumps(receipt, sort_keys=True))
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(journal.path), str(gate),
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
    assert receipts[0] == assert_complete(journal, acknowledged, original, request, digest)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_different_keys_can_derive_from_overlapping_source_snapshots(tmp_path, schema):
    journal, acknowledged, original, request, digest = prepare(tmp_path, schema)
    revision, before = original
    other, other_digest = canonical_promotion_request(request["sources"][1:],
        text="A separate explicit interpretation", reason="A different operator decision", **SCOPE)
    barrier = threading.Barrier(2)

    def embed(_):
        # Both requests bind the original source records before either commit.
        barrier.wait(timeout=10)
        return np.array([0., 1.])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(promote, journal, intent, intent_digest, key=key, embed=embed)
                   for key, intent, intent_digest in (("first-promotion", request, digest),
                                                     ("second-promotion", other, other_digest))]
        receipts = [future.result(timeout=30) for future in futures]
    assert_history(journal, acknowledged, before)
    final_revision, after = journal.read_snapshot()
    assert final_revision == revision + 2
    assert after["next_id"] == before["next_id"] + 2
    assert len(after["items"]) == len(before["items"]) + 2
    assert {receipt["result"]["memory"]["id"] for receipt in receipts} == {
        before["next_id"], before["next_id"] + 1}
    for receipt, intent, intent_digest in zip(receipts, (request, other), (digest, other_digest)):
        memory = receipt["result"]["memory"]
        assert encoded(memory) == encoded(expected_record(before, intent, memory["id"]))
        assert encoded(after["items"][memory["id"]]) == encoded(memory)
        assert promote(journal, intent, intent_digest, key=receipt["key"]) == receipt
    decisions = journal.decisions(after_revision=revision)
    assert len(decisions) == 2
    assert all(decision["operation"] == "promote-receipt-v1" and len(decision["changed"]) == 1
               and decision["deleted"] == [] for decision in decisions)
    if schema in (3, 4):
        events = [event for event in journal.outbox_events()["events"]
                  if event["operation"] == "promote-receipt-v1"]
        assert [event["source_revision"] for event in events] == [revision + 1, revision + 2]
        assert encoded(events[-1]["state"]) == encoded(after)
