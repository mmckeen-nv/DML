"""Independent adversarial oracles for explicit two-record supersession."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict
from daystrom_dml.services.receipt_supersession import (
    canonical_supersession_request,
    supersede_receipted,
)

SCOPE = dict(tenant_id="tenant", client_id=None, session_id=None, instance_id=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def record(ident=0):
    return dict(schema_version=1, id=ident, text=f"Independent source claim {ident}",
                embedding=[1., 0.], timestamp=1., salience=1., fidelity=1., level=0,
                summary_of=[], children=[], meta={**SCOPE, "provenance": {"revision": 1}})


@pytest.fixture
def journal(tmp_path):
    store = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    store.save({"items": [record(i) for i in range(3)], "lineage": [], "next_id": 3},
               expected_revision=0, operation="independent-fixture")
    return store


def request_for(source=None, replacement=None):
    source = record() if source is None else source
    replacement = record(1) if replacement is None else replacement
    return canonical_supersession_request(source["id"], replacement_memory_id=replacement["id"],
        expected_memory_digest=digest(source), expected_replacement_digest=digest(replacement),
        reason="An explicit owner correction", **SCOPE)


def invoke(journal, request=None, request_digest=None, *, key="supersede"):
    if request is None:
        request, request_digest = request_for()
    return supersede_receipted(journal, request=request, request_digest=request_digest,
        key=key, hydrate=lambda *_: None, degraded=lambda _: None)


def save_items(journal, items):
    revision, state = journal.read_snapshot()
    state["items"] = items
    journal.save(state, expected_revision=revision, operation="independent-change")


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("replacement_memory_id", False),
    ("expected_replacement_digest", "A" * 64), ("schema_version", True),
    ("reason", []), ("unknown", 1),
])
def test_invalid_direct_intent_never_reaches_authority(journal, monkeypatch, field, value):
    request, _ = request_for()
    request[field] = value
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: pytest.fail("invalid input reached authority"))
    with pytest.raises(ValueError):
        invoke(journal, request, digest(request))


def test_cyclic_intent_never_reaches_authority(journal, monkeypatch):
    request, request_digest = request_for()
    request["reason"] = request
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: pytest.fail("cyclic intent reached authority"))
    with pytest.raises(ValueError):
        invoke(journal, request, request_digest)


def test_caller_cannot_switch_either_record_or_scope_during_lookup(journal, monkeypatch):
    request, request_digest = request_for()
    original = deepcopy(request)
    lookup = journal.lookup_receipt

    def mutate_caller(**kwargs):
        request["scope"]["tenant_id"] = "other"
        request["memory_id"] = 1
        request["replacement_memory_id"] = 2
        request["reason"] = "Changed during I/O"
        return lookup(**kwargs)

    monkeypatch.setattr(journal, "lookup_receipt", mutate_caller)
    receipt = invoke(journal, request, request_digest)
    assert receipt["scope"] == original["scope"]
    assert receipt["request_digest"] == request_digest
    assert receipt["result"]["memory"]["id"] == 0
    assert receipt["result"]["memory"]["meta"]["superseded_by"] == 1
    assert journal.load()["items"][1:] == [record(1), record(2)]


@pytest.mark.parametrize("changed", [0, 1, 2])
def test_raw_writer_race_rechecks_both_full_records(journal, monkeypatch, changed):
    save = journal.save_with_receipt
    attempts = 0

    def competing_save(payload, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            writer = JournalStateStore(journal.path)
            state = writer.load()
            # Type changes are meaningful provenance changes, even when Python ==
            # would call these numbers equal.
            state["items"][changed]["meta"]["provenance"]["revision"] = 1.0
            save_items(writer, state["items"])
        return save(payload, **kwargs)

    monkeypatch.setattr(journal, "save_with_receipt", competing_save)
    if changed in (0, 1):
        with pytest.raises(ReceiptLifecycleConflict):
            invoke(journal)
        assert journal.read_snapshot()[0] == 2
        assert "superseded_by" not in journal.load()["items"][0]["meta"]
    else:
        receipt = invoke(journal)
        assert receipt["revision"] == 3
        assert journal.load()["items"][1] == record(1)
    assert type(journal.load()["items"][changed]["meta"]["provenance"]["revision"]) is float


@pytest.mark.parametrize("mutation", ["source-remove", "replacement-remove", "replacement-retire"])
def test_historical_retry_precedes_current_eligibility(journal, mutation):
    receipt = invoke(journal)
    items = journal.load()["items"]
    if mutation == "source-remove":
        items = items[1:]
    elif mutation == "replacement-remove":
        items = [items[0], items[2]]
    else:
        items[1]["meta"]["memory_state"] = "deleted"
    save_items(journal, items)
    before = journal.read_snapshot()
    assert invoke(journal) == receipt
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("metadata", [
    {"memory_state": "active", "lifecycle_state": "deleted"},
    {"memory_state": "active", "lifecycle_state": "superseded"},
    {"memory_state": "active", "retirement_decision": {}},
    {"memory_state": "active", "supersession_decision": {}},
])
def test_terminal_history_cannot_be_hidden_by_active_alias(journal, position, metadata):
    items = journal.load()["items"]
    items[position]["meta"].update(metadata)
    save_items(journal, items)
    request, request_digest = request_for(items[0], items[1])
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest)
    assert journal.read_snapshot() == before


def test_two_step_chain_retains_each_explicit_link_without_rewriting_history(journal):
    first = invoke(journal)
    items = journal.load()["items"]
    request, request_digest = request_for(items[1], items[2])
    second = invoke(journal, request, request_digest, key="second-correction")
    current = journal.load()["items"]
    assert current[0] == first["result"]["memory"]
    assert current[1] == second["result"]["memory"]
    assert current[0]["meta"]["superseded_by"] == 1
    assert current[1]["meta"]["superseded_by"] == 2
    assert current[2] == record(2)
    # A->B->C may not be silently collapsed into a fabricated A->C decision.
    assert invoke(journal) == first
    reverse, reverse_digest = request_for(current[2], current[0])
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, reverse, reverse_digest, key="cycle")


@pytest.mark.parametrize("corruption", [
    "wrong-link", "bool-link", "wrong-target", "wrong-source-digest",
    "wrong-replacement-digest", "extra-decision-field", "wrong-reason",
])
def test_generic_receipt_cannot_acknowledge_another_supersession(journal, corruption):
    request, request_digest = request_for()
    revision, state = journal.read_snapshot()
    item = state["items"][0]
    decision = {
        "schema_version": "dml-supersession-decision-v1",
        "prior_memory_digest": digest(record()), "replacement_memory_id": 1,
        "replacement_memory_digest": digest(record(1)),
        "reason": "An explicit owner correction",
    }
    item["meta"].update(memory_state="superseded", superseded_by=1,
                        supersession_decision=decision)
    if corruption == "wrong-link":
        item["meta"]["superseded_by"] = 2
    elif corruption == "bool-link":
        item["meta"]["superseded_by"] = True
    elif corruption == "wrong-target":
        decision["replacement_memory_id"] = 2
    elif corruption == "wrong-source-digest":
        decision["prior_memory_digest"] = "f" * 64
    elif corruption == "wrong-replacement-digest":
        decision["replacement_memory_digest"] = "f" * 64
    elif corruption == "extra-decision-field":
        decision["invented_authority"] = "system"
    else:
        decision["reason"] = "Different intent"
    journal.save_with_receipt(state, scope=SCOPE, key="supersede", request_digest=request_digest,
        result={"memory": item}, expected_revision=revision, operation="generic-import")
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest)
    assert journal.read_snapshot() == before


def test_revision_retry_rechecks_replacement_expiry(journal, monkeypatch):
    from daystrom_dml.services import receipt_supersession
    items = journal.load()["items"]
    items[1]["meta"]["expires_at"] = 100.0
    save_items(journal, items)
    request, request_digest = request_for(items[0], items[1])
    clock = [99.0]
    monkeypatch.setattr(receipt_supersession.time, "time", lambda: clock[0])
    save = journal.save_with_receipt
    attempts = 0

    def race_then_expire(payload, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            writer = JournalStateStore(journal.path)
            current = writer.load()["items"]
            current[2]["text"] = "Unrelated writer causes a rebase"
            save_items(writer, current)
            clock[0] = 100.0
        return save(payload, **kwargs)

    monkeypatch.setattr(journal, "save_with_receipt", race_then_expire)
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest)
    assert attempts == 1
    assert "superseded_by" not in journal.load()["items"][0]["meta"]
    assert journal.lookup_receipt(SCOPE, "supersede", request_digest) is None


def test_opposing_concurrent_decisions_cannot_create_a_cycle(journal):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    barrier = threading.Barrier(2)
    requests = [request_for(record(), record(1)), request_for(record(1), record())]

    def submit(position):
        request, request_digest = requests[position]
        barrier.wait(timeout=10)
        try:
            return invoke(journal, request, request_digest, key=f"edge-{position}")
        except ReceiptLifecycleConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (0, 1)))
    assert sum(receipt is not None for receipt in results) == 1
    linked = [item for item in journal.load()["items"] if "superseded_by" in item["meta"]]
    assert len(linked) == 1
    winner = next(receipt for receipt in results if receipt is not None)
    assert linked == [winner["result"]["memory"]]
    assert journal.read_snapshot()[0] == 2
    for position, (request, request_digest) in enumerate(requests):
        assert journal.lookup_receipt(SCOPE, f"edge-{position}", request_digest) == results[position]
