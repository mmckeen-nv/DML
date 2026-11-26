"""Independent semantic oracles for scoped retirement and historical replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict,
    canonical_retirement_request,
    retire_receipted,
)


SCOPE = dict(tenant_id="tenant", client_id=None, session_id=None, instance_id=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def record(ident=0):
    return dict(schema_version=1, id=ident, text="Original authority", embedding=[1., 0.],
                timestamp=1., salience=1., fidelity=1., level=0, summary_of=[],
                children=[], meta={**SCOPE, "source": {"trusted": False, "revision": 1}})


@pytest.fixture
def journal(tmp_path):
    store = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    store.save({"items": [record(), record(1)], "lineage": [], "next_id": 2},
               expected_revision=0, operation="independent-fixture")
    return store


def request_for(item=None):
    item = record() if item is None else item
    return canonical_retirement_request(item["id"], expected_memory_digest=digest(item),
                                        reason="Superseded instruction", **SCOPE)


def invoke(journal, request=None, request_digest=None, *, key="retire", hydrate=None):
    if request is None:
        request, request_digest = request_for()
    return retire_receipted(journal, request=request, request_digest=request_digest,
                            key=key, hydrate=hydrate or (lambda *_: None), degraded=lambda _: None)


@pytest.mark.parametrize("field,value", [("memory_id", True), ("schema_version", True),
                                         ("reason", []), ("unknown", 1)])
def test_direct_service_rejects_noncanonical_request_before_authority_lookup(journal, monkeypatch, field, value):
    request, _ = request_for()
    request[field] = value
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: pytest.fail("invalid intent reached authority"))
    with pytest.raises(ValueError):
        invoke(journal, request, digest(request))


def test_direct_service_rejects_cyclic_input_before_authority_lookup(journal, monkeypatch):
    request, request_digest = request_for()
    request["reason"] = request
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: pytest.fail("cyclic intent reached authority"))
    with pytest.raises(ValueError):
        invoke(journal, request, request_digest)


def test_direct_service_freezes_caller_intent_before_lookup(journal, monkeypatch):
    request, request_digest = request_for()
    original = deepcopy(request)
    lookup = journal.lookup_receipt

    def mutate_caller(**kwargs):
        request["scope"]["tenant_id"] = "different-tenant"
        request["memory_id"] = 1
        request["reason"] = "Caller mutation during I/O"
        return lookup(**kwargs)

    monkeypatch.setattr(journal, "lookup_receipt", mutate_caller)
    receipt = invoke(journal, request, request_digest)
    assert receipt["scope"] == original["scope"]
    assert receipt["request_digest"] == request_digest
    assert receipt["result"]["memory"]["id"] == 0
    assert journal.load()["items"][1] == record(1)


@pytest.mark.parametrize("mutation", ["target", "unrelated"])
def test_raw_journal_writer_between_read_and_commit_rechecks_full_record(journal, monkeypatch, mutation):
    save = journal.save_with_receipt
    original = deepcopy(journal.load())
    attempted = 0

    def competing_save(payload, **kwargs):
        nonlocal attempted
        attempted += 1
        if attempted == 1:
            writer = JournalStateStore(journal.path)
            revision, current = writer.read_snapshot()
            index = 0 if mutation == "target" else 1
            current["items"][index]["meta"]["source"]["revision"] = 2
            writer.save(current, expected_revision=revision, operation="raw-concurrent-writer")
        return save(payload, **kwargs)

    monkeypatch.setattr(journal, "save_with_receipt", competing_save)
    if mutation == "target":
        with pytest.raises(ReceiptLifecycleConflict):
            invoke(journal)
        assert journal.load()["items"][0]["meta"].get("memory_state") is None
        assert journal.read_snapshot()[0] == 2
    else:
        receipt = invoke(journal)
        assert receipt["result"]["memory"]["meta"]["memory_state"] == "deleted"
        assert journal.load()["items"][1]["meta"]["source"]["revision"] == 2
        assert journal.read_snapshot()[0] == 3
    assert journal.load()["items"][0]["text"] == original["items"][0]["text"]


@pytest.mark.parametrize("wrong_result", ["active", "different-id"])
def test_generic_receipt_cannot_falsely_attest_retirement(journal, wrong_result):
    request, request_digest = request_for()
    revision, payload = journal.read_snapshot()
    item = payload["items"][0 if wrong_result == "active" else 1]
    if wrong_result == "different-id":
        item["meta"]["memory_state"] = "deleted"
    journal.save_with_receipt(payload, scope=SCOPE, key="retire", request_digest=request_digest,
        result={"memory": item}, expected_revision=revision, operation="generic-import")
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest)
    assert journal.read_snapshot() == before


def test_historical_retirement_replays_after_later_physical_removal(journal):
    receipt = invoke(journal)
    revision, payload = journal.read_snapshot()
    payload["items"] = [payload["items"][1]]
    journal.save(payload, expected_revision=revision, operation="explicit-later-removal")
    before = journal.read_snapshot()
    assert invoke(journal) == receipt
    assert journal.read_snapshot() == before
