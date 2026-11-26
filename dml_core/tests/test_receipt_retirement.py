"""Retirement preconditions, historical replay, concurrency, and durable outcomes."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from daystrom_dml.journal import (
    IdempotencyConflict, JournalSchemaError, JournalStateStore,
    RECEIPT_FAULT_POINTS, RevisionConflict,
)
from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import (
    ReceiptCommitRejected, ReceiptCommitUncertain, canonical_request,
)
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, canonical_retirement_request,
    memory_digest, retire_receipted,
)

SCOPE = dict(tenant_id="tenant", client_id="client", session_id="session", instance_id="instance")


def record(ident=0):
    return MemoryItem(id=ident, text="Original text, preserved after retirement",
        embedding=np.array([1., 0., 0., 0.], dtype=np.float32), timestamp=123.0,
        salience=0.9, fidelity=0.8, level=0, summary_of=[10, 11],
        meta={**SCOPE, "no_merge": True, "source_trust": "untrusted", "source": "fixture",
              "typed": [True, 1, 1.0, None], "nested": {"confidence": 0.5}}).to_dict()


def seeded(tmp_path, schema=2):
    journal = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                                receipt_mode=True, outbox_mode=(schema == 3))
    original = record()
    payload = {"items": [original, record(1)], "lineage": [record(10)], "next_id": 12,
               "embedding_contract": {"opaque": "unchanged"}, "extra": {"typed": [1, True]}}
    journal.save_with_receipt(payload, scope=SCOPE, key="ingested",
        request_digest="a" * 64, result={"memory": original}, expected_revision=0)
    if schema == 4:
        target = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(journal.path, target)
        journal = JournalStateStore(target)
    return journal, copy.deepcopy(original)


def intent(original, **overrides):
    return canonical_retirement_request(**{"memory_id": original["id"],
        "expected_memory_digest": memory_digest(original), "reason": "Owner requested retirement",
        **SCOPE, **overrides})


def retire(journal, original=None, *, request=None, digest=None, key="retire", **callbacks):
    if request is None:
        request, digest = intent(original)
    return retire_receipted(journal, request=request, request_digest=digest, key=key,
        hydrate=callbacks.get("hydrate", lambda *_: None),
        degraded=callbacks.get("degraded", lambda _: None))


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_retirement_preserves_full_record_provenance_capacity_and_other_state(tmp_path, schema):
    journal, original = seeded(tmp_path, schema)
    revision, before = journal.read_snapshot()
    hydrated = []
    receipt = retire(journal, original, hydrate=lambda *args: hydrated.append(args))
    after_revision, after = journal.read_snapshot()
    expected = copy.deepcopy(before)
    expected["items"][0]["meta"]["memory_state"] = "deleted"
    expected["items"][0]["meta"]["retirement_decision"] = {
        "schema_version": "dml-retirement-decision-v1",
        "prior_memory_digest": memory_digest(original), "reason": "Owner requested retirement"}
    assert after == expected
    assert json.dumps(after, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert len(after["items"]) == len(before["items"]) and after["next_id"] == before["next_id"]
    assert receipt["result"]["memory"] == expected["items"][0]
    assert receipt["revision"] == after_revision == revision + 1
    assert hydrated == [(after_revision, after)]
    assert journal.schema_version == schema
    assert journal.decisions()[-1]["operation"] == "retire-receipt-v1"
    if schema in (3, 4):
        event = journal.outbox_events()["events"][-1]
        assert event["operation"] == "retire-receipt-v1"
        assert event["state"] == after
    assert JournalStateStore(journal.path).lookup_receipt(SCOPE, "retire", intent(original)[1]) == receipt


def test_memory_digest_is_complete_and_type_sensitive():
    original = record()
    expected = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert memory_digest(original) == expected
    for value in (1, 1.0, True, "1", None):
        changed = copy.deepcopy(original)
        changed["unknown_provenance"] = value
        assert memory_digest(changed) != expected
    hashes = {memory_digest({**original, "unknown_provenance": value}) for value in (1, 1.0, True, "1", None)}
    assert len(hashes) == 5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), (1, 2), {1: "integer-key"}, object()])
def test_memory_digest_rejects_noncanonical_metadata(bad):
    original = record()
    original["meta"]["bad"] = bad
    with pytest.raises(ValueError):
        memory_digest(original)


@pytest.mark.parametrize("overrides", [
    {"memory_id": True}, {"memory_id": -1}, {"memory_id": 0.0},
    {"expected_memory_digest": "A" * 64}, {"expected_memory_digest": "0" * 63},
    {"expected_memory_digest": None}, {"reason": " "}, {"reason": "é" * 513},
    {"reason": True}, {"tenant_id": None}, {"tenant_id": ""},
    {"client_id": True}, {"session_id": "é" * 129}, {"instance_id": " "},
])
def test_retirement_intent_validates_inputs(overrides):
    with pytest.raises(ValueError):
        intent(record(), **overrides)


@pytest.mark.parametrize("damage", ["digest", "schema", "extra", "scope", "bool-id", "cycle"])
def test_direct_service_rejects_invalid_intent_before_storage(tmp_path, monkeypatch, damage):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    if damage == "digest":
        digest = "b" * 64
    elif damage == "schema":
        request["schema_version"] = "dml-retire-request-v2"
    elif damage == "extra":
        request["unrecognized"] = "no"
    elif damage == "scope":
        del request["scope"]["client_id"]
    elif damage == "bool-id":
        request["memory_id"] = False
    else:
        request["scope"]["tenant_id"] = request
    def no_storage(*args, **kwargs):
        pytest.fail("Invalid intent must not access storage")
    monkeypatch.setattr(journal, "lookup_receipt", no_storage)
    with pytest.raises(ValueError):
        retire(journal, request=request, digest=digest)


def test_direct_service_detaches_scope_and_intent_before_io(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    lookup = journal.lookup_receipt
    def mutate_caller(**kwargs):
        request["scope"]["tenant_id"] = "other"
        request["memory_id"] = 1
        request["reason"] = "Changed after validation"
        return lookup(**kwargs)
    monkeypatch.setattr(journal, "lookup_receipt", mutate_caller)
    receipt = retire(journal, request=request, digest=digest)
    assert receipt["scope"] == SCOPE and receipt["result"]["memory"]["id"] == 0
    assert receipt["result"]["memory"]["meta"]["retirement_decision"]["reason"] == "Owner requested retirement"


@pytest.mark.parametrize("case", ["missing", "lineage", *SCOPE])
def test_missing_lineage_and_wrong_scope_share_error_without_mutation(tmp_path, case):
    journal, original = seeded(tmp_path)
    overrides = {case: "another"} if case in SCOPE else {"memory_id": 10 if case == "lineage" else 999}
    request, digest = intent(original, **overrides)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptMemoryNotFound, match="Memory is not available in the requested scope"):
        retire(journal, request=request, digest=digest)
    assert journal.read_snapshot() == before
    assert journal.lookup_receipt(request["scope"], "retire", digest) is None


def test_historical_replay_precedes_current_state_after_later_removal(tmp_path):
    journal, original = seeded(tmp_path)
    receipt = retire(journal, original)
    revision, payload = journal.read_snapshot()
    payload["items"].pop(0)
    journal.save(payload, expected_revision=revision)
    frozen = copy.deepcopy(receipt)
    receipt["result"]["memory"]["text"] = "Caller mutated its detached copy"
    assert retire(journal, original) == frozen
    assert journal.read_snapshot()[0] == revision + 1


@pytest.mark.parametrize("state", [{"memory_state": "deleted"}, {"lifecycle_state": " DELETED "},
                                    {"retirement_decision": {"source": "preexisting"}}])
def test_new_key_cannot_overwrite_retirement_state_or_decision(tmp_path, state):
    journal, original = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"].update(state)
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        retire(journal, payload["items"][0])
    assert journal.read_snapshot() == before


def test_new_key_cannot_retire_already_retired_record(tmp_path):
    journal, original = seeded(tmp_path)
    retired = retire(journal, original)["result"]["memory"]
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        retire(journal, retired, key="new-retirement")
    assert journal.read_snapshot() == before


def test_target_change_rejects_stale_full_record_digest(tmp_path):
    journal, original = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"]["typed"][0] = 1
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict, match="changed"):
        retire(journal, original)
    assert journal.read_snapshot() == before


def test_shared_key_namespace_rejects_ingestion_and_changed_retirement_intents(tmp_path):
    journal, original = seeded(tmp_path)
    with pytest.raises(IdempotencyConflict):
        retire(journal, original, key="ingested")
    receipt = retire(journal, original)
    request, digest = intent(original, reason="Different operator decision")
    with pytest.raises(IdempotencyConflict):
        retire(journal, request=request, digest=digest)
    _, ingestion_digest = canonical_request("Another memory", **SCOPE)
    with pytest.raises(IdempotencyConflict):
        journal.lookup_receipt(SCOPE, "retire", ingestion_digest)
    assert journal.read_snapshot()[0] == receipt["revision"]


@pytest.mark.parametrize("change_target", [False, True])
def test_cas_rebuilds_unrelated_changes_but_never_accepts_changed_target(tmp_path, monkeypatch, change_target):
    journal, original = seeded(tmp_path)
    save = journal.save_with_receipt
    attempts = []
    def race(payload, **kwargs):
        attempts.append(kwargs["expected_revision"])
        if len(attempts) == 1:
            other = JournalStateStore(journal.path)
            revision, current = other.read_snapshot()
            current["items"][0 if change_target else 1]["text"] = "Concurrent update"
            other.save(current, expected_revision=revision)
        return save(payload, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", race)
    if change_target:
        with pytest.raises(ReceiptLifecycleConflict):
            retire(journal, original)
        assert attempts == [1]
        assert journal.lookup_receipt(SCOPE, "retire", intent(original)[1]) is None
    else:
        assert retire(journal, original)["revision"] == 3
        assert attempts == [1, 2]
        assert journal.load()["items"][1]["text"] == "Concurrent update"


def test_three_cas_conflicts_stop_without_receipt(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    calls = []
    def conflict(*args, **kwargs):
        calls.append(kwargs["expected_revision"])
        raise RevisionConflict("Competing writer")
    monkeypatch.setattr(journal, "save_with_receipt", conflict)
    with pytest.raises(RevisionConflict, match="three"):
        retire(journal, original)
    assert calls == [1] * 3
    assert journal.lookup_receipt(SCOPE, "retire", intent(original)[1]) is None
    assert journal.load()["items"][0] == original


@pytest.mark.parametrize("point", RECEIPT_FAULT_POINTS)
def test_fault_reconciliation_never_invents_success_or_reverses_commit(tmp_path, point):
    journal, original = seeded(tmp_path)
    def fail(here):
        if here == point:
            raise OSError("Injected storage failure")
    journal._fault_hook = fail
    if point == "after_commit":
        receipt = retire(journal, original)
        assert receipt["revision"] == 2
    else:
        with pytest.raises(ReceiptCommitRejected):
            retire(journal, original)
        assert journal.load()["items"][0] == original
        assert journal.lookup_receipt(SCOPE, "retire", intent(original)[1]) is None
    journal._fault_hook = lambda _: None
    receipt = retire(journal, original)
    assert receipt["revision"] == journal.read_snapshot()[0] == 2


def test_unavailable_reconciliation_reports_uncertain_then_recovers(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    lookup, save = journal.lookup_receipt, journal.save_with_receipt
    attempted = False
    def lost_ack(*args, **kwargs):
        nonlocal attempted
        result = save(*args, **kwargs)
        assert result["revision"] == 2
        attempted = True
        raise OSError("Lost commit acknowledgement")
    def unavailable(*args, **kwargs):
        if attempted:
            raise OSError("Authority temporarily unavailable")
        return lookup(*args, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", lost_ack)
    monkeypatch.setattr(journal, "lookup_receipt", unavailable)
    with pytest.raises(ReceiptCommitUncertain):
        retire(journal, original)
    reopened = JournalStateStore(journal.path)
    assert retire(reopened, original)["revision"] == 2
    assert reopened.read_snapshot()[0] == 2


def test_hydration_and_telemetry_failures_do_not_revoke_durable_success(tmp_path):
    journal, original = seeded(tmp_path)
    errors = []
    def hydrate(*args):
        raise RuntimeError("Cache cannot hydrate")
    def degraded(exc):
        errors.append(str(exc))
        raise RuntimeError("Telemetry also unavailable")
    receipt = retire(journal, original, hydrate=hydrate, degraded=degraded)
    assert errors == ["Cache cannot hydrate"]
    assert retire(journal, original) == receipt


def test_sqlite_disk_quota_rolls_back_retirement_and_preserves_ingestion_receipt(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    ingestion = journal.lookup_receipt(SCOPE, "ingested", "a" * 64)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"]["large_provenance"] = "x" * 200_000
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    connect = journal._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(journal, "_connect", limited)
    with pytest.raises(ReceiptCommitRejected) as error:
        retire(journal, payload["items"][0])
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert "full" in str(error.value.__cause__).lower()
    reopened = JournalStateStore(journal.path)
    assert reopened.read_snapshot() == before
    assert reopened.lookup_receipt(SCOPE, "ingested", "a" * 64) == ingestion
    assert reopened.lookup_receipt(SCOPE, "retire", intent(payload["items"][0])[1]) is None


def test_retirement_serialization_failure_cannot_acknowledge_or_modify_memory(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    encode = journal._encode
    def fail_new_payload(payload):
        if isinstance(payload, dict) and any(
            item.get("meta", {}).get("memory_state") == "deleted"
            for item in payload.get("items", [])
        ):
            raise TypeError("Injected interrupted serialization")
        return encode(payload)
    monkeypatch.setattr(journal, "_encode", fail_new_payload)
    with pytest.raises(ReceiptCommitRejected) as error:
        retire(journal, original)
    assert isinstance(error.value.__cause__, TypeError)
    assert journal.read_snapshot() == before
    assert journal.lookup_receipt(SCOPE, "retire", intent(original)[1]) is None


@pytest.mark.parametrize("key", ["", " ", "é" * 129, 1, True, None])
def test_invalid_idempotency_keys_cannot_modify_authority(tmp_path, key):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    with pytest.raises(ValueError):
        retire(journal, original, key=key)
    assert journal.read_snapshot() == before


def test_schema_one_never_implicitly_upgrades(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3")
    with pytest.raises(JournalSchemaError):
        retire(journal, record())
    assert journal.schema_version == 1 and journal.read_snapshot()[0] == 0
