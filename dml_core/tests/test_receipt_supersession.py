"""Two-record supersession preconditions and atomic historical outcomes."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import sqlite3
import threading

import pytest

from daystrom_dml.journal import (
    IdempotencyConflict, JournalSchemaError, JournalStateStore,
    RECEIPT_FAULT_POINTS, RevisionConflict,
)
from daystrom_dml.services.receipt_ingestion import ReceiptCommitRejected, ReceiptCommitUncertain
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from daystrom_dml.services.receipt_supersession import (
    canonical_supersession_request, supersede_receipted,
)
from test_receipt_retirement import SCOPE, record, seeded as retirement_seeded


def seeded(tmp_path, schema=2):
    journal, _ = retirement_seeded(tmp_path, schema)
    revision, payload = journal.read_snapshot()
    payload["items"][1]["meta"].pop("source_trust")
    payload["items"][1]["text"] = "Explicit replacement fact"
    third = record(2)
    third["meta"].pop("source_trust")
    payload["items"].append(third)
    journal.save(payload, expected_revision=revision)
    return journal, copy.deepcopy(payload["items"][0]), copy.deepcopy(payload["items"][1])


def intent(source, replacement, **overrides):
    return canonical_supersession_request(**{"memory_id": source["id"],
        "replacement_memory_id": replacement["id"],
        "expected_memory_digest": memory_digest(source),
        "expected_replacement_digest": memory_digest(replacement),
        "reason": "Owner chose the corrected fact", **SCOPE, **overrides})


def supersede(journal, source=None, replacement=None, *, request=None, digest=None,
              key="supersede", **callbacks):
    if request is None:
        request, digest = intent(source, replacement)
    return supersede_receipted(journal, request=request, request_digest=digest, key=key,
        hydrate=callbacks.get("hydrate", lambda *_: None),
        degraded=callbacks.get("degraded", lambda _: None))


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_only_source_lifecycle_changes_and_replacement_remains_byte_exact(tmp_path, schema):
    journal, source, replacement = seeded(tmp_path, schema)
    revision, before = journal.read_snapshot()
    hydrated = []
    receipt = supersede(journal, source, replacement, hydrate=lambda *args: hydrated.append(args))
    after_revision, after = journal.read_snapshot()
    expected = copy.deepcopy(before)
    expected["items"][0]["meta"].update(memory_state="superseded", superseded_by=1,
        supersession_decision={"schema_version": "dml-supersession-decision-v1",
            "prior_memory_digest": memory_digest(source), "replacement_memory_id": 1,
            "replacement_memory_digest": memory_digest(replacement),
            "reason": "Owner chose the corrected fact"})
    assert json.dumps(after, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert memory_digest(after["items"][1]) == memory_digest(replacement)
    assert receipt["result"]["memory"] == expected["items"][0]
    assert receipt["revision"] == after_revision == revision + 1
    assert hydrated == [(after_revision, after)]
    assert journal.schema_version == schema
    assert journal.decisions()[-1]["operation"] == "supersede-receipt-v1"
    if schema in (3, 4):
        event = journal.outbox_events()["events"][-1]
        assert event["operation"] == "supersede-receipt-v1" and event["state"] == after
    assert supersede(JournalStateStore(journal.path), source, replacement) == receipt


@pytest.mark.parametrize("overrides", [
    {"memory_id": True}, {"memory_id": -1}, {"memory_id": 0.0},
    {"replacement_memory_id": True}, {"replacement_memory_id": -1},
    {"replacement_memory_id": 1.0}, {"replacement_memory_id": 0},
    {"expected_memory_digest": "A" * 64}, {"expected_replacement_digest": "A" * 64},
    {"expected_replacement_digest": None}, {"expected_replacement_digest": "0" * 63},
    {"reason": " "}, {"reason": "é" * 513}, {"reason": True},
    {"tenant_id": None}, {"client_id": True}, {"session_id": "é" * 129},
])
def test_canonical_request_rejects_ambiguous_inputs(overrides):
    with pytest.raises(ValueError):
        intent(record(), record(1), **overrides)


@pytest.mark.parametrize("damage", ["digest", "schema", "extra", "scope", "bool-id", "cycle", "link"])
def test_invalid_direct_requests_are_rejected_before_storage(tmp_path, monkeypatch, damage):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    if damage == "digest":
        digest = "b" * 64
    elif damage == "schema":
        request["schema_version"] = "dml-supersede-request-v2"
    elif damage == "extra":
        request["unexpected"] = True
    elif damage == "scope":
        request["scope"].pop("client_id")
    elif damage == "bool-id":
        request["replacement_memory_id"] = True
    elif damage == "link":
        request["replacement_memory_id"] = 2
    else:
        request["scope"]["tenant_id"] = request
    def no_storage(*args, **kwargs):
        pytest.fail("Malformed request accessed storage")
    monkeypatch.setattr(journal, "lookup_receipt", no_storage)
    with pytest.raises(ValueError):
        supersede(journal, request=request, digest=digest)


@pytest.mark.parametrize("which", ["memory_id", "replacement_memory_id"])
@pytest.mark.parametrize("case", ["missing", "lineage", *SCOPE])
def test_both_records_require_exact_full_scope_without_disclosing_presence(tmp_path, which, case):
    journal, source, replacement = seeded(tmp_path)
    if case in SCOPE:
        revision, payload = journal.read_snapshot()
        payload["items"][0 if which == "memory_id" else 1]["meta"][case] = "other"
        journal.save(payload, expected_revision=revision)
        request, digest = intent(source, replacement)
    else:
        request, digest = intent(source, replacement, **{which: 10 if case == "lineage" else 999})
    before = journal.read_snapshot()
    with pytest.raises(ReceiptMemoryNotFound, match="Memory is not available in the requested scope"):
        supersede(journal, request=request, digest=digest)
    assert journal.read_snapshot() == before
    assert journal.lookup_receipt(SCOPE, "supersede", digest) is None


@pytest.mark.parametrize("state", [
    {"memory_state": "deleted"}, {"lifecycle_state": "retired"},
    {"memory_state": "active", "lifecycle_state": " SUPERSEDED "},
    {"retirement_decision": None}, {"supersession_decision": {}}, {"superseded_by": 0},
])
def test_existing_source_decisions_cannot_be_overwritten(tmp_path, state):
    journal, _, replacement = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"].update(state)
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        supersede(journal, payload["items"][0], replacement)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("state", [
    {"memory_state": "deleted"}, {"lifecycle_state": "retired"},
    {"memory_state": "active", "lifecycle_state": "expired"},
    {"memory_state": "active", "lifecycle_state": "quarantined"},
    {"memory_state": "suppressed"}, {"namespace": " Quarantine "},
    {"source_trust": "untrusted"}, {"expires_at": 100}, {"expires_at": True},
    {"expires_at": "future"}, {"superseded_by": False},
    {"retirement_decision": None}, {"supersession_decision": None},
])
def test_replacement_cannot_bypass_retrieval_eligibility(tmp_path, state):
    journal, source, _ = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][1]["meta"].update(state)
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict, match="eligible"):
        supersede(journal, source, payload["items"][1])
    assert journal.read_snapshot() == before


def test_chains_are_explicit_and_historical_retry_survives_target_removal(tmp_path):
    journal, source, replacement = seeded(tmp_path)
    first = supersede(journal, source, replacement)
    third = journal.load()["items"][2]
    second = supersede(journal, replacement, third, key="second-link")
    state = journal.load()
    assert state["items"][0]["meta"]["superseded_by"] == 1
    assert state["items"][1]["meta"]["superseded_by"] == 2
    assert memory_digest(state["items"][2]) == memory_digest(third)
    with pytest.raises(ReceiptLifecycleConflict):
        supersede(journal, state["items"][0], third, key="rewrite-chain")
    state["items"] = [state["items"][0], state["items"][2]]
    journal.save(state, expected_revision=second["revision"])
    assert supersede(journal, source, replacement) == first


def test_concurrent_opposing_links_commit_only_one_edge(tmp_path):
    journal, _, _ = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"].pop("source_trust")
    journal.save(payload, expected_revision=revision)
    source, replacement = payload["items"][:2]
    journals = [JournalStateStore(journal.path), JournalStateStore(journal.path)]
    barrier = threading.Barrier(2)
    def link(index):
        left, right = (source, replacement) if index == 0 else (replacement, source)
        barrier.wait(timeout=10)
        try:
            return supersede(journals[index], left, right, key=f"opposing-{index}")
        except ReceiptLifecycleConflict as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(link, range(2)))
    receipts = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    assert len(receipts) == 1
    assert sum(isinstance(outcome, ReceiptLifecycleConflict) for outcome in outcomes) == 1
    final_revision, final = journal.read_snapshot()
    assert final_revision == revision + 2
    edges = [(item["id"], item["meta"]["superseded_by"])
             for item in final["items"] if "superseded_by" in item["meta"]]
    assert edges in [[(0, 1)], [(1, 0)]]
    assert receipts[0]["result"]["memory"]["id"] == edges[0][0]


@pytest.mark.parametrize("raced_record", [0, 1, 2])
def test_cas_revalidates_both_records_and_preserves_unrelated_writes(tmp_path, monkeypatch, raced_record):
    journal, source, replacement = seeded(tmp_path)
    initial_revision = journal.read_snapshot()[0]
    save, attempts = journal.save_with_receipt, []
    def race(payload, **kwargs):
        attempts.append(kwargs["expected_revision"])
        if len(attempts) == 1:
            other = JournalStateStore(journal.path)
            revision, current = other.read_snapshot()
            current["items"][raced_record]["meta"]["concurrent"] = True
            other.save(current, expected_revision=revision)
        return save(payload, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", race)
    if raced_record < 2:
        with pytest.raises(ReceiptLifecycleConflict, match="changed"):
            supersede(journal, source, replacement)
        assert attempts == [initial_revision]
        assert journal.lookup_receipt(SCOPE, "supersede", intent(source, replacement)[1]) is None
    else:
        assert supersede(journal, source, replacement)["revision"] == initial_revision + 2
        assert attempts == [initial_revision, initial_revision + 1]
    assert journal.load()["items"][raced_record]["meta"]["concurrent"] is True


def test_three_conflicts_stop_without_a_receipt(tmp_path, monkeypatch):
    journal, source, replacement = seeded(tmp_path)
    before = journal.read_snapshot()
    attempts = []
    def conflict(*args, **kwargs):
        attempts.append(kwargs["expected_revision"])
        raise RevisionConflict("Competing write")
    monkeypatch.setattr(journal, "save_with_receipt", conflict)
    with pytest.raises(RevisionConflict, match="three"):
        supersede(journal, source, replacement)
    assert attempts == [before[0]] * 3 and journal.read_snapshot() == before


def test_idempotency_namespace_rejects_ingestion_and_changed_link_intents(tmp_path):
    journal, source, replacement = seeded(tmp_path)
    with pytest.raises(IdempotencyConflict):
        supersede(journal, source, replacement, key="ingested")
    receipt = supersede(journal, source, replacement)
    for overrides in ({"reason": "Different decision"}, {"replacement_memory_id": 2}):
        request, digest = intent(source, replacement, **overrides)
        with pytest.raises(IdempotencyConflict):
            supersede(journal, request=request, digest=digest)
    assert journal.read_snapshot()[0] == receipt["revision"]


@pytest.mark.parametrize("point", RECEIPT_FAULT_POINTS)
def test_faults_reconcile_without_reversing_or_inventing_supersession(tmp_path, point):
    journal, source, replacement = seeded(tmp_path)
    before = journal.read_snapshot()
    def fail(here):
        if here == point:
            raise OSError("Injected storage failure")
    journal._fault_hook = fail
    if point == "after_commit":
        assert supersede(journal, source, replacement)["revision"] == before[0] + 1
    else:
        with pytest.raises(ReceiptCommitRejected):
            supersede(journal, source, replacement)
        assert journal.read_snapshot() == before
        assert journal.lookup_receipt(SCOPE, "supersede", intent(source, replacement)[1]) is None
    journal._fault_hook = lambda _: None
    assert supersede(journal, source, replacement)["revision"] == before[0] + 1
    assert memory_digest(journal.load()["items"][1]) == memory_digest(replacement)


def test_uncertain_ack_recovers_from_durable_receipt(tmp_path, monkeypatch):
    journal, source, replacement = seeded(tmp_path)
    lookup, save = journal.lookup_receipt, journal.save_with_receipt
    attempted = False
    def lost_ack(*args, **kwargs):
        nonlocal attempted
        save(*args, **kwargs)
        attempted = True
        raise OSError("Lost acknowledgement")
    def unavailable(*args, **kwargs):
        if attempted:
            raise OSError("Authority temporarily unavailable")
        return lookup(*args, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", lost_ack)
    monkeypatch.setattr(journal, "lookup_receipt", unavailable)
    with pytest.raises(ReceiptCommitUncertain):
        supersede(journal, source, replacement)
    reopened = JournalStateStore(journal.path)
    assert supersede(reopened, source, replacement)["revision"] == reopened.read_snapshot()[0] == 3


def test_hydration_and_observability_failures_preserve_success(tmp_path):
    journal, source, replacement = seeded(tmp_path)
    errors = []
    def hydrate(*args):
        raise RuntimeError("Cache cannot hydrate")
    def degraded(exc):
        errors.append(str(exc))
        raise RuntimeError("Telemetry unavailable")
    receipt = supersede(journal, source, replacement, hydrate=hydrate, degraded=degraded)
    assert errors == ["Cache cannot hydrate"]
    assert supersede(journal, source, replacement) == receipt


def test_sqlite_disk_quota_rolls_back_the_link_and_preserves_old_receipts(tmp_path, monkeypatch):
    journal, _, replacement = seeded(tmp_path)
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
        supersede(journal, payload["items"][0], replacement)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert "full" in str(error.value.__cause__).lower()
    reopened = JournalStateStore(journal.path)
    assert reopened.read_snapshot() == before
    assert reopened.lookup_receipt(SCOPE, "ingested", "a" * 64) == ingestion


def test_serialization_failure_cannot_modify_or_acknowledge_link(tmp_path, monkeypatch):
    journal, source, replacement = seeded(tmp_path)
    before = journal.read_snapshot()
    encode = journal._encode
    def fail_link(payload):
        if isinstance(payload, dict) and any("supersession_decision" in item.get("meta", {})
                                            for item in payload.get("items", [])):
            raise TypeError("Interrupted serialization")
        return encode(payload)
    monkeypatch.setattr(journal, "_encode", fail_link)
    with pytest.raises(ReceiptCommitRejected) as error:
        supersede(journal, source, replacement)
    assert isinstance(error.value.__cause__, TypeError)
    assert journal.read_snapshot() == before


def test_schema_one_never_implicitly_upgrades(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3")
    with pytest.raises(JournalSchemaError):
        supersede(journal, record(), record(1))
    assert journal.schema_version == 1 and journal.read_snapshot()[0] == 0
