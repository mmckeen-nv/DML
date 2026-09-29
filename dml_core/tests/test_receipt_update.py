"""Content-only receipts: scoped preconditions, embedding races and durability."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import sqlite3
import threading

import numpy as np
import pytest

from daystrom_dml.journal import (
    IdempotencyConflict, JournalSchemaError, JournalStateStore,
    RECEIPT_FAULT_POINTS, RevisionConflict,
)
from daystrom_dml.services.receipt_ingestion import (
    ReceiptCommitRejected, ReceiptCommitUncertain, ReceiptEmbeddingError,
    ReceiptEmbeddingCompatibilityError,
)
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from daystrom_dml.services.receipt_update import canonical_update_request, update_receipted
from test_receipt_retirement import SCOPE, record, seeded as retirement_seeded


IDENTITY = {"backend": "fixture", "revision": "revision-1", "model": None, "mode": "native"}
UPDATED_TEXT = "Explicitly corrected content"
UPDATED_VECTOR = [0.0, 1.0, 0.0, 0.0]


def seeded(tmp_path, schema=2):
    journal, original = retirement_seeded(tmp_path, schema)
    revision, payload = journal.read_snapshot()
    payload["embedding_contract"] = {"schema_version": "dml-embedding-contract-v1",
        "identity": copy.deepcopy(IDENTITY), "dimension": 4}
    journal.save(payload, expected_revision=revision)
    return journal, original


def intent(original, **overrides):
    return canonical_update_request(**{"memory_id": original["id"], "text": UPDATED_TEXT,
        "expected_memory_digest": memory_digest(original), "reason": "Owner corrected the content",
        **SCOPE, **overrides})


def update(journal, original=None, *, request=None, digest=None, key="update", **callbacks):
    if request is None:
        request, digest = intent(original)
    return update_receipted(journal, request=request, request_digest=digest, key=key,
        embed=callbacks.get("embed", lambda _: np.array(UPDATED_VECTOR, dtype=np.float32)),
        embedding_space=callbacks.get("embedding_space", lambda: copy.deepcopy(IDENTITY)),
        hydrate=callbacks.get("hydrate", lambda *_: None),
        degraded=callbacks.get("degraded", lambda _: None))


def no_backend(*args):
    pytest.fail("Invalid or already acknowledged update consulted a backend")


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_only_content_and_latest_decision_change_atomically(tmp_path, schema):
    journal, original = seeded(tmp_path, schema)
    revision, before = journal.read_snapshot()
    hydrated = []
    receipt = update(journal, original, hydrate=lambda *args: hydrated.append(args))
    after_revision, after = journal.read_snapshot()
    expected = copy.deepcopy(before)
    expected["items"][0]["text"] = UPDATED_TEXT
    expected["items"][0]["embedding"] = UPDATED_VECTOR
    expected["items"][0]["meta"]["content_update_decision"] = {
        "schema_version": "dml-content-update-decision-v1",
        "prior_memory_digest": memory_digest(original), "reason": "Owner corrected the content"}
    assert json.dumps(after, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert receipt["result"]["memory"] == after["items"][0]
    assert receipt["revision"] == after_revision == revision + 1
    assert hydrated == [(after_revision, after)]
    assert journal.schema_version == schema
    assert journal.decisions()[-1]["operation"] == "update-receipt-v1"
    if schema in (3, 4):
        event = journal.outbox_events()["events"][-1]
        assert event["operation"] == "update-receipt-v1" and event["state"] == after
    assert update(JournalStateStore(journal.path), original,
        embed=no_backend, embedding_space=no_backend) == receipt


@pytest.mark.parametrize("overrides", [
    {"memory_id": True}, {"memory_id": -1}, {"memory_id": 0.0},
    {"text": None}, {"text": " "}, {"text": 12}, {"text": ["text"]},
    {"text": "x" * (1024 * 1024)},
    {"expected_memory_digest": "A" * 64}, {"expected_memory_digest": None},
    {"expected_memory_digest": "0" * 63}, {"expected_memory_digest": 1},
    {"reason": " "}, {"reason": "é" * 513}, {"reason": True},
    {"tenant_id": None}, {"client_id": True}, {"session_id": "é" * 129},
])
def test_canonical_intent_rejects_ambiguous_inputs(overrides):
    with pytest.raises(ValueError):
        intent(record(), **overrides)


@pytest.mark.parametrize("damage", [
    "digest", "schema", "extra", "scope", "scope-type", "bool-id", "cycle",
    "text", "reason", "nan", "non-object", "non-string-key",
])
def test_malformed_direct_intents_fail_before_authority_or_backend(tmp_path, monkeypatch, damage):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    if damage == "digest":
        digest = "b" * 64
    elif damage == "schema":
        request["schema_version"] = "dml-update-request-v2"
    elif damage == "extra":
        request["unexpected"] = True
    elif damage == "scope":
        request["scope"].pop("client_id")
    elif damage == "scope-type":
        request["scope"] = []
    elif damage == "bool-id":
        request["memory_id"] = True
    elif damage == "cycle":
        request["scope"]["tenant_id"] = request
    elif damage == "text":
        request["text"] = "Other text"
    elif damage == "reason":
        request["reason"] = "Other reason"
    elif damage == "nan":
        request["text"] = float("nan")
    elif damage == "non-string-key":
        request[1] = "ambiguous"
    else:
        request = []
    monkeypatch.setattr(journal, "lookup_receipt", no_backend)
    with pytest.raises(ValueError):
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)


@pytest.mark.parametrize("case", ["missing", "lineage", *SCOPE, "missing-null-scope"])
def test_exact_live_scope_precedes_record_digest_and_embedding(tmp_path, case):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    if case in SCOPE or case == "missing-null-scope":
        revision, payload = journal.read_snapshot()
        if case == "missing-null-scope":
            payload["items"][0]["meta"].pop("client_id")
            request, digest = intent(payload["items"][0], client_id=None)
        else:
            payload["items"][0]["meta"][case] = "other"
        journal.save(payload, expected_revision=revision)
    else:
        request, digest = intent(original, memory_id=10 if case == "lineage" else 999)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptMemoryNotFound, match="Memory is not available in the requested scope"):
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("state", [
    {"memory_state": "deleted"}, {"memory_state": "retired"}, {"memory_state": "superseded"},
    {"lifecycle_state": "retired"}, {"memory_state": "active", "lifecycle_state": " SUPERSEDED "},
    {"retirement_decision": None}, {"supersession_decision": {}},
    {"superseded_by": 0}, {"superseded_by": False},
])
def test_terminal_lifecycle_cannot_be_revived_or_overwritten(tmp_path, state):
    journal, _ = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"].update(state)
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        update(journal, payload["items"][0], embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("decision", [None, {}, "old", [],
    {"schema_version": "dml-content-update-decision-v2", "prior_memory_digest": "a" * 64, "reason": "Prior"},
    {"schema_version": "dml-content-update-decision-v1", "prior_memory_digest": "A" * 64, "reason": "Prior"},
    {"schema_version": "dml-content-update-decision-v1", "prior_memory_digest": "a" * 64, "reason": " "},
    {"schema_version": "dml-content-update-decision-v1", "prior_memory_digest": "a" * 64, "reason": "\ud800"},
    {"schema_version": "dml-content-update-decision-v1", "prior_memory_digest": "a" * 64, "reason": "Prior", "extra": True},
])
def test_unknown_content_provenance_is_preserved_and_requires_explicit_resolution(tmp_path, decision):
    journal, _ = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"]["content_update_decision"] = decision
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict, match="unrecognized"):
        update(journal, payload["items"][0], embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("metadata", [
    {"memory_state": "quarantined"}, {"memory_state": "suppressed"},
    {"memory_state": "active", "lifecycle_state": "expired"},
    {"namespace": "quarantine"}, {"expires_at": 1}, {"expires_at": "unknown"},
    {"source_trust": "untrusted", "suppressed": True},
    {"superseded_by": None, "retention": {"policy": [True, 1, 1.0]}},
])
def test_updates_preserve_suppression_trust_expiry_and_unknown_metadata(tmp_path, metadata):
    journal, _ = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    payload["items"][0]["meta"].update(metadata)
    journal.save(payload, expected_revision=revision)
    expected_meta = copy.deepcopy(payload["items"][0]["meta"])
    receipt = update(journal, payload["items"][0])
    actual_meta = copy.deepcopy(receipt["result"]["memory"]["meta"])
    actual_meta.pop("content_update_decision")
    assert json.dumps(actual_meta, sort_keys=True) == json.dumps(expected_meta, sort_keys=True)


def test_stale_and_same_text_new_requests_do_not_reembed_or_create_revisions(tmp_path):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    request, digest = intent(original, text=original["text"])
    with pytest.raises(ReceiptLifecycleConflict, match="unchanged"):
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    request, digest = intent(original, expected_memory_digest="a" * 64)
    with pytest.raises(ReceiptLifecycleConflict, match="changed"):
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


def test_repeated_updates_replace_latest_decision_and_replay_all_historical_receipts(tmp_path):
    journal, original = seeded(tmp_path)
    first = update(journal, original)
    new_request, new_digest = intent(first["result"]["memory"], text="A later precise correction")
    second = update(journal, request=new_request, digest=new_digest, key="second")
    assert second["result"]["memory"]["meta"]["content_update_decision"]["prior_memory_digest"] == memory_digest(first["result"]["memory"])
    revision, payload = journal.read_snapshot()
    payload["items"] = [item for item in payload["items"] if item["id"] != original["id"]]
    payload.pop("embedding_contract")
    journal.save(payload, expected_revision=revision)
    assert update(journal, original, embed=no_backend, embedding_space=no_backend) == first
    assert update(journal, request=new_request, digest=new_digest, key="second",
        embed=no_backend, embedding_space=no_backend) == second


def test_shared_key_namespace_rejects_ingestion_and_changed_update_intents(tmp_path):
    journal, original = seeded(tmp_path)
    with pytest.raises(IdempotencyConflict):
        update(journal, original, key="ingested", embed=no_backend, embedding_space=no_backend)
    receipt = update(journal, original)
    for override in ({"text": "Another"}, {"reason": "Other reason"}, {"memory_id": 1}):
        request, digest = intent(original, **override)
        with pytest.raises(IdempotencyConflict):
            update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot()[0] == receipt["revision"]


@pytest.mark.parametrize("damage", ["text", "id", "scope", "decision", "empty-vector", "terminal"])
def test_generic_receipts_cannot_falsely_acknowledge_the_requested_operation(tmp_path, damage):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    revision, payload = journal.read_snapshot()
    result = payload["items"][0]
    result["text"] = UPDATED_TEXT
    result["meta"]["content_update_decision"] = {
        "schema_version": "dml-content-update-decision-v1",
        "prior_memory_digest": memory_digest(original), "reason": request["reason"]}
    if damage == "text":
        result["text"] = "Unexpected text"
    elif damage == "id":
        result["id"] = 20
    elif damage == "scope":
        result["meta"]["client_id"] = "other"
    elif damage == "decision":
        result["meta"]["content_update_decision"]["reason"] = "Unexpected reason"
    elif damage == "empty-vector":
        result["embedding"] = []
    else:
        result["meta"]["retirement_decision"] = None
    # Generic journals permit application-specific operations; their receipt
    # identity alone cannot establish that the service's contract was met.
    try:
        journal.save_with_receipt(payload, scope=SCOPE, key="update", request_digest=digest,
            result={"memory": result}, expected_revision=revision, operation="generic")
    except ValueError:
        assert damage == "scope"
        return  # The generic journal's own scope validation is stricter here.
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict, match="acknowledge"):
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("vector", [[], [[1., 0.]], [float("nan")], [float("inf")],
    [1e100], [1 + 2j], [True, False], ["1", "0"], None])
def test_invalid_model_vectors_never_mutate_authority(tmp_path, vector):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptEmbeddingError):
        update(journal, original, embed=lambda _: vector)
    assert journal.read_snapshot() == before
    assert journal.lookup_receipt(SCOPE, "update", intent(original)[1]) is None


@pytest.mark.parametrize("damage", ["missing", "identity", "dimension", "lineage-vector", "typed-identity"])
def test_persisted_embedding_contract_is_checked_before_model_calls(tmp_path, damage):
    journal, original = seeded(tmp_path)
    revision, payload = journal.read_snapshot()
    declared = copy.deepcopy(IDENTITY)
    if damage == "missing":
        payload.pop("embedding_contract")
    elif damage == "identity":
        payload["embedding_contract"]["identity"]["revision"] = "other"
    elif damage == "dimension":
        payload["embedding_contract"]["dimension"] = 3
    elif damage == "lineage-vector":
        payload["lineage"][0]["embedding"] = [1.0]
    else:
        declared["model"] = {"typed": True}
        payload["embedding_contract"]["identity"]["model"] = {"typed": 1}
    journal.save(payload, expected_revision=revision)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        update(journal, original, embed=no_backend, embedding_space=lambda: declared)
    assert journal.read_snapshot() == before


def test_prepared_vector_dimension_must_match_existing_space(tmp_path):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="dimensions differ"):
        update(journal, original, embed=lambda _: [1., 0., 0.])
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("timing", ["model", "cas"])
def test_same_identity_object_mutation_is_detected_without_committing(tmp_path, monkeypatch, timing):
    journal, original = seeded(tmp_path)
    identity = copy.deepcopy(IDENTITY)
    before = journal.read_snapshot()
    def embed(_):
        if timing == "model":
            identity["revision"] = "mutated"
        return UPDATED_VECTOR
    if timing == "cas":
        def race(payload, **kwargs):
            identity["revision"] = "mutated"
            raise RevisionConflict("Configuration changed after attempted CAS")
        monkeypatch.setattr(journal, "save_with_receipt", race)
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="identity changed"):
        update(journal, original, embed=embed, embedding_space=lambda: identity)
    assert journal.read_snapshot() == before


def test_retained_vector_cannot_be_changed_by_later_callbacks(tmp_path):
    journal, original = seeded(tmp_path)
    retained = np.array(UPDATED_VECTOR, dtype=np.float32)
    prepared = False
    def embed(_):
        nonlocal prepared
        prepared = True
        return retained
    def identity():
        if prepared:
            retained[:] = [1., 0., 0., 0.]
        return IDENTITY
    receipt = update(journal, original, embed=embed, embedding_space=identity)
    assert retained.tolist() != UPDATED_VECTOR
    assert receipt["result"]["memory"]["embedding"] == UPDATED_VECTOR


def test_model_is_outside_store_ownership_and_changed_target_is_rejected(tmp_path):
    journal, original = seeded(tmp_path)
    completed = threading.Event()
    def competing_write():
        from daystrom_dml.store_lock import store_write_lock
        with store_write_lock(journal.path.parent, operation="fixture-competing", timeout_ms=1000):
            other = JournalStateStore(journal.path)
            revision, payload = other.read_snapshot()
            payload["items"][0]["meta"]["concurrent"] = True
            other.save(payload, expected_revision=revision)
        completed.set()
    def embed(_):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(competing_write)
            assert completed.wait(5), "Model callback held store ownership"
            future.result(timeout=5)
        return UPDATED_VECTOR
    with pytest.raises(ReceiptLifecycleConflict, match="changed"):
        update(journal, original, embed=embed)
    assert journal.load()["items"][0]["meta"]["concurrent"] is True
    assert journal.lookup_receipt(SCOPE, "update", intent(original)[1]) is None


@pytest.mark.parametrize("race_target", ["target", "unrelated", "contract"])
def test_each_cas_attempt_rechecks_target_and_contract_and_rebases_other_changes(tmp_path, monkeypatch, race_target):
    journal, original = seeded(tmp_path)
    initial_revision = journal.read_snapshot()[0]
    save, attempts = journal.save_with_receipt, []
    def race(payload, **kwargs):
        attempts.append(kwargs["expected_revision"])
        if len(attempts) == 1:
            other = JournalStateStore(journal.path)
            revision, current = other.read_snapshot()
            if race_target == "contract":
                current["embedding_contract"]["identity"]["revision"] = "changed"
            else:
                current["items"][0 if race_target == "target" else 1]["meta"]["concurrent"] = True
            other.save(current, expected_revision=revision)
        return save(payload, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", race)
    if race_target == "unrelated":
        assert update(journal, original)["revision"] == initial_revision + 2
        assert attempts == [initial_revision, initial_revision + 1]
        assert journal.load()["items"][1]["meta"]["concurrent"] is True
    else:
        expected = ReceiptLifecycleConflict if race_target == "target" else ReceiptEmbeddingCompatibilityError
        with pytest.raises(expected):
            update(journal, original)
        assert attempts == [initial_revision]
        assert journal.lookup_receipt(SCOPE, "update", intent(original)[1]) is None


def test_three_cas_conflicts_stop_without_acknowledging_or_mutating(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    attempts = []
    def conflict(*args, **kwargs):
        attempts.append(kwargs["expected_revision"])
        raise RevisionConflict("Competing write")
    monkeypatch.setattr(journal, "save_with_receipt", conflict)
    with pytest.raises(RevisionConflict, match="three"):
        update(journal, original)
    assert attempts == [before[0]] * 3 and journal.read_snapshot() == before


@pytest.mark.parametrize("fail_model", [True, False])
def test_competing_receipt_wins_when_model_fails_or_configuration_changes(tmp_path, fail_model):
    journal, original = seeded(tmp_path)
    identity = copy.deepcopy(IDENTITY)
    committed = []
    def embed(_):
        committed.append(update(JournalStateStore(journal.path), original))
        identity["revision"] = "unavailable"
        if fail_model:
            raise OSError("Model temporarily unavailable")
        return UPDATED_VECTOR
    receipt = update(journal, original, embed=embed, embedding_space=lambda: identity)
    assert receipt == committed[0]
    assert journal.read_snapshot()[0] == receipt["revision"]


def test_failed_model_reconciliation_preserves_a_competing_intent_conflict(tmp_path):
    journal, original = seeded(tmp_path)
    other_request, other_digest = intent(original, text="Competing corrected text")
    committed = []
    def embed(_):
        committed.append(update(JournalStateStore(journal.path), request=other_request, digest=other_digest))
        raise OSError("Model unavailable")
    with pytest.raises(IdempotencyConflict):
        update(journal, original, embed=embed)
    assert journal.read_snapshot()[0] == committed[0]["revision"]
    assert journal.load()["items"][0]["text"] == "Competing corrected text"


def test_unreadable_guard_reconciliation_reports_uncertain_outcome(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    request, digest = intent(original, expected_memory_digest="a" * 64)
    lookup = journal.lookup_receipt
    lookups = 0
    def unavailable(**kwargs):
        nonlocal lookups
        lookups += 1
        if lookups > 1:
            raise OSError("Authority unavailable during outcome reconciliation")
        return lookup(**kwargs)
    monkeypatch.setattr(journal, "lookup_receipt", unavailable)
    with pytest.raises(ReceiptCommitUncertain) as error:
        update(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert isinstance(error.value.__cause__, OSError)
    assert JournalStateStore(journal.path).lookup_receipt(SCOPE, "update", digest) is None


def test_caller_intent_is_frozen_before_storage_and_model_callbacks(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    request, digest = intent(original)
    lookup = journal.lookup_receipt
    def mutate(**kwargs):
        request["text"] = "Changed by caller"
        request["scope"]["client_id"] = "other"
        return lookup(**kwargs)
    monkeypatch.setattr(journal, "lookup_receipt", mutate)
    receipt = update(journal, request=request, digest=digest)
    assert receipt["result"]["memory"]["text"] == UPDATED_TEXT
    assert receipt["result"]["memory"]["meta"]["client_id"] == SCOPE["client_id"]


@pytest.mark.parametrize("point", RECEIPT_FAULT_POINTS)
def test_storage_faults_reconcile_without_reversing_or_inventing_updates(tmp_path, point):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    def fail(here):
        if here == point:
            raise OSError("Injected storage failure")
    journal._fault_hook = fail
    if point == "after_commit":
        assert update(journal, original)["revision"] == before[0] + 1
    else:
        with pytest.raises(ReceiptCommitRejected):
            update(journal, original)
        assert journal.read_snapshot() == before
        assert journal.lookup_receipt(SCOPE, "update", intent(original)[1]) is None
    journal._fault_hook = lambda _: None
    assert update(journal, original)["revision"] == before[0] + 1


def test_uncertain_commit_recovers_its_historical_receipt_without_reembedding(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    lookup, save = journal.lookup_receipt, journal.save_with_receipt
    attempted = False
    def lost_ack(*args, **kwargs):
        nonlocal attempted
        save(*args, **kwargs)
        attempted = True
        raise OSError("Lost acknowledgement")
    def unavailable(*args, **kwargs):
        if attempted:
            raise OSError("Authority unavailable")
        return lookup(*args, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", lost_ack)
    monkeypatch.setattr(journal, "lookup_receipt", unavailable)
    with pytest.raises(ReceiptCommitUncertain):
        update(journal, original)
    reopened = JournalStateStore(journal.path)
    receipt = update(reopened, original, embed=no_backend, embedding_space=no_backend)
    assert receipt["revision"] == reopened.read_snapshot()[0]


def test_hydration_and_observability_failures_do_not_revoke_a_commit(tmp_path):
    journal, original = seeded(tmp_path)
    errors = []
    def hydrate(*args):
        raise RuntimeError("Cache cannot hydrate")
    def degraded(exc):
        errors.append(str(exc))
        raise RuntimeError("Telemetry unavailable")
    receipt = update(journal, original, hydrate=hydrate, degraded=degraded)
    assert errors == ["Cache cannot hydrate"]
    assert update(journal, original, embed=no_backend, embedding_space=no_backend) == receipt


def test_sqlite_disk_quota_rolls_back_content_and_preserves_old_receipts(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    ingestion = journal.lookup_receipt(SCOPE, "ingested", "a" * 64)
    connect = journal._connect
    @contextmanager
    def limited():
        with connect() as connection:
            pages = connection.execute("PRAGMA page_count").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={pages}")
            yield connection
    monkeypatch.setattr(journal, "_connect", limited)
    request, digest = intent(original, text="x" * 200_000)
    with pytest.raises(ReceiptCommitRejected) as error:
        update(journal, request=request, digest=digest)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert "full" in str(error.value.__cause__).lower()
    reopened = JournalStateStore(journal.path)
    assert reopened.read_snapshot() == before
    assert reopened.lookup_receipt(SCOPE, "ingested", "a" * 64) == ingestion


def test_serialization_interruption_cannot_modify_or_acknowledge_content(tmp_path, monkeypatch):
    journal, original = seeded(tmp_path)
    before = journal.read_snapshot()
    encode = journal._encode
    def fail_update(payload):
        if isinstance(payload, dict) and any("content_update_decision" in item.get("meta", {})
                                            for item in payload.get("items", [])):
            raise TypeError("Interrupted serialization")
        return encode(payload)
    monkeypatch.setattr(journal, "_encode", fail_update)
    with pytest.raises(ReceiptCommitRejected) as error:
        update(journal, original)
    assert isinstance(error.value.__cause__, TypeError)
    assert journal.read_snapshot() == before


def test_schema_one_is_never_upgraded_implicitly(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3")
    with pytest.raises(JournalSchemaError):
        update(journal, record(), embed=no_backend, embedding_space=no_backend)
    assert journal.schema_version == 1 and journal.read_snapshot()[0] == 0
