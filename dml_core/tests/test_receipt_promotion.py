"""Caller-directed snapshot derivation: provenance, races and durable outcomes."""
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
from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_ingestion import (
    ReceiptCapacityError, ReceiptCommitRejected, ReceiptCommitUncertain,
    ReceiptEmbeddingCompatibilityError, ReceiptEmbeddingError,
)
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from daystrom_dml.services.receipt_promotion import canonical_promotion_request, promote_receipted

SCOPE = dict(tenant_id="tenant", client_id="client", session_id="session", instance_id="instance")
IDENTITY = {"backend": "fixture", "revision": "revision-1", "model": None, "mode": "native"}
PROMOTED_TEXT = "The caller-confirmed summary of both sources"
PROMOTED_VECTOR = [0.0, 1.0, 0.0, 0.0]


def source(ident=0, **updates):
    record = MemoryItem(id=ident, text=f"Explicit source {ident}",
        embedding=np.array([1., 0., 0., 0.], dtype=np.float32), timestamp=123.0 + ident,
        salience=0.9 - ident / 100, fidelity=0.8 - ident / 100, level=0,
        meta={**SCOPE, "no_merge": False, "source_trust": "trusted", "source": f"source-{ident}",
              "provenance": {"typed": [True, 1, 1.0, None]}, "summary": "Obsolete cached summary",
              "unknown_policy": {"typed": [True, 1, 1.0]}}).to_dict()
    record.update(updates)
    return record


def seeded(tmp_path, schema=2):
    journal = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                                receipt_mode=True, outbox_mode=(schema == 3))
    originals = [source(0), source(1)]
    payload = {"items": originals + [source(2)], "lineage": [source(10)], "next_id": 12,
               "embedding_contract": {"schema_version": "dml-embedding-contract-v1",
                                      "identity": copy.deepcopy(IDENTITY), "dimension": 4},
               "extra": {"typed": [1, True]}}
    journal.save_with_receipt(payload, scope=SCOPE, key="ingested", request_digest="a" * 64,
        result={"memory": originals[0]}, expected_revision=0)
    if schema == 4:
        target = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(journal.path, target)
        journal = JournalStateStore(target)
    return journal, copy.deepcopy(originals)


def intent(records, **overrides):
    return canonical_promotion_request(**{
        "sources": [{"memory_id": record["id"], "expected_memory_digest": memory_digest(record)}
                    for record in records], "text": PROMOTED_TEXT,
        "reason": "Owner confirms a snapshot summary", **SCOPE, **overrides})


def promote(journal, records=None, *, request=None, digest=None, key="promotion", capacity=100, **callbacks):
    if request is None:
        request, digest = intent(records)
    return promote_receipted(journal, request=request, request_digest=digest, key=key,
        capacity=capacity, embed=callbacks.get("embed", lambda _: np.array(PROMOTED_VECTOR, dtype=np.float32)),
        embedding_space=callbacks.get("embedding_space", lambda: copy.deepcopy(IDENTITY)),
        hydrate=callbacks.get("hydrate", lambda *_: None), degraded=callbacks.get("degraded", lambda _: None))


def no_backend(*args):
    pytest.fail("Invalid or already acknowledged promotion consulted a backend")


def edited(journal, mutate):
    revision, payload = journal.read_snapshot()
    mutate(payload)
    journal.save(payload, expected_revision=revision)
    return payload["items"][:2]


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_atomic_derivation_preserves_every_source_and_has_complete_provenance(tmp_path, schema):
    journal, originals = seeded(tmp_path, schema)
    revision, before = journal.read_snapshot()
    hydrated = []
    receipt = promote(journal, list(reversed(originals)), hydrate=lambda *args: hydrated.append(args))
    after_revision, after = journal.read_snapshot()
    result = receipt["result"]["memory"]
    assert result["id"] == 12 and result["level"] == 1
    assert result["summary_of"] == result["children"] == [0, 1]
    assert result["text"] == PROMOTED_TEXT and result["embedding"] == PROMOTED_VECTOR
    for field in ("timestamp", "salience", "fidelity"):
        assert result[field] == min(record[field] for record in originals)
    expected_meta = copy.deepcopy(originals[0]["meta"])
    for key in ("source", "provenance", "summary"):
        expected_meta.pop(key)
    expected_meta["no_merge"] = True
    expected_meta["promotion_decision"] = {"schema_version": "dml-promotion-decision-v1",
        "reason": "Owner confirms a snapshot summary", "sources": [
            {"memory_digest": memory_digest(record), "memory": record} for record in originals]}
    assert result["meta"] == expected_meta
    expected = copy.deepcopy(before)
    expected["items"].append(result)
    expected["next_id"] = 13
    assert json.dumps(after, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert after_revision == receipt["revision"] == revision + 1
    assert hydrated == [(after_revision, after)] and journal.schema_version == schema
    assert journal.decisions()[-1]["operation"] == "promote-receipt-v1"
    if schema in (3, 4):
        event = journal.outbox_events()["events"][-1]
        assert event["operation"] == "promote-receipt-v1" and event["state"] == after
    assert promote(JournalStateStore(journal.path), originals, embed=no_backend, embedding_space=no_backend) == receipt


@pytest.mark.parametrize("override", [
    {"sources": []}, {"sources": None}, {"sources": ()},
    {"sources": [{"memory_id": 0, "expected_memory_digest": "a" * 64}] * 33},
    {"sources": [{"memory_id": True, "expected_memory_digest": "a" * 64}]},
    {"sources": [{"memory_id": -1, "expected_memory_digest": "a" * 64}]},
    {"sources": [{"memory_id": 1.0, "expected_memory_digest": "a" * 64}]},
    {"sources": [{"memory_id": 0, "expected_memory_digest": "A" * 64}]},
    {"sources": [{"memory_id": 0, "expected_memory_digest": "a" * 64, "extra": True}]},
    {"sources": [{"memory_id": 0, "expected_memory_digest": "a" * 64}] * 2},
    {"text": None}, {"text": " "}, {"text": True}, {"text": "x" * (1024 * 1024)},
    {"reason": " "}, {"reason": "é" * 513}, {"reason": True},
    {"tenant_id": None}, {"client_id": True}, {"session_id": "é" * 129},
], ids=[f"invalid-{index}" for index in range(20)])
def test_canonical_intent_rejects_ambiguous_inputs(override):
    with pytest.raises(ValueError):
        intent([source()], **override)


def test_source_order_has_one_canonical_intent_and_freezes_caller_objects():
    records = [source(0), source(1)]
    request, digest = intent(records)
    assert intent(list(reversed(records))) == (request, digest)
    sources = request["sources"]
    frozen, frozen_digest = intent(records, sources=sources)
    sources[0]["memory_id"] = 80
    assert frozen["sources"][0]["memory_id"] == 0 and frozen_digest == digest


@pytest.mark.parametrize("damage", ["digest", "schema", "extra", "scope", "scope-type", "order", "cycle", "text", "non-object", "non-string-key"])
def test_malformed_direct_intents_fail_before_authority_and_embedding(tmp_path, monkeypatch, damage):
    journal, originals = seeded(tmp_path)
    request, digest = intent(originals)
    if damage == "digest":
        digest = "b" * 64
    elif damage == "schema":
        request["schema_version"] = "dml-promotion-request-v2"
    elif damage == "extra":
        request["unexpected"] = True
    elif damage == "scope":
        request["scope"].pop("client_id")
    elif damage == "scope-type":
        request["scope"] = []
    elif damage == "order":
        request["sources"].reverse()
    elif damage == "cycle":
        request["sources"].append(request)
    elif damage == "text":
        request["text"] = "Other text"
    elif damage == "non-string-key":
        request[1] = "ambiguous"
    else:
        request = []
    monkeypatch.setattr(journal, "lookup_receipt", no_backend)
    with pytest.raises(ValueError):
        promote(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)


@pytest.mark.parametrize("case", ["missing", "lineage", *SCOPE, "missing-null-scope", "other-scope-stale-first"])
def test_all_source_scopes_are_checked_before_any_digest_or_model(tmp_path, case):
    journal, originals = seeded(tmp_path)
    request, digest = intent(originals)
    if case == "missing" or case == "lineage":
        request["sources"][1]["memory_id"] = 999 if case == "missing" else 10
        request, digest = intent(originals, sources=request["sources"])
    else:
        def change(payload):
            if case == "missing-null-scope":
                payload["items"][1]["meta"].pop("client_id")
            else:
                payload["items"][1]["meta"]["tenant_id" if case == "other-scope-stale-first" else case] = "other"
            if case == "other-scope-stale-first":
                payload["items"][0]["text"] = "Stale first record"
        edited(journal, change)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptMemoryNotFound, match="requested scope"):
        promote(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("metadata", [
    {"memory_state": "deleted"}, {"lifecycle_state": "retired"},
    {"memory_state": "active", "lifecycle_state": "quarantined"},
    {"memory_state": False}, {"lifecycle_state": 0}, {"memory_state": []},
    {"retirement_decision": None}, {"supersession_decision": {}}, {"superseded_by": 0},
    {"namespace": "quarantine"}, {"expires_at": 1}, {"expires_at": "unknown"},
    {"source_trust": "untrusted"}, {"source_trust": "Trusted"}, {"source_trust": None},
    {"promotion_decision": None}, {"abstracted_from": None},
    {"content_update_decision": None}, {"content_update_decision": {"schema_version": "unknown"}},
    {"no_merge": True}, {"no_merge": "false"}, {"no_merge": 0}, {"no_merge": None},
    {"merge_policy": " NEVER "},
], ids=[f"ineligible-{index}" for index in range(24)])
def test_ineligible_sources_never_prepare_or_mutate(tmp_path, metadata):
    journal, _ = seeded(tmp_path)
    originals = edited(journal, lambda payload: payload["items"][0]["meta"].update(metadata))
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        promote(journal, originals, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("damage", ["level", "summary", "children", "bool-child", "duplicate-child"])
def test_indirect_or_ambiguous_lineage_cannot_be_promoted(tmp_path, damage):
    journal, _ = seeded(tmp_path)
    values = {"level": ("level", 1), "summary": ("summary_of", [8]), "children": ("children", [8]),
              "bool-child": ("children", [False]), "duplicate-child": ("summary_of", [0, 0])}
    name, value = values[damage]
    originals = edited(journal, lambda payload: payload["items"][0].update({name: value}))
    with pytest.raises(ReceiptLifecycleConflict):
        promote(journal, originals, embed=no_backend, embedding_space=no_backend)


@pytest.mark.parametrize("key,value", [("claim_value", "conflict"), ("authority", "system"),
    ("unknown_policy", {"typed": [1, 1, 1.0]}), ("unknown_policy", {"typed": [True, 1.0, 1.0]}),
    ("source_trust", "verified"), ("expires_at", 1e20), ("no_merge", None), ("unknown", None)])
def test_unrecognized_metadata_and_authority_boundaries_are_exact(tmp_path, key, value):
    journal, _ = seeded(tmp_path)
    originals = edited(journal, lambda payload: payload["items"][1]["meta"].update({key: value}))
    with pytest.raises(ReceiptLifecycleConflict):
        promote(journal, originals, embed=no_backend, embedding_space=no_backend)


def test_singleton_keeps_append_only_policy_and_strips_old_summary(tmp_path):
    journal, _ = seeded(tmp_path)
    originals = edited(journal, lambda payload: payload["items"][0]["meta"].update(
        no_merge=True, merge_policy="never", source_trust="verified"))
    receipt = promote(journal, originals[:1])
    result = receipt["result"]["memory"]
    assert result["meta"]["no_merge"] is True and result["meta"]["merge_policy"] == "never"
    assert result["meta"]["source_trust"] == "verified" and "summary" not in result["meta"]
    assert result["meta"]["promotion_decision"]["sources"][0]["memory"] == originals[0]


def test_recognized_content_update_history_remains_in_proof_only(tmp_path):
    journal, _ = seeded(tmp_path)
    originals = edited(journal, lambda payload: payload["items"][0]["meta"].update(
        content_update_decision={"schema_version": "dml-content-update-decision-v1",
            "prior_memory_digest": "a" * 64, "reason": "Correction"}))
    result = promote(journal, originals)["result"]["memory"]
    assert "content_update_decision" not in result["meta"]
    assert result["meta"]["promotion_decision"]["sources"][0]["memory"] == originals[0]


@pytest.mark.parametrize("case", ["capacity", "invalid-capacity", "allocator", "source-proof", "final-output"])
def test_capacity_allocator_and_size_bounds_do_not_commit(tmp_path, case):
    journal, originals = seeded(tmp_path)
    capacity, text = 100, PROMOTED_TEXT
    if case == "capacity":
        capacity = 3
    elif case == "invalid-capacity":
        capacity = True
    elif case == "allocator":
        originals = edited(journal, lambda payload: payload.update(next_id=True))
    elif case == "source-proof":
        originals = edited(journal, lambda payload: payload["items"][0].update(text="x" * (1024 * 1024)))
    else:
        originals = edited(journal, lambda payload: payload["items"][0].update(text="x" * 600_000))
        text = "y" * 600_000
    before = journal.read_snapshot()
    request, digest = intent(originals, text=text)
    with pytest.raises(ReceiptCapacityError if case == "capacity" else ValueError):
        promote(journal, request=request, digest=digest, capacity=capacity,
            embed=(lambda _: PROMOTED_VECTOR) if case == "final-output" else no_backend)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("allocator", [0, 12, 90])
def test_allocator_never_reuses_a_live_or_lineage_id(tmp_path, allocator):
    journal, originals = seeded(tmp_path)
    edited(journal, lambda payload: payload.update(next_id=allocator))
    result = promote(journal, originals)["result"]["memory"]
    assert result["id"] == max(11, allocator)
    assert journal.load()["next_id"] == result["id"] + 1


def test_historical_replay_ignores_current_sources_capacity_models_and_expiry(tmp_path):
    journal, originals = seeded(tmp_path)
    receipt = promote(journal, originals)
    edited(journal, lambda payload: payload.update(items=[], lineage=[]))
    edited(journal, lambda payload: payload.pop("embedding_contract"))
    assert promote(journal, originals, capacity=0, embed=no_backend, embedding_space=no_backend,
        hydrate=no_backend, degraded=no_backend) == receipt


def test_same_key_namespace_rejects_different_operations_and_intents(tmp_path):
    journal, originals = seeded(tmp_path)
    with pytest.raises(IdempotencyConflict):
        promote(journal, originals, key="ingested", embed=no_backend, embedding_space=no_backend)
    receipt = promote(journal, originals)
    request, digest = intent(originals, text="Another decision")
    with pytest.raises(IdempotencyConflict):
        promote(journal, request=request, digest=digest, embed=no_backend, embedding_space=no_backend)
    assert journal.read_snapshot()[0] == receipt["revision"]


@pytest.mark.parametrize("vector", [[], [[1., 0.]], [float("nan")], [float("inf")], [1e100],
    [1 + 2j], [True, False], ["1", "0"], None], ids=[f"vector-{index}" for index in range(9)])
def test_invalid_vectors_cannot_commit_a_promotion(tmp_path, vector):
    journal, originals = seeded(tmp_path)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptEmbeddingError):
        promote(journal, originals, embed=lambda _: vector)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("damage", ["missing", "identity", "dimension", "lineage-vector", "typed-identity"])
def test_embedding_contract_is_validated_before_model_calls(tmp_path, damage):
    journal, originals = seeded(tmp_path)
    identity = copy.deepcopy(IDENTITY)
    def change(payload):
        if damage == "missing":
            payload.pop("embedding_contract")
        elif damage == "identity":
            payload["embedding_contract"]["identity"]["revision"] = "other"
        elif damage == "dimension":
            payload["embedding_contract"]["dimension"] = 3
        elif damage == "lineage-vector":
            payload["lineage"][0]["embedding"] = [1.0]
        else:
            identity["model"] = {"typed": True}
            payload["embedding_contract"]["identity"]["model"] = {"typed": 1}
    edited(journal, change)
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        promote(journal, originals, embed=no_backend, embedding_space=lambda: identity)


def test_caller_retained_vector_and_intent_are_frozen(tmp_path, monkeypatch):
    journal, originals = seeded(tmp_path)
    request, digest = intent(originals)
    retained = np.array(PROMOTED_VECTOR, dtype=np.float32)
    lookup = journal.lookup_receipt
    prepared = False
    def mutate(**kwargs):
        request["text"] = "Caller changed text"
        request["sources"][0]["memory_id"] = 90
        return lookup(**kwargs)
    def embed(_):
        nonlocal prepared
        prepared = True
        return retained
    def identity():
        if prepared:
            retained[:] = [1., 0., 0., 0.]
        return IDENTITY
    monkeypatch.setattr(journal, "lookup_receipt", mutate)
    result = promote(journal, request=request, digest=digest, embed=embed,
        embedding_space=identity)["result"]["memory"]
    assert result["text"] == PROMOTED_TEXT and result["embedding"] == PROMOTED_VECTOR
    assert result["summary_of"] == [0, 1]


def test_embedding_holds_no_store_ownership_and_all_sources_are_rechecked(tmp_path):
    journal, originals = seeded(tmp_path)
    completed = threading.Event()
    def competing_write():
        from daystrom_dml.store_lock import store_write_lock
        with store_write_lock(journal.path.parent, operation="fixture-competing", timeout_ms=1000):
            edited(JournalStateStore(journal.path), lambda payload: payload["items"][1]["meta"].update(concurrent=True))
        completed.set()
    def embed(_):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(competing_write)
            assert completed.wait(5), "Model callback held store ownership"
            future.result(timeout=5)
        return PROMOTED_VECTOR
    with pytest.raises(ReceiptLifecycleConflict, match="changed"):
        promote(journal, originals, embed=embed)
    assert len(journal.load()["items"]) == 3


@pytest.mark.parametrize("target", ["first-source", "last-source", "unrelated", "contract", "capacity"])
def test_every_cas_attempt_checks_sources_contract_and_capacity(tmp_path, monkeypatch, target):
    journal, originals = seeded(tmp_path)
    before_revision = journal.read_snapshot()[0]
    save, attempts = journal.save_with_receipt, []
    def race(payload, **kwargs):
        attempts.append(kwargs["expected_revision"])
        if len(attempts) == 1:
            def change(current):
                if target == "contract":
                    current["embedding_contract"]["identity"]["revision"] = "other"
                elif target == "capacity":
                    current["items"].append(source(30))
                else:
                    index = {"first-source": 0, "last-source": 1, "unrelated": 2}[target]
                    current["items"][index]["meta"]["concurrent"] = True
            edited(JournalStateStore(journal.path), change)
        return save(payload, **kwargs)
    monkeypatch.setattr(journal, "save_with_receipt", race)
    if target == "unrelated":
        assert promote(journal, originals)["revision"] == before_revision + 2
        assert attempts == [before_revision, before_revision + 1]
        assert journal.load()["items"][2]["meta"]["concurrent"] is True
    else:
        error = ReceiptEmbeddingCompatibilityError if target == "contract" else ReceiptCapacityError if target == "capacity" else ReceiptLifecycleConflict
        with pytest.raises(error):
            promote(journal, originals, capacity=4)
        assert attempts == [before_revision]
        assert journal.lookup_receipt(SCOPE, "promotion", intent(originals)[1]) is None


def test_expiry_is_rechecked_after_embedding(tmp_path, monkeypatch):
    journal, _ = seeded(tmp_path)
    now = [100.0]
    originals = edited(journal, lambda payload: [record["meta"].update(expires_at=101.) for record in payload["items"][:2]])
    monkeypatch.setattr("daystrom_dml.services.receipt_promotion.time.time", lambda: now[0])
    def embed(_):
        now[0] = 101.0
        return PROMOTED_VECTOR
    with pytest.raises(ReceiptLifecycleConflict, match="eligible"):
        promote(journal, originals, embed=embed)
    assert len(journal.load()["items"]) == 3


@pytest.mark.parametrize("fail_model", [True, False])
def test_competing_success_wins_over_model_failure_or_identity_drift(tmp_path, fail_model):
    journal, originals = seeded(tmp_path)
    identity = copy.deepcopy(IDENTITY)
    receipts = []
    def embed(_):
        receipts.append(promote(JournalStateStore(journal.path), originals))
        identity["revision"] = "unavailable"
        if fail_model:
            raise OSError("Model temporarily unavailable")
        return PROMOTED_VECTOR
    assert promote(journal, originals, embed=embed, embedding_space=lambda: identity) == receipts[0]
    assert len(journal.load()["items"]) == 4


def test_three_cas_conflicts_stop_without_an_acknowledgement(tmp_path, monkeypatch):
    journal, originals = seeded(tmp_path)
    before, attempts = journal.read_snapshot(), []
    def conflict(*args, **kwargs):
        attempts.append(kwargs["expected_revision"])
        raise RevisionConflict("Competing write")
    monkeypatch.setattr(journal, "save_with_receipt", conflict)
    with pytest.raises(RevisionConflict, match="three"):
        promote(journal, originals)
    assert attempts == [before[0]] * 3 and journal.read_snapshot() == before


@pytest.mark.parametrize("point", RECEIPT_FAULT_POINTS)
def test_storage_faults_neither_lose_sources_nor_invent_a_derived_result(tmp_path, point):
    journal, originals = seeded(tmp_path)
    before = journal.read_snapshot()
    def fail(here):
        if here == point:
            raise OSError("Injected storage failure")
    journal._fault_hook = fail
    if point == "after_commit":
        assert promote(journal, originals)["revision"] == before[0] + 1
    else:
        with pytest.raises(ReceiptCommitRejected):
            promote(journal, originals)
        assert journal.read_snapshot() == before
        assert journal.lookup_receipt(SCOPE, "promotion", intent(originals)[1]) is None
    journal._fault_hook = lambda _: None
    assert promote(journal, originals)["revision"] == before[0] + 1


def test_lost_acknowledgement_and_unreadable_authority_recover_without_models(tmp_path, monkeypatch):
    journal, originals = seeded(tmp_path)
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
        promote(journal, originals)
    reopened = JournalStateStore(journal.path)
    assert promote(reopened, originals, embed=no_backend, embedding_space=no_backend)["revision"] == reopened.read_snapshot()[0]


def test_projection_and_telemetry_failure_cannot_revoke_durable_success(tmp_path):
    journal, originals = seeded(tmp_path)
    errors = []
    def hydrate(*args):
        raise RuntimeError("Projection cannot hydrate")
    def degraded(exc):
        errors.append(str(exc))
        raise RuntimeError("Telemetry unavailable")
    receipt = promote(journal, originals, hydrate=hydrate, degraded=degraded)
    assert errors == ["Projection cannot hydrate"]
    assert promote(journal, originals, embed=no_backend, embedding_space=no_backend) == receipt


def test_real_disk_quota_rolls_back_output_and_keeps_old_receipts(tmp_path, monkeypatch):
    journal, originals = seeded(tmp_path)
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
    request, digest = intent(originals, text="x" * 200_000)
    with pytest.raises(ReceiptCommitRejected) as error:
        promote(journal, request=request, digest=digest)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert "full" in str(error.value.__cause__).lower()
    reopened = JournalStateStore(journal.path)
    assert reopened.read_snapshot() == before
    assert reopened.lookup_receipt(SCOPE, "ingested", "a" * 64) == ingestion


def test_serialization_interruption_never_publishes_output(tmp_path, monkeypatch):
    journal, originals = seeded(tmp_path)
    before = journal.read_snapshot()
    encode = journal._encode
    def fail_output(payload):
        if isinstance(payload, dict) and any("promotion_decision" in record.get("meta", {}) for record in payload.get("items", [])):
            raise TypeError("Interrupted serialization")
        return encode(payload)
    monkeypatch.setattr(journal, "_encode", fail_output)
    with pytest.raises(ReceiptCommitRejected):
        promote(journal, originals)
    assert journal.read_snapshot() == before


def test_schema_one_is_never_upgraded_implicitly(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.sqlite3")
    with pytest.raises(JournalSchemaError):
        promote(journal, [source()], embed=no_backend, embedding_space=no_backend)
    assert journal.schema_version == 1 and journal.read_snapshot()[0] == 0


@pytest.mark.parametrize("dimension", [0, 2], ids=["empty-proof-vector", "mixed-proof-dimensions"])
def test_historical_receipt_proof_requires_one_nonempty_embedding_space(tmp_path, dimension):
    journal, originals = seeded(tmp_path)
    first = promote(journal, originals)
    altered = copy.deepcopy(originals)
    altered[0]["embedding"] = [1.] * dimension
    request, digest = intent(altered)
    forged = copy.deepcopy(first["result"]["memory"])
    forged["meta"]["promotion_decision"]["sources"] = [
        {"memory_digest": memory_digest(record), "memory": record} for record in altered]
    revision, payload = journal.read_snapshot()
    payload["items"][-1] = forged
    journal.save_with_receipt(payload, scope=SCOPE, key="forged", request_digest=digest,
        result={"memory": forged}, expected_revision=revision, operation="generic")
    with pytest.raises(ReceiptLifecycleConflict, match="acknowledge"):
        promote(journal, request=request, digest=digest, key="forged", embed=no_backend,
            embedding_space=no_backend)
