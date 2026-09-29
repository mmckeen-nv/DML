"""Independent oracles for guarded content changes and historical replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_ingestion import (
    ReceiptEmbeddingCompatibilityError,
    ReceiptEmbeddingError,
)
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict
from daystrom_dml.services.receipt_update import (
    canonical_update_request,
    update_receipted,
)


SCOPE = dict(tenant_id="tenant", client_id=None, session_id=None, instance_id=None)
IDENTITY = dict(backend="independent.fixture", revision="immutable-v1", model=None, mode="native")
NEW_TEXT = "The owner corrected this exact stored claim."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def record(ident=0):
    return dict(schema_version=1, id=ident, text=f"Independent original claim {ident}",
                embedding=[1., 0.], timestamp=1., salience=0.75, fidelity=0.5,
                level=0, summary_of=[], children=[],
                meta={**SCOPE, "provenance": {"revision": 1},
                      "source_trust": "untrusted", "expires_at": 99.0})


@pytest.fixture
def journal(tmp_path):
    store = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    store.save({"items": [record(i) for i in range(3)], "lineage": [], "next_id": 3,
                "embedding_contract": {"schema_version": "dml-embedding-contract-v1",
                                       "identity": deepcopy(IDENTITY), "dimension": 2}},
               expected_revision=0, operation="independent-fixture")
    return store


def request_for(source=None, **overrides):
    source = record() if source is None else source
    values = dict(text=NEW_TEXT, expected_memory_digest=digest(source),
                  reason="Owner supplied a content correction", **SCOPE)
    values.update(overrides)
    return canonical_update_request(source["id"], **values)


def invoke(journal, request=None, request_digest=None, *, key="update", **overrides):
    if request is None:
        request, request_digest = request_for()
    options = dict(embed=lambda _: np.array([0., 1.], dtype=np.float32),
                   embedding_space=lambda: deepcopy(IDENTITY),
                   hydrate=lambda *_: None, degraded=lambda _: None)
    options.update(overrides)
    return update_receipted(journal, request=request, request_digest=request_digest,
                            key=key, **options)


def save_items(journal, items):
    revision, state = journal.read_snapshot()
    state["items"] = items
    journal.save(state, expected_revision=revision, operation="independent-change")


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("expected_memory_digest", "A" * 64),
    ("schema_version", True), ("reason", []), ("text", "  "),
    ("unknown", 1),
])
def test_invalid_direct_intent_never_reaches_authority(journal, monkeypatch, field, value):
    request, _ = request_for()
    request[field] = value
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: pytest.fail("invalid input reached authority"))
    with pytest.raises(ValueError):
        invoke(journal, request, digest(request))


def test_caller_cannot_change_intent_while_lookup_or_embedding_runs(journal, monkeypatch):
    request, request_digest = request_for()
    frozen = deepcopy(request)
    lookup = journal.lookup_receipt

    def mutate_caller(**kwargs):
        request["scope"]["tenant_id"] = "other"
        request["memory_id"] = 1
        request["text"] = "Changed during storage I/O"
        return lookup(**kwargs)

    def embed(text):
        assert text == frozen["text"]
        request["reason"] = "Changed inside the model callback"
        return [0., 1.]

    monkeypatch.setattr(journal, "lookup_receipt", mutate_caller)
    receipt = invoke(journal, request, request_digest, embed=embed)
    assert receipt["scope"] == frozen["scope"]
    assert receipt["request_digest"] == request_digest
    assert receipt["result"]["memory"]["id"] == 0
    assert receipt["result"]["memory"]["text"] == frozen["text"]
    assert receipt["result"]["memory"]["meta"]["content_update_decision"]["reason"] == frozen["reason"]
    assert journal.load()["items"][1:] == [record(1), record(2)]


def test_mutating_same_identity_object_during_embedding_fails_closed(journal):
    identity = deepcopy(IDENTITY)
    before = journal.read_snapshot()

    def embed(_text):
        identity["revision"] = "another-vector-space"
        return [0., 1.]

    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        invoke(journal, embedding_space=lambda: identity, embed=embed)
    assert journal.read_snapshot() == before


def test_identical_commit_between_initial_lookup_and_preparation_replays_receipt(journal, monkeypatch):
    read = journal.read_snapshot
    winner = []

    def concurrent_commit_before_snapshot():
        if not winner:
            writer = JournalStateStore(journal.path)
            winner.append(invoke(writer))
        return read()

    monkeypatch.setattr(journal, "read_snapshot", concurrent_commit_before_snapshot)
    result = invoke(journal, embed=lambda _: pytest.fail("already-committed request invoked model"))
    assert result == winner[0]
    assert journal.read_snapshot()[0] == 2


def test_identical_raw_commit_after_owned_lookup_replays_receipt(journal, monkeypatch):
    request, request_digest = request_for()
    read = journal.read_snapshot
    reads = 0
    winner = []

    def concurrent_commit_before_owned_snapshot():
        nonlocal reads
        reads += 1
        if reads == 2:
            # Journal CAS writers can bypass advisory service ownership. Their
            # atomic receipts are still authoritative and must be reconciled.
            writer = JournalStateStore(journal.path)
            revision, payload = writer.read_snapshot()
            item = payload["items"][0]
            item["text"] = request["text"]
            item["embedding"] = [0., 1.]
            item["meta"]["content_update_decision"] = {
                "schema_version": "dml-content-update-decision-v1",
                "prior_memory_digest": request["expected_memory_digest"],
                "reason": request["reason"]}
            winner.append(writer.save_with_receipt(payload, scope=SCOPE, key="update",
                request_digest=request_digest, result={"memory": item},
                expected_revision=revision, operation="update-receipt-v1"))
        return read()

    monkeypatch.setattr(journal, "read_snapshot", concurrent_commit_before_owned_snapshot)
    assert invoke(journal, request, request_digest) == winner[0]
    assert journal.read_snapshot()[0] == 2


def test_backend_retained_array_cannot_change_prepared_vector(journal):
    vector = np.array([0., 1.], dtype=np.float32)
    model_finished = False

    def embed(_text):
        nonlocal model_finished
        model_finished = True
        return vector

    def identity():
        if model_finished:
            vector[:] = [1., 0.]
        return IDENTITY

    receipt = invoke(journal, embed=embed, embedding_space=identity)
    assert vector.tolist() == [1., 0.]
    assert receipt["result"]["memory"]["embedding"] == [0., 1.]
    assert journal.load()["items"][0]["embedding"] == [0., 1.]


@pytest.mark.parametrize("vector", [[0., 1., 0.], [float("nan"), 1.],
                                   [float("inf"), 1.], [True, False],
                                   ["0", "1"], [], [[0., 1.]]])
def test_invalid_model_vector_cannot_commit_any_content_change(journal, vector):
    before = journal.read_snapshot()
    with pytest.raises(ReceiptEmbeddingError):
        invoke(journal, embed=lambda _: vector)
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("changed", [0, 1])
def test_raw_writer_rebase_preserves_unrelated_changes_but_rejects_target_changes(journal, monkeypatch, changed):
    save = journal.save_with_receipt
    attempts = 0

    def competing_save(payload, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            writer = JournalStateStore(journal.path)
            items = writer.load()["items"]
            # Python equality considers these equal. The full JSON digest must not.
            items[changed]["meta"]["provenance"]["revision"] = 1.0
            save_items(writer, items)
        return save(payload, **kwargs)

    monkeypatch.setattr(journal, "save_with_receipt", competing_save)
    if changed == 0:
        with pytest.raises(ReceiptLifecycleConflict):
            invoke(journal)
        assert attempts == 1
        assert journal.read_snapshot()[0] == 2
        assert journal.load()["items"][0]["text"] == record()["text"]
    else:
        receipt = invoke(journal)
        assert receipt["revision"] == 3
        assert attempts == 2
    assert type(journal.load()["items"][changed]["meta"]["provenance"]["revision"]) is float


@pytest.mark.parametrize("point", ["prepared", "cas-retry"])
def test_contract_change_is_detected_after_model_work_and_revision_rebase(journal, monkeypatch, point):
    save = journal.save_with_receipt
    attempted = False

    def change_contract():
        writer = JournalStateStore(journal.path)
        revision, payload = writer.read_snapshot()
        payload["embedding_contract"]["identity"]["revision"] = "different-model"
        writer.save(payload, expected_revision=revision, operation="independent-contract-change")

    def embed(_text):
        if point == "prepared":
            change_contract()
        return [0., 1.]

    def race(payload, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            change_contract()
        return save(payload, **kwargs)

    if point == "cas-retry":
        monkeypatch.setattr(journal, "save_with_receipt", race)
    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        invoke(journal, embed=embed)
    assert journal.read_snapshot()[0] == 2
    assert journal.load()["items"] == [record(i) for i in range(3)]


@pytest.mark.parametrize("mutation", ["removed", "retired", "contract-changed"])
def test_historical_receipt_replays_before_model_and_current_state(journal, mutation):
    receipt = invoke(journal)
    revision, state = journal.read_snapshot()
    if mutation == "removed":
        state["items"] = state["items"][1:]
    elif mutation == "retired":
        state["items"][0]["meta"]["memory_state"] = "deleted"
    else:
        state["embedding_contract"]["identity"]["revision"] = "new-contract"
    journal.save(state, expected_revision=revision, operation="independent-history-change")
    before = journal.read_snapshot()

    def unavailable(*_):
        pytest.fail("historical replay invoked the embedding backend or identity")

    assert invoke(journal, embed=unavailable, embedding_space=unavailable) == receipt
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("metadata", [
    {"memory_state": "active", "lifecycle_state": "retired"},
    {"memory_state": "active", "lifecycle_state": "superseded"},
    {"retirement_decision": None}, {"supersession_decision": {}},
    {"superseded_by": False},
    {"content_update_decision": {"schema_version": "dml-content-update-decision-v1"}},
    {"content_update_decision": {"schema_version": "future-format"}},
])
def test_prior_terminal_or_unknown_decisions_cannot_be_overwritten(journal, metadata):
    items = journal.load()["items"]
    items[0]["meta"].update(metadata)
    save_items(journal, items)
    request, request_digest = request_for(items[0])
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest, embed=lambda _: pytest.fail("ineligible target invoked model"))
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("corruption", [
    "wrong-text", "wrong-target", "wrong-prior-digest", "extra-decision-field",
    "wrong-reason", "retired-alias", "retirement-decision", "zero-vector",
])
def test_generic_receipt_cannot_acknowledge_another_content_update(journal, corruption):
    request, request_digest = request_for()
    revision, state = journal.read_snapshot()
    item = state["items"][0]
    item["text"] = request["text"]
    item["embedding"] = [0., 1.]
    decision = {"schema_version": "dml-content-update-decision-v1",
                "prior_memory_digest": request["expected_memory_digest"],
                "reason": request["reason"]}
    item["meta"]["content_update_decision"] = decision
    if corruption == "wrong-text":
        item["text"] = "An unrelated operation wrote a different text"
    elif corruption == "wrong-target":
        item["id"] = 3
        state["next_id"] = 4
    elif corruption == "wrong-prior-digest":
        decision["prior_memory_digest"] = "f" * 64
    elif corruption == "extra-decision-field":
        decision["invented_authority"] = "system"
    elif corruption == "wrong-reason":
        decision["reason"] = "Different intent"
    elif corruption == "retired-alias":
        item["meta"].update(memory_state="active", lifecycle_state="retired")
    elif corruption == "retirement-decision":
        item["meta"]["retirement_decision"] = None
    else:
        item["embedding"] = []
    journal.save_with_receipt(state, scope=SCOPE, key="update", request_digest=request_digest,
        result={"memory": item}, expected_revision=revision, operation="generic-import")
    before = journal.read_snapshot()
    with pytest.raises((ReceiptLifecycleConflict, ReceiptEmbeddingCompatibilityError)):
        invoke(journal, embed=lambda _: pytest.fail("malformed historical receipt invoked model"))
    assert journal.read_snapshot() == before


def test_second_update_preserves_all_other_fields_and_each_historical_receipt(journal):
    before = deepcopy(journal.load())
    first = invoke(journal)
    once = journal.load()["items"][0]
    request, request_digest = request_for(once, text="A second exact content correction")
    second = invoke(journal, request, request_digest, key="second-update", embed=lambda _: [0.5, 0.5])
    after = deepcopy(journal.load())
    expected = deepcopy(before)
    expected["items"][0]["text"] = request["text"]
    expected["items"][0]["embedding"] = [0.5, 0.5]
    expected["items"][0]["meta"]["content_update_decision"] = {
        "schema_version": "dml-content-update-decision-v1",
        "prior_memory_digest": digest(once), "reason": request["reason"]}
    assert after == expected
    assert invoke(journal) == first
    assert invoke(journal, request, request_digest, key="second-update") == second
    # Editing quarantined/untrusted content does not confer authority or freshness.
    assert after["items"][0]["timestamp"] == 1.
    assert after["items"][0]["meta"]["source_trust"] == "untrusted"
    assert after["items"][0]["meta"]["expires_at"] == 99.0
