"""Independent oracles for immutable source proof and explicit derivation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.receipt_ingestion import ReceiptEmbeddingCompatibilityError
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict, ReceiptMemoryNotFound
from daystrom_dml.services.receipt_promotion import canonical_promotion_request, promote_receipted


SCOPE = dict(tenant_id="independent", client_id=None, session_id=None, instance_id=None)
IDENTITY = dict(backend="independent.fixture", revision="immutable-v1", model=None, mode="native")
TEXT = "The owner explicitly combined these historical source snapshots."
REASON = "Owner reviewed the exact source records"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def record(ident=0):
    return dict(schema_version=1, id=ident, text=f"Independent source claim {ident}",
                embedding=[1., 0.], timestamp=float(10 + ident),
                salience=0.8 - ident * 0.1, fidelity=0.7 - ident * 0.1,
                level=0, summary_of=[ident], children=[ident],
                meta={**SCOPE, "source_trust": "trusted", "kind": "memory", "no_merge": False,
                      "authority": {"revision": 1}, "source": f"document-{ident}",
                      "provenance": {"uri": f"fixture://source/{ident}"},
                      "summary": f"Obsolete cached text from source {ident}"})


@pytest.fixture
def journal(tmp_path):
    store = JournalStateStore(tmp_path / "journal.sqlite3", receipt_mode=True)
    store.save({"items": [record(i) for i in range(3)], "lineage": [record(9)], "next_id": 20,
                "embedding_contract": {"schema_version": "dml-embedding-contract-v1",
                                       "identity": deepcopy(IDENTITY), "dimension": 2}},
               expected_revision=0, operation="independent-fixture")
    return store


def request_for(records=None, **overrides):
    records = [record(), record(1)] if records is None else records
    sources = [{"memory_id": item["id"], "expected_memory_digest": digest(item)} for item in records]
    values = dict(text=TEXT, reason=REASON, **SCOPE)
    values.update(overrides)
    return canonical_promotion_request(sources, **values)


def invoke(journal, request=None, request_digest=None, *, key="promotion", **overrides):
    if request is None:
        request, request_digest = request_for()
    callbacks = dict(embed=lambda _: np.array([0., 1.], dtype=np.float32),
                     embedding_space=lambda: deepcopy(IDENTITY), capacity=100,
                     hydrate=lambda *_: None, degraded=lambda _: None)
    callbacks.update(overrides)
    return promote_receipted(journal, request=request, request_digest=request_digest,
                            key=key, **callbacks)


def replace_items(journal, items):
    revision, payload = journal.read_snapshot()
    payload["items"] = items
    journal.save(payload, expected_revision=revision, operation="independent-source-change")


def new_intent(journal):
    return request_for(journal.load()["items"][:2])


def test_exact_sources_proof_and_conservative_result_without_source_mutation(journal):
    before_revision, before = journal.read_snapshot()
    receipt = invoke(journal)
    revision, after = journal.read_snapshot()
    result = receipt["result"]["memory"]
    assert result["id"] == 20 and after["next_id"] == 21
    assert encoded(after["items"][:-1]) == encoded(before["items"])
    assert encoded(after["lineage"]) == encoded(before["lineage"])
    assert result["summary_of"] == result["children"] == [0, 1]
    assert result["timestamp"] == 10.0
    assert result["salience"] == min(item["salience"] for item in before["items"][:2])
    assert result["fidelity"] == min(item["fidelity"] for item in before["items"][:2])
    assert result["level"] == 1 and result["text"] == TEXT and result["embedding"] == [0., 1.]
    expected_meta = {name: value for name, value in record()["meta"].items()
                     if name not in {"summary", "source", "provenance"}}
    expected_meta["no_merge"] = True
    expected_meta["promotion_decision"] = {
        "schema_version": "dml-promotion-decision-v1", "reason": REASON,
        "sources": [{"memory_digest": digest(record(i)), "memory": record(i)} for i in (0, 1)]}
    assert encoded(result["meta"]) == encoded(expected_meta)
    assert revision == before_revision + 1 == receipt["revision"]
    assert journal.decisions()[-1]["operation"] == "promote-receipt-v1"


def test_source_order_is_canonical_and_exact_replay_does_not_prepare(journal):
    request, request_digest = request_for([record(1), record()])
    assert (request, request_digest) == request_for()
    receipt = invoke(journal, request, request_digest)
    assert invoke(journal, embed=lambda _: pytest.fail("replay invoked model"),
                  embedding_space=lambda: pytest.fail("replay inspected backend")) == receipt


def test_caller_cannot_mutate_sources_scope_or_text_during_lookup(journal, monkeypatch):
    request, request_digest = request_for()
    lookup = journal.lookup_receipt

    def alter_request(**kwargs):
        request["sources"][1]["memory_id"] = 2
        request["scope"]["tenant_id"] = "different"
        request["text"] = "Unexpected replacement"
        return lookup(**kwargs)

    monkeypatch.setattr(journal, "lookup_receipt", alter_request)
    receipt = invoke(journal, request, request_digest)
    result = receipt["result"]["memory"]
    assert result["text"] == TEXT
    assert result["summary_of"] == [0, 1]
    assert result["meta"]["tenant_id"] == SCOPE["tenant_id"]


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_every_scope_is_checked_before_any_stale_source_disclosure(journal, field):
    items = journal.load()["items"]
    items[0]["text"] = "Stale first source"
    items[1]["meta"][field] = "other"
    replace_items(journal, items)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptMemoryNotFound, match="Memory is not available in the requested scope"):
        invoke(journal, embed=lambda _: pytest.fail("cross-scope source reached model"))
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("change", ["typed-authority", "unknown-present", "claim-value", "lifecycle-alias"],
                         ids=["typed-authority", "unknown-present", "claim-value", "lifecycle-alias"])
def test_merge_cannot_erase_an_authority_distinction(journal, change):
    items = journal.load()["items"]
    if change == "typed-authority":
        items[1]["meta"]["authority"]["revision"] = 1.0
    elif change == "unknown-present":
        items[1]["meta"]["unknown_policy"] = None
    elif change == "claim-value":
        for item in items[:2]:
            item["meta"]["claim_key"] = "office"
        items[0]["meta"]["claim_value"] = "Paris"
        items[1]["meta"]["claim_value"] = "Berlin"
    else:
        items[1]["meta"].update(memory_state="active", lifecycle_state=" RETIRED ")
    replace_items(journal, items)
    request, request_digest = new_intent(journal)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest,
               embed=lambda _: pytest.fail("incompatible authority reached model"))
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("value", [True, 1, "false", None], ids=["true", "integer", "text", "null"])
def test_multisource_no_merge_requires_unambiguous_permission(journal, value):
    items = journal.load()["items"]
    for item in items[:2]:
        item["meta"]["no_merge"] = value
    replace_items(journal, items)
    request, request_digest = new_intent(journal)
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest,
               embed=lambda _: pytest.fail("forbidden merge reached model"))


@pytest.mark.parametrize("children", [[True], [1.0], [1, 1]],
                         ids=["boolean-self-alias", "float-self-alias", "duplicate-self"])
def test_base_source_lineage_never_coerces_boolean_or_float_children(journal, children):
    items = journal.load()["items"]
    items[1]["children"] = children
    replace_items(journal, items)
    request, request_digest = new_intent(journal)
    before = journal.read_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest,
               embed=lambda _: pytest.fail("ambiguous lineage reached model"))
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("changed", [0, 1, 2], ids=["first-source", "last-source", "unrelated"])
def test_cas_rebase_rechecks_every_source_without_losing_unrelated_changes(journal, monkeypatch, changed):
    original_save = journal.save_with_receipt
    attempts = 0

    def competing_write(payload, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            writer = JournalStateStore(journal.path)
            items = writer.load()["items"]
            items[changed]["meta"]["authority"]["revision"] = 1.0
            replace_items(writer, items)
        return original_save(payload, **kwargs)

    monkeypatch.setattr(journal, "save_with_receipt", competing_write)
    if changed in (0, 1):
        with pytest.raises(ReceiptLifecycleConflict):
            invoke(journal)
        assert len(journal.load()["items"]) == 3
    else:
        receipt = invoke(journal)
        assert receipt["result"]["memory"]["id"] == 20
        assert len(journal.load()["items"]) == 4
    assert type(journal.load()["items"][changed]["meta"]["authority"]["revision"]) is float


def test_same_key_winner_that_fills_capacity_before_snapshot_is_replayed(journal, monkeypatch):
    original_read = journal.read_snapshot
    winner = []

    def concurrent_commit():
        if not winner:
            writer = JournalStateStore(journal.path)
            winner.append(invoke(writer))
        return original_read()

    monkeypatch.setattr(journal, "read_snapshot", concurrent_commit)
    assert invoke(journal, capacity=4,
                  embed=lambda _: pytest.fail("durable winner reached model")) == winner[0]


def test_same_identity_object_mutation_cannot_change_prepared_space(journal):
    identity = deepcopy(IDENTITY)
    before = journal.read_snapshot()

    def embed(_):
        identity["revision"] = "different-space"
        return [0., 1.]

    with pytest.raises(ReceiptEmbeddingCompatibilityError):
        invoke(journal, embed=embed, embedding_space=lambda: identity)
    assert journal.read_snapshot() == before


def test_retained_model_vector_cannot_be_changed_after_preparation(journal):
    vector = np.array([0., 1.], dtype=np.float32)
    prepared = False

    def embed(_):
        nonlocal prepared
        prepared = True
        return vector

    def identity():
        if prepared:
            vector[:] = [1., 0.]
        return IDENTITY

    result = invoke(journal, embed=embed, embedding_space=identity)["result"]["memory"]
    assert result["embedding"] == [0., 1.]


def test_historical_proof_replay_is_independent_of_current_source_and_contract(journal):
    request, request_digest = request_for()
    receipt = invoke(journal, request, request_digest)
    revision, payload = journal.read_snapshot()
    payload["items"] = []
    payload["lineage"] = []
    payload["embedding_contract"]["identity"]["revision"] = "later-space"
    payload["embedding_contract"]["dimension"] = 3
    journal.save(payload, expected_revision=revision, operation="independent-later-lifecycle")
    assert invoke(journal, request, request_digest,
                  embed=lambda _: pytest.fail("historical retry prepared a vector"),
                  embedding_space=lambda: pytest.fail("historical retry consulted current space")) == receipt


@pytest.mark.parametrize("damage", ["missing-proof", "wrong-source", "wrong-digest", "inflated-authority", "wrong-lineage", "source-id", "cached-summary", "wrong-vector-dimension"],
                         ids=["missing-proof", "wrong-source", "wrong-digest", "inflated-authority", "wrong-lineage", "source-id", "cached-summary", "wrong-vector-dimension"])
def test_generic_receipt_cannot_claim_an_unrelated_or_inflated_promotion(journal, monkeypatch, damage):
    request, request_digest = request_for()
    result = deepcopy(invoke(journal, request, request_digest))
    memory = result["result"]["memory"]
    if damage == "missing-proof":
        memory["meta"].pop("promotion_decision")
    elif damage == "wrong-source":
        memory["meta"]["promotion_decision"]["sources"][1]["memory"]["text"] = "Invented origin"
    elif damage == "wrong-digest":
        memory["meta"]["promotion_decision"]["sources"][1]["memory_digest"] = "0" * 64
    elif damage == "inflated-authority":
        memory["fidelity"] = 1.0
    elif damage == "wrong-lineage":
        memory["summary_of"] = [0, 2]
    elif damage == "source-id":
        memory["id"] = 0
    elif damage == "wrong-vector-dimension":
        memory["embedding"] = [1.0]
    else:
        memory["meta"]["summary"] = "A generic receipt should not reintroduce a stale source summary"
    monkeypatch.setattr(journal, "lookup_receipt", lambda **_: result)
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(journal, request, request_digest,
               embed=lambda _: pytest.fail("unrelated historical receipt reached model"))
