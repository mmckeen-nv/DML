"""Independent payload-copy and scope-isolation oracles for retention reports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_lifecycle import ReceiptMemoryNotFound
from daystrom_dml.services.receipt_promotion import canonical_promotion_request, promote_receipted
from daystrom_dml.services.retention import (
    RetentionInspectionUnsupported, canonical_retention_request, inspect_memory_retention,
)


SCOPE = {"tenant_id": "independent-retention", "client_id": "client", "session_id": None,
         "instance_id": "worker"}
IDENTITY = {"backend": "independent-retention", "revision": "v1", "model": None, "mode": "native"}
SURFACES = {"current_items", "current_lineage", "journal_snapshot", "receipts", "outbox_states"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def record(ident=0, *, scope=None):
    return {"schema_version": 1, "id": ident, "text": f"Independent private source {ident}",
            "embedding": [1., 0.], "timestamp": 100., "salience": .5, "fidelity": .5,
            "level": 0, "summary_of": [ident], "children": [ident],
            "meta": {**(SCOPE if scope is None else scope), "source_trust": "trusted",
                     "no_merge": True, "private_origin": f"private-source-{ident}"}}


def save(journal, items, *, lineage=None):
    revision, state = journal.read_snapshot()
    state["items"] = items
    if lineage is not None:
        state["lineage"] = lineage
    journal.save(state, expected_revision=revision, operation="independent-fixture-change")


def prepare(tmp_path, schema):
    journal = JournalStateStore(tmp_path / "source.sqlite3", receipt_mode=True,
                               outbox_mode=schema == 3, snapshot_interval=1)
    source = record()
    state = {"items": [source], "lineage": [], "next_id": 1,
             "embedding_contract": {"schema_version": "dml-embedding-contract-v1",
                                    "identity": IDENTITY, "dimension": 2}}
    journal.save_with_receipt(state, scope=SCOPE, key="private-original-retry-key",
                             request_digest=digest({"independent": "source"}),
                             result={"memory": source}, expected_revision=0)
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(journal.path, destination)
        journal = JournalStateStore(destination, snapshot_interval=1)
    request, request_digest = canonical_promotion_request(
        [{"memory_id": 0, "expected_memory_digest": digest(source)}],
        text="A private derived interpretation", reason="Reviewed independent source", **SCOPE)
    receipt = promote_receipted(
        journal, request=request, request_digest=request_digest,
        key="private-promotion-retry-key", embed=lambda _: np.array([0., 1.]),
        embedding_space=lambda: IDENTITY, capacity=100, hydrate=lambda *_: None,
        degraded=lambda _: None)
    return journal, source, receipt["result"]["memory"]


def inspect(journal, ident=0, **scope):
    return inspect_memory_retention(
        journal, request=canonical_retention_request(ident, **(scope or SCOPE)))


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_history_only_source_counts_literal_payload_occurrences_without_binding_aliases(tmp_path, schema):
    journal, source, derived = prepare(tmp_path, schema)
    save(journal, [derived])
    before = journal.read_snapshot()
    report = inspect(journal)
    assert set(report["surfaces"]) == SURFACES
    expected = {name: {"direct_records": 0, "embedded_source_records": 0} for name in SURFACES}
    expected["current_items"]["embedded_source_records"] = 1
    expected["journal_snapshot"]["embedded_source_records"] = 1
    expected["receipts"] = {"direct_records": 1, "embedded_source_records": 1}
    # Schema 4's migration baseline replaces the pre-migration full-state event:
    # baseline/original, source-plus-derived, then derived-only are three states.
    if schema in (3, 4):
        expected["outbox_states"] = {"direct_records": 2, "embedded_source_records": 2}
    assert report["surfaces"] == expected
    assert report["known_reference_count"] == (4 if schema == 2 else 8)
    assert report["source"]["revision"] == before[0]
    assert report["source"]["state_digest"] == digest(before[1])
    assert journal.read_snapshot() == before
    rendered = encoded(report)
    for private in (source["text"], derived["text"], "private-original-retry-key",
                    "private-promotion-retry-key", "private-source-0"):
        assert private not in rendered
    assert report["erasure_proven"] is False
    assert report["physical_erasure_supported"] is False
    assert report["retirement_is_erasure"] is False


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_foreign_carrier_cannot_disclose_same_scope_embedded_memory(tmp_path, field):
    journal, source, derived = prepare(tmp_path, 2)
    # A fresh journal has only a foreign carrier and no historical target receipt.
    isolated = JournalStateStore(tmp_path / "isolated.sqlite3", receipt_mode=True)
    derived["meta"][field] = "foreign-secret-value"
    isolated.save({"items": [derived], "lineage": []}, expected_revision=0)
    with pytest.raises(ReceiptMemoryNotFound) as embedded:
        inspect(isolated)
    with pytest.raises(ReceiptMemoryNotFound) as absent:
        inspect(isolated, 987654)
    assert str(embedded.value) == str(absent.value)
    assert "foreign-secret-value" not in str(embedded.value)
    assert source["text"] not in str(embedded.value)


@pytest.mark.parametrize("damage", ["unknown-version", "bad-digest", "foreign-source", "recursive",
                                    "boolean-source-id", "duplicate-source", "empty-vector"])
def test_same_scope_ambiguous_source_proof_never_claims_zero(tmp_path, damage):
    _, source, derived = prepare(tmp_path, 2)
    proof = derived["meta"]["promotion_decision"]
    binding = proof["sources"][0]
    if damage == "unknown-version":
        proof["schema_version"] = "dml-promotion-decision-v999"
    elif damage == "bad-digest":
        binding["memory_digest"] = "0" * 64
    elif damage == "foreign-source":
        binding["memory"]["meta"]["tenant_id"] = "private-foreign-tenant"
        binding["memory_digest"] = digest(binding["memory"])
    elif damage == "recursive":
        binding["memory"]["meta"]["promotion_decision"] = deepcopy(proof)
        binding["memory_digest"] = digest(binding["memory"])
    elif damage == "boolean-source-id":
        binding["memory"]["id"] = False
        binding["memory_digest"] = digest(binding["memory"])
    elif damage == "duplicate-source":
        proof["sources"].append(deepcopy(binding))
    else:
        binding["memory"]["embedding"] = []
        binding["memory_digest"] = digest(binding["memory"])
    isolated = JournalStateStore(tmp_path / "isolated.sqlite3", receipt_mode=True)
    isolated.save({"items": [derived], "lineage": []}, expected_revision=0)
    before = isolated.read_snapshot()
    with pytest.raises(RetentionInspectionUnsupported) as invalid:
        inspect(isolated)
    assert isolated.read_snapshot() == before
    assert source["text"] not in str(invalid.value)
    assert "private-foreign-tenant" not in str(invalid.value)


def test_unknown_proof_on_foreign_record_does_not_poison_same_scope_inventory(tmp_path):
    journal, source, derived = prepare(tmp_path, 2)
    foreign = record(999, scope={**SCOPE, "tenant_id": "foreign-private-tenant"})
    foreign["meta"]["promotion_decision"] = {"schema_version": "future-unknown"}
    baseline = inspect(journal)
    save(journal, [source, derived, foreign])
    report = inspect(journal)
    assert report["surfaces"] == baseline["surfaces"]
    assert report["known_reference_count"] == baseline["known_reference_count"]
    assert "foreign-private-tenant" not in encoded(report)


def test_metadata_copies_and_bare_lineage_ids_do_not_become_payload_occurrences(tmp_path):
    journal = JournalStateStore(tmp_path / "source.sqlite3", receipt_mode=True)
    carrier = record(20)
    carrier["meta"]["arbitrary_payload_copy"] = record(0)
    carrier["summary_of"] = carrier["children"] = [0]
    journal.save({"items": [carrier], "lineage": []}, expected_revision=0)
    with pytest.raises(ReceiptMemoryNotFound):
        inspect(journal)
    assert inspect(journal, 20)["known_reference_count"] == 2


@pytest.mark.parametrize("stored", ["missing", "null", False, 0, ""],
                         ids=["missing", "null", "false", "zero", "empty"])
def test_only_missing_and_null_optional_scope_are_equivalent(tmp_path, stored):
    journal = JournalStateStore(tmp_path / "source.sqlite3", receipt_mode=True)
    source = record()
    if stored == "missing":
        del source["meta"]["session_id"]
    else:
        source["meta"]["session_id"] = None if stored == "null" else stored
    journal.save({"items": [source], "lineage": []}, expected_revision=0)
    if stored in ("missing", "null"):
        assert inspect(journal)["known_reference_count"] == 2
    else:
        with pytest.raises(ReceiptMemoryNotFound):
            inspect(journal)


def test_retired_derived_memory_still_retains_original_source_proof(tmp_path):
    from daystrom_dml.services.receipt_lifecycle import (
        canonical_retirement_request, retire_receipted,
    )
    journal, _, derived = prepare(tmp_path, 2)
    save(journal, [derived])
    request, request_digest = canonical_retirement_request(
        derived["id"], expected_memory_digest=digest(derived), reason="Retire derived text", **SCOPE)
    retire_receipted(journal, request=request, request_digest=request_digest,
                    key="retire-derived", hydrate=lambda *_: None, degraded=lambda _: None)
    report = inspect(journal)
    assert report["surfaces"]["current_items"]["embedded_source_records"] == 1
    assert report["surfaces"]["receipts"]["embedded_source_records"] == 2
    assert report["known_reference_count"] == 5


def test_report_and_request_are_detached_from_inflight_caller_mutation(tmp_path):
    journal, _, _ = prepare(tmp_path, 2)
    request = canonical_retention_request(0, **SCOPE)
    expected_request = deepcopy(request)
    before = journal.read_snapshot()

    def mutate(point):
        if point == "retention_after_snapshot":
            request["memory_id"] = 999
            request["scope"]["tenant_id"] = "foreign"

    journal._fault_hook = mutate
    report = inspect_memory_retention(journal, request=request)
    assert report["request"] == expected_request
    report["request"]["scope"]["tenant_id"] = "altered-result"
    report["surfaces"]["receipts"]["direct_records"] = 999
    report["uninspected_surfaces"].clear()
    journal._fault_hook = lambda _: None
    assert inspect(journal)["request"] == expected_request
    assert inspect(journal)["surfaces"]["receipts"]["direct_records"] == 1
    assert inspect(journal)["uninspected_surfaces"]
    assert journal.read_snapshot() == before


@pytest.mark.parametrize("bad_id", [True, False, 0., -1, "0", None],
                         ids=["true", "false", "float", "negative", "text", "null"])
def test_target_identity_is_never_coerced(bad_id):
    with pytest.raises(ValueError):
        canonical_retention_request(bad_id, **SCOPE)


@pytest.mark.parametrize("mutation", ["unknown-field", "missing-scope", "wrong-version"])
def test_noncanonical_request_is_rejected_before_journal_access(tmp_path, monkeypatch, mutation):
    journal, _, _ = prepare(tmp_path, 2)
    request = canonical_retention_request(0, **SCOPE)
    if mutation == "unknown-field":
        request["erase"] = True
    elif mutation == "missing-scope":
        del request["scope"]["session_id"]
    else:
        request["schema_version"] = "future-retention-request"
    monkeypatch.setattr(journal, "read_retention_view", lambda: pytest.fail("invalid request opened journal"))
    with pytest.raises(ValueError):
        inspect_memory_retention(journal, request=request)
