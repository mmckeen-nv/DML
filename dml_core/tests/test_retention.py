"""Literal retained occurrences, strict scope, and immutable inspection results."""
from __future__ import annotations

import copy
from contextlib import closing
import hashlib
import json
import sqlite3

import pytest

from daystrom_dml.journal import JournalSchemaError, JournalStateStore
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.receipt_lifecycle import ReceiptMemoryNotFound, memory_digest
from daystrom_dml.services.retention import (
    RetentionInspectionUnsupported, UNINSPECTED_SURFACES,
    canonical_retention_request, inspect_memory_retention,
)

SCOPE = {"tenant_id": "tenant", "client_id": "client", "session_id": None, "instance_id": None}


def record(ident=0, *, scope=None):
    return {"schema_version": 1, "id": ident, "text": "private-text-sentinel",
            "embedding": [1.0, 0.0], "timestamp": 123.0, "salience": 0.8,
            "fidelity": 0.7, "level": 0, "summary_of": [],
            "meta": {**(SCOPE if scope is None else scope), "source_trust": "trusted", "no_merge": False}}


def derived(source=None):
    source = record() if source is None else copy.deepcopy(source)
    result = record(1)
    result.update(level=1, summary_of=[source["id"]], children=[source["id"]])
    result["meta"].update(no_merge=True, promotion_decision={
        "schema_version": "dml-promotion-decision-v1", "reason": "private-reason-sentinel",
        "sources": [{"memory_digest": memory_digest(source), "memory": source}],
    })
    return result


def seeded(tmp_path, schema=2, *, payload=None, acknowledged=None):
    source, promoted = record(), derived()
    payload = ({"items": [source, promoted], "lineage": [copy.deepcopy(source)], "next_id": 2}
               if payload is None else payload)
    acknowledged = promoted if acknowledged is None else acknowledged
    journal = JournalStateStore(tmp_path / "source" / "journal.sqlite3",
                                receipt_mode=True, outbox_mode=(schema == 3))
    journal.save_with_receipt(payload, scope=SCOPE, key="private-key-sentinel", request_digest="a" * 64,
                             result={"memory": acknowledged}, expected_revision=0)
    if schema == 4:
        destination = tmp_path / "migrated" / "journal.sqlite3"
        upgrade_outbox_journal(journal.path, destination)
        journal = JournalStateStore(destination)
    return journal


def inspect(journal, memory_id=0, **scope):
    return inspect_memory_retention(journal, request=canonical_retention_request(memory_id, **{**SCOPE, **scope}))


def rows(journal):
    with closing(sqlite3.connect(journal.path)) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: connection.execute(f'SELECT * FROM "{name}"').fetchall() for name in names}


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_counts_literal_records_and_proofs_without_chasing_outbox_bindings(tmp_path, schema):
    journal = seeded(tmp_path, schema)
    before = rows(journal)
    report = inspect(journal)
    assert set(report) == {"schema_version", "request", "source", "coverage", "surfaces",
                           "known_reference_count", "physical_erasure_supported", "erasure_proven",
                           "retirement_is_erasure", "uninspected_surfaces"}
    assert report["surfaces"] == {
        "current_items": {"direct_records": 1, "embedded_source_records": 1},
        "current_lineage": {"direct_records": 1, "embedded_source_records": 0},
        "journal_snapshot": {"direct_records": 2, "embedded_source_records": 1},
        "receipts": {"direct_records": 0, "embedded_source_records": 1},
        "outbox_states": {"direct_records": 0 if schema == 2 else 2,
                          "embedded_source_records": 0 if schema == 2 else 1},
    }
    assert report["known_reference_count"] == (7 if schema == 2 else 10)
    assert report["source"]["journal_schema_version"] == schema
    assert report["source"]["revision"] == (2 if schema == 4 else 1)
    assert report["physical_erasure_supported"] is False
    assert report["erasure_proven"] is False and report["retirement_is_erasure"] is False
    assert report["uninspected_surfaces"] == list(UNINSPECTED_SURFACES)
    raw = json.dumps(report)
    for private in ("private-text-sentinel", "private-key-sentinel", "private-reason-sentinel",
                    "source_trust", "promotion_decision", str(journal.path)):
        assert private not in raw
    assert rows(journal) == before


def test_request_and_returned_views_are_detached(tmp_path):
    journal = seeded(tmp_path)
    intent = canonical_retention_request(0, **SCOPE)
    expected = copy.deepcopy(intent)

    def mutate_caller(point):
        if point == "retention_after_snapshot":
            intent["memory_id"] = 99
            intent["scope"]["tenant_id"] = "other"

    journal._fault_hook = mutate_caller
    report = inspect_memory_retention(journal, request=intent)
    assert report["request"] == expected
    view = journal.read_retention_view()
    assert set(view) == {"store_id", "schema_version", "revision", "state_digest",
                         "state", "snapshot", "receipts", "outbox"}
    state_digest = hashlib.sha256(json.dumps(view["state"], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert view["state_digest"] == state_digest
    view["state"]["items"][0]["text"] = "changed"
    view["snapshot"]["items"].clear()
    view["receipts"].clear()
    report["uninspected_surfaces"].clear()
    report["surfaces"]["current_items"]["direct_records"] = 100
    fresh = inspect(journal)
    assert fresh["known_reference_count"] == 7
    assert fresh["uninspected_surfaces"] == list(UNINSPECTED_SURFACES)
    assert journal.read_retention_view()["state"]["items"][0]["text"] == "private-text-sentinel"


@pytest.mark.parametrize("memory_id", [True, False, -1, 1.0, "1", None])
def test_strict_target_id(memory_id):
    with pytest.raises(ValueError):
        canonical_retention_request(memory_id, **SCOPE)


@pytest.mark.parametrize("change", [
    {"tenant_id": None}, {"tenant_id": ""}, {"tenant_id": " "}, {"tenant_id": True},
    {"client_id": False}, {"session_id": 0}, {"instance_id": 1.0},
    {"tenant_id": "é" * 129}, {"client_id": "é" * 129}, {"session_id": ""},
], ids=[f"scope-{index}" for index in range(10)])
def test_strict_scope_values(change):
    with pytest.raises(ValueError):
        canonical_retention_request(0, **{**SCOPE, **change})


@pytest.mark.parametrize("change", [
    lambda request: request.update(schema_version="future"),
    lambda request: request.update(extra=True),
    lambda request: request.pop("schema_version"),
    lambda request: request["scope"].pop("session_id"),
    lambda request: request["scope"].update(extra="scope"),
    lambda request: request.update(scope=None),
], ids=[f"request-{index}" for index in range(6)])
def test_malformed_request_rejected_before_journal_access(change):
    class Unavailable:
        def read_retention_view(self):
            pytest.fail("Invalid request reached authority")

    request = canonical_retention_request(0, **SCOPE)
    change(request)
    with pytest.raises(ValueError):
        inspect_memory_retention(Unavailable(), request=request)


def test_optional_absence_in_legacy_record_matches_explicit_null_scope(tmp_path):
    legacy = record()
    del legacy["meta"]["session_id"]
    del legacy["meta"]["instance_id"]
    journal = seeded(tmp_path, payload={"items": [legacy]}, acknowledged=legacy)
    report = inspect(journal)
    assert report["known_reference_count"] == 3
    assert report["surfaces"]["receipts"]["direct_records"] == 1
    assert report["surfaces"]["journal_snapshot"]["direct_records"] == 1


@pytest.mark.parametrize("scope", [
    {"tenant_id": "other"}, {"client_id": "other"}, {"session_id": "other"}, {"instance_id": "other"},
], ids=["tenant", "client", "session", "instance"])
def test_wrong_scope_and_missing_memory_have_uniform_failure(tmp_path, scope):
    journal = seeded(tmp_path)
    with pytest.raises(ReceiptMemoryNotFound) as wrong:
        inspect(journal, **scope)
    with pytest.raises(ReceiptMemoryNotFound) as absent:
        inspect(journal, 900)
    assert str(wrong.value) == str(absent.value)


def test_history_only_and_embedded_only_targets_are_reported(tmp_path):
    target = record()
    journal = seeded(tmp_path / "history", payload={"items": [target]}, acknowledged=target)
    journal.save({"items": [], "lineage": [], "next_id": 1}, expected_revision=1)
    report = inspect(journal)
    assert report["surfaces"]["current_items"]["direct_records"] == 0
    assert report["surfaces"]["receipts"]["direct_records"] == 1
    assert report["surfaces"]["journal_snapshot"]["direct_records"] == 1
    assert report["known_reference_count"] == 2
    promoted = derived()
    second = seeded(tmp_path / "embedded", payload={"items": [promoted]}, acknowledged=promoted)
    proof_report = inspect(second)
    assert proof_report["known_reference_count"] == 3
    assert all(counts["direct_records"] == 0 for counts in proof_report["surfaces"].values())


def test_retired_or_updated_container_keeps_historical_source_evidence(tmp_path):
    promoted = derived()
    journal = seeded(tmp_path, payload={"items": [promoted]}, acknowledged=promoted)
    changed = copy.deepcopy(promoted)
    changed["text"] = "updated derived content"
    changed["meta"]["memory_state"] = "deleted"
    changed["meta"]["retirement_decision"] = {"schema_version": "dml-retirement-decision-v1",
        "prior_memory_digest": memory_digest(promoted), "reason": "retired"}
    journal.save({"items": [changed]}, expected_revision=1)
    assert inspect(journal)["known_reference_count"] == 3


def test_foreign_malformed_proofs_are_not_interpreted(tmp_path):
    target = record()
    foreign = record(9, scope={**SCOPE, "tenant_id": "foreign"})
    foreign["meta"]["promotion_decision"] = {"schema_version": "unrecognized", "sources": [target]}
    journal = seeded(tmp_path, 3, payload={"items": [target, foreign]}, acknowledged=target)
    assert inspect(journal)["known_reference_count"] == 4
    with pytest.raises(RetentionInspectionUnsupported):
        inspect(journal, 9, tenant_id="foreign")


def test_arbitrary_metadata_copies_are_explicitly_outside_structured_coverage(tmp_path):
    container = record(9)
    container["meta"]["arbitrary_copy"] = record()
    journal = seeded(tmp_path, payload={"items": [container]}, acknowledged=container)
    with pytest.raises(ReceiptMemoryNotFound):
        inspect(journal)
    report = inspect(journal, 9)
    assert "arbitrary_metadata_and_semantic_copies" in report["uninspected_surfaces"]


def _bad_proof(record, case):
    proof = record["meta"]["promotion_decision"]
    binding = proof["sources"][0]
    source = binding["memory"]
    if case == "version":
        proof["schema_version"] = "future"
    elif case == "extra-field":
        proof["extra"] = True
    elif case == "empty":
        proof["sources"] = []
    elif case == "many":
        proof["sources"] *= 33
    elif case == "duplicate":
        proof["sources"] *= 2
    elif case == "reason":
        proof["reason"] = ""
    elif case == "digest":
        binding["memory_digest"] = "a" * 64
    elif case == "recursive":
        source["meta"]["promotion_decision"] = {"schema_version": "future"}
    elif case == "indirect":
        source["summary_of"] = [8]
    elif case == "source-trust":
        source["meta"]["source_trust"] = "untrusted"
    elif case == "source-scope":
        source["meta"]["tenant_id"] = "foreign"
    elif case == "source-level":
        source["level"] = 1
    elif case == "container-level":
        record["level"] = 2
    elif case == "container-id":
        record["id"] = 0
    elif case == "children":
        record["children"] = [False]
    elif case == "source-vector":
        source["embedding"] = []
    elif case == "output-vector":
        record["embedding"] = [1.0]
    elif case == "oversized":
        source["text"] = "x" * (1024 * 1024)
    if case not in {"digest", "empty"}:
        binding["memory_digest"] = memory_digest(source)


@pytest.mark.parametrize("case", [
    "version", "extra-field", "empty", "many", "duplicate", "reason", "digest", "recursive",
    "indirect", "source-trust", "source-scope", "source-level", "container-level", "container-id",
    "children", "source-vector", "output-vector", "oversized",
])
def test_same_scope_unrecognized_proof_refuses_incomplete_report(tmp_path, case):
    promoted = derived()
    _bad_proof(promoted, case)
    journal = seeded(tmp_path, payload={"items": [promoted]}, acknowledged=promoted)
    with pytest.raises(RetentionInspectionUnsupported) as failure:
        inspect(journal)
    assert str(failure.value) == "Scoped memory has an unsupported retained shape"


def test_empty_receipt_authority_is_uniformly_not_found(tmp_path):
    journal = JournalStateStore(tmp_path / "empty.sqlite3", receipt_mode=True)
    view = journal.read_retention_view()
    assert view["revision"] == 0 and view["snapshot"] is None
    assert view["receipts"] == view["outbox"] == []
    with pytest.raises(ReceiptMemoryNotFound):
        inspect(journal)


def test_schema_one_authority_is_rejected_without_upgrade(tmp_path):
    journal = JournalStateStore(tmp_path / "old.sqlite3")
    before = rows(journal)
    with pytest.raises(JournalSchemaError):
        inspect(journal)
    assert rows(journal) == before and journal.schema_version == 1
