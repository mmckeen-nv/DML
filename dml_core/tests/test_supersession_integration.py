"""Public supersession boundary: exact scope, paired CAS, replay and replicas."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.journal import IdempotencyConflict, JournalIntegrityError, RevisionConflict
from daystrom_dml.provider_server import create_app
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.projection import SQLiteProjection, reconcile
from daystrom_dml.services.receipt_ingestion import ReceiptCommitRejected, ReceiptCommitUncertain
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory


@pytest.fixture(params=[2, 3, 4])
def authority(request, factory, tmp_path):
    version = request.param
    adapter = factory(directory=tmp_path / "source",
                      persistence={"journal": True, "receipts": True, "outbox": version == 3})
    original = append(adapter, text="My previous preference", meta={"provenance": {"source": "trusted-note"}})
    replacement = append(adapter, text="My corrected preference", key="replacement")
    if version == 4:
        adapter.close(persist=False)
        target = tmp_path / "migrated" / "dml_state.sqlite3"
        upgrade_outbox_journal(adapter._journal.path, target)
        adapter = factory(directory=target.parent,
                          persistence={"journal": True, "receipts": True, "outbox": True})
    assert adapter._journal.schema_version == version
    return adapter, original, replacement


def supersession_request(adapter, **overrides):
    records = adapter._journal.load()["items"]
    return {"memory_id": records[0]["id"], "replacement_memory_id": records[1]["id"],
            "expected_memory_digest": memory_digest(records[0]),
            "expected_replacement_digest": memory_digest(records[1]),
            "reason": "Owner corrected this preference", "idempotency_key": "supersede-1",
            "tenant_id": "owner", **overrides}


def seed(factory):
    adapter = factory()
    append(adapter)
    append(adapter, text="Corrected preference", key="replacement")
    return adapter


def test_supersession_preserves_history_replacement_and_replays_without_embedding(authority, factory):
    adapter, ingestion, replacement = authority
    before = deepcopy(adapter._journal.load())
    request = supersession_request(adapter)
    calls = adapter.embedder.calls
    adapter.embedder.fail = True
    receipt = adapter.supersede_memory_receipted(**request)
    after = adapter._journal.load()
    assert adapter.embedder.calls == calls
    expected = deepcopy(before)
    expected["items"][0]["meta"].update({
        "memory_state": "superseded", "superseded_by": request["replacement_memory_id"],
        "supersession_decision": {
            "schema_version": "dml-supersession-decision-v1",
            "prior_memory_digest": request["expected_memory_digest"],
            "replacement_memory_id": request["replacement_memory_id"],
            "replacement_memory_digest": request["expected_replacement_digest"],
            "reason": request["reason"]}})
    assert after == expected
    assert adapter.supersede_memory_receipted(**request) == receipt
    assert append(adapter, text="My previous preference", meta={"provenance": {"source": "trusted-note"}}) == ingestion
    assert append(adapter, text="My corrected preference", key="replacement") == replacement
    directory = adapter._journal.path.parent
    version = adapter._journal.schema_version
    adapter.close(persist=False)
    failing = CountingEmbedder(fail=True)
    restarted = factory(directory=directory, embedder=failing,
                        persistence={"journal": True, "receipts": True, "outbox": version in (3, 4)})
    assert restarted.supersede_memory_receipted(**request) == receipt
    assert failing.calls == 0
    assert restarted._journal.load() == after


@pytest.mark.parametrize("record_index", [0, 1])
def test_either_record_change_rejects_stale_decision_without_mutation(authority, record_index):
    adapter, _, _ = authority
    request = supersession_request(adapter)
    revision, state = adapter._journal.read_snapshot()
    state["items"][record_index]["meta"]["authority"] = "changed-owner"
    adapter._journal.save(state, expected_revision=revision, operation="test-authority-change")
    before = adapter._journal.verified_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        adapter.supersede_memory_receipted(**request)
    assert adapter._journal.verified_snapshot() == before
    adapter.supersede_memory_receipted(**supersession_request(adapter))
    with pytest.raises(IdempotencyConflict):
        adapter.supersede_memory_receipted(**request)


def test_failed_hydration_never_serves_stale_source_and_refresh_repairs(authority, monkeypatch):
    adapter, _, _ = authority
    assert len(adapter.retrieve_context("preference", tenant_id="owner")["items"]) == 2
    request = supersession_request(adapter)
    original_import = adapter.store.import_state
    adapter.query_cache.get("old-query", lambda _: np.ones(4))

    def fail(_payload):
        raise RuntimeError("private tenant memory")

    monkeypatch.setattr(adapter.store, "import_state", fail)
    receipt = adapter.supersede_memory_receipted(**request)
    assert adapter._last_observed_state is None
    assert adapter.durability_status() == {"status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert adapter.supersede_memory_receipted(**request) == receipt
    with pytest.raises(RuntimeError, match="private tenant memory"):
        adapter.retrieve_context("preference", tenant_id="owner")
    monkeypatch.setattr(adapter.store, "import_state", original_import)
    report = adapter.retrieve_context("preference", tenant_id="owner")
    assert [item["id"] for item in report["items"]] == [str(request["replacement_memory_id"])]
    assert "My previous preference" not in report["raw_context"]
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert not adapter.query_cache.values


def test_snapshot_delta_and_outbox_preserve_link_and_retrieve_only_replacement(authority, tmp_path):
    adapter, _, _ = authority
    projection = SQLiteProjection(tmp_path / "projection" / "state.sqlite")
    reconcile(adapter._journal, projection)
    assert len(adapter.query_projection("preference", backend=projection, tenant_id="owner", as_of=100)["results"]) == 2
    request = supersession_request(adapter)
    replacement_before = deepcopy(adapter._journal.load()["items"][1])
    adapter.supersede_memory_receipted(**request)
    assert adapter.sync_projection(projection)["matches_pinned_source"] is True
    projected = projection.read()["items"]
    assert projected[0]["meta"]["memory_state"] == "superseded"
    assert projected[0]["meta"]["superseded_by"] == request["replacement_memory_id"]
    assert projected[1] == replacement_before
    results = adapter.query_projection("preference", backend=projection, tenant_id="owner", as_of=100)["results"]
    assert [item["memory"]["id"] for item in results] == [request["replacement_memory_id"]]
    report = adapter.retrieve_context("preference", tenant_id="owner")
    assert [item["id"] for item in report["items"]] == [str(request["replacement_memory_id"])]
    rebuilt = SQLiteProjection(tmp_path / "rebuilt" / "state.sqlite")
    reconcile(adapter._journal, rebuilt)
    assert rebuilt.read()["items"] == projected
    if adapter._journal.schema_version in (3, 4):
        consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
        assert adapter.deliver_outbox(consumer)["backlog"] == 0
        event = consumer.read()["last_event"]
        assert event["state"]["items"] == adapter._journal.load()["items"]
        assert event["receipt"]["key"] == request["idempotency_key"]
        assert event["operation"] == "supersede-receipt-v1"
        decision = adapter._journal.decisions()[-1]
        assert decision["deleted"] == []
        assert [(change["bucket"], change["id"]) for change in decision["changed"]] == [
            ("items", str(request["memory_id"]))]


@pytest.mark.parametrize("guard", ["disabled", "mirror", "nested", "legacy-schema"])
def test_supersession_preserves_explicit_mutation_guards(factory, guard):
    adapter = seed(factory)
    request = supersession_request(adapter)
    before = adapter._journal.verified_snapshot()
    if guard == "disabled":
        adapter._receipts_enabled = False
    elif guard == "mirror":
        adapter.mirror_agentic_memory_to_rag = True
    elif guard == "legacy-schema":
        adapter._journal._schema_version = 1
    if guard == "nested":
        with adapter.mutation_transaction("supersession-cannot-nest"):
            with pytest.raises(ValueError, match="nested"):
                adapter.supersede_memory_receipted(**request)
    else:
        with pytest.raises(ValueError):
            adapter.supersede_memory_receipted(**request)
    if guard == "legacy-schema":
        adapter._journal._schema_version = 2
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
@pytest.mark.parametrize("target", ["memory_id", "replacement_memory_id"])
def test_http_exact_scope_and_absent_targets_are_indistinguishable(factory, monkeypatch, field, target):
    monkeypatch.setenv("DML_API_TOKEN", "supersession-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    request = supersession_request(adapter)
    before = adapter._journal.verified_snapshot()
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    assert client.post("/api/memory/supersede/receipt", json=request).status_code == 401
    headers = {"Authorization": "Bearer supersession-test-token"}
    wrong = client.post("/api/memory/supersede/receipt", json={**request, field: "other"}, headers=headers)
    absent = client.post("/api/memory/supersede/receipt", json={**request, target: 999}, headers=headers)
    assert wrong.status_code == absent.status_code == 404
    assert wrong.json() == absent.json() == {"detail": {"code": "receipt_memory_not_found"}}
    assert adapter._journal.verified_snapshot() == before
    first = client.post("/api/memory/supersede/receipt", json=request, headers=headers)
    assert first.status_code == 200, first.text
    assert client.post("/api/memory/supersede/receipt", json=request, headers=headers).json() == first.json()
    conflict = client.post("/api/memory/supersede/receipt", json={**request, "reason": "Changed"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("memory_id", "0"), ("memory_id", 0.0), ("memory_id", -1),
    ("replacement_memory_id", True), ("replacement_memory_id", "1"),
    ("replacement_memory_id", 1.0), ("replacement_memory_id", -1),
    ("expected_memory_digest", 123), ("expected_memory_digest", "not-a-digest"),
    ("expected_replacement_digest", 123), ("expected_replacement_digest", "not-a-digest"),
    ("reason", True), ("reason", 123), ("reason", ""),
    ("idempotency_key", True), ("tenant_id", 1), ("client_id", False),
    ("session_id", 1), ("instance_id", 1), ("unexpected", "must reject"),
])
def test_http_supersession_strictly_rejects_coercion_and_unknown_fields(factory, monkeypatch, field, value):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    before = adapter._journal.verified_snapshot()
    request = supersession_request(adapter, **{field: value})
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post("/api/memory/supersede/receipt", json=request)
    assert response.status_code == 422, response.text
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("error,status,code,retry", [
    (ReceiptMemoryNotFound, 404, "receipt_memory_not_found", False),
    (ReceiptLifecycleConflict, 409, "receipt_lifecycle_conflict", False),
    (IdempotencyConflict, 409, "idempotency_conflict", False),
    (RevisionConflict, 409, "revision_conflict", True),
    (TimeoutError, 503, "receipt_ownership_unavailable", True),
    (ReceiptCommitUncertain, 503, "receipt_outcome_unavailable", True),
    (JournalIntegrityError, 503, "receipt_outcome_unavailable", True),
    (ReceiptCommitRejected, 503, "receipt_not_committed", True),
    (ValueError, 400, "invalid_or_unsupported_receipt_request", False),
])
def test_http_supersession_errors_are_sanitized(factory, monkeypatch, error, status, code, retry):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    request = supersession_request(adapter)

    def fail(**_kwargs):
        raise error("private tenant record and storage credential")

    monkeypatch.setattr(adapter, "supersede_memory_receipted", fail)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post("/api/memory/supersede/receipt", json=request)
    assert response.status_code == status
    expected = {"code": code, **({"retry_same_key": True} if retry else {})}
    assert response.json() == {"detail": expected}
    assert "private" not in response.text
