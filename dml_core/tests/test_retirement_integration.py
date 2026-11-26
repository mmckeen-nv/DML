"""Public-boundary retirement oracles: scope, retries, hydration and replicas."""
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
    receipt = append(adapter, meta={"provenance": {"source": "trusted-note"}})
    if version == 4:
        adapter.close(persist=False)
        target = tmp_path / "migrated" / "dml_state.sqlite3"
        upgrade_outbox_journal(adapter._journal.path, target)
        adapter = factory(directory=target.parent,
                          persistence={"journal": True, "receipts": True, "outbox": True})
    assert adapter._journal.schema_version == version
    return adapter, receipt


def retirement_request(adapter, **overrides):
    record = adapter._journal.load()["items"][0]
    return {"memory_id": record["id"], "expected_memory_digest": memory_digest(record),
            "reason": "Owner withdrew this preference", "idempotency_key": "retire-1",
            "tenant_id": "owner", **overrides}


def test_retirement_retains_record_history_and_replays_without_embedder(authority, factory):
    adapter, ingestion = authority
    before = deepcopy(adapter._journal.load())
    request = retirement_request(adapter)
    calls = adapter.embedder.calls
    adapter.embedder.fail = True
    receipt = adapter.retire_memory_receipted(**request)
    after = adapter._journal.load()
    assert adapter.embedder.calls == calls
    assert after["next_id"] == before["next_id"]
    assert after["lineage"] == before["lineage"]
    expected = before["items"][0]
    expected["meta"]["memory_state"] = "deleted"
    expected["meta"]["retirement_decision"] = {
        "schema_version": "dml-retirement-decision-v1",
        "prior_memory_digest": request["expected_memory_digest"], "reason": request["reason"]}
    assert after["items"] == [expected]
    assert adapter.retire_memory_receipted(**request) == receipt
    assert append(adapter, meta={"provenance": {"source": "trusted-note"}}) == ingestion
    directory = adapter._journal.path.parent
    version = adapter._journal.schema_version
    adapter.close(persist=False)
    failing = CountingEmbedder(fail=True)
    restarted = factory(directory=directory, embedder=failing,
                        persistence={"journal": True, "receipts": True, "outbox": version in (3, 4)})
    assert restarted.retire_memory_receipted(**request) == receipt
    assert failing.calls == 0
    assert restarted._journal.load() == after


def test_target_metadata_change_rejects_stale_decision_without_mutation(authority):
    adapter, _ = authority
    request = retirement_request(adapter)
    revision, state = adapter._journal.read_snapshot()
    state["items"][0]["meta"]["authority"] = "changed-owner"
    adapter._journal.save(state, expected_revision=revision, operation="test-authority-change")
    before = adapter._journal.verified_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        adapter.retire_memory_receipted(**request)
    assert adapter._journal.verified_snapshot() == before
    fresh = retirement_request(adapter)
    adapter.retire_memory_receipted(**fresh)
    with pytest.raises(IdempotencyConflict):
        adapter.retire_memory_receipted(**request)


def test_hydration_failure_returns_durable_receipt_and_refresh_suppresses_memory(authority, monkeypatch):
    adapter, _ = authority
    assert adapter.retrieve_context("preference", tenant_id="owner")["items"]
    request = retirement_request(adapter)
    original_import = adapter.store.import_state
    adapter.query_cache.get("old-query", lambda _: np.ones(4))

    def fail(_payload):
        raise RuntimeError("private tenant memory")

    monkeypatch.setattr(adapter.store, "import_state", fail)
    receipt = adapter.retire_memory_receipted(**request)
    assert adapter._last_observed_state is None
    assert adapter.durability_status() == {"status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert adapter.retire_memory_receipted(**request) == receipt
    # A failed refresh must not return the stale in-memory record.
    with pytest.raises(RuntimeError, match="private tenant memory"):
        adapter.retrieve_context("preference", tenant_id="owner")
    monkeypatch.setattr(adapter.store, "import_state", original_import)
    report = adapter.retrieve_context("preference", tenant_id="owner")
    assert report["items"] == [] and report["raw_context"] == ""
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert not adapter.query_cache.values


def test_projection_delta_and_outbox_preserve_tombstone_and_filter_retrieval(authority, tmp_path):
    adapter, _ = authority
    projection = SQLiteProjection(tmp_path / "projection" / "state.sqlite")
    reconcile(adapter._journal, projection)
    assert adapter.query_projection("preference", backend=projection, tenant_id="owner", as_of=100)["results"]
    request = retirement_request(adapter)
    adapter.retire_memory_receipted(**request)
    assert adapter.sync_projection(projection)["matches_pinned_source"] is True
    assert projection.read()["items"][0]["meta"]["memory_state"] == "deleted"
    assert adapter.query_projection("preference", backend=projection, tenant_id="owner", as_of=100)["results"] == []
    assert adapter.retrieve_context("preference", tenant_id="owner")["items"] == []
    rebuilt = SQLiteProjection(tmp_path / "rebuilt" / "state.sqlite")
    reconcile(adapter._journal, rebuilt)
    assert rebuilt.read()["items"] == projection.read()["items"]
    if adapter._journal.schema_version in (3, 4):
        consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
        assert adapter.deliver_outbox(consumer)["backlog"] == 0
        event = consumer.read()["last_event"]
        assert event["state"]["items"] == adapter._journal.load()["items"]
        assert event["receipt"] is not None
        assert event["operation"] == "retire-receipt-v1"
        assert event["receipt"]["key"] == request["idempotency_key"]
        decision = adapter._journal.decisions()[-1]
        assert decision["deleted"] == []
        assert [(change["bucket"], change["id"]) for change in decision["changed"]] == [
            ("items", str(request["memory_id"]))]


@pytest.mark.parametrize("guard", ["disabled", "mirror", "nested"])
def test_retirement_preserves_explicit_mutation_guards(factory, guard):
    adapter = factory()
    append(adapter)
    request = retirement_request(adapter)
    before = adapter._journal.verified_snapshot()
    if guard == "disabled":
        adapter._receipts_enabled = False
    elif guard == "mirror":
        adapter.mirror_agentic_memory_to_rag = True
    if guard == "nested":
        with adapter.mutation_transaction("retirement-cannot-nest"):
            with pytest.raises(ValueError, match="nested"):
                adapter.retire_memory_receipted(**request)
    else:
        with pytest.raises(ValueError):
            adapter.retire_memory_receipted(**request)
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_http_scope_mismatch_matches_absent_target_and_auth_blocks_mutation(factory, monkeypatch, field):
    monkeypatch.setenv("DML_API_TOKEN", "retirement-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = factory()
    append(adapter)
    request = retirement_request(adapter)
    before = adapter._journal.verified_snapshot()
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    assert client.post("/api/memory/retire/receipt", json=request).status_code == 401
    headers = {"Authorization": "Bearer retirement-test-token"}
    wrong = client.post("/api/memory/retire/receipt", json={**request, field: "other"}, headers=headers)
    absent = client.post("/api/memory/retire/receipt", json={**request, "memory_id": 999}, headers=headers)
    assert wrong.status_code == absent.status_code == 404
    assert wrong.json() == absent.json() == {"detail": {"code": "receipt_memory_not_found"}}
    assert adapter._journal.verified_snapshot() == before
    first = client.post("/api/memory/retire/receipt", json=request, headers=headers)
    assert first.status_code == 200, first.text
    assert client.post("/api/memory/retire/receipt", json=request, headers=headers).json() == first.json()
    conflict = client.post("/api/memory/retire/receipt", json={**request, "reason": "Changed"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("memory_id", "0"), ("memory_id", 0.0), ("memory_id", -1),
    ("expected_memory_digest", 123), ("expected_memory_digest", "not-a-digest"),
    ("reason", True), ("reason", 123), ("reason", ""),
    ("idempotency_key", True), ("tenant_id", 1), ("client_id", False),
    ("session_id", 1), ("instance_id", 1), ("unexpected", "must reject"),
])
def test_http_retirement_strictly_rejects_coercion_and_unknown_fields(factory, monkeypatch, field, value):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = factory()
    append(adapter)
    before = adapter._journal.verified_snapshot()
    request = retirement_request(adapter, **{field: value})
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    response = client.post("/api/memory/retire/receipt", json=request)
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
def test_http_retirement_errors_are_sanitized(factory, monkeypatch, error, status, code, retry):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = factory()
    append(adapter)
    request = retirement_request(adapter)

    def fail(**_kwargs):
        raise error("private tenant record and storage credential")

    monkeypatch.setattr(adapter, "retire_memory_receipted", fail)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post("/api/memory/retire/receipt", json=request)
    assert response.status_code == status
    expected = {"code": code, **({"retry_same_key": True} if retry else {})}
    assert response.json() == {"detail": expected}
    assert "private" not in response.text
