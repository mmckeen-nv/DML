"""Independent adapter/API oracles for the append-only receipt boundary."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import IdempotencyConflict, JournalIntegrityError
from daystrom_dml.provider_server import create_app
from daystrom_dml.services.receipt_ingestion import (
    ReceiptCapacityError,
    ReceiptCommitRejected,
    ReceiptCommitUncertain,
    ReceiptEmbeddingError,
)


class CountingEmbedder:
    receipt_embedding_identity = "test-v1"

    def __init__(self, callback=None, fail=False):
        self.calls = 0
        self.callback = callback
        self.fail = fail

    def embed(self, text):
        self.calls += 1
        if self.callback:
            self.callback()
        if self.fail:
            raise RuntimeError("private backend credential")
        return np.ones(4, dtype=np.float32)


@pytest.fixture
def factory(tmp_path):
    instances = []

    def build(*, embedder=None, directory=None, **overrides):
        config = {
            "storage_dir": str(directory or tmp_path),
            "model_name": "dummy", "embedding_model": None,
            "checkpoint_interval_seconds": 0,
            "persistence": {"journal": True, "receipts": True, "enable": False},
            "rag_store": {"enable": False},
            "dpm": {"enable": False, "include_in_context": False},
            **overrides,
        }
        instance = DMLAdapter(config_overrides=config, embedder=embedder or CountingEmbedder(),
                              start_aging_loop=False)
        instances.append(instance)
        return instance

    yield build
    for instance in reversed(instances):
        instance.close(persist=False)


def append(adapter, text="Remember my preference", key="request-1", **kwargs):
    return adapter.ingest_memory_receipted(text, idempotency_key=key,
                                         tenant_id=kwargs.pop("tenant_id", "owner"), **kwargs)


def test_restart_retry_returns_original_receipt_without_embedding(factory):
    first = factory()
    receipt = append(first)
    first.close(persist=False)
    failing = CountingEmbedder(fail=True)
    restarted = factory(embedder=failing)
    assert append(restarted) == receipt
    assert failing.calls == 0
    assert len(restarted._journal.load()["items"]) == 1


def test_conflicting_retry_fails_before_embedding_and_preserves_commit(factory):
    instance = factory()
    receipt = append(instance)
    count = instance.embedder.calls
    with pytest.raises(IdempotencyConflict):
        append(instance, text="A changed preference")
    assert instance.embedder.calls == count
    assert append(instance) == receipt
    assert len(instance._journal.load()["items"]) == 1


@pytest.mark.parametrize("member", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_each_scope_member_namespaces_the_same_key(factory, member):
    instance = factory()
    first = append(instance)
    second = append(instance, **{member: "different"})
    assert first["scope"] != second["scope"]
    assert first["result"]["memory"]["id"] != second["result"]["memory"]["id"]
    assert append(instance) == first
    assert append(instance, **{member: "different"}) == second


def test_nested_metadata_is_frozen_before_embedding(factory):
    metadata = {"source": {"tags": ["original"]}}
    original = deepcopy(metadata)
    embedder = CountingEmbedder(callback=lambda: metadata["source"]["tags"].append("changed"))
    instance = factory(embedder=embedder)
    receipt = append(instance, meta=metadata)
    assert receipt["result"]["memory"]["meta"]["source"] == original["source"]
    assert append(instance, meta=original) == receipt
    assert embedder.calls == 1


def test_receipt_is_refused_inside_legacy_atomic_batch(factory):
    instance = factory()
    before = instance._journal.revision
    with pytest.raises(ValueError, match="legacy writes"):
        with instance.atomic_batch("invalid-receipt-nesting"):
            append(instance)
    assert instance.embedder.calls == 0
    assert instance._journal.revision == before
    assert instance._journal.load()["items"] == []


def test_receipt_refuses_nested_transaction_before_prelookup_or_embedding(factory):
    instance = factory()
    receipt = append(instance)
    calls = instance.embedder.calls
    with instance.mutation_transaction("receipt-cannot-nest"):
        with pytest.raises(ValueError, match="nested"):
            append(instance)
    assert instance.embedder.calls == calls
    assert append(instance) == receipt


def test_receipt_refuses_external_rag_mirroring(factory):
    instance = factory(mirror_agentic_memory_to_rag=True)
    with pytest.raises(ValueError, match="mirroring"):
        append(instance)
    assert instance.embedder.calls == 0
    assert instance._journal.load()["items"] == []


def test_capacity_rejection_does_not_evict_acknowledged_memory(factory):
    instance = factory()
    instance.store.capacity = 1
    receipt = append(instance)
    before = instance._journal.read_snapshot()
    with pytest.raises(ReceiptCapacityError):
        append(instance, key="second", text="Another memory")
    assert instance._journal.read_snapshot() == before
    assert append(instance) == receipt


def test_hydration_failure_keeps_receipt_and_refresh_repairs_runtime(factory, monkeypatch):
    instance = factory()
    import_state = instance.store.import_state

    def broken_import(_payload):
        raise RuntimeError("private memory text must not appear in health")

    monkeypatch.setattr(instance.store, "import_state", broken_import)
    receipt = append(instance)
    assert instance._journal.load()["items"][0]["text"] == "Remember my preference"
    assert instance.durability_status() == {
        "status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert append(instance) == receipt
    monkeypatch.setattr(instance.store, "import_state", import_state)
    assert instance.refresh_if_changed()
    assert instance.durability_status() == {"status": "ok", "failures": {}}
    assert [item.text for item in instance.store.items()] == ["Remember my preference"]


def test_historical_receipt_survives_deletion_and_ids_are_not_reused(factory):
    instance = factory()
    receipt = append(instance)
    revision, payload = instance._journal.read_snapshot()
    payload["items"] = []
    payload["lineage"] = []
    instance._journal.save(payload, expected_revision=revision, operation="explicit-delete")
    instance.refresh_if_changed()
    assert instance.store.items() == []
    assert append(instance) == receipt
    assert instance._journal.load()["items"] == []
    assert instance.store.export_state()["next_id"] > receipt["result"]["memory"]["id"]
    later = append(instance, key="later", text="A later memory")
    assert later["result"]["memory"]["id"] > receipt["result"]["memory"]["id"]


def test_receipt_journal_refuses_lattice_only_checkpoint_and_export(factory, tmp_path):
    instance = factory()
    receipt = append(instance)
    with pytest.raises(ValueError, match="receipt"):
        instance.create_checkpoint()
    target = tmp_path / "lossy-export.json"
    with pytest.raises(ValueError, match="receipt"):
        instance._journal.export_snapshot(target)
    assert not target.exists()
    assert append(instance) == receipt


@pytest.mark.parametrize("point", ["after_receipt", "before_commit"])
def test_precommit_failure_is_absent_and_same_key_retry_can_commit(factory, point):
    instance = factory()

    def fail(event):
        if event == point:
            raise OSError("injected disk failure")

    instance._journal._fault_hook = fail
    with pytest.raises(ReceiptCommitRejected):
        append(instance)
    assert instance._journal.load()["items"] == []
    instance._journal._fault_hook = lambda _event: None
    receipt = append(instance)
    assert append(instance) == receipt
    assert len(instance._journal.load()["items"]) == 1


def test_postcommit_failure_resolves_durable_receipt_without_compensation(factory):
    instance = factory()

    def fail(event):
        if event == "after_commit":
            raise OSError("lost commit acknowledgement")

    instance._journal._fault_hook = fail
    receipt = append(instance)
    assert receipt["revision"] == instance._journal.revision
    assert append(instance) == receipt
    assert len(instance._journal.load()["items"]) == 1
    assert len(instance.store.items()) == 1


def test_unreadable_postcommit_outcome_is_uncertain_then_retry_recovers(factory, monkeypatch):
    instance = factory()
    committed = False
    lookup = instance._journal.lookup_receipt

    def fault(event):
        nonlocal committed
        if event == "after_commit":
            committed = True
            raise OSError("lost acknowledgement")

    def unavailable(**kwargs):
        if committed:
            raise OSError("temporarily unreadable database")
        return lookup(**kwargs)

    instance._journal._fault_hook = fault
    monkeypatch.setattr(instance._journal, "lookup_receipt", unavailable)
    with pytest.raises(ReceiptCommitUncertain):
        append(instance)
    monkeypatch.setattr(instance._journal, "lookup_receipt", lookup)
    receipt = append(instance)
    assert receipt["revision"] == instance._journal.read_snapshot()[0]
    assert len(instance._journal.load()["items"]) == 1


def test_http_auth_replay_conflict_and_unknown_fields(factory, monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "receipt-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    instance = factory()
    client = TestClient(create_app(adapter_factory=lambda: instance))
    request = {"text": "HTTP memory", "idempotency_key": "http-key", "tenant_id": "owner"}
    assert client.post("/api/remember/receipt", json=request).status_code == 401
    assert instance._journal.load()["items"] == []
    headers = {"Authorization": "Bearer receipt-test-token"}
    first = client.post("/api/remember/receipt", json=request, headers=headers)
    assert first.status_code == 200
    assert client.post("/api/remember/receipt", json=request, headers=headers).json() == first.json()
    conflict = client.post("/api/remember/receipt", json={**request, "text": "Changed"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"
    unknown = client.post("/api/remember/receipt", json={**request, "silently_ignored": True}, headers=headers)
    assert unknown.status_code == 422
    assert len(instance._journal.load()["items"]) == 1


@pytest.mark.parametrize("error, code", [
    (ReceiptCommitUncertain, "receipt_outcome_unavailable"),
    (ReceiptCommitRejected, "receipt_not_committed"),
    (ReceiptEmbeddingError, "embedding_unavailable"),
    (JournalIntegrityError, "receipt_outcome_unavailable"),
])
def test_http_storage_failures_are_sanitized_and_retryable(factory, monkeypatch, error, code):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    instance = factory()

    def fail(**_kwargs):
        raise error("private tenant memory and backend credential")

    monkeypatch.setattr(instance, "ingest_memory_receipted", fail)
    client = TestClient(create_app(adapter_factory=lambda: instance))
    response = client.post("/api/remember/receipt", json={"text": "memory", "idempotency_key": "k"})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": code, "retry_same_key": True}}
    assert "private" not in response.text


def test_http_lock_timeout_is_sanitized_retryable_and_does_not_commit(factory, monkeypatch):
    from daystrom_dml.services import receipt_ingestion

    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    instance = factory()

    @contextmanager
    def timeout(*_args, **_kwargs):
        raise TimeoutError("private storage path")
        yield

    monkeypatch.setattr(receipt_ingestion, "store_write_lock", timeout)
    client = TestClient(create_app(adapter_factory=lambda: instance), raise_server_exceptions=False)
    response = client.post("/api/remember/receipt", json={"text": "memory", "idempotency_key": "k"})
    assert response.status_code == 503
    assert response.json()["detail"]["retry_same_key"] is True
    assert "private" not in response.text
    assert instance._journal.load()["items"] == []
