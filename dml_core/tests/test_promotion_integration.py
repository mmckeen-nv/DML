"""Public explicit derivation: authority guards, immutable lineage and delivery."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.journal import IdempotencyConflict, JournalIntegrityError, RevisionConflict
from daystrom_dml.provider_server import create_app
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal
from daystrom_dml.services.projection import SQLiteProjection, reconcile
from daystrom_dml.services.receipt_ingestion import (
    ReceiptCapacityError, ReceiptCommitRejected, ReceiptCommitUncertain,
    ReceiptEmbeddingCompatibilityError, ReceiptEmbeddingError,
)
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from daystrom_dml.store_lock import store_write_lock
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory
SOURCE = "The owner prefers green notebooks"
SECOND = "The owner uses notebooks for meeting notes"
DERIVED = "The owner prefers green notebooks for meeting notes"
ENDPOINT = "/api/memory/promote/receipt"
META = {"source_trust": "trusted", "authority": 3, "namespace": "personal"}


class TextEmbedder(CountingEmbedder):
    receipt_embedding_identity = "promotion-test-v1"

    def embed(self, text):
        super().embed(text)
        return np.array([0, 1, 0, 0] if text == DERIVED else [1, 0, 0, 0], dtype=np.float32)


class UnavailableEmbedder:
    @property
    def embed(self):
        raise AssertionError("Historical replay contacted the embedding backend")

    @property
    def receipt_embedding_identity(self):
        raise AssertionError("Historical replay contacted the embedding identity")


def seed(factory, *, multiple=False, directory=None, persistence=None, scope=None):
    adapter = factory(directory=directory, embedder=TextEmbedder(),
                      **({"persistence": persistence} if persistence else {}))
    scope = scope or {"tenant_id": "owner"}
    receipts = [append(adapter, text=SOURCE, meta={**META, "summary": SOURCE,
                "source": "first", "provenance": {"document": "first"}}, **scope)]
    if multiple:
        receipts.append(append(adapter, text=SECOND, key="source-2", meta={**META, "summary": SECOND,
                        "source": "second", "provenance": {"document": "second"}}, **scope))
        revision, state = adapter._journal.read_snapshot()
        for index, record in enumerate(state["items"]):
            record["meta"]["no_merge"] = False
            record["timestamp"] = 10.0 + index
            record["salience"] = 0.9 - index * 0.2
            record["fidelity"] = 0.8 - index * 0.2
        # Explicit imported merge policy. Normal receipt ingestion never relaxes it.
        adapter._journal.save(state, expected_revision=revision, operation="test-import-merge-policy")
        adapter.refresh_if_changed()
    return adapter, receipts


@pytest.fixture(params=[2, 3, 4])
def authority(request, factory, tmp_path):
    version = request.param
    adapter, receipts = seed(factory, multiple=True, directory=tmp_path / "authority",
        persistence={"journal": True, "receipts": True, "outbox": version == 3})
    if version == 4:
        adapter.close(persist=False)
        target = tmp_path / "migrated" / "dml_state.sqlite3"
        upgrade_outbox_journal(adapter._journal.path, target)
        adapter = factory(directory=target.parent, embedder=TextEmbedder(),
                          persistence={"journal": True, "receipts": True, "outbox": True})
    assert adapter._journal.schema_version == version
    return adapter, receipts


def promotion_request(adapter, *, ids=None, **overrides):
    records = adapter._journal.load()["items"]
    if ids is not None:
        records = [record for record in records if record["id"] in ids]
    return {"sources": [{"memory_id": record["id"], "expected_memory_digest": memory_digest(record)}
                        for record in records],
            "text": DERIVED, "reason": "Owner explicitly approved this derived statement",
            "idempotency_key": "promotion-1", "tenant_id": "owner", **overrides}


def test_merge_preserves_sources_and_adds_one_conservative_derived_memory(authority):
    adapter, _ = authority
    before = deepcopy(adapter._journal.load())
    request = promotion_request(adapter)
    count = adapter.embedder.calls
    receipt = adapter.promote_memories_receipted(**request)
    after = adapter._journal.load()
    result = receipt["result"]["memory"]
    assert after["items"][:-1] == before["items"]
    assert after["items"][-1] == result
    assert after["lineage"] == before["lineage"]
    assert after["embedding_contract"] == before["embedding_contract"]
    assert after["next_id"] == result["id"] + 1
    assert result["id"] >= before["next_id"]
    assert result["text"] == DERIVED
    assert result["embedding"] == [0.0, 1.0, 0.0, 0.0]
    assert result["level"] == 1
    assert result["summary_of"] == result["children"] == [item["id"] for item in before["items"]]
    assert result["timestamp"] == min(item["timestamp"] for item in before["items"])
    assert result["salience"] == min(item["salience"] for item in before["items"])
    assert result["fidelity"] == min(item["fidelity"] for item in before["items"])
    assert all(result["meta"][name] == value for name, value in META.items())
    assert result["meta"]["no_merge"] is True
    assert not set(result["meta"]) & {"summary", "source", "source_id", "source_ids", "provenance", "content_update_decision"}
    assert result["meta"]["promotion_decision"] == {
        "schema_version": "dml-promotion-decision-v1", "reason": request["reason"],
        "sources": [{"memory_digest": memory_digest(record), "memory": record}
                    for record in before["items"]]}
    assert adapter.embedder.calls == count + 1
    assert adapter.promote_memories_receipted(**request) == receipt
    assert adapter.embedder.calls == count + 1
    assert adapter._journal.load() == after


def test_singleton_public_receipt_can_promote_without_relaxing_no_merge(factory):
    adapter, receipts = seed(factory)
    original = receipts[0]["result"]["memory"]
    receipt = adapter.promote_memories_receipted(**promotion_request(adapter))
    result = receipt["result"]["memory"]
    assert original["meta"]["no_merge"] is True
    assert result["meta"]["no_merge"] is True
    assert result["summary_of"] == [original["id"]]
    assert adapter._journal.load()["items"] == [original, result]
    assert append(adapter, text=SOURCE, meta={**META, "summary": SOURCE,
                  "source": "first", "provenance": {"document": "first"}}) == receipts[0]


def test_retry_after_restart_and_later_source_retirement_uses_immutable_history(authority, factory):
    adapter, _ = authority
    request = promotion_request(adapter)
    receipt = adapter.promote_memories_receipted(**request)
    source = adapter._journal.load()["items"][0]
    adapter.retire_memory_receipted(source["id"], expected_memory_digest=memory_digest(source),
        reason="The source is no longer current", idempotency_key="retire-source", tenant_id="owner")
    before = adapter._journal.verified_snapshot()
    directory, version = adapter._journal.path.parent, adapter._journal.schema_version
    adapter.close(persist=False)
    restarted = factory(directory=directory, embedder=UnavailableEmbedder(),
        persistence={"journal": True, "receipts": True, "outbox": version in (3, 4)})
    restarted.store.capacity = 1
    assert restarted.promote_memories_receipted(**request) == receipt
    assert restarted._journal.verified_snapshot() == before
    assert receipt["result"]["memory"]["text"] == DERIVED


def test_hydration_failure_keeps_success_and_refresh_restores_derived_context(authority, monkeypatch):
    adapter, _ = authority
    request = promotion_request(adapter)
    import_state = adapter.store.import_state
    adapter.query_cache.get("old-query", lambda _: np.ones(4))

    def fail(_payload):
        raise RuntimeError("private memory and storage credential")

    monkeypatch.setattr(adapter.store, "import_state", fail)
    receipt = adapter.promote_memories_receipted(**request)
    assert adapter._last_observed_state is None
    assert adapter.durability_status() == {"status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert adapter.promote_memories_receipted(**request) == receipt
    with pytest.raises(RuntimeError, match="private memory"):
        adapter.retrieve_context(DERIVED, tenant_id="owner")
    monkeypatch.setattr(adapter.store, "import_state", import_state)
    report = adapter.retrieve_context(DERIVED, tenant_id="owner")
    assert DERIVED in report["raw_context"]
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert not adapter.query_cache.values
    assert adapter.store.items()[-1].cached_summary() == DERIVED


def test_projection_context_and_outbox_observe_sources_and_derived_record_together(authority, tmp_path):
    adapter, _ = authority
    before = deepcopy(adapter._journal.load()["items"])
    projection = SQLiteProjection(tmp_path / "projection" / "state.sqlite")
    reconcile(adapter._journal, projection)
    receipt = adapter.promote_memories_receipted(**promotion_request(adapter))
    expected = before + [receipt["result"]["memory"]]
    assert adapter.sync_projection(projection)["matches_pinned_source"] is True
    assert projection.read()["items"] == expected
    results = adapter.query_projection(DERIVED, backend=projection, tenant_id="owner", as_of=100)["results"]
    assert results[0]["memory"] == expected[-1]
    assert {row["memory"]["id"] for row in results} == {record["id"] for record in expected}
    report = adapter.retrieve_context(DERIVED, tenant_id="owner")
    assert DERIVED in report["raw_context"]
    assert adapter.store.items()[-1].cached_summary() == DERIVED
    rebuilt = SQLiteProjection(tmp_path / "rebuilt" / "state.sqlite")
    reconcile(adapter._journal, rebuilt)
    assert rebuilt.read()["items"] == expected
    if adapter._journal.schema_version in (3, 4):
        consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
        assert adapter.deliver_outbox(consumer)["backlog"] == 0
        event = consumer.read()["last_event"]
        assert event["operation"] == "promote-receipt-v1"
        assert event["state"]["items"] == expected
        binding = event["receipt"]
        assert binding == {"scope": receipt["scope"], "key": receipt["key"],
            "request_digest": receipt["request_digest"], "digest": hashlib.sha256(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()}
    decision = adapter._journal.decisions()[-1]
    assert decision["deleted"] == []
    assert [(change["bucket"], change["id"]) for change in decision["changed"]] == [
        ("items", str(receipt["result"]["memory"]["id"]))]


@pytest.mark.parametrize("guard", ["disabled", "mirror", "nested", "legacy-schema"])
def test_explicit_receipt_guards_are_preserved(factory, guard):
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    before = adapter._journal.verified_snapshot()
    if guard == "disabled":
        adapter._receipts_enabled = False
    elif guard == "mirror":
        adapter.mirror_agentic_memory_to_rag = True
    elif guard == "legacy-schema":
        adapter._journal._schema_version = 1
    adapter.embedder = UnavailableEmbedder()
    if guard == "nested":
        with adapter.mutation_transaction("promotion-cannot-nest"):
            with pytest.raises(ValueError, match="nested"):
                adapter.promote_memories_receipted(**request)
    else:
        with pytest.raises(ValueError):
            adapter.promote_memories_receipted(**request)
    if guard == "legacy-schema":
        adapter._journal._schema_version = 2
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_http_auth_scope_and_missing_source_do_not_disclose_records(factory, monkeypatch, field):
    monkeypatch.setenv("DML_API_TOKEN", "promotion-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    assert client.post(ENDPOINT, json=request).status_code == 401
    headers = {"Authorization": "Bearer promotion-test-token"}
    before = adapter._journal.verified_snapshot()
    backend = adapter.embedder
    adapter.embedder = UnavailableEmbedder()
    wrong = client.post(ENDPOINT, json={**request, field: "other"}, headers=headers)
    absent_request = deepcopy(request)
    absent_request["sources"][0]["memory_id"] = 999
    absent = client.post(ENDPOINT, json=absent_request, headers=headers)
    assert wrong.status_code == absent.status_code == 404
    assert wrong.json() == absent.json() == {"detail": {"code": "receipt_memory_not_found"}}
    assert adapter._journal.verified_snapshot() == before
    adapter.embedder = backend
    first = client.post(ENDPOINT, json=request, headers=headers)
    assert first.status_code == 200, first.text
    adapter.embedder = UnavailableEmbedder()
    assert client.post(ENDPOINT, json=request, headers=headers).json() == first.json()
    conflict = client.post(ENDPOINT, json={**request, "text": "Changed result"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": {"code": "idempotency_conflict"}}


@pytest.mark.parametrize("field,value", [
    ("sources", None), ("sources", []), ("sources", [True]), ("sources", {}),
    ("text", True), ("text", 123), ("text", None), ("text", ""),
    ("reason", True), ("reason", 123), ("reason", ""),
    ("idempotency_key", True), ("tenant_id", 1), ("client_id", False),
    ("session_id", 1), ("instance_id", 1), ("unexpected", "reject"),
    ("meta", {"source_trust": "verified"}), ("embedding", [0, 1, 0, 0]),
    ("level", 2), ("authority", "higher"),
])
def test_http_strict_top_level_request_rejects_coercion_and_authority_injection(factory, monkeypatch, field, value):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    before = adapter._journal.verified_snapshot()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json=promotion_request(adapter, **{field: value}))
    assert response.status_code == 422, response.text
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("memory_id", "0"), ("memory_id", 0.0), ("memory_id", -1),
    ("expected_memory_digest", 123), ("expected_memory_digest", "not-a-digest"),
    ("expected_memory_digest", "A" * 64), ("meta", {"no_merge": False}),
    ("source_trust", "verified"),
])
def test_http_strict_source_request_rejects_coercion_and_extra_fields(factory, monkeypatch, field, value):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    before = adapter._journal.verified_snapshot()
    request = promotion_request(adapter)
    request["sources"][0][field] = value
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == 422, response.text
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("error,status,code,retry", [
    (ReceiptMemoryNotFound, 404, "receipt_memory_not_found", False),
    (ReceiptLifecycleConflict, 409, "receipt_lifecycle_conflict", False),
    (ReceiptCapacityError, 409, "receipt_capacity_exceeded", False),
    (IdempotencyConflict, 409, "idempotency_conflict", False),
    (RevisionConflict, 409, "revision_conflict", True),
    (TimeoutError, 503, "receipt_ownership_unavailable", True),
    (ReceiptCommitUncertain, 503, "receipt_outcome_unavailable", True),
    (JournalIntegrityError, 503, "receipt_outcome_unavailable", True),
    (ReceiptCommitRejected, 503, "receipt_not_committed", True),
    (ReceiptEmbeddingError, 503, "embedding_unavailable", True),
    (ReceiptEmbeddingCompatibilityError, 503, "embedding_unavailable", True),
    (ValueError, 400, "invalid_or_unsupported_receipt_request", False),
])
def test_http_failures_are_sanitized(factory, monkeypatch, error, status, code, retry):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    request = promotion_request(adapter)

    def fail(**_kwargs):
        raise error("private tenant record and storage credential")

    monkeypatch.setattr(adapter, "promote_memories_receipted", fail)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == status
    assert response.json() == {"detail": {"code": code, **({"retry_same_key": True} if retry else {})}}
    assert "private" not in response.text


def test_http_fully_populated_scope_survives_derivation(factory, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    scope = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}
    adapter, _ = seed(factory, scope=scope)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(
        ENDPOINT, json=promotion_request(adapter, **scope))
    assert response.status_code == 200, response.text
    assert response.json()["scope"] == scope
    assert all(response.json()["result"]["memory"]["meta"][name] == value for name, value in scope.items())
    assert DERIVED in adapter.retrieve_context(DERIVED, **scope)["raw_context"]
    assert adapter.retrieve_context(DERIVED, tenant_id="owner")["items"] == []


@pytest.mark.parametrize("change", [
    {"source_trust": "untrusted"}, {"source_trust": None},
    {"memory_state": "quarantined"}, {"lifecycle_state": "suppressed"},
    {"memory_state": "active", "lifecycle_state": "quarantined"},
    {"expires_at": 1}, {"authority": True}, {"claim_key": "color", "claim_value": "blue"},
    {"no_merge": True},
])
def test_http_merge_authority_conflict_fails_before_backend(factory, monkeypatch, change):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory, multiple=True)
    revision, state = adapter._journal.read_snapshot()
    state["items"][1]["meta"].update(change)
    adapter._journal.save(state, expected_revision=revision, operation="test-conflicting-import")
    request = promotion_request(adapter)
    before = adapter._journal.verified_snapshot()
    adapter.embedder = UnavailableEmbedder()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": {"code": "receipt_lifecycle_conflict"}}
    assert adapter._journal.verified_snapshot() == before


def test_capacity_rejection_never_evicts_acknowledged_memories(factory, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    adapter.store.capacity = 1
    before = adapter._journal.verified_snapshot()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "receipt_capacity_exceeded"}}
    assert adapter._journal.verified_snapshot() == before
    adapter.store.capacity = 2
    receipt = adapter.promote_memories_receipted(**request)
    adapter.store.capacity = 1
    adapter.embedder = UnavailableEmbedder()
    assert adapter.promote_memories_receipted(**request) == receipt


def test_model_preparation_releases_writer_ownership_and_preserves_unrelated_change(factory):
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    other = append(adapter, text="Unrelated memory", key="other", meta=deepcopy(META))["result"]["memory"]
    entered, release = Event(), Event()

    def blocked():
        assert not int(getattr(adapter._mutation_local, "depth", 0))
        entered.set()
        if not release.wait(5):
            raise AssertionError("Test did not release the embedding backend")

    adapter.embedder.callback = blocked

    def change_unrelated():
        with store_write_lock(adapter._journal.path.parent, operation="test-during-promotion-embed", timeout_ms=1000):
            revision, state = adapter._journal.read_snapshot()
            state["items"][1]["meta"]["authority"] = "independent-change"
            adapter._journal.save(state, expected_revision=revision, operation="test-during-promotion-embed")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(adapter.promote_memories_receipted, **request)
        try:
            assert entered.wait(3)
            executor.submit(change_unrelated).result(timeout=3)
        finally:
            release.set()
        receipt = future.result(timeout=5)
    state = adapter._journal.load()
    assert state["items"][-1] == receipt["result"]["memory"]
    assert state["items"][1]["id"] == other["id"]
    assert state["items"][1]["meta"]["authority"] == "independent-change"


def test_all_source_scopes_are_checked_before_any_digest_failure(factory, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory, multiple=True)
    request = promotion_request(adapter)
    request["sources"][0]["expected_memory_digest"] = "0" * 64
    revision, state = adapter._journal.read_snapshot()
    state["items"][1]["meta"]["tenant_id"] = "another-owner"
    adapter._journal.save(state, expected_revision=revision, operation="test-other-tenant")
    adapter.embedder = UnavailableEmbedder()
    before = adapter._journal.verified_snapshot()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": {"code": "receipt_memory_not_found"}}
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("case,status", [("too-many-sources", 422), ("duplicate-source", 400),
    ("blank-text", 400), ("oversized-utf8-text", 400)], ids=lambda value: str(value))
def test_http_bounded_request_rejects_before_backend(factory, monkeypatch, case, status):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    if case == "too-many-sources":
        request["sources"] = [{"memory_id": i, "expected_memory_digest": "0" * 64} for i in range(33)]
    elif case == "duplicate-source":
        request["sources"] *= 2
    elif case == "blank-text":
        request["text"] = " \n\t"
    else:
        request["text"] = "\U0001f4dd" * (256 * 1024)
    adapter.embedder = UnavailableEmbedder()
    before = adapter._journal.verified_snapshot()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == status, response.text[:200]
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("failure", ["backend", "identity-before", "identity-during"])
def test_http_real_backend_failures_preserve_sources_and_allow_retry(factory, monkeypatch, failure):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter, _ = seed(factory)
    request = promotion_request(adapter)
    before = adapter._journal.verified_snapshot()
    count = adapter.embedder.calls
    if failure == "backend":
        adapter.embedder.fail = True
    elif failure == "identity-before":
        adapter.embedder.receipt_embedding_identity = "different-space"
    else:
        adapter.embedder.callback = lambda: setattr(adapter.embedder, "receipt_embedding_identity", "different-space")
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == 503, response.text
    assert response.json() == {"detail": {"code": "embedding_unavailable", "retry_same_key": True}}
    assert "private" not in response.text
    assert adapter._journal.verified_snapshot() == before
    assert adapter.embedder.calls == count + (failure != "identity-before")
    adapter.embedder.fail = False
    adapter.embedder.callback = None
    adapter.embedder.receipt_embedding_identity = TextEmbedder.receipt_embedding_identity
    receipt = adapter.promote_memories_receipted(**request)
    assert receipt["result"]["memory"]["text"] == DERIVED
