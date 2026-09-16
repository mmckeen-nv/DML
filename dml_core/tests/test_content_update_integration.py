"""Public content edits: scoped CAS, compatible vectors and historical retries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
    ReceiptCommitRejected, ReceiptCommitUncertain, ReceiptEmbeddingCompatibilityError,
    ReceiptEmbeddingError,
)
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, memory_digest,
)
from daystrom_dml.store_lock import store_write_lock
from test_receipt_adapter import CountingEmbedder, append, factory as receipt_factory

factory = receipt_factory
ORIGINAL = "My previous preference is blue"
UPDATED = "My corrected preference is green"
META = {"provenance": {"source": "owner-note", "tags": ["original"]},
        "source_trust": "trusted", "authority": 3, "typed": True}
ENDPOINT = "/api/memory/update/receipt"


class TextEmbedder(CountingEmbedder):
    receipt_embedding_identity = "content-update-test-v1"

    def embed(self, text):
        super().embed(text)
        return np.array([1, 0, 0, 0] if text == ORIGINAL else [0, 1, 0, 0],
                        dtype=np.float32)


class UnavailableEmbedder:
    @property
    def embed(self):
        raise AssertionError("Historical retry touched the embedding backend")

    @property
    def receipt_embedding_identity(self):
        raise AssertionError("Historical retry touched the embedding identity")


@pytest.fixture(params=[2, 3, 4])
def authority(request, factory, tmp_path):
    version = request.param
    adapter = factory(directory=tmp_path / "source", embedder=TextEmbedder(),
                      persistence={"journal": True, "receipts": True, "outbox": version == 3})
    ingestion = append(adapter, text=ORIGINAL, meta=deepcopy(META))
    if version == 4:
        adapter.close(persist=False)
        target = tmp_path / "migrated" / "dml_state.sqlite3"
        upgrade_outbox_journal(adapter._journal.path, target)
        adapter = factory(directory=target.parent, embedder=TextEmbedder(),
                          persistence={"journal": True, "receipts": True, "outbox": True})
    assert adapter._journal.schema_version == version
    return adapter, ingestion


def update_request(adapter, **overrides):
    source = adapter._journal.load()["items"][0]
    return {"memory_id": source["id"], "text": UPDATED,
            "expected_memory_digest": memory_digest(source),
            "reason": "Owner corrected the text", "idempotency_key": "update-1",
            "tenant_id": "owner", **overrides}


def seed(factory):
    adapter = factory(embedder=TextEmbedder())
    append(adapter, text=ORIGINAL, meta=deepcopy(META))
    return adapter


def test_content_update_changes_only_text_vector_and_decision(authority):
    adapter, ingestion = authority
    before = deepcopy(adapter._journal.load())
    request = update_request(adapter)
    count = adapter.embedder.calls
    receipt = adapter.update_memory_receipted(**request)
    after = adapter._journal.load()
    expected = deepcopy(before)
    record = expected["items"][0]
    record["text"] = UPDATED
    record["embedding"] = [0.0, 1.0, 0.0, 0.0]
    record["meta"]["content_update_decision"] = {
        "schema_version": "dml-content-update-decision-v1",
        "prior_memory_digest": request["expected_memory_digest"], "reason": request["reason"]}
    assert after == expected
    assert receipt["result"]["memory"] == record
    assert adapter.embedder.calls == count + 1
    assert adapter.update_memory_receipted(**request) == receipt
    assert append(adapter, text=ORIGINAL, meta=deepcopy(META)) == ingestion
    assert adapter.embedder.calls == count + 1
    next_request = update_request(adapter, text="A later corrected preference", idempotency_key="update-2")
    next_receipt = adapter.update_memory_receipted(**next_request)
    assert next_receipt["revision"] == receipt["revision"] + 1
    assert next_receipt["result"]["memory"]["meta"]["content_update_decision"]["prior_memory_digest"] == memory_digest(record)
    assert adapter.update_memory_receipted(**request) == receipt
    assert adapter._journal.load()["items"][0] == next_receipt["result"]["memory"]


def test_update_retry_after_restart_uses_history_without_embedding(authority, factory):
    adapter, _ = authority
    request = update_request(adapter)
    receipt = adapter.update_memory_receipted(**request)
    before = adapter._journal.verified_snapshot()
    directory = adapter._journal.path.parent
    version = adapter._journal.schema_version
    adapter.close(persist=False)
    failing = TextEmbedder(fail=True)
    restarted = factory(directory=directory, embedder=failing,
                        persistence={"journal": True, "receipts": True, "outbox": version in (3, 4)})
    assert restarted.update_memory_receipted(**request) == receipt
    assert failing.calls == 0
    restarted.embedder = UnavailableEmbedder()
    assert restarted.update_memory_receipted(**request) == receipt
    assert restarted._journal.verified_snapshot() == before


@pytest.mark.parametrize("terminal", ["retirement", "supersession"])
def test_historical_update_receipt_survives_later_lifecycle_decision(authority, terminal):
    adapter, _ = authority
    request = update_request(adapter)
    receipt = adapter.update_memory_receipted(**request)
    current = adapter._journal.load()["items"][0]
    lifecycle = {"memory_id": current["id"], "expected_memory_digest": memory_digest(current),
                 "reason": "Owner replaced this information", "idempotency_key": "later-lifecycle",
                 "tenant_id": "owner"}
    if terminal == "retirement":
        adapter.retire_memory_receipted(**lifecycle)
    else:
        replacement = append(adapter, text="An independent replacement", key="replacement")["result"]["memory"]
        adapter.supersede_memory_receipted(**lifecycle, replacement_memory_id=replacement["id"],
                                           expected_replacement_digest=memory_digest(replacement))
    before = adapter._journal.verified_snapshot()
    adapter.embedder = UnavailableEmbedder()
    assert adapter.update_memory_receipted(**request) == receipt
    with pytest.raises(ReceiptLifecycleConflict):
        adapter.update_memory_receipted(**update_request(adapter, idempotency_key="new-update", text="Must not revive"))
    assert adapter._journal.verified_snapshot() == before


def test_same_text_and_stale_decision_reject_before_backend_properties(authority):
    adapter, _ = authority
    request = update_request(adapter)
    before = adapter._journal.verified_snapshot()
    adapter.embedder = UnavailableEmbedder()
    with pytest.raises(ReceiptLifecycleConflict):
        adapter.update_memory_receipted(**{**request, "text": ORIGINAL})
    assert adapter._journal.verified_snapshot() == before
    revision, state = adapter._journal.read_snapshot()
    state["items"][0]["meta"]["typed"] = 1
    adapter._journal.save(state, expected_revision=revision, operation="test-concurrent-metadata")
    changed = adapter._journal.verified_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        adapter.update_memory_receipted(**request)
    assert adapter._journal.verified_snapshot() == changed


def test_failed_hydration_preserves_success_and_never_serves_old_text(authority, monkeypatch):
    adapter, _ = authority
    assert ORIGINAL in adapter.retrieve_context(ORIGINAL, tenant_id="owner")["raw_context"]
    request = update_request(adapter)
    original_import = adapter.store.import_state
    adapter.query_cache.get("old-query", lambda _: np.ones(4))

    def fail(_payload):
        raise RuntimeError("private tenant memory")

    monkeypatch.setattr(adapter.store, "import_state", fail)
    receipt = adapter.update_memory_receipted(**request)
    assert adapter._last_observed_state is None
    assert adapter.durability_status() == {"status": "degraded", "failures": {"receipt_runtime": "RuntimeError"}}
    assert adapter.update_memory_receipted(**request) == receipt
    with pytest.raises(RuntimeError, match="private tenant memory"):
        adapter.retrieve_context(UPDATED, tenant_id="owner")
    monkeypatch.setattr(adapter.store, "import_state", original_import)
    report = adapter.retrieve_context(UPDATED, tenant_id="owner")
    assert UPDATED in report["raw_context"]
    assert ORIGINAL not in report["raw_context"]
    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert not adapter.query_cache.values
    assert adapter.store.items()[0].embedding.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_snapshot_delta_and_outbox_deliver_new_text_and_vector(authority, tmp_path):
    adapter, _ = authority
    projection = SQLiteProjection(tmp_path / "projection" / "state.sqlite")
    reconcile(adapter._journal, projection)
    assert projection.read()["items"][0]["embedding"] == [1.0, 0.0, 0.0, 0.0]
    request = update_request(adapter)
    receipt = adapter.update_memory_receipted(**request)
    assert adapter.sync_projection(projection)["matches_pinned_source"] is True
    projected = projection.read()["items"]
    assert projected == [receipt["result"]["memory"]]
    results = adapter.query_projection(UPDATED, backend=projection, tenant_id="owner", as_of=100)["results"]
    assert [item["memory"] for item in results] == projected
    report = adapter.retrieve_context(UPDATED, tenant_id="owner")
    assert UPDATED in report["raw_context"]
    assert ORIGINAL not in report["raw_context"]
    rebuilt = SQLiteProjection(tmp_path / "rebuilt" / "state.sqlite")
    reconcile(adapter._journal, rebuilt)
    assert rebuilt.read()["items"] == projected
    if adapter._journal.schema_version in (3, 4):
        consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "state.sqlite")
        assert adapter.deliver_outbox(consumer)["backlog"] == 0
        event = consumer.read()["last_event"]
        assert event["state"]["items"] == projected
        assert event["receipt"]["key"] == request["idempotency_key"]
        assert event["operation"] == "update-receipt-v1"
    decision = adapter._journal.decisions()[-1]
    assert decision["deleted"] == []
    assert [(change["bucket"], change["id"]) for change in decision["changed"]] == [
        ("items", str(request["memory_id"]))]


@pytest.mark.parametrize("guard", ["disabled", "mirror", "nested", "legacy-schema"])
def test_update_preserves_explicit_mutation_guards(factory, guard):
    adapter = seed(factory)
    request = update_request(adapter)
    before = adapter._journal.verified_snapshot()
    if guard == "disabled":
        adapter._receipts_enabled = False
    elif guard == "mirror":
        adapter.mirror_agentic_memory_to_rag = True
    elif guard == "legacy-schema":
        adapter._journal._schema_version = 1
    adapter.embedder = UnavailableEmbedder()
    if guard == "nested":
        with adapter.mutation_transaction("update-cannot-nest"):
            with pytest.raises(ValueError, match="nested"):
                adapter.update_memory_receipted(**request)
    else:
        with pytest.raises(ValueError):
            adapter.update_memory_receipted(**request)
    if guard == "legacy-schema":
        adapter._journal._schema_version = 2
    assert adapter._journal.verified_snapshot() == before


@pytest.mark.parametrize("field", ["tenant_id", "client_id", "session_id", "instance_id"])
def test_http_exact_scope_and_absent_target_are_indistinguishable(factory, monkeypatch, field):
    monkeypatch.setenv("DML_API_TOKEN", "content-update-test-token")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    request = update_request(adapter)
    before = adapter._journal.verified_snapshot()
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    assert client.post(ENDPOINT, json=request).status_code == 401
    headers = {"Authorization": "Bearer content-update-test-token"}
    backend = adapter.embedder
    adapter.embedder = UnavailableEmbedder()
    wrong = client.post(ENDPOINT, json={**request, field: "other"}, headers=headers)
    absent = client.post(ENDPOINT, json={**request, "memory_id": 999}, headers=headers)
    assert wrong.status_code == absent.status_code == 404
    assert wrong.json() == absent.json() == {"detail": {"code": "receipt_memory_not_found"}}
    assert adapter._journal.verified_snapshot() == before
    adapter.embedder = backend
    first = client.post(ENDPOINT, json=request, headers=headers)
    assert first.status_code == 200, first.text
    adapter.embedder = UnavailableEmbedder()
    assert client.post(ENDPOINT, json=request, headers=headers).json() == first.json()
    conflict = client.post(ENDPOINT, json={**request, "text": "Different content"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"


@pytest.mark.parametrize("field,value", [
    ("memory_id", True), ("memory_id", "0"), ("memory_id", 0.0), ("memory_id", -1),
    ("expected_memory_digest", 123), ("expected_memory_digest", "not-a-digest"),
    ("text", True), ("text", 123), ("text", ["text"]), ("text", None), ("text", ""),
    ("reason", True), ("reason", 123), ("reason", ""),
    ("idempotency_key", True), ("tenant_id", 1), ("client_id", False),
    ("session_id", 1), ("instance_id", 1), ("unexpected", "must reject"),
    ("meta", {"source_trust": "trusted"}), ("embedding", [0, 1, 0, 0]),
])
def test_http_update_strictly_rejects_coercion_and_unknown_fields(factory, monkeypatch, field, value):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    before = adapter._journal.verified_snapshot()
    request = update_request(adapter, **{field: value})
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
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
    (ReceiptEmbeddingError, 503, "embedding_unavailable", True),
    (ReceiptEmbeddingCompatibilityError, 503, "embedding_unavailable", True),
    (ValueError, 400, "invalid_or_unsupported_receipt_request", False),
])
def test_http_update_errors_are_sanitized(factory, monkeypatch, error, status, code, retry):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    request = update_request(adapter)

    def fail(**_kwargs):
        raise error("private tenant record and storage credential")

    monkeypatch.setattr(adapter, "update_memory_receipted", fail)
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=request)
    assert response.status_code == status
    expected = {"code": code, **({"retry_same_key": True} if retry else {})}
    assert response.json() == {"detail": expected}
    assert "private" not in response.text


def test_http_update_preserves_fully_populated_scope(factory, monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = factory(embedder=TextEmbedder())
    scope = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "instance"}
    append(adapter, text=ORIGINAL, meta=deepcopy(META), **scope)
    request = update_request(adapter, **scope)
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    response = client.post(ENDPOINT, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["scope"] == scope
    assert {name: response.json()["result"]["memory"]["meta"][name] for name in scope} == scope
    assert UPDATED in adapter.retrieve_context(UPDATED, **scope)["raw_context"]
    assert adapter.retrieve_context(UPDATED, tenant_id="owner")["items"] == []


@pytest.mark.parametrize("metadata", [{"source_trust": "untrusted"}, {"memory_state": "quarantined"}, {"expires_at": 1}])
def test_content_edit_preserves_retrieval_suppression(factory, metadata):
    adapter = factory(embedder=TextEmbedder())
    append(adapter, text=ORIGINAL, meta={**deepcopy(META), **metadata})
    receipt = adapter.update_memory_receipted(**update_request(adapter))
    assert all(receipt["result"]["memory"]["meta"][name] == value for name, value in metadata.items())
    report = adapter.retrieve_context(UPDATED, tenant_id="owner", as_of=100)
    assert report["items"] == []
    assert UPDATED not in report["raw_context"]


@pytest.mark.parametrize("text", [" \n\t", "\U0001f4dd" * (256 * 1024)])
def test_http_invalid_text_content_rejects_before_backend(factory, monkeypatch, text):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    before = adapter._journal.verified_snapshot()
    adapter.embedder = UnavailableEmbedder()
    response = TestClient(create_app(adapter_factory=lambda: adapter)).post(ENDPOINT, json=update_request(adapter, text=text))
    assert response.status_code == 400
    assert response.json() == {"detail": {"code": "invalid_or_unsupported_receipt_request"}}
    assert adapter._journal.verified_snapshot() == before


def test_updated_text_overrides_historical_cached_summary_without_rewriting_provenance(factory):
    adapter = factory(embedder=TextEmbedder())
    ingestion = append(adapter, text=ORIGINAL, meta={**deepcopy(META), "summary": ORIGINAL})
    assert ORIGINAL in adapter.retrieve_context(ORIGINAL, tenant_id="owner")["raw_context"]
    receipt = adapter.update_memory_receipted(**update_request(adapter))
    assert receipt["result"]["memory"]["meta"]["summary"] == ORIGINAL
    report = adapter.retrieve_context(UPDATED, tenant_id="owner")
    assert UPDATED in report["raw_context"]
    assert ORIGINAL not in report["raw_context"]
    item = adapter.store.items()[0]
    assert item.cached_summary() == UPDATED
    assert item.cached_summary(max_len=12) == UPDATED[:9] + "..."
    assert UPDATED in adapter._format_ltm_entries([item])
    assert append(adapter, text=ORIGINAL, meta={**deepcopy(META), "summary": ORIGINAL}) == ingestion
    assert ingestion["result"]["memory"]["text"] == ORIGINAL
    assert ingestion["result"]["memory"]["meta"]["summary"] == ORIGINAL


@pytest.mark.parametrize("failure", ["backend", "identity-before", "identity-during"])
def test_actual_embedding_failure_is_retryable_without_mutation(factory, monkeypatch, failure):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    adapter = seed(factory)
    request = update_request(adapter)
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
    receipt = adapter.update_memory_receipted(**request)
    assert receipt["result"]["memory"]["text"] == UPDATED


def test_embedding_preparation_releases_writer_ownership_and_rebases_unrelated_change(factory):
    adapter = seed(factory)
    append(adapter, text="Unrelated memory", key="other")
    request = update_request(adapter)
    entered, release = Event(), Event()

    def blocked():
        assert not int(getattr(adapter._mutation_local, "depth", 0))
        entered.set()
        if not release.wait(5):
            raise AssertionError("Test did not release the embedding backend")

    adapter.embedder.callback = blocked

    def write_unrelated():
        with store_write_lock(adapter._journal.path.parent, operation="test-during-embed", timeout_ms=1000):
            revision, state = adapter._journal.read_snapshot()
            state["items"][1]["meta"]["authority"] = "independent-change"
            adapter._journal.save(state, expected_revision=revision, operation="test-during-embed")

    with ThreadPoolExecutor(max_workers=2) as executor:
        updating = executor.submit(adapter.update_memory_receipted, **request)
        try:
            assert entered.wait(3)
            executor.submit(write_unrelated).result(timeout=3)
        finally:
            release.set()
        receipt = updating.result(timeout=5)
    state = adapter._journal.load()
    assert state["items"][0] == receipt["result"]["memory"]
    assert state["items"][1]["meta"]["authority"] == "independent-change"
    assert state["items"][0]["text"] == UPDATED
