"""Embedding-space identity and dimension are independent receipt release gates."""
from __future__ import annotations

import numpy as np
import pytest

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.journal import IdempotencyConflict, JournalStateStore, upgrade_receipt_journal
from daystrom_dml.services.receipt_ingestion import (
    ReceiptEmbeddingCompatibilityError, append_receipted, canonical_request,
)


class VersionedEmbedder:
    def __init__(self, revision="revision-a", dimension=4):
        self.receipt_embedding_identity = revision
        self.dimension = dimension
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return np.ones(self.dimension, dtype=np.float32)


@pytest.fixture
def factory(tmp_path):
    adapters = []
    def create(embedder, **overrides):
        config = {"storage_dir": str(tmp_path), "model_name": "dummy", "embedding_model": None,
            "persistence": {"journal": True, "receipts": True, "enable": False},
            "rag_store": {"enable": False}, "checkpoint_interval_seconds": 0,
            "dpm": {"enable": False, "include_in_context": False}, **overrides}
        adapter = DMLAdapter(config_overrides=config, embedder=embedder, start_aging_loop=False)
        adapters.append(adapter)
        return adapter
    yield create
    for adapter in adapters:
        adapter.close(persist=False)


def append(adapter, key="one"):
    return adapter.ingest_memory_receipted("original memory", idempotency_key=key, tenant_id="tenant")


def test_same_dimension_changed_model_rejects_new_work_but_replays_before_embed(factory):
    first = factory(VersionedEmbedder())
    receipt = append(first)
    changed = VersionedEmbedder(revision="revision-b")
    second = factory(changed)
    assert changed.calls == 0, "startup must not probe unavailable or incompatible models"
    assert append(second) == receipt
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="identity differs"):
        append(second, "new")
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="identity differs"):
        second.retrieve_context("query", tenant_id="tenant")
    assert changed.calls == 0
    assert first._journal.read_snapshot()[0] == 1


def test_changed_dimension_same_identity_rejects_before_commit_and_ranking(factory):
    embedder = VersionedEmbedder()
    adapter = factory(embedder)
    append(adapter)
    original = adapter._journal.read_snapshot()
    embedder.dimension = 7
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="dimensions differ"):
        append(adapter, "new")
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="dimensions differ"):
        adapter.retrieve_context("query", tenant_id="tenant")
    assert adapter._journal.read_snapshot() == original


def test_unknown_custom_identity_fails_closed_only_for_new_work(factory):
    adapter = factory(VersionedEmbedder())
    receipt = append(adapter)
    adapter.embedder.receipt_embedding_identity = None
    assert append(adapter) == receipt
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="immutable"):
        append(adapter, "new")
    assert adapter.embedder.calls == 1


def test_config_identity_declaration_and_legacy_write_boundary(factory):
    embedder = VersionedEmbedder(revision=None)
    adapter = factory(embedder, persistence={"journal": True, "receipts": True,
        "receipt_embedding_identity": "pinned-artifact-sha256-example"})
    receipt = append(adapter)
    contract = adapter._journal.load()["embedding_contract"]
    assert contract["identity"]["revision"] == "pinned-artifact-sha256-example"
    assert contract["dimension"] == 4
    with pytest.raises(ValueError, match="only receipted"):
        adapter.ingest_memory("legacy", tenant_id="tenant")
    with pytest.raises(ValueError, match="only receipted"):
        with adapter.atomic_batch("legacy-batch"):
            pytest.fail("legacy batch body must not run")
    adapter.close()
    assert append(adapter) == receipt
    assert adapter._journal.read_snapshot()[0] == 1
    assert embedder.calls == 1


def test_migrated_unidentified_vectors_require_explicit_conversion(tmp_path, factory):
    legacy = JournalStateStore(tmp_path / "legacy.sqlite3")
    legacy.save({"items": [{"id": 0, "schema_version": 1, "text": "old", "level": 0,
        "embedding": [1., 1., 1., 1.], "timestamp": 1., "salience": 1., "fidelity": 1., "meta": {}}]})
    upgrade_receipt_journal(legacy.path, tmp_path / "dml_state.sqlite3")
    embedder = VersionedEmbedder()
    adapter = factory(embedder)
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="lack an embedding identity"):
        append(adapter)
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="lack an embedding identity"):
        adapter.retrieve_context("query", tenant_id="tenant")
    assert embedder.calls == 0
    assert adapter._journal.read_snapshot()[0] == 1


def test_observer_failure_does_not_revoke_committed_receipt(tmp_path):
    journal = JournalStateStore(tmp_path / "journal.db", receipt_mode=True)
    request, digest = canonical_request("original memory", tenant_id="tenant")
    def fail(*args):
        raise RuntimeError("observer failure")
    receipt = append_receipted(journal, request=request, request_digest=digest, key="key",
        embed=lambda _: np.ones(4), embedding_space=lambda: {"test": "v1"}, capacity=10,
        hydrate=fail, degraded=fail)
    assert receipt == journal.lookup_receipt(request["scope"], "key", digest)


def test_reconciliation_preserves_typed_idempotency_conflict(tmp_path, monkeypatch):
    journal = JournalStateStore(tmp_path / "journal.db", receipt_mode=True)
    request, digest = canonical_request("original memory", tenant_id="tenant")
    def conflicting_lookup(**kwargs):
        raise IdempotencyConflict("competing request")
    def uncertain_save(*args, **kwargs):
        monkeypatch.setattr(journal, "lookup_receipt", conflicting_lookup)
        raise OSError("lost commit response")
    monkeypatch.setattr(journal, "save_with_receipt", uncertain_save)
    with pytest.raises(IdempotencyConflict, match="competing"):
        append_receipted(journal, request=request, request_digest=digest, key="key",
            embed=lambda _: np.ones(4), embedding_space=lambda: {"test": "v1"}, capacity=10,
            hydrate=lambda *_: None, degraded=lambda _: None)


def test_failed_hydration_cannot_reuse_query_from_another_embedding_identity(factory, monkeypatch):
    embedder = VersionedEmbedder(revision="A", dimension=2)
    monkeypatch.setattr(embedder, "embed", lambda _: np.array([1., 0.])
        if embedder.receipt_embedding_identity == "A" else np.array([0., 1.]))
    adapter = factory(embedder)
    assert adapter._embed_query("q").tolist() == [1., 0.]
    # Reproduce both an earlier lookup and an unrelated legacy cache entry.
    adapter.query_cache.get("q", lambda _: np.array([1., 0.]))
    embedder.receipt_embedding_identity = "B"
    restore = adapter.store.import_state
    def failed_hydration(_):
        raise RuntimeError("projection unavailable")
    monkeypatch.setattr(adapter.store, "import_state", failed_hydration)
    receipt = append(adapter)
    assert receipt["result"]["memory"]["embedding"] == [0., 1.]
    monkeypatch.setattr(adapter.store, "import_state", restore)
    assert adapter._embed_query("q").tolist() == [0., 1.]
    assert adapter._journal.read_snapshot()[0] == 1


def test_inflight_query_cannot_publish_an_old_identity(factory, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered, release = Event(), Event()
    embedder = VersionedEmbedder(revision="A", dimension=2)
    def delayed_embedding(_):
        identity = embedder.receipt_embedding_identity
        if identity == "A":
            entered.set()
            assert release.wait(10)
        return np.array([1., 0.]) if identity == "A" else np.array([0., 1.])
    monkeypatch.setattr(embedder, "embed", delayed_embedding)
    adapter = factory(embedder)
    with ThreadPoolExecutor(max_workers=1) as executor:
        old = executor.submit(adapter._embed_query, "q")
        assert entered.wait(10)
        embedder.receipt_embedding_identity = "B"
        assert adapter._embed_query("q").tolist() == [0., 1.]
        release.set()
        with pytest.raises(ReceiptEmbeddingCompatibilityError, match="identity changed"):
            old.result(timeout=10)
    assert adapter._embed_query("q").tolist() == [0., 1.]
    assert not adapter.query_cache.values and not adapter.query_cache.pending


def test_query_identity_remains_pinned_until_retrieval_owns_the_store(factory, monkeypatch):
    from contextlib import contextmanager
    embedder = VersionedEmbedder(revision="A", dimension=2)
    adapter = factory(embedder)
    owned = adapter._mutation_transaction
    @contextmanager
    def interleaved(operation):
        if operation == "retrieve-context":
            embedder.receipt_embedding_identity = "B"
            append(adapter)
        with owned(operation):
            yield
    monkeypatch.setattr(adapter, "_mutation_transaction", interleaved)
    with pytest.raises(ReceiptEmbeddingCompatibilityError, match="before retrieval ownership"):
        adapter.retrieve_context("q", tenant_id="tenant")
    assert adapter._journal.read_snapshot()[0] == 1


@pytest.mark.parametrize("method, arguments, keyword_arguments", [
    ("query_database", ("query",), {}),
    ("_retrieve_items", ("query",), {}),
    ("_retrieve_ltm_items", ("query", 3), {}),
    ("suggest_workflows_for_task", ("query",), {}),
    ("run_generation", ("query",), {}),
    ("generate_with_controller", ("query",), {}),
    ("_run_generation_with_controller", ("query",), {"max_new_tokens": 8, "session_id": "session"}),
    ("build_preamble", ("query",), {}),
    ("retrieval_report", ("query",), {}),
])
def test_receipt_profile_refuses_unqualified_legacy_retrieval_before_embedding(factory, method, arguments, keyword_arguments):
    embedder = VersionedEmbedder()
    adapter = factory(embedder)
    with pytest.raises(ValueError, match="only tenant-scoped retrieve_context"):
        getattr(adapter, method)(*arguments, **keyword_arguments)
    assert embedder.calls == 0


@pytest.mark.parametrize("tenant", [None, "", " ", 7])
def test_receipt_retrieval_requires_explicit_tenant_before_embedding(factory, tenant):
    embedder = VersionedEmbedder()
    adapter = factory(embedder)
    with pytest.raises(ValueError, match="explicit nonempty tenant_id"):
        adapter.retrieve_context("query", tenant_id=tenant)
    assert embedder.calls == 0
