"""Regression coverage for write and recall scope boundaries."""
import numpy as np
import pytest
import argparse
from fastapi.testclient import TestClient

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.provider_server import create_app
from daystrom_dml.provider_cli import _client


@pytest.fixture
def adapter(tmp_path):
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy", "embedding_model": None,
            "storage_dir": str(tmp_path), "persistence": {"enable": False},
        },
        start_aging_loop=False,
    )
    adapter.embedder.embed = lambda text: np.ones(16, dtype=np.float32)
    yield adapter
    adapter.close()


@pytest.mark.parametrize("key", [
    "tenant_id", "client_id", "session_id", "instance_id", "thread_id",
    "project_id", "relationship_id", "kind", "phase",
])
def test_merge_never_crosses_scope_or_kind(adapter, key):
    first, _ = adapter.store.ingest("private alpha", np.ones(16), meta={key: "alpha"})
    second, merged = adapter.store.ingest("private beta", np.ones(16), meta={key: "beta"})
    assert not merged
    assert first.id != second.id
    assert first.text == "private alpha"


def test_merge_selects_eligible_neighbor_and_preserves_no_merge(adapter):
    protected, _ = adapter.store.ingest("protected", np.ones(16), meta={"no_merge": True})
    first, merged = adapter.store.ingest("first", np.ones(16))
    assert not merged
    second, merged = adapter.store.ingest("second", np.ones(16))
    assert merged and second.id == first.id
    assert protected.text == "protected"


def test_explicit_scope_wins_over_metadata(adapter):
    item = adapter.ingest_memory(
        "scoped memory", tenant_id="owner", kind="note",
        meta={"tenant_id": "other", "session_id": "hidden", "kind": "workflow", "source": "test"},
    )
    assert item.meta["tenant_id"] == "owner"
    assert item.meta["session_id"] is None
    assert item.meta["kind"] == "note"
    assert item.meta["source"] == "test"


@pytest.mark.parametrize("key", ["client_id", "session_id", "instance_id"])
@pytest.mark.parametrize("semantic_match", [True, False])
def test_recall_omitted_scope_never_reads_private_subscope(adapter, monkeypatch, key, semantic_match):
    adapter.ingest("subscope private memory", meta={"tenant_id": "owner", key: "private"})
    if not semantic_match:
        monkeypatch.setattr(adapter.store, "retrieve_filtered", lambda *a, **kw: [])
    report = adapter.retrieve_context("memory", tenant_id="owner")
    assert not report["items"]
    assert "subscope private memory" not in report["raw_context"]


def test_provider_auth_and_reserved_metadata(adapter, monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "secret")
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    with TestClient(create_app(adapter_factory=lambda: adapter)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/api/recall", json={"query": "memory"}).status_code == 401
        response = client.post("/api/remember", headers={"Authorization": "Bearer secret"}, json={
            "text": "scoped memory", "tenant_id": "owner", "kind": "note",
            "meta": {"tenant_id": "other", "session_id": "hidden", "kind": "workflow"},
        })
        assert response.status_code == 200
        item = adapter.store.items()[0]
        assert item.meta["tenant_id"] == "owner"
        assert item.meta["session_id"] is None
        assert item.meta["kind"] == "note"


@pytest.mark.parametrize("variable", ["DML_API_TOKEN", "DML_ADMIN_TOKEN"])
def test_provider_cli_sends_configured_credential(monkeypatch, variable):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv(variable, "secret")
    with _client(argparse.Namespace(base_url="http://localhost:8765", timeout_s=5)) as client:
        assert client.headers["Authorization"] == "Bearer secret"


def test_query_cache_preserves_exact_input_and_reuses_identical_queries(adapter):
    calls = []

    def embed(text):
        calls.append(text)
        return np.array([len(calls)], dtype=np.float32)

    adapter.embedder.embed = embed
    for text in ["US", "us", " us", "us "]:
        first = adapter._embed_query(text)
        assert np.array_equal(adapter._embed_query(text), first)
    assert calls == ["US", "us", " us", "us "]


def test_query_cache_remains_bounded_under_concurrent_reads(adapter):
    from concurrent.futures import ThreadPoolExecutor

    adapter._query_embedding_cache_size = 2
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(adapter._embed_query, ["a", "b", "c"] * 100))
    assert all(np.all(result == 1) for result in results)
    assert len(adapter._query_embedding_cache) <= 2
