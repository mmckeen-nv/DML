"""Performance contracts: bounded work, identical scope, and durable failure behavior."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from daystrom_dml.dml_adapter import DMLAdapter, PersistenceCommitError
from daystrom_dml.embeddings import embed_texts, OllamaEmbedder
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.persistent_index import PersistentVectorIndex
from daystrom_dml.provider_client import ProviderClient
from daystrom_dml.provider_server import create_app
from daystrom_dml.memory_store import MemoryItem


class BatchEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        return np.array([1, len(text) / 100, 0, 0], dtype=np.float32)

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [self.embed(text) for text in texts]


def make_adapter(path, *, journal=False):
    return DMLAdapter(
        config_overrides={"model_name": "dummy", "embedding_model": None,
                          "storage_dir": str(path), "theta_merge": 2.0,
                          "persistence": {"enable": False, "journal": journal},
                          "skip_rag_state_import": True},
        embedder=BatchEmbedder(), start_aging_loop=False,
    )


@pytest.fixture
def adapter(tmp_path):
    instance = make_adapter(tmp_path)
    yield instance
    instance.close()


def test_index_batch_one_commit_and_failed_write_rolls_back(tmp_path, monkeypatch):
    index = PersistentVectorIndex(tmp_path / "index.json")
    calls = []
    original = index._flush
    monkeypatch.setattr(index, "_flush", lambda: (calls.append(1), original()))
    index.extend([np.ones(4)] * 100, [{"text": str(i)} for i in range(100)])
    assert len(calls) == 1
    assert len(PersistentVectorIndex(index.path)._vectors) == 100
    before = index.path.read_bytes()
    index.search(np.ones(4))
    matrix = index._matrix
    index.search(np.ones(4))
    assert index._matrix is matrix

    def fail():
        raise OSError("disk failed")

    monkeypatch.setattr(index, "_flush", fail)
    with pytest.raises(OSError):
        index.extend([np.ones(4)] * 2, [{"text": "new"}] * 2)
    assert len(index._vectors) == 100
    assert index.path.read_bytes() == before
    assert index._matrix is matrix


def test_index_batch_validates_before_mutating(tmp_path):
    index = PersistentVectorIndex(tmp_path / "index.json")
    with pytest.raises(ValueError, match="lengths"):
        index.extend([np.ones(4)] * 2, [{"text": "one"}])
    assert not index._vectors and not index.path.exists()
    with pytest.raises(ValueError, match="dimension"):
        index.extend([np.ones(4), np.ones(3)], [{}, {}])
    assert not index._vectors


def test_native_embedding_batches_validate_shape_and_count():
    embedder = BatchEmbedder()
    assert len(embed_texts(embedder, ["a"] * 5, batch_size=2)) == 5
    assert list(map(len, embedder.calls)) == [2, 2, 1]
    embedder.embed_many = lambda texts: [np.ones(4)]
    with pytest.raises(ValueError, match="count"):
        embed_texts(embedder, ["a", "b"])
    embedder.embed_many = lambda texts: [np.array([np.nan])] * len(texts)
    with pytest.raises(ValueError, match="invalid"):
        embed_texts(embedder, ["a"])


def test_ollama_batch_uses_one_request(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"embeddings": [[3, 4], [0, 2]]}, request=httpx.Request("POST", url))

    monkeypatch.setattr("daystrom_dml.embeddings.requests.post", post)
    result = embed_texts(OllamaEmbedder("test"), ["a", "", "b"])
    assert len(calls) == 1 and calls[0][0].endswith("/api/embed")
    assert calls[0][1]["json"]["input"] == ["a", "b"]
    assert calls[0][1]["json"]["truncate"] is False
    np.testing.assert_allclose(result[0], [0.6, 0.8])
    np.testing.assert_array_equal(result[1], [0, 0])


def test_memory_batch_one_commit_preserves_scope_and_rolls_back(adapter, monkeypatch):
    records = [{"text": f"note {i}", "tenant_id": "owner", "session_id": "s",
                "meta": {"tenant_id": "intruder"}} for i in range(5)]
    commits = []
    original = adapter._persist_dml_state
    monkeypatch.setattr(adapter, "_persist_dml_state", lambda: (commits.append(1), original()))
    result = adapter.ingest_memory_batch(records, batch_size=2)
    assert len(result) == 5 and len(commits) == 1
    assert list(map(len, adapter.embedder.calls)) == [2, 2, 1]
    assert all(item.meta["tenant_id"] == "owner" for item in result)
    before = adapter.store.export_state()

    def fail():
        raise PersistenceCommitError("injected commit failure")

    monkeypatch.setattr(adapter, "_persist_dml_state", fail)
    with pytest.raises(PersistenceCommitError):
        adapter.ingest_memory_batch(records)
    assert adapter.store.export_state() == before
    monkeypatch.setattr(adapter, "_persist_dml_state", original)


@pytest.mark.parametrize("fails", [False, True])
def test_concurrent_embedding_misses_share_one_call(adapter, fails):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def embed(text):
        calls.append(text)
        entered.set()
        assert release.wait(5)
        if fails:
            raise RuntimeError("provider unavailable")
        return np.ones(4)

    adapter.embedder.embed = embed
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(adapter._embed_query, "same") for _ in range(8)]
        assert entered.wait(5)
        # Wait until the seven followers have entered Future.result().
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pending = adapter._query_embedding_inflight.get("same")
            if pending and len(pending._condition._waiters) == 7:
                break
            time.sleep(0.001)
        else:
            pytest.fail("embedding followers did not attach")
        release.set()
        for future in futures:
            if fails:
                with pytest.raises(RuntimeError, match="unavailable"):
                    future.result()
            else:
                np.testing.assert_array_equal(future.result(), np.ones(4))
    assert calls == ["same"]
    assert not adapter._query_embedding_inflight
    if fails:
        adapter.embedder.embed = lambda text: np.zeros(4)
        np.testing.assert_array_equal(adapter._embed_query("same"), np.zeros(4))


def populate(store, count=300):
    rng = np.random.default_rng(41)
    store._items = [MemoryItem(i, f"record {i}", rng.normal(size=16).astype(np.float32), 1000,
                               0, 1, 0, {"tenant_id": str(i % 2), "kind": "note", "lattice_row": 0,
                                         "lattice_col": 0, "lattice_layer": 0, "lattice_neighbors": [i]})
                    for i in range(count)]
    store._lineage = {item.id: item for item in store._items}
    store._id = count
    store._invalidate_embedding_cache()


def test_scope_matrix_reuse_invalidation_and_exact_equivalence(adapter):
    store = adapter.store
    populate(store)
    query = store.items()[0].embedding
    reference = store.retrieve_filtered(query, tenant_id="0", strict_scope=False, top_k=8)
    actual = store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8)
    assert [item.id for item in actual] == [item.id for item in reference]
    cached = next(iter(store._scope_matrices.values()))[1]
    assert store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8)
    assert next(iter(store._scope_matrices.values()))[1] is cached
    store.ingest("new private memory", query, meta={"tenant_id": "1", "no_merge": True})
    assert not store._scope_matrices
    assert all(item.meta["tenant_id"] == "0" for item in store.retrieve_filtered(query, tenant_id="0", strict_scope=True))


def test_scope_cache_has_byte_bound(adapter):
    populate(adapter.store)
    adapter.store._scope_cache_bytes = 1
    adapter.store.retrieve_filtered(np.ones(16), tenant_id="0", strict_scope=True)
    assert not adapter.store._scope_matrices


def test_approximate_search_indexes_only_requested_scope_and_kind(adapter):
    pytest.importorskip("faiss")
    store = adapter.store
    populate(store, 600)
    query = store.items()[0].embedding
    expected = store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8, kinds=["note"])
    store._ann_min_items = 100
    found = store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8, kinds=["note"])
    assert all(item.meta["tenant_id"] == "0" for item in found)
    assert len({item.id for item in found} & {item.id for item in expected}) / 8 >= 0.875
    index = next(iter(store._ann_indexes.values()))
    assert index.ntotal == 300
    store._invalidate_embedding_cache()
    assert not store._ann_indexes


def payload():
    return {"items": [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}],
            "lineage": [{"id": 1, "text": "one"}], "next_id": 3, "repair_queue": []}


def test_journal_incremental_replay_order_deletion_and_snapshot(tmp_path):
    journal = JournalStateStore(tmp_path / "store.sqlite3", snapshot_interval=2)
    state = payload()
    journal.save(state)
    assert journal.last_changed_rows == 3
    assert JournalStateStore(journal.path).load() == state
    state["items"][1]["text"] = "changed"
    journal.save(state)
    assert journal.last_changed_rows == 1
    with sqlite3.connect(journal.path) as connection:
        assert json.loads(connection.execute("SELECT payload FROM snapshot").fetchone()[0]) == state
    state["items"].reverse()
    state["lineage"].clear()
    journal.save(state)
    assert JournalStateStore(journal.path).load() == state
    out = journal.export_snapshot(tmp_path / "snapshot.json")
    assert json.loads(out.read_text()) == state
    revision = journal.stamp()[0]
    journal.save(state)
    assert journal.stamp()[0] == revision


def test_journal_failed_transaction_is_not_acknowledged_or_replayed(tmp_path, monkeypatch):
    journal = JournalStateStore(tmp_path / "store.sqlite3")
    state = payload()
    journal.save(state)
    original = journal._connect

    @contextmanager
    def fail_before_commit():
        with original() as connection:
            yield connection
            raise OSError("injected commit failure")

    monkeypatch.setattr(journal, "_connect", fail_before_commit)
    changed = copy.deepcopy(state)
    changed["items"][0]["text"] = "must roll back"
    with pytest.raises(OSError):
        journal.save(changed)
    assert JournalStateStore(journal.path).load() == state


def test_journal_adapter_refreshes_other_writer_and_reopens(tmp_path):
    first, second = make_adapter(tmp_path, journal=True), make_adapter(tmp_path, journal=True)
    try:
        first.ingest_memory("first", tenant_id="a")
        second.ingest_memory("second", tenant_id="b")
        assert first.refresh_if_changed()
        assert {item.text for item in first.store.items()} == {"first", "second"}
    finally:
        first.close()
        second.close()
    reopened = make_adapter(tmp_path, journal=True)
    try:
        assert {item.text for item in reopened.store.items()} == {"first", "second"}
    finally:
        reopened.close()


def test_journal_rejects_implicit_legacy_migration(tmp_path):
    (tmp_path / "dml_store.json").write_text(json.dumps(payload()))
    with pytest.raises(ValueError, match="explicit snapshot import"):
        make_adapter(tmp_path, journal=True)


def test_journal_migration_preserves_existing_memories(tmp_path):
    from scripts.dml_journal import import_snapshot

    previous = make_adapter(tmp_path)
    previous.ingest_memory("existing memory", tenant_id="owner")
    previous.close()
    source = tmp_path / "dml_store.json"
    original = source.read_bytes()
    database = tmp_path / "dml_state.sqlite3"
    import_snapshot(source, database)
    with pytest.raises(ValueError, match="overwrite"):
        import_snapshot(source, database)
    reopened = make_adapter(tmp_path, journal=True)
    try:
        assert [item.text for item in reopened.store.items()] == ["existing memory"]
        assert source.read_bytes() == original
    finally:
        reopened.close()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_journal_export_cannot_overwrite_database_files(tmp_path, suffix):
    store = JournalStateStore(tmp_path / "store.sqlite3")
    store.save(payload())
    with pytest.raises(ValueError, match="overwrite"):
        store.export_snapshot(Path(str(store.path) + suffix))
    assert store.load() == payload()


def test_empty_journal_is_initialized_once(tmp_path):
    store = JournalStateStore(tmp_path / "store.sqlite3")
    store.save({"items": [], "lineage": []})
    assert store.stamp()[0] == 1
    store.save({"items": [], "lineage": []})
    assert store.stamp()[0] == 1


def test_ann_revalidates_changed_metadata_before_returning(adapter):
    pytest.importorskip("faiss")
    populate(adapter.store, 600)
    store = adapter.store
    store._ann_min_items = 100
    query = store.items()[0].embedding
    assert store.retrieve_filtered(query, tenant_id="0", strict_scope=True)
    for item in store._items:
        item.meta["tenant_id"] = "private"
    assert not store.retrieve_filtered(query, tenant_id="0", strict_scope=True)


def test_provider_batch_endpoint_and_pooled_client(adapter):
    with TestClient(create_app(adapter_factory=lambda: adapter)) as client:
        result = client.post("/api/remember/batch", json={"records": [{"text": "memory", "tenant_id": "owner"}]})
        assert result.status_code == 200 and result.json()["count"] == 1
        assert client.post("/api/remember/batch", json={"records": []}).status_code == 422
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"status": "ok"})
    with ProviderClient(token="secret", transport=httpx.MockTransport(handler)) as client:
        connection = client._client
        client.remember_many([{"text": "memory", "tenant_id": "owner"}])
        client.recall("memory", tenant_id="owner")
        assert client._client is connection
    assert [request.url.path for request in calls] == ["/api/remember/batch", "/api/recall"]
    assert all(request.headers["Authorization"] == "Bearer secret" for request in calls)
    assert connection.is_closed
