import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "daystrom_dml" / "rag_store.py"
SPEC = importlib.util.spec_from_file_location("daystrom_dml.rag_store", MODULE_PATH)
assert SPEC and SPEC.loader  # pragma: no cover - sanity check for test setup
rag_module = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("daystrom_dml.rag_store", rag_module)
SPEC.loader.exec_module(rag_module)
PersistentRAGStore = rag_module.PersistentRAGStore


@pytest.fixture(scope="module", autouse=True)
def _require_faiss():
    pytest.importorskip("faiss")


def _vec(values):
    return np.asarray(values, dtype=np.float32)


def test_roundtrip_add_and_search(tmp_path):
    index_path = tmp_path / "index.faiss"
    meta_path = tmp_path / "meta.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    alpha_id = store.add("def alpha():\n    return 1", _vec([1.0, 0.0, 0.0]), {"source": "alpha.py"})
    beta_id = store.add("def beta():\n    return 2", _vec([0.0, 1.0, 0.0]), {"source": "beta.py"})
    assert alpha_id != beta_id
    results = store.search(_vec([1.0, 0.0, 0.0]), top_k=1)
    assert results
    top = results[0]
    assert top["id"] == alpha_id
    assert top["meta"]["source"] == "alpha.py"
    assert "return 1" in top["text"]


def test_persist_and_reload(tmp_path):
    index_path = tmp_path / "store.faiss"
    meta_path = tmp_path / "store.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    store.add("class Gamma:\n    pass", _vec([0.0, 0.0, 1.0]), {"source": "gamma.py"})
    store.add("class Delta:\n    pass", _vec([0.7, 0.1, 0.2]), {"source": "delta.py"})
    store.persist()

    restored = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    restored.load()
    query = _vec([0.65, 0.15, 0.2])
    results = restored.search(query, top_k=1)
    assert results
    assert results[0]["meta"]["source"] == "delta.py"
    assert "Delta" in results[0]["text"]


def test_load_rejects_torn_index_metadata_pair(tmp_path):
    index_path = tmp_path / "torn.faiss"
    meta_path = tmp_path / "torn.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    store.add("one document", _vec([1.0, 0.0, 0.0]), {"source": "one"})
    store.persist()

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    current_meta = store.manifest_path.parent / manifest["current"]["metadata"]
    payload = json.loads(current_meta.read_text(encoding="utf-8"))
    payload["records"] = []
    current_meta.write_text(json.dumps(payload), encoding="utf-8")

    restored = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    assert restored.load() is False
    assert restored.search(_vec([1.0, 0.0, 0.0]), top_k=1) == []


def test_load_recovers_previous_complete_generation(tmp_path):
    index_path = tmp_path / "recover.faiss"
    meta_path = tmp_path / "recover.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    store.add("stable document", _vec([1.0, 0.0, 0.0]), {"source": "stable"})
    store.persist()
    store.add("new document", _vec([0.0, 1.0, 0.0]), {"source": "new"})
    store.persist()

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    current_meta = store.manifest_path.parent / manifest["current"]["metadata"]
    current_meta.write_text("corrupt", encoding="utf-8")

    restored = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    assert restored.load() is True
    results = restored.search(_vec([1.0, 0.0, 0.0]), top_k=3)
    assert [item["meta"]["source"] for item in results] == ["stable"]


def test_persist_prunes_generations_older_than_manifest_rollback(tmp_path):
    index_path = tmp_path / "bounded.faiss"
    meta_path = tmp_path / "bounded.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)

    for position in range(4):
        store.add(f"document {position}", _vec([1.0, float(position), 0.0]))
        store.persist()

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    retained = {
        manifest[slot][field]
        for slot in ("current", "previous")
        for field in ("index", "metadata")
    }
    generations = {
        path.name
        for pattern in ("bounded.faiss.*", "bounded.json.*")
        for path in tmp_path.glob(pattern)
        if path.name != store.manifest_path.name
    }

    assert generations == retained


def test_runtime_snapshot_restore_is_atomic_and_complete(tmp_path):
    index_path = tmp_path / "snapshot.faiss"
    meta_path = tmp_path / "snapshot.json"
    store = PersistentRAGStore(enable=True, index_path=index_path, meta_path=meta_path, dim=3)
    store.add("stable document", _vec([1.0, 0.0, 0.0]), {"source": "stable"})
    snapshot = store.snapshot_state()

    store.add("transient document", _vec([0.0, 1.0, 0.0]), {"source": "transient"})
    store.restore_state(snapshot)

    results = store.search(_vec([1.0, 0.0, 0.0]), top_k=3)
    assert [item["meta"]["source"] for item in results] == ["stable"]
    assert store._index is not None
    assert store._index.ntotal == 1

    store.restore_state(snapshot)
    assert store._index is not None
    assert store._index.ntotal == 1
