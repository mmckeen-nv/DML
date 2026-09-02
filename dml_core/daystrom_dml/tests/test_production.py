import json
from pathlib import Path

import numpy as np
import pytest

import daystrom_dml.checkpoint as checkpoint_module
import daystrom_dml.persistent_index as persistent_index_module
from daystrom_dml import server
from daystrom_dml.checkpoint import CheckpointManager
from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.persistent_index import PersistentVectorIndex
from daystrom_dml.config import load_config
from scripts.benchmark import run_benchmark

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def build_adapter(tmp_path: Path) -> DMLAdapter:
    storage = tmp_path / "storage"
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(storage),
            "checkpoint_interval_seconds": 0,
        },
        start_aging_loop=False,
    )


def test_checkpoint_creation(tmp_path):
    adapter = build_adapter(tmp_path)
    adapter.ingest("Quantum data bus alignment logs")
    checkpoint_path = adapter.create_checkpoint()
    adapter.close()
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "dml" in payload and "rag" in payload
    assert payload["stats"]["count"] >= 1


def test_persistent_vector_index_roundtrip(tmp_path):
    index_path = tmp_path / "index.json"
    index = PersistentVectorIndex(index_path)
    vec = np.ones(8, dtype=np.float32)
    index.add(vec, {"text": "alpha", "tokens": 4, "meta": {"source": "doc"}})
    results = index.search(vec, top_k=1)
    assert results and results[0]["meta"]["source"] == "doc"
    reloaded = PersistentVectorIndex(index_path)
    results_reload = reloaded.search(vec, top_k=1)
    assert results_reload and results_reload[0]["text"] == "alpha"


def test_router_auto_mode_prefers_literal(tmp_path):
    adapter = build_adapter(tmp_path)
    adapter.ingest(
        "function fetchUserProfile() calls the /api/users endpoint for account lookup.",
        meta={"doc_path": "app/api.py"},
    )
    report = adapter.query_database("show fetchUserProfile implementation", mode="auto")
    adapter.close()
    assert report["mode"] in {"literal", "hybrid"}
    assert "Source: app/api.py" in report["context"]


def test_metrics_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(server, "VISUALIZER_URL", "http://dummy.local")
    with TestClient(server.app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200


def test_visualizer_state_endpoint(monkeypatch):
    monkeypatch.setattr(server, "VISUALIZER_URL", "http://dummy.local")
    monkeypatch.setattr(
        server.visualizer_bridge,
        "latest_prompt",
        lambda: {"prompt": "hello", "timestamp": 1.0},
    )
    with TestClient(server.app) as client:
        response = client.get("/visualizer/state")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["payload"]["prompt"] == "hello"


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DML_CHECKPOINT_INTERVAL_SECONDS", "15")
    settings = load_config(overrides={"storage_dir": str(tmp_path)})
    assert settings.checkpoint_interval_seconds == 15


def test_benchmark_runner(tmp_path):
    adapter = build_adapter(tmp_path)
    metrics = run_benchmark(adapter, iterations=3)
    adapter.close()
    assert metrics["iterations"] == 3.0
    assert metrics["avg_latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Crash-durability: checkpoint and persistent-index writers must route through
# the shared atomic_io helpers (unique sibling temps, fsync, dir-sync, bounded
# Windows retries) rather than bare write_text + Path.replace.
# ---------------------------------------------------------------------------


def test_checkpoint_routes_through_atomic_write_text(tmp_path, monkeypatch):
    calls: list[str] = []
    real = checkpoint_module.atomic_write_text

    def spy(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "atomic_write_text", spy)

    manager = CheckpointManager(tmp_path, provider=lambda: {"gen": 1})
    path = manager.checkpoint()

    assert calls, "checkpoint() did not call atomic_write_text"
    assert calls[-1] == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"gen": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_failure_preserves_prior_state(tmp_path, monkeypatch):
    manager = CheckpointManager(tmp_path, provider=lambda: {"gen": 1})
    first_path = manager.checkpoint()
    assert first_path.exists()

    def boom(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(checkpoint_module, "atomic_write_text", boom)

    with pytest.raises(OSError, match="simulated disk full"):
        manager.checkpoint()

    assert first_path.exists()
    assert json.loads(first_path.read_text(encoding="utf-8")) == {"gen": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_persistent_index_flush_routes_through_atomic_write_text(tmp_path, monkeypatch):
    calls: list[str] = []
    real = persistent_index_module.atomic_write_text

    def spy(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(persistent_index_module, "atomic_write_text", spy)

    index_path = tmp_path / "index.json"
    index = PersistentVectorIndex(index_path)
    index.add(np.ones(4, dtype=np.float32), {"text": "alpha", "tokens": 1, "meta": {}})

    assert calls, "_flush() did not call atomic_write_text"
    assert calls[-1] == str(index_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_persistent_index_flush_failure_preserves_prior_state(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index = PersistentVectorIndex(index_path)
    index.add(np.ones(4, dtype=np.float32), {"text": "first", "tokens": 1, "meta": {}})
    assert index_path.exists()

    prior_content = index_path.read_text(encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(persistent_index_module, "atomic_write_text", boom)

    with pytest.raises(OSError, match="simulated disk full"):
        index.add(np.zeros(4, dtype=np.float32), {"text": "second", "tokens": 1, "meta": {}})

    assert index_path.read_text(encoding="utf-8") == prior_content
    assert list(tmp_path.glob("*.tmp")) == []
    reloaded = PersistentVectorIndex(index_path)
    assert len(reloaded._vectors) == 1
    assert reloaded._payloads[0]["text"] == "first"
