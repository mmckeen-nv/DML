from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import daystrom_dml.dml_adapter as adapter_module
from daystrom_dml.dml_adapter import DMLAdapter, PersistenceCommitError


class FixedEmbedder:
    def embed(self, text: str) -> np.ndarray:
        del text
        return np.ones(16, dtype=np.float32)


def make_adapter(tmp_path: Path) -> DMLAdapter:
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {
                "enable": True,
                "path": "dml_state.jsonl",
                "interval_sec": 0,
            },
            "rag_store": {"enable": False},
        },
        embedder=FixedEmbedder(),
        start_aging_loop=False,
    )


def test_ingest_propagates_dml_commit_failure_and_marks_degraded(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(adapter_module, "save_persisted_memories", fail_save)

    with pytest.raises(PersistenceCommitError, match="DML state") as exc_info:
        adapter.ingest("must not report success")

    assert isinstance(exc_info.value.__cause__, OSError)
    status = adapter.durability_status()
    assert status["status"] == "degraded"
    assert status["failures"]["dml"] == "OSError: simulated disk full"


def test_successful_dml_retry_clears_component_failure(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    real_save = adapter_module.save_persisted_memories
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient replacement failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "save_persisted_memories", fail_once)

    with pytest.raises(PersistenceCommitError):
        adapter.ingest("retryable durable memory")

    adapter._persist_dml_state()

    assert adapter.durability_status() == {"status": "ok", "failures": {}}
    assert adapter._persistence_path.exists()


def test_rag_commit_failure_propagates_and_marks_only_rag(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    real_atomic_write = adapter_module.atomic_write_text

    def fail_rag(path, content, *args, **kwargs):
        if Path(path) == adapter.rag_state_path:
            raise PermissionError("simulated read-only volume")
        return real_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_rag)

    with pytest.raises(PersistenceCommitError, match="RAG state") as exc_info:
        adapter._persist_rag_state()

    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert adapter.durability_status() == {
        "status": "degraded",
        "failures": {"rag": "PermissionError: simulated read-only volume"},
    }


def test_public_ingest_propagates_rag_commit_failure(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    real_atomic_write = adapter_module.atomic_write_text

    def fail_rag(path, content, *args, **kwargs):
        if Path(path) == adapter.rag_state_path:
            raise PermissionError("simulated RAG volume failure")
        return real_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_rag)

    with pytest.raises(PersistenceCommitError, match="RAG state"):
        adapter.ingest("public mutation must fail closed")

    assert adapter._persistence_path.exists()
    assert adapter.durability_status() == {
        "status": "degraded",
        "failures": {"rag": "PermissionError: simulated RAG volume failure"},
    }
