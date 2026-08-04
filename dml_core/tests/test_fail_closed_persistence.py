from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import daystrom_dml.dml_adapter as adapter_module
from daystrom_dml.dml_adapter import (
    DMLAdapter,
    PersistenceCommitError,
    PersistenceRollbackError,
)


class FixedEmbedder:
    def embed(self, text: str) -> np.ndarray:
        del text
        return np.ones(16, dtype=np.float32)


def make_adapter(tmp_path: Path, *, persistent_rag: bool = False) -> DMLAdapter:
    rag_store = {"enable": False}
    if persistent_rag:
        rag_store = {
            "enable": True,
            "path": "persistent_rag.faiss",
            "meta_path": "persistent_rag.json",
            "backend": "faiss",
            "dim": 16,
        }
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
            "rag_store": rag_store,
        },
        embedder=FixedEmbedder(),
        start_aging_loop=False,
    )


def test_ingest_propagates_dml_commit_failure_and_marks_degraded(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    assert adapter.rag_store.descriptors() == []

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(adapter_module, "save_persisted_memories", fail_save)

    with pytest.raises(PersistenceCommitError, match="DML state") as exc_info:
        adapter.ingest("must not report success")

    assert isinstance(exc_info.value.__cause__, OSError)
    status = adapter.durability_status()
    assert status["status"] == "degraded"
    assert status["failures"]["dml"] == "OSError: simulated disk full"
    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0


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
    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0
    assert adapter.durability_status() == {
        "status": "degraded",
        "failures": {"rag": "PermissionError: simulated RAG volume failure"},
    }

    reloaded = make_adapter(tmp_path)
    assert reloaded.memory_count() == 0
    assert reloaded.rag_store.catalog_summary()["count"] == 0
    reloaded.close(persist=False)


def test_failed_compensating_write_surfaces_rollback_error(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    real_save = adapter_module.save_persisted_memories
    real_atomic_write = adapter_module.atomic_write_text
    save_attempts = 0

    def fail_compensation(*args, **kwargs):
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 2:
            raise OSError("rollback disk failure")
        return real_save(*args, **kwargs)

    def fail_rag(path, content, *args, **kwargs):
        if Path(path) == adapter.rag_state_path:
            raise PermissionError("original RAG failure")
        return real_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "save_persisted_memories", fail_compensation)
    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_rag)

    with pytest.raises(PersistenceRollbackError) as exc_info:
        adapter.ingest("rollback failure must be explicit")

    assert isinstance(exc_info.value.original_error, PersistenceCommitError)
    assert isinstance(exc_info.value.rollback_error, OSError)
    assert exc_info.value.__cause__ is exc_info.value.original_error
    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0
    status = adapter.durability_status()
    assert status["status"] == "degraded"
    assert "rag" in status["failures"]
    assert status["failures"]["rollback"] == "OSError: rollback disk failure"


def test_nested_success_is_compensated_when_outer_mutation_fails(tmp_path):
    adapter = make_adapter(tmp_path)

    @adapter_module._serialized_mutation
    def outer_mutation(target):
        target.ingest("nested mutation")
        raise RuntimeError("outer operation failed")

    with pytest.raises(RuntimeError, match="outer operation failed"):
        outer_mutation(adapter)

    assert adapter.memory_count() == 0
    assert adapter.rag_store.catalog_summary()["count"] == 0
    reloaded = make_adapter(tmp_path)
    assert reloaded.memory_count() == 0
    assert reloaded.rag_store.catalog_summary()["count"] == 0
    reloaded.close(persist=False)


def test_persistent_rag_commit_is_compensated_after_legacy_failure(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    adapter = make_adapter(tmp_path, persistent_rag=True)
    real_atomic_write = adapter_module.atomic_write_text

    def fail_legacy_rag(path, content, *args, **kwargs):
        if Path(path) == adapter.rag_state_path:
            raise PermissionError("legacy RAG failure after persistent commit")
        return real_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "atomic_write_text", fail_legacy_rag)

    with pytest.raises(PersistenceCommitError, match="RAG state"):
        adapter.ingest("persistent RAG must compensate")

    assert adapter.memory_count() == 0
    assert adapter.persistent_rag_store is not None
    assert adapter.persistent_rag_store.search(np.ones(16), top_k=2) == []

    reloaded = make_adapter(tmp_path, persistent_rag=True)
    assert reloaded.memory_count() == 0
    assert reloaded.persistent_rag_store is not None
    assert reloaded.persistent_rag_store.search(np.ones(16), top_k=2) == []
    reloaded.close(persist=False)
