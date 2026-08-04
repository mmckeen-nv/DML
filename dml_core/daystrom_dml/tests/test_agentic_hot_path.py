"""Regression tests for the procedural-memory hot path and shared state."""
from __future__ import annotations

import json

import numpy as np
import pytest

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.summarizer import DummySummarizer


class _FixedEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[sum(text.encode("utf-8")) % self.dim] = 1.0
        return vector


def _adapter(tmp_path, *, mirror=False, rag=False):
    config = {
        "model_name": "dummy",
        "embedding_model": None,
        "storage_dir": str(tmp_path),
        "theta_merge": 2.0,
        "token_budget": 600,
        "dml_top_k": 10,
        "metrics_enabled": False,
        "mirror_agentic_memory_to_rag": mirror,
        "persistence": {"enable": True, "path": "dml_state.jsonl", "interval_sec": 0},
    }
    if rag:
        config["rag_store"] = {
            "enable": True,
            "path": "rag_index.faiss",
            "meta_path": "rag_meta.json",
            "backend": "faiss",
            "dim": 8,
        }
    return DMLAdapter(
        config_overrides=config,
        embedder=_FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


def test_agentic_write_avoids_rag_by_default(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    rag_calls = []
    monkeypatch.setattr(adapter.rag_store, "add_document", lambda *args, **kwargs: rag_calls.append((args, kwargs)))

    adapter.ingest_agentic(
        "Tool: exact_tool\nArguments: {\"value\": 1}\nVerified result: success",
        "action",
        {"memory_class": "tool_cookbook_event", "status": "success"},
    )

    assert adapter.stats()["count"] == 1
    assert rag_calls == []
    adapter.close(persist=False)


def test_agentic_rag_mirroring_remains_opt_in(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, mirror=True)
    rag_calls = []
    monkeypatch.setattr(adapter.rag_store, "add_document", lambda *args, **kwargs: rag_calls.append((args, kwargs)))

    adapter.ingest_agentic(
        "Tool: exact_tool\nArguments: {\"value\": 1}\nVerified result: success",
        "action",
        {"memory_class": "tool_cookbook_event", "status": "success"},
    )

    assert len(rag_calls) == 1
    adapter.close(persist=False)


def test_external_state_change_refreshes_before_read(tmp_path):
    reader = _adapter(tmp_path)
    writer = _adapter(tmp_path)
    writer.ingest_memory(
        "Externally persisted verified procedure",
        tenant_id="tenant-a",
        meta={"memory_class": "tool_cookbook_event", "status": "success"},
    )

    assert reader.stats()["count"] == 0
    assert reader.refresh_if_changed() is True
    assert any("Externally persisted" in item.text for item in reader.store.items())

    reader.close(persist=False)
    writer.close(persist=False)


def test_concurrent_reinforce_refreshes_before_persisting(tmp_path):
    first = _adapter(tmp_path)
    second = _adapter(tmp_path)

    first.reinforce("first prompt", "first durable response")
    second.reinforce("second prompt", "second durable response")

    restored = _adapter(tmp_path)
    texts = [item.text for item in restored.store.items()]
    assert any("first durable response" in text for text in texts)
    assert any("second durable response" in text for text in texts)

    first.close(persist=False)
    second.close(persist=False)
    restored.close(persist=False)


def test_stale_reader_close_cannot_overwrite_newer_state(tmp_path):
    stale_reader = _adapter(tmp_path)
    writer = _adapter(tmp_path)
    writer.ingest_memory("new durable state", tenant_id="tenant-a")
    stale_reader.close(persist=False)
    writer.close(persist=False)

    verifier = _adapter(tmp_path)
    assert any(item.text == "new durable state" for item in verifier.store.items())
    verifier.close(persist=False)


def test_procedural_query_expands_selected_success_and_excludes_failure(tmp_path):
    adapter = _adapter(tmp_path)
    scaffold = (
        "Tool name: exact_tool_name\n"
        "Argument structure: {\"path\": \"<path>\", \"mode\": \"safe\"}\n"
        "Validated call scaffold: exact_tool_name(path=\"sample.txt\", mode=\"safe\")\n"
        "Verification: returned status=ok\n"
        "Verified result: artifact created"
    )
    adapter.ingest_memory(
        scaffold,
        tenant_id="tenant-a",
        meta={"memory_class": "tool_cookbook_event", "status": "success", "source": "procedure:success"},
    )
    adapter.ingest_memory(
        "Tool name: broken_tool\nTraceback (most recent call last): failed",
        tenant_id="tenant-a",
        kind="error",
        meta={"memory_class": "tool_cookbook_event", "status": "failed", "source": "procedure:failed"},
    )
    adapter.ingest_memory(
        "unrelated material " * 2000,
        tenant_id="tenant-a",
        meta={"source": "unrelated:large"},
    )

    report = adapter.query_database("exact tool argument structure validated scaffold", mode="semantic")

    assert "exact_tool_name" in report["context"]
    assert '"path": "<path>"' in report["context"]
    assert "broken_tool" not in report["context"]
    assert "procedure:success" in report["source_docs"]
    assert len(report["context"]) <= 16000
    adapter.close(persist=False)


def test_persistent_rag_formatting_excludes_suppressed_records(tmp_path):
    adapter = _adapter(tmp_path)
    formatted = adapter._format_rag_matches(
        [
            {
                "id": 7,
                "text": "suppressed procedure",
                "score": 1.0,
                "meta": {"memory_state": "suppressed", "source": "procedure:suppressed"},
            }
        ]
    )
    assert formatted == []
    adapter.close(persist=False)


def test_durable_rag_load_suppresses_legacy_replay(tmp_path):
    pytest.importorskip("faiss")
    first = _adapter(tmp_path, rag=True)
    first.ingest("explicit document", meta={"source": "document:one"})
    first.close()
    assert first.persistent_rag_store is not None
    count_before = len(first.persistent_rag_store._records)
    assert count_before == 1

    (tmp_path / "rag_store.json").write_text(
        json.dumps({"documents": [{"text": "legacy duplicate", "meta": {"source": "legacy"}}]}),
        encoding="utf-8",
    )
    second = _adapter(tmp_path, rag=True)
    try:
        assert second._persistent_rag_loaded is True
        assert second.persistent_rag_store is not None
        assert len(second.persistent_rag_store._records) == count_before
    finally:
        second.close(persist=False)
