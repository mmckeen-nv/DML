from __future__ import annotations

import json
from pathlib import Path

from scripts.embedding_compatibility_status import (
    format_markdown_report,
    format_report,
    format_status_line,
    write_markdown_report,
)

import numpy as np

from daystrom_dml.dml_adapter import (
    DMLAdapter,
    KNOWLEDGE_ENTRY_PREVIEW_CHARS,
    KNOWLEDGE_MAX_ENTRIES,
)
from daystrom_dml.memory_store import MemoryStore
from daystrom_dml.summarizer import DummySummarizer


def make_store(**kwargs) -> MemoryStore:
    defaults = dict(
        summarizer=DummySummarizer(),
        beta_a=0.08,
        beta_r=0.2,
        eta=0.15,
        gamma=0.02,
        kappa=0.5,
        tau_s=0.3,
        theta_merge=0.92,
        K=4,
        capacity=20,
        start_aging_loop=False,
        similarity_threshold=0.0,
    )
    defaults.update(kwargs)
    return MemoryStore(**defaults)


def test_ingest_retrieve_reinforce(tmp_path):
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 50,
            "token_budget": 120,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
        },
        start_aging_loop=False,
    )
    adapter.ingest("The cat likes playing with yarn.")
    adapter.ingest("We observed the dog napping in the sun.")
    context = adapter.build_preamble("What animals are playful?")
    assert "Daystrom Memory Lattice" in context
    assert "=== User Prompt ===" in context

    before = adapter.stats()["count"]
    adapter.reinforce("What animals are playful?", "Cats enjoy play time.")
    after = adapter.stats()["count"]
    assert after >= before


def test_decay_adjusts_abstraction_level():
    store = make_store(tau_s=0.8)
    embedding = np.ones(8, dtype=np.float32)
    item, _ = store.ingest("Test memory", embedding, salience=1.0, fidelity=1.0)
    target_time = item.timestamp + 3600 * 24
    store.decay_step(now=target_time)
    items = store.items()
    levels = [it.level for it in items]
    assert any(level >= 1 for level in levels)
    assert any(it.fidelity <= 1.0 for it in items)


def test_merging_stabilises_memory_count():
    store = make_store()
    vec = np.ones(16, dtype=np.float32)
    first, merged_flag = store.ingest("Alpha observation", vec, salience=0.5)
    assert merged_flag is False
    second, merged_flag = store.ingest("Alpha observation repeated", vec, salience=0.5)
    assert merged_flag is True
    assert first.id == second.id
    assert len(store.items()) == 1
    item = store.items()[0]
    assert item.meta.get("merges", 0) >= 1


def test_capacity_eviction_prefers_stale_items() -> None:
    store = make_store(capacity=2, theta_merge=2.0)
    vec = np.ones(8, dtype=np.float32)
    old, _ = store.ingest("Old observation", vec, salience=0.5, fidelity=1.0)
    old.timestamp -= 3600 * 24
    mid, _ = store.ingest("Mid observation", vec, salience=0.5, fidelity=1.0)
    new, _ = store.ingest("New observation", vec, salience=0.9, fidelity=1.0)
    remaining = {item.id for item in store.items()}
    assert old.id not in remaining
    assert mid.id in remaining
    assert new.id in remaining


def test_knowledge_report_limits_payload(tmp_path):
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": KNOWLEDGE_MAX_ENTRIES + 50,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
        },
        start_aging_loop=False,
    )
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3
    total = KNOWLEDGE_MAX_ENTRIES + 20
    for idx in range(total):
        adapter.ingest(f"Entry {idx}: {long_text}")
    report = adapter.knowledge_report()
    dml = report["dml"]
    assert dml["count"] == total
    assert len(dml["entries"]) == KNOWLEDGE_MAX_ENTRIES
    assert dml["truncated"] is True
    assert all(len(entry["summary"]) <= KNOWLEDGE_ENTRY_PREVIEW_CHARS for entry in dml["entries"])


def test_similarity_threshold_filters_irrelevant_memories():
    store = make_store(similarity_threshold=0.25)
    on_topic_vec = np.ones(8, dtype=np.float32)
    off_topic_vec = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float32)

    store.ingest("Climate change impacts ecosystems", on_topic_vec, salience=0.8)
    store.ingest("Recipe for blueberry pancakes", off_topic_vec, salience=0.9)

    results = store.retrieve(on_topic_vec, top_k=2)

    assert any("Climate" in item.text for item in results)
    assert all("pancakes" not in item.text for item in results)


def test_similarity_threshold_backfills_top_k_after_filtering():
    store = make_store(similarity_threshold=0.2)
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    high_similarity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    borderline_high_salience = np.array([-0.2, 1.0, 0.0, 0.0], dtype=np.float32)
    mid_similarity = np.array([0.35, 1.0, 0.0, 0.0], dtype=np.float32)

    store.ingest("Direct match", high_similarity, salience=0.5)
    store.ingest("Off topic but salient", borderline_high_salience, salience=80.0)
    store.ingest("Related but quieter", mid_similarity, salience=0.5)

    results = store.retrieve(query_vec, top_k=2)

    assert len(results) == 2
    texts = {item.text for item in results}
    assert "Off topic but salient" not in texts
    assert {"Direct match", "Related but quieter"} == texts


def test_embedding_compatibility_migration_writes_report(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
        },
        start_aging_loop=False,
    )
    payload = {
        "items": [
            {"text": "legacy-memory-a", "embedding": [1.0, 2.0]},
            {"text": "legacy-memory-b", "embedding": [3.0, 4.0]},
        ]
    }

    report = adapter._ensure_embedding_compatibility(payload)

    assert report["status"] == "migrated"
    assert report["phase"] == "done"
    assert report["phase_detail"].startswith("completed compatibility migration:")
    assert report["total_items"] == 2
    assert report["checked"] == 2
    assert report["remaining_items"] == 0
    assert report["last_checked_index"] == 2
    assert report["last_completed_item_index"] == 2
    assert report["last_completed_item_preview"] == "legacy-memory-b"
    assert report["progress_pct"] == 100.0
    assert report["current_item_index"] == 2
    assert report["current_item_preview"] is None
    assert report["started_at"]
    assert report["updated_at"]
    assert report["phase_started_at"]
    assert report["mismatched"] == 2
    assert report["reembedded"] == 2
    assert report["failed"] == 0
    assert report["target_dim"] > 0
    for entry in payload["items"]:
        assert len(entry["embedding"]) == report["target_dim"]

    report_path = adapter.storage_dir / "embedding_compatibility_report.json"
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["status"] == "migrated"
    assert written["phase"] == "done"
    assert written["phase_detail"].startswith("completed compatibility migration:")
    assert written["total_items"] == 2
    assert written["remaining_items"] == 0
    assert written["last_checked_index"] == 2
    assert written["last_completed_item_index"] == 2
    assert written["last_completed_item_preview"] == "legacy-memory-b"
    assert written["progress_pct"] == 100.0
    assert written["current_item_index"] == 2
    assert written["current_item_preview"] is None
    assert written["started_at"]
    assert written["updated_at"]
    assert written["phase_started_at"]
    assert written["mismatched"] == 2
    assert written["reembedded"] == 2

    rendered = format_report(written, report_path=Path(report_path))
    assert "status: migrated" in rendered
    assert "phase: done" in rendered
    assert "progress: 100.00% (2/2, remaining=0)" in rendered
    assert "last_completed: index=2 preview=legacy-memory-b" in rendered

    status_line = format_status_line(written, report_path=Path(report_path))
    assert "migration_status=migrated" in status_line
    assert "phase=done" in status_line
    assert "progress=100.00%" in status_line
    assert f"report={report_path}" in status_line

    markdown = format_markdown_report(written, report_path=Path(report_path))
    assert "# DML Ollama Live-Store Migration Status" in markdown
    assert "- status_line: `migration_status=migrated | phase=done | progress=100.00%" in markdown
    assert "- status: `migrated`" in markdown
    assert "- progress: `100.00% (2/2, remaining=0)`" in markdown

    markdown_path = tmp_path / "migration-status.md"
    write_markdown_report(written, report_path=Path(report_path), output_path=markdown_path)
    assert markdown_path.exists()
    assert "Generated from the durable live-store migration artifact" in markdown_path.read_text(encoding="utf-8")
