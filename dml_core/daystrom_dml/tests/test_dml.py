from __future__ import annotations

import json
from pathlib import Path

from scripts.embedding_compatibility_status import (
    format_markdown_report,
    format_progress_snapshot,
    format_report,
    format_status_line,
    write_markdown_report,
    write_progress_snapshot,
)

import numpy as np

from daystrom_dml.dml_adapter import (
    DMLAdapter,
    KNOWLEDGE_ENTRY_PREVIEW_CHARS,
    KNOWLEDGE_MAX_ENTRIES,
)
from daystrom_dml.embeddings import RandomEmbedder
from daystrom_dml.gpt_runner import GPTRunner
from daystrom_dml.memory_store import MemoryStore
from daystrom_dml.summarizer import DummySummarizer


class FixedEmbedder:
    def embed(self, _text: str) -> np.ndarray:
        return np.ones(16, dtype=np.float32)


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


def test_generation_prompt_keeps_memory_provider_silent(tmp_path):
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
    prompt = adapter._compose_prompt(
        "Who controls inventory?",
        "=== Daystrom Memory Lattice ===\n- L0 (f=1.00): Quartermaster Imani Vale controls inventory.\n=== User Prompt ===\nWho controls inventory?",
    )

    assert "Quartermaster Imani Vale controls inventory." in prompt
    assert "Daystrom Memory Lattice" not in prompt
    assert "RAG Retrieval" not in prompt
    assert prompt.count("=== User Prompt ===") == 1
    assert "Who controls inventory?" in prompt
    assert "Do not mention DML, RAG, retrieved context" in prompt




def test_adapter_imports_legacy_rag_state_by_default(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "rag_store.json").write_text(
        json.dumps({"documents": [{"text": "working foreground RAG document", "meta": {"source": "legacy"}}]}),
        encoding="utf-8",
    )

    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(storage),
            "persistence": {"enable": False},
        },
        start_aging_loop=False,
    )
    try:
        assert adapter.rag_store.catalog_summary()["count"] == 1
    finally:
        adapter.close()

def test_adapter_can_skip_legacy_rag_state_import(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "rag_store.json").write_text(
        json.dumps({"documents": [{"text": "stale foreground RAG document", "meta": {"source": "legacy"}}]}),
        encoding="utf-8",
    )

    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(storage),
            "persistence": {"enable": False},
            "skip_rag_state_import": True,
        },
        start_aging_loop=False,
    )
    try:
        assert adapter.rag_store.catalog_summary()["count"] == 0
    finally:
        adapter.close()

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


def test_ingest_assigns_first_class_lattice_metadata():
    store = make_store(theta_merge=2.0)
    vectors = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    ]

    for idx, vec in enumerate(vectors):
        store.ingest(f"Durable memory {idx}", vec, salience=0.5)

    items = store.items()
    assert len(items) == 3
    metas = [item.meta or {} for item in items]
    assert all(meta.get("lattice_policy") == "semantic-topic-time-v1" for meta in metas)
    assert all(isinstance(meta.get("lattice_row"), int) for meta in metas)
    assert all(isinstance(meta.get("lattice_col"), int) for meta in metas)
    assert all(isinstance(meta.get("lattice_layer"), int) for meta in metas)
    assert any(meta.get("lattice_neighbors") for meta in metas)


def test_import_repairs_legacy_items_with_missing_lattice_metadata():
    store = make_store(theta_merge=2.0)
    payload = {
        "items": [
            {
                "id": 0,
                "text": "Legacy A",
                "timestamp": 1.0,
                "salience": 0.5,
                "fidelity": 1.0,
                "level": 1,
                "meta": {},
                "summary_of": [0],
                "embedding": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "id": 1,
                "text": "Legacy B",
                "timestamp": 2.0,
                "salience": 0.5,
                "fidelity": 1.0,
                "level": 2,
                "meta": {},
                "summary_of": [1],
                "embedding": [0.9, 0.1, 0.0, 0.0],
            },
        ]
    }

    store.import_state(payload)

    items = store.items()
    metas = [item.meta or {} for item in items]
    assert all(meta.get("lattice_policy") == "semantic-topic-time-v1" for meta in metas)
    assert all("lattice_row" in meta and "lattice_col" in meta for meta in metas)
    assert {meta.get("lattice_layer") for meta in metas} == {1, 2}
    assert any(meta.get("lattice_neighbors") for meta in metas)


def test_gpt_runner_summary_strips_instruction_preface() -> None:
    noisy = (
        "Here is a summary of the content in 256 characters or less:\n\n"
        '"Maintain Citizen Snips continuity using compact DML state."'
    )

    assert (
        GPTRunner._clean_summary_output(noisy, max_len=256)
        == "Maintain Citizen Snips continuity using compact DML state."
    )


def test_adapter_merge_preserves_incoming_conflict_metadata(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "theta_merge": 0.5,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
        },
        embedder=FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest(
        "Deploy mode is automatic.",
        meta={
            "tenant_id": "alpha",
            "namespace": "ops",
            "conflict_key": "deploy_mode",
            "claim_value": "automatic",
        },
    )
    adapter.ingest(
        "Deploy mode is manual.",
        meta={
            "tenant_id": "alpha",
            "namespace": "ops",
            "conflict_key": "deploy_mode",
            "claim_value": "manual",
            "conflict_state": "conflicted",
            "conflicts_with": [{"id": 0, "claim_value": "automatic"}],
        },
    )

    item = adapter.store.items()[0]
    assert item.meta["conflict_state"] == "conflicted"
    assert item.meta["claim_value"] == "manual"
    assert item.meta["conflicts_with"][0]["claim_value"] == "automatic"


def test_no_merge_metadata_keeps_continuity_checkpoints_distinct(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "theta_merge": 0.5,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
        },
        embedder=FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    for idx in range(2):
        adapter.ingest(
            f"thread: session-{idx}\nnext_action: run {idx}",
            meta={
                "tenant_id": "openclaw",
                "session_id": f"session-{idx}",
                "namespace": "active_continuity",
                "merge_policy": "never",
                "no_merge": True,
            },
        )

    assert len(adapter.store.items()) == 2
    assert {item.meta["session_id"] for item in adapter.store.items()} == {"session-0", "session-1"}


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


def test_retrieve_context_respects_tenant_scope(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest_memory(
        "Tenant alpha deployment uses blue release lanes.",
        tenant_id="alpha",
        kind="note",
    )
    adapter.ingest_memory(
        "Tenant beta deployment uses green release lanes.",
        tenant_id="beta",
        kind="note",
    )

    report = adapter.retrieve_context("deployment release lanes", tenant_id="alpha", top_k=5)

    assert report["items"]
    assert {item["meta"]["tenant_id"] for item in report["items"]} == {"alpha"}
    assert all("embedding" not in item for item in report["items"])
    assert "Tenant alpha" in report["raw_context"]
    assert "Tenant beta" not in report["raw_context"]


def test_retrieve_context_respects_single_user_session_scope(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "theta_merge": 2.0,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest_memory(
        "OPENCLAW-SESSION-A builds the installer plan.",
        tenant_id="openclaw",
        session_id="session-a",
        kind="note",
    )
    adapter.ingest_memory(
        "OPENCLAW-SESSION-B debugs the provider UI.",
        tenant_id="openclaw",
        session_id="session-b",
        kind="note",
    )

    session_report = adapter.retrieve_context(
        "openclaw sessions",
        tenant_id="openclaw",
        session_id="session-a",
        top_k=5,
    )
    assert session_report["items"]
    assert {item["meta"]["session_id"] for item in session_report["items"]} == {"session-a"}
    assert "OPENCLAW-SESSION-A" in session_report["raw_context"]
    assert "OPENCLAW-SESSION-B" not in session_report["raw_context"]

    tenant_report = adapter.retrieve_context(
        "openclaw sessions",
        tenant_id="openclaw",
        top_k=5,
    )
    assert {item["meta"]["session_id"] for item in tenant_report["items"]} == {"session-a", "session-b"}
    assert "OPENCLAW-SESSION-A" in tenant_report["raw_context"]
    assert "OPENCLAW-SESSION-B" in tenant_report["raw_context"]


def test_survival_ledger_carries_long_horizon_anchors(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "theta_merge": 2.0,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
            "token_budget": 80,
            "dml_context_max_items": 2,
        },
        embedder=FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    for idx, anchor in enumerate(
        [
            "HORIZON-ANCHOR-005",
            "HORIZON-ANCHOR-025",
            "HORIZON-ANCHOR-050",
            "HORIZON-ANCHOR-075",
            "HORIZON-ANCHOR-100",
        ],
        start=1,
    ):
        adapter.ingest_memory(
            f"Compaction cycle {idx} preserves durable milestone {anchor}.",
            tenant_id="openclaw",
            session_id="long-run",
            kind="note",
            meta={
                "compaction_cycle": idx,
                "virtual_tokens": idx * 250_000_000,
                "source": "compactor",
            },
        )
    for idx in range(6):
        adapter.ingest_memory(
            f"Recent noisy execution detail {idx} should not erase survival anchors.",
            tenant_id="openclaw",
            session_id="long-run",
            kind="note",
        )

    report = adapter.retrieve_context(
        "Which HORIZON anchors survived?",
        tenant_id="openclaw",
        session_id="long-run",
        top_k=1,
    )

    assert report["survival_ledger_included"] is True
    assert "Survival ledger" in report["raw_context"]
    assert "HORIZON-ANCHOR-005" in report["raw_context"]
    assert "HORIZON-ANCHOR-100" in report["raw_context"]


def test_survival_ledger_is_scoped_to_session(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "theta_merge": 2.0,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=FixedEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    adapter.ingest_memory(
        "Compaction checkpoint keeps SESSION-A-ANCHOR-777.",
        tenant_id="openclaw",
        session_id="session-a",
        kind="note",
        meta={"compaction_cycle": 1},
    )
    adapter.ingest_memory(
        "Compaction checkpoint keeps SESSION-B-ANCHOR-999.",
        tenant_id="openclaw",
        session_id="session-b",
        kind="note",
        meta={"compaction_cycle": 1},
    )

    report = adapter.retrieve_context(
        "Which session anchor survived?",
        tenant_id="openclaw",
        session_id="session-a",
        top_k=5,
    )

    assert "SESSION-A-ANCHOR-777" in report["raw_context"]
    assert "SESSION-B-ANCHOR-999" not in report["raw_context"]


def test_retrieve_context_applies_phase_filtering(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Plan memory should stay out of execute context.", meta={"kind": "plan"})
    adapter.ingest("Action memory should appear in execute context.", meta={"kind": "action"})

    report = adapter.retrieve_context("memory context", phase="execute", top_k=5)

    assert report["items"]
    assert {item["meta"]["kind"] for item in report["items"]} == {"action"}
    assert "Action memory" in report["raw_context"]
    assert "Plan memory" not in report["raw_context"]


def test_retrieve_context_respects_token_budget_and_omits_vectors(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
            "token_budget": 8,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Short memory one.", meta={"kind": "note"})
    adapter.ingest("This second memory is intentionally long enough to exceed the tiny budget.", meta={"kind": "note"})

    report = adapter.retrieve_context("memory", top_k=5)

    assert report["context_tokens"] <= 8
    assert all("embedding" not in item for item in report["items"])


def test_retrieval_report_caps_context_items_and_summary_size(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
            "dml_context_max_items": 2,
            "dml_context_summary_chars": 72,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    for idx in range(5):
        adapter.ingest(
            f"Memory {idx} contains detailed operational context about telemetry trimming and context packing.",
            meta={"kind": "note"},
        )

    report = adapter.retrieval_report("telemetry context packing", top_k=5)

    assert len(report["entries"]) <= 2
    assert all(len(entry["summary"]) <= 72 for entry in report["entries"])


def test_asteria_answer_key_scores_base_dml_and_rag(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    key = adapter._answer_key_for_prompt("What are the fuel reserves for Asteria Crossing?")
    full = adapter._evaluate_answer_accuracy(
        "Fuel reserves include 41,200 tonnes deuterium slush, 7,900 tonnes helium-3, "
        "320 kilograms antimatter catalyst, 8,400 tonnes argon, and 19,000 tonnes shield ice.",
        key,
    )
    partial = adapter._evaluate_answer_accuracy("Fuel reserves include argon.", key)

    assert full["scored"] is True
    assert full["score"] == 1.0
    assert partial["score"] < full["score"]
    assert "41,200 tonnes deuterium slush" in partial["missing"]


def test_retrieve_context_falls_back_to_recent_memory_when_similarity_filters_all(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 1.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Recent context should remain available under strict thresholds.", meta={"kind": "note"})

    report = adapter.retrieve_context("unrelated query", top_k=5)

    assert report["items"]
    assert "Recent context" in report["raw_context"]


def test_retrieve_context_excludes_quarantined_memories_by_default(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest(
        "Quarantined legacy archive memory should stay out of normal retrieval.",
        meta={"kind": "note", "source": "legacy", "memory_state": "quarantined"},
    )
    adapter.ingest(
        "Active continuity memory should retrieve normally.",
        meta={"kind": "note", "source": "active", "memory_state": "active"},
    )

    report = adapter.retrieve_context("legacy archive continuity memory", top_k=5)

    assert report["items"]
    assert all(item["meta"].get("memory_state") != "quarantined" for item in report["items"])
    assert "Active continuity" in report["raw_context"]
    assert "Quarantined legacy" not in report["raw_context"]


def test_retrieve_context_can_include_quarantined_memories_when_requested(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest(
        "Quarantined legacy archive memory can be inspected on demand.",
        meta={"kind": "note", "source": "legacy", "memory_state": "quarantined"},
    )

    report = adapter.retrieve_context(
        "legacy archive memory",
        top_k=5,
        include_quarantined=True,
    )

    assert report["include_quarantined"] is True
    assert any(item["meta"].get("memory_state") == "quarantined" for item in report["items"])
    assert "Quarantined legacy" in report["raw_context"]


def test_scoped_recent_fallback_does_not_cross_tenant_boundary(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 1.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest_memory(
        "Tenant alpha context must not leak through recent fallback.",
        tenant_id="alpha",
        client_id="client-a",
        session_id="session-a",
        instance_id="instance-a",
        kind="observation",
        meta={"phase": "execute"},
    )

    report = adapter.retrieve_context(
        "unrelated query",
        tenant_id="beta",
        client_id="client-b",
        session_id="session-b",
        instance_id="instance-b",
        phase="execute",
        top_k=5,
    )

    assert report["items"] == []
    assert "Tenant alpha" not in report["raw_context"]


def test_recent_fallback_respects_phase(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 1.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Plan fallback should stay out of execute context.", meta={"kind": "action", "phase": "plan"})
    adapter.ingest("Execute fallback should appear in execute context.", meta={"kind": "action", "phase": "execute"})

    report = adapter.retrieve_context("unrelated query", phase="execute", top_k=5)

    assert report["items"]
    assert {item["meta"]["phase"] for item in report["items"]} == {"execute"}
    assert "Execute fallback" in report["raw_context"]
    assert "Plan fallback" not in report["raw_context"]


def test_personality_matrix_overlay_is_added_to_context(tmp_path) -> None:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "schema_version": "dpm.replay-overlay.v1",
                "overlay_id": "overlay:relationship:test:active-read",
                "mode": "active-read",
                "generated_at": "2026-05-18T00:00:00Z",
                "scope": {
                    "primary": "relationship",
                    "thread_id": None,
                    "project_id": "project:test",
                    "relationship_id": "relationship:test",
                },
                "retrieval_order_applied": ["relationship"],
                "overlay": {
                    "persona_summary": "Be direct and careful.",
                    "style_directives": ["Prefer direct answers."],
                    "do_not_do": ["Do not override explicit instructions."],
                    "open_questions": [],
                    "max_chars": 120,
                    "rendered_text": "Direct, careful, and privacy-safe.",
                },
                "effective_constraints": {
                    "explicit_instruction_precedence": "always_override",
                    "narrowest_scope_wins": True,
                    "cross_scope_fallback_requires_compatibility": True,
                    "writes_allowed": False,
                },
                "sources": [
                    {
                        "source_id": "relationship:test",
                        "scope": "relationship",
                        "kind": "relationship_summary",
                        "included": True,
                        "priority": 1,
                        "confidence": 0.9,
                        "updated_at": "2026-05-18T00:00:00Z",
                        "summary": "Stable directness preference.",
                    }
                ],
                "audit": {
                    "included_source_ids": ["relationship:test"],
                    "excluded_sources": [],
                    "conflicts_detected": [],
                    "notes": [],
                },
                "override_state": {
                    "has_explicit_instruction": False,
                    "instruction_source_id": None,
                    "override_applied": False,
                    "suppressed_source_ids": [],
                    "effective_for_turn": [],
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-read",
                "overlay_path": str(overlay_path),
                "max_overlay_chars": 80,
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Runtime memory remains available beside the matrix.", meta={"kind": "note"})

    report = adapter.retrieve_context("runtime memory", top_k=5)

    assert report["personality_overlay"]["schema_version"] == "dpm.replay-overlay.v1"
    assert report["personality_overlay"]["mode"] == "active-read"
    assert "=== Personality Matrix ===" in report["raw_context"]
    assert "Direct, careful, and privacy-safe." in report["raw_context"]
    assert "Runtime memory" in report["raw_context"]


def test_personality_matrix_can_render_preference_graph(tmp_path) -> None:
    graph_path = tmp_path / "preference-graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "dpm.preference-graph.v1",
                "graph_id": "preference-graph:relationship:test",
                "subject_id": "relationship:test",
                "generated_at": "2026-05-18T00:00:00Z",
                "nodes": [
                    {
                        "id": "pref.directness",
                        "kind": "interaction_style",
                        "label": "Directness",
                        "scope": "relationship",
                        "state": "active",
                        "weight": 0.9,
                        "confidence": 0.8,
                        "polarity": "prefer_high",
                        "value": {"target": 0.9},
                        "updated_at": "2026-05-18T00:00:00Z",
                    }
                ],
                "edges": [],
                "audit": {},
            }
        ),
        encoding="utf-8",
    )
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-read",
                "preference_graph_path": str(graph_path),
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    overlay = adapter.personality_overlay(prompt="answer directly")

    assert overlay["schema_version"] == "dpm.replay-overlay.v1"
    assert overlay["sources"][0]["kind"] == "preference_graph"
    assert "Prefer directness." in overlay["overlay"]["rendered_text"]


def test_personality_matrix_active_write_records_preference_graph(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "preference_graph_path": str(graph_path),
                "relationship_id": "relationship:test",
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    result = adapter.record_personality_preference(
        "I prefer concise direct answers.",
        source_id="turn:test",
    )

    assert result["status"] == "recorded"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["schema_version"] == "dpm.preference-graph.v1"
    assert graph["nodes"]
    node = graph["nodes"][0]
    assert node["id"].startswith("pref.")
    assert node["state"] == "active"
    assert node["evidence"]["support_count"] == 1
    assert node["provenance"][0]["source_id"] == "turn:test"

    overlay = adapter.personality_overlay(prompt="help")
    assert overlay["sources"][0]["kind"] == "preference_graph"
    assert overlay["effective_constraints"]["writes_allowed"] is True


def test_personality_matrix_active_read_does_not_write_preference_graph(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-read",
                "preference_graph_path": str(graph_path),
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    result = adapter.record_personality_preference("I prefer terse replies.")

    assert result is None
    assert not graph_path.exists()


def test_personality_matrix_prefers_newer_cross_session_graph_over_stale_overlay(tmp_path) -> None:
    overlay_path = tmp_path / "static-overlay.json"
    graph_path = tmp_path / "dpm-graph.json"
    overlay_path.write_text(
        json.dumps(
            {
                "schema_version": "dpm.replay-overlay.v1",
                "overlay_id": "overlay:relationship:test:active-write",
                "mode": "active-write",
                "generated_at": "2026-01-01T00:00:00Z",
                "scope": {"primary": "relationship", "relationship_id": "relationship:test"},
                "retrieval_order_applied": ["relationship"],
                "overlay": {
                    "persona_summary": "Prefer stale static overlay.",
                    "style_directives": ["Prefer stale static overlay."],
                    "do_not_do": [],
                    "open_questions": [],
                    "max_chars": 200,
                    "rendered_text": "Preferences: Prefer stale static overlay.",
                },
                "sources": [],
                "audit": {},
            }
        ),
        encoding="utf-8",
    )
    first = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "overlay_path": str(overlay_path),
                "preference_graph_path": str(graph_path),
                "relationship_id": "relationship:test",
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    first.record_personality_preference("I prefer evolving graph overlays.", source_id="turn:one")

    second = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "overlay_path": str(overlay_path),
                "preference_graph_path": str(graph_path),
                "relationship_id": "relationship:test",
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    overlay = second.personality_overlay(prompt="help")
    assert overlay is not None
    rendered = overlay["overlay"]["rendered_text"]

    assert "evolving graph overlays" in rendered.lower()
    assert "stale static overlay" not in rendered.lower()


def test_personality_matrix_explicit_correction_conflicts_with_existing_preference_across_sessions(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    config = {
        "model_name": "dummy",
        "embedding_model": None,
        "storage_dir": str(tmp_path / "storage"),
        "persistence": {"enable": False},
        "metrics_enabled": False,
        "dpm": {
            "enable": True,
            "mode": "active-write",
            "preference_graph_path": str(graph_path),
            "relationship_id": "relationship:test",
        },
    }
    first = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    first.record_personality_preference("I prefer concise direct answers.", source_id="turn:one")

    second = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    second.record_personality_preference("Do not prefer concise direct answers.", source_id="turn:two", explicit=True)

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph["nodes"]

    assert len(nodes) == 1
    node = nodes[0]
    assert node["state"] == "conflicted"
    assert node["evidence"]["support_count"] == 1
    assert node["evidence"]["contradiction_count"] == 1
    assert node["provenance"][-1]["source_id"] == "turn:two"


def test_reinforce_can_record_explicit_dpm_preference(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "preference_graph_path": str(graph_path),
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    adapter.reinforce(
        "For this project, please use terse status updates.",
        "Understood.",
        meta={"dpm_preference": True, "source": "turn:reinforce"},
    )

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["nodes"]
    assert graph["nodes"][0]["provenance"][0]["source_id"] == "turn:reinforce"


def test_personality_matrix_active_write_survives_restart(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    config = {
        "model_name": "dummy",
        "embedding_model": None,
        "storage_dir": str(tmp_path / "storage"),
        "persistence": {"enable": False},
        "metrics_enabled": False,
        "dpm": {
            "enable": True,
            "mode": "active-write",
            "preference_graph_path": str(graph_path),
        },
    }
    adapter = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.record_personality_preference("I prefer compact status updates.", source_id="turn:restart")
    adapter.close()

    reloaded = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    try:
        overlay = reloaded.personality_overlay(prompt="status")
    finally:
        reloaded.close()

    assert overlay["sources"][0]["kind"] == "preference_graph"
    assert "compact status updates" in overlay["overlay"]["rendered_text"]


def test_personality_matrix_can_suppress_and_delete_preference(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "preference_graph_path": str(graph_path),
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    result = adapter.record_personality_preference("I prefer compact status updates.")
    node_id = result["node_id"]

    suppressed = adapter.suppress_personality_preference(node_id, reason="test_suppression")
    graph = adapter.personality_graph()
    node = next(node for node in graph["nodes"] if node["id"] == node_id)
    assert suppressed["status"] == "suppressed"
    assert node["state"] == "suppressed"

    deleted = adapter.delete_personality_preference(node_id)
    graph = adapter.personality_graph()
    assert deleted["status"] == "deleted"
    assert all(node["id"] != node_id for node in graph["nodes"])


def test_personality_matrix_overlay_respects_token_budget(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "preference_graph_path": str(graph_path),
                "token_budget": 4,
                "max_overlay_chars": 200,
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.record_personality_preference(
        "I prefer compact status updates with careful explicit audit context and no ornamental phrasing.",
        explicit=True,
    )

    overlay = adapter.personality_overlay(prompt="status")

    assert overlay["overlay"]["rendered_text"]
    assert len(overlay["overlay"]["rendered_text"].split()) <= 8


def test_personality_matrix_overlay_dedupes_and_cuts_on_boundaries(tmp_path) -> None:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(
        json.dumps(
            {
                "schema_version": "dpm.replay-overlay.v1",
                "overlay_id": "overlay:relationship:test:active-read",
                "mode": "active-read",
                "generated_at": "2026-05-18T00:00:00Z",
                "scope": {"primary": "relationship", "relationship_id": "relationship:test"},
                "retrieval_order_applied": ["relationship"],
                "overlay": {
                    "style_directives": ["Prefer direct answers.", "Prefer direct answers."],
                    "do_not_do": ["Current-turn instructions override the DPM overlay."],
                    "max_chars": 72,
                    "rendered_text": (
                        "Prefer direct answers. Prefer direct answers. "
                        "Avoid decorative filler words while preserving useful context."
                    ),
                },
                "sources": [],
                "audit": {},
            }
        ),
        encoding="utf-8",
    )
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-read",
                "overlay_path": str(overlay_path),
                "max_overlay_chars": 72,
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    overlay = adapter.personality_overlay(prompt="Use more detail for this answer.")
    body = overlay["overlay"]

    assert body["style_directives"] == ["Prefer direct answers."]
    assert body["do_not_do"] == ["Current-turn instructions override the DPM overlay."]
    assert body["rendered_text"] == "Prefer direct answers. Avoid decorative filler words while preserving"
    assert not body["rendered_text"].endswith("preserv")


def test_ingest_memory_persists_scoped_items(tmp_path) -> None:
    storage_dir = tmp_path / "storage"
    config = {
        "model_name": "dummy",
        "embedding_model": None,
        "storage_dir": str(storage_dir),
        "persistence": {"enable": True, "path": "dml_state.jsonl"},
        "rag_store": {"enable": False},
        "metrics_enabled": False,
        "similarity_threshold": 0.0,
    }
    adapter = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest_memory(
        "Scoped durable memory survives adapter restart.",
        tenant_id="alpha",
        kind="note",
    )
    adapter.close()

    reloaded = DMLAdapter(
        config_overrides=config,
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    try:
        report = reloaded.retrieve_context("durable memory", tenant_id="alpha", top_k=5)
    finally:
        reloaded.close()

    assert report["items"]
    assert report["items"][0]["meta"]["tenant_id"] == "alpha"
    assert "Scoped durable memory" in report["raw_context"]


def test_retrieve_context_falls_back_to_legacy_unscoped_memories(tmp_path) -> None:
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Legacy unscoped memory survives tenant-aware retrieval.", meta={"kind": "note"})

    report = adapter.retrieve_context("legacy memory", tenant_id="openclaw", top_k=5)

    assert report["items"]
    assert report["items"][0]["meta"].get("tenant_id") is None
    assert "Legacy unscoped memory" in report["raw_context"]


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
    assert "freshness: fresh updated_age_s=" in rendered

    status_line = format_status_line(written, report_path=Path(report_path))
    assert "migration_status=migrated" in status_line
    assert "phase=done" in status_line
    assert "progress=100.00%" in status_line
    assert "freshness=fresh" in status_line
    assert "updated_age_s=" in status_line
    assert f"report={report_path}" in status_line

    snapshot = format_progress_snapshot(written, report_path=Path(report_path))
    assert snapshot["migration_status"] == "migrated"
    assert snapshot["phase"] == "done"
    assert snapshot["progress"] == {"pct": 100.0, "checked": 2, "total": 2, "remaining": 0}
    assert snapshot["current_item"] == {"index": 2, "preview": "-"}
    assert snapshot["last_completed"] == {"index": 2, "preview": "legacy-memory-b"}
    assert snapshot["migration_counts"]["reembedded"] == 2
    assert snapshot["timing"]["freshness"] == "fresh"
    assert snapshot["timing"]["updated_age_s"] is not None
    assert snapshot["status_line"] == status_line
    assert snapshot["report_path"] == str(report_path)

    markdown = format_markdown_report(written, report_path=Path(report_path))
    assert "# DML Ollama Live-Store Migration Status" in markdown
    assert "- status_line: `migration_status=migrated | phase=done | progress=100.00%" in markdown
    assert "- freshness: `fresh` (updated_age_s=" in markdown
    assert "- status: `migrated`" in markdown
    assert "- progress: `100.00% (2/2, remaining=0)`" in markdown

    markdown_path = tmp_path / "migration-status.md"
    write_markdown_report(written, report_path=Path(report_path), output_path=markdown_path)
    assert markdown_path.exists()
    assert "Generated from the durable live-store migration artifact" in markdown_path.read_text(encoding="utf-8")

    snapshot_path = tmp_path / "migration-snapshot.json"
    write_progress_snapshot(written, report_path=Path(report_path), output_path=snapshot_path)
    assert snapshot_path.exists()
    snapshot_written = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_written == snapshot


def test_personality_matrix_mixed_positive_preference_with_avoid_clause_stays_positive(tmp_path) -> None:
    graph_path = tmp_path / "dpm-graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "preference_graph_path": str(graph_path),
                "relationship_id": "relationship:test",
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    adapter.record_personality_preference(
        "Mark prefers Citizen Snips to sound warm and human; avoid rigid mechanical writing.",
        source_id="turn:mixed",
        explicit=True,
    )

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    node = graph["nodes"][0]
    assert node["polarity"] == "prefer_high"

    overlay = adapter.personality_overlay(prompt="voice")
    assert overlay is not None
    rendered = overlay["overlay"]["rendered_text"]
    assert "Mark prefers Citizen Snips to sound warm and human" in rendered
    assert "restrained" not in rendered


def test_personality_matrix_prefers_user_preference_note_over_repair_provenance(tmp_path) -> None:
    graph_path = tmp_path / "preference-graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "dpm.preference-graph.v1",
                "graph_id": "preference-graph:relationship:test",
                "subject_id": "relationship:test",
                "generated_at": "2026-06-29T00:00:00Z",
                "nodes": [
                    {
                        "id": "pref.voice",
                        "kind": "interaction_style",
                        "label": "Voice",
                        "scope": "relationship",
                        "state": "active",
                        "weight": 0.9,
                        "confidence": 0.9,
                        "polarity": "prefer_high",
                        "value": {"target": 0.9},
                        "updated_at": "2026-06-29T00:00:00Z",
                        "provenance": [
                            {
                                "type": "current_turn_preference",
                                "source_id": "turn:user",
                                "observed_at": "2026-06-29T00:00:00Z",
                                "note": "Mark prefers Citizen Snips to sound warm and personable.",
                            },
                            {
                                "type": "polarity_repair",
                                "source_id": "repair",
                                "observed_at": "2026-06-29T00:01:00Z",
                                "note": "Repair implementation note should not become persona text.",
                            },
                        ],
                    }
                ],
                "edges": [],
                "audit": {},
            }
        ),
        encoding="utf-8",
    )
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-read",
                "preference_graph_path": str(graph_path),
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    overlay = adapter.personality_overlay(prompt="voice")
    assert overlay is not None
    rendered = overlay["overlay"]["rendered_text"]
    assert "warm and personable" in rendered
    assert "Repair implementation note" not in rendered


def test_personality_evolution_records_interaction_and_renders_hard_laws(tmp_path) -> None:
    evolution_path = tmp_path / "storage" / "dpm_evolution_graph.json"
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "storage_dir": str(tmp_path / "storage"),
            "persistence": {"enable": False},
            "metrics_enabled": False,
            "dpm": {
                "enable": True,
                "mode": "active-write",
                "evolution_graph_path": str(evolution_path),
                "relationship_id": "relationship:test",
                "max_overlay_chars": 700,
                "token_budget": 180,
            },
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )

    result = adapter.record_personality_interaction(
        "This is too mechanical; be warmer and more personable.",
        "Understood — I will loosen up and keep the mechanics in the background.",
        source_id="turn:evolution",
        meta={"task_type": "creative_personality", "feedback_dimension": "mechanicality", "feedback_valence": -0.8},
    )

    assert result is not None
    assert result["status"] == "recorded"
    graph = json.loads(evolution_path.read_text(encoding="utf-8"))
    assert graph["schema_version"] == "dpm.evolution-graph.v1"
    assert graph["state_traces"]
    assert graph["traits"]["mechanicality"]["fast"] < 0.24
    assert graph["traits"]["warmth"]["fast"] > 0.62

    overlay = adapter.personality_overlay(prompt="rewrite this script with personality")
    assert overlay is not None
    rendered = overlay["overlay"]["rendered_text"]
    assert "Current tendency:" in rendered
    assert "Context adaptation:" in rendered
    assert "Explicit current-turn user instructions override personality tendencies" in rendered
    assert overlay["effective_constraints"]["hard_laws_immutable"] is True
