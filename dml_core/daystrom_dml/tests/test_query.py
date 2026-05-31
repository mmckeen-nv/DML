from __future__ import annotations

import tempfile
from pathlib import Path

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.embeddings import RandomEmbedder
from daystrom_dml.summarizer import DummySummarizer
from daystrom_dml.gpt_runner import GPTRunner


def make_adapter():
    storage_dir = Path(tempfile.mkdtemp(prefix="dml-test-storage-"))
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 100,
            "top_k": 4,
            "literal_context": 1,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


def test_local_completion_backend_answers_from_user_prompt_not_raw_context():
    runner = GPTRunner("dummy", backend="auto")
    output = runner.generate(
        "=== Daystrom Memory Lattice ===\n"
        "- L0 (f=1.00): Continuity memory preserves decisions and tests.\n"
        "=== User Prompt ===\n"
        "What does continuity preserve?",
        max_new_tokens=64,
    )

    assert "=== Daystrom Memory Lattice ===" not in output
    assert "=== User Prompt ===" not in output
    assert "retrieved context" not in output.lower()
    assert "Continuity memory preserves decisions and tests." in output


def test_local_completion_backend_uses_private_grounding_notes():
    runner = GPTRunner("dummy", backend="auto")
    output = runner.generate(
        "Answer the user directly and naturally. "
        "Treat the notes below as private grounding, not as something to announce.\n\n"
        "=== Private Grounding Notes ===\n"
        "- L0 (f=1.00): Quartermaster Ada Sol controls inventory with lockbox ORCHID-17.\n"
        "=== User Prompt ===\n"
        "Who controls inventory?",
        max_new_tokens=64,
    )

    assert "Answer the user directly" not in output
    assert "private grounding" not in output
    assert "Ada Sol" in output
    assert "ORCHID-17" in output


def test_query_database_literal_mode_auto():
    adapter = make_adapter()
    adapter.ingest(
        "# User service documentation",
        meta={"doc_path": "docs/user_api.md", "chunk_index": 0},
    )
    adapter.ingest(
        (
            "def fetchUserProfile(user_id: str) -> dict:\n"
            '    """Fetches a single user profile."""\n'
            '    return client.get(f"/users/{user_id}")'
        ),
        meta={"doc_path": "docs/user_api.md", "chunk_index": 1},
    )
    adapter.ingest(
        "# Related helper",
        meta={"doc_path": "docs/user_api.md", "chunk_index": 2},
    )

    result = adapter.query_database("Show API call to fetchUserProfile")
    assert result["mode"] == "literal"
    assert "fetchUserProfile" in result["context"]
    assert "docs/user_api.md" in result["source_docs"]
    assert result["tokens"] > 0
    assert result["latency_ms"] >= 0


def test_query_database_keeps_literal_hits_when_persistent_rag_has_weak_matches(tmp_path):
    adapter = DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 100,
            "top_k": 4,
            "dml_top_k": 4,
            "literal_context": 1,
            "storage_dir": str(tmp_path),
            "persistence": {"enable": False},
            "rag_store": {
                "enable": True,
                "path": "rag_index.faiss",
                "meta_path": "rag_meta.json",
                "backend": "faiss",
                "dim": 48,
            },
            "similarity_threshold": 0.0,
        },
        embedder=RandomEmbedder(dim=48),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )
    adapter.ingest("Old irrelevant memory about starship maintenance.", meta={"source": "old"})
    adapter.ingest(
        "The unique smoke phrase is DML_SMOKE_literal_priority_blue_circuit.",
        meta={"source": "smoke-test"},
    )

    result = adapter.query_database("What is the unique smoke phrase?", mode="hybrid")

    assert "DML_SMOKE_literal_priority_blue_circuit" in result["context"]
    assert result["context"].split("\n\n", 1)[0].find("DML_SMOKE_literal_priority_blue_circuit") != -1
    assert "smoke-test" in result["source_docs"]

    report = adapter.retrieval_report("What is the unique smoke phrase?")
    assert report["entries"]
    assert "DML_SMOKE_literal_priority_blue_circuit" in report["entries"][0]["summary"]


def test_query_database_semantic_summary():
    adapter = make_adapter()
    adapter.ingest(
        "January average temperature was 5C while February averaged 6C.",
        meta={"doc_path": "reports/weather_2023.txt", "chunk_index": 0},
    )
    adapter.ingest(
        "Summer months peaked at 30C on average, cooling to 10C in autumn.",
        meta={"doc_path": "reports/weather_2023.txt", "chunk_index": 1},
    )

    result = adapter.query_database("Summarize average temperatures from reports last year")
    assert result["mode"] == "semantic"
    assert "temperature" in result["context"].lower()
    assert "reports/weather_2023.txt" in result["source_docs"]
    assert result["tokens"] > 0
