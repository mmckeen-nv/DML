"""
Optimized synthetic benchmarks for DML vs RAG.
Lightweight performance comparison with efficient ingestion and retrieval.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, List

from daystrom_dml.dml_adapter import DMLAdapter


DEFAULT_MODES = ("semantic", "literal", "hybrid", "agent")
VOCAB = [
    "quantum", "warp", "lattice", "plasma", "neutrino", "protocol", "diagnostic",
    "hyperdrive", "tensor", "fusion", "relay", "synthesis", "analysis", "resonance",
    "photon", "entanglement", "spacetime", "chronometer", "flux", "capacitor",
    "torus", "singularity", "dilithium", "antimatter", "warp drive", "subspace",
    "quark", "gluon", "boson", "lepton", "neutron", "proton", "atom", "molecule",
    "cell", "organism", "species", "ecosystem", "biosphere", "atmosphere", "gravity",
    "magnetism", "electricity", "radiation", "energy", "mass", "velocity", "acceleration",
    "momentum", "inertia", "force", "work", "power", "thermodynamics", "entropy",
]


def _synthetic_document(idx: int, rng: random.Random) -> str:
    """Generate synthetic document with concept references."""
    words = [rng.choice(VOCAB) for _ in range(60)]
    words[idx % len(words)] = f"concept_{idx}"
    return " ".join(words)


def _synthetic_query(idx: int) -> str:
    """Generate synthetic query asking for concept explanation."""
    return f"Explain concept_{idx}"


def _ingest_corpus(adapter: DMLAdapter, count: int, rng: random.Random) -> dict[str, Any]:
    """Ingest synthetic corpus into DML with minimal overhead."""
    ingest_start = time.perf_counter()
    for idx in range(count):
        text = _synthetic_document(idx, rng)
        adapter.ingest(
            text,
            meta={
                "doc_path": f"doc_{idx}.txt",
                "tenant_id": "bench",
            },
        )
    ingest_duration = (time.perf_counter() - ingest_start) * 1000.0
    return {
        "docs_ingested": count,
        "ingest_duration_ms": ingest_duration,
        "ingest_cost_tokens": count * 60,
    }


def _rag_ingest_docs(count: int, rng: random.Random, storage_dir: Path) -> dict[str, Any]:
    """Simulate RAG ingestion with minimal overhead."""
    vector_store_path = storage_dir / "vector_store"
    vector_store_path.mkdir(parents=True, exist_ok=True)

    ingest_start = time.perf_counter()
    for idx in range(count):
        text = _synthetic_document(idx, rng)
        doc_id = f"doc_{idx}"
        metadata = {"doc_path": f"doc_{idx}.txt", "tenant_id": "bench"}

        # Lightweight embedding (128-dim for speed)
        words = text.split()[:20]
        embedding = [random.random() for _ in range(128)]

        (vector_store_path / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }))
    ingest_duration = (time.perf_counter() - ingest_start) * 1000.0

    return {
        "docs_ingested": count,
        "ingest_duration_ms": ingest_duration,
        "ingest_cost_tokens": count * 60,
        "vector_store_size_bytes": vector_store_path.stat().st_size,
    }


def _run_dml_mode(
    adapter: "DMLAdapter",
    prompts: List[str],
    mode: str,
) -> dict[str, Any]:
    """Run DML retrieval in specified mode."""
    latencies: List[float] = []
    tokens: List[int] = []

    for prompt in prompts:
        retrieval_start = time.perf_counter()
        if mode == "agent":
            report = adapter.retrieve_context(
                prompt,
                tenant_id="bench",
                client_id="bench",
                session_id=None,
                instance_id=None,
                kinds=None,
                top_k=None,
            )
            retrieval_duration = (time.perf_counter() - retrieval_start) * 1000.0
            context_tokens = int(report.get("context_tokens", 0))
        else:
            report = adapter.query_database(prompt, mode=mode)
            retrieval_duration = (time.perf_counter() - retrieval_start) * 1000.0
            context_tokens = int(report.get("tokens", 0))

        retrieval_latencies.append(retrieval_duration)
        tokens.append(context_tokens)

    avg_latency = statistics.mean(latencies) if latencies else 0
    total_tokens = sum(tokens) + sum(len(p.split()) for p in prompts)
    avg_context_tokens = statistics.mean(tokens) if tokens else 0

    return {
        "mode": mode,
        "type": "dml",
        "avg_latency_ms": round(avg_latency, 3),
        "avg_retrieval_latency_ms": round(avg_latency, 3),
        "avg_tokens": round(total_tokens / len(prompts) if prompts else 0, 2),
        "total_tokens": total_tokens,
        "context_tokens": round(avg_context_tokens, 2),
    }


def _run_rag_mode(
    storage_dir: Path,
    prompts: List[str],
    mode: str,
) -> dict[str, Any]:
    """Run RAG retrieval with lightweight simulation."""
    latencies: List[float] = []
    tokens: List[int] = []

    vector_store_path = storage_dir / "vector_store"

    for prompt in prompts:
        retrieval_start = time.perf_counter()

        # Simple keyword matching (no real embeddings)
        results = []
        query_words = set(prompt.split())

        for doc_file in vector_store_path.glob("*.json"):
            data = json.loads(doc_file.read_text())
            doc_text = data["text"]
            doc_words = set(doc_text.split()[:20])

            similarity = len(query_words & doc_words) / max(len(query_words), 1)
            if similarity > 0.3:
                results.append((similarity, doc_text))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = [text for score, text in results[:2]]

        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000.0
        retrieval_latencies.append(retrieval_duration)

        context_text = "\n".join(top_results) if top_results else "No relevant documents"
        context_tokens = len(context_text.split())

        tokens.append(context_tokens)

    avg_latency = statistics.mean(latencies) if latencies else 0
    total_tokens = sum(tokens) + sum(len(p.split()) for p in prompts)
    avg_context_tokens = statistics.mean(tokens) if tokens else 0

    return {
        "mode": mode,
        "type": "rag",
        "avg_latency_ms": round(avg_latency, 3),
        "avg_retrieval_latency_ms": round(avg_latency, 3),
        "avg_tokens": round(total_tokens / len(prompts) if prompts else 0, 2),
        "total_tokens": total_tokens,
        "context_tokens": round(avg_context_tokens, 2),
    }


def run_benchmark(
    *,
    corpus_sizes: List[int] = [100, 500, 1000],
    query_counts: List[int] = [10, 50],
    seed: int = 42,
) -> List[dict[str, Any]]:
    """Run optimized benchmark suite."""
    rng = random.Random(seed)
    results: List[dict[str, Any]] = []

    print(f"Running benchmarks...")
    print(f"Configuration: corpus_sizes={corpus_sizes}, query_counts={query_counts}")
    print(f"Seed: {seed}\n")

    for corpus_size in corpus_sizes:
        for query_count in query_counts:
            prompts = [_synthetic_query(idx) for idx in range(query_count)]

            print(f"{'='*70}")
            print(f"Corpus: {corpus_size} docs, Queries: {query_count}")
            print(f"{'='*70}")

            with tempfile.TemporaryDirectory(prefix="dml-bench-opt-") as tmpdir:
                tmpdir_path = Path(tmpdir)

                # DML
                print(f"\n[DML]")
                adapter = DMLAdapter(
                    config_overrides={"storage_dir": tmpdir_path, "model_name": "dummy"},
                    start_aging_loop=False,
                )
                try:
                    ingest = _ingest_corpus(adapter, corpus_size, rng)
                    print(f"  Ingest: {ingest['ingest_duration_ms']:.1f}ms ({corpus_size} docs)")

                    for mode in DEFAULT_MODES:
                        result = _run_dml_mode(adapter, prompts, mode)
                        results.append(result)
                        print(f"  {mode:8s}: {result['avg_latency_ms']:.1f}ms, tokens={result['avg_tokens']}")

                    # RAG
                    print(f"\n[RAG]")
                    rag_ingest = _rag_ingest_docs(corpus_size, rng, tmpdir_path)
                    print(f"  Ingest: {rag_ingest['ingest_duration_ms']:.1f}ms ({corpus_size} docs)")

                    for mode in ["semantic", "literal", "hybrid"]:
                        result = _run_rag_mode(tmpdir_path, prompts, mode)
                        results.append(result)
                        print(f"  {mode:8s}: {result['avg_latency_ms']:.1f}ms, tokens={result['avg_tokens']}")

                finally:
                    try:
                        adapter.close()
                    except:
                        pass

    return results


def analyze_results(results: List[dict[str, Any]]) -> dict[str, Any]:
    """Analyze benchmark results."""
    analysis = {}

    for corpus_size in [100, 500, 1000]:
        analysis[corpus_size] = {"dml": {}, "rag": {}}

        for mode in DEFAULT_MODES + [("rag", "semantic"), ("rag", "literal"), ("rag", "hybrid")]:
            mode_key = mode[1] if mode[0] == "rag" else mode
            mode_type = mode[0]

            mode_results = [
                r for r in results
                if r.get("avg_tokens", 0) > 0 and
                r.get("mode") == mode_key and
                r.get("type") == mode_type
            ]

            if mode_results:
                total_tokens = sum(r["total_tokens"] for r in mode_results)
                avg_latency = statistics.mean([r["avg_latency_ms"] for r in mode_results])

                analysis[corpus_size][mode_type][mode_key] = {
                    "avg_tokens": round(total_tokens / len(mode_results), 2),
                    "avg_latency_ms": round(avg_latency, 3),
                    "context_tokens": round(statistics.mean([r["context_tokens"] for r in mode_results]), 2),
                    "runs": len(mode_results),
                }

    return analysis


def write_results(results: List[dict[str, Any]], analysis: dict[str, Any], path: Path) -> None:
    """Write results to files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = path.with_suffix(".csv")
    fieldnames = ["mode", "type", "avg_latency_ms", "avg_tokens", "total_tokens", "context_tokens"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # JSON
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps({"results": results, "analysis": analysis}, indent=2))

    # Summary
    summary_path = path.parent / f"benchmark_summary.txt"
    summary_path.write_text(f"""# DML vs RAG Benchmark Results

## Corpus Sizes: {list(analysis.keys())}

## Total Tokens per Mode

### DML
""")

    for corpus_size in analysis:
        for mode_name, metrics in analysis[corpus_size]["dml"].items():
            summary_path.write_text(
                f"**{corpus_size} docs** - {mode_name}: {metrics['avg_tokens']} tokens, "
                f"{metrics['avg_latency_ms']}ms latency\n"
            )

        summary_path.write_text(f"\n### RAG\n")

        for mode_name, metrics in analysis[corpus_size]["rag"].items():
            summary_path.write_text(
                f"**{corpus_size} docs** - {mode_name}: {metrics['avg_tokens']} tokens, "
                f"{metrics['avg_latency_ms']}ms latency\n"
            )

    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"Results: {path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DML vs RAG benchmark")
    parser.add_argument("--corpus-sizes", type=int, nargs="+", default=[100, 500, 1000],
                        help="Corpus sizes")
    parser.add_argument("--query-counts", type=int, nargs="+", default=[10, 50],
                        help="Query counts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("examples/bench/optimized_results"),
                        help="Output path")
    args = parser.parse_args()

    results = run_benchmark(
        corpus_sizes=args.corpus_sizes,
        query_counts=args.query_counts,
        seed=args.seed,
    )

    analysis = analyze_results(results)
    write_results(results, analysis, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()