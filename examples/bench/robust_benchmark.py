"""
Robust synthetic benchmarks for DML vs RAG.
Measures retrieval time, latency, token usage, and total costs.
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
from typing import Any, Iterable, List

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
    words[(idx + 1) % len(words)] = f"detailed_explanation_{idx}"
    return " ".join(words)


def _synthetic_query(idx: int) -> str:
    """Generate synthetic query asking for concept explanation."""
    return f"Explain concept_{idx} with historical context and technical details"


def _ingest_corpus(adapter: DMLAdapter, count: int, rng: random.Random) -> dict[str, Any]:
    """Ingest synthetic corpus into DML."""
    ingest_start = time.perf_counter()
    for idx in range(count):
        text = _synthetic_document(idx, rng)
        adapter.ingest(
            text,
            meta={
                "doc_path": f"synthetic/doc_{idx}.txt",
                "tenant_id": "bench",
                "client_id": "bench",
                "category": f"category_{idx % 5}",
            },
        )
    ingest_duration = (time.perf_counter() - ingest_start) * 1000.0
    return {
        "docs_ingested": count,
        "ingest_duration_ms": ingest_duration,
        "ingest_cost_tokens": count * 60,
    }


def _rag_ingest_docs(count: int, rng: random.Random, storage_dir: Path) -> dict[str, Any]:
    """Simulate RAG ingestion by creating vector store."""
    vector_store_path = storage_dir / "vector_store"
    vector_store_path.mkdir(parents=True, exist_ok=True)

    ingest_start = time.perf_counter()
    for idx in range(count):
        text = _synthetic_document(idx, rng)
        doc_id = f"doc_{idx}"
        metadata = {
            "doc_path": f"synthetic/doc_{idx}.txt",
            "tenant_id": "bench",
            "category": f"category_{idx % 5}",
        }
        # Simulate vector embedding
        embeddings = []
        words = text.split()
        for word in words[:30]:
            embeddings.append([random.random() for _ in range(1536)])

        (vector_store_path / f"{doc_id}.json").write_text(json.dumps({
            "id": doc_id,
            "text": text,
            "embedding": embeddings,
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
    prompts: Iterable[str],
    mode: str,
    run_id: int,
) -> dict[str, Any]:
    """Run DML retrieval in specified mode."""
    latencies: List[float] = []
    tokens: List[int] = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    outputs: List[dict[str, str]] = []
    sample_output: str | None = None

    for idx, prompt in enumerate(prompts):
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
            retrieval_latencies.append(retrieval_duration)
            context_tokens = int(report.get("context_tokens", 0))
            tokens.append(context_tokens)
            augmented = adapter._compose_prompt(prompt, report.get("raw_context", ""))
        else:
            report = adapter.query_database(prompt, mode=mode)
            retrieval_duration = (time.perf_counter() - retrieval_start) * 1000.0
            retrieval_latencies.append(retrieval_duration)
            tokens.append(int(report.get("tokens", 0)))
            augmented = adapter._compose_prompt(prompt, report.get("context", ""))

        gen_start = time.perf_counter()
        response = adapter.runner.generate(augmented)
        generation_duration = (time.perf_counter() - gen_start) * 1000.0
        generation_latencies.append(generation_duration)

        outputs.append({"prompt": prompt, "response": response})
        if sample_output is None:
            sample_output = response

    total_tokens = sum(tokens) + sum(len(p.split()) + len(r.split()) for p, r in outputs)

    return {
        "run_id": run_id,
        "mode": mode,
        "type": "dml",
        "avg_latency_ms": round(statistics.mean(retrieval_latencies + generation_latencies), 3) if (retrieval_latencies + generation_latencies) else 0,
        "avg_retrieval_latency_ms": round(statistics.mean(retrieval_latencies), 3) if retrieval_latencies else 0,
        "avg_generation_latency_ms": round(statistics.mean(generation_latencies), 3) if generation_latencies else 0,
        "avg_tokens": round(total_tokens / len(tokens) if tokens else 0, 2),
        "total_tokens": total_tokens,
        "context_tokens": round(sum(tokens) / len(tokens) if tokens else 0, 2),
    }


def _run_rag_mode(
    storage_dir: Path,
    prompts: Iterable[str],
    mode: str,
    run_id: int,
) -> dict[str, Any]:
    """Run RAG retrieval (simulated with vector store)."""
    latencies: List[float] = []
    tokens: List[int] = []
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    outputs: List[dict[str, str]] = []
    sample_output: str | None = None

    for idx, prompt in enumerate(prompts):
        retrieval_start = time.perf_counter()

        vector_store_path = storage_dir / "vector_store"
        results = []
        for doc_file in vector_store_path.glob("*.json"):
            data = json.loads(doc_file.read_text())
            doc_text = data["text"]
            query_words = set(prompt.split()[:10])
            doc_words = set(doc_text.split()[:30])

            similarity = len(query_words & doc_words) / max(len(query_words), 1)
            if similarity > 0.1:
                results.append((similarity, doc_text))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = [text for score, text in results[:3]]

        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000.0
        retrieval_latencies.append(retrieval_duration)

        context = "\n\n".join([f"Source: {idx}"] + top_results)
        augmented = f"Context:\n{context}\n\n{prompt}"

        gen_start = time.perf_counter()
        response = f"Based on retrieved documents, here's the answer: {top_results[0][:100] if top_results else 'No relevant documents found.'}"
        generation_duration = (time.perf_counter() - gen_start) * 1000.0
        generation_latencies.append(generation_duration)

        outputs.append({"prompt": prompt, "response": response})
        if sample_output is None:
            sample_output = response

        tokens.append(len(prompt.split()) + len(response.split()))

    total_tokens = sum(tokens)
    context_tokens = sum(len(text.split()) for text in top_results[:3] if top_results) if top_results else 0

    return {
        "run_id": run_id,
        "mode": mode,
        "type": "rag",
        "avg_latency_ms": round(statistics.mean(retrieval_latencies + generation_latencies), 3) if (retrieval_latencies + generation_latencies) else 0,
        "avg_retrieval_latency_ms": round(statistics.mean(retrieval_latencies), 3) if retrieval_latencies else 0,
        "avg_generation_latency_ms": round(statistics.mean(generation_latencies), 3) if generation_latencies else 0,
        "avg_tokens": round(total_tokens / len(tokens) if tokens else 0, 2),
        "total_tokens": total_tokens,
        "context_tokens": round(context_tokens, 2),
    }


def run_benchmark(
    *,
    corpus_sizes: List[int] = [100, 1000],
    query_counts: List[int] = [10, 50],
    seed: int = 42,
    runs: int = 2,
) -> List[dict[str, Any]]:
    """Run comprehensive benchmark suite."""
    rng = random.Random(seed)

    results: List[dict[str, Any]] = []
    total_runs = 0

    for corpus_size in corpus_sizes:
        for query_count in query_counts:
            for run_id in range(runs):
                total_runs += 1
                prompts = [_synthetic_query(idx) for idx in range(query_count)]

                print(f"\n{'='*70}")
                print(f"Run {total_runs}: corpus_size={corpus_size}, queries={query_count}")
                print(f"{'='*70}")

                with tempfile.TemporaryDirectory(prefix="dml-bench-robust-") as tmpdir:
                    tmpdir_path = Path(tmpdir)

                    # Ingest for DML
                    print(f"\n[DML] Ingesting {corpus_size} documents...")
                    adapter = DMLAdapter(
                        config_overrides={"storage_dir": tmpdir_path, "model_name": "dummy", "embedding_model": None},
                        start_aging_loop=False,
                    )
                    try:
                        dml_ingest = _ingest_corpus(adapter, corpus_size, rng)
                        print(f"    Ingest duration: {dml_ingest['ingest_duration_ms']:.2f}ms")

                        # Ingest for RAG
                        print(f"[RAG] Ingesting {corpus_size} documents...")
                        rag_ingest = _rag_ingest_docs(corpus_size, rng, tmpdir_path)
                        print(f"    Ingest duration: {rag_ingest['ingest_duration_ms']:.2f}ms")

                        # Run DML modes
                        print(f"\n[DML] Running benchmarks...")
                        for mode in DEFAULT_MODES:
                            print(f"  - {mode.upper()}")
                            dml_result = _run_dml_mode(adapter, prompts, mode, run_id)
                            results.append(dml_result)

                        # Run RAG modes
                        print(f"\n[RAG] Running benchmarks...")
                        for mode in ["semantic", "literal", "hybrid"]:
                            print(f"  - {mode.upper()}")
                            rag_result = _run_rag_mode(tmpdir_path, prompts, mode, run_id)
                            results.append(rag_result)

                    finally:
                        try:
                            adapter.close()
                        except:
                            pass

    return results


def analyze_results(results: List[dict[str, Any]]) -> dict[str, Any]:
    """Analyze and aggregate benchmark results."""
    analysis = {}

    for corpus_size in [100, 1000]:
        analysis[corpus_size] = {}
        for mode_type in ["rag", "dml"]:
            analysis[corpus_size][mode_type] = {}

            for mode in DEFAULT_MODES + [("rag", "semantic"), ("rag", "literal"), ("rag", "hybrid")]:
                mode_key = mode[1] if mode[0] == "rag" else mode
                mode_type_key = mode[0]

                mode_results = [
                    r for r in results
                    if r.get("avg_tokens", 0) > 0 and
                    r.get("mode") == mode_key and
                    r.get("type") == mode_type_key
                ]

                if mode_results:
                    total_tokens = sum(r["total_tokens"] for r in mode_results)
                    avg_latency = statistics.mean([r["avg_latency_ms"] for r in mode_results])

                    analysis[corpus_size][mode_type_key][mode_key] = {
                        "avg_tokens": round(total_tokens / len(mode_results), 2),
                        "avg_latency_ms": round(avg_latency, 3),
                        "context_tokens": round(statistics.mean([r["context_tokens"] for r in mode_results]), 2),
                        "count": len(mode_results),
                    }

    return analysis


def write_results(results: List[dict[str, Any]], analysis: dict[str, Any], path: Path) -> None:
    """Write results to CSV and JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = path.with_suffix(".csv")
    fieldnames = [
        "run_id", "mode", "type",
        "avg_latency_ms", "avg_retrieval_latency_ms", "avg_generation_latency_ms",
        "avg_tokens", "total_tokens", "context_tokens",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_for_csv = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(row_for_csv)

    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "raw_results": results,
        "analysis": analysis,
    }, indent=2))

    summary_path = path.parent / f"benchmark_summary_{path.stem}.txt"
    summary_path.write_text(f"""# Benchmark Summary

## Corpus Sizes: {list(analysis.keys())}

## Performance Comparison

### Total Token Usage (RAG vs DML)
""")

    for corpus_size in analysis:
        summary_path.write_text(f"\n## Corpus: {corpus_size} documents\n\n")
        for mode_type in ["rag", "dml"]:
            if mode_type in analysis[corpus_size]:
                summary_path.write_text(f"\n### {mode_type.upper()}\n\n")
                for mode_name, metrics in analysis[corpus_size][mode_type].items():
                    summary_path.write_text(
                        f"- {mode_name:12s}: tokens={metrics['avg_tokens']:8.2f}, "
                        f"latency={metrics['avg_latency_ms']:7.3f}ms, "
                        f"context={metrics['context_tokens']:8.2f}, "
                        f"runs={metrics['count']}\n"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust DML vs RAG benchmark")
    parser.add_argument("--corpus-sizes", type=int, nargs="+", default=[100, 1000],
                        help="Corpus sizes to test")
    parser.add_argument("--query-counts", type=int, nargs="+", default=[10, 50],
                        help="Number of queries per test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per configuration")
    parser.add_argument("--output", type=Path, default=Path("examples/bench/robust_results"),
                        help="Output path prefix")
    args = parser.parse_args()

    print("Running robust benchmarks...")
    print(f"Configuration: corpus_sizes={args.corpus_sizes}, query_counts={args.query_counts}")
    print(f"Seed: {args.seed}, Runs: {args.runs}")

    results = run_benchmark(
        corpus_sizes=args.corpus_sizes,
        query_counts=args.query_counts,
        seed=args.seed,
        runs=args.runs,
    )

    analysis = analyze_results(results)
    write_results(results, analysis, args.output)

    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"{'='*70}")
    print(f"Total runs: {len(results)}")
    print(f"Results saved to: {args.output}")
    print(f"Summary: {args.output}.txt")


if __name__ == "__main__":  # pragma: no cover
    main()