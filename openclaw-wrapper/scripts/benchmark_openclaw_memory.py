#!/usr/bin/env python3
"""Benchmark DML as an OpenClaw memory substrate.

Tracks token savings + latency + weak-supervision relevance metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

WORKSPACE = Path("/Users/markmckeen/.openclaw/workspace")
DML_CORE = WORKSPACE / "dml" / "dml_core"
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (DML_CORE, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore
from tuning_utils import (  # type: ignore
    infer_intent_terms,
    noise_score,
    relevance_score,
    rewrite_query,
    should_keep_chunk,
    smart_chunks,
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def synthetic_memories(n: int) -> list[str]:
    base = [
        "Battlebot chassis update completed with hardened aluminum wedge front and dual side armor skirts.",
        "Weapon subsystem changed to hammer-saw hybrid mount with increased top clearance and reduced wobble.",
        "Wheel layout adjusted to inset dual-side geometry for improved turning control under weapon recoil.",
        "Material update: body switched to heat-treated aluminum, weapon to tool steel, wheels to high-grip composite.",
        "Validation step passed: anti-blob constraints satisfied with directional primitive stack and hard-surface budget.",
        "USD export supports native operator and converter fallback with GLB manifest handoff.",
    ]
    return [f"mem-{i:03d}: {base[i % len(base)]} Iteration {i}." for i in range(n)]


def file_memories(paths: Iterable[Path], *, max_chunks_per_file: int = 48, filter_noise: bool = True) -> tuple[list[str], dict]:
    out: list[str] = []
    stats = {"raw_chunks": 0, "kept_chunks": 0, "dropped_chunks": 0}
    for path in paths:
        try:
            raw = path.read_text(errors="ignore")
        except Exception:
            continue
        chunks = smart_chunks(raw)
        for idx, chunk in enumerate(chunks[:max_chunks_per_file]):
            stats["raw_chunks"] += 1
            if filter_noise and not should_keep_chunk(chunk):
                stats["dropped_chunks"] += 1
                continue
            stats["kept_chunks"] += 1
            out.append(f"[{path.name}#{idx}] {chunk}")
    return out, stats


def resolve_paths(*, input_files: list[str], input_globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for f in input_files:
        p = Path(f).expanduser()
        if p.exists() and p.is_file():
            out.append(p)
    for g in input_globs:
        out.extend([p for p in WORKSPACE.glob(g) if p.is_file()])
    dedup, seen = [], set()
    for p in sorted(out):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup


def ensure_gpu_embedder(adapter: DMLAdapter) -> None:
    embedder = getattr(adapter, "embedder", None)
    model = getattr(embedder, "_model", None)
    if model is None:
        raise RuntimeError("GPU benchmark requires SentenceTransformer embedder (fallback detected)")
    device = str(getattr(model, "device", getattr(model, "_target_device", ""))).lower()
    if "cuda" not in device:
        raise RuntimeError(f"GPU benchmark requires CUDA embedder; got {device or 'unknown'}")


def _extract_items(report: dict) -> list[dict]:
    items = report.get("items", [])
    return items if isinstance(items, list) else []


def _safe_text(item: dict) -> str:
    return str(item.get("text") or item.get("summary") or "")


def _dcg(scores: list[float]) -> float:
    total = 0.0
    for idx, score in enumerate(scores):
        denom = 1.0 if idx == 0 else math.log2(idx + 2)
        total += ((2**score) - 1) / max(1.0, denom)
    return total


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered))) - 1
    return ordered[min(rank, len(ordered) - 1)]


def _reset_storage_dir(storage_dir: str) -> str:
    """Reset benchmark index state for reproducible runs.

    Safety guard: only allow paths under workspace/data.
    """

    target = Path(storage_dir).expanduser().resolve()
    safe_root = (WORKSPACE / "data").resolve()
    if safe_root not in target.parents and target != safe_root:
        raise ValueError(f"Refusing to reset storage outside {safe_root}: {target}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def compute_delta(current: dict, previous: dict) -> dict:
    """Compute benchmark deltas (positive is better except latency/noise)."""

    def _num(obj: dict, key: str) -> float:
        try:
            return float(obj.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    return {
        "from_status": str(previous.get("status", "unknown")),
        "token_savings_pct_delta": round(_num(current, "avg_token_savings_pct") - _num(previous, "avg_token_savings_pct"), 2),
        "latency_ms_delta": round(_num(current, "avg_latency_ms") - _num(previous, "avg_latency_ms"), 2),
        "precision_at_k_delta": round(_num(current, "avg_precision_at_k") - _num(previous, "avg_precision_at_k"), 3),
        "ndcg_at_k_delta": round(_num(current, "avg_ndcg_at_k") - _num(previous, "avg_ndcg_at_k"), 3),
        "retrieval_noise_delta": round(_num(current, "avg_retrieval_noise_score") - _num(previous, "avg_retrieval_noise_score"), 3),
    }


def run_benchmark(*, config_path: str, storage_dir: str, memories: int, top_k: int, queries: list[str], corpus: list[str], corpus_label: str, query_expand: bool, fresh_storage: bool) -> dict:
    if fresh_storage:
        storage_dir = _reset_storage_dir(storage_dir)

    adapter = DMLAdapter(
        config_path=config_path,
        config_overrides={
            "storage_dir": storage_dir,
            "embedding_device": "cuda",
            "dml.agentic_mode.enabled": True,
        },
    )
    try:
        ensure_gpu_embedder(adapter)

        if not corpus:
            corpus = synthetic_memories(memories)
            corpus_label = "synthetic"

        for text in corpus:
            adapter.ingest(text, meta={"kind": "note", "source": f"benchmark:{corpus_label}"})

        baseline_context = "\n".join(corpus)
        baseline_tokens = estimate_tokens(baseline_context)

        per_query = []
        latencies, dml_tokens = [], []
        precision_scores, ndcg_scores, noise_scores = [], [], []

        for query in queries:
            expanded = rewrite_query(query) if query_expand else query
            intent = infer_intent_terms(query)

            t0 = time.perf_counter()
            report = adapter.retrieve_context(expanded, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000.0
            tokens = int(report.get("context_tokens", 0))
            items = _extract_items(report)
            ranked_texts = [_safe_text(i) for i in items[:top_k]]
            rel = [relevance_score(t, intent) for t in ranked_texts]

            prec_at_k = sum(1 for r in rel if r > 0.0) / max(1, len(rel))
            dcg = _dcg(rel)
            ideal = sorted(rel, reverse=True)
            idcg = _dcg(ideal) or 1.0
            ndcg = dcg / idcg
            avg_noise = statistics.mean([noise_score(t) for t in ranked_texts]) if ranked_texts else 0.0

            latencies.append(ms)
            dml_tokens.append(tokens)
            precision_scores.append(prec_at_k)
            ndcg_scores.append(ndcg)
            noise_scores.append(avg_noise)

            per_query.append(
                {
                    "query": query,
                    "expanded_query": expanded,
                    "latency_ms": round(ms, 2),
                    "dml_tokens": tokens,
                    "baseline_tokens": baseline_tokens,
                    "token_savings_pct": round((1 - (tokens / baseline_tokens)) * 100.0, 2),
                    "precision_at_k": round(prec_at_k, 3),
                    "ndcg_at_k": round(ndcg, 3),
                    "retrieval_noise_score": round(avg_noise, 3),
                }
            )

        avg_dml_tokens = statistics.mean(dml_tokens) if dml_tokens else 0.0
        savings_pct = (1 - (avg_dml_tokens / baseline_tokens)) * 100.0 if baseline_tokens else 0.0

        return {
            "status": "ok",
            "gpu_embedding": True,
            "corpus": corpus_label,
            "storage_dir": storage_dir,
            "fresh_storage": fresh_storage,
            "memories_ingested": len(corpus),
            "top_k": top_k,
            "baseline_tokens": baseline_tokens,
            "avg_dml_tokens": round(avg_dml_tokens, 2),
            "avg_token_savings_pct": round(savings_pct, 2),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(_p95(latencies), 2),
            "avg_precision_at_k": round(statistics.mean(precision_scores), 3) if precision_scores else 0.0,
            "avg_ndcg_at_k": round(statistics.mean(ndcg_scores), 3) if ndcg_scores else 0.0,
            "avg_retrieval_noise_score": round(statistics.mean(noise_scores), 3) if noise_scores else 0.0,
            "queries": per_query,
        }
    finally:
        adapter.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=str(WORKSPACE / "skills" / "daystrom-dml" / "config" / "dml_gpu_only.yaml"))
    ap.add_argument("--storage-dir", default=str(WORKSPACE / "data" / "dml-gpu-benchmark"))
    ap.add_argument("--memories", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--input-file", action="append", default=[])
    ap.add_argument("--input-glob", action="append", default=[])
    ap.add_argument("--max-chunks-per-file", type=int, default=48)
    ap.add_argument("--filter-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--query-expand", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--compare-with",
        default="",
        help="Optional path to previous benchmark JSON to compute deltas.",
    )
    ap.add_argument(
        "--fresh-storage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset benchmark storage directory before ingest (default: true)",
    )
    args = ap.parse_args()

    queries = args.query or [
        "How do I export USD and what is the fallback path?",
        "Summarize anti-blob chassis constraints and primitive stack rules.",
        "What wheel layout and weapon mount choices were made for the battlebot?",
    ]

    paths = resolve_paths(input_files=args.input_file, input_globs=args.input_glob)
    corpus, corpus_stats = file_memories(paths, max_chunks_per_file=max(1, args.max_chunks_per_file), filter_noise=args.filter_noise)
    corpus_label = "battlebot-files" if paths else "synthetic"

    result = run_benchmark(
        config_path=args.config_path,
        storage_dir=args.storage_dir,
        memories=max(10, args.memories),
        top_k=max(1, args.top_k),
        queries=queries,
        corpus=corpus,
        corpus_label=corpus_label,
        query_expand=args.query_expand,
        fresh_storage=args.fresh_storage,
    )
    result["input_files"] = [str(p) for p in paths]
    result["corpus_filtering"] = corpus_stats

    if args.compare_with:
        compare_path = Path(args.compare_with).expanduser()
        if compare_path.exists() and compare_path.is_file():
            previous = json.loads(compare_path.read_text())
            result["delta_vs_compare"] = compute_delta(result, previous)
            result["compared_to"] = str(compare_path)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
