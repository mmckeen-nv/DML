"""Reproducible CPU benchmarks for persistence and scoped recall.

Run: python -m scripts.performance_benchmark --output results.json
Only synthetic data in a temporary directory is used.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from daystrom_dml.atomic_io import atomic_write_text
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.memory_store import MemoryItem, MemoryStore
from daystrom_dml.persistent_index import PersistentVectorIndex
from daystrom_dml.summarizer import DummySummarizer


def timed(call):
    started = time.perf_counter()
    value = call()
    return (time.perf_counter() - started) * 1000, value


def percentiles(values):
    return {name: round(float(np.percentile(values, percentile)), 3)
            for name, percentile in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99))}


def make_store(size, tenants, dimension=32):
    rng = np.random.default_rng(4107)
    vectors = rng.normal(size=(size, dimension)).astype(np.float32)
    store = MemoryStore(DummySummarizer(), beta_a=0.08, beta_r=0.2, eta=0.15, gamma=0.02,
                        kappa=0.5, tau_s=0.3, theta_merge=0.92, K=4, capacity=size + 1,
                        start_aging_loop=False)
    store._items = [MemoryItem(i, f"synthetic memory {i}", vectors[i], 1000.0, 0.0, 1.0, 0,
                               {"tenant_id": str(i % tenants), "kind": "note",
                                "lattice_row": 0, "lattice_col": 0, "lattice_layer": 0,
                                "lattice_neighbors": [i]}) for i in range(size)]
    store._lineage = {item.id: item for item in store._items}
    store._id = size
    store._invalidate_embedding_cache()
    return store


def recall_benchmark(size, tenants, queries, *, ann=False):
    store = make_store(size, tenants)
    vectors = np.random.default_rng(982).normal(size=(queries, 32)).astype(np.float32)
    original, optimized, expected = [], [], []
    for query in vectors:
        elapsed, items = timed(lambda: store.retrieve_filtered(query, tenant_id="0", top_k=8))
        original.append(elapsed)
        expected.append({item.id for item in items})
    cold, _ = timed(lambda: store.retrieve_filtered(vectors[0], tenant_id="0", strict_scope=True, top_k=8))
    for query, ids in zip(vectors, expected):
        elapsed, items = timed(lambda: store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8))
        assert {item.id for item in items} == ids, "Exact scoped recall changed"
        optimized.append(elapsed)
    result = {"records": size, "tenants": tenants, "queries": queries, "k": 8,
              "reference": percentiles(original), "cached_exact": percentiles(optimized),
              "cold_scope_build_ms": round(cold, 3), "exact_recall_at_k": 1.0,
              "p95_speedup": round(float(np.percentile(original, 95) / np.percentile(optimized, 95)), 2)}
    if ann and size >= 10000:
        try:
            import faiss
        except ImportError:
            result["ann"] = {"status": "unavailable"}
        else:
            faiss.omp_set_num_threads(1)
            store._ann_min_items = 1000
            build, _ = timed(lambda: store.retrieve_filtered(vectors[0], tenant_id="0", strict_scope=True, top_k=8))
            latencies, recalls = [], []
            for query, ids in zip(vectors, expected):
                elapsed, items = timed(lambda: store.retrieve_filtered(query, tenant_id="0", strict_scope=True, top_k=8))
                latencies.append(elapsed)
                recalls.append(len(ids.intersection(item.id for item in items)) / max(1, len(ids)))
            result["ann"] = {**percentiles(latencies), "build_ms": round(build, 3),
                             "recall_at_8": round(float(np.mean(recalls)), 4),
                             "index_count": len(store._ann_indexes)}
    store.close()
    return result


def insertion_benchmark(directory, count):
    vectors = [np.ones(16, dtype=np.float32)] * count
    payloads = [{"text": f"synthetic {i}"} for i in range(count)]
    results = {}
    for name, batch in (("individual", False), ("batch", True)):
        index = PersistentVectorIndex(directory / f"{name}-{count}.json")
        counts = {"commits": 0, "bytes_written": 0}
        flush = index._flush

        def counted_flush():
            flush()
            counts["commits"] += 1
            counts["bytes_written"] += index.path.stat().st_size

        index._flush = counted_flush
        if batch:
            elapsed, _ = timed(lambda: index.extend(vectors, payloads))
        else:
            elapsed, _ = timed(lambda: [index.add(vector, payload) for vector, payload in zip(vectors, payloads)])
        assert len(PersistentVectorIndex(index.path)._vectors) == count
        results[name] = {"elapsed_ms": round(elapsed, 3), **counts}
    return {"records": count, **results}


def journal_benchmark(directory, size=10000):
    state = {"items": [{"id": i, "text": f"memory {i}", "embedding": [0.1] * 16} for i in range(size)],
             "lineage": [], "next_id": size, "repair_queue": []}
    journal = JournalStateStore(directory / "journal.sqlite3", snapshot_interval=128)
    journal.save(state)
    snapshot_latencies, journal_latencies, changed = [], [], []
    for i in range(10):
        state["items"][i]["text"] += " updated"
        elapsed, _ = timed(lambda: atomic_write_text(directory / "full.json", json.dumps(state, indent=2)))
        snapshot_latencies.append(elapsed)
        elapsed, _ = timed(lambda: journal.save(state))
        journal_latencies.append(elapsed)
        changed.append(journal.last_changed_rows)
    assert JournalStateStore(journal.path).load() == state
    store = make_store(size, 64)
    old_copy, _ = timed(lambda: copy.deepcopy(store.export_state()))
    native_copy, _ = timed(store.snapshot_state)
    return {"records": size, "full_json": percentiles(snapshot_latencies),
            "journal": percentiles(journal_latencies), "changed_rows_per_commit": changed,
            "rollback_snapshot_ms": {"json_expansion": round(old_copy, 3), "native_arrays": round(native_copy, 3)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args(argv)
    if args.queries < 1 or any(size < 8 for size in args.sizes):
        parser.error("queries must be positive and sizes must be at least eight")
    result = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
              "scope": "synthetic CPU microbenchmarks; no external model calls", "recall": [], "insertion": []}
    with tempfile.TemporaryDirectory(prefix="dml-perf-") as temporary:
        directory = Path(temporary)
        for count in (1, 100, 1000):
            result["insertion"].append(insertion_benchmark(directory, count))
            print(f"insertions: {count}", flush=True)
        for size in args.sizes:
            for tenants in (1, 64):
                result["recall"].append(recall_benchmark(size, tenants, args.queries, ann=tenants == 1))
                print(f"recall: {size} records, {tenants} tenants", flush=True)
        result["journal"] = journal_benchmark(directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
