"""Independent SQLite/embeddings/recency baseline and offline comparison runner.

Synthetic retrieval only: no LLM task-success, TTFT, or production value claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import tempfile
import time

import numpy as np


class SQLiteBaseline:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, text TEXT, vector BLOB, scope TEXT, timestamp REAL)")
        self.connection.commit()

    def remember(self, memory_id, text, vector, *, scope, timestamp):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
            raise ValueError("invalid embedding")
        with self.connection:
            self.connection.execute("INSERT INTO memory VALUES (?,?,?,?,?)", (memory_id, text, vector.tobytes(), json.dumps(scope, sort_keys=True), timestamp))

    def recall(self, vector, *, scope, now, top_k=8, budget=600):
        rows = self.connection.execute("SELECT id,text,vector,timestamp FROM memory WHERE scope=? ORDER BY id", (json.dumps(scope, sort_keys=True),)).fetchall()
        ranked = []
        vector = np.asarray(vector, dtype=np.float32)
        for memory_id, text, raw, timestamp in rows:
            stored = np.frombuffer(raw, dtype=np.float32)
            if stored.shape != vector.shape:
                raise ValueError("embedding identity/dimension mismatch")
            denominator = np.linalg.norm(vector) * np.linalg.norm(stored)
            cosine = float(np.dot(vector, stored) / denominator) if denominator else 0.0
            recency = 1 / (1 + max(0., now - timestamp) / 3600)
            ranked.append((cosine + .15 * recency, memory_id, text))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        lines, ids = [], []
        for _, memory_id, text in ranked[:top_k]:
            # Independent deterministic prompt compaction, same character/token
            # estimator family as the legacy adapter; no DML services are reused.
            excerpt = text[:220]
            candidate = "\n".join(lines + [excerpt])
            if max(1, len(candidate) // 4) > budget:
                break
            lines.append(excerpt)
            ids.append(memory_id)
        return {"ids": ids, "context": "\n".join(lines)}

    def close(self):
        self.connection.close()


class OfflineEmbedder:
    def embed(self, text):
        result = np.zeros(64, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            key = hashlib.sha256(word.encode()).digest()
            result[int.from_bytes(key[:2], "little") % 64] += 1
        return result


def run_comparison(*, turns=1000):
    from daystrom_dml.dml_adapter import DMLAdapter
    if turns not in {1000, 10000, 100000}:
        raise ValueError("turns must be 1000, 10000 or 100000")
    with tempfile.TemporaryDirectory(prefix="dml-baseline-") as root:
        root = Path(root)
        embedder = OfflineEmbedder()
        baseline = SQLiteBaseline(root / "baseline.sqlite3")
        dml = DMLAdapter(config_overrides={"storage_dir": str(root / "dml"), "model_name": "dummy",
            "embedding_model": None, "persistence": {"journal": True, "enable": False},
            "rag_store": {"enable": False}, "theta_merge": 2., "token_budget": 600,
            "survival_ledger_enabled": False, "similarity_threshold": 0., "metrics_enabled": False,
            "dpm": {"enable": False, "include_in_context": False}}, embedder=embedder, start_aging_loop=False)
        now = 2000000000.
        scope = {"tenant_id": "benchmark", "client_id": None, "session_id": "episode", "instance_id": None}
        measurements = {name: {"retrieval_ms": [], "evidence_hits": 0} for name in ("sqlite_baseline", "dml")}
        try:
            # Fixed small corpus isolates retrieval cost across turn counts.
            # This is explicitly not a 100k-record ingestion/growth benchmark.
            for index in range(100):
                text = f"Record {index} has verified value value{index}."
                baseline.remember(index, text, embedder.embed(text), scope=scope, timestamp=now)
                item = dml.ingest_memory(text, **scope, meta={"source": "fixture", "no_merge": True})
                item.timestamp = now
            with dml.atomic_batch("pin-fixture-time"):
                pass
            for turn in range(turns):
                index = turn % 100
                query = f"Recall record {index} value{index}"
                start = time.perf_counter()
                result = baseline.recall(embedder.embed(query), scope=scope, now=now)
                measurements["sqlite_baseline"]["retrieval_ms"].append((time.perf_counter()-start)*1000)
                measurements["sqlite_baseline"]["evidence_hits"] += f"value{index}." in result["context"]
                start = time.perf_counter()
                result = dml.retrieve_context(query, **scope, as_of=now)
                measurements["dml"]["retrieval_ms"].append((time.perf_counter()-start)*1000)
                measurements["dml"]["evidence_hits"] += f"value{index}." in result["raw_context"]
            for value in measurements.values():
                samples = value.pop("retrieval_ms")
                value.update(retrieval_ms_p50=float(np.percentile(samples,50)), retrieval_ms_p95=float(np.percentile(samples,95)),
                             retrieval_ms_p99=float(np.percentile(samples,99)), evidence_recall=value["evidence_hits"]/turns)
            return {"schema_version": 1, "mode": "offline_retrieval_smoke", "turns": turns, "memory_records": 100,
                    "agent_outcomes": None, "value_gate": "not_evaluated", "measurements": measurements,
                    "configuration": {"embedding": "sha256-token-histogram-64-v1", "top_k": 8, "token_budget": 600,
                                      "seed": "deterministic-record-cycle-v1", "now": now, "python": platform.python_version(),
                                      "numpy": np.__version__, "sqlite": sqlite3.sqlite_version},
                    "limitations": ["No model or tools invoked", "Fixed 100-record corpus; no growth claim",
                                     "Compaction/framing differ; neither uses an LLM", "No task-success or production speedup claim"]}
        finally:
            dml.close()
            baseline.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=1000, choices=[1000,10000,100000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-events", type=Path, help="Summarize real terminal task JSONL instead of running a simulation")
    args = parser.parse_args(argv)
    if args.task_events:
        from daystrom_dml.services.evaluation import summarize_outcomes
        report = summarize_outcomes([json.loads(line) for line in args.task_events.read_text().splitlines() if line.strip()])
    else:
        report = run_comparison(turns=args.turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
