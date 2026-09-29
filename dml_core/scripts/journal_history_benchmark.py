"""Measure journal history-validation cost on an isolated, fixed-size lattice.

This CPU microbenchmark is evidence about history growth, not agent performance
or a production SLO. No external store or model is contacted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
import time

from daystrom_dml.journal import JournalStateStore


def run_history_benchmark(milestones=(1, 64, 256, 1000), samples=10) -> dict:
    if type(samples) is not int or samples < 2:
        raise ValueError("at least two samples are required")
    if not milestones or any(type(n) is not int or n < 1 for n in milestones) or sorted(set(milestones)) != list(milestones):
        raise ValueError("milestones must be strictly increasing positive integers")
    measurements = []
    with tempfile.TemporaryDirectory(prefix="dml-journal-history-") as directory:
        store = JournalStateStore(Path(directory) / "history.sqlite3")
        payload = {"items": [{"id": n, "text": f"evidence-{n}"} for n in range(10)], "lineage": []}
        current_revision = 0
        for milestone in milestones:
            while current_revision < milestone:
                payload["benchmark_step"] = current_revision + 1
                store.save(payload, expected_revision=current_revision, operation="history-benchmark")
                current_revision += 1
            durations = {}
            for name, operation in (("read", store.read_snapshot), ("stamp", store.stamp),
                                    ("no_change_save", lambda: store.save(payload, expected_revision=current_revision)),
                                    ("reopen", lambda: JournalStateStore(store.path))):
                values = []
                for _ in range(samples):
                    started = time.perf_counter_ns()
                    operation()
                    values.append((time.perf_counter_ns() - started) / 1_000_000)
                durations[name] = {"samples_ms": values, "p50_ms": statistics.median(values), "max_ms": max(values)}
            assert store.read_snapshot() == (current_revision, payload)
            measurements.append({"revision": current_revision, "live_records": 10, "operations": durations})
    return {"schema_version": "dml-journal-history-cost-v1", "mode": "isolated_cpu_microbenchmark",
            "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
            "platform": platform.system(), "measurements": measurements,
            "limitations": ["Fixed 10-record lattice; metadata changes grow decision history",
                            "No model, HTTP workers, network filesystem or power-loss testing",
                            "Single local run, warm samples; no production latency or scale guarantee"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_history_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
