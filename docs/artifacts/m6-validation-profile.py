"""Reproduce a single-client validation-cost diagnostic; not a capacity gate.

Run with the repository's test environment, for example:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/artifacts/m6-validation-profile.py --output scan.json

Use --source-root to measure an isolated checkout of a different revision with
the same recipe. Output contains relative source hashes and structured profiler
counts, never temporary database paths. All state is synthetic and disposable.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch


def run(source_root):
    # Test campaign/output variables must not enter the strict profile config.
    for key in list(os.environ):
        if key.startswith("DML_"):
            del os.environ[key]
    sys.path[:0] = [str(source_root / "dml_core"), str(source_root / "dml_core/tests")]
    from profile_concurrency_fixture import fixed_clock, prepare_history
    from profile_crash_fixture import SCOPE, make_adapter

    observations = []
    with TemporaryDirectory(prefix="dml-validation-profile-") as temporary, fixed_clock():
        directory = Path(temporary) / "authority"
        prepare_history(directory, 3)  # Seven committed, independently checked seed records.
        adapter = make_adapter(directory, 3)
        peer = None
        try:
            for index in range(30):
                adapter.ingest_memory_receipted(
                    f"Synthetic profile record {index}", **SCOPE,
                    idempotency_key=f"profile-record-{index}",
                    meta={"source_trust": "trusted"},
                )
            scans = 0
            original = adapter._journal._read_snapshot

            def observe(connection):
                nonlocal scans
                scans += 1
                return original(connection)

            with patch.object(adapter._journal, "_read_snapshot", observe):
                for kind in ("unchanged", "unchanged", "unchanged", "changed"):
                    if kind == "changed":
                        peer = make_adapter(directory, 3)
                        peer.ingest_memory_receipted(
                            "Synthetic peer record", **SCOPE, idempotency_key="profile-peer",
                            meta={"source_trust": "trusted"},
                        )
                    scans = 0
                    started = monotonic()
                    adapter.retrieve_context("Concurrent facts", **SCOPE, top_k=10)
                    observations.append({"kind": kind, "scans": scans,
                                         "seconds": monotonic() - started})
                profiler = cProfile.Profile()
                with profiler:
                    for _ in range(25):
                        adapter._journal.read_snapshot()
                revision = adapter._journal.revision
        finally:
            if peer is not None:
                peer.close()
            adapter.close()

    selected = []
    names = {"read_snapshot", "_read_snapshot", "_validate_outbox", "validate_outbox_event",
             "_validate_owned_outbox_event", "_encode", "_decode", "_checked"}
    stats = pstats.Stats(profiler)
    for (filename, _line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        path = Path(filename)
        if function in names and path.is_relative_to(source_root):
            selected.append({"path": path.relative_to(source_root).as_posix(), "function": function,
                             "primitive_calls": primitive, "calls": calls,
                             "own_seconds": own, "cumulative_seconds": cumulative})
    files = [*sorted((source_root / "dml_core/daystrom_dml").rglob("*.py")),
             source_root / "dml_core/tests/profile_concurrency_fixture.py",
             source_root / "dml_core/tests/profile_crash_fixture.py"]
    hashes = {path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    reviewed = {"dml_core/daystrom_dml/dml_adapter.py", "dml_core/daystrom_dml/journal.py",
                "dml_core/daystrom_dml/services/persistence.py",
                "dml_core/daystrom_dml/services/journal_outbox.py",
                "dml_core/tests/profile_concurrency_fixture.py",
                "dml_core/tests/profile_crash_fixture.py"}
    return {
        "format": "dml-validation-profile-diagnostic-v1",
        "purpose": "Single-client cost diagnostic; not a latency or capacity qualification",
        "recipe": {"schema": 3, "seed_commits": 7, "additional_ingests": 30,
                   "peer_ingests": 1, "unchanged_recalls": 3, "changed_recalls": 1,
                   "profiled_snapshot_reads": 25, "synthetic_embedder": True},
        "runtime": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                    "platform": platform.system(), "blas_threads": {
                        key: os.environ.get(key) for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "revision": revision, "observations": observations,
        "profile": {"total_calls": stats.total_calls, "total_seconds": stats.total_tt,
                    "selected_functions": sorted(selected, key=lambda item: (item["path"], item["function"]))},
        "source_file_sha256": {path: value for path, value in hashes.items() if path in reviewed},
        "source_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root.resolve())
    # Exclusive creation preserves earlier measurements and their provenance.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
