#!/usr/bin/env python3
"""Simple polling worker for low-latency DML queue ingestion."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _queued_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("status", "queued") == "queued":
            count += 1
    return count


def run_once(args: argparse.Namespace) -> dict:
    queued_before = _queued_count(args.queue_path)
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(
        {
            "OPENCLAW_HOME": str(args.openclaw_home),
            "OPENCLAW_WORKSPACE": str(args.openclaw_workspace),
            "DML_STORE": str(args.storage_dir),
        }
    )
    proc = subprocess.run(
        [str(args.worker_script)],
        text=True,
        capture_output=True,
        check=False,
        timeout=args.timeout_s,
        env=env,
    )
    return {
        "ts": _utc_now(),
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "queued_before": queued_before,
        "queued_after": _queued_count(args.queue_path),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "stdout": proc.stdout.strip()[-1200:],
        "stderr": proc.stderr.strip()[-1200:],
    }


def build_parser() -> argparse.ArgumentParser:
    home = Path(os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))).expanduser()
    workspace = Path(os.environ.get("OPENCLAW_WORKSPACE", str(home / "workspace"))).expanduser()
    parser = argparse.ArgumentParser(description="Run the DML background ingest worker")
    parser.add_argument("--openclaw-home", type=Path, default=home)
    parser.add_argument("--openclaw-workspace", type=Path, default=workspace)
    parser.add_argument("--queue-path", type=Path, default=workspace / "out" / "dml_ingest_queue.jsonl")
    parser.add_argument("--storage-dir", type=Path, default=Path(os.environ.get("DML_STORE", str(home / "dml-store"))))
    parser.add_argument("--worker-script", type=Path, default=workspace / "skills" / "daystrom-dml" / "scripts" / "run_dml_ingest_worker.sh")
    parser.add_argument("--interval-s", type=float, default=15.0)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while True:
        try:
            print(json.dumps(run_once(args), sort_keys=True), flush=True)
        except subprocess.TimeoutExpired as exc:
            print(json.dumps({"ts": _utc_now(), "status": "timeout", "timeout_s": args.timeout_s, "cmd": exc.cmd}), flush=True)
        if args.once or STOP:
            return 0
        time.sleep(max(1.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
