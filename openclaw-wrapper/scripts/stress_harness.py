#!/usr/bin/env python3
"""Concurrency and isolation stress harness for the DML wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DML_SCRIPT = SCRIPT_DIR / "dml_memory.py"


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    elapsed_ms: float
    payload: dict[str, Any]
    stdout: str
    stderr: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _run_json_command(argv: list[str], *, timeout_s: float) -> CommandResult:
    started = time.perf_counter()
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_s, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        payload = {"status": "invalid_json", "stdout_preview": proc.stdout[:1200]}
    return CommandResult(argv, proc.returncode, round(elapsed_ms, 2), payload, proc.stdout, proc.stderr)


def _prefix(args: argparse.Namespace, storage_dir: Path) -> list[str]:
    argv = [
        args.python,
        str(args.dml_script),
        "--storage-dir",
        str(storage_dir),
        "--audit-actor",
        args.audit_actor,
    ]
    if args.config_path:
        argv.extend(["--config-path", args.config_path])
    if not args.require_gpu:
        argv.append("--no-require-gpu")
    argv.extend(["--lock-timeout-ms", str(args.lock_timeout_ms)])
    return argv


def _marker(run_id: str, index: int) -> str:
    return f"{run_id}-WRITER-{index:03d}"


def _ingest_command(args: argparse.Namespace, storage_dir: Path, index: int) -> list[str]:
    tenant = f"{args.run_id}-tenant-{index % max(1, args.tenants)}"
    session = f"session-{index % max(1, args.sessions)}"
    marker = _marker(args.run_id, index)
    meta = {
        "source": "stress_harness",
        "namespace": "concurrency",
        "stress_run_id": args.run_id,
        "writer_index": index,
    }
    text = f"{marker}: stress writer {index} durable memory for {tenant} {session}."
    return [
        *_prefix(args, storage_dir),
        "ingest",
        "--tenant-id",
        tenant,
        "--session-id",
        session,
        "--kind",
        "note",
        "--summary-policy",
        "skip",
        "--no-chunk",
        "--no-filter-noise",
        "--meta",
        json.dumps(meta, separators=(",", ":"), sort_keys=True),
        "--text",
        text,
    ]


def _run_concurrent_writes(
    args: argparse.Namespace,
    storage_dir: Path,
    runner: Callable[[list[str]], CommandResult],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(runner, _ingest_command(args, storage_dir, idx)): idx for idx in range(args.writes)}
        for future in as_completed(future_map):
            idx = future_map[future]
            result = future.result()
            results.append(
                {
                    "index": idx,
                    "marker": _marker(args.run_id, idx),
                    "returncode": result.returncode,
                    "status": result.payload.get("status"),
                    "chunks_ingested": result.payload.get("chunks_ingested"),
                    "latency_ms": result.elapsed_ms,
                    "stderr_preview": result.stderr[:800],
                }
            )
    results.sort(key=lambda item: item["index"])
    return results


def _retrieve_case(
    args: argparse.Namespace,
    storage_dir: Path,
    runner: Callable[[list[str]], CommandResult],
    *,
    tenant: str,
    session: str,
    expected: str,
    forbidden: list[str],
) -> dict[str, Any]:
    result = runner(
        [
            *_prefix(args, storage_dir),
            "retrieve",
            "--tenant-id",
            tenant,
            "--session-id",
            session,
            "--query",
            expected,
            "--top-k",
            str(args.top_k),
            "--ground-truth-policy",
            "never",
            "--no-reform-memory",
            "--no-query-expand",
        ]
    )
    blob = json.dumps(result.payload, sort_keys=True, default=str)
    expected_hit = expected in blob
    forbidden_hits = [marker for marker in forbidden if marker in blob]
    return {
        "tenant_id": tenant,
        "session_id": session,
        "expected": expected,
        "expected_hit": expected_hit,
        "forbidden_hits": forbidden_hits,
        "passed": result.returncode == 0 and expected_hit and not forbidden_hits,
        "latency_ms": result.elapsed_ms,
        "status": result.payload.get("status"),
    }


def _run_isolation_checks(
    args: argparse.Namespace,
    storage_dir: Path,
    runner: Callable[[list[str]], CommandResult],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if args.writes <= 0:
        return checks
    first = 0
    second = 1 if args.writes > 1 else 0
    first_tenant = f"{args.run_id}-tenant-{first % max(1, args.tenants)}"
    first_session = f"session-{first % max(1, args.sessions)}"
    second_marker = _marker(args.run_id, second)
    checks.append(
        _retrieve_case(
            args,
            storage_dir,
            runner,
            tenant=first_tenant,
            session=first_session,
            expected=_marker(args.run_id, first),
            forbidden=[second_marker] if second != first else [],
        )
    )
    if args.tenants > 1 and args.writes > 1:
        other_idx = next((idx for idx in range(args.writes) if idx % args.tenants != first % args.tenants), None)
        if other_idx is not None:
            checks.append(
                _retrieve_case(
                    args,
                    storage_dir,
                    runner,
                    tenant=first_tenant,
                    session=first_session,
                    expected=_marker(args.run_id, first),
                    forbidden=[_marker(args.run_id, other_idx)],
                )
            )
    return checks


def _run_store_checks(
    args: argparse.Namespace,
    storage_dir: Path,
    runner: Callable[[list[str]], CommandResult],
) -> dict[str, Any]:
    verify = runner([*_prefix(args, storage_dir), "verify"])
    audit = runner([*_prefix(args, storage_dir), "audit-tail", "--limit", str(max(args.writes + 5, 20))])
    state_path = storage_dir / "dml_state.jsonl"
    missing_markers = {_marker(args.run_id, idx) for idx in range(args.writes)}
    if state_path.exists():
        with state_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                for marker in tuple(missing_markers):
                    if marker in line:
                        missing_markers.remove(marker)
                if not missing_markers:
                    break
    return {
        "verify": {
            "passed": verify.returncode == 0 and verify.payload.get("status") == "ok",
            "status": verify.payload.get("status"),
            "loaded_count": verify.payload.get("loaded_count"),
            "errors": verify.payload.get("errors"),
            "latency_ms": verify.elapsed_ms,
        },
        "audit": {
            "passed": audit.returncode == 0 and audit.payload.get("status") == "ok",
            "status": audit.payload.get("status"),
            "event_count": (audit.payload.get("audit") or {}).get("event_count"),
            "latency_ms": audit.elapsed_ms,
        },
        "markers": {
            "passed": not missing_markers,
            "expected": args.writes,
            "missing": sorted(missing_markers),
        },
    }


def run_stress(args: argparse.Namespace, runner: Callable[[list[str]], CommandResult] | None = None) -> dict[str, Any]:
    temp_root: Path | None = None
    if args.storage_dir:
        storage_dir = Path(args.storage_dir).expanduser().resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="dml-stress-"))
        storage_dir = temp_root / "store"
    command_runner = runner or (lambda argv: _run_json_command(argv, timeout_s=args.timeout_s))
    try:
        writes = _run_concurrent_writes(args, storage_dir, command_runner)
        isolation = _run_isolation_checks(args, storage_dir, command_runner)
        checks = _run_store_checks(args, storage_dir, command_runner)
        write_ok = all(item["returncode"] == 0 and item["status"] == "ok" and item.get("chunks_ingested", 0) >= 1 for item in writes)
        isolation_ok = all(item["passed"] for item in isolation)
        checks_ok = all(item["passed"] for item in checks.values())
        summary = {
            "schema_version": "dml.stress-harness.v1",
            "created_at": _utc_now(),
            "run_id": args.run_id,
            "status": "pass" if write_ok and isolation_ok and checks_ok else "fail",
            "storage_dir": str(storage_dir),
            "write_count": len(writes),
            "write_ok": write_ok,
            "isolation_ok": isolation_ok,
            "checks_ok": checks_ok,
            "writes": writes,
            "isolation": isolation,
            "checks": checks,
        }
        return summary
    finally:
        if temp_root is not None and not args.keep_store:
            import shutil

            shutil.rmtree(temp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DML wrapper concurrency/isolation stress checks")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dml-script", type=Path, default=DEFAULT_DML_SCRIPT)
    parser.add_argument("--config-path")
    parser.add_argument("--storage-dir")
    parser.add_argument("--run-id", default=f"stress-{int(time.time())}")
    parser.add_argument("--writes", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tenants", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--lock-timeout-ms", type=int, default=30000)
    parser.add_argument("--audit-actor", default="stress-harness")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-store", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_stress(args)
    print(_json_dumps(summary))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
