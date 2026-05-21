#!/usr/bin/env python3
"""Smoke test continuity resume ordering across several OpenClaw sessions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DML_SCRIPT = SCRIPT_DIR / "dml_memory.py"


def _run(argv: list[str], timeout_s: float) -> dict[str, Any]:
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_s, check=False)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"status": "invalid_json", "stdout": proc.stdout[-1200:]}
    payload["_returncode"] = proc.returncode
    payload["_stderr"] = proc.stderr[-1200:]
    return payload


def _base(args: argparse.Namespace, storage_dir: Path) -> list[str]:
    argv = [
        args.python,
        str(args.dml_script),
        "--storage-dir",
        str(storage_dir),
        "--config-path",
        args.config_path,
        "--lock-timeout-ms",
        str(args.lock_timeout_ms),
        "--audit-actor",
        "resume-quality-smoke",
    ]
    if not args.require_gpu:
        argv.append("--no-require-gpu")
    return argv


def run_smoke(args: argparse.Namespace, runner: Callable[[list[str]], dict[str, Any]] | None = None) -> dict[str, Any]:
    temp_root: Path | None = None
    if args.storage_dir:
        storage_dir = Path(args.storage_dir).expanduser().resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="dml-resume-smoke-"))
        storage_dir = temp_root / "store"
    command_runner = runner or (lambda argv: _run(argv, args.timeout_s))
    try:
        writes = []
        for idx in range(args.sessions):
            updated_at = f"2026-05-21T12:0{idx}:00Z"
            next_action = f"{args.run_id}-NEXT-{idx}"
            payload = command_runner(
                [
                    *_base(args, storage_dir),
                    "handoff",
                    "--tenant-id",
                    args.tenant_id,
                    "--session-id",
                    f"session-{idx}",
                    "--thread",
                    f"session-{idx}",
                    "--state",
                    "executing",
                    "--task",
                    "resume quality smoke",
                    "--next-action",
                    next_action,
                    "--updated-at",
                    updated_at,
                ]
            )
            writes.append({"index": idx, "next_action": next_action, "status": payload.get("status"), "returncode": payload.get("_returncode")})
        resume = command_runner(
            [
                *_base(args, storage_dir),
                "resume",
                "--tenant-id",
                args.tenant_id,
                "--top-k",
                str(max(args.sessions + 2, 12)),
            ]
        )
        expected = f"{args.run_id}-NEXT-{args.sessions - 1}"
        latest = resume.get("latest_checkpoint") or {}
        passed = all(item["returncode"] == 0 and item["status"] == "ok" for item in writes) and latest.get("next_action") == expected
        return {
            "schema_version": "dml.resume-quality-smoke.v1",
            "status": "pass" if passed else "fail",
            "storage_dir": str(storage_dir),
            "expected_next_action": expected,
            "latest_checkpoint": latest,
            "writes": writes,
            "resume_returncode": resume.get("_returncode"),
        }
    finally:
        if temp_root is not None and not args.keep_store:
            import shutil

            shutil.rmtree(temp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DML resume quality smoke")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dml-script", type=Path, default=DEFAULT_DML_SCRIPT)
    parser.add_argument("--config-path", default=str(SCRIPT_DIR.parent / "config" / "dml_portable_linux.yaml"))
    parser.add_argument("--storage-dir")
    parser.add_argument("--tenant-id", default="openclaw")
    parser.add_argument("--run-id", default=f"resume-{int(time.time())}")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--lock-timeout-ms", type=int, default=30000)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-store", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
