#!/usr/bin/env python3
"""Recall-quality eval harness for the OpenClaw DML wrapper.

The harness writes a tiny deterministic fixture set, runs retrieve/resume
queries through the same CLI surface used by agents, and emits JSON plus a
Markdown summary. It is intended as a low-cost regression gate for recall,
scope isolation, and compaction-continuity behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
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


@dataclass(frozen=True)
class EvalCase:
    name: str
    action: str
    query: str
    tenant_id: str
    expected_markers: tuple[str, ...]
    forbidden_markers: tuple[str, ...] = ()
    session_id: str | None = None
    top_k: int = 6


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
        payload = {
            "status": "invalid_json",
            "stdout_preview": proc.stdout[:1200],
        }
    return CommandResult(
        argv=argv,
        returncode=proc.returncode,
        elapsed_ms=round(elapsed_ms, 2),
        payload=payload,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _command_prefix(args: argparse.Namespace, storage_dir: Path) -> list[str]:
    prefix = [
        args.python,
        str(args.dml_script),
        "--storage-dir",
        str(storage_dir),
        "--audit-actor",
        args.audit_actor,
    ]
    if args.config_path:
        prefix.extend(["--config-path", args.config_path])
    if not args.require_gpu:
        prefix.append("--no-require-gpu")
    prefix.extend(["--lock-timeout-ms", str(args.lock_timeout_ms)])
    return prefix


def _fixture_texts(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "tenant_id": f"{run_id}-alpha",
            "session_id": "session-alpha",
            "kind": "plan",
            "marker": f"{run_id}-ALPHA-EXPORT-PLAN",
            "text": (
                f"{run_id}-ALPHA-EXPORT-PLAN: Alpha tenant export plan. "
                "Use staged checksum backups before promoting the memory store."
            ),
            "meta": {"source": "recall_eval", "namespace": "project_plan"},
        },
        {
            "tenant_id": f"{run_id}-beta",
            "session_id": "session-beta",
            "kind": "note",
            "marker": f"{run_id}-BETA-BILLING-NOTE",
            "text": (
                f"{run_id}-BETA-BILLING-NOTE: Beta tenant billing note. "
                "This should not appear during alpha tenant recall."
            ),
            "meta": {"source": "recall_eval", "namespace": "billing"},
        },
        {
            "tenant_id": f"{run_id}-alpha",
            "session_id": "session-two",
            "kind": "note",
            "marker": f"{run_id}-ALPHA-SESSION-TWO",
            "text": (
                f"{run_id}-ALPHA-SESSION-TWO: Alpha second session note. "
                "This session should not leak into session-alpha recall."
            ),
            "meta": {"source": "recall_eval", "namespace": "session_scope"},
        },
        {
            "tenant_id": f"{run_id}-alpha",
            "session_id": "session-alpha",
            "kind": "plan",
            "marker": f"{run_id}-CONTINUITY-NEXT-ACTION",
            "text": "\n".join(
                [
                    "[source:rolling_thread_checkpoint]",
                    "thread: recall-eval",
                    "state: hardening",
                    "task: validate continuity resume",
                    f"next_action: {run_id}-CONTINUITY-NEXT-ACTION run eval report review",
                ]
            ),
            "meta": {
                "source": "rolling_thread_checkpoint",
                "namespace": "active_continuity",
                "memory_state": "active",
                "thread": "recall-eval",
                "state": "hardening",
                "task": "validate continuity resume",
                "next_action": f"{run_id}-CONTINUITY-NEXT-ACTION run eval report review",
            },
        },
    ]


def _eval_cases(run_id: str) -> list[EvalCase]:
    return [
        EvalCase(
            name="tenant_recall_and_isolation",
            action="retrieve",
            query=f"{run_id} alpha export plan checksum backups",
            tenant_id=f"{run_id}-alpha",
            session_id="session-alpha",
            expected_markers=(f"{run_id}-ALPHA-EXPORT-PLAN",),
            forbidden_markers=(f"{run_id}-BETA-BILLING-NOTE", f"{run_id}-ALPHA-SESSION-TWO"),
        ),
        EvalCase(
            name="session_scope_isolation",
            action="retrieve",
            query=f"{run_id} alpha second session note",
            tenant_id=f"{run_id}-alpha",
            session_id="session-alpha",
            expected_markers=(f"{run_id}-ALPHA-EXPORT-PLAN",),
            forbidden_markers=(f"{run_id}-ALPHA-SESSION-TWO", f"{run_id}-BETA-BILLING-NOTE"),
        ),
        EvalCase(
            name="continuity_resume",
            action="resume",
            query=f"{run_id} active continuity checkpoint next action",
            tenant_id=f"{run_id}-alpha",
            session_id="session-alpha",
            expected_markers=(f"{run_id}-CONTINUITY-NEXT-ACTION",),
            forbidden_markers=(f"{run_id}-BETA-BILLING-NOTE",),
            top_k=12,
        ),
    ]


def _contains_marker(payload: dict[str, Any], marker: str) -> bool:
    return marker.lower() in json.dumps(payload, sort_keys=True, default=str).lower()


def _marker_rank(payload: dict[str, Any], marker: str) -> int | None:
    marker_lower = marker.lower()
    for idx, item in enumerate(payload.get("items") or [], start=1):
        if marker_lower in json.dumps(item, sort_keys=True, default=str).lower():
            return idx
    raw_context = str(payload.get("raw_context") or "").lower()
    if marker_lower in raw_context:
        return 1
    return None


def _score_case(case: EvalCase, result: CommandResult) -> dict[str, Any]:
    payload = result.payload
    expected_hits = {marker: _contains_marker(payload, marker) for marker in case.expected_markers}
    forbidden_hits = {marker: _contains_marker(payload, marker) for marker in case.forbidden_markers}
    ranks = {marker: _marker_rank(payload, marker) for marker in case.expected_markers}
    passed = (
        result.returncode == 0
        and str(payload.get("status", "ok")) in {"ok", "degraded"}
        and all(expected_hits.values())
        and not any(forbidden_hits.values())
    )
    return {
        "name": case.name,
        "action": case.action,
        "passed": passed,
        "query": case.query,
        "tenant_id": case.tenant_id,
        "session_id": case.session_id,
        "returncode": result.returncode,
        "status": payload.get("status"),
        "latency_ms": result.elapsed_ms,
        "reported_latency_ms": payload.get("retrieve_total_latency_ms") or payload.get("resume_total_latency_ms"),
        "items_seen": len(payload.get("items") or []),
        "memory_confidence": payload.get("memory_confidence"),
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
        "expected_ranks": ranks,
        "stderr_preview": result.stderr[:1200],
    }


def _run_ingests(
    args: argparse.Namespace,
    *,
    storage_dir: Path,
    prefix: list[str],
    runner: Callable[[list[str]], CommandResult],
    run_id: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fixture in _fixture_texts(run_id):
        meta = dict(fixture["meta"])
        meta["recall_eval_run_id"] = run_id
        argv = [
            *prefix,
            "ingest",
            "--tenant-id",
            fixture["tenant_id"],
            "--session-id",
            fixture["session_id"],
            "--kind",
            fixture["kind"],
            "--summary-policy",
            "skip",
            "--no-chunk",
            "--filter-noise",
            "--meta",
            json.dumps(meta, separators=(",", ":"), sort_keys=True),
            "--text",
            fixture["text"],
        ]
        result = runner(argv)
        results.append(
            {
                "marker": fixture["marker"],
                "tenant_id": fixture["tenant_id"],
                "session_id": fixture["session_id"],
                "returncode": result.returncode,
                "status": result.payload.get("status"),
                "latency_ms": result.elapsed_ms,
                "chunks_ingested": result.payload.get("chunks_ingested"),
                "stderr_preview": result.stderr[:1200],
            }
        )
    return results


def _run_cases(
    args: argparse.Namespace,
    *,
    prefix: list[str],
    runner: Callable[[list[str]], CommandResult],
    run_id: str,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for case in _eval_cases(run_id):
        if case.action == "resume":
            argv = [
                *prefix,
                "resume",
                "--query",
                case.query,
                "--tenant-id",
                case.tenant_id,
                "--top-k",
                str(case.top_k),
                "--fallback-items",
                "3",
            ]
        else:
            argv = [
                *prefix,
                "retrieve",
                "--query",
                case.query,
                "--tenant-id",
                case.tenant_id,
                "--top-k",
                str(case.top_k),
                "--no-query-expand",
                "--ground-truth-policy",
                "never",
                "--no-reform-memory",
            ]
        if case.session_id:
            argv.extend(["--session-id", case.session_id])
        scored.append(_score_case(case, runner(argv)))
    return scored


def _summarize(run_id: str, storage_dir: Path, ingests: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    ingest_ok = all(item.get("returncode") == 0 and item.get("status") == "ok" for item in ingests)
    case_ok = all(case.get("passed") for case in cases)
    latencies = [float(item["latency_ms"]) for item in ingests + cases if item.get("latency_ms") is not None]
    return {
        "schema_version": "dml.recall-eval.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "storage_dir": str(storage_dir),
        "status": "pass" if ingest_ok and case_ok else "fail",
        "ingest_count": len(ingests),
        "ingest_ok": ingest_ok,
        "case_count": len(cases),
        "case_pass_count": sum(1 for case in cases if case.get("passed")),
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "ingests": ingests,
        "cases": cases,
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# DML Recall Eval",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['case_pass_count']}/{summary['case_count']}`",
        f"- Avg latency: `{summary['avg_latency_ms']} ms`",
        f"- Max latency: `{summary['max_latency_ms']} ms`",
        "",
        "| Case | Status | Latency ms | Expected | Forbidden |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for case in summary["cases"]:
        expected = ", ".join(f"{k}:{v}" for k, v in case["expected_hits"].items())
        forbidden = ", ".join(f"{k}:{v}" for k, v in case["forbidden_hits"].items()) or "none"
        lines.append(
            f"| `{case['name']}` | `{'pass' if case['passed'] else 'fail'}` | "
            f"{case['latency_ms']} | {expected} | {forbidden} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_reports(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recall_eval_report.json"
    md_path = output_dir / "recall_eval_report.md"
    json_path.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_eval(args: argparse.Namespace, runner: Callable[[list[str]], CommandResult] | None = None) -> dict[str, Any]:
    run_id = args.run_id or f"recall-eval-{int(time.time())}"
    temp_root: Path | None = None
    if args.storage_dir:
        storage_dir = Path(args.storage_dir).expanduser().resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="dml-recall-eval-"))
        storage_dir = temp_root / "store"

    command_runner = runner or (lambda argv: _run_json_command(argv, timeout_s=args.timeout_s))
    prefix = _command_prefix(args, storage_dir)
    try:
        ingests = _run_ingests(args, storage_dir=storage_dir, prefix=prefix, runner=command_runner, run_id=run_id)
        cases = _run_cases(args, prefix=prefix, runner=command_runner, run_id=run_id)
        summary = _summarize(run_id, storage_dir, ingests, cases)
        if args.output_dir:
            summary["reports"] = _write_reports(summary, Path(args.output_dir).expanduser().resolve())
        return summary
    finally:
        if temp_root is not None and not args.keep_store:
            shutil.rmtree(temp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DML recall quality eval fixtures")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run dml_memory.py")
    parser.add_argument("--dml-script", type=Path, default=DEFAULT_DML_SCRIPT)
    parser.add_argument("--config-path")
    parser.add_argument("--storage-dir", help="Store under test. Defaults to a temporary isolated store.")
    parser.add_argument("--output-dir", default="out/recall-eval", help="Directory for JSON and Markdown reports")
    parser.add_argument("--run-id", help="Stable marker prefix. Defaults to a timestamped id.")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--lock-timeout-ms", type=int, default=250)
    parser.add_argument("--audit-actor", default="recall-eval")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-store", action="store_true", help="Keep temporary store for debugging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_eval(args)
    print(_json_dumps(summary))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
