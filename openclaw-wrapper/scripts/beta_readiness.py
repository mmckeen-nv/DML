#!/usr/bin/env python3
"""Portable beta readiness gate for the DML OpenClaw wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DML_SCRIPT = SCRIPT_DIR / "dml_memory.py"
DEFAULT_RECALL_EVAL_SCRIPT = SCRIPT_DIR / "recall_eval.py"


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
    return CommandResult(
        argv=argv,
        returncode=proc.returncode,
        elapsed_ms=round(elapsed_ms, 2),
        payload=payload,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _dml_prefix(args: argparse.Namespace) -> list[str]:
    prefix = [
        args.python,
        str(args.dml_script),
        "--storage-dir",
        str(args.storage_dir),
        "--audit-actor",
        args.audit_actor,
    ]
    if args.config_path:
        prefix.extend(["--config-path", args.config_path])
    if not args.require_gpu:
        prefix.append("--no-require-gpu")
    prefix.extend(["--lock-timeout-ms", str(args.lock_timeout_ms)])
    return prefix


def _step_result(
    name: str,
    result: CommandResult,
    *,
    pass_statuses: set[str],
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    status = str(result.payload.get("status") or "missing")
    passed = (allow_nonzero or result.returncode == 0) and status in pass_statuses
    return {
        "name": name,
        "passed": passed,
        "returncode": result.returncode,
        "status": status,
        "latency_ms": result.elapsed_ms,
        "payload": result.payload,
        "stderr_preview": result.stderr[:1200],
    }


def _recall_eval_args(args: argparse.Namespace) -> list[str]:
    argv = [
        args.python,
        str(args.recall_eval_script),
        "--python",
        args.python,
        "--dml-script",
        str(args.dml_script),
        "--run-id",
        args.recall_run_id or f"beta-readiness-{int(time.time())}",
        "--timeout-s",
        str(args.timeout_s),
        "--lock-timeout-ms",
        str(args.lock_timeout_ms),
        "--audit-actor",
        args.audit_actor,
    ]
    if args.config_path:
        argv.extend(["--config-path", args.config_path])
    if args.require_gpu:
        argv.append("--require-gpu")
    return argv


def _summarize(args: argparse.Namespace, steps: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [step["name"] for step in steps if not step.get("passed")]
    warnings: list[str] = []
    conflicts_step = next((step for step in steps if step["name"] == "conflicts"), None)
    if conflicts_step:
        conflict_count = int(conflicts_step["payload"].get("conflict_group_count") or 0)
        if conflict_count > args.max_unresolved_conflicts:
            failed.append("conflict_budget")
        elif conflict_count:
            warnings.append(f"unresolved_conflicts={conflict_count}")
    max_latency = max((float(step["latency_ms"]) for step in steps), default=0.0)
    if max_latency > args.max_step_latency_ms:
        warnings.append(f"max_step_latency_ms={round(max_latency, 2)}")
    return {
        "schema_version": "dml.beta-readiness.v1",
        "created_at": _utc_now(),
        "status": "pass" if not failed else "fail",
        "storage_dir": str(args.storage_dir),
        "failed_steps": sorted(set(failed)),
        "warnings": warnings,
        "max_step_latency_ms": round(max_latency, 2),
        "steps": steps,
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# DML Beta Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Storage: `{summary['storage_dir']}`",
        f"- Max step latency: `{summary['max_step_latency_ms']} ms`",
        "",
        "| Step | Status | Latency ms |",
        "| --- | --- | ---: |",
    ]
    for step in summary["steps"]:
        lines.append(f"| `{step['name']}` | `{'pass' if step['passed'] else 'fail'}` | {step['latency_ms']} |")
    if summary["failed_steps"]:
        lines.extend(["", "Failed steps:", ", ".join(f"`{name}`" for name in summary["failed_steps"])])
    if summary["warnings"]:
        lines.extend(["", "Warnings:", ", ".join(f"`{warning}`" for warning in summary["warnings"])])
    lines.append("")
    return "\n".join(lines)


def _write_reports(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "beta_readiness_report.json"
    md_path = output_dir / "beta_readiness_report.md"
    json_path.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_gate(args: argparse.Namespace, runner: Callable[[list[str]], CommandResult] | None = None) -> dict[str, Any]:
    command_runner = runner or (lambda argv: _run_json_command(argv, timeout_s=args.timeout_s))
    prefix = _dml_prefix(args)
    steps: list[dict[str, Any]] = []

    health = command_runner([*prefix, "health"])
    steps.append(_step_result("health", health, pass_statuses={"ok", "degraded"}))

    verify = command_runner([*prefix, "verify"])
    steps.append(_step_result("verify", verify, pass_statuses={"ok"}))

    conflicts = command_runner(
        [
            *prefix,
            "conflicts",
            "--tenant-id",
            args.tenant_id,
            "--limit",
            str(args.conflict_limit),
        ]
    )
    steps.append(_step_result("conflicts", conflicts, pass_statuses={"ok"}))

    audit = command_runner([*prefix, "audit-tail", "--limit", str(args.audit_limit)])
    steps.append(_step_result("audit-tail", audit, pass_statuses={"ok"}))

    if not args.skip_recall_eval:
        recall = command_runner(_recall_eval_args(args))
        steps.append(_step_result("recall-eval", recall, pass_statuses={"pass"}))

    summary = _summarize(args, steps)
    if args.output_dir:
        summary["reports"] = _write_reports(summary, Path(args.output_dir).expanduser().resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run portable DML beta readiness checks")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dml-script", type=Path, default=DEFAULT_DML_SCRIPT)
    parser.add_argument("--recall-eval-script", type=Path, default=DEFAULT_RECALL_EVAL_SCRIPT)
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--tenant-id", default="openclaw")
    parser.add_argument("--audit-actor", default="beta-readiness")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--lock-timeout-ms", type=int, default=500)
    parser.add_argument("--conflict-limit", type=int, default=20)
    parser.add_argument("--audit-limit", type=int, default=10)
    parser.add_argument("--max-unresolved-conflicts", type=int, default=0)
    parser.add_argument("--max-step-latency-ms", type=float, default=5000.0)
    parser.add_argument("--recall-run-id")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-recall-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_gate(args)
    print(_json_dumps(summary))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
