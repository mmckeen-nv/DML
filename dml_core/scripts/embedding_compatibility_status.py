"""Render a human-readable status view for DML embedding compatibility migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_REPORT_PATH = Path("/home/nvidia/.openclaw/workspace/data/dml-gpu-prod/embedding_compatibility_report.json")


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_report(report: Dict[str, Any], *, report_path: Path) -> str:
    status = report.get("status", "unknown")
    phase = report.get("phase", "unknown")
    detail = report.get("phase_detail") or "no detail"
    checked = int(report.get("checked") or 0)
    total = int(report.get("total_items") or 0)
    remaining = int(report.get("remaining_items") or max(total - checked, 0))
    progress_pct = float(report.get("progress_pct") or 0.0)
    mismatched = int(report.get("mismatched") or 0)
    reembedded = int(report.get("reembedded") or 0)
    failed = int(report.get("failed") or 0)
    target_dim = int(report.get("target_dim") or 0)
    current_idx = int(report.get("current_item_index") or 0)
    current_preview = report.get("current_item_preview") or "-"
    last_done_idx = int(report.get("last_completed_item_index") or 0)
    last_done_preview = report.get("last_completed_item_preview") or "-"
    started_at = report.get("started_at") or "-"
    updated_at = report.get("updated_at") or "-"
    elapsed_ms = float(report.get("elapsed_ms") or 0.0)

    return "\n".join(
        [
            f"report_path: {report_path}",
            f"status: {status}",
            f"phase: {phase}",
            f"detail: {detail}",
            f"progress: {progress_pct:.2f}% ({checked}/{total}, remaining={remaining})",
            f"migration_counts: mismatched={mismatched} reembedded={reembedded} failed={failed} target_dim={target_dim}",
            f"current_item: index={current_idx} preview={current_preview}",
            f"last_completed: index={last_done_idx} preview={last_done_preview}",
            f"timing: started_at={started_at} updated_at={updated_at} elapsed_ms={elapsed_ms:.2f}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print DML embedding compatibility migration status")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to embedding_compatibility_report.json",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of the formatted status view")
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")

    report = load_report(args.report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report, report_path=args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
