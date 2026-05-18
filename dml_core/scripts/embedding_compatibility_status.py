"""Render a human-readable status view for DML embedding compatibility migration."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_HOME = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = REPO_HOME / "data" / "embedding_compatibility_report.json"
DEFAULT_MARKDOWN_PATH = REPO_HOME / "out" / "dml-ollama-live-store-migration-status.md"
DEFAULT_SNAPSHOT_PATH = REPO_HOME / "out" / "dml-ollama-live-store-migration-snapshot.json"


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso8601(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _freshness(updated_at: Any, *, now: datetime | None = None) -> Dict[str, Any]:
    updated = _parse_iso8601(updated_at)
    if updated is None:
        return {"updated_age_s": None, "freshness": "unknown"}
    now_utc = now or datetime.now(timezone.utc)
    age_s = max((now_utc - updated).total_seconds(), 0.0)
    if age_s <= 60:
        freshness = "fresh"
    elif age_s <= 300:
        freshness = "recent"
    else:
        freshness = "stale"
    return {"updated_age_s": round(age_s, 1), "freshness": freshness}


def _report_fields(report: Dict[str, Any], *, report_path: Path) -> Dict[str, Any]:
    checked = int(report.get("checked") or 0)
    total = int(report.get("total_items") or 0)
    freshness = _freshness(report.get("updated_at"))
    return {
        "report_path": report_path,
        "status": report.get("status", "unknown"),
        "phase": report.get("phase", "unknown"),
        "detail": report.get("phase_detail") or "no detail",
        "checked": checked,
        "total": total,
        "remaining": int(report.get("remaining_items") or max(total - checked, 0)),
        "progress_pct": float(report.get("progress_pct") or 0.0),
        "mismatched": int(report.get("mismatched") or 0),
        "reembedded": int(report.get("reembedded") or 0),
        "failed": int(report.get("failed") or 0),
        "target_dim": int(report.get("target_dim") or 0),
        "current_idx": int(report.get("current_item_index") or 0),
        "current_preview": report.get("current_item_preview") or "-",
        "last_done_idx": int(report.get("last_completed_item_index") or 0),
        "last_done_preview": report.get("last_completed_item_preview") or "-",
        "started_at": report.get("started_at") or "-",
        "updated_at": report.get("updated_at") or "-",
        "elapsed_ms": float(report.get("elapsed_ms") or 0.0),
        "updated_age_s": freshness["updated_age_s"],
        "freshness": freshness["freshness"],
    }


def format_status_line(report: Dict[str, Any], *, report_path: Path) -> str:
    fields = _report_fields(report, report_path=report_path)
    return " | ".join(
        [
            f"migration_status={fields['status']}",
            f"phase={fields['phase']}",
            f"progress={fields['progress_pct']:.2f}%",
            f"checked={fields['checked']}/{fields['total']}",
            f"remaining={fields['remaining']}",
            f"current={fields['current_idx']}",
            f"last_completed={fields['last_done_idx']}",
            f"freshness={fields['freshness']}",
            f"updated_age_s={fields['updated_age_s'] if fields['updated_age_s'] is not None else '-'}",
            f"report={fields['report_path']}",
        ]
    )


def format_progress_snapshot(report: Dict[str, Any], *, report_path: Path) -> Dict[str, Any]:
    fields = _report_fields(report, report_path=report_path)
    return {
        "report_path": str(fields["report_path"]),
        "migration_status": fields["status"],
        "phase": fields["phase"],
        "phase_detail": fields["detail"],
        "progress": {
            "pct": round(fields["progress_pct"], 2),
            "checked": fields["checked"],
            "total": fields["total"],
            "remaining": fields["remaining"],
        },
        "current_item": {
            "index": fields["current_idx"],
            "preview": fields["current_preview"],
        },
        "last_completed": {
            "index": fields["last_done_idx"],
            "preview": fields["last_done_preview"],
        },
        "migration_counts": {
            "mismatched": fields["mismatched"],
            "reembedded": fields["reembedded"],
            "failed": fields["failed"],
            "target_dim": fields["target_dim"],
        },
        "timing": {
            "started_at": fields["started_at"],
            "updated_at": fields["updated_at"],
            "elapsed_ms": round(fields["elapsed_ms"], 2),
            "updated_age_s": fields["updated_age_s"],
            "freshness": fields["freshness"],
        },
        "status_line": format_status_line(report, report_path=report_path),
    }


def format_report(report: Dict[str, Any], *, report_path: Path) -> str:
    fields = _report_fields(report, report_path=report_path)
    return "\n".join(
        [
            f"report_path: {fields['report_path']}",
            f"status: {fields['status']}",
            f"phase: {fields['phase']}",
            f"detail: {fields['detail']}",
            f"progress: {fields['progress_pct']:.2f}% ({fields['checked']}/{fields['total']}, remaining={fields['remaining']})",
            (
                "migration_counts: "
                f"mismatched={fields['mismatched']} reembedded={fields['reembedded']} "
                f"failed={fields['failed']} target_dim={fields['target_dim']}"
            ),
            f"current_item: index={fields['current_idx']} preview={fields['current_preview']}",
            f"last_completed: index={fields['last_done_idx']} preview={fields['last_done_preview']}",
            f"freshness: {fields['freshness']} updated_age_s={fields['updated_age_s'] if fields['updated_age_s'] is not None else '-'}",
            (
                "timing: "
                f"started_at={fields['started_at']} updated_at={fields['updated_at']} "
                f"elapsed_ms={fields['elapsed_ms']:.2f}"
            ),
        ]
    )


def format_markdown_report(report: Dict[str, Any], *, report_path: Path) -> str:
    fields = _report_fields(report, report_path=report_path)
    return "\n".join(
        [
            "# DML Ollama Live-Store Migration Status",
            "",
            f"- status_line: `{format_status_line(report, report_path=report_path)}`",
            f"- report_path: `{fields['report_path']}`",
            f"- status: `{fields['status']}`",
            f"- phase: `{fields['phase']}`",
            f"- detail: {fields['detail']}",
            f"- progress: `{fields['progress_pct']:.2f}% ({fields['checked']}/{fields['total']}, remaining={fields['remaining']})`",
            (
                "- migration_counts: "
                f"`mismatched={fields['mismatched']} reembedded={fields['reembedded']} "
                f"failed={fields['failed']} target_dim={fields['target_dim']}`"
            ),
            f"- current_item: `index={fields['current_idx']}` preview=`{fields['current_preview']}`",
            f"- last_completed: `index={fields['last_done_idx']}` preview=`{fields['last_done_preview']}`",
            (
                "- freshness: "
                f"`{fields['freshness']}` "
                f"(updated_age_s={fields['updated_age_s'] if fields['updated_age_s'] is not None else '-'})"
            ),
            (
                "- timing: "
                f"`started_at={fields['started_at']} updated_at={fields['updated_at']} "
                f"elapsed_ms={fields['elapsed_ms']:.2f}`"
            ),
            "",
            "Generated from the durable live-store migration artifact; no separate state store is introduced.",
        ]
    )


def write_markdown_report(report: Dict[str, Any], *, report_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(report, report_path=report_path), encoding="utf-8")
    return output_path


def write_progress_snapshot(report: Dict[str, Any], *, report_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(format_progress_snapshot(report, report_path=report_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print DML embedding compatibility migration status")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to embedding_compatibility_report.json",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of the formatted status view")
    parser.add_argument("--one-line", action="store_true", help="Print a compact single-line status summary")
    parser.add_argument(
        "--snapshot-json",
        action="store_true",
        help="Print a derived progress-only JSON snapshot from the durable migration artifact",
    )
    parser.add_argument(
        "--write-markdown",
        nargs="?",
        const=str(DEFAULT_MARKDOWN_PATH),
        help="Write a markdown status card to the given path (defaults to the workspace out/ card path)",
    )
    parser.add_argument(
        "--write-snapshot-json",
        nargs="?",
        const=str(DEFAULT_SNAPSHOT_PATH),
        help="Write the derived progress-only JSON snapshot to the given path (defaults to the workspace out/ snapshot path)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")

    report = load_report(args.report)
    if args.write_markdown:
        written = write_markdown_report(report, report_path=args.report, output_path=Path(args.write_markdown))
        print(f"wrote_markdown: {written}")
        return 0
    if args.write_snapshot_json:
        written = write_progress_snapshot(report, report_path=args.report, output_path=Path(args.write_snapshot_json))
        print(f"wrote_snapshot_json: {written}")
        return 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.snapshot_json:
        print(json.dumps(format_progress_snapshot(report, report_path=args.report), indent=2, sort_keys=True))
    elif args.one_line:
        print(format_status_line(report, report_path=args.report))
    else:
        print(format_report(report, report_path=args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
