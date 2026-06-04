#!/usr/bin/env python3
"""Read-only stale-memory detector for Daystrom DML JSONL state files.

The detector emits review candidates only; it never mutates the source store.
Use it as an observability input for review dashboards and hygiene gates.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "dml.stale-memory-report.v1"


def _parse_ts(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            obj.setdefault("_line_no", line_no)
            items.append(obj)
    return items


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _age_days(item: dict[str, Any], now: float) -> float | None:
    meta = _meta(item)
    ts = (
        _parse_ts(meta.get("updated_at"))
        or _parse_ts(item.get("updated_at"))
        or _parse_ts(item.get("timestamp"))
        or _parse_ts(meta.get("created_at"))
        or _parse_ts(item.get("created_at"))
    )
    if ts is None:
        return None
    return max(0.0, (now - ts) / 86400.0)


def _score(
    item: dict[str, Any],
    *,
    now: float,
    stale_after_days: float,
    low_fidelity: float,
) -> tuple[float, list[str], str]:
    meta = _meta(item)
    state = str(meta.get("memory_state") or meta.get("lifecycle_state") or "active").lower()
    reasons: list[str] = []
    score = 0.0

    age = _age_days(item, now)
    if age is None:
        score += 0.15
        reasons.append("missing_timestamp")
    elif age >= stale_after_days:
        score += min(0.35, 0.15 + (age - stale_after_days) / max(stale_after_days, 1.0) * 0.20)
        reasons.append("old_age")

    fidelity = item.get("fidelity", meta.get("fidelity"))
    if isinstance(fidelity, (int, float)) and float(fidelity) < low_fidelity:
        score += 0.25
        reasons.append("low_fidelity")

    quality = meta.get("quality_score")
    if isinstance(quality, (int, float)) and float(quality) < 0.45:
        score += 0.20
        reasons.append("low_quality_score")

    has_summary = bool(str(meta.get("summary") or "").strip()) or bool(meta.get("abstracted"))
    if not has_summary:
        score += 0.10
        reasons.append("missing_summary")

    ttl_days = meta.get("ttl_days")
    constraints = meta.get("constraints")
    if ttl_days is None and isinstance(constraints, dict):
        ttl_days = constraints.get("ttl_days")
    if isinstance(ttl_days, (int, float)) and age is not None and age > float(ttl_days):
        score += 0.30
        reasons.append("ttl_expired")

    if state in {"suppressed", "deleted"}:
        score -= 0.25
        reasons.append(f"already_{state}")

    score = round(max(0.0, min(1.0, score)), 4)
    if score >= 0.70:
        action = "suppress_candidate"
    elif score >= 0.40:
        action = "review"
    elif "missing_summary" in reasons or "old_age" in reasons:
        action = "refresh_candidate"
    else:
        action = "keep"
    return score, reasons, action


def build_report(
    path: Path,
    *,
    stale_after_days: float,
    low_fidelity: float,
    limit: int,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    items = _load_jsonl(path)
    candidates: list[dict[str, Any]] = []
    for item in items:
        meta = _meta(item)
        score, reasons, action = _score(
            item,
            now=timestamp,
            stale_after_days=stale_after_days,
            low_fidelity=low_fidelity,
        )
        if action == "keep":
            continue
        candidates.append(
            {
                "id": str(item.get("id") or item.get("memory_id") or item.get("_line_no")),
                "line_no": item.get("_line_no"),
                "stale_score": score,
                "recommended_action": action,
                "reasons": reasons,
                "state": str(meta.get("memory_state") or meta.get("lifecycle_state") or "active"),
                "source": meta.get("source"),
                "namespace": meta.get("namespace"),
                "summary": str(meta.get("summary") or item.get("text") or "")[:220],
            }
        )
    candidates.sort(key=lambda row: (-float(row["stale_score"]), str(row["id"])))
    shown = min(max(0, limit), len(candidates))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "state_file": str(path),
        "status": "ok",
        "thresholds": {"stale_after_days": stale_after_days, "low_fidelity": low_fidelity},
        "totals": {
            "items_scanned": len(items),
            "candidates": len(candidates),
            "shown": shown,
        },
        "candidates": candidates[:shown],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only stale-memory detector for Daystrom DML JSONL state files")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--stale-after-days", type=float, default=90.0)
    parser.add_argument("--low-fidelity", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.state_file.expanduser().resolve(),
        stale_after_days=args.stale_after_days,
        low_fidelity=args.low_fidelity,
        limit=max(1, args.limit),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
