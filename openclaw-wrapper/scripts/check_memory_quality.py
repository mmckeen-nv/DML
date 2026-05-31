#!/usr/bin/env python3
"""Lightweight memory quality guardrail check for persisted DML state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_items(state_file: Path) -> list[dict]:
    lines = state_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= 1:
        return []
    items: list[dict] = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            items.append(rec)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    durable_home = Path(os.environ.get("DAYSTROM_DML_HOME", "/Users/markmckeen/.openclaw/daystrom-dml-v2")).resolve()
    ap.add_argument("--state-file", default=str(durable_home / "data" / "dml_state.jsonl"))
    ap.add_argument("--old-fidelity-threshold", type=float, default=0.55)
    ap.add_argument("--min-old-summary-ratio", type=float, default=0.60)
    args = ap.parse_args()

    state_file = Path(args.state_file)
    if not state_file.exists():
        print(json.dumps({"status": "fail", "reason": f"missing state file: {state_file}"}, indent=2))
        return 2

    items = load_items(state_file)
    old_items = [x for x in items if float(x.get("fidelity") or 0.0) < args.old_fidelity_threshold]

    def is_summarized(item: dict) -> bool:
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            return False
        summary = str(meta.get("summary") or "").strip()
        return bool(summary) or bool(meta.get("abstracted")) or bool(meta.get("abstracted_from"))

    summarized_old = [x for x in old_items if is_summarized(x)]
    ratio = (len(summarized_old) / len(old_items)) if old_items else 1.0
    status = "pass" if ratio >= args.min_old_summary_ratio else "fail"

    payload = {
        "status": status,
        "state_file": str(state_file),
        "totals": {
            "all_items": len(items),
            "old_items": len(old_items),
            "summarized_old_items": len(summarized_old),
            "old_summary_ratio": round(ratio, 4),
        },
        "thresholds": {
            "old_fidelity_threshold": args.old_fidelity_threshold,
            "min_old_summary_ratio": args.min_old_summary_ratio,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
