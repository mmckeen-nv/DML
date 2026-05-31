#!/usr/bin/env python3
"""Audit critical code/docs for DML cookbook retrieval flags."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path("/Users/markmckeen/.openclaw/workspace")

REQUIRED_TOKENS = (
    "--ground-truth-policy",
    "low-confidence",
    "--ground-truth-mode",
    "hybrid",
    "--reform-memory",
    "--no-strict-ground-truth",
)

TARGETS = [
    ROOT / "agentic-framework" / "orchestrator.py",
    ROOT / "skills" / "daystrom-dml" / "SKILL.md",
    ROOT / "skills" / "daystrom-dml" / "DEPLOY_PROD.md",
]


def main() -> int:
    failures: list[str] = []
    for path in TARGETS:
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        missing = [token for token in REQUIRED_TOKENS if token not in text]
        if missing:
            failures.append(f"{path}: missing tokens={missing}")

    if failures:
        print("COOKBOOK_AUDIT=FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("COOKBOOK_AUDIT=PASS")
    for path in TARGETS:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
