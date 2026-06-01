#!/usr/bin/env python3
"""Smoke checks for the deterministic DCN evaluation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _install_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    dml_core = repo_root / "dml_core"
    if str(dml_core) not in sys.path:
        sys.path.insert(0, str(dml_core))


def main() -> int:
    _install_repo_path()
    from daystrom_dml.cognition.evaluation import DCNEvalHarness, smoke_eval_cases

    harness = DCNEvalHarness(clock=lambda: 0.0)
    first = harness.run_suite(smoke_eval_cases(), suite_id="hermes-plugin-dcn-eval-smoke")
    second = harness.run_suite(smoke_eval_cases(), suite_id="hermes-plugin-dcn-eval-smoke")

    assert first.passed is True, first.to_dict()
    assert first.deterministic_hash == second.deterministic_hash, (first.deterministic_hash, second.deterministic_hash)
    assert first.summary["case_count"] == 3, first.summary
    assert first.summary["max_pollution_score"] == 0.0, first.summary
    assert first.summary["blocked_polluting_items"] == 1, first.summary

    rendered = json.dumps(first.to_dict(), sort_keys=True)
    for forbidden in ("raw_transcript", "tool_calls", "prompt_scaffold", "sk-"):
        assert forbidden not in rendered.lower(), forbidden

    print("daystrom_dml dcn eval smoke passed", json.dumps(first.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
