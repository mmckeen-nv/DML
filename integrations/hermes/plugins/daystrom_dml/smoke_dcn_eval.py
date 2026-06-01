#!/usr/bin/env python3
"""Smoke checks for the deterministic DCN evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _install_repo_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    dml_core = repo_root / "dml_core"
    if str(dml_core) not in sys.path:
        sys.path.insert(0, str(dml_core))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic DCN eval smoke and optionally write a sanitized artifact")
    parser.add_argument("--output", help="Write the sanitized eval artifact JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _install_repo_path()
    from daystrom_dml.cognition.evaluation import DCNEvalHarness, smoke_eval_cases

    harness = DCNEvalHarness(clock=lambda: 0.0)
    first = harness.run_suite(smoke_eval_cases(), suite_id="hermes-plugin-dcn-eval-smoke")
    second = harness.run_suite(smoke_eval_cases(), suite_id="hermes-plugin-dcn-eval-smoke")
    artifact = first.artifact()

    assert first.passed is True, first.to_dict()
    assert first.deterministic_hash == second.deterministic_hash, (first.deterministic_hash, second.deterministic_hash)
    assert artifact == second.artifact(), (artifact.get("artifact_hash"), second.artifact().get("artifact_hash"))
    assert artifact["schema_version"] == "dcn-eval-artifact-v1", artifact
    assert artifact["summary"]["case_count"] == 9, artifact["summary"]
    assert artifact["summary"]["max_pollution_score"] == 0.0, artifact["summary"]
    assert artifact["summary"]["blocked_polluting_items"] == 3, artifact["summary"]
    case_ids = set(artifact["coverage"]["case_ids"])
    assert {
        "code_verification_tool_policy",
        "setup_retrieval_semantic",
        "debugging_requires_verification",
        "side_effect_merge_requires_confirmation",
        "metadata_long_horizon_memory",
    } <= case_ids, case_ids
    assert {"code_change", "debugging", "admin"} <= set(artifact["coverage"]["task_types"]), artifact["coverage"]
    assert {"direct", "dml_context"} <= set(artifact["coverage"]["frontier_modes"]), artifact["coverage"]
    assert {"low", "medium"} <= set(artifact["coverage"]["risk_levels"]), artifact["coverage"]
    assert artifact["coverage"]["confirmation_required_cases"] >= 1, artifact["coverage"]
    assert artifact["readiness"]["ready"] is True, artifact["readiness"]
    assert artifact["readiness"]["failed_gates"] == [], artifact["readiness"]
    assert artifact["readiness"]["gate_count"] == 15, artifact["readiness"]

    rendered = json.dumps(artifact, sort_keys=True)
    for forbidden in ("raw_transcript", "tool_calls", "prompt_scaffold", "sk-"):
        assert forbidden not in rendered.lower(), forbidden
    for flag in artifact["redaction_policy"].values():
        assert flag is False, artifact["redaction_policy"]

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("daystrom_dml dcn eval smoke passed", json.dumps({**first.summary, "artifact_hash": artifact["artifact_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
