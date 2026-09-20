"""Failure-inclusive accounting for experimental agent episodes.

Known totals are lower bounds whenever a call is unmeasured. Structural checks
cannot authenticate an imported verdict or turn injected execution into a live
model result. The legacy ``evaluation.summarize_outcomes`` contract is unchanged.
"""
from __future__ import annotations

import math
import statistics

from ..contracts.agent_episode import (
    AgentEpisodeError, QUALITY_PAIRS, TERMINAL_VERSION, canonical_json, decode_json,
    evidence_digest, validate_episode_events, validate_terminal, validate_verifier,
)


def build_terminal(events, verifier, *, status, latency_ms, retrieval_ms,
                   answer=None, usage_unknown=False):
    """Derive usage from raw acknowledged calls, retaining uncertain dispatches.

``usage_unknown`` is set by the parent supervisor when a lost dispatch boundary
could conceal work. An unacknowledged model request independently makes both
input and output usage unknown. No maintenance model is used by this harness.
    """
    validate_episode_events(events, require_terminal=False)
    validate_verifier(verifier)
    if type(usage_unknown) is not bool:
        raise AgentEpisodeError("usage_unknown must be a boolean")
    first = events[0]
    known = {"input": 0, "output": 0}
    unknown = {"input": 0, "output": 0}
    pending = None
    effects_unknown = False
    known_latency = 0
    known_retrieval = 0
    unknown_retrieval = False
    for event in events[1:]:
        kind, payload = event["kind"], event["payload"]
        if kind in ("model_requested", "tool_requested"):
            pending = event
        elif kind in ("model_completed", "model_failed"):
            if payload["latency_ms"] is not None:
                known_latency += payload["latency_ms"]
            for side in known:
                value = payload[f"{side}_token_count"]
                if value is None:
                    unknown[side] += 1
                else:
                    known[side] += value
            pending = None
        elif kind in ("tool_completed", "tool_failed"):
            if payload["latency_ms"] is not None:
                known_latency += payload["latency_ms"]
            if payload["name"] == "retrieve":
                if payload["latency_ms"] is None:
                    unknown_retrieval = True
                else:
                    known_retrieval += payload["latency_ms"]
            if kind == "tool_failed" and payload["effects"] == "unknown":
                effects_unknown = True
            pending = None
    if pending is not None:
        if pending["kind"] == "model_requested":
            unknown["input"] += 1
            unknown["output"] += 1
        elif pending["payload"]["name"] != "retrieve":
            effects_unknown = True
        else:
            unknown_retrieval = True
    if usage_unknown:
        unknown = {side: max(count, 1) for side, count in unknown.items()}
        if status in ("killed", "timeout", "runner_error"):
            effects_unknown = True
        unknown_retrieval = True
    expected_retrieval = None if unknown_retrieval else known_retrieval
    if retrieval_ms != expected_retrieval:
        raise AgentEpisodeError("Retrieval latency differs from raw operation measurements")
    terminal = {
        "schema_version": TERMINAL_VERSION, "episode_id": first["episode_id"],
        "task_id": first["task_id"], "execution_path": first["payload"]["execution_path"],
        "status": status, "success": status == "completed" and verifier["success"],
        "answer": answer, "verifier": verifier, "evidence_digest": evidence_digest(events),
        "input_tokens": None if unknown["input"] else known["input"],
        "output_tokens": None if unknown["output"] else known["output"],
        "maintenance_tokens": 0,
        "known_input_tokens": known["input"], "known_output_tokens": known["output"],
        "unknown_input_calls": unknown["input"], "unknown_output_calls": unknown["output"],
        "usage_unknown": usage_unknown, "effects_unknown": effects_unknown,
        "latency_ms": latency_ms, "retrieval_ms": retrieval_ms, "ttft_ms": None,
    }
    validate_terminal(terminal)
    if latency_ms < known_latency:
        raise AgentEpisodeError("Total latency cannot be below its serial measured operations")
    return decode_json(canonical_json(terminal))


def summarize_episode_outcomes(terminals):
    """Summarize every attempted terminal, without filling missing evidence.

Incomplete cost/quality totals remain null. Separate known sums and coverage
make their lower bounds and missing denominators visible. Mixed execution paths
are rejected so test injection is never pooled with concrete local generation.
    """
    if type(terminals) is not list:
        raise AgentEpisodeError("Expected a list of terminal outcomes")
    seen = set()
    paths = set()
    for terminal in terminals:
        validate_terminal(terminal)
        key = (terminal["episode_id"], terminal["task_id"])
        if key in seen:
            raise AgentEpisodeError("Each attempted task must have exactly one terminal")
        seen.add(key)
        paths.add(terminal["execution_path"])
    if len(paths) > 1:
        raise AgentEpisodeError("Cannot pool injected tests with concrete local execution")
    successes = sum(terminal["success"] for terminal in terminals)
    total_known = sum(terminal["known_input_tokens"] + terminal["known_output_tokens"]
                      + terminal["maintenance_tokens"] for terminal in terminals)
    unknown_tasks = sum(terminal["input_tokens"] is None or terminal["output_tokens"] is None
                        for terminal in terminals)
    exact_tokens = total_known if not unknown_tasks else None

    def distribution(name):
        values = sorted(terminal[name] for terminal in terminals if terminal[name] is not None)
        def percentile(q):
            return values[max(0, math.ceil(q * len(values)) - 1)] if values else None
        return {"samples": len(values), "unknown_tasks": len(terminals) - len(values),
                "p50": statistics.median(values) if values else None,
                "p95": percentile(.95), "p99": percentile(.99)}

    def quality(numerator, denominator):
        measured = [terminal["verifier"] for terminal in terminals
                    if terminal["verifier"][numerator] is not None]
        left = sum(report[numerator] for report in measured)
        right = sum(report[denominator] for report in measured)
        complete = bool(terminals) and len(measured) == len(terminals)
        return {"rate": left / right if complete and right else None,
                "known_numerator": left, "known_denominator": right,
                "measured_tasks": len(measured), "unknown_tasks": len(terminals) - len(measured)}

    result = {
        "schema_version": TERMINAL_VERSION,
        "execution_path": next(iter(paths)) if paths else None,
        "validation_scope": "structural_consistency_only",
        "attempted_tasks": len(terminals), "completed_tasks": successes,
        "task_success_rate": successes / len(terminals) if terminals else None,
        "total_tokens": exact_tokens, "known_total_tokens": total_known,
        "unknown_usage_tasks": unknown_tasks,
        "tokens_per_completed_task": exact_tokens / successes if exact_tokens is not None and successes else None,
        "known_tokens_per_completed_task": total_known / successes if successes else None,
        "unknown_effect_tasks": sum(terminal["effects_unknown"] for terminal in terminals),
        "ttft_ms": distribution("ttft_ms"), "total_latency_ms": distribution("latency_ms"),
        "retrieval_overhead_ms": distribution("retrieval_ms"),
        "status_counts": {status: sum(terminal["status"] == status for terminal in terminals)
                          for status in sorted({terminal["status"] for terminal in terminals})},
    }
    for side in ("input", "output"):
        unknown_count = sum(terminal[f"unknown_{side}_calls"] for terminal in terminals)
        known_count = sum(terminal[f"known_{side}_tokens"] for terminal in terminals)
        result[f"{side}_tokens"] = None if unknown_count else known_count
        result[f"known_{side}_tokens"] = known_count
        result[f"unknown_{side}_calls"] = unknown_count
    for name, pair in zip(("contradiction", "false_memory", "repeated_error"), QUALITY_PAIRS):
        detail = quality(*pair)
        result[f"{name}_rate"] = detail["rate"]
        result[f"{name}_coverage"] = detail
    canonical_json(result)
    return result
