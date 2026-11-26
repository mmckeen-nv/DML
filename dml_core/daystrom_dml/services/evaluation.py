"""Reduce verified terminal task events without turning missing evidence into zero."""
from __future__ import annotations

import math
import statistics

TASK_OUTCOME_VERSION = "dml-task-outcome-v1"


def summarize_outcomes(events: list[dict]) -> dict:
    seen = set()
    required = ("input_tokens", "output_tokens", "maintenance_tokens", "latency_ms", "retrieval_ms")
    optional = ("ttft_ms", "repeat_errors", "repeat_opportunities", "contradictions", "factual_outputs",
                "false_memory_claims", "recalled_claims", "store_bytes", "memory_records", "recovery_ms")
    for event in events:
        if event.get("schema_version") != TASK_OUTCOME_VERSION:
            raise ValueError("unsupported task outcome schema")
        key = (event.get("episode_id"), event.get("task_id"))
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            raise ValueError("each terminal task must have a unique episode/task identity")
        seen.add(key)
        if type(event.get("success")) is not bool:
            raise ValueError("success must be a verified boolean")
        for name in required + optional:
            value = event.get(name)
            if value is None and name in optional:
                continue
            if value is None or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if event.get("ttft_ms") is not None and event["ttft_ms"] > event["latency_ms"]:
            raise ValueError("TTFT cannot exceed total task latency")
        for numerator, denominator in (("repeat_errors", "repeat_opportunities"), ("contradictions", "factual_outputs"), ("false_memory_claims", "recalled_claims")):
            if event.get(numerator) is not None and event.get(denominator) is not None and event[numerator] > event[denominator]:
                raise ValueError(f"{numerator} exceeds {denominator}")
    success = sum(event["success"] for event in events)
    tokens = sum(event[name] for event in events for name in ("input_tokens", "output_tokens", "maintenance_tokens"))
    def rate(numerator, denominator):
        if not events or any(event.get(numerator) is None or event.get(denominator) is None for event in events):
            return None
        total = sum(event[denominator] for event in events)
        return sum(event[numerator] for event in events) / total if total else None
    def distribution(name):
        values = sorted(event[name] for event in events if event.get(name) is not None)
        def percentile(q):
            return values[max(0, math.ceil(q * len(values)) - 1)] if values else None
        return {"samples": len(values), "p50": statistics.median(values) if values else None,
                "p95": percentile(.95), "p99": percentile(.99)}
    return {"schema_version": TASK_OUTCOME_VERSION, "attempted_tasks": len(events), "completed_tasks": success,
            "task_success_rate": success / len(events) if events else None,
            "total_tokens": tokens, "tokens_per_completed_task": tokens / success if success else None,
            "repeated_error_rate": rate("repeat_errors", "repeat_opportunities"),
            "contradiction_rate": rate("contradictions", "factual_outputs"),
            "false_memory_rate": rate("false_memory_claims", "recalled_claims"),
            "ttft_ms": distribution("ttft_ms"), "total_latency_ms": distribution("latency_ms"),
            "retrieval_overhead_ms": distribution("retrieval_ms"), "recovery_ms": distribution("recovery_ms"),
            "store_growth": [{"episode_id": e["episode_id"], "task_id": e["task_id"],
                              "store_bytes": e.get("store_bytes"), "memory_records": e.get("memory_records")} for e in events]}
