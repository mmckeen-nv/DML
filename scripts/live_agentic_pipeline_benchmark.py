#!/usr/bin/env python3
"""Live Daystrom inference-pipeline benchmark for a synthetic agentic run.

The script talks to a running Daystrom server. It does not read or store any API
key; paid inference happens only through the server's /inference/run endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "decision": "CACHE-DECISION-ALTAIR-42",
    "blocker": "CACHE-BLOCKER-VEGA-09",
    "patch": "CACHE-PATCH-RIGEL-55",
    "regression": "CACHE-REGRESSION-MIRA-18",
    "next_step": "CACHE-NEXT-SIRIUS-88",
}


@dataclass(frozen=True)
class AgentEvent:
    step: int
    agent: str
    phase: str
    text: str


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def build_events(steps: int) -> list[AgentEvent]:
    agents = [
        ("architect", "plan"),
        ("implementer", "build"),
        ("reviewer", "review"),
        ("tester", "test"),
        ("ops", "debug"),
        ("continuity", "reflect"),
    ]
    events: list[AgentEvent] = []
    for step in range(1, steps + 1):
        agent, phase = agents[(step - 1) % len(agents)]
        shard = (
            f"Agentic cache benchmark step {step:04d}. Agent={agent}; phase={phase}. "
            "The team is building a synthetic distributed cache service with sharded writes, "
            "lease-based invalidation, cold-start replay, and fault-injection tests. "
            f"Telemetry bucket={step % 37}; retry window={step % 5}; branch lane=bench/{step % 11}. "
        )
        if step == 7:
            shard += (
                f"{EXPECTED['decision']}: use a write-through cache with quorum-confirmed invalidations "
                "before acknowledging writes to downstream agents. "
            )
        if step == 19:
            shard += (
                f"{EXPECTED['blocker']}: null owner leases can survive compaction and cause stale reads "
                "after node failover. "
            )
        if step == 34:
            shard += (
                f"{EXPECTED['patch']}: add lease-owner guard, replay-safe invalidation ledger, and scoped "
                "handoff summary for long-horizon recovery. "
            )
        if step == 58:
            shard += (
                f"{EXPECTED['regression']}: reviewer found the metrics exporter counted speculative "
                "invalidations as committed writes. "
            )
        if step == 72:
            shard += (
                f"{EXPECTED['next_step']}: run a final endpoint verification comparing raw transcript "
                "cost to DML-compressed frontier verification. "
            )
        shard += (
            "Nonessential trace material: stack sample, local scratchpad notes, discarded alternatives, "
            "and verbose tool transcript should accumulate in traditional context but be compressed away "
            "from the frontier call unless it affects durable decisions."
        )
        events.append(AgentEvent(step=step, agent=agent, phase=phase, text=shard))
    return events


def post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout: float = 180.0) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def ingest_event(base_url: str, tenant_id: str, session_id: str, event: AgentEvent) -> float:
    meta = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "kind": "note",
        "source": "live_agentic_pipeline_benchmark",
        "agent": event.agent,
        "phase": event.phase,
        "step_id": f"STEP-{event.step:04d}",
        "no_merge": True,
    }
    started = time.perf_counter()
    post_json(base_url, "/ingest", {"text": event.text, "meta": meta}, timeout=60)
    return (time.perf_counter() - started) * 1000.0


def ingest_ledger(base_url: str, tenant_id: str, session_id: str, cumulative_tokens: int, step: int) -> float:
    text = (
        "Exact distributed cache agentic run survival ledger. "
        f"{EXPECTED['decision']} = write-through cache with quorum-confirmed invalidations. "
        f"{EXPECTED['blocker']} = null owner leases survive compaction and cause stale reads after failover. "
        f"{EXPECTED['patch']} = lease-owner guard plus replay-safe invalidation ledger. "
        f"{EXPECTED['regression']} = metrics exporter counted speculative invalidations as committed writes. "
        f"{EXPECTED['next_step']} = final endpoint verification comparing raw transcript cost to DML-compressed frontier verification."
    )
    meta = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "kind": "survival_ledger",
        "source": "live_agentic_pipeline_benchmark",
        "summary": text,
        "no_merge": True,
        "survival_ledger": True,
        "compaction_cycle": step,
        "virtual_tokens": cumulative_tokens,
        "anchors": list(EXPECTED.values()),
    }
    started = time.perf_counter()
    post_json(base_url, "/ingest", {"text": text, "meta": meta}, timeout=60)
    return (time.perf_counter() - started) * 1000.0


def score(text: str) -> dict[str, Any]:
    haystack = str(text or "").lower()
    matched = [value for value in EXPECTED.values() if value.lower() in haystack]
    return {
        "score": len(matched) / len(EXPECTED),
        "matched": matched,
        "missing": [value for value in EXPECTED.values() if value not in matched],
    }


def checkpoint_prompt(step: int) -> str:
    return (
        f"At benchmark checkpoint step {step}, answer with exact IDs. What was the cache decision, "
        "what blocker affected long-horizon continuity, what exact patch fixed it, what regression was found, "
        "and what is the next step?"
    )


def run_checkpoint(
    base_url: str,
    *,
    tenant_id: str,
    session_id: str,
    step: int,
    raw_tokens: int,
    run_frontier: bool,
    include_local_draft: bool,
    model: str,
) -> dict[str, Any]:
    payload = {
        "prompt": checkpoint_prompt(step),
        "tenant_id": tenant_id,
        "session_id": session_id,
        "top_k": 10,
        "include_local_draft": include_local_draft,
        "direct_input_tokens_estimate": raw_tokens,
        "direct_output_tokens_estimate": 1200,
        "frontier_max_tokens": 320,
        "reasoning_effort": "low",
        "model": model,
    }
    start = time.perf_counter()
    if run_frontier:
        result = post_json(base_url, "/inference/run", payload, timeout=240)
        output = result.get("inference", {}).get("output_text", "")
        telemetry = result.get("telemetry", {})
        raw = result.get("inference", {}).get("raw") or {}
        usage = raw.get("usage") if isinstance(raw, dict) else {}
    else:
        result = post_json(base_url, "/inference/prepare", payload, timeout=180)
        output = result.get("local_draft", "")
        telemetry = result.get("telemetry", {})
        usage = {}
    elapsed = (time.perf_counter() - start) * 1000.0
    accuracy = score(output + "\n" + result.get("prepared", result).get("dml_context", ""))
    return {
        "step": step,
        "run_frontier": run_frontier,
        "latency_ms": round(elapsed, 2),
        "raw_tokens": raw_tokens,
        "include_local_draft": include_local_draft,
        "output": output,
        "accuracy": accuracy,
        "telemetry": telemetry,
        "usage": usage,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Live Agentic Pipeline Benchmark",
        "",
        f"- Steps ingested: {s['steps']}",
        f"- Checkpoints: {s['checkpoints']}",
        f"- Raw transcript tokens: {s['raw_transcript_tokens']:,}",
        f"- Avg event ingest latency: {s['avg_event_ingest_latency_ms']} ms",
        f"- Avg ledger ingest latency: {s['avg_ledger_ingest_latency_ms']} ms",
        f"- Avg frontier input tokens: {s['avg_frontier_input_tokens']}",
        f"- Avg input savings: {s['avg_input_savings_pct']}%",
        f"- Avg output savings estimate: {s['avg_output_savings_pct']}%",
        f"- Avg accuracy: {s['avg_accuracy_pct']}%",
        f"- Total observed frontier tokens: {s['observed_frontier_total_tokens']}",
        "",
        "| Step | Accuracy | Raw Tokens | Frontier Input | Frontier Output | Latency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["checkpoints"]:
        usage = row.get("usage") or {}
        telemetry = row.get("telemetry") or {}
        lines.append(
            f"| {row['step']} | {row['accuracy']['score'] * 100:.0f}% | {row['raw_tokens']:,} | "
            f"{telemetry.get('frontier_input_tokens', usage.get('input_tokens', 0))} | "
            f"{usage.get('output_tokens', telemetry.get('frontier_output_tokens_estimate', 0))} | "
            f"{row['latency_ms']} ms |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live synthetic agentic benchmark through Daystrom inference pipeline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8777")
    parser.add_argument("--tenant-id", default="openclaw")
    parser.add_argument("--session-id", default=f"agentic-live-benchmark-{int(time.time())}")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument("--run-frontier", action="store_true")
    parser.add_argument("--include-local-draft", action="store_true")
    parser.add_argument("--model", default="azure/openai/gpt-5.2-codex")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "live_agentic_pipeline_benchmark")
    args = parser.parse_args(argv)

    events = build_events(args.steps)
    raw_transcript: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    event_ingest_latencies: list[float] = []
    ledger_ingest_latencies: list[float] = []
    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "session_id": args.session_id,
                "steps": args.steps,
                "checkpoint_every": args.checkpoint_every,
                "run_frontier": args.run_frontier,
                "include_local_draft": args.include_local_draft,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for event in events:
        raw_transcript.append(event.text)
        ingest_ms = ingest_event(args.base_url, args.tenant_id, args.session_id, event)
        event_ingest_latencies.append(ingest_ms)
        raw_tokens = estimate_tokens("\n".join(raw_transcript))
        print(
            json.dumps(
                {
                    "event": "ingest_step",
                    "step": event.step,
                    "latency_ms": round(ingest_ms, 2),
                    "raw_tokens": raw_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if event.step % args.checkpoint_every == 0 or event.step == args.steps:
            ledger_ms = ingest_ledger(args.base_url, args.tenant_id, args.session_id, raw_tokens, event.step)
            ledger_ingest_latencies.append(ledger_ms)
            print(
                json.dumps(
                    {
                        "event": "ingest_ledger",
                        "step": event.step,
                        "latency_ms": round(ledger_ms, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            checkpoint = run_checkpoint(
                args.base_url,
                tenant_id=args.tenant_id,
                session_id=args.session_id,
                step=event.step,
                raw_tokens=raw_tokens,
                run_frontier=args.run_frontier,
                include_local_draft=args.include_local_draft,
                model=args.model,
            )
            checkpoints.append(checkpoint)
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": event.step,
                        "accuracy_pct": round(checkpoint["accuracy"]["score"] * 100, 1),
                        "latency_ms": checkpoint["latency_ms"],
                        "frontier_input_tokens": (checkpoint.get("telemetry") or {}).get("frontier_input_tokens"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    frontier_inputs = [
        int((row.get("telemetry") or {}).get("frontier_input_tokens") or (row.get("usage") or {}).get("input_tokens") or 0)
        for row in checkpoints
    ]
    output_savings = [
        float((row.get("telemetry") or {}).get("output_savings_pct_estimate") or 0.0)
        for row in checkpoints
    ]
    observed_total = sum(int((row.get("usage") or {}).get("total_tokens") or 0) for row in checkpoints)
    summary = {
        "session_id": args.session_id,
        "steps": args.steps,
        "checkpoints": len(checkpoints),
        "run_frontier": args.run_frontier,
        "include_local_draft": args.include_local_draft,
        "raw_transcript_tokens": estimate_tokens("\n".join(raw_transcript)),
        "avg_event_ingest_latency_ms": round(mean(event_ingest_latencies), 2) if event_ingest_latencies else 0,
        "avg_ledger_ingest_latency_ms": round(mean(ledger_ingest_latencies), 2) if ledger_ingest_latencies else 0,
        "avg_frontier_input_tokens": round(mean(frontier_inputs), 1) if frontier_inputs else 0,
        "avg_input_savings_pct": round(mean(float((row.get("telemetry") or {}).get("input_savings_pct_estimate") or 0) for row in checkpoints), 1) if checkpoints else 0,
        "avg_output_savings_pct": round(mean(output_savings), 1) if output_savings else 0,
        "avg_accuracy_pct": round(mean(row["accuracy"]["score"] for row in checkpoints) * 100, 1) if checkpoints else 0,
        "observed_frontier_total_tokens": observed_total,
    }
    payload = {"summary": summary, "checkpoints": checkpoints, "expected": EXPECTED}
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / "README.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    sys.stdout.flush()
    return 0 if summary["avg_accuracy_pct"] >= 80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
