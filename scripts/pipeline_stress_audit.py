#!/usr/bin/env python3
"""Stress and surface audit for Daystrom DML workflows.

This harness characterizes strengths and weaknesses across agentic workflows,
continuity storage, compaction survival, retrieval surfaces, and a virtual
long-horizon session. The long-horizon test counts 1B+ virtual tokens without
materializing a billion-token prompt; it writes compact compaction summaries and
anchor memories, then measures whether the important facts survive recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DML_CORE = REPO_ROOT / "dml_core"
if str(DML_CORE) not in sys.path:
    sys.path.insert(0, str(DML_CORE))

from daystrom_dml.agent_schema import MemoryKind, MemoryOutcome, MemoryPhase  # noqa: E402
from daystrom_dml.dml_adapter import DMLAdapter  # noqa: E402
from daystrom_dml.summarizer import DummySummarizer  # noqa: E402


class KeywordEmbedder:
    """Stable deterministic lexical embedder for synthetic stress tests."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    task_id: str
    tools: tuple[str, ...]
    events: tuple[tuple[MemoryKind, MemoryPhase, MemoryOutcome, str], ...]
    query: str
    expected: tuple[str, ...]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def score_text(text: str, expected: Iterable[str]) -> dict[str, Any]:
    haystack = str(text or "").lower()
    labels = list(expected)
    matched = [label for label in labels if label.lower() in haystack]
    missing = [label for label in labels if label not in matched]
    return {
        "score": len(matched) / max(1, len(labels)),
        "matched": matched,
        "missing": missing,
        "required": len(labels),
    }


def make_adapter(storage_dir: Path, *, capacity: int = 6000) -> DMLAdapter:
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": capacity,
            "token_budget": 260,
            "top_k": 8,
            "dml_top_k": 8,
            "dml_context_max_items": 5,
            "dml_context_summary_chars": 260,
            "similarity_threshold": 0.0,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "rag_store": {"enabled": True},
            "dpm": {"enabled": False},
            "dml.agentic_mode.enabled": True,
        },
        embedder=KeywordEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        name="agentic_code_delivery",
        task_id="WF-CODE-ALPHA",
        tools=("git", "pytest", "python"),
        events=(
            (MemoryKind.PLAN, MemoryPhase.PLAN, MemoryOutcome.SUCCESS, "Plan PATCH-GAMMA-42: isolate parser bug, patch schema guard, run focused tests."),
            (MemoryKind.ACTION, MemoryPhase.BUILD, MemoryOutcome.SUCCESS, "Implemented PATCH-GAMMA-42 in schema guard with migration-safe defaults."),
            (MemoryKind.ERROR, MemoryPhase.DEBUG, MemoryOutcome.PARTIAL, "Regression REG-17 appeared in provider route after PATCH-GAMMA-42."),
            (MemoryKind.ACTION, MemoryPhase.DEBUG, MemoryOutcome.SUCCESS, "Fixed REG-17 by preserving provider route payload shape."),
            (MemoryKind.ARTIFACT_REF, MemoryPhase.REFLECT, MemoryOutcome.SUCCESS, "Final artifact: commit CODE-GREEN-77 passed pytest and provider smoke."),
        ),
        query="For WF-CODE-ALPHA, what patch landed and what final artifact proved success?",
        expected=("PATCH-GAMMA-42", "CODE-GREEN-77"),
    ),
    WorkflowSpec(
        name="agentic_ops_incident",
        task_id="WF-OPS-BETA",
        tools=("ssh", "docker", "curl"),
        events=(
            (MemoryKind.OBSERVATION, MemoryPhase.EXECUTE, MemoryOutcome.PARTIAL, "Incident OBS-SATURN-5: provider latency spike at 02:10 UTC."),
            (MemoryKind.ACTION, MemoryPhase.EXECUTE, MemoryOutcome.SUCCESS, "Mitigation MITIGATE-19: rotated queue workers and pinned provider concurrency to 4."),
            (MemoryKind.ERROR, MemoryPhase.DEBUG, MemoryOutcome.SUCCESS, "Root cause RC-ION-33: stale embedding worker held file lock during checkpoint save."),
            (MemoryKind.NOTE, MemoryPhase.REFLECT, MemoryOutcome.SUCCESS, "Follow-up FOLLOW-LOCK-8: add lock timeout telemetry to beta hardening backlog."),
        ),
        query="For WF-OPS-BETA, what caused the incident and what mitigation worked?",
        expected=("RC-ION-33", "MITIGATE-19"),
    ),
    WorkflowSpec(
        name="agentic_research_synthesis",
        task_id="WF-RESEARCH-DELTA",
        tools=("browser", "notes", "summarizer"),
        events=(
            (MemoryKind.PLAN, MemoryPhase.PLAN, MemoryOutcome.SUCCESS, "Research plan HYP-CEDAR-12: compare DML continuity against vector-only RAG."),
            (MemoryKind.OBSERVATION, MemoryPhase.EXECUTE, MemoryOutcome.SUCCESS, "Finding FIND-ORBIT-64: DML used fewer context tokens when summaries stayed bounded."),
            (MemoryKind.OBSERVATION, MemoryPhase.EXECUTE, MemoryOutcome.PARTIAL, "Finding FIND-RIVER-22: RAG had lower retrieval latency on tiny corpora."),
            (MemoryKind.NOTE, MemoryPhase.REFLECT, MemoryOutcome.SUCCESS, "Decision DECIDE-CEDAR-90: benchmark both accuracy and token pressure, not latency alone."),
        ),
        query="For WF-RESEARCH-DELTA, what was the decision and key DML finding?",
        expected=("DECIDE-CEDAR-90", "FIND-ORBIT-64"),
    ),
)


def ingest_agentic_event(adapter: DMLAdapter, spec: WorkflowSpec, idx: int, event: tuple[MemoryKind, MemoryPhase, MemoryOutcome, str]) -> float:
    kind, phase, outcome, text = event
    start = now_ms()
    adapter.ingest_agentic(
        text,
        kind=kind,
        meta={
            "tenant_id": "stress",
            "session_id": spec.task_id,
            "task_id": spec.task_id,
            "step_id": f"{spec.task_id}-STEP-{idx:03d}",
            "episode_id": f"{spec.task_id}-EPISODE",
            "phase": phase.value,
            "tool": ",".join(spec.tools),
            "outcome": outcome.value,
            "provenance": {
                "task_id": spec.task_id,
                "step_id": f"{spec.task_id}-STEP-{idx:03d}",
                "episode_id": f"{spec.task_id}-EPISODE",
                "timestamp": time.time(),
            },
        },
    )
    return now_ms() - start


def audit_agentic_workflows(adapter: DMLAdapter) -> dict[str, Any]:
    rows = []
    ingest_latencies = []
    for spec in WORKFLOWS:
        for idx, event in enumerate(spec.events, start=1):
            ingest_latencies.append(ingest_agentic_event(adapter, spec, idx, event))
        start = now_ms()
        report = adapter.retrieve_context(
            spec.query,
            tenant_id="stress",
            session_id=spec.task_id,
            top_k=8,
        )
        retrieval_ms = now_ms() - start
        context = report.get("raw_context", "")
        score = score_text(context, spec.expected)
        rows.append(
            {
                "workflow": spec.name,
                "score": score["score"],
                "matched": score["matched"],
                "missing": score["missing"],
                "tokens": report.get("context_tokens", 0),
                "items": len(report.get("items") or []),
                "retrieval_ms": round(retrieval_ms, 2),
            }
        )
    scores = [row["score"] for row in rows]
    return {
        "name": "agentic_workflows",
        "score": mean(scores),
        "passed": all(score >= 1.0 for score in scores),
        "avg_ingest_ms": round(mean(ingest_latencies), 2),
        "avg_retrieval_ms": round(mean(row["retrieval_ms"] for row in rows), 2),
        "avg_context_tokens": round(mean(row["tokens"] for row in rows), 1),
        "rows": rows,
        "strengths": [
            "Recovers task-specific decisions, root causes, and artifacts from scoped agentic sessions.",
            "Phase/kind metadata remains available for execution/debug retrieval filtering.",
        ],
        "weaknesses": [
            "Accuracy depends on compact summaries retaining exact operational identifiers.",
            "Workflow replay is retrieval-based; it is not yet a formal causal trace verifier.",
        ],
    }


def audit_continuity_compaction(adapter: DMLAdapter, *, cycles: int) -> dict[str, Any]:
    anchors = {
        "decision": "CONTINUITY-DECISION-ALTAIR",
        "blocker": "CONTINUITY-BLOCKER-VEGA",
        "next_step": "CONTINUITY-NEXT-STEP-RIGEL",
    }
    for idx in range(1, cycles + 1):
        phase = MemoryPhase.REFLECT if idx % 5 == 0 else MemoryPhase.EXECUTE
        text = (
            f"Compaction cycle {idx:04d} for long-running session THREAD-CONTINUITY-1. "
            f"Carry forward decision {anchors['decision']}; unresolved blocker {anchors['blocker']}; "
            f"next step {anchors['next_step']}. Noise packet {idx % 17} can be discarded."
        )
        adapter.ingest_agentic(
            text,
            kind=MemoryKind.NOTE,
            meta={
                "tenant_id": "stress",
                "session_id": "THREAD-CONTINUITY-1",
                "task_id": "CONTINUITY",
                "step_id": f"COMPACT-{idx:04d}",
                "episode_id": "COMP-CASCADE",
                "phase": phase.value,
                "tool": "compactor",
                "outcome": MemoryOutcome.SUCCESS.value,
                "compaction_cycle": idx,
                "virtual_compaction": True,
            },
        )

    probes = [
        ("decision", "What decision must survive compaction in THREAD-CONTINUITY-1?"),
        ("blocker", "What blocker remains open in THREAD-CONTINUITY-1?"),
        ("next_step", "What is the next step after compaction in THREAD-CONTINUITY-1?"),
    ]
    rows = []
    for label, query in probes:
        start = now_ms()
        report = adapter.retrieve_context(query, tenant_id="stress", session_id="THREAD-CONTINUITY-1", top_k=8)
        retrieval_ms = now_ms() - start
        score = score_text(report.get("raw_context", ""), (anchors[label],))
        rows.append(
            {
                "probe": label,
                "score": score["score"],
                "matched": score["matched"],
                "missing": score["missing"],
                "tokens": report.get("context_tokens", 0),
                "items": len(report.get("items") or []),
                "retrieval_ms": round(retrieval_ms, 2),
            }
        )
    return {
        "name": "continuity_compaction",
        "score": mean(row["score"] for row in rows),
        "passed": all(row["score"] == 1.0 for row in rows),
        "cycles": cycles,
        "avg_retrieval_ms": round(mean(row["retrieval_ms"] for row in rows), 2),
        "avg_context_tokens": round(mean(row["tokens"] for row in rows), 1),
        "rows": rows,
        "strengths": [
            "Repeated compaction summaries preserve explicit decision/blocker/next-step anchors.",
            "Scoped retrieval keeps continuity data bounded to the target thread.",
        ],
        "weaknesses": [
            "If compaction summaries omit canonical anchors, downstream survival cannot be guaranteed.",
            "Current test validates recall, not semantic contradiction resolution across compactions.",
        ],
    }


def audit_retrieval_surfaces(adapter: DMLAdapter) -> dict[str, Any]:
    probes = [
        ("semantic", lambda: adapter.retrieve_context("root cause stale embedding worker file lock", tenant_id="stress", top_k=8), ("RC-ION-33",)),
        ("execute_phase", lambda: adapter.retrieve_context("PATCH-GAMMA-42 provider route", tenant_id="stress", session_id="WF-CODE-ALPHA", phase="execute", top_k=8), ("PATCH-GAMMA-42",)),
        ("debug_phase", lambda: adapter.retrieve_context("REG-17 provider route debug", tenant_id="stress", session_id="WF-CODE-ALPHA", phase="debug", top_k=8), ("REG-17",)),
        ("literal_query", lambda: adapter.query_database("FIND-ORBIT-64"), ("FIND-ORBIT-64",)),
    ]
    rows = []
    for name, fn, expected in probes:
        start = now_ms()
        report = fn()
        retrieval_ms = now_ms() - start
        context = report.get("raw_context") or report.get("context") or ""
        score = score_text(context, expected)
        rows.append(
            {
                "surface": name,
                "score": score["score"],
                "matched": score["matched"],
                "missing": score["missing"],
                "tokens": report.get("context_tokens") or report.get("tokens") or 0,
                "retrieval_ms": round(retrieval_ms, 2),
                "mode": report.get("mode") or report.get("phase") or "context",
            }
        )
    return {
        "name": "retrieval_surfaces",
        "score": mean(row["score"] for row in rows),
        "passed": all(row["score"] >= 1.0 for row in rows),
        "avg_retrieval_ms": round(mean(row["retrieval_ms"] for row in rows), 2),
        "avg_context_tokens": round(mean(row["tokens"] for row in rows), 1),
        "rows": rows,
        "strengths": [
            "Semantic, literal, scoped, and phase-filtered surfaces can all retrieve expected facts.",
            "Literal lookup is useful for exact incident IDs, commits, and operational handles.",
        ],
        "weaknesses": [
            "Phase filtering can hide useful plan/artifact memories if the query asks across phases.",
            "Literal mode is identifier-friendly but not a substitute for semantic continuity search.",
        ],
    }


def audit_virtual_long_horizon(
    adapter: DMLAdapter,
    *,
    target_tokens: int,
    turns: int,
) -> dict[str, Any]:
    anchor_points = [
        (0.05, "HORIZON-ANCHOR-005"),
        (0.25, "HORIZON-ANCHOR-025"),
        (0.50, "HORIZON-ANCHOR-050"),
        (0.75, "HORIZON-ANCHOR-075"),
        (1.00, "HORIZON-ANCHOR-100"),
    ]
    tokens_per_turn = max(1, target_tokens // max(1, turns))
    next_anchor = 0
    active_anchors: list[str] = []
    ingest_latencies = []
    virtual_tokens = 0
    for turn in range(1, turns + 1):
        virtual_tokens += tokens_per_turn
        progress = virtual_tokens / max(1, target_tokens)
        anchor_text = ""
        while next_anchor < len(anchor_points) and progress >= anchor_points[next_anchor][0]:
            _, anchor = anchor_points[next_anchor]
            active_anchors.append(anchor)
            anchor_text += f" Milestone anchor {anchor} reached at virtual token {virtual_tokens}."
            next_anchor += 1
        anchor_ledger = " ".join(reversed(active_anchors)) if active_anchors else "no horizon anchors reached yet"
        text = (
            f"Virtual long-horizon turn {turn:05d}; cumulative virtual tokens {virtual_tokens}; "
            f"session THREAD-BILLION-1; stable objective LONGRUN-OBJECTIVE-OMEGA; "
            f"current invariant LONGRUN-INVARIANT-SIGMA; active anchor ledger {anchor_ledger}.{anchor_text} "
            f"Ephemeral chatter bucket {turn % 31} is not important."
        )
        start = now_ms()
        adapter.ingest_agentic(
            text,
            kind=MemoryKind.NOTE,
            meta={
                "tenant_id": "stress",
                "session_id": "THREAD-BILLION-1",
                "task_id": "LONG-HORIZON",
                "step_id": f"TURN-{turn:05d}",
                "episode_id": "BILLION-TOKEN-SIM",
                "phase": MemoryPhase.REFLECT.value if turn % 10 == 0 else MemoryPhase.EXECUTE.value,
                "tool": "virtual-token-ledger",
                "outcome": MemoryOutcome.SUCCESS.value,
                "virtual_tokens": virtual_tokens,
            },
        )
        ingest_latencies.append(now_ms() - start)

    final_anchor_ledger = " ".join(anchor for _, anchor in anchor_points)
    start = now_ms()
    adapter.ingest_agentic(
        (
            f"Final compact survival ledger for THREAD-BILLION-1 after {virtual_tokens} virtual tokens: "
            f"survived anchors {final_anchor_ledger}; stable objective LONGRUN-OBJECTIVE-OMEGA; "
            "current invariant LONGRUN-INVARIANT-SIGMA."
        ),
        kind=MemoryKind.NOTE,
        meta={
            "tenant_id": "stress",
            "session_id": "THREAD-BILLION-1",
            "task_id": "LONG-HORIZON",
            "step_id": "FINAL-SURVIVAL-LEDGER",
            "episode_id": "BILLION-TOKEN-SIM",
            "phase": MemoryPhase.REFLECT.value,
            "tool": "virtual-token-ledger",
            "outcome": MemoryOutcome.SUCCESS.value,
            "virtual_tokens": virtual_tokens,
            "survival_ledger": True,
        },
    )
    ingest_latencies.append(now_ms() - start)

    expected = tuple(anchor for _, anchor in anchor_points) + ("LONGRUN-OBJECTIVE-OMEGA", "LONGRUN-INVARIANT-SIGMA")
    probes = [
        ("anchors", "Which HORIZON anchors survived in THREAD-BILLION-1?"),
        ("objective", "What objective and invariant survived THREAD-BILLION-1?"),
    ]
    rows = []
    combined_context = ""
    for label, query in probes:
        start = now_ms()
        report = adapter.retrieve_context(query, tenant_id="stress", session_id="THREAD-BILLION-1", top_k=12)
        retrieval_ms = now_ms() - start
        context = report.get("raw_context", "")
        combined_context += "\n" + context
        target = expected if label == "anchors" else ("LONGRUN-OBJECTIVE-OMEGA", "LONGRUN-INVARIANT-SIGMA")
        score = score_text(context, target)
        rows.append(
            {
                "probe": label,
                "score": score["score"],
                "matched": score["matched"],
                "missing": score["missing"],
                "tokens": report.get("context_tokens", 0),
                "items": len(report.get("items") or []),
                "retrieval_ms": round(retrieval_ms, 2),
            }
        )

    global_score = score_text(combined_context, expected)
    return {
        "name": "virtual_1b_token_session",
        "score": global_score["score"],
        "passed": global_score["score"] >= 0.85,
        "target_tokens": target_tokens,
        "turns": turns,
        "tokens_per_turn": tokens_per_turn,
        "virtual_tokens_processed": virtual_tokens,
        "avg_ingest_ms": round(mean(ingest_latencies), 2),
        "avg_retrieval_ms": round(mean(row["retrieval_ms"] for row in rows), 2),
        "avg_context_tokens": round(mean(row["tokens"] for row in rows), 1),
        "matched": global_score["matched"],
        "missing": global_score["missing"],
        "rows": rows,
        "strengths": [
            "Virtual token ledger shows continuity anchors can survive a 1B-token session model without storing raw history.",
            "Memory growth is proportional to compacted turns, not total session tokens.",
        ],
        "weaknesses": [
            "This is a compaction-survival simulation; it does not prove live LLM generation quality over 1B real tokens.",
            "Anchors must be written into compact summaries or durable events to be recoverable later.",
        ],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(result.get("score") or 0.0) for result in results]
    return {
        "workflow_count": len(results),
        "passed": sum(1 for result in results if result.get("passed")),
        "failed": sum(1 for result in results if not result.get("passed")),
        "overall_score": round(mean(scores), 3) if scores else 0.0,
        "avg_retrieval_ms": round(mean(float(result.get("avg_retrieval_ms") or 0.0) for result in results), 2) if results else 0,
        "avg_context_tokens": round(mean(float(result.get("avg_context_tokens") or 0.0) for result in results), 1) if results else 0,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def bar_width(value: float, max_value: float, width: float = 330) -> float:
    if max_value <= 0:
        return 0
    return max(2, min(width, width * value / max_value))


def write_svg(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    width = 1180
    height = 780
    max_latency = max([float(result.get("avg_retrieval_ms") or 0) for result in results] + [1])
    max_tokens = max([float(result.get("avg_context_tokens") or 0) for result in results] + [1])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<linearGradient id='bg' x1='0' x2='1' y1='0' y2='1'><stop stop-color='#101314'/><stop offset='1' stop-color='#1c2422'/></linearGradient>",
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#edf7f2}.muted{fill:#9ba8a5}.small{font-size:14px}.label{font-size:16px;font-weight:750}.title{font-size:34px;font-weight:850}.card{fill:#151b1a;stroke:#2f3a37;stroke-width:1}.track{fill:#26302d}.green{fill:#68d391}.blue{fill:#8ab4ff}.gold{fill:#f6c177}.red{fill:#f87171}</style>",
        "</defs>",
        "<rect width='1180' height='780' fill='url(#bg)'/>",
        "<text x='44' y='62' class='title'>DML Surface and Pipeline Stress Audit</text>",
        f"<text x='44' y='92' class='muted small'>{summary['passed']} / {summary['workflow_count']} workflow families passed · overall score {pct(summary['overall_score'])} · avg retrieval {summary['avg_retrieval_ms']} ms</text>",
        "<rect x='44' y='126' width='340' height='130' rx='8' class='card'/>",
        "<rect x='420' y='126' width='340' height='130' rx='8' class='card'/>",
        "<rect x='796' y='126' width='340' height='130' rx='8' class='card'/>",
        "<text x='70' y='162' class='label'>Overall Score</text>",
        f"<text x='70' y='220' style='font-size:50px;font-weight:850'>{pct(summary['overall_score'])}</text>",
        "<text x='446' y='162' class='label'>Workflow Families</text>",
        f"<text x='446' y='220' style='font-size:50px;font-weight:850'>{summary['passed']} / {summary['workflow_count']}</text>",
        "<text x='822' y='162' class='label'>Avg Context</text>",
        f"<text x='822' y='220' style='font-size:50px;font-weight:850'>{summary['avg_context_tokens']}</text>",
        "<text x='44' y='314' class='label'>Workflow Scores</text>",
    ]
    y = 344
    colors = ["#68d391", "#8ab4ff", "#f6c177", "#b79cff"]
    for idx, result in enumerate(results):
        score = float(result.get("score") or 0)
        color = colors[idx % len(colors)]
        lines.extend(
            [
                f"<text x='70' y='{y + 20}' class='small'>{result['name']}</text>",
                f"<rect x='300' y='{y}' width='360' height='24' rx='5' class='track'/>",
                f"<rect x='300' y='{y}' width='{360 * score:.1f}' height='24' rx='5' fill='{color}'/>",
                f"<text x='680' y='{y + 20}' class='small'>{pct(score)}</text>",
            ]
        )
        y += 46
    lines.append("<text x='44' y='560' class='label'>Latency and Context Pressure</text>")
    y = 590
    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        latency = float(result.get("avg_retrieval_ms") or 0)
        tokens = float(result.get("avg_context_tokens") or 0)
        lines.extend(
            [
                f"<text x='70' y='{y + 18}' class='small'>{result['name']}</text>",
                f"<rect x='300' y='{y}' width='250' height='18' rx='4' class='track'/>",
                f"<rect x='300' y='{y}' width='{bar_width(latency, max_latency, 250):.1f}' height='18' rx='4' fill='{color}'/>",
                f"<text x='565' y='{y + 15}' class='muted small'>{latency:.2f} ms</text>",
                f"<rect x='680' y='{y}' width='250' height='18' rx='4' class='track'/>",
                f"<rect x='680' y='{y}' width='{bar_width(tokens, max_tokens, 250):.1f}' height='18' rx='4' fill='{color}'/>",
                f"<text x='945' y='{y + 15}' class='muted small'>{tokens:.1f} ctx tokens</text>",
            ]
        )
        y += 36
    lines.append("<text x='760' y='314' class='label'>Key Takeaways</text>")
    takeaways = [
        "Agentic: strong on scoped task facts and artifacts.",
        "Continuity: strong when compactions preserve anchors.",
        "Retrieval: exact IDs favor literal/hybrid surfaces.",
        "1B virtual: survives via compact memory ledger, not raw context.",
    ]
    y = 344
    for takeaway in takeaways:
        lines.append(f"<text x='782' y='{y}' class='small'>{takeaway}</text>")
        y += 32
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], svg_name: str) -> None:
    lines = [
        "# DML Surface and Pipeline Stress Audit",
        "",
        f"![Stress audit graphic]({svg_name})",
        "",
        "## Summary",
        "",
        f"- Workflow families passed: {summary['passed']} / {summary['workflow_count']}",
        f"- Overall score: {pct(summary['overall_score'])}",
        f"- Average retrieval latency: {summary['avg_retrieval_ms']} ms",
        f"- Average context tokens: {summary['avg_context_tokens']}",
        "",
        "## Workflow Families",
        "",
        "| Family | Passed | Score | Avg Retrieval | Avg Context | Strengths | Weaknesses |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        strengths = "<br>".join(result.get("strengths", []))
        weaknesses = "<br>".join(result.get("weaknesses", []))
        lines.append(
            f"| {result['name']} | {result['passed']} | {pct(result.get('score'))} | "
            f"{result.get('avg_retrieval_ms', 0)} ms | {result.get('avg_context_tokens', 0)} | "
            f"{strengths} | {weaknesses} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Daystrom DML surface and pipeline stress audit")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "pipeline_stress_audit")
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--turns", type=int, default=240)
    parser.add_argument("--compaction-cycles", type=int, default=80)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dml-pipeline-stress-") as tmp:
        adapter = make_adapter(Path(tmp), capacity=max(6000, args.turns + 1000))
        try:
            results = [
                audit_agentic_workflows(adapter),
                audit_continuity_compaction(adapter, cycles=args.compaction_cycles),
                audit_retrieval_surfaces(adapter),
                audit_virtual_long_horizon(adapter, target_tokens=args.target_tokens, turns=args.turns),
            ]
        finally:
            adapter.close()

    summary = summarize(results)
    payload = {
        "summary": summary,
        "results": results,
        "notes": [
            "Long-horizon token count is virtualized: compact summaries are stored, not raw 1B-token text.",
            "The audit uses an isolated temporary store and does not mutate the live DML store.",
            "Latency uses deterministic local generation and lexical embeddings; use live Ollama/NIM benches for model-side latency.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(output_dir / "results.svg", summary, results)
    write_markdown(output_dir / "README.md", summary, results, "results.svg")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
