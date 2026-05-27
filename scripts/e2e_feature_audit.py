#!/usr/bin/env python3
"""Run an isolated end-to-end Daystrom DML feature audit.

The audit intentionally uses a temporary store and a tiny factual corpus so it
does not depend on, mutate, or ship demo memories from the user's live store.
It writes JSON, Markdown, and SVG summaries under ``out/e2e_feature_audit``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DML_CORE = REPO_ROOT / "dml_core"
if str(DML_CORE) not in sys.path:
    sys.path.insert(0, str(DML_CORE))

from fastapi.testclient import TestClient  # noqa: E402

from daystrom_dml.dml_adapter import DMLAdapter  # noqa: E402
from daystrom_dml.provider_cli import _app_profile  # noqa: E402
from daystrom_dml.provider_server import create_app  # noqa: E402
from daystrom_dml.summarizer import DummySummarizer  # noqa: E402


@dataclass(frozen=True)
class AuditCase:
    name: str
    query: str
    expected: tuple[str, ...]


class KeywordEmbedder:
    """Deterministic lexical embedder for fast isolated E2E checks."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            bucket = hash(token) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0


CORPUS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "Mission Zephyr inventory controller is Quartermaster Ada Sol. "
        "The inventory lockbox code is ORCHID-17. Medical stores include 42 trauma kits.",
        {"tenant_id": "openclaw", "session_id": "alpha", "kind": "inventory", "source": "zephyr-manifest"},
    ),
    (
        "Mission Zephyr propulsion uses argon ion drive HELIOS-9. "
        "Fuel reserve is 18.4 kiloliters argon and burn margin is 11 percent.",
        {"tenant_id": "openclaw", "session_id": "alpha", "kind": "propulsion", "source": "zephyr-propulsion"},
    ),
    (
        "Mission Zephyr crew: Ada Sol owns inventory, Dr. Ren Kaito owns medical, "
        "Mira Vale owns comms, and Theo March owns habitat.",
        {"tenant_id": "openclaw", "session_id": "alpha", "kind": "crew", "source": "zephyr-crew"},
    ),
    (
        "Mission Zephyr landing cache contains six beacon anchors, two shelter ceramics, "
        "and the Larkspur survey drone.",
        {"tenant_id": "openclaw", "session_id": "alpha", "kind": "landing", "source": "zephyr-cache"},
    ),
    (
        "Tenant beta unrelated note: the archive color is violet and the spare keyboard is in lab three.",
        {"tenant_id": "beta", "session_id": "beta-session", "kind": "noise", "source": "scope-control"},
    ),
)

CASES: tuple[AuditCase, ...] = (
    AuditCase("inventory", "Who controls Mission Zephyr inventory and what is the lockbox code?", ("Ada Sol", "ORCHID-17")),
    AuditCase("propulsion", "What propulsion system does Mission Zephyr use and how much argon reserve remains?", ("HELIOS-9", "18.4")),
    AuditCase("crew", "Who owns medical, comms, and habitat on Mission Zephyr?", ("Ren Kaito", "Mira Vale", "Theo March")),
    AuditCase("landing", "What is in the Mission Zephyr landing cache?", ("beacon anchors", "shelter ceramics", "Larkspur")),
)


def score_text(text: str, expected: Iterable[str]) -> dict[str, Any]:
    haystack = str(text or "").lower()
    labels = list(expected)
    matched = [label for label in labels if label.lower() in haystack]
    missing = [label for label in labels if label not in matched]
    score = len(matched) / max(1, len(labels))
    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "required": len(labels),
    }


def make_adapter(storage_dir: Path) -> DMLAdapter:
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 128,
            "token_budget": 220,
            "top_k": 6,
            "dml_top_k": 6,
            "dml_context_max_items": 4,
            "dml_context_summary_chars": 220,
            "similarity_threshold": 0.0,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "rag_store": {"enabled": True},
            "dpm": {"enabled": False},
        },
        embedder=KeywordEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


def run_step(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    timer = Timer()
    try:
        payload = fn()
        status = "pass" if payload.get("passed", True) else "fail"
    except Exception as exc:  # pragma: no cover - audit resilience
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        status = "fail"
    return {"name": name, "status": status, "latency_ms": round(timer.ms, 2), **payload}


def audit_adapter(adapter: DMLAdapter) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def ingest() -> dict[str, Any]:
        latencies = []
        for text, meta in CORPUS:
            timer = Timer()
            adapter.ingest(text, meta=meta)
            latencies.append(timer.ms)
        return {
            "passed": len(adapter.store.items()) >= 4,
            "items": len(adapter.store.items()),
            "avg_ingest_ms": round(mean(latencies), 2),
            "max_ingest_ms": round(max(latencies), 2),
        }

    results.append(run_step("adapter.ingest", ingest))

    def scoped_recall() -> dict[str, Any]:
        report = adapter.retrieve_context(
            "Mission Zephyr inventory lockbox tenant scope",
            tenant_id="openclaw",
            session_id="alpha",
            top_k=6,
        )
        context = report.get("raw_context", "")
        score = score_text(context, ("Ada Sol", "ORCHID-17"))
        beta_leak = "violet" in context.lower()
        return {
            "passed": score["score"] == 1.0 and not beta_leak,
            "accuracy": score,
            "context_tokens": report.get("context_tokens"),
            "items": len(report.get("items") or []),
            "reported_latency_ms": report.get("latency_ms"),
            "tenant_leak": beta_leak,
        }

    results.append(run_step("adapter.retrieve_context scoped", scoped_recall))

    def compare() -> dict[str, Any]:
        per_case = []
        for case in CASES:
            report = adapter.compare_responses(case.query, top_k=6, max_new_tokens=96, allow_reinforce=False)
            dml_score = score_text(report.get("dml", {}).get("response", ""), case.expected)
            rag_backend = next((item for item in report.get("rag_backends", []) if item.get("available")), {})
            rag_score = score_text(rag_backend.get("response", ""), case.expected)
            base_score = score_text(report.get("base", {}).get("response", ""), case.expected)
            per_case.append(
                {
                    "case": case.name,
                    "base_accuracy": base_score["score"],
                    "dml_accuracy": dml_score["score"],
                    "rag_accuracy": rag_score["score"],
                    "dml_retrieval_ms": report.get("dml", {}).get("retrieval_latency_ms", 0),
                    "dml_generation_ms": report.get("dml", {}).get("generation_latency_ms", 0),
                    "rag_retrieval_ms": rag_backend.get("retrieval_latency_ms", 0),
                    "rag_generation_ms": rag_backend.get("generation_latency_ms", 0),
                    "dml_tokens": report.get("dml", {}).get("context_tokens", 0),
                    "rag_tokens": rag_backend.get("context_tokens", 0),
                    "dml_nodes": len(report.get("dml", {}).get("entries") or []),
                    "rag_docs": len(rag_backend.get("documents") or []),
                }
            )
        return {
            "passed": mean(item["dml_accuracy"] for item in per_case) >= 0.75,
            "cases": per_case,
            "base_accuracy": round(mean(item["base_accuracy"] for item in per_case), 3),
            "dml_accuracy": round(mean(item["dml_accuracy"] for item in per_case), 3),
            "rag_accuracy": round(mean(item["rag_accuracy"] for item in per_case), 3),
            "dml_total_ms": round(mean(item["dml_retrieval_ms"] + item["dml_generation_ms"] for item in per_case), 2),
            "rag_total_ms": round(mean(item["rag_retrieval_ms"] + item["rag_generation_ms"] for item in per_case), 2),
            "dml_context_tokens": round(mean(item["dml_tokens"] for item in per_case), 1),
            "rag_context_tokens": round(mean(item["rag_tokens"] for item in per_case), 1),
        }

    results.append(run_step("adapter.compare_responses DML/RAG", compare))
    return results


def audit_provider(adapter: DMLAdapter) -> list[dict[str, Any]]:
    client = TestClient(create_app(adapter_factory=lambda: adapter))
    results: list[dict[str, Any]] = []

    results.append(
        run_step(
            "provider.health",
            lambda: {
                "passed": client.get("/health").json().get("status") == "ok",
            },
        )
    )

    def remember_recall() -> dict[str, Any]:
        text = "Provider smoke memory says Zephyr audit token is COBALT-29."
        remember = client.post(
            "/api/remember",
            json={
                "text": text,
                "tenant_id": "openclaw",
                "session_id": "provider",
                "kind": "smoke_test",
                "meta": {"source": "e2e-audit"},
            },
        )
        recall = client.post(
            "/api/recall",
            json={"query": "Zephyr audit token", "tenant_id": "openclaw", "session_id": "provider", "top_k": 4},
        )
        payload = recall.json()
        score = score_text(payload.get("raw_context", ""), ("COBALT-29",))
        return {
            "passed": remember.status_code == 200 and score["score"] == 1.0,
            "accuracy": score,
            "context_tokens": payload.get("context_tokens"),
            "items": len(payload.get("items") or []),
        }

    results.append(run_step("provider.remember+recall", remember_recall))

    def ollama_routes() -> dict[str, Any]:
        tags = client.get("/api/tags").json()
        chat = client.post(
            "/api/chat",
            json={
                "model": "daystrom-dml:memory",
                "messages": [{"role": "user", "content": "Zephyr audit token"}],
                "tenant_id": "openclaw",
                "session_id": "provider",
            },
        ).json()
        embed = client.post("/api/embed", json={"model": "daystrom-dml:memory", "input": ["hello", "world"]}).json()
        score = score_text(chat.get("message", {}).get("content", ""), ("COBALT-29",))
        return {
            "passed": bool(tags.get("models")) and len(embed.get("embeddings") or []) == 2 and score["score"] == 1.0,
            "accuracy": score,
            "models": [model.get("name") for model in tags.get("models", [])],
            "embeddings": len(embed.get("embeddings") or []),
        }

    results.append(run_step("ollama-compatible routes", ollama_routes))
    return results


def audit_profiles(output_dir: Path) -> list[dict[str, Any]]:
    profiles = ["openclaw", "hermes", "generic"]

    def profile_case(app: str) -> dict[str, Any]:
        profile = _app_profile(app, base_url="http://127.0.0.1:8765", tenant_id=app, storage_dir=str(output_dir / "store"))
        return {
            "passed": profile.get("provider") == "daystrom-dml" and "mcp" in profile and "environment" in profile,
            "app": app,
            "has_mcp": "mcp" in profile,
            "has_environment": "environment" in profile,
        }

    return [run_step(f"profile.{app}", lambda app=app: profile_case(app)) for app in profiles]


def flatten_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for group in results for step in group.get("steps", [])]
    compare = next((step for step in steps if step["name"] == "adapter.compare_responses DML/RAG"), {})
    return {
        "total_steps": len(steps),
        "passed_steps": sum(1 for step in steps if step.get("status") == "pass"),
        "failed_steps": sum(1 for step in steps if step.get("status") != "pass"),
        "avg_step_latency_ms": round(mean(step.get("latency_ms", 0) for step in steps), 2) if steps else 0,
        "base_accuracy": compare.get("base_accuracy"),
        "dml_accuracy": compare.get("dml_accuracy"),
        "rag_accuracy": compare.get("rag_accuracy"),
        "dml_total_ms": compare.get("dml_total_ms"),
        "rag_total_ms": compare.get("rag_total_ms"),
        "dml_context_tokens": compare.get("dml_context_tokens"),
        "rag_context_tokens": compare.get("rag_context_tokens"),
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def bar(width: float, max_width: float, value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0
    return max(2, min(max_width, width * (value / max_value)))


def write_svg(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    width = 1100
    height = 760
    steps = [step for group in results for step in group.get("steps", [])]
    compare = next((step for step in steps if step["name"] == "adapter.compare_responses DML/RAG"), {})
    accuracy_rows = [
        ("Base", float(summary.get("base_accuracy") or 0), "#9ba8a5"),
        ("DML", float(summary.get("dml_accuracy") or 0), "#68d391"),
        ("RAG", float(summary.get("rag_accuracy") or 0), "#8ab4ff"),
    ]
    latency_rows = [
        ("DML", float(summary.get("dml_total_ms") or 0), "#68d391"),
        ("RAG", float(summary.get("rag_total_ms") or 0), "#8ab4ff"),
    ]
    token_rows = [
        ("DML", float(summary.get("dml_context_tokens") or 0), "#68d391"),
        ("RAG", float(summary.get("rag_context_tokens") or 0), "#8ab4ff"),
    ]
    max_latency = max([row[1] for row in latency_rows] + [1])
    max_tokens = max([row[1] for row in token_rows] + [1])
    pass_rate = summary["passed_steps"] / max(1, summary["total_steps"])

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760">',
        "<defs>",
        "<linearGradient id='bg' x1='0' x2='1' y1='0' y2='1'><stop stop-color='#101314'/><stop offset='1' stop-color='#1b2120'/></linearGradient>",
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#edf7f2}.muted{fill:#9ba8a5}.small{font-size:14px}.label{font-size:16px;font-weight:700}.title{font-size:34px;font-weight:800}.card{fill:#151b1a;stroke:#2f3a37;stroke-width:1}.track{fill:#26302d}.ok{fill:#68d391}.warn{fill:#f6c177}.bad{fill:#f87171}</style>",
        "</defs>",
        "<rect width='1100' height='760' fill='url(#bg)'/>",
        "<text x='44' y='62' class='title'>Daystrom DML E2E Feature Audit</text>",
        f"<text x='44' y='92' class='muted small'>{summary['passed_steps']} / {summary['total_steps']} checks passed · average step latency {summary['avg_step_latency_ms']} ms</text>",
        "<rect x='44' y='126' width='314' height='150' rx='8' class='card'/>",
        "<rect x='393' y='126' width='314' height='150' rx='8' class='card'/>",
        "<rect x='742' y='126' width='314' height='150' rx='8' class='card'/>",
        "<text x='68' y='164' class='label'>Feature Pass Rate</text>",
        f"<text x='68' y='224' style='font-size:52px;font-weight:800'>{pct(pass_rate)}</text>",
        f"<rect x='68' y='242' width='250' height='12' rx='6' class='track'/><rect x='68' y='242' width='{250 * pass_rate:.1f}' height='12' rx='6' class='ok'/>",
        "<text x='417' y='164' class='label'>DML Accuracy</text>",
        f"<text x='417' y='224' style='font-size:52px;font-weight:800'>{pct(summary.get('dml_accuracy'))}</text>",
        "<text x='766' y='164' class='label'>Avg DML Latency</text>",
        f"<text x='766' y='224' style='font-size:52px;font-weight:800'>{summary.get('dml_total_ms', 0)} ms</text>",
        "<text x='44' y='332' class='label'>Answer Accuracy</text>",
    ]
    y = 360
    for label, value, color in accuracy_rows:
        lines.extend(
            [
                f"<text x='70' y='{y + 20}' class='small'>{label}</text>",
                f"<rect x='145' y='{y}' width='360' height='24' rx='5' class='track'/>",
                f"<rect x='145' y='{y}' width='{360 * value:.1f}' height='24' rx='5' fill='{color}'/>",
                f"<text x='525' y='{y + 20}' class='small'>{pct(value)}</text>",
            ]
        )
        y += 42

    lines.append("<text x='44' y='514' class='label'>Latency and Context Size</text>")
    y = 542
    for label, value, color in latency_rows:
        lines.extend(
            [
                f"<text x='70' y='{y + 20}' class='small'>{label} latency</text>",
                f"<rect x='190' y='{y}' width='320' height='22' rx='5' class='track'/>",
                f"<rect x='190' y='{y}' width='{bar(320, 320, value, max_latency):.1f}' height='22' rx='5' fill='{color}'/>",
                f"<text x='530' y='{y + 18}' class='small'>{value:.1f} ms</text>",
            ]
        )
        y += 38
    for label, value, color in token_rows:
        lines.extend(
            [
                f"<text x='70' y='{y + 20}' class='small'>{label} tokens</text>",
                f"<rect x='190' y='{y}' width='320' height='22' rx='5' class='track'/>",
                f"<rect x='190' y='{y}' width='{bar(320, 320, value, max_tokens):.1f}' height='22' rx='5' fill='{color}'/>",
                f"<text x='530' y='{y + 18}' class='small'>{value:.1f}</text>",
            ]
        )
        y += 38

    lines.append("<text x='650' y='332' class='label'>Feature Checks</text>")
    y = 360
    for step in steps[:9]:
        color = "#68d391" if step.get("status") == "pass" else "#f87171"
        lines.extend(
            [
                f"<circle cx='672' cy='{y + 8}' r='7' fill='{color}'/>",
                f"<text x='692' y='{y + 13}' class='small'>{step['name']}</text>",
                f"<text x='1010' y='{y + 13}' text-anchor='end' class='muted small'>{step.get('latency_ms', 0)} ms</text>",
            ]
        )
        y += 34
    if len(steps) > 9:
        lines.append(f"<text x='692' y='{y + 13}' class='muted small'>+ {len(steps) - 9} more checks in JSON report</text>")

    if compare.get("cases"):
        lines.append("<text x='650' y='690' class='muted small'>Per-query case results are included in results.json.</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], svg_name: str) -> None:
    lines = [
        "# Daystrom DML E2E Feature Audit",
        "",
        f"![E2E audit graphic]({svg_name})",
        "",
        "## Summary",
        "",
        f"- Checks passed: {summary['passed_steps']} / {summary['total_steps']}",
        f"- Average step latency: {summary['avg_step_latency_ms']} ms",
        f"- Base accuracy: {pct(summary.get('base_accuracy'))}",
        f"- DML accuracy: {pct(summary.get('dml_accuracy'))}",
        f"- RAG accuracy: {pct(summary.get('rag_accuracy'))}",
        f"- Avg DML latency: {summary.get('dml_total_ms')} ms",
        f"- Avg RAG latency: {summary.get('rag_total_ms')} ms",
        f"- Avg DML context tokens: {summary.get('dml_context_tokens')}",
        f"- Avg RAG context tokens: {summary.get('rag_context_tokens')}",
        "",
        "## Feature Checks",
        "",
        "| Feature | Status | Latency | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for group in results:
        for step in group.get("steps", []):
            detail = ""
            if "accuracy" in step and isinstance(step["accuracy"], dict):
                detail = f"accuracy {pct(step['accuracy'].get('score'))}"
            elif "dml_accuracy" in step:
                detail = f"DML {pct(step.get('dml_accuracy'))}, RAG {pct(step.get('rag_accuracy'))}"
            elif "items" in step:
                detail = f"{step['items']} items"
            lines.append(f"| {step['name']} | {step['status']} | {step.get('latency_ms', 0)} ms | {detail} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated Daystrom DML E2E feature audit")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "e2e_feature_audit")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dml-e2e-store-") as tmp:
        adapter = make_adapter(Path(tmp))
        try:
            results = [
                {"group": "adapter", "steps": audit_adapter(adapter)},
                {"group": "provider", "steps": audit_provider(adapter)},
                {"group": "profiles", "steps": audit_profiles(output_dir)},
            ]
        finally:
            adapter.close()

    summary = flatten_summary(results)
    payload = {
        "summary": summary,
        "results": results,
        "notes": [
            "Audit uses an isolated temporary store and does not mutate the live DML store.",
            "Corpus is intentionally tiny and factual; it is test data, not bundled installer data.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(output_dir / "results.svg", summary, results)
    write_markdown(output_dir / "README.md", summary, results, "results.svg")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["failed_steps"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
