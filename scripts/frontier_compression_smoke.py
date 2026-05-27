#!/usr/bin/env python3
"""Smoke test DML as a frontier-model compression proxy.

The script does not call paid frontier APIs. It builds a long synthetic agent
history, stores it in DML, asks a local Ollama model for a cheap draft when
available, then estimates the prompt/output token delta between:

1. sending the full raw history directly to a frontier model, and
2. sending compact DML context plus a local draft for frontier verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DML_CORE = REPO_ROOT / "dml_core"
if str(DML_CORE) not in sys.path:
    sys.path.insert(0, str(DML_CORE))

from daystrom_dml.agent_schema import MemoryKind, MemoryOutcome, MemoryPhase  # noqa: E402
from daystrom_dml.dml_adapter import DMLAdapter  # noqa: E402
from daystrom_dml.summarizer import DummySummarizer  # noqa: E402
from daystrom_dml import utils  # noqa: E402


class KeywordEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def estimate_tokens(text: str) -> int:
    return utils.estimate_tokens(text)


def make_adapter(storage_dir: Path) -> DMLAdapter:
    return DMLAdapter(
        config_overrides={
            "model_name": "dummy",
            "embedding_model": None,
            "capacity": 5000,
            "token_budget": 900,
            "dml_top_k": 10,
            "dml_context_max_items": 6,
            "dml_context_summary_chars": 420,
            "similarity_threshold": 0.0,
            "storage_dir": str(storage_dir),
            "persistence": {"enable": False},
            "rag_store": {"enabled": False},
            "dpm": {"enabled": False},
            "dml.agentic_mode.enabled": True,
        },
        embedder=KeywordEmbedder(),
        summarizer=DummySummarizer(),
        start_aging_loop=False,
    )


def build_history(turns: int) -> list[str]:
    durable = [
        "FINAL-DECISION-ATLAS-42: Ship the provider as an Ollama-compatible memory service before adding a desktop shell.",
        "USER-PREF-ORION-17: User prefers concise answers with direct engineering tradeoffs and no ceremonial phrasing.",
        "BLOCKER-VEGA-09: Long-horizon agents lose late anchors unless survival ledger state is explicitly carried forward.",
        "PATCH-RIGEL-55: Force scoped survival ledger into retrieve_context before normal compact context.",
        "BENCH-CYGNUS-31: DML should be judged on accuracy, context tokens, final model latency, and recovery after compaction.",
        "NEXT-STEP-SIRIUS-88: Build a frontier compression proxy that routes local drafts to expensive models only when needed.",
    ]
    filler_topics = [
        "provider UI polish",
        "agentic scratch promotion",
        "visualizer camera control",
        "RAG comparison latency",
        "synthetic mission dataset",
        "OpenClaw installer path",
        "personality matrix overlay",
        "memory curation",
    ]
    rows: list[str] = []
    for idx in range(1, turns + 1):
        topic = filler_topics[idx % len(filler_topics)]
        anchor = durable[idx % len(durable)] if idx % 17 == 0 else ""
        rows.append(
            f"Turn {idx:04d}: worked on {topic}. "
            f"Observed telemetry bucket {idx % 29}, retry count {idx % 5}, and local note shard {idx % 13}. "
            f"{anchor} "
            "Most of this turn is ordinary process detail and should not be sent wholesale to a frontier model."
        )
    rows.append("Final survival ledger: " + " ".join(durable))
    return rows


def seed_dml(adapter: DMLAdapter, history: list[str]) -> None:
    for idx, text in enumerate(history, start=1):
        is_compaction = idx % 17 == 0 or text.startswith("Final survival ledger")
        adapter.ingest_agentic(
            text,
            kind=MemoryKind.NOTE,
            meta={
                "tenant_id": "frontier-smoke",
                "session_id": "long-agent-run",
                "task_id": "FRONTIER-COMPRESSION",
                "step_id": f"TURN-{idx:04d}",
                "episode_id": "THEORETICAL-SMOKE",
                "phase": MemoryPhase.REFLECT.value if is_compaction else MemoryPhase.EXECUTE.value,
                "tool": "compactor" if is_compaction else "agent",
                "outcome": MemoryOutcome.SUCCESS.value,
                "compaction_cycle": idx if is_compaction else None,
                "virtual_tokens": estimate_tokens("\n".join(history[:idx])) if is_compaction else None,
            },
        )


def ollama_models(base_url: str) -> list[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return []
    return [model.get("name", "") for model in response.json().get("models", []) if model.get("name")]


def choose_model(models: list[str], preferred: str | None) -> str | None:
    if preferred:
        return preferred if preferred in models else preferred
    for candidate in ("gemma3:4b", "llama3:8b"):
        if candidate in models:
            return candidate
    for model in models:
        if "embed" not in model.lower():
            return model
    return None


def local_generate(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, float, dict[str, Any]]:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1, "top_p": 0.9},
        },
        timeout=180,
    )
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    payload = response.json()
    return payload.get("response", "").strip(), elapsed_ms, payload


def score_answer(text: str, expected: list[str]) -> dict[str, Any]:
    haystack = text.lower()
    matched = [item for item in expected if item.lower() in haystack]
    return {
        "score": len(matched) / max(1, len(expected)),
        "matched": matched,
        "missing": [item for item in expected if item not in matched],
    }


def estimate_cost(tokens: int, dollars_per_million: float) -> float:
    return (tokens / 1_000_000.0) * dollars_per_million


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Frontier Compression Smoke Test",
        "",
        "This is a theoretical cost/token smoke test. It does not call OpenAI, Claude, or any paid frontier endpoint.",
        "",
        "## Summary",
        "",
        f"- Local draft model: {s['local_model']}",
        f"- Direct frontier input tokens: {s['direct_input_tokens']}",
        f"- DML proxy frontier input tokens: {s['proxy_input_tokens']}",
        f"- Estimated input token savings: {s['input_token_savings_pct']}%",
        f"- Estimated frontier output token savings: {s['output_token_savings_pct']}%",
        f"- Local draft latency: {s['local_draft_latency_ms']} ms",
        f"- DML context tokens: {s['dml_context_tokens']}",
        f"- DML retrieval latency: {s['dml_retrieval_latency_ms']} ms",
        f"- DML context anchor recall score: {s['dml_context_recall_pct']}%",
        f"- Draft anchor recall score: {s['draft_recall_pct']}%",
        "",
        "## Interpretation",
        "",
        "DML is acting as an external compression and draft layer, not provider-internal speculative decoding.",
        "The frontier model would receive a compact memory context plus a local draft to verify/correct instead of the raw long history.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test DML as a frontier compression proxy")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "frontier_compression_smoke")
    parser.add_argument("--turns", type=int, default=360)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--local-max-tokens", type=int, default=220)
    parser.add_argument("--direct-output-tokens", type=int, default=900)
    parser.add_argument("--verify-output-tokens", type=int, default=420)
    parser.add_argument("--frontier-input-price", type=float, default=5.0)
    parser.add_argument("--frontier-output-price", type=float, default=15.0)
    args = parser.parse_args(argv)

    question = (
        "For the long agent run, what was the final provider decision, what blocker "
        "affected long-horizon continuity, what exact patch fixed it, and what should the next step be?"
    )
    expected = ["FINAL-DECISION-ATLAS-42", "BLOCKER-VEGA-09", "PATCH-RIGEL-55", "NEXT-STEP-SIRIUS-88"]
    history = build_history(args.turns)
    raw_history = "\n".join(history)
    direct_prompt = f"Full prior session:\n{raw_history}\n\nUser question:\n{question}"

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dml-frontier-smoke-") as tmp:
        adapter = make_adapter(Path(tmp))
        try:
            seed_dml(adapter, history)
            retrieval_start = time.perf_counter()
            report = adapter.retrieve_context(
                question,
                tenant_id="frontier-smoke",
                session_id="long-agent-run",
                top_k=8,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        finally:
            adapter.close()

    dml_context = report.get("raw_context", "")
    local_prompt = (
        "Use the compact DML memory context to draft a factual answer. "
        "Preserve exact anchor IDs when present.\n\n"
        f"{dml_context}\n\nQuestion:\n{question}\n\nDraft:"
    )
    local_model = choose_model(ollama_models(args.ollama_base_url), args.ollama_model)
    draft = ""
    local_latency_ms: float | None = None
    ollama_payload: dict[str, Any] = {}
    if local_model:
        try:
            draft, local_latency_ms, ollama_payload = local_generate(
                args.ollama_base_url,
                local_model,
                local_prompt,
                args.local_max_tokens,
            )
        except requests.RequestException as exc:
            draft = f"[local draft unavailable: {exc}]"
    else:
        local_model = "unavailable"
        draft = "[local draft unavailable: no generative Ollama model detected]"

    verify_prompt = (
        "You are a frontier model verifier. Use the DML context and local draft below. "
        "Correct mistakes, preserve exact IDs, and answer concisely.\n\n"
        f"DML context:\n{dml_context}\n\nLocal draft:\n{draft}\n\nQuestion:\n{question}\n\nFinal:"
    )
    direct_input_tokens = estimate_tokens(direct_prompt)
    proxy_input_tokens = estimate_tokens(verify_prompt)
    local_prompt_tokens = estimate_tokens(local_prompt)
    draft_tokens = estimate_tokens(draft)
    input_savings = max(0, direct_input_tokens - proxy_input_tokens)
    output_savings = max(0, args.direct_output_tokens - args.verify_output_tokens)
    context_score = score_answer(dml_context, expected)
    draft_score = score_answer(draft, expected)
    summary = {
        "local_model": local_model,
        "history_turns": args.turns,
        "direct_input_tokens": direct_input_tokens,
        "proxy_input_tokens": proxy_input_tokens,
        "local_prompt_tokens": local_prompt_tokens,
        "local_draft_tokens": draft_tokens,
        "dml_context_tokens": report.get("context_tokens", 0),
        "dml_retrieval_latency_ms": round(retrieval_ms, 2),
        "local_draft_latency_ms": round(local_latency_ms, 2) if local_latency_ms is not None else None,
        "input_token_savings": input_savings,
        "input_token_savings_pct": round((input_savings / max(1, direct_input_tokens)) * 100, 1),
        "direct_output_tokens_assumed": args.direct_output_tokens,
        "proxy_verify_output_tokens_assumed": args.verify_output_tokens,
        "output_token_savings": output_savings,
        "output_token_savings_pct": round((output_savings / max(1, args.direct_output_tokens)) * 100, 1),
        "estimated_direct_frontier_cost": round(
            estimate_cost(direct_input_tokens, args.frontier_input_price)
            + estimate_cost(args.direct_output_tokens, args.frontier_output_price),
            6,
        ),
        "estimated_proxy_frontier_cost": round(
            estimate_cost(proxy_input_tokens, args.frontier_input_price)
            + estimate_cost(args.verify_output_tokens, args.frontier_output_price),
            6,
        ),
        "dml_context_recall_pct": round(context_score["score"] * 100, 1),
        "draft_recall_pct": round(draft_score["score"] * 100, 1),
    }
    payload = {
        "summary": summary,
        "question": question,
        "expected": expected,
        "context_score": context_score,
        "draft_score": draft_score,
        "dml_context": dml_context,
        "local_draft": draft,
        "frontier_verify_prompt_preview": verify_prompt[:2400],
        "ollama_usage": {
            key: ollama_payload.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if key in ollama_payload
        },
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / "README.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
