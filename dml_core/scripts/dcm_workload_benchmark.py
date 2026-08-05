"""Run the digest-only DCM multi-strategy workload benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from daystrom_dml.context.benchmark import (
    BenchmarkConfig,
    DeterministicEvidenceClient,
    Strategy,
    default_workload,
    extended_workload,
    run_workload,
    stress_workload,
)
from daystrom_dml.context.probe import OpenAICompatibleModelClient, atomic_write_json
from daystrom_dml.embeddings import OllamaEmbedder


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use deterministic evidence-presence model")
    parser.add_argument("--allow-network", action="store_true", help="required for live endpoint calls")
    parser.add_argument("--endpoint-url", default=os.environ.get("DCM_MODEL_PROBE_ENDPOINT_URL", ""))
    parser.add_argument("--model", default=os.environ.get("DCM_MODEL_PROBE_MODEL", ""))
    parser.add_argument("--api-key", default=os.environ.get("DCM_MODEL_PROBE_API_KEY"))
    parser.add_argument("--context-budget-tokens", type=int, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--rag-top-k", type=int, default=1)
    parser.add_argument("--dcm-semantic-candidates", type=int, default=2)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("DCM_BENCHMARK_EMBEDDING_MODEL", ""),
        help="optional Ollama embedding model for the production semantic catalog path",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("DCM_BENCHMARK_EMBEDDING_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--suite",
        choices=("regression", "stress", "extended"),
        default="regression",
        help="sanitized workload suite to run",
    )
    parser.add_argument("--strategy", action="append", choices=[item.value for item in Strategy])
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    if not args.offline and not args.allow_network:
        parser.error("live benchmark requires explicit --allow-network")
    if args.embedding_model and not args.allow_network:
        parser.error("embedding-backed retrieval requires explicit --allow-network")
    if not args.offline and (not args.endpoint_url or not args.model):
        parser.error("live benchmark requires --endpoint-url and --model")

    endpoint = args.endpoint_url or "http://127.0.0.1:11434/v1/chat/completions"
    model = args.model or "deterministic-evidence-model"
    config = BenchmarkConfig(
        endpoint_url=endpoint,
        model_id=model,
        runtime_id="openai-compatible" if not args.offline else "offline-deterministic",
        context_budget_tokens=args.context_budget_tokens,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        rag_top_k=args.rag_top_k,
        dcm_semantic_candidates=args.dcm_semantic_candidates,
        embedding_model_id=args.embedding_model or None,
        embedding_endpoint_url=args.embedding_base_url if args.embedding_model else None,
    )
    client = (
        DeterministicEvidenceClient()
        if args.offline
        else OpenAICompatibleModelClient(api_key=args.api_key)
    )
    semantic_embedder = (
        OllamaEmbedder(args.embedding_model, base_url=args.embedding_base_url.rstrip("/"))
        if args.embedding_model
        else None
    )
    strategies = [Strategy(value) for value in args.strategy] if args.strategy else list(Strategy)
    workload = {
        "regression": default_workload,
        "stress": stress_workload,
        "extended": extended_workload,
    }[args.suite]()
    report = run_workload(
        workload,
        client,
        config,
        strategies=strategies,
        semantic_embedder=semantic_embedder,
    )
    payload = report.to_dict()
    print(json.dumps(payload, sort_keys=True, indent=2))
    if args.output_json:
        atomic_write_json(Path(args.output_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
