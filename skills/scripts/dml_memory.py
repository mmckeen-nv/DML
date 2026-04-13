#!/usr/bin/env python3
"""Small CLI wrapper to use local DML as an OpenClaw memory substrate."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

WORKSPACE = Path("/home/nvidia/.openclaw/workspace")
DML_CORE = WORKSPACE / "dml" / "dml_core"
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (DML_CORE, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from daystrom_dml.agent_schema import MemoryKind  # type: ignore
from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore
from tuning_utils import (  # type: ignore
    infer_intent_terms,
    noise_score,
    relevance_score,
    rewrite_query,
    should_keep_chunk,
    smart_chunks,
)


def _parse_meta(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --meta JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--meta must decode to a JSON object")
    return parsed


def _kind(value: str) -> MemoryKind:
    token = value.strip().lower()
    mapping = {
        "action": MemoryKind.ACTION,
        "observation": MemoryKind.OBSERVATION,
        "note": MemoryKind.NOTE,
        "insight": MemoryKind.NOTE,
        "plan": MemoryKind.PLAN,
        "planning": MemoryKind.PLAN,
        "execution": MemoryKind.ACTION,
        "result": MemoryKind.NOTE,
        "error": MemoryKind.ERROR,
        "artifact": MemoryKind.ARTIFACT_REF,
    }
    return mapping.get(token, MemoryKind.ACTION)


def _backend_proof(adapter: DMLAdapter) -> dict:
    embedder = getattr(adapter, "embedder", None)
    embedder_class = type(embedder).__name__ if embedder is not None else None
    embedding_model = None
    embedding_device_cfg = None
    try:
        embedding_model = adapter.config.get("embedding_model")
        embedding_device_cfg = adapter.config.get("embedding_device")
    except Exception:
        pass

    backend = {
        "embedding_model": embedding_model,
        "embedding_device_cfg": embedding_device_cfg,
        "embedder_class": embedder_class,
        "embedder_backend": "unknown",
        "embedder_ready": False,
        "embedder_target_device": None,
        "runner_backend_class": type(getattr(adapter.runner, "_backend", None)).__name__,
        "runner_is_dummy": bool(getattr(adapter.runner, "is_dummy", False)),
        "llm_backend_cfg": adapter.config.get("llm_backend"),
        "llm_model_name": adapter.config.get("model_name"),
        "storage_dir": str(getattr(adapter, "storage_dir", "")),
    }

    model_name = str(embedding_model or "")
    if model_name.startswith("ollama:"):
        backend["embedder_backend"] = "ollama"
        backend["embedder_target_device"] = "ollama-managed"
        backend["ollama_model_name"] = model_name.split(":", 1)[1].strip() or None
        backend["ollama_base_url"] = getattr(embedder, "base_url", None)
        backend["ollama_dim"] = getattr(embedder, "_dim", None)
        backend["embedder_ready"] = bool(
            embedder is not None
            and (
                embedder_class == "OllamaEmbedder"
                or embedder_class.endswith("OllamaEmbedder")
                or backend["ollama_base_url"]
                or backend["ollama_dim"]
            )
        )
        return backend

    model = getattr(embedder, "_model", None)
    target = str(getattr(model, "device", getattr(model, "_target_device", ""))).strip() or None
    backend["embedder_backend"] = "sentence_transformers"
    backend["embedder_ready"] = model is not None
    backend["embedder_target_device"] = target
    return backend



def _assert_gpu_only(adapter: DMLAdapter) -> None:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("GPU-only mode requires torch with CUDA support") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only mode enabled, but CUDA is not available")

    backend = _backend_proof(adapter)
    if backend.get("embedder_backend") == "ollama":
        if not backend.get("embedder_ready"):
            raise RuntimeError("GPU-only mode requires Ollama embedder when using ollama:* embedding_model")
        if str(backend.get("embedding_device_cfg") or "").lower() != "cuda":
            raise RuntimeError(
                f"GPU-only Ollama mode requires embedding_device config to stay explicit as cuda, got: {backend.get('embedding_device_cfg') or 'unknown'}"
            )
        return

    if not backend.get("embedder_ready"):
        raise RuntimeError("GPU-only mode requires SentenceTransformer embedder (fallback embedder detected)")

    target = str(backend.get("embedder_target_device") or "").lower()
    if "cuda" not in target:
        raise RuntimeError(f"GPU-only mode requires CUDA embedding device, got: {target or 'unknown'}")


def _adapter(storage_dir: str, config_path: str | None, require_gpu: bool) -> DMLAdapter:
    adapter = DMLAdapter(
        config_path=config_path,
        config_overrides={
            "storage_dir": storage_dir,
            "dml.agentic_mode.enabled": True,
            "embedding_device": "cuda" if require_gpu else None,
        },
    )
    if require_gpu:
        _assert_gpu_only(adapter)
    return adapter


def cmd_ingest(args: argparse.Namespace) -> int:
    meta = _parse_meta(args.meta)
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    try:
        payload_meta = {**meta, "kind": _kind(args.kind).value}
        chunks = smart_chunks(args.text, chunk_chars=max(180, args.chunk_chars), overlap=max(0, args.chunk_overlap)) if args.chunk else [args.text]
        kept = 0
        for chunk in chunks:
            if args.filter_noise and not should_keep_chunk(chunk):
                continue
            adapter.ingest(chunk, meta=payload_meta)
            kept += 1
    finally:
        adapter.close()
    print(json.dumps({"status": "ok", "action": "ingest", "kind": args.kind, "chunks_ingested": kept}, indent=2))
    return 0


def _memory_confidence(report: dict, *, query: str) -> float:
    items = report.get("items") or []
    if not items:
        return 0.0

    intent = infer_intent_terms(query)
    rel_scores = []
    noise_scores = []
    for item in items:
        text = str(item.get("text") or item.get("summary") or "")
        rel_scores.append(relevance_score(text, intent))
        noise_scores.append(noise_score(text))

    avg_rel = sum(rel_scores) / max(1, len(rel_scores))
    avg_noise = sum(noise_scores) / max(1, len(noise_scores))
    hit_factor = min(1.0, len(items) / 4.0)
    conf = (0.55 * avg_rel) + (0.30 * (1.0 - avg_noise)) + (0.15 * hit_factor)
    return max(0.0, min(1.0, conf))


def _reform_memory_from_ground_truth(*, adapter: DMLAdapter, query: str, ground_truth: dict, tag: str = "rag-reform") -> int:
    context = str(ground_truth.get("context") or "").strip()
    if not context:
        return 0

    chunks = smart_chunks(context, chunk_chars=700, overlap=80)
    kept = 0
    for chunk in chunks[:6]:
        if not should_keep_chunk(chunk):
            continue
        adapter.ingest(
            f"[reformed:{tag}] query={query}\n{chunk}",
            meta={"kind": "note", "source": "ground_truth_reform", "tag": tag},
        )
        kept += 1
    return kept


def _query_ground_truth_with_timeout(*, adapter: DMLAdapter, query: str, mode: str, timeout_ms: int) -> dict:
    if timeout_ms <= 0:
        return adapter.query_database(query, mode=mode)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.query_database, query, mode=mode)
        try:
            return future.result(timeout=max(0.001, timeout_ms / 1000.0))
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(f"ground-truth query timed out after {timeout_ms}ms") from exc


def _attach_ground_truth(
    report: dict,
    *,
    adapter: DMLAdapter,
    query: str,
    mode: str,
    strict: bool = False,
    timeout_ms: int = 2500,
) -> None:
    try:
        report["ground_truth"] = _query_ground_truth_with_timeout(
            adapter=adapter,
            query=query,
            mode=mode,
            timeout_ms=timeout_ms,
        )
        report["ground_truth_status"] = "ok"
        report["ground_truth_timeout_ms"] = timeout_ms
    except Exception as exc:
        report["ground_truth"] = None
        report["ground_truth_error"] = str(exc)
        report["ground_truth_status"] = "timeout" if isinstance(exc, TimeoutError) else "error"
        report["ground_truth_timeout_ms"] = timeout_ms
        if strict:
            raise


def cmd_retrieve(args: argparse.Namespace) -> int:
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    try:
        query = rewrite_query(args.query) if args.query_expand else args.query
        report = adapter.retrieve_context(
            query,
            tenant_id=args.tenant_id,
            client_id=args.client_id,
            session_id=args.session_id,
            instance_id=args.instance_id,
            top_k=args.top_k,
        )
        report["query_original"] = args.query
        report["query_effective"] = query

        confidence = _memory_confidence(report, query=query)
        report["memory_confidence"] = round(confidence, 4)

        need_ground_truth = args.with_ground_truth and (
            args.ground_truth_policy == "always"
            or (args.ground_truth_policy == "low-confidence" and confidence < args.confidence_threshold)
        )
        report["ground_truth_triggered"] = bool(need_ground_truth)

        if need_ground_truth:
            _attach_ground_truth(
                report,
                adapter=adapter,
                query=query,
                mode=args.ground_truth_mode,
                strict=args.strict_ground_truth,
                timeout_ms=max(0, args.ground_truth_timeout_ms),
            )
            if args.reform_memory and report.get("ground_truth"):
                reformed = _reform_memory_from_ground_truth(
                    adapter=adapter,
                    query=query,
                    ground_truth=report["ground_truth"],
                    tag="low_confidence_repair",
                )
                report["memory_reformed_chunks"] = reformed
    finally:
        adapter.close()
    print(json.dumps(report, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-dir", default=str(WORKSPACE / "data" / "dml-gpu"))
    parser.add_argument(
        "--config-path",
        default=str(WORKSPACE / "skills" / "daystrom-dml" / "config" / "dml_gpu_only.yaml"),
        help="Optional DML YAML config path",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if CUDA embedding path is not active (default: true)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--text", required=True)
    ing.add_argument("--kind", default="action")
    ing.add_argument("--meta", help="JSON object")
    ing.add_argument("--chunk", action=argparse.BooleanOptionalAction, default=True)
    ing.add_argument("--chunk-chars", type=int, default=620)
    ing.add_argument("--chunk-overlap", type=int, default=90)
    ing.add_argument("--filter-noise", action=argparse.BooleanOptionalAction, default=True)
    ing.set_defaults(func=cmd_ingest)

    ret = sub.add_parser("retrieve")
    ret.add_argument("--query", required=True)
    ret.add_argument("--top-k", type=int, default=6)
    ret.add_argument("--tenant-id", default="openclaw")
    ret.add_argument("--client-id")
    ret.add_argument("--session-id")
    ret.add_argument("--instance-id")
    ret.add_argument("--query-expand", action=argparse.BooleanOptionalAction, default=True)
    ret.add_argument("--with-ground-truth", action=argparse.BooleanOptionalAction, default=True)
    ret.add_argument("--ground-truth-mode", default="hybrid", choices=["auto", "semantic", "literal", "hybrid"])
    ret.add_argument(
        "--ground-truth-policy",
        default="low-confidence",
        choices=["always", "low-confidence", "never"],
        help="When to run sidecar ground-truth retrieval",
    )
    ret.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.46,
        help="Trigger ground-truth when memory_confidence < threshold (low-confidence policy)",
    )
    ret.add_argument(
        "--reform-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When ground-truth triggers, ingest condensed ground-truth chunks back into memory",
    )
    ret.add_argument(
        "--strict-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail retrieve command if parallel ground-truth query fails (default: false)",
    )
    ret.add_argument(
        "--ground-truth-timeout-ms",
        type=int,
        default=2500,
        help="Timeout for sidecar ground-truth query in milliseconds (default: 2500)",
    )
    ret.set_defaults(func=cmd_retrieve)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
