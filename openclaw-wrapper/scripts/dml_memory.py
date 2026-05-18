#!/usr/bin/env python3
"""Small CLI wrapper to use local DML as an OpenClaw memory substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

DEFAULT_WORKSPACE = Path("/Users/markmckeen/.openclaw/workspace")
WORKSPACE = Path(os.environ.get("OPENCLAW_WORKSPACE", str(DEFAULT_WORKSPACE))).resolve()
DAYSTROM_DML_HOME = Path(os.environ.get("DAYSTROM_DML_HOME", "/Users/markmckeen/.openclaw/daystrom-dml-v2")).resolve()
WRAPPER_HOME = DAYSTROM_DML_HOME / "openclaw-wrapper"
DML_PROJECT = DAYSTROM_DML_HOME / "dml"
LEGACY_DML_PROJECT = WORKSPACE / "projects" / "dml"
OLDER_DML_PROJECT = WORKSPACE / "dml"


def _resolve_existing(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _ensure_gpu_venv_runtime() -> None:
    """Re-exec inside the dedicated GPU venv when launched from the wrong Python."""
    if os.environ.get("DML_SKIP_VENV_REEXEC") == "1":
        return

    explicit_venv = os.environ.get("DAYSTROM_DML_VENV")
    target_venv = _resolve_existing(
        Path(explicit_venv).resolve() if explicit_venv else None,
        WORKSPACE / ".venv-dmlgpu",
        DAYSTROM_DML_HOME / ".venv-dml",
        DML_PROJECT / ".venv-dmlgpu",
        DML_PROJECT / ".venv",
        LEGACY_DML_PROJECT / ".venv-dmlgpu",
        LEGACY_DML_PROJECT / ".venv",
        OLDER_DML_PROJECT / ".venv-dmlgpu",
        OLDER_DML_PROJECT / ".venv",
    )
    if target_venv is None:
        return

    target_python = target_venv / "bin" / "python"
    current_prefix = Path(getattr(sys, "prefix", "") or "").resolve()
    current_venv = Path(os.environ.get("VIRTUAL_ENV", "") or current_prefix).resolve()

    try:
        already_target = current_prefix == target_venv.resolve() or current_venv == target_venv.resolve()
    except Exception:
        already_target = str(current_prefix) == str(target_venv) or str(current_venv) == str(target_venv)

    if already_target or not target_python.exists():
        return

    env = os.environ.copy()
    env["DML_SKIP_VENV_REEXEC"] = "1"
    env["VIRTUAL_ENV"] = str(target_venv)
    env["PATH"] = f"{target_venv / 'bin'}:{env.get('PATH', '')}"
    os.execve(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


DML_CORE = _resolve_existing(
    Path(os.environ.get("DAYSTROM_DML_CORE", "")).resolve() if os.environ.get("DAYSTROM_DML_CORE") else None,
    DML_PROJECT / "dml_core",
    LEGACY_DML_PROJECT / "dml_core",
    OLDER_DML_PROJECT / "dml_core",
)
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (DML_CORE, SCRIPT_DIR):
    if p is not None and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from daystrom_dml.agent_schema import MemoryKind  # type: ignore
from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore
from tuning_utils import (  # type: ignore
    continuity_focus_score,
    continuity_handoff_summary,
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


def _normalize_text_for_dedup(text: str) -> str:
    return " ".join((text or "").strip().split())


def _text_digest(text: str) -> str:
    normalized = _normalize_text_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _dedup_index_path(storage_dir: str) -> Path:
    return Path(storage_dir) / ".ingest_dedup_sha256.txt"


def _load_dedup_index(storage_dir: str) -> set[str]:
    path = _dedup_index_path(storage_dir)
    if not path.exists():
        return set()
    try:
        return {line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}
    except Exception:
        return set()


def _append_dedup_digest(storage_dir: str, digest: str) -> None:
    path = _dedup_index_path(storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(digest + "\n")


def _adapter(storage_dir: str, config_path: str | None, require_gpu: bool) -> DMLAdapter:
    graph_path = os.environ.get("DAYSTROM_DPM_GRAPH_PATH") or str(Path(storage_dir) / "dpm_preference_graph.json")
    dpm_mode = os.environ.get("DAYSTROM_DPM_MODE", "active-write")
    dpm_enable = os.environ.get("DAYSTROM_DPM_ENABLE", "1").strip().lower() not in {"0", "false", "no", "off"}
    adapter = DMLAdapter(
        config_path=config_path,
        config_overrides={
            "storage_dir": storage_dir,
            "dml.agentic_mode.enabled": True,
            "embedding_device": "cuda" if require_gpu else None,
            "strict_llm_required": False,
            "dpm": {
                "enable": dpm_enable,
                "mode": dpm_mode,
                "preference_graph_path": graph_path,
                "relationship_id": os.environ.get("DAYSTROM_DPM_RELATIONSHIP_ID", "relationship:openclaw"),
                "project_id": os.environ.get("DAYSTROM_DPM_PROJECT_ID", "project:openclaw"),
                "token_budget": int(os.environ.get("DAYSTROM_DPM_TOKEN_BUDGET", "80")),
            },
        },
    )
    if require_gpu:
        _assert_gpu_only(adapter)
    return adapter


def cmd_ingest(args: argparse.Namespace) -> int:
    meta = _parse_meta(args.meta)
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    seen = _load_dedup_index(args.storage_dir)
    try:
        payload_meta = {**meta, "kind": _kind(args.kind).value}
        chunks = smart_chunks(args.text, chunk_chars=max(180, args.chunk_chars), overlap=max(0, args.chunk_overlap)) if args.chunk else [args.text]
        kept = 0
        skipped_duplicate = 0
        for chunk in chunks:
            if args.filter_noise and not should_keep_chunk(chunk):
                continue
            digest = _text_digest(chunk)
            if digest in seen:
                skipped_duplicate += 1
                continue
            adapter.ingest(chunk, meta=payload_meta, persist=False)
            _append_dedup_digest(args.storage_dir, digest)
            seen.add(digest)
            kept += 1
        adapter._persist_all()
    finally:
        adapter.close()
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "ingest",
                "kind": args.kind,
                "chunks_ingested": kept,
                "chunks_skipped_duplicate": skipped_duplicate,
            },
            indent=2,
        )
    )
    return 0


def cmd_migration_status(args: argparse.Namespace) -> int:
    storage_dir = Path(args.storage_dir)
    local_report = storage_dir / "embedding_compatibility_report.json"
    top_level_report = WORKSPACE / "out" / "dml-migration-progress.json"
    report_path = local_report if local_report.exists() else top_level_report
    if not report_path.exists():
        print(
            json.dumps(
                {
                    "status": "missing",
                    "action": "migration-status",
                    "storage_dir": str(storage_dir),
                    "report_path": str(report_path),
                    "message": "no migration progress report found",
                },
                indent=2,
            )
        )
        return 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2))
    return 0


def cmd_migrate_embeddings(args: argparse.Namespace) -> int:
    storage_dir = Path(args.storage_dir)
    persistence_path = storage_dir / "dml_state.jsonl"
    if not persistence_path.exists():
        print(
            json.dumps(
                {
                    "status": "missing",
                    "action": "migrate-embeddings",
                    "storage_dir": str(storage_dir),
                    "message": "no persisted dml_state.jsonl found",
                },
                indent=2,
            )
        )
        return 0
    from daystrom_dml.persistence import load_state, save_state  # type: ignore

    items = load_state(persistence_path)
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    try:
        payload = {"items": [item.to_dict() for item in items]}
        report = adapter._ensure_embedding_compatibility(payload, max_items=args.max_items)
        adapter.store.import_state(payload)
        save_state(adapter.store.items(), persistence_path)
    finally:
        adapter.close()
    report["action"] = "migrate-embeddings"
    report["storage_dir"] = str(storage_dir)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _memory_confidence(report: dict, *, query: str) -> float:
    items = report.get("items") or []
    if not items:
        return 0.0

    intent = infer_intent_terms(query)
    query_continuity = continuity_focus_score(query)
    rel_scores = []
    noise_scores = []
    continuity_scores = []
    effective_noise_scores = []
    for idx, item in enumerate(items):
        text = str(item.get("text") or item.get("summary") or "")
        rel = relevance_score(text, intent)
        noise = noise_score(text)
        continuity = continuity_focus_score(text)

        rel_scores.append(rel)
        noise_scores.append(noise)
        continuity_scores.append(continuity)

        effective_noise = noise
        if query_continuity >= 0.12 and idx > 0 and continuity < 0.12 and rel < 0.25 and noise >= 0.35:
            effective_noise = min(noise, 0.22)
        effective_noise_scores.append(effective_noise)

    avg_rel = sum(rel_scores) / max(1, len(rel_scores))
    avg_noise = sum(effective_noise_scores) / max(1, len(effective_noise_scores))
    avg_continuity = sum(continuity_scores) / max(1, len(continuity_scores))
    top_rel = max(rel_scores) if rel_scores else 0.0
    top_continuity = max(continuity_scores) if continuity_scores else 0.0
    top_noise = effective_noise_scores[0] if effective_noise_scores else 1.0
    hit_factor = min(1.0, len(items) / 4.0)

    # Reward the strongest matching continuity memory more than we punish a secondary noisy tail.
    conf = (
        (0.40 * avg_rel)
        + (0.20 * top_rel)
        + (0.15 * avg_continuity)
        + (0.10 * top_continuity)
        + (0.10 * (1.0 - avg_noise))
        + (0.05 * (1.0 - min(1.0, top_noise)))
        + (0.10 * hit_factor)
    )
    return max(0.0, min(1.0, conf))


def _reform_memory_from_ground_truth(
    *, adapter: DMLAdapter, storage_dir: str, query: str, ground_truth: dict, tag: str = "rag-reform"
) -> int:
    context = str(ground_truth.get("context") or "").strip()
    if not context:
        return 0

    seen = _load_dedup_index(storage_dir)
    chunks = smart_chunks(context, chunk_chars=700, overlap=80)
    kept = 0
    for chunk in chunks[:6]:
        if not should_keep_chunk(chunk):
            continue
        payload = f"[reformed:{tag}] query={query}\n{chunk}"
        digest = _text_digest(payload)
        if digest in seen:
            continue
        adapter.ingest(
            payload,
            meta={"kind": "note", "source": "ground_truth_reform", "tag": tag},
        )
        _append_dedup_digest(storage_dir, digest)
        seen.add(digest)
        kept += 1
    return kept


def _query_ground_truth_with_timeout(*, adapter: DMLAdapter, query: str, mode: str, timeout_ms: int | None) -> tuple[dict, float]:
    started = time.perf_counter()
    timeout_s = None if timeout_ms is None else max(0.001, timeout_ms / 1000.0)

    if timeout_s is None:
        return adapter.query_database(query, mode=mode), (time.perf_counter() - started) * 1000.0

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(adapter.query_database, query, mode)
    try:
        result = fut.result(timeout=timeout_s)
        return result, (time.perf_counter() - started) * 1000.0
    except FutureTimeoutError:
        fut.cancel()
        raise
    finally:
        # Do not block on executor shutdown after timeout.
        pool.shutdown(wait=False, cancel_futures=True)


def _attach_ground_truth(
    report: dict,
    *,
    adapter: DMLAdapter,
    query: str,
    mode: str,
    strict: bool = False,
    timeout_ms: int | None = None,
) -> None:
    try:
        gt, elapsed_ms = _query_ground_truth_with_timeout(adapter=adapter, query=query, mode=mode, timeout_ms=timeout_ms)
        report["ground_truth"] = gt
        report["ground_truth_status"] = "ok"
        report["ground_truth_latency_ms"] = round(elapsed_ms, 2)
    except FutureTimeoutError:
        report["ground_truth"] = None
        report["ground_truth_status"] = "timeout"
        report["ground_truth_error"] = f"ground truth timed out after {timeout_ms}ms"
        report["ground_truth_timeout_ms"] = timeout_ms
        if strict:
            raise RuntimeError(report["ground_truth_error"])
    except Exception as exc:
        report["ground_truth"] = None
        report["ground_truth_error"] = str(exc)
        report["ground_truth_status"] = "error"
        if strict:
            raise


def cmd_retrieve(args: argparse.Namespace) -> int:
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    started = time.perf_counter()
    try:
        embed_model = None
        with_device = None
        try:
            embed_model = adapter.config.get("embedding_model")
        except Exception:
            embed_model = None
        try:
            model = getattr(getattr(adapter, "embedder", None), "_model", None)
            with_device = str(getattr(model, "device", getattr(model, "_target_device", ""))).strip() or None
        except Exception:
            with_device = None
        query = rewrite_query(args.query) if args.query_expand else args.query
        report = adapter.retrieve_context(
            query,
            tenant_id=args.tenant_id,
            client_id=args.client_id,
            session_id=args.session_id,
            instance_id=args.instance_id,
            top_k=args.top_k,
        )
        items = report.get("items") or []
        raw_lines = ["=== Retrieved Context ==="]
        for item in items:
            text = str(item.get("text") or item.get("summary") or "")
            source = str((item.get("meta") or {}).get("source") or "unknown")
            summary = continuity_handoff_summary(text) or str((item.get("meta") or {}).get("summary") or "").strip() or text.strip()
            raw_lines.append(f"- [{source}] {summary[:220]}")
        if len(raw_lines) > 1:
            matrix_block = ""
            if report.get("raw_context") and "=== Personality Matrix ===" in str(report.get("raw_context")):
                matrix_block = str(report.get("raw_context")).split("=== Retrieved Context ===", 1)[0].strip()
            compact_context = "\n".join(raw_lines)
            report["raw_context"] = f"{matrix_block}\n\n{compact_context}".strip() if matrix_block else compact_context
            report["context_tokens"] = max(1, len(report["raw_context"].split()))

        report["query_original"] = args.query
        report["query_effective"] = query
        report["embedding_provider"] = "local"
        report["embedding_model"] = embed_model
        report["embedding_device"] = with_device

        confidence = _memory_confidence(report, query=query)
        report["memory_confidence"] = round(confidence, 4)

        with_ground_truth = bool(getattr(args, "with_ground_truth", True))
        ground_truth_policy = str(getattr(args, "ground_truth_policy", "low-confidence"))
        confidence_threshold = float(getattr(args, "confidence_threshold", 0.46))
        strict_ground_truth = bool(getattr(args, "strict_ground_truth", False))
        reform_memory = bool(getattr(args, "reform_memory", True))

        need_ground_truth = with_ground_truth and (
            ground_truth_policy == "always"
            or (ground_truth_policy == "low-confidence" and confidence < confidence_threshold)
        )
        report["ground_truth_policy"] = ground_truth_policy
        report["ground_truth_confidence_threshold"] = confidence_threshold
        report["ground_truth_triggered"] = bool(need_ground_truth)
        report["memory_reformed_chunks"] = 0
        if not with_ground_truth:
            report["ground_truth_reason"] = "disabled"
        elif ground_truth_policy == "never":
            report["ground_truth_reason"] = "policy_never"
        elif ground_truth_policy == "always":
            report["ground_truth_reason"] = "policy_always"
        elif need_ground_truth:
            report["ground_truth_reason"] = "low_confidence"
        else:
            report["ground_truth_reason"] = "confidence_ok"

        if need_ground_truth:
            _attach_ground_truth(
                report,
                adapter=adapter,
                query=query,
                mode=args.ground_truth_mode,
                strict=strict_ground_truth,
                timeout_ms=args.ground_truth_timeout_ms,
            )
            if reform_memory and report.get("ground_truth"):
                reformed = _reform_memory_from_ground_truth(
                    adapter=adapter,
                    storage_dir=args.storage_dir,
                    query=query,
                    ground_truth=report["ground_truth"],
                    tag="low_confidence_repair",
                )
                report["memory_reformed_chunks"] = reformed
    finally:
        adapter.close()
    report["retrieve_total_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_backend_proof(args: argparse.Namespace) -> int:
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    try:
        report = _backend_proof(adapter)
        report["status"] = "ok"
        report["action"] = "backend-proof"
    finally:
        adapter.close()
    print(json.dumps(report, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-dir", default=str(DAYSTROM_DML_HOME / "data"))
    parser.add_argument(
        "--config-path",
        default=str(WRAPPER_HOME / "config" / "dml_gpu_only.yaml"),
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
        default=False,
        help="When ground-truth triggers, ingest condensed ground-truth chunks back into memory (default: false)",
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
        default=1800,
        help="Timeout budget for sidecar ground-truth retrieval (default: 1800ms)",
    )
    ret.set_defaults(func=cmd_retrieve)

    proof = sub.add_parser("backend-proof")
    proof.set_defaults(func=cmd_backend_proof)

    mig = sub.add_parser("migration-status")
    mig.set_defaults(func=cmd_migration_status)

    migrate = sub.add_parser("migrate-embeddings")
    migrate.add_argument("--max-items", type=int, default=50)
    migrate.set_defaults(func=cmd_migrate_embeddings)

    return parser


def main() -> int:
    _ensure_gpu_venv_runtime()
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
