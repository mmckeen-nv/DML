#!/usr/bin/env python3
"""Small CLI wrapper to use local DML as an OpenClaw memory substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import errno
import fcntl
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from datetime import datetime, timezone
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


def _compact_value(value: str, *, limit: int) -> str:
    clean = " ".join(value.split()).strip()
    if len(clean) <= limit:
        return clean
    shortened = clean[: max(0, limit - 3)].rstrip()
    cut = max(shortened.rfind(". "), shortened.rfind("; "), shortened.rfind(", "), shortened.rfind(" "))
    if cut >= max(12, limit // 2):
        shortened = shortened[:cut].rstrip()
    return shortened.rstrip(" ,;:.") + "..."


def _cheap_summary(text: str, *, max_chars: int) -> str:
    clean = _normalize_text_for_dedup(text)
    if len(clean) <= max_chars:
        return clean
    return _compact_value(clean, limit=max_chars)


def _line_value(text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix) :].strip()
            return value or None
    return None


def _structured_summary_from_meta_or_text(meta: dict, text: str, *, max_chars: int) -> str | None:
    continuity_summary = continuity_handoff_summary(text, max_len=max_chars)
    if continuity_summary:
        return continuity_summary

    fields = {
        "thread": str(meta.get("thread") or _line_value(text, "thread") or "").strip(),
        "state": str(meta.get("state") or _line_value(text, "state") or "").strip(),
        "task": str(meta.get("task") or _line_value(text, "task") or "").strip(),
        "next": str(meta.get("next_action") or _line_value(text, "next_action") or "").strip(),
    }
    if not any(fields.values()):
        return None
    parts = []
    for label, value in fields.items():
        if value and value.lower() not in {"unknown", "none", "null"}:
            parts.append(f"{label}: {_compact_value(value, limit=64)}")
    if not parts:
        return None
    return _compact_value(" | ".join(parts), limit=max_chars)


def _apply_summary_policy(base_meta: dict, text: str, *, policy: str, max_chars: int) -> tuple[dict, str]:
    meta = dict(base_meta)
    if str(meta.get("summary") or "").strip():
        return meta, "cheap"
    if meta.get("skip_summary"):
        return meta, "skip"
    if policy == "llm":
        return meta, "llm"
    if policy == "skip":
        meta["skip_summary"] = True
        return meta, "skip"

    structured = _structured_summary_from_meta_or_text(meta, text, max_chars=max_chars)
    if structured:
        meta["summary"] = structured
        meta.setdefault("summary_source", "deterministic")
        return meta, "cheap"

    if policy == "cheap" or len(_normalize_text_for_dedup(text)) <= max_chars:
        meta["summary"] = _cheap_summary(text, max_chars=max_chars)
        meta.setdefault("summary_source", "deterministic")
        return meta, "cheap"

    return meta, "llm"


def _text_digest(text: str) -> str:
    normalized = _normalize_text_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _dedup_index_path(storage_dir: str) -> Path:
    return Path(storage_dir) / ".ingest_dedup_sha256.txt"


def _state_file_path(storage_dir: str) -> Path:
    return Path(storage_dir) / "dml_state.jsonl"


def _lock_file_path(storage_dir: str) -> Path:
    return Path(storage_dir) / ".dml_store.lock"


def _lock_metadata_path(storage_dir: str) -> Path:
    return Path(storage_dir) / ".dml_store.lock.json"


def _read_lock_metadata(storage_dir: str) -> dict | None:
    path = _lock_metadata_path(storage_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@contextmanager
def _store_write_lock(storage_dir: str, *, operation: str, timeout_ms: int = 0):
    lock_path = _lock_file_path(storage_dir)
    lock_meta_path = _lock_metadata_path(storage_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                waited_ms = (time.perf_counter() - started) * 1000.0
                if timeout_ms <= 0 or waited_ms >= timeout_ms:
                    holder = _read_lock_metadata(storage_dir) or {}
                    raise TimeoutError(
                        json.dumps(
                            {
                                "lock_path": str(lock_path),
                                "operation": operation,
                                "waited_ms": round(waited_ms, 2),
                                "holder": holder,
                            },
                            sort_keys=True,
                        )
                    )
                time.sleep(min(0.05, max(0.005, (timeout_ms - waited_ms) / 1000.0)))
        meta = {
            "operation": operation,
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lock_path": str(lock_path),
        }
        lock_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        yield {"path": str(lock_path), "metadata_path": str(lock_meta_path), **meta}
    finally:
        if acquired:
            try:
                if _read_lock_metadata(storage_dir) and lock_meta_path.exists():
                    lock_meta_path.unlink()
            except Exception:
                pass
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _lock_failure_report(action: str, exc: TimeoutError, started: float) -> dict:
    details: dict = {}
    try:
        details = json.loads(str(exc))
    except Exception:
        details = {"error": str(exc)}
    return {
        "status": "blocked",
        "action": action,
        "error": "store_write_lock_held",
        "lock": details,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _backup_root(storage_dir: str, backup_dir: str | None = None) -> Path:
    return Path(backup_dir).expanduser() if backup_dir else Path(storage_dir) / "backups"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_exists(src: Path, dest: Path) -> dict | None:
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "source": str(src),
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": _sha256_file(dest),
    }


def _prune_backups(root: Path, *, keep: int) -> list[str]:
    if keep <= 0 or not root.exists():
        return []
    manifests = sorted(root.glob("*/backup_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[str] = []
    for manifest in manifests[keep:]:
        backup_dir = manifest.parent
        shutil.rmtree(backup_dir, ignore_errors=True)
        removed.append(str(backup_dir))
    return removed


def _create_backup(storage_dir: str, *, backup_dir: str | None = None, label: str = "manual", keep: int = 20) -> dict:
    state_path = _state_file_path(storage_dir)
    if not state_path.exists():
        raise FileNotFoundError(f"state file missing: {state_path}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label).strip("-") or "manual"
    root = _backup_root(storage_dir, backup_dir)
    target = root / f"{ts}-{safe_label}"
    target.mkdir(parents=True, exist_ok=False)

    files = []
    for src, name in [
        (state_path, "dml_state.jsonl"),
        (_dedup_index_path(storage_dir), ".ingest_dedup_sha256.txt"),
        (Path(storage_dir) / "embedding_compatibility_report.json", "embedding_compatibility_report.json"),
        (Path(storage_dir) / "dpm_preference_graph.json", "dpm_preference_graph.json"),
    ]:
        copied = _copy_if_exists(src, target / name)
        if copied:
            files.append(copied)

    manifest = {
        "schema_version": "dml.backup-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": safe_label,
        "storage_dir": str(storage_dir),
        "backup_dir": str(target),
        "files": files,
    }
    manifest_path = target / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["pruned_backups"] = _prune_backups(root, keep=keep)
    return manifest


def _read_state_health(storage_dir: str) -> dict:
    path = _state_file_path(storage_dir)
    report = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "checksum_ok": False,
        "header_count": 0,
        "record_count": 0,
        "count_ok": False,
        "embedding_dimensions": [],
        "active_continuity_count": 0,
        "quarantined_count": 0,
        "summary_count": 0,
        "unscoped_count": 0,
        "records_by_tenant": {},
        "active_continuity_by_tenant": {},
        "errors": [],
    }
    if not path.exists():
        report["errors"].append("state_file_missing")
        return report
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        report["readable"] = True
    except Exception as exc:
        report["errors"].append(f"state_file_unreadable: {exc}")
        return report
    if not lines:
        report["errors"].append("state_file_empty")
        return report
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        report["errors"].append(f"invalid_header_json: {exc}")
        return report

    report["type"] = header.get("type")
    report["version"] = header.get("version")
    report["created_at"] = header.get("created_at")
    report["header_count"] = int(header.get("count") or 0)
    payload_lines = lines[1:]
    expected_checksum = str(header.get("checksum") or "")
    actual_checksum = hashlib.sha256("\n".join(payload_lines).encode("utf-8")).hexdigest()
    report["checksum"] = {"expected": expected_checksum, "actual": actual_checksum}
    report["checksum_ok"] = bool(expected_checksum and expected_checksum == actual_checksum)
    if not report["checksum_ok"]:
        report["errors"].append("checksum_mismatch")

    dims: set[int] = set()
    record_count = 0
    for index, raw in enumerate(payload_lines, start=2):
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            report["errors"].append(f"invalid_record_json_line_{index}: {exc}")
            continue
        record_count += 1
        embedding = record.get("embedding") or []
        if isinstance(embedding, list):
            dims.add(len(embedding))
        meta = record.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        if meta.get("namespace") == "active_continuity" or meta.get("source") in CONTINUITY_SOURCES:
            report["active_continuity_count"] += 1
            tenant_key = str(meta.get("tenant_id") or "__unscoped__")
            report["active_continuity_by_tenant"][tenant_key] = report["active_continuity_by_tenant"].get(tenant_key, 0) + 1
        tenant_id = meta.get("tenant_id")
        if tenant_id is None:
            report["unscoped_count"] += 1
            tenant_key = "__unscoped__"
        else:
            tenant_key = str(tenant_id)
        report["records_by_tenant"][tenant_key] = report["records_by_tenant"].get(tenant_key, 0) + 1
        if str(meta.get("memory_state") or "").lower() in {"quarantined", "suppressed", "deleted"}:
            report["quarantined_count"] += 1
        if str(meta.get("summary") or "").strip():
            report["summary_count"] += 1

    report["record_count"] = record_count
    report["count_ok"] = report["header_count"] == record_count
    if not report["count_ok"]:
        report["errors"].append("record_count_mismatch")
    report["embedding_dimensions"] = sorted(dims)
    if len(dims) > 1:
        report["errors"].append("mixed_embedding_dimensions")
    report["summary_ratio"] = round(report["summary_count"] / record_count, 4) if record_count else 1.0
    return report


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
    started = time.perf_counter()
    meta = _parse_meta(args.meta)
    try:
        lock_ctx = _store_write_lock(args.storage_dir, operation="ingest", timeout_ms=args.lock_timeout_ms)
        lock = lock_ctx.__enter__()
    except TimeoutError as exc:
        print(json.dumps(_lock_failure_report("ingest", exc, started), indent=2, default=str))
        return 2
    adapter = None
    try:
        adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
        seen = _load_dedup_index(args.storage_dir)
        payload_meta = {
            "tenant_id": args.tenant_id,
            "client_id": args.client_id,
            "session_id": args.session_id,
            "instance_id": args.instance_id,
            **meta,
            "kind": _kind(args.kind).value,
        }
        chunks = smart_chunks(args.text, chunk_chars=max(180, args.chunk_chars), overlap=max(0, args.chunk_overlap)) if args.chunk else [args.text]
        kept = 0
        skipped_duplicate = 0
        cheap_summaries = 0
        skipped_summaries = 0
        llm_summaries_allowed = 0
        for chunk in chunks:
            if args.filter_noise and not should_keep_chunk(chunk):
                continue
            digest = _text_digest(chunk)
            if digest in seen:
                skipped_duplicate += 1
                continue
            chunk_meta, summary_mode = _apply_summary_policy(
                payload_meta,
                chunk,
                policy=args.summary_policy,
                max_chars=args.summary_max_chars,
            )
            if summary_mode == "cheap":
                cheap_summaries += 1
            elif summary_mode == "skip":
                skipped_summaries += 1
            else:
                llm_summaries_allowed += 1
            adapter.ingest(chunk, meta=chunk_meta, persist=False)
            if chunk_meta.get("dpm_preference"):
                adapter.record_personality_preference(
                    chunk,
                    scope=str(chunk_meta.get("dpm_scope") or "relationship"),
                    source_id=str(chunk_meta.get("source") or "wrapper:ingest"),
                    explicit=True,
                    meta=chunk_meta,
                )
            _append_dedup_digest(args.storage_dir, digest)
            seen.add(digest)
            kept += 1
        adapter._persist_all()
    finally:
        if adapter is not None:
            adapter.close()
        lock_ctx.__exit__(None, None, None)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "ingest",
                "kind": args.kind,
                "chunks_ingested": kept,
                "chunks_skipped_duplicate": skipped_duplicate,
                "summary_policy": args.summary_policy,
                "cheap_summaries": cheap_summaries,
                "skipped_summaries": skipped_summaries,
                "llm_summaries_allowed": llm_summaries_allowed,
                "lock": lock,
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
    started = time.perf_counter()
    try:
        lock_ctx = _store_write_lock(args.storage_dir, operation="migrate-embeddings", timeout_ms=args.lock_timeout_ms)
        lock = lock_ctx.__enter__()
    except TimeoutError as exc:
        print(json.dumps(_lock_failure_report("migrate-embeddings", exc, started), indent=2, default=str))
        return 2
    storage_dir = Path(args.storage_dir)
    persistence_path = storage_dir / "dml_state.jsonl"
    adapter = None
    try:
        if not persistence_path.exists():
            print(
                json.dumps(
                    {
                        "status": "missing",
                        "action": "migrate-embeddings",
                        "storage_dir": str(storage_dir),
                        "message": "no persisted dml_state.jsonl found",
                        "lock": lock,
                    },
                    indent=2,
                )
            )
            return 0
        from daystrom_dml.persistence import load_state, save_state  # type: ignore

        items = load_state(persistence_path)
        adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
        payload = {"items": [item.to_dict() for item in items]}
        report = adapter._ensure_embedding_compatibility(payload, max_items=args.max_items)
        adapter.store.import_state(payload)
        save_state(adapter.store.items(), persistence_path)
    finally:
        if adapter is not None:
            adapter.close()
        lock_ctx.__exit__(None, None, None)
    report["action"] = "migrate-embeddings"
    report["storage_dir"] = str(storage_dir)
    report["lock"] = lock
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
            include_quarantined=bool(getattr(args, "include_quarantined", False)),
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


CONTINUITY_SOURCES = {
    "rolling_thread_checkpoint",
    "continuity_checkpoint",
    "dpm_continuity_checkpoint",
}


def _is_active_continuity_item(item: dict) -> bool:
    meta = item.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    source = str(meta.get("source") or "").strip()
    namespace = str(meta.get("namespace") or "").strip()
    memory_state = str(meta.get("memory_state") or "active").strip().lower()
    if memory_state in {"quarantined", "suppressed", "deleted"}:
        return False
    return namespace == "active_continuity" or source in CONTINUITY_SOURCES


def _continuity_resume_context(items: list[dict]) -> tuple[str, dict]:
    raw_lines = ["=== Active Continuity Resume ==="]
    latest: dict[str, str] = {}
    for item in items:
        text = str(item.get("text") or item.get("summary") or "")
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        source = str(meta.get("source") or "unknown")
        summary = continuity_handoff_summary(text) or str(meta.get("summary") or "").strip() or text.strip()
        thread = str(meta.get("thread") or "").strip() or _line_value(text, "thread")
        state = str(meta.get("state") or "").strip() or _line_value(text, "state")
        task = str(meta.get("task") or "").strip() or _line_value(text, "task")
        next_action = str(meta.get("next_action") or "").strip() or _line_value(text, "next_action")
        updated_at = (
            str(meta.get("updated_at") or meta.get("captured_at") or "").strip()
            or _line_value(text, "updated_at")
            or _line_value(text, "captured_at")
            or None
        )
        if not latest and any([thread, state, task, next_action, updated_at]):
            latest = {
                k: v
                for k, v in {
                    "thread": thread,
                    "state": state,
                    "task": task,
                    "next_action": next_action,
                    "updated_at": updated_at,
                }.items()
                if v
            }
        label = source
        if thread:
            label = f"{label}:{thread}"
        raw_lines.append(f"- [{label}] {summary[:260]}")
    return "\n".join(raw_lines), latest


def cmd_resume(args: argparse.Namespace) -> int:
    adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
    started = time.perf_counter()
    try:
        report = adapter.retrieve_context(
            args.query,
            tenant_id=args.tenant_id,
            client_id=args.client_id,
            session_id=args.session_id,
            instance_id=args.instance_id,
            top_k=args.top_k,
            include_quarantined=False,
        )
    finally:
        adapter.close()

    items = [item for item in (report.get("items") or []) if isinstance(item, dict)]
    continuity_items = [item for item in items if _is_active_continuity_item(item)]
    context_items = continuity_items or items[: max(0, args.fallback_items)]
    raw_context, latest = _continuity_resume_context(context_items)

    report.update(
        {
            "status": report.get("status", "ok"),
            "action": "resume",
            "query_original": args.query,
            "items_seen": len(items),
            "continuity_items": len(continuity_items),
            "fallback_used": not bool(continuity_items),
            "latest_checkpoint": latest,
            "raw_context": raw_context,
            "context_tokens": max(1, len(raw_context.split())),
            "resume_total_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    )
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


def cmd_health(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    state = _read_state_health(args.storage_dir)
    dedup_path = _dedup_index_path(args.storage_dir)
    migration_report = Path(args.storage_dir) / "embedding_compatibility_report.json"
    queue_status_path = WORKSPACE / "out" / "continuity-ingest-status.json"
    report = {
        "status": "ok",
        "action": "health",
        "contract_version": "dml-agent-memory-v1",
        "storage_dir": args.storage_dir,
        "config_path": args.config_path,
        "state": state,
        "dedup_index": {
            "path": str(dedup_path),
            "exists": dedup_path.exists(),
        },
        "migration_report": {
            "path": str(migration_report),
            "exists": migration_report.exists(),
        },
        "continuity_worker": {
            "status_path": str(queue_status_path),
            "status_exists": queue_status_path.exists(),
        },
        "store_lock": {
            "path": str(_lock_file_path(args.storage_dir)),
            "metadata_path": str(_lock_metadata_path(args.storage_dir)),
            "metadata": _read_lock_metadata(args.storage_dir),
        },
        "backend": None,
        "latency_ms": 0.0,
    }
    errors = list(state.get("errors") or [])
    if args.probe_backend:
        adapter = _adapter(args.storage_dir, args.config_path, args.require_gpu)
        try:
            backend = _backend_proof(adapter)
            backend["status"] = "ok"
            report["backend"] = backend
        except Exception as exc:
            errors.append(f"backend_probe_failed: {exc}")
            report["backend"] = {"status": "error", "error": str(exc)}
        finally:
            adapter.close()

    if state.get("exists") and not errors:
        report["status"] = "ok"
    elif state.get("exists") and state.get("readable"):
        report["status"] = "degraded"
    else:
        report["status"] = "fail"
    report["errors"] = errors
    report["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] in {"ok", "degraded"} else 1


def cmd_backup(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    try:
        with _store_write_lock(args.storage_dir, operation="backup", timeout_ms=args.lock_timeout_ms) as lock:
            manifest = _create_backup(
                args.storage_dir,
                backup_dir=args.backup_dir,
                label=args.label,
                keep=args.keep,
            )
        report = {
            "status": "ok",
            "action": "backup",
            "contract_version": "dml-agent-memory-v1",
            "backup": manifest,
            "lock": lock,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        print(json.dumps(report, indent=2, default=str))
        return 0
    except TimeoutError as exc:
        print(json.dumps(_lock_failure_report("backup", exc, started), indent=2, default=str))
        return 2
    except Exception as exc:
        report = {
            "status": "fail",
            "action": "backup",
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        print(json.dumps(report, indent=2, default=str))
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    state = _read_state_health(args.storage_dir)
    errors = list(state.get("errors") or [])
    loaded_count = None
    try:
        from daystrom_dml.persistence import load_state  # type: ignore

        items = load_state(_state_file_path(args.storage_dir))
        loaded_count = len(items)
        if loaded_count != state.get("record_count"):
            errors.append("loader_count_mismatch")
    except Exception as exc:
        errors.append(f"persistence_loader_failed: {exc}")

    suggestions = []
    if "state_file_missing" in errors:
        suggestions.append("restore from the latest verified backup")
    if "checksum_mismatch" in errors or "record_count_mismatch" in errors:
        suggestions.append("run restore with a backup whose manifest checksum matches")
    if "mixed_embedding_dimensions" in errors:
        suggestions.append("run migrate-embeddings or restore a store with a single embedding dimension")
    if any(error.startswith("persistence_loader_failed") for error in errors):
        suggestions.append("inspect invalid JSONL records or restore from backup")

    report = {
        "status": "ok" if not errors else "fail",
        "action": "verify",
        "contract_version": "dml-agent-memory-v1",
        "storage_dir": args.storage_dir,
        "state": state,
        "loaded_count": loaded_count,
        "errors": errors,
        "suggestions": suggestions,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "ok" else 1


def _load_backup_manifest(path: str) -> tuple[Path, dict]:
    p = Path(path).expanduser()
    if p.is_dir():
        manifest_path = p / "backup_manifest.json"
    else:
        manifest_path = p
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent, manifest


def _manifest_file(manifest: dict, name: str) -> dict | None:
    for file_report in manifest.get("files") or []:
        if Path(str(file_report.get("path") or "")).name == name:
            return file_report
    return None


def cmd_restore(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    try:
        with _store_write_lock(args.storage_dir, operation="restore", timeout_ms=args.lock_timeout_ms) as lock:
            backup_path, manifest = _load_backup_manifest(args.backup)
            state_entry = _manifest_file(manifest, "dml_state.jsonl")
            if not state_entry:
                raise FileNotFoundError("backup manifest does not include dml_state.jsonl")
            source_state = Path(str(state_entry["path"])).expanduser()
            if not source_state.is_absolute():
                source_state = backup_path / source_state
            if not source_state.exists():
                raise FileNotFoundError(f"backup state missing: {source_state}")
            actual = _sha256_file(source_state)
            expected = str(state_entry.get("sha256") or "")
            if expected and actual != expected:
                raise ValueError("backup state checksum mismatch")

            pre_restore = None
            if _state_file_path(args.storage_dir).exists() and not args.no_pre_restore_backup:
                pre_restore = _create_backup(
                    args.storage_dir,
                    backup_dir=args.backup_dir,
                    label="pre-restore",
                    keep=args.keep,
                )

            target = _state_file_path(args.storage_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".restore-tmp")
            shutil.copy2(source_state, tmp)
            tmp.replace(target)

            for optional_name in [".ingest_dedup_sha256.txt", "embedding_compatibility_report.json", "dpm_preference_graph.json"]:
                entry = _manifest_file(manifest, optional_name)
                if not entry:
                    continue
                source = Path(str(entry["path"])).expanduser()
                if not source.is_absolute():
                    source = backup_path / source
                if not source.exists():
                    continue
                dest = Path(args.storage_dir) / optional_name
                tmp_optional = dest.with_suffix(dest.suffix + ".restore-tmp") if dest.suffix else dest.with_name(dest.name + ".restore-tmp")
                shutil.copy2(source, tmp_optional)
                tmp_optional.replace(dest)

            verify = _read_state_health(args.storage_dir)
        status = "ok" if not verify.get("errors") else "degraded"
        report = {
            "status": status,
            "action": "restore",
            "contract_version": "dml-agent-memory-v1",
            "restored_from": str(source_state),
            "pre_restore_backup": pre_restore,
            "state": verify,
            "lock": lock,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        print(json.dumps(report, indent=2, default=str))
        return 0 if status == "ok" else 1
    except TimeoutError as exc:
        print(json.dumps(_lock_failure_report("restore", exc, started), indent=2, default=str))
        return 2
    except Exception as exc:
        report = {
            "status": "fail",
            "action": "restore",
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        print(json.dumps(report, indent=2, default=str))
        return 1


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
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=0,
        help="Milliseconds to wait for the shared store write lock before returning blocked (default: 0)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest")
    ing.add_argument("--text", required=True)
    ing.add_argument("--kind", default="action")
    ing.add_argument("--meta", help="JSON object")
    ing.add_argument("--tenant-id", default="openclaw")
    ing.add_argument("--client-id")
    ing.add_argument("--session-id")
    ing.add_argument("--instance-id")
    ing.add_argument("--chunk", action=argparse.BooleanOptionalAction, default=True)
    ing.add_argument("--chunk-chars", type=int, default=620)
    ing.add_argument("--chunk-overlap", type=int, default=90)
    ing.add_argument("--filter-noise", action=argparse.BooleanOptionalAction, default=True)
    ing.add_argument(
        "--summary-policy",
        default="auto",
        choices=["auto", "llm", "cheap", "skip"],
        help="How ingest populates cached summaries before storage (default: auto)",
    )
    ing.add_argument(
        "--summary-max-chars",
        type=int,
        default=220,
        help="Maximum deterministic summary length for auto/cheap policies",
    )
    ing.set_defaults(func=cmd_ingest)

    ret = sub.add_parser("retrieve")
    ret.add_argument("--query", required=True)
    ret.add_argument("--top-k", type=int, default=6)
    ret.add_argument("--tenant-id", default="openclaw")
    ret.add_argument("--client-id")
    ret.add_argument("--session-id")
    ret.add_argument("--instance-id")
    ret.add_argument(
        "--include-quarantined",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include quarantined/suppressed memories in retrieval results (default: false)",
    )
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

    resume = sub.add_parser("resume")
    resume.add_argument(
        "--query",
        default="active continuity checkpoint compaction handoff resume next action",
        help="Continuity-focused retrieval query",
    )
    resume.add_argument("--top-k", type=int, default=12)
    resume.add_argument("--tenant-id", default="openclaw")
    resume.add_argument("--client-id")
    resume.add_argument("--session-id")
    resume.add_argument("--instance-id")
    resume.add_argument(
        "--fallback-items",
        type=int,
        default=3,
        help="Generic retrieved items to return if no active continuity checkpoint is found",
    )
    resume.set_defaults(func=cmd_resume)

    proof = sub.add_parser("backend-proof")
    proof.set_defaults(func=cmd_backend_proof)

    health = sub.add_parser("health")
    health.add_argument(
        "--probe-backend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Instantiate the adapter and verify embedding/LLM backend surfaces",
    )
    health.set_defaults(func=cmd_health)

    backup = sub.add_parser("backup")
    backup.add_argument("--backup-dir")
    backup.add_argument("--label", default="manual")
    backup.add_argument("--keep", type=int, default=20)
    backup.set_defaults(func=cmd_backup)

    verify = sub.add_parser("verify")
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("restore")
    restore.add_argument("--backup", required=True, help="Backup directory or backup_manifest.json path")
    restore.add_argument("--backup-dir", help="Where to place the pre-restore backup")
    restore.add_argument("--keep", type=int, default=20)
    restore.add_argument("--no-pre-restore-backup", dest="no_pre_restore_backup", action="store_true")
    restore.set_defaults(no_pre_restore_backup=False, func=cmd_restore)

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
