#!/usr/bin/env python3
"""Import legacy ``newdml`` archive memories into a current DML store.

The legacy archive stores DML/RAG payloads as JSON with old embeddings. This
tool intentionally ignores those embeddings and re-embeds memory text through
the current DML adapter so imported memories match the active runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEGACY_DML_MEMBER = "newdml/data/dml_store.json"
REPORT_NAME = "newdml_import_report.json"
DEDUP_NAME = ".ingest_dedup_sha256.txt"
HEXISH_RE = re.compile(r"\b[a-f0-9]{32,}\b", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    root = _repo_root()
    candidates = [
        root / "dml_core",
        root / "openclaw-wrapper" / "scripts",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _text_digest(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8", errors="ignore")).hexdigest()


def _read_json_member(archive: Path, member_name: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:*") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError as exc:
            raise SystemExit(f"archive is missing {member_name}") from exc
        extracted = tar.extractfile(member)
        if extracted is None:
            raise SystemExit(f"could not read {member_name} from archive")
        return json.load(extracted)


def _load_legacy_items(archive: Path) -> list[dict[str, Any]]:
    payload = _read_json_member(archive, LEGACY_DML_MEMBER)
    items = payload.get("items")
    if not isinstance(items, list):
        raise SystemExit(f"{LEGACY_DML_MEMBER} does not contain an items list")
    return [item for item in items if isinstance(item, dict)]


def _load_existing_digests(storage_dir: Path) -> set[str]:
    digests: set[str] = set()
    dedup_path = storage_dir / DEDUP_NAME
    if dedup_path.exists():
        digests.update(
            line.strip()
            for line in dedup_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        )

    state_path = storage_dir / "dml_state.jsonl"
    if state_path.exists():
        for raw in state_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = str(record.get("text") or "")
            if text.strip():
                digests.add(_text_digest(text))

    json_state_path = storage_dir / "dml_store.json"
    if json_state_path.exists():
        try:
            payload = json.loads(json_state_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for entry in payload.get("items") or []:
            if isinstance(entry, dict):
                text = str(entry.get("text") or "")
                if text.strip():
                    digests.add(_text_digest(text))
    return digests


def _looks_like_metadata_noise(item: dict[str, Any], text: str) -> bool:
    meta = item.get("meta") or {}
    doc_path = str(meta.get("doc_path") or "").lower()
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if doc_path.endswith(".csv"):
        return True

    comma_count = normalized.count(",")
    hex_hits = len(HEXISH_RE.findall(normalized))
    words = WORD_RE.findall(normalized)
    word_count = len(words)
    alpha_chars = sum(1 for ch in normalized if ch.isalpha())
    digit_chars = sum(1 for ch in normalized if ch.isdigit())
    punctuation_chars = sum(1 for ch in normalized if not ch.isalnum() and not ch.isspace())
    alpha_ratio = alpha_chars / max(1, len(normalized))
    digit_ratio = digit_chars / max(1, len(normalized))
    punctuation_ratio = punctuation_chars / max(1, len(normalized))
    raw_tokens = normalized.split()
    noisy_tokens = [
        token
        for token in raw_tokens
        if len(token) > 24
        or not re.search(r"[A-Za-z]", token)
        or (sum(1 for ch in token if ch.isdigit()) / max(1, len(token))) > 0.30
        or (sum(1 for ch in token if not ch.isalnum() and ch not in {"'", "-", "’"}) / max(1, len(token))) > 0.25
    ]
    noisy_token_ratio = len(noisy_tokens) / max(1, len(raw_tokens))
    if hex_hits >= 4:
        return True
    if comma_count >= 18 and alpha_ratio < 0.45:
        return True
    if word_count < 18 and len(normalized) > 240:
        return True
    if len(normalized) > 180 and digit_ratio > 0.18:
        return True
    if len(normalized) > 180 and punctuation_ratio > 0.16:
        return True
    if len(raw_tokens) >= 20 and noisy_token_ratio > 0.34:
        return True
    return False


def _iter_importable_items(
    items: Iterable[dict[str, Any]],
    *,
    filter_noise: bool,
) -> Iterable[tuple[dict[str, Any], str, str]]:
    seen_in_archive: set[str] = set()
    for item in items:
        text = str(item.get("text") or "")
        normalized = _normalize_text(text)
        if not normalized:
            continue
        if filter_noise and _looks_like_metadata_noise(item, normalized):
            continue
        digest = _text_digest(normalized)
        if digest in seen_in_archive:
            continue
        seen_in_archive.add(digest)
        yield item, text, digest


def _short_summary(text: str, max_len: int = 256) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _quality_score(item: dict[str, Any], text: str) -> float:
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0
    meta = item.get("meta") or {}
    score = 1.0
    if _looks_like_metadata_noise(item, normalized):
        score -= 0.55
    if str(meta.get("doc_path") or "").lower().endswith(".csv"):
        score -= 0.25
    hex_hits = len(HEXISH_RE.findall(normalized))
    if hex_hits:
        score -= min(0.35, hex_hits * 0.08)
    raw_tokens = normalized.split()
    glued = sum(1 for token in raw_tokens if len(token) > 18 and re.search(r"[A-Za-z]", token))
    if raw_tokens:
        score -= min(0.25, (glued / len(raw_tokens)) * 0.8)
    return max(0.0, min(1.0, round(score, 4)))


def _legacy_meta(item: dict[str, Any], archive: Path, text: str, *, target_state: str) -> dict[str, Any]:
    legacy_meta = dict(item.get("meta") or {})
    meta: dict[str, Any] = {
        "kind": "note",
        "namespace": "legacy_archive",
        "memory_state": target_state,
        "source": "old_openclaw_newdml_archive",
        "archive": archive.name,
        "quality_score": _quality_score(item, text),
        "legacy_id": item.get("id"),
        "legacy_timestamp": item.get("timestamp"),
        "legacy_level": item.get("level"),
        "legacy_salience": item.get("salience"),
        "legacy_fidelity": item.get("fidelity"),
        "summary": _short_summary(text),
        "skip_summary": True,
    }
    if legacy_meta.get("doc_path"):
        meta["legacy_doc_path"] = legacy_meta.get("doc_path")
    if legacy_meta.get("merges") is not None:
        meta["legacy_merges"] = legacy_meta.get("merges")
    if legacy_meta.get("prompt"):
        meta["legacy_prompt"] = legacy_meta.get("prompt")
    if legacy_meta.get("response_excerpt"):
        meta["legacy_response_excerpt"] = str(legacy_meta.get("response_excerpt"))[:500]
    return {k: v for k, v in meta.items() if v is not None}


def _backup_store(storage_dir: Path) -> Path:
    backup_dir = storage_dir / "backups" / f"pre-newdml-import-{_utc_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "dml_state.jsonl",
        "dml_store.json",
        "rag_store.json",
        "rag_meta.json",
        "rag_index.faiss",
        "embedding_compatibility_report.json",
        DEDUP_NAME,
    ]:
        source = storage_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    return backup_dir


def _write_report(storage_dir: Path, report: dict[str, Any]) -> Path:
    path = storage_dir / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_digests(storage_dir: Path, digests: list[str]) -> None:
    if not digests:
        return
    path = storage_dir / DEDUP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for digest in digests:
            handle.write(digest + "\n")


def _dry_run(args: argparse.Namespace, items: list[dict[str, Any]], existing_digests: set[str]) -> dict[str, Any]:
    raw_unique = list(_iter_importable_items(items, filter_noise=False))
    importable = list(_iter_importable_items(items, filter_noise=args.filter_noise))
    duplicates = sum(1 for _, _, digest in importable if digest in existing_digests)
    selected = [entry for entry in importable if entry[2] not in existing_digests]
    if args.limit is not None:
        selected = selected[: args.limit]
    doc_paths: dict[str, int] = {}
    dims: dict[str, int] = {}
    for item, _, _ in importable:
        dim = str(len(item.get("embedding") or []))
        dims[dim] = dims.get(dim, 0) + 1
        doc_path = str((item.get("meta") or {}).get("doc_path") or "")
        if doc_path:
            doc_paths[doc_path] = doc_paths.get(doc_path, 0) + 1
    return {
        "status": "dry-run",
        "archive": str(args.archive),
        "storage_dir": str(args.storage_dir),
        "legacy_items": len(items),
        "noise_filter_enabled": bool(args.filter_noise),
        "target_state": args.target_state,
        "skipped_by_noise_filter": len(raw_unique) - len(importable),
        "importable_unique_texts": len(importable),
        "duplicates_in_target": duplicates,
        "selected_for_import": len(selected),
        "legacy_embedding_dims": dims,
        "top_legacy_doc_paths": sorted(doc_paths.items(), key=lambda pair: pair[1], reverse=True)[:20],
    }


def _apply(args: argparse.Namespace, items: list[dict[str, Any]], existing_digests: set[str]) -> dict[str, Any]:
    _ensure_import_path()
    from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore
    from daystrom_dml.summarizer import DummySummarizer  # type: ignore

    storage_dir = Path(args.storage_dir).expanduser().resolve()
    backup_dir = _backup_store(storage_dir)
    selected = [
        entry
        for entry in _iter_importable_items(items, filter_noise=args.filter_noise)
        if entry[2] not in existing_digests
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    adapter = DMLAdapter(
        config_path=str(args.config_path) if args.config_path else None,
        config_overrides={
            "storage_dir": str(storage_dir),
            "strict_embedding_required": True,
            "strict_llm_required": False,
            "dml.agentic_mode.enabled": True,
        },
        summarizer=DummySummarizer(),
    )
    imported = 0
    failed = 0
    failure_samples: list[str] = []
    appended_digests: list[str] = []
    started = time.perf_counter()
    try:
        for item, text, digest in selected:
            try:
                meta = _legacy_meta(item, Path(args.archive), text, target_state=args.target_state)
                adapter.ingest(text, meta=meta, persist=False)
            except Exception:
                failed += 1
                failure_samples.append(str(item.get("id")))
                if args.stop_on_error:
                    raise
                continue
            imported += 1
            appended_digests.append(digest)
            if args.progress_every and imported % args.progress_every == 0:
                print(json.dumps({"event": "progress", "imported": imported, "failed": failed}), flush=True)
        adapter._persist_all()
    finally:
        adapter.close()
    _append_digests(storage_dir, appended_digests)
    elapsed = time.perf_counter() - started
    return {
        "status": "applied",
        "archive": str(args.archive),
        "storage_dir": str(storage_dir),
        "backup_dir": str(backup_dir),
        "selected_for_import": len(selected),
        "imported": imported,
        "failed": failed,
        "failure_legacy_ids": failure_samples[:20],
        "noise_filter_enabled": bool(args.filter_noise),
        "target_state": args.target_state,
        "elapsed_sec": round(elapsed, 3),
        "avg_sec_per_import": round(elapsed / imported, 4) if imported else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--storage-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--filter-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-state",
        choices=("quarantined", "active"),
        default="quarantined",
        help="Lifecycle state assigned to imported memories (default: quarantined)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.archive = Path(args.archive).expanduser().resolve()
    args.storage_dir = Path(args.storage_dir).expanduser().resolve()
    if args.config_path:
        args.config_path = Path(args.config_path).expanduser().resolve()
    if not args.archive.exists():
        raise SystemExit(f"archive not found: {args.archive}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    items = _load_legacy_items(args.archive)
    existing_digests = _load_existing_digests(args.storage_dir)
    report = (
        _dry_run(args, items, existing_digests)
        if args.dry_run
        else _apply(args, items, existing_digests)
    )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report_path = _write_report(args.storage_dir, report)
    report["report_path"] = str(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
