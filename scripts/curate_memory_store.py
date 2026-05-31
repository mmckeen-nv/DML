#!/usr/bin/env python3
"""Review and curate DML memory lifecycle states."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    core = _repo_root() / "dml_core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))


def _state_path(storage_dir: Path) -> Path:
    return storage_dir.expanduser().resolve() / "dml_state.jsonl"


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = path.parent / "backups" / f"pre-curation-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return target


def _load_items(storage_dir: Path):
    _ensure_import_path()
    from daystrom_dml.persistence import load_state  # type: ignore

    return load_state(_state_path(storage_dir))


def _save_items(storage_dir: Path, items) -> Path:
    _ensure_import_path()
    from daystrom_dml.persistence import save_state  # type: ignore

    return save_state(items, _state_path(storage_dir))


def _state(item) -> str:
    meta = item.meta or {}
    return str(meta.get("memory_state") or meta.get("lifecycle_state") or "active").strip().lower() or "active"


def _matches(item, args: argparse.Namespace) -> bool:
    meta = item.meta or {}
    if args.ids is not None and item.id not in args.ids:
        return False
    if args.source is not None and meta.get("source") != args.source:
        return False
    if args.namespace is not None and meta.get("namespace") != args.namespace:
        return False
    if args.state is not None and _state(item) != args.state:
        return False
    if args.min_quality is not None:
        quality = meta.get("quality_score")
        if not isinstance(quality, (int, float)) or float(quality) < args.min_quality:
            return False
    return True


def _summary(item, max_len: int) -> str:
    meta = item.meta or {}
    text = str(meta.get("summary") or item.text or "").strip()
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def cmd_review(args: argparse.Namespace) -> int:
    items = _load_items(args.storage_dir)
    matches = [item for item in items if _matches(item, args)]
    limit = max(1, args.limit)
    payload: dict[str, Any] = {
        "status": "ok",
        "action": "review",
        "storage_dir": str(args.storage_dir.expanduser().resolve()),
        "total_items": len(items),
        "matched": len(matches),
        "shown": min(limit, len(matches)),
        "state_counts": dict(Counter(_state(item) for item in items)),
        "source_counts": dict(Counter(str((item.meta or {}).get("source") or "unknown") for item in items)),
        "items": [],
    }
    for item in matches[:limit]:
        meta = item.meta or {}
        payload["items"].append(
            {
                "id": item.id,
                "state": _state(item),
                "source": meta.get("source"),
                "namespace": meta.get("namespace"),
                "quality_score": meta.get("quality_score"),
                "legacy_id": meta.get("legacy_id"),
                "legacy_doc_path": meta.get("legacy_doc_path"),
                "summary": _summary(item, args.summary_chars),
            }
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _mutate(args: argparse.Namespace, new_state: str) -> int:
    path = _state_path(args.storage_dir)
    items = _load_items(args.storage_dir)
    matched = [item for item in items if _matches(item, args)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "action": args.cmd,
                    "matched": len(matched),
                    "ids": [item.id for item in matched[: args.limit]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    backup_path = _backup(path)
    now = datetime.now(timezone.utc).isoformat()
    changed = 0
    limit = max(1, args.limit)
    selected_ids = []
    for item in matched[:limit]:
        meta = dict(item.meta or {})
        if meta.get("memory_state") == new_state:
            continue
        meta["memory_state"] = new_state
        meta["curated_at"] = now
        meta["curation_action"] = args.cmd
        item.meta = meta
        selected_ids.append(item.id)
        changed += 1
    save_path = _save_items(args.storage_dir, items)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": args.cmd,
                "new_state": new_state,
                "matched": len(matched),
                "changed": changed,
                "ids": selected_ids,
                "state_path": str(save_path),
                "backup_path": str(backup_path) if backup_path else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    return _mutate(args, "active")


def cmd_suppress(args: argparse.Namespace) -> int:
    return _mutate(args, "suppressed")


def cmd_delete(args: argparse.Namespace) -> int:
    return _mutate(args, "deleted")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--source")
    parser.add_argument("--namespace")
    parser.add_argument("--state")
    parser.add_argument("--min-quality", type=float)
    parser.add_argument("--limit", type=int, default=25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    review = sub.add_parser("review")
    _add_filters(review)
    review.add_argument("--summary-chars", type=int, default=180)
    review.set_defaults(func=cmd_review)

    for name, func in {
        "promote": cmd_promote,
        "suppress": cmd_suppress,
        "delete": cmd_delete,
    }.items():
        p = sub.add_parser(name)
        _add_filters(p)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
