#!/usr/bin/env python3
"""Seed a square synthetic lattice into the Daystrom Memory Lattice store."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_path() -> None:
    core = _repo_root() / "dml_core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))


def _node_text(row: int, col: int, size: int, lattice_id: str) -> str:
    quadrant = (
        "northwest" if row < size / 2 and col < size / 2 else
        "northeast" if row < size / 2 else
        "southwest" if col < size / 2 else
        "southeast"
    )
    return (
        f"Synthetic square lattice {lattice_id} node r{row:02d} c{col:02d}. "
        f"This memory is part of a {size} by {size} DML visualization grid in the {quadrant} quadrant. "
        f"It links to adjacent lattice memories so retrieval can show local neighborhood activation, "
        f"token usage, salience, fidelity, and stable square topology."
    )


def _summary(row: int, col: int, size: int, lattice_id: str) -> str:
    return (
        f"Square lattice {lattice_id} node ({row}, {col}) in a {size}x{size} synthetic DML demo grid."
    )


def _source(row: int, col: int, lattice_id: str) -> str:
    return f"synthetic/square-lattice/{lattice_id}/r{row:02d}-c{col:02d}"


def _neighbors(row: int, col: int, size: int, id_map: dict[tuple[int, int], int]) -> list[int]:
    coords = (
        (row - 1, col),
        (row, col + 1),
        (row + 1, col),
        (row, col - 1),
    )
    return [
        id_map[(next_row, next_col)]
        for next_row, next_col in coords
        if 0 <= next_row < size and 0 <= next_col < size
    ]


def _drop_existing_lattice(adapter: Any, lattice_id: str | None) -> int:
    """Remove existing synthetic lattice nodes from the active store."""

    items = list(adapter.store.items())
    kept = []
    removed = 0
    for item in items:
        meta = item.meta or {}
        is_square = meta.get("synthetic_lattice") == "square"
        matches_id = lattice_id is None or meta.get("lattice_id") == lattice_id
        if is_square and matches_id:
            removed += 1
            continue
        kept.append(item)
    if removed:
        adapter.store.import_state(
            {
                "items": [item.to_dict() for item in kept],
                "lineage": [item.to_dict() for item in kept],
                "repair_queue": [],
                "next_id": (max((item.id for item in kept), default=-1) + 1),
            }
        )
    return removed


def seed_square_lattice(*, size: int, lattice_id: str, replace: bool, storage_dir: Path | None) -> dict[str, Any]:
    _ensure_import_path()
    from daystrom_dml.dml_adapter import DMLAdapter  # type: ignore

    overrides: dict[str, Any] = {}
    if storage_dir is not None:
        overrides["storage_dir"] = str(storage_dir.expanduser().resolve())
    adapter = DMLAdapter(config_overrides=overrides or None, start_aging_loop=False)
    try:
        removed = _drop_existing_lattice(adapter, lattice_id if replace else None) if replace else 0
        created = []
        id_map: dict[tuple[int, int], int] = {}
        now = time.time()
        for row in range(size):
            for col in range(size):
                text = _node_text(row, col, size, lattice_id)
                meta = {
                    "source": _source(row, col, lattice_id),
                    "summary": _summary(row, col, size, lattice_id),
                    "kind": "synthetic_lattice_node",
                    "synthetic_lattice": "square",
                    "lattice_id": lattice_id,
                    "lattice_row": row,
                    "lattice_col": col,
                    "lattice_size": size,
                    "lattice_created_at": now,
                    "no_merge": True,
                }
                item, merged = adapter.store.ingest(
                    text,
                    adapter.embedder.embed(text),
                    salience=0.45 + 0.45 * ((row + col) / max(1, (size - 1) * 2)),
                    fidelity=1.0,
                    level=0,
                    meta=meta,
                )
                if merged:
                    raise RuntimeError(f"synthetic lattice node unexpectedly merged: {row},{col}")
                created.append(item)
                id_map[(row, col)] = item.id

        for item in created:
            meta = item.meta or {}
            row = int(meta["lattice_row"])
            col = int(meta["lattice_col"])
            neighbor_ids = _neighbors(row, col, size, id_map)
            meta["lattice_neighbors"] = neighbor_ids
            meta["lattice_degree"] = len(neighbor_ids)
            item.meta = meta
            # Keep native lineage semantics intact for the store.
            item.summary_of = [item.id]

        adapter._persist_all()  # Persist both durable JSONL and legacy JSON snapshots.
        return {
            "status": "ok",
            "lattice_id": lattice_id,
            "size": size,
            "created": len(created),
            "removed": removed,
            "total_memories": len(adapter.store.items()),
            "storage_dir": str(adapter.storage_dir.resolve()),
            "persistence_path": str(getattr(adapter, "_persistence_path", "")),
        }
    finally:
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12, help="Grid width and height")
    parser.add_argument("--lattice-id", default="demo-square", help="Synthetic lattice identifier")
    parser.add_argument("--storage-dir", type=Path, default=None, help="Override DML storage directory")
    parser.add_argument("--replace", action="store_true", help="Replace existing square lattice nodes with the same id")
    args = parser.parse_args()
    size = max(2, min(40, int(args.size)))
    report = seed_square_lattice(
        size=size,
        lattice_id=str(args.lattice_id),
        replace=bool(args.replace),
        storage_dir=args.storage_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
