from __future__ import annotations

import json
import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np

from daystrom_dml.memory_store import MemoryItem
from daystrom_dml.persistence import load_state, save_state


def _load_curator():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "curate_memory_store.py"
    spec = importlib.util.spec_from_file_location("curate_memory_store", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


curate_memory_store = _load_curator()


def _item(idx: int, text: str, *, state: str, quality: float = 1.0) -> MemoryItem:
    return MemoryItem(
        id=idx,
        text=text,
        embedding=np.ones(8, dtype=np.float32),
        timestamp=1.0,
        salience=1.0,
        fidelity=1.0,
        level=0,
        meta={
            "source": "old_openclaw_newdml_archive",
            "namespace": "legacy_archive",
            "memory_state": state,
            "quality_score": quality,
        },
    )


def test_curator_promotes_quarantined_memories(tmp_path, capsys):
    storage_dir = tmp_path / "store"
    save_state(
        [
            _item(1, "Good legacy memory", state="quarantined", quality=0.9),
            _item(2, "Weak legacy memory", state="quarantined", quality=0.2),
        ],
        storage_dir / "dml_state.jsonl",
    )

    args = Namespace(
        storage_dir=storage_dir,
        ids=None,
        source="old_openclaw_newdml_archive",
        namespace=None,
        state="quarantined",
        min_quality=0.8,
        limit=10,
        dry_run=False,
        cmd="promote",
    )

    assert curate_memory_store.cmd_promote(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["changed"] == 1

    items = load_state(storage_dir / "dml_state.jsonl")
    states = {item.id: item.meta["memory_state"] for item in items}
    assert states == {1: "active", 2: "quarantined"}


def test_curator_review_reports_state_counts(tmp_path, capsys):
    storage_dir = tmp_path / "store"
    save_state(
        [
            _item(1, "Active memory", state="active"),
            _item(2, "Quarantined memory", state="quarantined"),
        ],
        storage_dir / "dml_state.jsonl",
    )

    args = Namespace(
        storage_dir=storage_dir,
        ids=None,
        source=None,
        namespace=None,
        state=None,
        min_quality=None,
        limit=5,
        summary_chars=120,
    )

    assert curate_memory_store.cmd_review(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_counts"] == {"active": 1, "quarantined": 1}
    assert payload["matched"] == 2
