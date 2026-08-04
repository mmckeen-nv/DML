"""Portable atomic file replacement regression tests."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from daystrom_dml.atomic_io import atomic_write_text


def test_atomic_write_replaces_content_and_leaves_no_temp_files(tmp_path: Path):
    target = tmp_path / "state.json"
    atomic_write_text(target, json.dumps({"generation": 1}))
    atomic_write_text(target, json.dumps({"generation": 2}))

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 2}
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_concurrent_atomic_writers_never_create_torn_content(tmp_path: Path):
    target = tmp_path / "shared.json"
    payloads = [json.dumps({"writer": index, "body": "x" * 4096}) for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda payload: atomic_write_text(target, payload), payloads))

    result = target.read_text(encoding="utf-8")
    assert result in payloads
    assert json.loads(result)["body"] == "x" * 4096
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
