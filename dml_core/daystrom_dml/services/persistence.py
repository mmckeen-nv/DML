"""Lattice storage selection, validation and commits behind a narrow boundary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from ..journal import JournalStateStore
from ..memory_store import MemoryItem
from ..persistence import validate_snapshot


class LatticePersistence:
    def __init__(self, *, json_path: Path, jsonl_path: Path, use_jsonl: bool,
                 journal: JournalStateStore | None,
                 read_jsonl: Callable, write_jsonl: Callable, write_text: Callable):
        self.json_path = json_path
        self.jsonl_path = jsonl_path
        self.use_jsonl = use_jsonl
        self.journal = journal
        self._read_jsonl, self._write_jsonl, self._write_text = read_jsonl, write_jsonl, write_text

    @property
    def path(self) -> Path:
        return self.journal.path if self.journal else self.jsonl_path if self.use_jsonl else self.json_path

    def stamp(self) -> tuple[int, int] | None:
        if self.journal:
            return self.journal.stamp()
        try:
            stat = self.path.stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def load(self, *, startup: bool = False) -> dict | None:
        if self.journal:
            if startup and self.journal.stamp()[0] == 0 and (self.json_path.exists() or self.jsonl_path.exists()):
                raise ValueError("Journal migration requires an explicit snapshot import")
            payload = self.journal.load()
        elif self.use_jsonl and self.jsonl_path.exists():
            payload = {"items": [item.to_dict() for item in self._read_jsonl(self.jsonl_path)]}
        elif self.json_path.exists() and (startup or not self.use_jsonl):
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        elif startup:
            return None
        else:
            raise FileNotFoundError("Previously initialized DML state is missing")
        validate_snapshot(payload)
        return payload

    def commit(self, *, payload: dict | None, items: Sequence[MemoryItem] | None,
               expected_revision: int | None, operation: str) -> tuple[int, int] | None:
        if self.journal:
            if payload is None:
                raise ValueError("Journal commit requires a lattice payload")
            self.journal.save(payload, expected_revision=expected_revision, operation=operation)
            return self.journal.revision, self.journal.path.stat().st_mtime_ns
        if self.use_jsonl:
            if items is None:
                raise ValueError("JSONL commit requires memory items")
            self._write_jsonl(items, self.jsonl_path)
        else:
            if payload is None:
                raise ValueError("JSON commit requires a lattice payload")
            self._write_text(self.json_path, json.dumps(payload, indent=2, allow_nan=False))
        return self.stamp()
