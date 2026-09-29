"""Durable persistence helpers for the Daystrom Memory Lattice."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np

from . import utils
from .atomic_io import atomic_write_text
from .memory_store import MemoryItem

_PERSISTENCE_VERSION = 1
_PERSISTENCE_TYPE = "daystrom_dml.memory"


class PersistenceFormatError(ValueError):
    """Persisted state is corrupt or uses an unsupported schema."""


def validate_record(record: dict) -> None:
    """Validate v1 records before constructing memory; never coerce bad data."""
    if not isinstance(record, dict):
        raise PersistenceFormatError("Memory record must be an object")
    if type(record.get("schema_version", 1)) is not int or record.get("schema_version", 1) != 1:
        raise PersistenceFormatError("Unsupported memory record schema version")
    for key in ("id", "level"):
        if type(record.get(key)) is not int or record[key] < 0:
            raise PersistenceFormatError(f"Invalid memory {key}")
    if not isinstance(record.get("text"), str):
        raise PersistenceFormatError("Invalid memory text")
    for key in ("timestamp", "salience", "fidelity"):
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise PersistenceFormatError(f"Invalid memory {key}")
    if record.get("meta") is not None and not isinstance(record.get("meta"), dict):
        raise PersistenceFormatError("Invalid memory metadata")
    children = record.get("summary_of", [])
    if not isinstance(children, list) or any(type(v) is not int or v < 0 for v in children):
        raise PersistenceFormatError("Invalid memory lineage")
    vector = record.get("embedding")
    if not isinstance(vector, list) or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in vector
    ):
        raise PersistenceFormatError("Invalid memory embedding")
    # Reject values that overflow the float32 representation used at runtime.
    if any(abs(v) > float(np.finfo(np.float32).max) for v in vector):
        raise PersistenceFormatError("Memory embedding exceeds float32 range")


def validate_snapshot(payload: dict) -> None:
    """Accept the documented legacy-v1 JSON shape, rejecting partial records."""
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise PersistenceFormatError("Lattice snapshot must contain an items list")
    if type(payload.get("schema_version", 1)) is not int or payload.get("schema_version", 1) != 1:
        raise PersistenceFormatError("Unsupported lattice schema version")
    for bucket in ("items", "lineage"):
        records = payload.get(bucket, [])
        if not isinstance(records, list):
            raise PersistenceFormatError(f"Invalid snapshot {bucket}")
        seen = set()
        for record in records:
            validate_record(record)
            if record["id"] in seen:
                raise PersistenceFormatError(f"Duplicate {bucket} IDs")
            seen.add(record["id"])


def _item_to_record(item: MemoryItem) -> dict:
    record = {
        "schema_version": 1,
        "id": item.id,
        "text": item.text,
        "level": item.level,
        "fidelity": item.fidelity,
        "salience": item.salience,
        "timestamp": item.timestamp,
        "meta": item.meta or {},
        "summary_of": list(item.summary_of or []),
        "embedding": utils.ensure_serializable(item.embedding),
    }
    validate_record(record)
    return record


def save_state(items: Sequence[MemoryItem], path: str | Path) -> Path:
    """Persist ``items`` to ``path`` as newline delimited JSON."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [_item_to_record(item) for item in items]
    if len({record["id"] for record in records}) != len(records):
        raise PersistenceFormatError("Duplicate memory IDs")
    payload_lines = [json.dumps(record, separators=(",", ":"), sort_keys=True) for record in records]
    payload_bytes = "\n".join(payload_lines).encode("utf-8")
    checksum = hashlib.sha256(payload_bytes).hexdigest()
    header = {
        "type": _PERSISTENCE_TYPE,
        "version": _PERSISTENCE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "checksum": checksum,
    }
    header_line = json.dumps(header, separators=(",", ":"), sort_keys=True)
    content = header_line
    if payload_lines:
        content += "\n" + "\n".join(payload_lines)
    atomic_write_text(target, content)
    return target


def load_state(path: str | Path) -> List[MemoryItem]:
    """Load persisted memories from ``path``."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if not lines:
        raise PersistenceFormatError("Empty persistence file is not a valid store")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid persistence header
        raise PersistenceFormatError("Invalid persistence header") from exc
    if not isinstance(header, dict):
        raise PersistenceFormatError("Persistence header must be an object")
    if header.get("type") != _PERSISTENCE_TYPE:
        raise ValueError(f"Unsupported persistence payload: {header.get('type')}")
    if type(header.get("version")) is not int or header["version"] != _PERSISTENCE_VERSION:
        raise PersistenceFormatError("Unsupported persistence version")
    if type(header.get("count")) is not int or header["count"] < 0:
        raise PersistenceFormatError("Invalid persistence record count")
    data_lines = lines[1:]
    expected_checksum = header.get("checksum")
    payload_bytes = "\n".join(data_lines).encode("utf-8")
    actual_checksum = hashlib.sha256(payload_bytes).hexdigest()
    if expected_checksum != actual_checksum:
        raise PersistenceFormatError("Persistence checksum mismatch")
    items: List[MemoryItem] = []
    seen: set[int] = set()
    for raw in data_lines:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PersistenceFormatError("Invalid persistence record") from exc
        validate_record(record)
        if record["id"] in seen:
            raise PersistenceFormatError("Duplicate memory IDs")
        seen.add(record["id"])
        embedding = np.asarray(record.get("embedding") or [], dtype=np.float32)
        item = MemoryItem(
            id=int(record.get("id", 0)),
            text=str(record.get("text") or ""),
            level=int(record.get("level", 0)),
            fidelity=float(record.get("fidelity") or 0.0),
            salience=float(record.get("salience") or 0.0),
            timestamp=float(record.get("timestamp") or 0.0),
            meta=record.get("meta") or {},
            summary_of=list(record.get("summary_of") or []),
            embedding=embedding,
        )
        items.append(item)
    count = header.get("count")
    if isinstance(count, int) and count != len(items):  # pragma: no cover - diagnostic only
        raise ValueError("Persistence record count mismatch")
    return items
