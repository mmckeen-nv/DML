"""Explicit import/export for the optional incremental lattice journal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.persistence import load_state
from daystrom_dml.store_lock import store_write_lock


def import_snapshot(source: Path, database: Path) -> None:
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        items = load_state(source)
        payload = {"items": [item.to_dict() for item in items]}
    else:
        if isinstance(payload, dict) and payload.get("type") == "daystrom_dml.memory":
            payload = {"items": [item.to_dict() for item in load_state(source)]}
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("source must be a full lattice JSON or checksummed JSONL snapshot")
    with store_write_lock(database.parent, operation="journal-import", timeout_ms=30000):
        store = JournalStateStore(database)
        if store.stamp()[0] != 0:
            raise ValueError("refusing to overwrite an initialized journal")
        store.save(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    incoming = commands.add_parser("import")
    incoming.add_argument("source", type=Path)
    incoming.add_argument("database", type=Path)
    outgoing = commands.add_parser("export")
    outgoing.add_argument("database", type=Path)
    outgoing.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.operation == "import":
        import_snapshot(args.source, args.database)
    else:
        if not args.database.is_file():
            raise FileNotFoundError(args.database)
        if args.output.resolve() == args.database.resolve():
            raise ValueError("export output must differ from the journal database")
        JournalStateStore(args.database).export_snapshot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
