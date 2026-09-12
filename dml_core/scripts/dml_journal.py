"""Explicit import/export for the optional incremental lattice journal."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.persistence import load_state, validate_snapshot
from daystrom_dml.store_lock import store_write_lock


def upgrade_legacy(source: Path, destination: Path) -> None:
    """Copy the released schema-0 journal into a separate schema-1 database.

    Stop all writers before switching paths. The original is a rollback copy;
    no in-place downgrade or implicit startup migration is supported.
    """
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("upgrade requires a new, separate destination")
    with store_write_lock(source.parent, operation="journal-upgrade", timeout_ms=30000):
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 0:
                raise ValueError("upgrade accepts only the legacy schema-0 journal")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("legacy journal integrity check failed")
            row = connection.execute("SELECT metadata FROM state WHERE id=1").fetchone()
            if row is None:
                raise ValueError("legacy journal state is missing")
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("invalid legacy journal metadata")
            payload.update(items=[], lineage=[])
            for bucket, key, position, raw in connection.execute("SELECT bucket,key,position,payload FROM records ORDER BY bucket,position"):
                record = json.loads(raw)
                if bucket not in {"items", "lineage"} or str(record.get("id")) != key or position != len(payload[bucket]):
                    raise ValueError("invalid legacy journal record identity/order")
                payload[bucket].append(record)
        validate_snapshot(payload)
        journal = JournalStateStore(destination)
        journal.save(payload, expected_revision=0, operation="schema-0-upgrade")
        if journal.load() != payload:
            raise ValueError("upgraded journal verification failed")


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
    validate_snapshot(payload)
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
    upgrade = commands.add_parser("upgrade")
    upgrade.add_argument("source", type=Path)
    upgrade.add_argument("destination", type=Path)
    decisions = commands.add_parser("decisions")
    decisions.add_argument("database", type=Path)
    decisions.add_argument("--after-revision", type=int, default=0)
    decisions.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.operation == "import":
        import_snapshot(args.source, args.database)
    elif args.operation == "upgrade":
        upgrade_legacy(args.source, args.destination)
    elif args.operation == "decisions":
        if not args.database.is_file():
            raise FileNotFoundError(args.database)
        print(json.dumps(JournalStateStore(args.database).decisions(after_revision=args.after_revision, limit=args.limit), indent=2))
    else:
        if not args.database.is_file():
            raise FileNotFoundError(args.database)
        if args.output.resolve() == args.database.resolve():
            raise ValueError("export output must differ from the journal database")
        JournalStateStore(args.database).export_snapshot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
