"""Explicit bounded delivery from a schema-3 authority to a local consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer, deliver_outbox, outbox_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing schema-3 authoritative journal")
    parser.add_argument("target", type=Path, help="Consumer journal in a separate directory")
    commands = parser.add_subparsers(dest="operation", required=True)
    sync = commands.add_parser("sync", help="Deliver one ordered page and verify durable acceptance")
    sync.add_argument("--limit", type=int, default=100)
    commands.add_parser("status", help="Report observed lag without creating a consumer")
    args = parser.parse_args(argv)
    try:
        if not args.source.is_file():
            raise FileNotFoundError("An existing authority is required")
        if args.operation == "status" and not args.target.is_file():
            raise FileNotFoundError("An existing consumer is required")
        if args.source.resolve().parent == args.target.resolve().parent:
            raise ValueError("Authority and consumer require separate directories")
        if args.operation == "sync" and not 1 <= args.limit <= 1000:
            raise ValueError("Invalid delivery limit")
        source = JournalStateStore(args.source)
        if source.schema_version != 3:
            raise ValueError("Outbox source requires journal schema 3")
        consumer = SQLiteOutboxConsumer(args.target)
        result = deliver_outbox(source, consumer, limit=args.limit) if args.operation == "sync" else outbox_status(source, consumer)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
