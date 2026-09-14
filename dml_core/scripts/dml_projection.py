"""Explicit synchronization and verified reads of a disposable vector projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.projection import (
    SQLiteProjection, projection_status, query_projection, reconcile,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing schema-2/3 authoritative journal")
    parser.add_argument("target", type=Path, help="Projection journal in a separate directory")
    commands = parser.add_subparsers(dest="operation", required=True)
    sync = commands.add_parser("sync", help="Atomically update the disposable projection from authority")
    sync.add_argument("--incremental", action="store_true", help="Deliver changed records with a checked base and complete ID manifest")
    commands.add_parser("status", help="Report revision lag without changing projection contents")
    query = commands.add_parser("query", help="Read only a projection matching a pinned authority snapshot")
    query.add_argument("request", type=Path, help="JSON: vector, embedding_identity, scope, top_k, as_of")
    args = parser.parse_args(argv)
    try:
        if not args.source.is_file():
            raise FileNotFoundError("An existing authority is required")
        if args.operation != "sync" and not args.target.is_file():
            raise FileNotFoundError("An existing projection is required")
        if args.source.resolve().parent == args.target.resolve().parent:
            raise ValueError("Authority and projection require separate directories")
        source = JournalStateStore(args.source)
        if source.schema_version not in (2, 3):
            raise ValueError("Projection source requires journal schema 2 or 3")
        target = SQLiteProjection(args.target)
        if args.operation == "sync":
            if args.incremental:
                from daystrom_dml.services.projection_delta import reconcile_incremental
                result = reconcile_incremental(source, target)
            else:
                result = reconcile(source, target)
        elif args.operation == "status":
            result = projection_status(source, target)
        else:
            request = json.loads(args.request.read_text(encoding="utf-8"))
            if not isinstance(request, dict) or set(request) - {"vector", "embedding_identity", "scope", "top_k", "as_of"}:
                raise ValueError("Invalid projection query request")
            result = query_projection(source, target, **request)
    except Exception as exc:
        # Storage paths, request contents and backend exception messages are
        # not an operator error-reporting channel. Preserve the typed outcome.
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
