"""Incremental SQLite WAL persistence for the complete lattice state.

SQLite supplies transactional replay and fsync; periodic logical snapshots live
in the same transaction as the corresponding rows. This backend is opt-in.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .atomic_io import atomic_write_text


class JournalStateStore:
    def __init__(self, path: Path, *, snapshot_interval: int = 128):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = max(1, snapshot_interval)
        self.last_changed_rows = 0
        self._revision = -1
        self._encoded = {}
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS records (bucket TEXT, key TEXT, position INTEGER, payload TEXT, PRIMARY KEY(bucket,key))")
            connection.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, metadata TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, payload TEXT)")
            connection.execute("INSERT OR IGNORE INTO state VALUES (1,0,'{}')")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    def stamp(self) -> tuple[int, int]:
        with self._connect() as connection:
            revision = connection.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]
        return int(revision), self.path.stat().st_mtime_ns

    @staticmethod
    def _encode(payload) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def load(self) -> dict:
        with self._connect() as connection:
            connection.execute("BEGIN")
            revision, metadata = connection.execute("SELECT revision,metadata FROM state WHERE id=1").fetchone()
            rows = connection.execute("SELECT bucket,key,position,payload FROM records ORDER BY bucket,position").fetchall()
        payload = json.loads(metadata)
        payload.update(items=[], lineage=[])
        encoded = {}
        for bucket, key, position, raw in rows:
            payload[bucket].append(json.loads(raw))
            encoded[(bucket, key)] = (position, raw)
        self._revision, self._encoded = revision, encoded
        return payload

    def save(self, payload: dict) -> None:
        encoded = {}
        for bucket in ("items", "lineage"):
            for position, record in enumerate(payload.get(bucket, [])):
                key = (bucket, str(record["id"]))
                if key in encoded:
                    raise ValueError("duplicate journal record id")
                encoded[key] = (position, self._encode(record))
        metadata = self._encode({key: value for key, value in payload.items() if key not in {"items", "lineage"}})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision, previous_meta = connection.execute("SELECT revision,metadata FROM state WHERE id=1").fetchone()
            previous = self._encoded
            if revision != self._revision:
                previous = {(bucket, key): (position, raw) for bucket, key, position, raw in
                            connection.execute("SELECT bucket,key,position,payload FROM records")}
            changed = [(bucket, key, *value) for (bucket, key), value in encoded.items() if previous.get((bucket, key)) != value]
            deleted = list(previous.keys() - encoded.keys())
            if revision == 0 or changed or deleted or previous_meta != metadata:
                connection.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?)", changed)
                connection.executemany("DELETE FROM records WHERE bucket=? AND key=?", deleted)
                revision += 1
                connection.execute("UPDATE state SET revision=?,metadata=? WHERE id=1", (revision, metadata))
                if revision == 1 or revision % self.snapshot_interval == 0:
                    connection.execute("INSERT OR REPLACE INTO snapshot VALUES (1,?,?)", (revision, self._encode(payload)))
        # Do not advance the cache unless COMMIT returned successfully.
        self._revision, self._encoded = revision, encoded
        self.last_changed_rows = len(changed) + len(deleted)

    def export_snapshot(self, path: Path) -> Path:
        """Export a current portable JSON snapshot for offline tooling/backups."""
        reserved = {self.path.resolve(), Path(str(self.path) + "-wal").resolve(), Path(str(self.path) + "-shm").resolve()}
        if Path(path).resolve() in reserved:
            raise ValueError("snapshot output must not overwrite the journal or its sidecars")
        return atomic_write_text(path, self._encode(self.load()))

    def checkpoint(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
