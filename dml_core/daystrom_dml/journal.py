"""Versioned SQLite WAL state with atomic decisions and optimistic concurrency.

The database is authoritative for the lattice only. External RAG/KV stores are
not part of this transaction. Schema 0 requires an explicit side-by-side upgrade.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .atomic_io import atomic_write_text

JOURNAL_SCHEMA_VERSION = 1
FAULT_POINTS = (
    "after_begin", "after_records", "after_state", "after_snapshot",
    "after_decision", "before_commit", "after_commit",
)


class JournalIntegrityError(ValueError):
    """State cannot be trusted; recovery is required."""


class JournalSchemaError(JournalIntegrityError):
    """Unsupported schema; explicit upgrade or a compatible reader is required."""


class RevisionConflict(RuntimeError):
    """The caller attempted to overwrite a revision it has not read."""


def _encode(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _checked(raw: str, checksum: str):
    if _digest(raw) != checksum:
        raise JournalIntegrityError("Journal checksum mismatch")
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise JournalIntegrityError("Invalid journal JSON") from exc


class JournalStateStore:
    def __init__(self, path: Path, *, snapshot_interval: int = 128,
                 fault_hook: Callable[[str], None] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = max(1, snapshot_interval)
        self.last_changed_rows = 0
        self._revision = -1
        self._encoded = {}
        self._fault_hook = fault_hook or (lambda _point: None)
        self.identity_path = self.path.with_name(self.path.name + ".identity.json")
        existed = self.path.exists()
        if not existed and self.identity_path.exists():
            raise JournalIntegrityError("Previously initialized journal is missing; recovery required")
        # Only construction may create a new database. Normal operations use rw
        # and therefore cannot silently recreate a deleted authoritative file.
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if existed:
                self._validate_version(version)
            else:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("CREATE TABLE records (bucket TEXT, key TEXT, position INTEGER, payload TEXT, checksum TEXT NOT NULL, PRIMARY KEY(bucket,key))")
                    connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, metadata TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, payload TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE decisions (revision INTEGER PRIMARY KEY, payload TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE identity (id INTEGER PRIMARY KEY CHECK(id=1), store_id TEXT NOT NULL)")
                    connection.execute("INSERT INTO identity VALUES (1,?)", (uuid.uuid4().hex,))
                    connection.execute("INSERT INTO state VALUES (1,0,?,?)", ("{}", _digest("{}")))
                    connection.execute(f"PRAGMA user_version={JOURNAL_SCHEMA_VERSION}")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise JournalIntegrityError("Journal integrity check failed")
            row = connection.execute("SELECT store_id FROM identity WHERE id=1").fetchone()
            if row is None:
                raise JournalIntegrityError("Journal identity is missing")
            self._identity = {"schema_version": 1, "store_id": row[0]}
            if self.identity_path.exists():
                if json.loads(self.identity_path.read_text(encoding="utf-8")) != self._identity:
                    raise JournalIntegrityError("Journal identity mismatch")
            else:
                atomic_write_text(self.identity_path, _encode(self._identity))
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError("Journal could not be opened; recovery required") from exc
        finally:
            connection.close()

    @staticmethod
    def _validate_version(version):
        if version != JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError(
                f"Unsupported journal schema {version}; expected {JOURNAL_SCHEMA_VERSION}. "
                "Schema 0 requires explicit dml-journal upgrade to a new database."
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
        try:
            self._validate_version(connection.execute("PRAGMA user_version").fetchone()[0])
            row = connection.execute("SELECT store_id FROM identity WHERE id=1").fetchone()
            if row is None or row[0] != self._identity["store_id"]:
                raise JournalIntegrityError("Journal identity changed")
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    @property
    def revision(self) -> int:
        """Revision of the last snapshot read or successfully committed here."""
        return self._revision

    def stamp(self) -> tuple[int, int]:
        with self._connect() as connection:
            row = connection.execute("SELECT revision FROM state WHERE id=1").fetchone()
            if row is None:
                raise JournalIntegrityError("Journal state is missing")
        return int(row[0]), self.path.stat().st_mtime_ns

    _encode = staticmethod(_encode)

    def _read_snapshot(self, connection):
        row = connection.execute("SELECT revision,metadata,checksum FROM state WHERE id=1").fetchone()
        if row is None:
            raise JournalIntegrityError("Journal state is missing")
        revision, metadata, checksum = row
        payload = _checked(metadata, checksum)
        if not isinstance(payload, dict):
            raise JournalIntegrityError("Invalid journal metadata")
        rows = connection.execute("SELECT bucket,key,position,payload,checksum FROM records ORDER BY bucket,position").fetchall()
        decision = None
        if revision:
            event = connection.execute("SELECT payload,checksum FROM decisions WHERE revision=?", (revision,)).fetchone()
            if event is None:
                raise JournalIntegrityError("Missing commit decision")
            decision = _checked(*event)
            if decision.get("record_count") != len(rows) or decision.get("revision") != revision:
                raise JournalIntegrityError("Journal record count/revision mismatch")
        payload.update(items=[], lineage=[])
        encoded = {}
        for bucket, key, position, raw, checksum in rows:
            record = _checked(raw, checksum)
            if bucket not in {"items", "lineage"} or not isinstance(record, dict) or str(record.get("id")) != key:
                raise JournalIntegrityError("Invalid journal record identity")
            if position != len(payload[bucket]):
                raise JournalIntegrityError("Journal record positions are incomplete")
            payload[bucket].append(record)
            encoded[(bucket, key)] = (position, raw)
        if decision and decision.get("state_digest") != _digest(_encode(payload)):
            raise JournalIntegrityError("Journal committed state digest mismatch")
        return revision, payload, encoded

    def read_snapshot(self) -> tuple[int, dict]:
        """Return a detached state and its revision from one read transaction."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            revision, payload, encoded = self._read_snapshot(connection)
        self._revision, self._encoded = revision, encoded
        return revision, payload

    def load(self) -> dict:
        return self.read_snapshot()[1]

    def save(self, payload: dict, *, expected_revision: int | None = None,
             operation: str = "persist") -> None:
        if not isinstance(payload, dict):
            raise ValueError("journal state must be an object")
        if type(payload.get("schema_version", 1)) is not int or payload.get("schema_version", 1) != 1:
            raise JournalSchemaError("Unsupported lattice schema version")
        encoded = {}
        for bucket in ("items", "lineage"):
            records = payload.get(bucket, [])
            if not isinstance(records, list):
                raise ValueError("journal records must be lists")
            for position, record in enumerate(records):
                if not isinstance(record, dict) or type(record.get("id")) is not int or record["id"] < 0:
                    raise ValueError("journal record requires a nonnegative integer id")
                if type(record.get("schema_version", 1)) is not int or record.get("schema_version", 1) != 1:
                    raise JournalSchemaError("Unsupported journal record schema version")
                key = (bucket, str(record["id"]))
                if key in encoded:
                    raise ValueError("duplicate journal record id")
                encoded[key] = (position, self._encode(record))
        metadata = self._encode({key: value for key, value in payload.items() if key not in {"items", "lineage"}})
        normalized = {**payload, "items": payload.get("items", []), "lineage": payload.get("lineage", [])}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._fault_hook("after_begin")
            revision, current, previous = self._read_snapshot(connection)
            previous_meta = self._encode({key: value for key, value in current.items() if key not in {"items", "lineage"}})
            if expected_revision is not None and revision != expected_revision:
                raise RevisionConflict(f"Expected revision {expected_revision}, found {revision}")
            changed = [(bucket, key, *value, _digest(value[1])) for (bucket, key), value in encoded.items() if previous.get((bucket, key)) != value]
            deleted = sorted(previous.keys() - encoded.keys())
            if revision == 0 or changed or deleted or previous_meta != metadata:
                connection.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?)", changed)
                connection.executemany("DELETE FROM records WHERE bucket=? AND key=?", deleted)
                self._fault_hook("after_records")
                revision += 1
                connection.execute("UPDATE state SET revision=?,metadata=?,checksum=? WHERE id=1", (revision, metadata, _digest(metadata)))
                self._fault_hook("after_state")
                if revision == 1 or revision % self.snapshot_interval == 0:
                    raw = self._encode(payload)
                    connection.execute("INSERT OR REPLACE INTO snapshot VALUES (1,?,?,?)", (revision, raw, _digest(raw)))
                self._fault_hook("after_snapshot")
                decision = self._encode({
                    "schema_version": 1, "revision": revision, "operation": operation,
                    "changed": [{"bucket": bucket, "id": key, "digest": checksum} for bucket, key, _, _, checksum in changed],
                    "deleted": [{"bucket": bucket, "id": key} for bucket, key in deleted],
                    "record_count": len(encoded), "state_digest": _digest(self._encode(normalized)),
                })
                connection.execute("INSERT INTO decisions VALUES (?,?,?)", (revision, decision, _digest(decision)))
                self._fault_hook("after_decision")
            self._fault_hook("before_commit")
        self._fault_hook("after_commit")
        self._revision, self._encoded = revision, encoded
        self.last_changed_rows = len(changed) + len(deleted)

    def decisions(self, *, after_revision: int = 0, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("decision page size must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute("SELECT payload,checksum FROM decisions WHERE revision>? ORDER BY revision LIMIT ?", (after_revision, limit)).fetchall()
        return [_checked(*row) for row in rows]

    def export_snapshot(self, path: Path) -> Path:
        reserved = {self.path.resolve(), self.identity_path.resolve(), Path(str(self.path) + "-wal").resolve(), Path(str(self.path) + "-shm").resolve()}
        if Path(path).resolve() in reserved:
            raise ValueError("snapshot output must not overwrite the journal or its sidecars")
        return atomic_write_text(path, self._encode(self.load()))

    def checkpoint(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
