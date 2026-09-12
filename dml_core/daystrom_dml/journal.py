"""Versioned SQLite WAL state with atomic decisions and optimistic concurrency.

The database is authoritative for the lattice only. External RAG/KV stores are
not part of this transaction. Schema 0 requires an explicit side-by-side upgrade.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .atomic_io import atomic_write_text
from .store_lock import acquire_file_lock, release_file_lock

JOURNAL_SCHEMA_VERSION = 1
INIT_FAULT_POINTS = (
    "init_before_connect", "init_after_connect", "init_after_begin",
    "init_after_schema", "init_after_commit", "init_before_identity",
    "init_after_identity",
)
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


def _is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JournalIntegrityError("Duplicate journal JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise JournalIntegrityError(f"Non-finite journal JSON value: {value}")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise JournalIntegrityError("Non-finite journal JSON number")
    return result


def _decode(raw):
    if not isinstance(raw, str):
        raise JournalIntegrityError("Journal JSON must be text")
    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant, parse_float=_finite_float)
    except (TypeError, ValueError, RecursionError) as exc:
        raise JournalIntegrityError("Invalid journal JSON") from exc


def _checked(raw: str, checksum: str):
    if not isinstance(raw, str) or not _is_digest(checksum) or _digest(raw) != checksum:
        raise JournalIntegrityError("Journal checksum mismatch")
    return _decode(raw)


def _version(payload):
    if type(payload.get("schema_version", 1)) is not int or payload.get("schema_version", 1) != 1:
        raise JournalSchemaError("Unsupported journal payload schema version")


def _normalized(payload):
    if not isinstance(payload, dict):
        raise JournalIntegrityError("Journal state must be an object")
    _version(payload)
    result = {**payload, "items": payload.get("items", []), "lineage": payload.get("lineage", [])}
    for bucket in ("items", "lineage"):
        records = result[bucket]
        if not isinstance(records, list):
            raise JournalIntegrityError("Journal records must be lists")
        seen = set()
        for record in records:
            if not isinstance(record, dict) or type(record.get("id")) is not int or record["id"] < 0:
                raise JournalIntegrityError("Journal record requires a nonnegative integer id")
            _version(record)
            if record["id"] in seen:
                raise JournalIntegrityError("Duplicate journal record id")
            seen.add(record["id"])
    return result


def _valid_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {"schema_version", "store_id"}:
        raise JournalIntegrityError("Invalid journal identity")
    _version(identity)
    store_id = identity["store_id"]
    if not isinstance(store_id, str) or len(store_id) != 32 or any(c not in "0123456789abcdef" for c in store_id):
        raise JournalIntegrityError("Invalid journal store identity")
    return identity


def _read_identity(path):
    try:
        return _valid_identity(_decode(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError) as exc:
        raise JournalIntegrityError("Journal identity marker cannot be read") from exc


@contextmanager
def _initialization_lock(path: Path):
    # A distinct, persistent inode avoids nesting the adapter/CLI store lock.
    # Never unlink this file: waiters must continue locking the same inode.
    with path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + 30.0
        while True:
            try:
                acquire_file_lock(handle)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out initializing journal {path}")
                time.sleep(0.01)
        try:
            yield
        finally:
            release_file_lock(handle)


class JournalStateStore:
    def __init__(self, path: Path, *, snapshot_interval: int = 128,
                 fault_hook: Callable[[str], None] | None = None):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = max(1, snapshot_interval)
        self.last_changed_rows = 0
        self._revision = -1
        self._encoded: dict[tuple[str, str], tuple[int, str]] = {}
        self._fault_hook = fault_hook or (lambda _point: None)
        self.identity_path = self.path.with_name(self.path.name + ".identity.json")
        self.initialization_lock_path = self.path.with_name(self.path.name + ".init.lock")
        with _initialization_lock(self.initialization_lock_path):
            self._initialize()

    def _initialize(self):
        existed = self.path.exists()
        if not existed and self.identity_path.exists():
            raise JournalIntegrityError("Previously initialized journal is missing; recovery required")
        # Existence and schema creation are serialized across processes. A kill
        # before schema commit leaves version 0, which is preserved and rejected;
        # only a fully validated v1 database may finish identity publication.
        if not existed:
            self._fault_hook("init_before_connect")
        try:
            if existed:
                connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
            else:
                connection = sqlite3.connect(self.path, timeout=30)
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError("Journal could not be opened; recovery required") from exc
        try:
            if existed:
                self._validate_version(connection.execute("PRAGMA user_version").fetchone()[0])
            else:
                self._fault_hook("init_after_connect")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._fault_hook("init_after_begin")
                    connection.execute("CREATE TABLE records (bucket TEXT, key TEXT, position INTEGER, payload TEXT, checksum TEXT NOT NULL, PRIMARY KEY(bucket,key))")
                    connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, metadata TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, payload TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE decisions (revision INTEGER PRIMARY KEY, payload TEXT, checksum TEXT NOT NULL)")
                    connection.execute("CREATE TABLE identity (id INTEGER PRIMARY KEY CHECK(id=1), store_id TEXT NOT NULL)")
                    connection.execute("INSERT INTO identity VALUES (1,?)", (uuid.uuid4().hex,))
                    connection.execute("INSERT INTO state VALUES (1,0,?,?)", ("{}", _digest("{}")))
                    connection.execute(f"PRAGMA user_version={JOURNAL_SCHEMA_VERSION}")
                    self._fault_hook("init_after_schema")
                self._fault_hook("init_after_commit")
            with connection:
                connection.execute("BEGIN")
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise JournalIntegrityError("Journal integrity check failed")
                self._validate_layout(connection)
                rows = connection.execute("SELECT id,store_id FROM identity").fetchall()
                if len(rows) != 1 or rows[0][0] != 1:
                    raise JournalIntegrityError("Journal identity is missing or ambiguous")
                self._identity = _valid_identity({"schema_version": 1, "store_id": rows[0][1]})
                self._read_snapshot(connection)
            if self.identity_path.exists():
                if _read_identity(self.identity_path) != self._identity:
                    raise JournalIntegrityError("Journal identity mismatch")
            else:
                self._fault_hook("init_before_identity")
                atomic_write_text(self.identity_path, _encode(self._identity))
                self._fault_hook("init_after_identity")
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

    @staticmethod
    def _validate_layout(connection):
        columns = {
            "records": [("bucket", "TEXT", 0, 1), ("key", "TEXT", 0, 2), ("position", "INTEGER", 0, 0), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "state": [("id", "INTEGER", 0, 1), ("revision", "INTEGER", 0, 0), ("metadata", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "snapshot": [("id", "INTEGER", 0, 1), ("revision", "INTEGER", 0, 0), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "decisions": [("revision", "INTEGER", 0, 1), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "identity": [("id", "INTEGER", 0, 1), ("store_id", "TEXT", 1, 0)],
        }
        objects = connection.execute("SELECT type,name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'").fetchall()
        if set(objects) != {("table", name) for name in columns}:
            raise JournalSchemaError("Unexpected journal schema objects")
        for table, expected in columns.items():
            actual = [(row[1], row[2].upper(), row[3], row[5]) for row in connection.execute(f"PRAGMA table_info({table})")]
            if actual != expected:
                raise JournalSchemaError(f"Unexpected journal table layout: {table}")

    @contextmanager
    def _connect(self):
        try:
            connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError("Journal could not be opened; recovery required") from exc
        try:
            try:
                self._validate_version(connection.execute("PRAGMA user_version").fetchone()[0])
                self._validate_layout(connection)
                rows = connection.execute("SELECT id,store_id FROM identity").fetchall()
                if len(rows) != 1 or rows[0] != (1, self._identity["store_id"]):
                    raise JournalIntegrityError("Journal identity changed")
                marker = _read_identity(self.identity_path)
                if marker != self._identity:
                    raise JournalIntegrityError("Journal identity marker changed")
                connection.execute("PRAGMA synchronous=FULL")
            except sqlite3.DatabaseError as exc:
                raise JournalIntegrityError("Journal connection integrity check failed") from exc
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
            connection.execute("BEGIN")
            revision, _, _ = self._read_snapshot(connection)
        return revision, self.path.stat().st_mtime_ns

    _encode = staticmethod(_encode)

    @staticmethod
    def _decision(sql_revision, raw, checksum):
        decision = _checked(raw, checksum)
        if not isinstance(decision, dict):
            raise JournalIntegrityError("Journal decision must be an object")
        _version(decision)
        if set(decision) != {"schema_version", "revision", "operation", "changed", "deleted", "record_count", "state_digest"}:
            raise JournalIntegrityError("Invalid journal decision fields")
        revision = decision["revision"]
        if type(revision) is not int or revision <= 0 or revision != sql_revision:
            raise JournalIntegrityError("Journal decision revision mismatch")
        if not isinstance(decision["operation"], str) or not decision["operation"].strip():
            raise JournalIntegrityError("Invalid journal decision operation")
        if type(decision["record_count"]) is not int or decision["record_count"] < 0 or not _is_digest(decision["state_digest"]):
            raise JournalIntegrityError("Invalid journal decision state metadata")
        seen = set()
        for action in ("changed", "deleted"):
            entries = decision[action]
            if not isinstance(entries, list):
                raise JournalIntegrityError("Invalid journal decision changes")
            for entry in entries:
                expected = {"bucket", "id", "digest"} if action == "changed" else {"bucket", "id"}
                if not isinstance(entry, dict) or set(entry) != expected:
                    raise JournalIntegrityError("Invalid journal decision record")
                key = entry["id"]
                if not isinstance(key, str) or not key.isascii() or not key.isdigit() or (key != "0" and key.startswith("0")):
                    raise JournalIntegrityError("Invalid journal decision record id")
                bucket = entry["bucket"]
                if not isinstance(bucket, str) or bucket not in {"items", "lineage"} or (bucket, key) in seen:
                    raise JournalIntegrityError("Invalid or duplicate journal decision identity")
                seen.add((bucket, key))
                if action == "changed" and not _is_digest(entry["digest"]):
                    raise JournalIntegrityError("Invalid journal decision record digest")
        return decision

    def _read_snapshot(self, connection):
        rows = connection.execute("SELECT id,revision,metadata,checksum FROM state").fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise JournalIntegrityError("Journal state is missing or ambiguous")
        _, revision, metadata, checksum = rows[0]
        if type(revision) is not int or revision < 0:
            raise JournalIntegrityError("Invalid journal state revision")
        payload = _checked(metadata, checksum)
        if not isinstance(payload, dict) or {"items", "lineage"} & payload.keys():
            raise JournalIntegrityError("Invalid journal metadata")
        _version(payload)
        rows = connection.execute("SELECT bucket,key,position,payload,checksum FROM records ORDER BY bucket,position").fetchall()
        snapshots = connection.execute("SELECT id,revision,payload,checksum FROM snapshot").fetchall()
        events = connection.execute("SELECT revision,payload,checksum FROM decisions ORDER BY revision").fetchall()
        if revision == 0 and (payload != {} or rows or snapshots or events):
            raise JournalIntegrityError("Uncommitted journal revision contains state")
        if len(events) != revision:
            raise JournalIntegrityError("Journal decision history is incomplete")
        decisions = {}
        expected_records = {}
        for expected_revision, event in enumerate(events, 1):
            if event[0] != expected_revision:
                raise JournalIntegrityError("Journal decision history is not contiguous")
            decision = self._decision(*event)
            for entry in decision["deleted"]:
                key = (entry["bucket"], entry["id"])
                if key not in expected_records:
                    raise JournalIntegrityError("Journal decision deletes an absent record")
                del expected_records[key]
            for entry in decision["changed"]:
                expected_records[(entry["bucket"], entry["id"])] = entry["digest"]
            if len(expected_records) != decision["record_count"]:
                raise JournalIntegrityError("Journal decision record count mismatch")
            decisions[expected_revision] = decision
        payload.update(items=[], lineage=[])
        encoded = {}
        for bucket, key, position, raw, checksum in rows:
            record = _checked(raw, checksum)
            if not isinstance(bucket, str) or bucket not in {"items", "lineage"} or not isinstance(record, dict) or type(record.get("id")) is not int or record["id"] < 0 or str(record["id"]) != key:
                raise JournalIntegrityError("Invalid journal record identity")
            _version(record)
            if type(position) is not int or position != len(payload[bucket]):
                raise JournalIntegrityError("Journal record positions are incomplete")
            if expected_records.get((bucket, key)) != checksum:
                raise JournalIntegrityError("Journal record differs from committed decision history")
            payload[bucket].append(record)
            encoded[(bucket, key)] = (position, raw)
        if len(encoded) != len(expected_records):
            raise JournalIntegrityError("Journal record count mismatch")
        if revision and decisions[revision]["state_digest"] != _digest(_encode(payload)):
            raise JournalIntegrityError("Journal committed state digest mismatch")
        if revision:
            if len(snapshots) != 1 or snapshots[0][0] != 1:
                raise JournalIntegrityError("Journal snapshot is missing or ambiguous")
            _, snapshot_revision, raw, checksum = snapshots[0]
            if type(snapshot_revision) is not int or not 1 <= snapshot_revision <= revision:
                raise JournalIntegrityError("Invalid journal snapshot revision")
            snapshot = _normalized(_checked(raw, checksum))
            if decisions[snapshot_revision]["state_digest"] != _digest(_encode(snapshot)):
                raise JournalIntegrityError("Journal snapshot differs from its commit decision")
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
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a nonempty string")
        # Freeze the exact serialized input before acquiring the write lock.
        # Caller mutations during I/O must not produce a split commit.
        payload = _decode(self._encode(payload))
        normalized = _normalized(payload)
        encoded = {}
        for bucket in ("items", "lineage"):
            for position, record in enumerate(normalized[bucket]):
                encoded[(bucket, str(record["id"]))] = (position, self._encode(record))
        metadata = self._encode({key: value for key, value in payload.items() if key not in {"items", "lineage"}})
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
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("decision page size must be an integer between 1 and 1000")
        if type(after_revision) is not int or after_revision < 0:
            raise ValueError("after_revision must be a nonnegative integer")
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._read_snapshot(connection)
            rows = connection.execute("SELECT revision,payload,checksum FROM decisions WHERE revision>? ORDER BY revision LIMIT ?", (after_revision, limit)).fetchall()
        return [self._decision(*row) for row in rows]

    def export_snapshot(self, path: Path) -> Path:
        reserved = {self.path.resolve(), self.identity_path.resolve(), self.initialization_lock_path.resolve(), (self.path.parent / ".dml_store.lock").resolve(), (self.path.parent / ".dml_store.lock.json").resolve(), Path(str(self.path) + "-wal").resolve(), Path(str(self.path) + "-shm").resolve()}
        if Path(path).resolve() in reserved:
            raise ValueError("snapshot output must not overwrite the journal or its sidecars")
        return atomic_write_text(path, self._encode(self.load()))

    def checkpoint(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._read_snapshot(connection)
            # A connection cannot checkpoint while holding its own read lock.
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
