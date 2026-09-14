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
RECEIPT_JOURNAL_SCHEMA_VERSION = 2
OUTBOX_JOURNAL_SCHEMA_VERSION = 3
MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION = 4
OUTBOX_EVENT_FORMAT = "dml-journal-outbox-v1"
RECEIPT_SCOPE_KEYS = {"tenant_id", "client_id", "session_id", "instance_id"}
INIT_FAULT_POINTS = (
    "init_before_connect", "init_after_connect", "init_after_begin",
    "init_after_schema", "init_after_commit", "init_before_identity",
    "init_after_identity",
)
FAULT_POINTS = (
    "after_begin", "after_records", "after_state", "after_snapshot",
    "after_decision", "before_commit", "after_commit",
)

RECEIPT_FAULT_POINTS = (*FAULT_POINTS[:4], "after_receipt", *FAULT_POINTS[4:])
OUTBOX_FAULT_POINTS = (*RECEIPT_FAULT_POINTS[:-2], "after_outbox", *RECEIPT_FAULT_POINTS[-2:])


class IdempotencyConflict(RuntimeError):
    """A scoped idempotency key was already committed for another request."""


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


def _receipt_scope(scope):
    if not isinstance(scope, dict) or set(scope) != RECEIPT_SCOPE_KEYS:
        raise ValueError("receipt scope requires tenant_id, client_id, session_id and instance_id")
    if not isinstance(scope["tenant_id"], str) or not scope["tenant_id"].strip():
        raise ValueError("receipt tenant_id must be a nonempty string")
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in scope.values()):
        raise ValueError("receipt scope values must be nonempty strings or null")
    if len(_encode(scope).encode("utf-8")) > 4096:
        raise ValueError("receipt scope exceeds 4096 bytes")
    return dict(scope)


def _receipt_identity(scope, key, request_digest):
    scope = _receipt_scope(scope)
    if not isinstance(key, str) or not key.strip() or len(key.encode("utf-8")) > 256:
        raise ValueError("idempotency key must be a nonempty string of at most 256 UTF-8 bytes")
    if not _is_digest(request_digest):
        raise ValueError("request_digest must be a canonical SHA-256 digest")
    return scope, key, request_digest


def _validated_receipt(receipt):
    if not isinstance(receipt, dict) or set(receipt) != {"schema_version", "store_id", "scope", "key", "request_digest", "revision", "result"}:
        raise JournalIntegrityError("Invalid journal receipt fields")
    _version(receipt)
    _valid_identity({"schema_version": 1, "store_id": receipt["store_id"]})
    try:
        _receipt_identity(receipt["scope"], receipt["key"], receipt["request_digest"])
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JournalIntegrityError("Invalid journal receipt identity") from exc
    if type(receipt["revision"]) is not int or receipt["revision"] <= 0:
        raise JournalIntegrityError("Invalid journal receipt revision")
    result = receipt["result"]
    if not isinstance(result, dict) or set(result) != {"memory"} or not isinstance(result["memory"], dict):
        raise JournalIntegrityError("Receipt result must contain exactly one committed memory")
    memory = result["memory"]
    if type(memory.get("id")) is not int or memory["id"] < 0:
        raise JournalIntegrityError("Invalid receipt memory identity")
    _version(memory)
    meta = memory.get("meta")
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in receipt["scope"].items()):
        raise JournalIntegrityError("Receipt memory scope mismatch")
    return receipt


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
                 fault_hook: Callable[[str], None] | None = None,
                 receipt_mode: bool = False, outbox_mode: bool = False):
        if type(receipt_mode) is not bool:
            raise ValueError("receipt_mode must be a boolean")
        if type(outbox_mode) is not bool or (outbox_mode and not receipt_mode):
            raise ValueError("outbox_mode must be a boolean and requires receipt_mode=True")
        self._receipt_mode = receipt_mode
        self._outbox_mode = outbox_mode
        self._schema_version = OUTBOX_JOURNAL_SCHEMA_VERSION if outbox_mode else (RECEIPT_JOURNAL_SCHEMA_VERSION if receipt_mode else JOURNAL_SCHEMA_VERSION)
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = max(1, snapshot_interval)
        self.last_changed_rows = 0
        self._revision = -1
        self._encoded: dict[tuple[str, str], tuple[int, str]] = {}
        self._fault_hook = fault_hook or (lambda _point: None)
        self.identity_path = self.path.with_name(self.path.name + ".identity.json")
        self.initialization_lock_path = self.path.with_name(self.path.name + ".init.lock")
        self.migration_path = self.path.with_name(self.path.name + ".migration.json")
        with _initialization_lock(self.initialization_lock_path):
            self._initialize()

    def _initialize(self):
        if self.migration_path.exists():
            raise JournalIntegrityError("Incomplete journal migration; preserve destination and retry with a new path")
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
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                self._validate_version(version)
                if self._outbox_mode and version not in {OUTBOX_JOURNAL_SCHEMA_VERSION, MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION}:
                    raise JournalSchemaError("Outbox mode requires schema 3 or 4; schema-2 journals require explicit upgrade_outbox_journal to a new destination")
                if self._receipt_mode and version < RECEIPT_JOURNAL_SCHEMA_VERSION:
                    raise JournalSchemaError("Receipt mode requires explicit upgrade_receipt_journal to a new schema-2 database")
                self._schema_version = version
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
                    if self._schema_version >= RECEIPT_JOURNAL_SCHEMA_VERSION:
                        self._create_receipt_table(connection)
                    if self._schema_version >= OUTBOX_JOURNAL_SCHEMA_VERSION:
                        connection.execute("CREATE TABLE outbox (revision INTEGER PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)")
                    connection.execute(f"PRAGMA user_version={self._schema_version}")
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
        if version not in {JOURNAL_SCHEMA_VERSION, RECEIPT_JOURNAL_SCHEMA_VERSION, OUTBOX_JOURNAL_SCHEMA_VERSION, MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION}:
            raise JournalSchemaError(
                f"Unsupported journal schema {version}; expected 1, 2, 3 or 4. "
                "Schema 0 requires explicit dml-journal upgrade to a new database."
            )

    @staticmethod
    def _create_receipt_table(connection):
        connection.execute("CREATE TABLE receipts (scope TEXT NOT NULL, key TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(scope,key))")

    @property
    def schema_version(self) -> int:
        return self._schema_version

    def _validate_layout(self, connection):
        columns = {
            "records": [("bucket", "TEXT", 0, 1), ("key", "TEXT", 0, 2), ("position", "INTEGER", 0, 0), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "state": [("id", "INTEGER", 0, 1), ("revision", "INTEGER", 0, 0), ("metadata", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "snapshot": [("id", "INTEGER", 0, 1), ("revision", "INTEGER", 0, 0), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "decisions": [("revision", "INTEGER", 0, 1), ("payload", "TEXT", 0, 0), ("checksum", "TEXT", 1, 0)],
            "identity": [("id", "INTEGER", 0, 1), ("store_id", "TEXT", 1, 0)],
        }
        if self._schema_version >= RECEIPT_JOURNAL_SCHEMA_VERSION:
            columns["receipts"] = [("scope", "TEXT", 1, 1), ("key", "TEXT", 1, 2), ("revision", "INTEGER", 1, 0), ("payload", "TEXT", 1, 0), ("checksum", "TEXT", 1, 0)]
        if self._schema_version >= OUTBOX_JOURNAL_SCHEMA_VERSION:
            columns["outbox"] = [("revision", "INTEGER", 0, 1), ("payload", "TEXT", 1, 0), ("checksum", "TEXT", 1, 0)]
        if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION:
            columns["migration_origin"] = [("id", "INTEGER", 0, 1), ("payload", "TEXT", 1, 0), ("checksum", "TEXT", 1, 0)]
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
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                self._validate_version(version)
                if version != self._schema_version:
                    raise JournalSchemaError("Journal schema changed while open")
                if self.migration_path.exists():
                    raise JournalIntegrityError("Incomplete journal migration")
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

    def _migration_origin(self, connection):
        from .services.journal_outbox import validate_migration_origin

        rows = connection.execute("SELECT id,payload,checksum FROM migration_origin").fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise JournalIntegrityError("Migration origin is missing or ambiguous")
        return validate_migration_origin(_checked(rows[0][1], rows[0][2]))

    def _decision(self, sql_revision, raw, checksum, *, expected_schema=None):
        decision = _checked(raw, checksum)
        if not isinstance(decision, dict):
            raise JournalIntegrityError("Journal decision must be an object")
        schema = self._schema_version if expected_schema is None else expected_schema
        if type(decision.get("schema_version")) is not int or decision["schema_version"] != schema:
            raise JournalSchemaError("Journal decision schema mismatch")
        fields = {"schema_version", "revision", "operation", "changed", "deleted", "record_count", "state_digest"}
        if schema >= RECEIPT_JOURNAL_SCHEMA_VERSION:
            fields.add("receipt")
        if schema >= OUTBOX_JOURNAL_SCHEMA_VERSION:
            fields.add("outbox_digest")
            if not _is_digest(decision.get("outbox_digest")):
                raise JournalIntegrityError("Invalid decision outbox binding")
        if set(decision) != fields:
            raise JournalIntegrityError("Invalid journal decision fields")
        if schema >= RECEIPT_JOURNAL_SCHEMA_VERSION and decision["receipt"] is not None:
            binding = decision["receipt"]
            if not isinstance(binding, dict) or set(binding) != {"scope", "key", "request_digest", "digest"} or not _is_digest(binding["digest"]):
                raise JournalIntegrityError("Invalid decision receipt binding")
            try:
                _receipt_identity(binding["scope"], binding["key"], binding["request_digest"])
            except (TypeError, ValueError, UnicodeError) as exc:
                raise JournalIntegrityError("Invalid decision receipt identity") from exc
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
        if self._schema_version >= OUTBOX_JOURNAL_SCHEMA_VERSION:
            identity_rows = connection.execute("SELECT id,store_id FROM identity").fetchall()
            if identity_rows != [(1, self._identity["store_id"])]:
                raise JournalIntegrityError("Journal outbox transaction identity changed")
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
        origin = self._migration_origin(connection) if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION else None
        decisions = {}
        expected_records = {}
        for expected_revision, event in enumerate(events, 1):
            if event[0] != expected_revision:
                raise JournalIntegrityError("Journal decision history is not contiguous")
            expected_schema = RECEIPT_JOURNAL_SCHEMA_VERSION if origin is not None and expected_revision <= origin["source_revision"] else self._schema_version
            decision = self._decision(*event, expected_schema=expected_schema)
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
        if self._schema_version >= RECEIPT_JOURNAL_SCHEMA_VERSION:
            expected_receipts = {}
            for decision_revision, decision in decisions.items():
                binding = decision["receipt"]
                if binding is not None:
                    identity = (_encode(binding["scope"]), binding["key"])
                    if identity in expected_receipts:
                        raise JournalIntegrityError("Duplicate receipt in decision history")
                    expected_receipts[identity] = (decision_revision, binding)
            receipt_rows = connection.execute("SELECT scope,key,revision,payload,checksum FROM receipts").fetchall()
            if len(receipt_rows) != len(expected_receipts):
                raise JournalIntegrityError("Journal receipt history is incomplete")
            for scope, key, receipt_revision, raw, checksum in receipt_rows:
                if not isinstance(scope, str) or not isinstance(key, str):
                    raise JournalIntegrityError("Invalid receipt row identity")
                receipt = _validated_receipt(_checked(raw, checksum))
                expected = expected_receipts.get((scope, key))
                if expected is None or type(receipt_revision) is not int or expected[0] != receipt_revision:
                    raise JournalIntegrityError("Receipt has no matching commit decision")
                binding = expected[1]
                if receipt["revision"] != receipt_revision or receipt["store_id"] != self._identity["store_id"] or _encode(receipt["scope"]) != scope or receipt["key"] != key or receipt["request_digest"] != binding["request_digest"] or checksum != binding["digest"]:
                    raise JournalIntegrityError("Receipt differs from committed decision")
                memory = receipt["result"]["memory"]
                if not any(entry["bucket"] == "items" and entry["id"] == str(memory["id"]) and entry["digest"] == _digest(_encode(memory)) for entry in decisions[receipt_revision]["changed"]):
                    # A receipt may acknowledge an existing unchanged record.
                    historical_digest = None
                    for prior in range(1, receipt_revision + 1):
                        for entry in decisions[prior]["deleted"]:
                            if entry["bucket"] == "items" and entry["id"] == str(memory["id"]):
                                historical_digest = None
                        for entry in decisions[prior]["changed"]:
                            if entry["bucket"] == "items" and entry["id"] == str(memory["id"]):
                                historical_digest = entry["digest"]
                    if historical_digest != _digest(_encode(memory)):
                        raise JournalIntegrityError("Receipt memory was not present at its commit")
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
        if self._schema_version >= OUTBOX_JOURNAL_SCHEMA_VERSION:
            self._validate_outbox(connection, revision, decisions)
        return revision, payload, encoded

    def _validate_outbox(self, connection, revision, decisions):
        from .services.journal_outbox import validate_outbox_event

        rows = connection.execute("SELECT revision,payload,checksum FROM outbox ORDER BY revision").fetchall()
        origin = self._migration_origin(connection) if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION else None
        boundary = origin["source_revision"] if origin is not None else 0
        if origin is not None:
            if revision <= boundary or origin["history_digest"] != _digest(_encode([decisions[index] for index in range(1, boundary + 1)])):
                raise JournalIntegrityError("Migration origin history mismatch")
            if boundary and decisions[boundary]["state_digest"] != origin["source_digest"]:
                raise JournalIntegrityError("Migration origin state mismatch")
            if not boundary and origin["source_digest"] != _digest(_encode({"items": [], "lineage": []})):
                raise JournalIntegrityError("Empty migration origin state mismatch")
        if len(rows) != revision - boundary:
            raise JournalIntegrityError("Journal outbox history is incomplete")
        previous = {}
        for expected_revision, (sql_revision, raw, checksum) in enumerate(rows, boundary + 1):
            event = validate_outbox_event(_checked(raw, checksum))
            if origin is None:
                if event["schema_version"] != 1:
                    raise JournalSchemaError("Schema-3 outbox requires version-1 events")
            elif event["schema_version"] != 2 or _encode(event["origin"]) != _encode(origin):
                raise JournalIntegrityError("Outbox migration origin mismatch")
            if sql_revision != expected_revision or event["source_revision"] != expected_revision:
                raise JournalIntegrityError("Journal outbox history is not contiguous")
            if event["source_store_id"] != self._identity["store_id"]:
                raise JournalIntegrityError("Journal outbox belongs to another authority")
            decision = decisions[expected_revision]
            base_decision = {key: value for key, value in decision.items() if key != "outbox_digest"}
            if decision["outbox_digest"] != event["checksum"] or event["decision_digest"] != _digest(_encode(base_decision)):
                raise JournalIntegrityError("Journal outbox decision binding mismatch")
            if event["source_digest"] != decision["state_digest"] or event["operation"] != decision["operation"] or _encode(event["receipt"]) != _encode(decision["receipt"]):
                raise JournalIntegrityError("Journal outbox differs from its commit decision")
            current = {}
            for bucket in ("items", "lineage"):
                for position, record in enumerate(event["state"][bucket]):
                    current[(bucket, str(record["id"]))] = (position, _encode(record))
            if origin is not None and expected_revision == boundary + 1:
                if event["operation"] != "outbox-migration-baseline-v1" or event["receipt"] is not None or event["source_digest"] != origin["source_digest"]:
                    raise JournalIntegrityError("Invalid migration baseline")
                previous = current
            changed = [{"bucket": bucket, "id": key, "digest": _digest(value[1])}
                       for (bucket, key), value in current.items() if previous.get((bucket, key)) != value]
            deleted = [{"bucket": bucket, "id": key} for bucket, key in sorted(previous.keys() - current.keys())]
            if _encode(changed) != _encode(decision["changed"]) or _encode(deleted) != _encode(decision["deleted"]) or len(current) != decision["record_count"]:
                raise JournalIntegrityError("Journal outbox state transition differs from decision history")
            previous = current

    def outbox_events(self, *, after_revision: int = 0, limit: int = 100) -> dict:
        """Read one bounded event page and authority head from a verified snapshot.

        Schema 3 retains a complete state per committed revision. Schema 4
        starts with an explicit migration baseline; older decisions remain
        auditable without fabricated full-state events. The page bounds event
        count, not bytes or verification cost; all history is verified.
        Delivery positions are consumer-owned and never mutate this history.
        """
        from .services.journal_outbox import validate_outbox_event

        if self._schema_version not in {OUTBOX_JOURNAL_SCHEMA_VERSION, MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION}:
            raise JournalSchemaError("Transactional outbox requires a schema-3 or schema-4 journal")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("outbox page size must be an integer between 1 and 1000")
        if type(after_revision) is not int or after_revision < 0:
            raise ValueError("after_revision must be a nonnegative integer")
        with self._connect() as connection:
            connection.execute("BEGIN")
            identity_rows = connection.execute("SELECT id,store_id FROM identity").fetchall()
            if identity_rows != [(1, self._identity["store_id"])]:
                raise JournalIntegrityError("Journal outbox identity changed")
            revision, _, _ = self._read_snapshot(connection)
            origin = self._migration_origin(connection) if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION else None
            if origin is not None and 0 < after_revision <= origin["source_revision"]:
                raise ValueError("after_revision precedes the migrated outbox baseline")
            if after_revision > revision:
                raise ValueError("after_revision exceeds the verified authority head")
            rows = connection.execute("SELECT payload,checksum FROM outbox WHERE revision>? ORDER BY revision LIMIT ?", (after_revision, limit)).fetchall()
            events = [validate_outbox_event(_checked(*row)) for row in rows]
            prefix_rows = connection.execute("SELECT payload,checksum FROM outbox WHERE revision<=? ORDER BY revision", (after_revision,)).fetchall()
            prefix_digest = _digest(_encode([_checked(*row)["checksum"] for row in prefix_rows]))
            next_revision = events[-1]["source_revision"] if events else after_revision
            page = {"schema_version": 1, "source_store_id": identity_rows[0][1],
                    "head_revision": revision, "after_revision": after_revision,
                    "next_revision": next_revision, "has_more": next_revision < revision,
                    "prefix_digest": prefix_digest, "events": events}
            if origin is not None:
                page.update(schema_version=2, origin=origin)
            return page

    def read_snapshot(self) -> tuple[int, dict]:
        """Return a detached state and its revision from one read transaction."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            revision, payload, encoded = self._read_snapshot(connection)
        self._revision, self._encoded = revision, encoded
        return revision, payload

    def verified_snapshot(self) -> tuple[str, int, dict]:
        """Read verified authority identity, revision and state together."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                rows = connection.execute("SELECT id,store_id FROM identity").fetchall()
            except sqlite3.DatabaseError as exc:
                raise JournalIntegrityError("Journal snapshot identity cannot be read") from exc
            if len(rows) != 1 or rows[0] != (1, self._identity["store_id"]):
                raise JournalIntegrityError("Journal snapshot identity changed")
            revision, payload, _ = self._read_snapshot(connection)
            return rows[0][1], revision, payload

    def load(self) -> dict:
        return self.read_snapshot()[1]

    def _require_receipts(self):
        if self._schema_version < RECEIPT_JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError("Receipts require explicit upgrade_receipt_journal to a new schema-2 database")

    @staticmethod
    def _lookup_receipt(connection, scope, key, request_digest):
        row = connection.execute("SELECT payload,checksum FROM receipts WHERE scope=? AND key=?", (_encode(scope), key)).fetchone()
        if row is None:
            return None
        receipt = _validated_receipt(_checked(*row))
        if receipt["request_digest"] != request_digest:
            raise IdempotencyConflict("Idempotency key is already committed for a different request")
        return receipt

    def lookup_receipt(self, scope: dict, key: str, request_digest: str) -> dict | None:
        """Validate authority and return the original detached receipt, if committed."""
        self._require_receipts()
        scope, key, request_digest = _receipt_identity(scope, key, request_digest)
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._read_snapshot(connection)
            return self._lookup_receipt(connection, scope, key, request_digest)

    def save(self, payload: dict, *, expected_revision: int | None = None,
             operation: str = "persist") -> None:
        self._save(payload, expected_revision=expected_revision, operation=operation)

    def save_with_receipt(self, payload: dict, *, scope: dict, key: str,
                          request_digest: str, result: dict, expected_revision: int,
                          operation: str = "ingest_receipt") -> dict:
        """Commit one memory and its receipt atomically, or return the prior receipt.

        A retry lookup precedes CAS inside the write transaction. ``result`` must
        contain exactly ``memory``, equal to an item in the committed snapshot.
        """
        self._require_receipts()
        scope, key, request_digest = _receipt_identity(scope, key, request_digest)
        spec = _decode(_encode({"scope": scope, "key": key, "request_digest": request_digest, "result": result}))
        receipt = self._save(payload, expected_revision=expected_revision, operation=operation, receipt_spec=spec)
        assert receipt is not None
        return receipt

    def _save(self, payload: dict, *, expected_revision: int | None = None,
              operation: str = "persist", receipt_spec: dict | None = None) -> dict | None:
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
            if receipt_spec is not None:
                prior_receipt = self._lookup_receipt(connection, receipt_spec["scope"], receipt_spec["key"], receipt_spec["request_digest"])
                if prior_receipt is not None:
                    self._revision, self._encoded = revision, previous
                    self.last_changed_rows = 0
                    return prior_receipt
            previous_meta = self._encode({key: value for key, value in current.items() if key not in {"items", "lineage"}})
            if expected_revision is not None and revision != expected_revision:
                raise RevisionConflict(f"Expected revision {expected_revision}, found {revision}")
            changed = [(bucket, key, *value, _digest(value[1])) for (bucket, key), value in encoded.items() if previous.get((bucket, key)) != value]
            deleted = sorted(previous.keys() - encoded.keys())
            receipt = None
            if receipt_spec is not None:
                receipt = _validated_receipt({"schema_version": 1, "store_id": self._identity["store_id"], "revision": revision + 1, **receipt_spec})
                memory = receipt["result"]["memory"]
                if not any(record["id"] == memory["id"] and self._encode(record) == self._encode(memory) for record in normalized["items"]):
                    raise JournalIntegrityError("Receipt memory does not match the committed snapshot")
            if receipt is not None or revision == 0 or changed or deleted or previous_meta != metadata:
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
                binding = None
                if receipt is not None:
                    raw_receipt = self._encode(receipt)
                    receipt_digest = _digest(raw_receipt)
                    connection.execute("INSERT INTO receipts VALUES (?,?,?,?,?)", (self._encode(receipt["scope"]), receipt["key"], revision, raw_receipt, receipt_digest))
                    binding = {"scope": receipt["scope"], "key": receipt["key"], "request_digest": receipt["request_digest"], "digest": receipt_digest}
                    self._fault_hook("after_receipt")
                event = {
                    "schema_version": self._schema_version, "revision": revision, "operation": operation,
                    "changed": [{"bucket": bucket, "id": key, "digest": checksum} for bucket, key, _, _, checksum in changed],
                    "deleted": [{"bucket": bucket, "id": key} for bucket, key in deleted],
                    "record_count": len(encoded), "state_digest": _digest(self._encode(normalized)),
                }
                if self._schema_version >= RECEIPT_JOURNAL_SCHEMA_VERSION:
                    event["receipt"] = binding
                outbox = None
                if self._schema_version >= OUTBOX_JOURNAL_SCHEMA_VERSION:
                    outbox = {
                        "schema_version": 1, "event_format": OUTBOX_EVENT_FORMAT,
                        "source_store_id": self._identity["store_id"], "source_revision": revision,
                        "source_digest": event["state_digest"], "operation": operation,
                        "receipt": binding, "state": normalized,
                        "decision_digest": _digest(self._encode(event)),
                    }
                    if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION:
                        outbox.update(schema_version=2, event_format="dml-journal-outbox-v2", origin=self._migration_origin(connection))
                    outbox["checksum"] = _digest(self._encode(outbox))
                    event["outbox_digest"] = outbox["checksum"]
                decision = self._encode(event)
                connection.execute("INSERT INTO decisions VALUES (?,?,?)", (revision, decision, _digest(decision)))
                self._fault_hook("after_decision")
                if outbox is not None:
                    raw_outbox = self._encode(outbox)
                    connection.execute("INSERT INTO outbox VALUES (?,?,?)", (revision, raw_outbox, _digest(raw_outbox)))
                    self._fault_hook("after_outbox")
            self._fault_hook("before_commit")
        self._fault_hook("after_commit")
        self._revision, self._encoded = revision, encoded
        self.last_changed_rows = len(changed) + len(deleted)
        return receipt

    def decisions(self, *, after_revision: int = 0, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("decision page size must be an integer between 1 and 1000")
        if type(after_revision) is not int or after_revision < 0:
            raise ValueError("after_revision must be a nonnegative integer")
        with self._connect() as connection:
            connection.execute("BEGIN")
            self._read_snapshot(connection)
            rows = connection.execute("SELECT revision,payload,checksum FROM decisions WHERE revision>? ORDER BY revision LIMIT ?", (after_revision, limit)).fetchall()
            origin = self._migration_origin(connection) if self._schema_version == MIGRATED_OUTBOX_JOURNAL_SCHEMA_VERSION else None
            return [self._decision(*row, expected_schema=RECEIPT_JOURNAL_SCHEMA_VERSION if origin is not None and row[0] <= origin["source_revision"] else self._schema_version) for row in rows]

    def export_snapshot(self, path: Path) -> Path:
        if self._schema_version >= RECEIPT_JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError("Receipt/outbox journals require a full database backup; lattice-only export loses receipt authority")
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


RECEIPT_MIGRATION_FAULT_POINTS = (
    "migration_after_marker", "migration_after_backup", "migration_before_commit",
    "migration_after_commit", "migration_after_identity", "migration_after_publish",
)


def upgrade_receipt_journal(source: Path, destination: Path, *,
                            fault_hook: Callable[[str], None] | None = None) -> None:
    """Explicitly copy schema 1 to schema 2 without resetting identity or history.

    Stop writers before cutover. A durable migration marker blocks every normal
    opener until the destination is fully verified and its identity published.
    Interrupted targets are preserved for recovery; retry using a new path.
    This upgrades lattice journals only; there are no receipts to translate yet.
    """
    from contextlib import closing
    from .atomic_io import _sync_directory
    from .store_lock import store_write_lock

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or not source.is_file():
        raise ValueError("receipt upgrade requires an existing source and a new separate destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_name(destination.name + ".migration.json")
    identity_path = destination.with_name(destination.name + ".identity.json")
    init_lock = destination.with_name(destination.name + ".init.lock")
    fault = fault_hook or (lambda _point: None)
    with store_write_lock(source.parent, operation="journal-receipt-upgrade", timeout_ms=30000):
        original = JournalStateStore(source)
        if original.schema_version != JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError("receipt upgrade accepts only a schema-1 source")
        with _initialization_lock(init_lock):
            if any(path.exists() for path in (destination, marker, identity_path, Path(str(destination) + "-wal"), Path(str(destination) + "-shm"))):
                raise ValueError("receipt upgrade requires an unused destination with no journal sidecars")
            with original._connect() as source_connection:
                source_connection.execute("BEGIN")
                revision, payload, _ = original._read_snapshot(source_connection)
                atomic_write_text(marker, _encode({"schema_version": 1, "source_store_id": original._identity["store_id"], "source_revision": revision}))
                fault("migration_after_marker")
                # A target can never masquerade as an empty authoritative store:
                # its marker is durable before the database name is created.
                destination.touch(exist_ok=False)
                with closing(sqlite3.connect(destination, timeout=30)) as target:
                    source_connection.backup(target)
                    fault("migration_after_backup")
                    target.execute("PRAGMA journal_mode=WAL")
                    target.execute("PRAGMA synchronous=FULL")
                    with target:
                        target.execute("BEGIN IMMEDIATE")
                        JournalStateStore._create_receipt_table(target)
                        for event_revision, raw, checksum in target.execute("SELECT revision,payload,checksum FROM decisions ORDER BY revision").fetchall():
                            event = original._decision(event_revision, raw, checksum)
                            event.update(schema_version=RECEIPT_JOURNAL_SCHEMA_VERSION, receipt=None)
                            encoded = _encode(event)
                            target.execute("UPDATE decisions SET payload=?,checksum=? WHERE revision=?", (encoded, _digest(encoded), event_revision))
                        target.execute(f"PRAGMA user_version={RECEIPT_JOURNAL_SCHEMA_VERSION}")
                        fault("migration_before_commit")
                    fault("migration_after_commit")
                    verifier = JournalStateStore.__new__(JournalStateStore)
                    verifier._schema_version = RECEIPT_JOURNAL_SCHEMA_VERSION
                    verifier._identity = dict(original._identity)
                    with target:
                        target.execute("BEGIN")
                        if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                            raise JournalIntegrityError("Upgraded receipt journal integrity check failed")
                        verifier._validate_layout(target)
                        new_revision, new_payload, _ = verifier._read_snapshot(target)
                        if new_revision != revision or new_payload != payload:
                            raise JournalIntegrityError("Upgraded receipt journal differs from source snapshot")
                    atomic_write_text(identity_path, _encode(original._identity))
                    fault("migration_after_identity")
                marker.unlink()
                _sync_directory(destination.parent)
                fault("migration_after_publish")
