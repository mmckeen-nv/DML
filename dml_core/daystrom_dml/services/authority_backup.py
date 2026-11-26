"""Offline, full-authority backup and restore for receipt journals.

An internally valid old backup cannot prove the absence of later acknowledged
commits. Compare separately retained client receipts before cutover. These tools
never perform cutover and do not claim physical power-loss qualification.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable

from ..atomic_io import _sync_directory, atomic_write_text
from ..journal import (
    JournalIntegrityError, JournalSchemaError, JournalStateStore, RevisionConflict,
    _decode, _digest, _encode, _initialization_lock, _read_identity, _validated_receipt,
    require_patched_sqlite,
)
from ..store_lock import store_write_lock

DATABASE = "dml_state.sqlite3"
IDENTITY = DATABASE + ".identity.json"
MARKER = DATABASE + ".migration.json"
MANIFEST = "backup.json"
BACKUP_SCHEMA = "dml-authority-backup-v1"
BACKUP_FAULT_POINTS = (
    "before_copy", "after_copy", "after_identity", "after_manifest",
    "before_publish", "after_publish",
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_RECEIPTS_BYTES = 16 * 1024 * 1024
_TABLES = {
    "records": "bucket,key", "state": "id", "snapshot": "id",
    "decisions": "revision", "identity": "id", "receipts": "scope,key",
    "outbox": "revision", "migration_origin": "id",
}


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise JournalIntegrityError("An existing regular file without a symlink is required")


def _file_identity(path: Path) -> dict[str, Any]:
    _regular(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _authority(connection: sqlite3.Connection, identity: dict, *, sealed: bool = False) -> dict:
    """Use the journal's complete validation without startup writes or adoption."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if type(version) is not int or version not in {2, 3, 4}:
        raise JournalSchemaError("Backup requires journal schema 2, 3 or 4")
    if not sealed and connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        raise JournalIntegrityError("Backup requires an authority in WAL mode")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise JournalIntegrityError("Authority integrity check failed")
    if connection.execute("SELECT id,store_id FROM identity").fetchall() != [(1, identity["store_id"])]:
        raise JournalIntegrityError("Authority identity differs from its sidecar")
    verifier = JournalStateStore.__new__(JournalStateStore)
    verifier._schema_version = version
    verifier._identity = identity
    verifier._validate_layout(connection)
    revision, payload, _ = verifier._read_snapshot(connection)
    names = ["records", "state", "snapshot", "decisions", "identity", "receipts"]
    if version >= 3:
        names.append("outbox")
    if version == 4:
        names.append("migration_origin")
    rows = {name: connection.execute(f"SELECT * FROM {name} ORDER BY {_TABLES[name]}").fetchall()
            for name in names}
    return {
        "store_id": identity["store_id"], "journal_schema_version": version,
        "revision": revision, "state_digest": _digest(_encode(payload)),
        "authority_digest": _digest(_encode(rows)), "receipt_count": len(rows["receipts"]),
    }


def _open(path: Path, *, sealed: bool = False) -> sqlite3.Connection:
    if sealed:
        with path.open("rb") as handle:
            header = handle.read(20)
        if header[:16] != b"SQLite format 3\0" or header[18:20] != b"\x02\x02":
            raise JournalIntegrityError("Sealed authority requires a WAL-format database header")
        if any(os.path.lexists(Path(str(path) + suffix)) for suffix in ("-wal", "-shm", "-journal")):
            raise JournalIntegrityError("Sealed authority cannot depend on journal sidecars")
    suffix = "?mode=ro&immutable=1" if sealed else "?mode=ro"
    return sqlite3.connect(path.resolve().as_uri() + suffix, uri=True, timeout=30)


def _read_report(database: Path, identity: dict, *, sealed: bool = False) -> dict:
    with closing(_open(database, sealed=sealed)) as connection:
        connection.execute("BEGIN")
        return _authority(connection, identity, sealed=sealed)


def _reserve(source_directory: Path, destination: Path) -> Path:
    raw = Path(destination)
    if os.path.lexists(raw):
        raise ValueError("A new, separate destination directory is required")
    destination = raw.resolve()
    source_directory = source_directory.resolve()
    if destination == source_directory or source_directory in destination.parents or destination in source_directory.parents:
        raise ValueError("Destination must be separate from the source directory")
    # The caller provisions a parent on its chosen filesystem; never create an
    # unexpected directory hierarchy or reuse another operation's destination.
    staging = Path(tempfile.mkdtemp(prefix=".dml-backup-staging-", dir=destination.parent))
    atomic_write_text(staging / MARKER, _encode({"operation": "authority-backup-or-restore", "schema_version": 1}))
    # Publish the blocked reservation atomically: the requested path is never
    # visible as an empty, apparently uninitialized store. A persistent parent
    # lock serializes cooperating backup/restore publishers; do not unlink it.
    publish_lock = destination.parent / ".dml_backup_publish.lock"
    if publish_lock.is_symlink():
        raise ValueError("Backup publication lock must not be a symlink")
    with _initialization_lock(publish_lock):
        if os.path.lexists(destination):
            raise ValueError("Destination was reserved by another operation")
        os.rename(staging, destination)
        _sync_directory(destination.parent)
    return destination


def _sync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _receipt_check(connection: sqlite3.Connection, authority: dict, receipts: list[dict] | None) -> dict:
    if receipts is None:
        return {"provided": 0, "matched": 0, "coverage": "not_provided"}
    if type(receipts) is not list or len(_encode(receipts).encode("utf-8")) > MAX_RECEIPTS_BYTES:
        raise ValueError("Receipts must be a JSON array no larger than 16 MiB")
    for value in receipts:
        receipt = _validated_receipt(value)
        if receipt["store_id"] != authority["store_id"] or receipt["revision"] > authority["revision"]:
            raise JournalIntegrityError("Retained receipt is outside this backup authority or revision")
        row = connection.execute("SELECT payload FROM receipts WHERE scope=? AND key=?",
                                 (_encode(receipt["scope"]), receipt["key"])).fetchone()
        if row is None or _encode(_decode(row[0])) != _encode(receipt):
            raise JournalIntegrityError("Retained receipt does not match backup history")
    return {"provided": len(receipts), "matched": len(receipts), "coverage": "provided_receipts_only"}


def _verify(directory: Path, *, receipts: list[dict] | None = None, incomplete: bool = False) -> dict:
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise JournalIntegrityError("Backup must be a regular directory")
    expected = {DATABASE, IDENTITY, MANIFEST} | ({MARKER} if incomplete else set())
    if {path.name for path in directory.iterdir()} != expected:
        raise JournalIntegrityError("Backup is incomplete or contains unexpected files")
    _regular(directory / MANIFEST)
    if (directory / MANIFEST).stat().st_size > MAX_MANIFEST_BYTES:
        raise JournalIntegrityError("Backup manifest exceeds its size limit")
    manifest = _decode((directory / MANIFEST).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "created_at_utc", "authority", "files"}
            or manifest["schema_version"] != BACKUP_SCHEMA
            or not isinstance(manifest["created_at_utc"], str)
            or not isinstance(manifest["files"], dict) or set(manifest["files"]) != {DATABASE, IDENTITY}):
        raise JournalIntegrityError("Unsupported backup manifest")
    for name in (DATABASE, IDENTITY):
        recorded = manifest["files"][name]
        if (not isinstance(recorded, dict) or set(recorded) != {"sha256", "bytes"}
                or type(recorded["bytes"]) is not int or recorded["bytes"] <= 0
                or recorded != _file_identity(directory / name)):
            raise JournalIntegrityError("Backup file differs from its manifest")
    identity = _read_identity(directory / IDENTITY)
    with closing(_open(directory / DATABASE, sealed=True)) as connection:
        connection.execute("BEGIN")
        authority = _authority(connection, identity, sealed=True)
        if _encode(authority) != _encode(manifest["authority"]):
            raise JournalIntegrityError("Backup authority differs from its manifest")
        comparison = _receipt_check(connection, authority, receipts)
    return {**manifest, "receipt_comparison": comparison}


def verify_backup(directory: Path, *, receipts: list[dict] | None = None) -> dict:
    """Verify a sealed backup; optional receipts attest only supplied history."""
    require_patched_sqlite()
    try:
        return _verify(Path(directory), receipts=receipts)
    except sqlite3.DatabaseError as exc:
        raise JournalIntegrityError("Backup database cannot be verified") from exc


def backup_authority(source: Path, destination: Path, *,
                     fault_hook: Callable[[str], None] | None = None) -> dict:
    """Back up a stopped-writer authority into an unused directory, never cut over.

    A failed destination is quarantine evidence. Preserve it and retry a new
    location; do not remove its migration marker or provision it as a new store.
    """
    require_patched_sqlite()
    source = Path(source)
    _regular(source)
    source = source.resolve()
    identity_path = source.with_name(source.name + ".identity.json")
    _regular(identity_path)
    marker = source.with_name(source.name + ".migration.json")
    if os.path.lexists(marker):
        raise JournalIntegrityError("Incomplete source cannot be backed up")
    fault = fault_hook or (lambda _point: None)
    with store_write_lock(source.parent, operation="authority-backup", timeout_ms=30000):
        identity = _read_identity(identity_path)
        with closing(_open(source)) as origin:
            origin.execute("BEGIN")
            report = _authority(origin, identity)
            destination = _reserve(source.parent, Path(destination))
            fault("before_copy")
            with closing(sqlite3.connect(destination / DATABASE)) as target:
                origin.backup(target)
                target.execute("PRAGMA journal_mode=WAL")
                target.execute("PRAGMA synchronous=FULL")
            _sync_file(destination / DATABASE)
            fault("after_copy")
            atomic_write_text(destination / IDENTITY, _encode(identity))
            fault("after_identity")
            if _read_report(destination / DATABASE, identity, sealed=True) != report:
                raise JournalIntegrityError("Copied authority differs from the captured source")
        if (_read_identity(identity_path) != identity or os.path.lexists(marker)
                or _read_report(source, identity) != report):
            raise RevisionConflict("Source advanced or changed; stop writers and retry a new destination")
        manifest = {"schema_version": BACKUP_SCHEMA,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "authority": report,
                    "files": {name: _file_identity(destination / name) for name in (DATABASE, IDENTITY)}}
        atomic_write_text(destination / MANIFEST, _encode(manifest))
        fault("after_manifest")
        result = _verify(destination, incomplete=True)
        fault("before_publish")
        (destination / MARKER).unlink()
        _sync_directory(destination)
        fault("after_publish")
        return result


def restore_backup(source: Path, destination: Path, *, receipts: list[dict] | None = None,
                   fault_hook: Callable[[str], None] | None = None) -> dict:
    """Restore a verified captured revision into a new directory; never cut over."""
    source = Path(source)
    verified = verify_backup(source, receipts=receipts)
    fault = fault_hook or (lambda _point: None)
    destination = _reserve(source.resolve(), Path(destination))
    fault("before_copy")
    # Copy the sealed bytes rather than opening an operational SQLite store.
    # Revalidation below detects a changed backup during the copy.
    for name, point in ((DATABASE, "after_copy"), (IDENTITY, "after_identity"), (MANIFEST, "after_manifest")):
        with (source / name).open("rb") as origin, (destination / name).open("xb") as target:
            for block in iter(lambda: origin.read(1024 * 1024), b""):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        fault(point)
    result = _verify(destination, receipts=receipts, incomplete=True)
    if result != verified or verify_backup(source, receipts=receipts) != verified:
        raise JournalIntegrityError("Backup changed while restoring")
    fault("before_publish")
    (destination / MARKER).unlink()
    _sync_directory(destination)
    fault("after_publish")
    return result
