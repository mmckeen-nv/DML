"""Lattice storage selection, validation and commits behind a narrow boundary."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol, Sequence, TypedDict

from ..journal import JournalStateStore, RevisionConflict
from ..memory_store import MemoryItem
from ..persistence import validate_snapshot
from .transactions import PersistenceCommitError


LOGGER = logging.getLogger(__name__)
Stamp = tuple[int, int] | None


class LatticePersistence:
    def __init__(self, *, json_path: Path, jsonl_path: Path, use_jsonl: bool,
                 journal: JournalStateStore | None,
                 read_jsonl: Callable, write_jsonl: Callable, write_text: Callable):
        self.json_path = json_path
        self.jsonl_path = jsonl_path
        self.use_jsonl = use_jsonl
        self.journal = journal
        self._read_jsonl, self._write_jsonl, self._write_text = read_jsonl, write_jsonl, write_text

    @property
    def path(self) -> Path:
        return self.journal.path if self.journal else self.jsonl_path if self.use_jsonl else self.json_path

    def stamp(self) -> tuple[int, int] | None:
        if self.journal:
            return self.journal.stamp()
        try:
            stat = self.path.stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def load(self, *, startup: bool = False) -> dict | None:
        """Retain the payload-only interface for existing storage callers."""
        return self.load_with_revision(startup=startup)[0]

    def load_with_revision(self, *, startup: bool = False) -> tuple[dict | None, int | None]:
        """Return validated payload and the revision read in that same snapshot.

        The journal's last-read revision is advisory shared runtime state; a
        concurrent preparation read may replace it before a caller imports its
        payload. File formats have no journal revision and return ``None``.
        """
        revision = None
        if self.journal:
            if startup and self.journal.stamp()[0] == 0 and (self.json_path.exists() or self.jsonl_path.exists()):
                raise ValueError("Journal migration requires an explicit snapshot import")
            revision, payload = self.journal.read_snapshot()
        elif self.use_jsonl and self.jsonl_path.exists():
            payload = {"items": [item.to_dict() for item in self._read_jsonl(self.jsonl_path)]}
        elif self.json_path.exists() and (startup or not self.use_jsonl):
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        elif startup:
            return None, None
        else:
            raise FileNotFoundError("Previously initialized DML state is missing")
        validate_snapshot(payload)
        return payload, revision

    def commit(self, *, payload: dict | None, items: Sequence[MemoryItem] | None,
               expected_revision: int | None, operation: str) -> tuple[int, int] | None:
        if self.journal:
            if payload is None:
                raise ValueError("Journal commit requires a lattice payload")
            self.journal.save(payload, expected_revision=expected_revision, operation=operation)
            return self.journal.revision, self.journal.path.stat().st_mtime_ns
        if self.use_jsonl:
            if items is None:
                raise ValueError("JSONL commit requires memory items")
            self._write_jsonl(items, self.jsonl_path)
        else:
            if payload is None:
                raise ValueError("JSON commit requires a lattice payload")
            self._write_text(self.json_path, json.dumps(payload, indent=2, allow_nan=False))
        return self.stamp()


class LatticeRuntime(Protocol):
    def snapshot_state(self) -> dict[str, Any]: ...
    def import_state(self, payload: dict[str, Any] | None) -> None: ...
    def export_state(self) -> dict[str, Any]: ...
    def items(self) -> Sequence[MemoryItem]: ...


class RAGRuntime(Protocol):
    def snapshot_state(self) -> dict[str, Any]: ...
    def restore_state(self, payload: dict[str, Any]) -> None: ...
    def import_state(self, payload: dict[str, Any] | None) -> None: ...
    def export_state(self) -> dict[str, Any]: ...


class PersistentRAG(Protocol):
    manifest_path: Path

    def snapshot_state(self) -> dict[str, Any]: ...
    def restore_state(self, snapshot: dict[str, Any]) -> None: ...
    def persist(self) -> None: ...
    def load(self) -> bool: ...


class MutationSnapshot(TypedDict):
    dml: dict[str, Any]
    rag: dict[str, Any]
    persistent_rag: dict[str, Any] | None


@dataclass
class PersistenceState:
    """Owned component observations and health; adapter aliases are compatibility views."""

    persist_lock: RLock = field(default_factory=RLock)
    refresh_lock: RLock = field(default_factory=RLock)
    failures: dict[str, str] = field(default_factory=dict)
    observed_lattice: Stamp = None
    observed_rag: Stamp = None
    observed_persistent_rag: Stamp = None
    lattice_recovery_error: BaseException | None = None


_UNKNOWN = object()


class PersistenceCoordinator:
    """Coordinate lattice and RAG components under caller-owned mutation ownership.

    Runtime restoration precedes compensating writes, in lattice/persistent-RAG/
    legacy-RAG order. This is exception recovery under cooperating writer locks,
    not crash atomicity across files. File writes read a pre-write SHA-256 digest
    so a publish-then-raise cannot be mistaken for an unchanged authority based
    on timestamps alone. Journal failure recovery verifies revision and payload
    before permitting compensation; successful journal writes add no extra read.
    """

    def __init__(
        self, *, state: PersistenceState, lattice: LatticePersistence,
        runtime: LatticeRuntime, rag: RAGRuntime, rag_path: Path,
        persistent_rag: Callable[[], PersistentRAG | None],
        invalidate_cache: Callable[[], None],
        write_text: Callable[[Path, str], Any],
        mark_committed: Callable[[str], None],
        require_legacy: Callable[[], None], operation: Callable[[], str],
    ) -> None:
        self.state = state
        self.lattice = lattice
        self.runtime, self.rag, self.rag_path = runtime, rag, rag_path
        self._persistent_rag = persistent_rag
        self._invalidate_cache = invalidate_cache
        self._write_text = write_text
        self._mark_committed = mark_committed
        self._require_legacy = require_legacy
        self._operation = operation

    @staticmethod
    def path_stamp(path: Path) -> Stamp:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return int(stat.st_mtime_ns), int(stat.st_size)

    @staticmethod
    def _file_fingerprint(path: Path) -> bytes | None:
        try:
            with path.open("rb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                return digest.digest()
        except FileNotFoundError:
            return None

    def _before_file_write(self, path: Path) -> bytes | None | object:
        try:
            return self._file_fingerprint(path)
        except Exception:
            # A write can still succeed when the prior generation is unreadable.
            # Failure after such an attempt has an unknown outcome.
            return _UNKNOWN

    def _reconcile_file_failure(self, component: str, path: Path, before: object) -> None:
        after: object
        try:
            after = self._file_fingerprint(path)
        except BaseException:
            after = _UNKNOWN
        if before is _UNKNOWN or after is _UNKNOWN or before != after:
            self._mark_committed(component)

    def capture_snapshot(self) -> MutationSnapshot:
        persistent = self._persistent_rag()
        persistent_snapshot = persistent.snapshot_state() if persistent is not None else None
        return {
            "dml": self.runtime.snapshot_state(),
            "rag": self.rag.snapshot_state(),
            "persistent_rag": persistent_snapshot,
        }

    def rollback(
        self, snapshot: MutationSnapshot, committed_components: set[str], *,
        persist_lattice: Callable[[], None] | None = None,
    ) -> None:
        """Restore runtime, then compensate only writes that committed or may have."""
        persistent = self._persistent_rag()
        self.runtime.import_state(copy.deepcopy(snapshot["dml"]))
        self.rag.restore_state(snapshot["rag"])
        persistent_snapshot = snapshot.get("persistent_rag")
        if persistent is not None and persistent_snapshot is not None:
            persistent.restore_state(persistent_snapshot)
        self._invalidate_cache()

        if "dml" in committed_components:
            if self.state.lattice_recovery_error is not None:
                raise self.state.lattice_recovery_error
            (persist_lattice or self.persist_lattice)()
        if "persistent_rag" in committed_components and persistent is not None:
            persistent.persist()
            self.state.observed_persistent_rag = self.path_stamp(persistent.manifest_path)
        if "rag" in committed_components:
            payload = {"documents": copy.deepcopy(snapshot["rag"].get("documents") or [])}
            self._write_text(self.rag_path, json.dumps(payload, indent=2))
            self.state.observed_rag = self.path_stamp(self.rag_path)

    def refresh(self, *, state_stamp: Callable[[], Stamp] | None = None,
                include_auxiliary: bool = True) -> bool:
        """Import changed components; caller handles reload logging and metrics."""
        persistent = self._persistent_rag()
        with self.state.refresh_lock:
            changed = False
            current = (state_stamp or self.lattice.stamp)()
            if current is None and self.state.observed_lattice is not None:
                raise FileNotFoundError("Previously initialized DML state is missing")
            if current is not None and current != self.state.observed_lattice:
                if self.lattice.journal:
                    payload, loaded_revision = self.lattice.load_with_revision()
                    assert loaded_revision is not None
                    current = (loaded_revision, current[1])
                else:
                    payload = self.lattice.load()
                self.runtime.import_state(payload)
                self._invalidate_cache()
                self.state.observed_lattice = current
                self.state.lattice_recovery_error = None
                self.clear_failure("receipt_runtime")
                changed = True

            if not include_auxiliary:
                return changed

            rag_stamp = self.path_stamp(self.rag_path)
            if rag_stamp is not None and rag_stamp != self.state.observed_rag:
                self.rag.import_state(json.loads(self.rag_path.read_text(encoding="utf-8")))
                self.state.observed_rag = rag_stamp
                changed = True

            if persistent is not None:
                persistent_stamp = self.path_stamp(persistent.manifest_path)
                if (persistent_stamp is not None
                        and persistent_stamp != self.state.observed_persistent_rag
                        and persistent.load()):
                    self.state.observed_persistent_rag = persistent_stamp
                    changed = True
            return changed

    def _reconcile_lattice_failure(
        self, *, before: object, payload: dict[str, Any] | None,
        expected_revision: int, error: BaseException,
    ) -> None:
        journal = self.lattice.journal
        if journal is None:
            self._reconcile_file_failure("dml", self.lattice.path, before)
            return
        # A rejected CAS is a definite non-commit, even if the peer's payload
        # happens to equal ours. It never authorizes replacing peer state.
        if isinstance(error, RevisionConflict):
            return
        try:
            revision, authority = journal.read_snapshot()
            if revision == expected_revision:
                return
            self._mark_committed("dml")
            if revision != expected_revision + 1 or authority != payload:
                raise RuntimeError("Lattice authority diverged after failed commit")
            previous = self.state.observed_lattice
            self.state.observed_lattice = (revision, previous[1] if previous else 0)
            self.state.lattice_recovery_error = None
        except BaseException as recovery_error:
            self._mark_committed("dml")
            self.state.lattice_recovery_error = recovery_error

    def persist_lattice(self) -> None:
        self._require_legacy()
        with self.state.persist_lock:
            attempted = False
            before: object = _UNKNOWN
            payload = None
            expected_revision = (
                self.state.observed_lattice[0]
                if self.lattice.journal and self.state.observed_lattice else 0
            )
            try:
                jsonl = self.lattice.use_jsonl and self.lattice.journal is None
                payload = None if jsonl else self.runtime.export_state()
                items = self.runtime.items() if jsonl else None
                if self.lattice.journal is None:
                    before = self._before_file_write(self.lattice.path)
                operation = self._operation()
                attempted = True
                stamp = self.lattice.commit(
                    payload=payload, items=items, expected_revision=expected_revision,
                    operation=operation,
                )
                self._mark_committed("dml")
                self.state.observed_lattice = stamp
                self.state.lattice_recovery_error = None
            except BaseException as exc:
                if attempted:
                    self._reconcile_lattice_failure(
                        before=before, payload=payload,
                        expected_revision=expected_revision, error=exc,
                    )
                self.record_failure("dml", exc)
                if isinstance(exc, Exception):
                    raise PersistenceCommitError("DML state persistence failed") from exc
                raise
            self.clear_failure("dml")

    def _write_rag_component(self, component: str, path: Path, write: Callable[[], Any]) -> Stamp:
        before = self._before_file_write(path)
        try:
            write()
            self._mark_committed(component)
            return self.path_stamp(path)
        except BaseException:
            self._reconcile_file_failure(component, path, before)
            raise

    def persist_rag(self) -> None:
        persistent = self._persistent_rag()
        with self.state.persist_lock:
            try:
                if persistent is not None:
                    self.state.observed_persistent_rag = self._write_rag_component(
                        "persistent_rag", persistent.manifest_path, persistent.persist,
                    )
                data = self.rag.export_state()
                self.rag_path.parent.mkdir(parents=True, exist_ok=True)
                self.state.observed_rag = self._write_rag_component(
                    "rag", self.rag_path,
                    lambda: self._write_text(self.rag_path, json.dumps(data, indent=2)),
                )
            except BaseException as exc:
                self.record_failure("rag", exc)
                LOGGER.exception("Failed to persist RAG state to %s", self.rag_path)
                if isinstance(exc, Exception):
                    raise PersistenceCommitError(f"RAG state persistence failed: {self.rag_path}") from exc
                raise
            self.clear_failure("rag")

    def persist_all(
        self, *, persist_lattice: Callable[[], None] | None = None,
        persist_rag: Callable[[], None] | None = None,
    ) -> None:
        (persist_lattice or self.persist_lattice)()
        (persist_rag or self.persist_rag)()

    def record_failure(self, component: str, error: BaseException) -> None:
        with self.state.persist_lock:
            self.state.failures[component] = f"{type(error).__name__}: {error}"

    def clear_failure(self, component: str) -> None:
        with self.state.persist_lock:
            self.state.failures.pop(component, None)

    def durability_status(self) -> dict[str, Any]:
        with self.state.persist_lock:
            failures = dict(self.state.failures)
        return {"status": "degraded" if failures else "ok", "failures": failures}
