"""Nestable ownership and rollback for legacy durable mutations.

The coordinator owns transaction state, but knows nothing about storage formats
or adapter objects. Callbacks supply ownership, snapshots, and compensation.
Rollback also runs for Python control-flow exceptions; abrupt process death
still requires the backing store's own recovery guarantees.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from typing import Callable, ContextManager, Generic, Iterator, TypeVar


Snapshot = TypeVar("Snapshot")
_MISSING = object()


class PersistenceCommitError(RuntimeError):
    """Raised when a configured durability write cannot confirm success."""


class PersistenceRollbackError(RuntimeError):
    """Raised when a failed mutation cannot restore its pre-mutation state."""

    def __init__(
        self, original_error: BaseException, rollback_error: BaseException
    ) -> None:
        super().__init__(
            f"Persistence failed and rollback also failed: {type(rollback_error).__name__}"
        )
        self.original_error = original_error
        self.rollback_error = rollback_error


class TransactionCoordinator(Generic[Snapshot]):
    """Coordinate ownership and compensation through explicit capabilities.

    ``acquire_ownership`` must serialize all cooperating writers until its
    context exits. ``refresh`` runs after ownership is acquired and before any
    snapshot or mutation. ``record_rollback_failure`` handles its own locking.
    ``local`` is exposed solely for compatibility with legacy nesting checks.
    A forked child must construct a fresh adapter: inherited runtime and locks
    cannot safely be reused even if thread-local nesting were reset.
    """

    def __init__(
        self,
        *,
        acquire_ownership: Callable[[str], ContextManager[object]],
        refresh: Callable[[], object],
        capture: Callable[[], Snapshot],
        rollback: Callable[[Snapshot, set[str]], None],
        record_rollback_failure: Callable[[BaseException], None],
    ) -> None:
        self._acquire_ownership = acquire_ownership
        self._refresh = refresh
        self._capture = capture
        self._rollback = rollback
        self._record_rollback_failure = record_rollback_failure
        self.local = threading.local()
        self._owner_pid = os.getpid()

    def _require_owner_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Inherited transaction coordinator cannot be used after fork; "
                "create a fresh adapter in the child process"
            )

    @contextmanager
    def transaction(self, operation: str) -> Iterator[None]:
        """Own and refresh a transaction; nested calls reuse outer ownership."""

        self._require_owner_process()
        depth = int(getattr(self.local, "depth", 0))
        if depth:
            self.local.depth = depth + 1
            try:
                yield
            finally:
                self.local.depth = depth
            return

        with self._acquire_ownership(operation):
            previous_operation = getattr(self.local, "operation", _MISSING)
            self.local.depth = 1
            self.local.operation = operation
            try:
                self._refresh()
                yield
            finally:
                self.local.depth = 0
                if previous_operation is _MISSING:
                    del self.local.operation
                else:
                    self.local.operation = previous_operation

    @contextmanager
    def mutation(
        self,
        operation: str,
        *,
        ownership: Callable[[], ContextManager[object]] | None = None,
        persist: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        """Run a rollback-capable mutation with optional persistence on exit.

        ``ownership`` lets compatibility wrappers retain their transaction
        interception point; it must delegate to ``transaction``. A batch passes
        ``persist`` so its final write participates in the same rollback frame.
        """

        self._require_owner_process()
        scope = self.transaction(operation) if ownership is None else ownership()
        with scope:
            snapshot = self._capture()
            previous_commits = getattr(self.local, "committed_components", None)
            committed_components: set[str] = set()
            self.local.committed_components = committed_components
            try:
                yield
                if persist is not None:
                    persist()
            except BaseException as original_error:
                # Compensating writes must not become new mutation commits.
                self.local.committed_components = None
                try:
                    self._rollback(snapshot, committed_components)
                except BaseException as rollback_error:
                    root_rollback_error = (
                        rollback_error.__cause__
                        if isinstance(rollback_error, PersistenceCommitError)
                        and isinstance(rollback_error.__cause__, BaseException)
                        else rollback_error
                    )
                    self._record_rollback_failure(root_rollback_error)
                    raise PersistenceRollbackError(
                        original_error, root_rollback_error
                    ) from original_error
                raise
            finally:
                # Even successful compensation publishes the inner snapshot,
                # which may contain changes from its uncommitted parent. Keep
                # every touched component so an outer rollback can restore its
                # own snapshot instead of leaving those parent changes durable.
                if previous_commits is not None:
                    previous_commits.update(committed_components)
                self.local.committed_components = previous_commits

    def mark_committed(self, component: str) -> None:
        """Track a durable write in this thread's current mutation frame."""

        self._require_owner_process()
        committed = getattr(self.local, "committed_components", None)
        if committed is not None:
            committed.add(component)
