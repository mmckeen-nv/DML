"""Ownership, savepoints, and failure boundaries without an adapter fixture."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import pytest

import daystrom_dml.services.transactions as transaction_module
from daystrom_dml.services.transactions import (
    PersistenceCommitError,
    PersistenceRollbackError,
    TransactionCoordinator,
)


class TransactionHarness:
    """Small mutable resource with visible durable writes and compensation."""

    def __init__(self):
        self.events = []
        self.runtime = []
        self.durable = {"dml": [], "rag": []}
        self.owned = False
        self.coordinator = TransactionCoordinator(
            acquire_ownership=self.ownership,
            refresh=self.refresh,
            capture=self.capture,
            rollback=self.rollback,
            record_rollback_failure=self.record_failure,
        )

    @contextmanager
    def ownership(self, operation):
        assert not self.owned
        self.events.append(("acquire", operation))
        self.owned = True
        try:
            yield
        finally:
            self.events.append(("release", operation))
            self.owned = False

    def refresh(self):
        assert self.owned
        self.events.append("refresh")

    def capture(self):
        assert self.owned
        self.events.append("capture")
        return list(self.runtime)

    def rollback(self, snapshot, components):
        assert self.owned
        assert self.coordinator.local.committed_components is None
        self.events.append(("rollback", set(components)))
        self.runtime[:] = snapshot
        for component in components:
            self.durable[component] = list(snapshot)
        # A compensation callback may share the regular persistence entry point.
        self.coordinator.mark_committed("compensation")

    def record_failure(self, failure):
        assert self.owned
        self.events.append(("rollback-failure", failure))

    def persist(self, component):
        assert self.owned
        self.events.append(("persist", component))
        self.durable[component] = list(self.runtime)
        self.coordinator.mark_committed(component)


def test_outer_refresh_precedes_capture_body_and_final_persistence():
    target = TransactionHarness()
    coordinator = target.coordinator
    with coordinator.mutation("batch", persist=lambda: target.persist("dml")):
        target.events.append("body")
        target.runtime.append("memory")
    assert target.durable["dml"] == ["memory"]
    assert target.events == [
        ("acquire", "batch"), "refresh", "capture", "body",
        ("persist", "dml"), ("release", "batch"),
    ]
    assert coordinator.local.depth == 0
    assert not hasattr(coordinator.local, "operation")
    assert coordinator.local.committed_components is None


def test_nested_transaction_reuses_ownership_and_restores_depth_after_failure():
    target = TransactionHarness()
    coordinator = target.coordinator
    coordinator.local.operation = "previous"
    with coordinator.transaction("outer"):
        assert coordinator.local.depth == 1
        with pytest.raises(ValueError, match="nested"):
            with coordinator.transaction("inner"):
                assert coordinator.local.depth == 2
                assert coordinator.local.operation == "outer"
                raise ValueError("nested")
        assert coordinator.local.depth == 1
        assert coordinator.local.operation == "outer"
    assert coordinator.local.operation == "previous"
    assert coordinator.local.depth == 0
    assert target.events == [("acquire", "outer"), "refresh", ("release", "outer")]


def test_failed_refresh_releases_ownership_without_snapshot_or_body():
    target = TransactionHarness()
    failure = RuntimeError("external snapshot unavailable")

    def refresh():
        assert target.owned
        raise failure

    target.coordinator._refresh = refresh
    with pytest.raises(RuntimeError) as caught:
        with target.coordinator.mutation("refresh-failure"):
            pytest.fail("Mutation ran after failed refresh")
    assert caught.value is failure
    assert target.events == [
        ("acquire", "refresh-failure"), ("release", "refresh-failure"),
    ]
    assert target.coordinator.local.depth == 0
    assert not hasattr(target.coordinator.local, "operation")


def test_failed_snapshot_does_not_attempt_compensation():
    target = TransactionHarness()
    failure = ValueError("cannot snapshot")

    def capture():
        raise failure

    target.coordinator._capture = capture
    with pytest.raises(ValueError) as caught:
        with target.coordinator.mutation("capture-failure"):
            pytest.fail("Mutation ran without snapshot")
    assert caught.value is failure
    assert target.events == [
        ("acquire", "capture-failure"), "refresh", ("release", "capture-failure"),
    ]
    assert target.coordinator.local.depth == 0


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_body_failure_rolls_back_runtime_and_durable_writes(failure_type):
    target = TransactionHarness()
    failure = failure_type("mutation interrupted")
    with pytest.raises(failure_type) as caught:
        with target.coordinator.mutation("failed"):
            target.runtime.append("transient")
            target.persist("dml")
            raise failure
    assert caught.value is failure
    assert target.runtime == target.durable["dml"] == []
    assert target.events[-2:] == [("rollback", {"dml"}), ("release", "failed")]
    assert target.coordinator.local.depth == 0
    assert target.coordinator.local.committed_components is None
    assert not hasattr(target.coordinator.local, "operation")


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt, SystemExit])
def test_final_persist_failure_rolls_back_completed_components(failure_type):
    target = TransactionHarness()
    failure = failure_type("second component failed")

    def persist():
        target.persist("dml")
        raise failure

    with pytest.raises(failure_type) as caught:
        with target.coordinator.mutation("batch", persist=persist):
            target.runtime.append("batch mutation")
    assert caught.value is failure
    assert target.runtime == target.durable["dml"] == []
    assert target.events[-3:] == [
        ("persist", "dml"), ("rollback", {"dml"}), ("release", "batch"),
    ]


def test_failed_batch_body_never_runs_final_persistence():
    target = TransactionHarness()
    with pytest.raises(ValueError, match="body failure"):
        with target.coordinator.mutation("batch", persist=lambda: target.persist("dml")):
            target.runtime.append("transient")
            raise ValueError("body failure")
    assert target.runtime == []
    assert not any(event == ("persist", "dml") for event in target.events)


def test_successful_nested_writes_merge_for_outer_compensation():
    target = TransactionHarness()
    coordinator = target.coordinator
    with pytest.raises(ValueError, match="outer failure"):
        with coordinator.mutation("outer"):
            target.runtime.append("outer")
            target.persist("dml")
            outer_commits = coordinator.local.committed_components
            with coordinator.mutation("inner", persist=lambda: target.persist("rag")):
                target.runtime.append("inner")
                assert coordinator.local.committed_components is not outer_commits
            assert coordinator.local.committed_components is outer_commits
            assert outer_commits == {"dml", "rag"}
            raise ValueError("outer failure")
    assert target.runtime == target.durable["dml"] == target.durable["rag"] == []
    assert target.events.count("refresh") == 1
    assert target.events[-2] == ("rollback", {"dml", "rag"})


def test_caught_inner_failure_restores_savepoint_and_retains_touched_components():
    target = TransactionHarness()
    coordinator = target.coordinator
    with coordinator.mutation("outer", persist=lambda: target.persist("dml")):
        target.runtime.append("outer")
        outer_commits = coordinator.local.committed_components
        with pytest.raises(ValueError, match="inner failure"):
            with coordinator.mutation("inner"):
                target.runtime.append("inner")
                target.persist("rag")
                raise ValueError("inner failure")
        assert target.runtime == ["outer"]
        assert coordinator.local.committed_components is outer_commits
        assert outer_commits == {"rag"}
        assert coordinator.local.depth == 1
    assert target.durable["dml"] == target.durable["rag"] == ["outer"]
    assert coordinator.local.committed_components is None


@pytest.mark.parametrize("rollback_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("wrapped", [False, True])
def test_failed_compensation_records_root_error_and_preserves_original(rollback_type, wrapped):
    target = TransactionHarness()
    original = KeyboardInterrupt("interrupted mutation")
    root = rollback_type("compensation failed")

    def rollback(snapshot, components):
        assert target.owned
        assert target.coordinator.local.committed_components is None
        if wrapped:
            raise PersistenceCommitError("compensation write failed") from root
        raise root

    target.coordinator._rollback = rollback
    with pytest.raises(PersistenceRollbackError) as caught:
        with target.coordinator.mutation("failed-compensation"):
            target.persist("dml")
            raise original
    assert caught.value.original_error is original
    assert caught.value.rollback_error is root
    assert caught.value.__cause__ is original
    assert target.events[-2:] == [
        ("rollback-failure", root), ("release", "failed-compensation"),
    ]
    assert target.coordinator.local.committed_components is None
    assert target.coordinator.local.depth == 0


def test_inner_compensation_failure_retains_writes_for_outer_compensation():
    target = TransactionHarness()
    coordinator = target.coordinator
    rollback = target.rollback
    original = ValueError("inner failed")
    compensation_failure = OSError("inner compensation failed")
    snapshots = []

    def fail_inner(snapshot, components):
        snapshots.append((list(snapshot), set(components)))
        if len(snapshots) == 1:
            raise compensation_failure
        rollback(snapshot, components)

    coordinator._rollback = fail_inner
    with pytest.raises(PersistenceRollbackError) as caught:
        with coordinator.mutation("outer"):
            target.runtime.append("outer")
            target.persist("dml")
            with coordinator.mutation("inner"):
                target.runtime.append("inner")
                target.persist("rag")
                raise original
    assert caught.value.original_error is original
    assert caught.value.rollback_error is compensation_failure
    assert snapshots == [(["outer"], {"rag"}), ([], {"dml", "rag"})]
    assert target.runtime == target.durable["dml"] == target.durable["rag"] == []


@pytest.mark.parametrize("catch_inner", [False, True])
def test_inner_compensation_does_not_publish_failed_outer_mutation(catch_inner):
    target = TransactionHarness()
    coordinator = target.coordinator
    original = ValueError("inner failed")
    with pytest.raises(ValueError) as caught:
        with coordinator.mutation("outer"):
            target.runtime.append("uncommitted outer change")
            try:
                with coordinator.mutation("inner"):
                    target.runtime.append("inner change")
                    target.persist("rag")
                    raise original
            except ValueError:
                if not catch_inner:
                    raise
            raise original
    assert caught.value is original
    assert target.runtime == target.durable["rag"] == []


def test_explicit_ownership_hook_keeps_refresh_and_rollback_inside_the_hook():
    target = TransactionHarness()
    coordinator = target.coordinator

    @contextmanager
    def intercepted():
        target.events.append("hook-before")
        try:
            with coordinator.transaction("intercepted"):
                yield
        finally:
            target.events.append("hook-after")

    with pytest.raises(ValueError):
        with coordinator.mutation("public-name", ownership=intercepted):
            raise ValueError("failure")
    assert target.events == [
        "hook-before", ("acquire", "intercepted"), "refresh", "capture",
        ("rollback", set()), ("release", "intercepted"), "hook-after",
    ]


def test_marking_outside_a_mutation_does_not_leave_pending_commits():
    target = TransactionHarness()
    coordinator = target.coordinator
    coordinator.mark_committed("outside")
    with coordinator.transaction("ownership-only"):
        coordinator.mark_committed("owned-but-not-a-mutation")
    with coordinator.mutation("next"):
        assert coordinator.local.committed_components == set()
    coordinator.mark_committed("after")
    assert coordinator.local.committed_components is None


def test_nesting_operation_and_commit_frames_are_thread_local():
    barrier = threading.Barrier(2)
    events = []
    events_lock = threading.Lock()

    def record(event):
        with events_lock:
            events.append((threading.current_thread().name, event))

    @contextmanager
    def ownership(operation):
        record(("acquire", operation))
        try:
            yield
        finally:
            record(("release", operation))

    coordinator = TransactionCoordinator(
        acquire_ownership=ownership,
        refresh=lambda: record("refresh"),
        capture=lambda: [],
        rollback=lambda snapshot, components: record(("rollback", set(components))),
        record_rollback_failure=lambda failure: pytest.fail(str(failure)),
    )

    def mutate(name, fail):
        try:
            with coordinator.mutation(name):
                coordinator.mark_committed(name)
                with coordinator.transaction("nested"):
                    barrier.wait(timeout=5)
                    assert coordinator.local.depth == 2
                    assert coordinator.local.operation == name
                    assert coordinator.local.committed_components == {name}
                    barrier.wait(timeout=5)
                if fail:
                    raise ValueError(name)
        except ValueError:
            assert fail
        assert coordinator.local.depth == 0
        assert coordinator.local.committed_components is None
        assert not hasattr(coordinator.local, "operation")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(mutate, "first", True), pool.submit(mutate, "second", False)]
        for future in futures:
            future.result(timeout=10)
    acquired = [event for _, event in events if isinstance(event, tuple) and event[0] == "acquire"]
    assert sorted(acquired) == [("acquire", "first"), ("acquire", "second")]
    assert sum(event == "refresh" for _, event in events) == 2
    assert [event for _, event in events if isinstance(event, tuple) and event[0] == "rollback"] == [
        ("rollback", {"first"}),
    ]
    assert not hasattr(coordinator.local, "depth")
    assert not hasattr(coordinator.local, "committed_components")


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("entry", ["transaction", "mutation", "ownership-hook", "mark"])
def test_inherited_coordinator_refuses_foreign_process_before_callbacks(monkeypatch, depth, entry):
    target = TransactionHarness()
    coordinator = target.coordinator
    coordinator.local.depth = depth
    coordinator.local.operation = "parent"
    parent_components = {"dml"}
    coordinator.local.committed_components = parent_components
    child_pid = transaction_module.os.getpid() + 1
    monkeypatch.setattr(transaction_module.os, "getpid", lambda: child_pid)

    def unexpected_ownership():
        pytest.fail("A forked coordinator called the ownership hook")

    with pytest.raises(RuntimeError, match="create a fresh adapter in the child process"):
        if entry == "mark":
            coordinator.mark_committed("child-write")
        else:
            if entry == "transaction":
                scope = coordinator.transaction("child")
            elif entry == "mutation":
                scope = coordinator.mutation("child", persist=lambda: target.persist("rag"))
            else:
                scope = coordinator.mutation("child", ownership=unexpected_ownership)
            with scope:
                pytest.fail("A forked coordinator entered the mutation body")

    assert target.events == []
    assert coordinator.local.depth == depth
    assert coordinator.local.operation == "parent"
    assert coordinator.local.committed_components is parent_components
    assert parent_components == {"dml"}


def test_fresh_coordinator_in_child_process_gets_its_own_ownership(monkeypatch):
    parent = TransactionHarness()
    child_pid = transaction_module.os.getpid() + 1
    monkeypatch.setattr(transaction_module.os, "getpid", lambda: child_pid)
    child = TransactionHarness()
    with child.coordinator.mutation("child", persist=lambda: child.persist("dml")):
        child.runtime.append("child memory")
    assert child.durable["dml"] == ["child memory"]
    assert child.events == [
        ("acquire", "child"), "refresh", "capture", ("persist", "dml"), ("release", "child"),
    ]
    assert parent.events == []
