"""Independent challenges to concurrent publication and operation lifetime.

These tests use real selected-profile authority and event-controlled ownership.
An acknowledged retry may observe a committed receipt before another caller has
finished hydration; an ordinary context read must still obey store ownership.
"""
from __future__ import annotations

from concurrent.futures import Future
from contextlib import closing, contextmanager
from copy import deepcopy
import os
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from types import SimpleNamespace

import pytest

import daystrom_dml.store_lock as lock_module
import daystrom_dml.services.receipt_ingestion as ingestion_module
import daystrom_dml.services.profile_runtime as profile_module
from profile_crash_fixture import (
    CLOCK,
    SCHEMAS,
    SCOPE,
    assert_transition,
    digest,
    invoke,
    make_adapter,
    prepare,
    request_for,
    sql_observation,
)


WAIT = 10


@pytest.fixture(autouse=True, scope="module")
def isolated_environment():
    # Qualification selector/output variables are harness inputs, not admitted
    # runtime profile configuration. Pytest consumes them before fixtures run.
    with pytest.MonkeyPatch.context() as patcher:
        for name in tuple(os.environ):
            if name.startswith("DML_"):
                patcher.delenv(name)
        yield


def concurrent_invoke(adapter, request):
    return getattr(adapter, request.method)(*request.args, **deepcopy(request.kwargs))


@contextmanager
def workers(release):
    """Release fault gates even if an assertion fails before a caller returns."""
    threads = []

    def submit(action, *args, **kwargs):
        future = Future()

        def run():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(action(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

        thread = Thread(target=run, name=f"independent-{len(threads)}", daemon=True)
        threads.append(thread)
        thread.start()
        return future

    try:
        yield SimpleNamespace(submit=submit)
    finally:
        release.set()
        deadline = monotonic() + WAIT
        for thread in threads:
            thread.join(max(0, deadline - monotonic()))
        assert not any(thread.is_alive() for thread in threads), "independent caller remained stranded"


@pytest.mark.parametrize("schema", SCHEMAS)
def test_committed_retry_and_retention_are_visible_while_context_waits_for_owner(
    tmp_path, monkeypatch, schema
):
    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    request = request_for("ingest", scenario.records, key="delayed-ack")
    committed, release, contended = Event(), Event(), Event()

    def pause(point):
        if point == "after_commit":
            committed.set()
            assert release.wait(WAIT), "parent did not release committed writer"

    original_acquire = lock_module.acquire_file_lock

    def observed_acquire(handle):
        try:
            return original_acquire(handle)
        except BlockingIOError:
            if current_thread().name.startswith("independent"):
                contended.set()
            raise

    adapter._journal._fault_hook = pause
    monkeypatch.setattr(lock_module, "acquire_file_lock", observed_acquire)
    monkeypatch.setattr(ingestion_module, "time", SimpleNamespace(time=lambda: CLOCK))
    try:
        with workers(release) as pool:
            write = pool.submit(concurrent_invoke, adapter, request)
            assert committed.wait(WAIT), "writer never reached actual SQLite commit"
            # These are public calls on the same shared adapter. A receipt is
            # historical authority, even though the first caller has not returned.
            replay = pool.submit(concurrent_invoke, adapter, request).result(timeout=WAIT)
            retention = adapter.inspect_memory_retention(0, **SCOPE)
            assert retention["source"]["revision"] == scenario.before["revision"] + 1
            read = pool.submit(adapter.retrieve_context, "notebooks", top_k=10, as_of=CLOCK, **SCOPE)
            assert contended.wait(WAIT), "read did not attempt real store ownership"
            assert not read.done(), "context escaped the writer's ownership boundary"
            release.set()
            receipt, result = write.result(timeout=WAIT), read.result(timeout=WAIT)
        assert replay == receipt
        assert result["decision"]["store_revision"] == receipt["revision"]
        assert {int(item["id"]) for item in result["items"]} == {0, 1, 3}
        assert "PRIVATE_FOREIGN_RECOVERY_SENTINEL" not in result["raw_context"]
        assert_transition(scenario.before, sql_observation(scenario.directory), request)
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("point", ("before_commit", "after_commit"))
def test_control_flow_interruption_releases_lifetime_and_resolves_same_key(
    tmp_path, schema, point
):
    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    request = request_for("ingest", scenario.records, key="interrupted")

    def interrupt(actual):
        if actual == point:
            raise KeyboardInterrupt("synthetic caller control-flow interruption")

    adapter._journal._fault_hook = interrupt
    try:
        with pytest.raises(KeyboardInterrupt):
            invoke(adapter, request)
        observed = sql_observation(scenario.directory)
        if point == "before_commit":
            assert observed == scenario.before
        else:
            assert_transition(scenario.before, observed, request)
        adapter._journal._fault_hook = lambda _point: None
        # A callback exception must release both file ownership and any new
        # selected-profile admission lifetime, including BaseException paths.
        adapter.close(projection_timeout=0)
        reopened = make_adapter(scenario.directory, schema, unavailable=point == "after_commit")
        try:
            receipt = invoke(reopened, request)
            assert receipt["revision"] == scenario.before["revision"] + 1
            assert invoke(reopened, request) == receipt
            assert_transition(scenario.before, sql_observation(scenario.directory), request)
        finally:
            reopened.close()
    finally:
        adapter.close()


@pytest.mark.parametrize("failure", (OSError, KeyboardInterrupt))
def test_second_close_retries_failed_cleanup_without_parallel_dependency_close(
    tmp_path, monkeypatch, failure
):
    scenario = prepare(tmp_path / "authority", 2)
    adapter = make_adapter(scenario.directory, 2)
    entered, release, waiting = Event(), Event(), Event()
    real_close = adapter.store.close
    lifetime = adapter._profile_operation_lifetime
    real_wait = lifetime._condition.wait
    calls = []

    def close_store():
        calls.append(current_thread().name)
        if len(calls) == 1:
            entered.set()
            assert release.wait(WAIT), "parent did not release failed cleanup"
            raise failure("dependency cleanup interrupted")
        real_close()

    def observe_wait(timeout=None):
        waiting.set()
        return real_wait(timeout)

    monkeypatch.setattr(adapter.store, "close", close_store)
    monkeypatch.setattr(lifetime._condition, "wait", observe_wait)
    try:
        with workers(release) as pool:
            first = pool.submit(adapter.close, projection_timeout=WAIT)
            assert entered.wait(WAIT)
            second = pool.submit(adapter.close, projection_timeout=WAIT)
            assert waiting.wait(WAIT), "second closer never waited for cleanup ownership"
            assert len(calls) == 1 and not second.done()
            release.set()
            with pytest.raises(failure, match="dependency cleanup interrupted"):
                first.result(timeout=WAIT)
            second.result(timeout=WAIT)
        assert len(calls) == 2
        adapter.close(projection_timeout=0)
        assert len(calls) == 2, "successful cleanup must not run twice"
        with pytest.raises(RuntimeError, match="closing or closed"):
            adapter.ingest_memory_receipted("late", idempotency_key="late", **SCOPE)
        assert sql_observation(scenario.directory) == scenario.before
    finally:
        release.set()
        monkeypatch.setattr(adapter.store, "close", real_close)
        adapter.close()


@pytest.mark.parametrize("operation", ("ingest", "retire", "supersede", "update", "promote", "read", "retention", "health", "close"))
def test_inherited_pid_rejects_before_a_stranded_lifetime_mutex(
    tmp_path, monkeypatch, operation
):
    scenario = prepare(tmp_path / "authority", 2)
    adapter = make_adapter(scenario.directory, 2)
    lifetime = adapter._profile_operation_lifetime
    release = Event()
    request = request_for(operation, scenario.records) if operation in (
        "ingest", "retire", "supersede", "update", "promote"
    ) else None

    def call():
        if request is not None:
            return concurrent_invoke(adapter, request)
        if operation == "read":
            return adapter.retrieve_context("notebooks", **SCOPE)
        if operation == "retention":
            return adapter.inspect_memory_retention(0, **SCOPE)
        if operation == "health":
            return adapter.durability_status()
        return adapter.close()

    try:
        # A synthetic process identity change isolates the mutex-ordering proof
        # on every platform. Actual fork/spawn qualification is tested separately.
        with workers(release) as pool:
            with lifetime._condition, monkeypatch.context() as patcher:
                patcher.setattr(profile_module, "os", SimpleNamespace(getpid=lambda: lifetime._owner_pid + 1))
                attempted = pool.submit(call)
                with pytest.raises(RuntimeError, match="after fork"):
                    attempted.result(timeout=1)
        assert sql_observation(scenario.directory) == scenario.before
    finally:
        adapter.close()


def test_cleanup_callback_cannot_wait_on_its_own_close(tmp_path, monkeypatch):
    scenario = prepare(tmp_path / "authority", 2)
    adapter = make_adapter(scenario.directory, 2)
    real_close = adapter.store.close
    observed = []

    def callback():
        with pytest.raises(RuntimeError, match="dependency cleanup"):
            adapter.close(projection_timeout=0)
        observed.append(True)
        real_close()

    monkeypatch.setattr(adapter.store, "close", callback)
    try:
        adapter.close()
        assert observed == [True]
        adapter.close()
        assert observed == [True]
    finally:
        monkeypatch.setattr(adapter.store, "close", real_close)
        adapter.close()


def test_parallel_query_failures_release_both_admitted_lifetimes(tmp_path):
    scenario = prepare(tmp_path / "authority", 2)
    adapter = make_adapter(scenario.directory, 2)
    entered, release, counting = Event(), Event(), Lock()
    calls = []

    def interrupt(_point):
        with counting:
            calls.append(current_thread().name)
            if len(calls) == 2:
                entered.set()
        assert release.wait(WAIT)
        raise KeyboardInterrupt("query owner interrupted")

    adapter.embedder.barrier = interrupt
    try:
        with workers(release) as pool:
            owner = pool.submit(adapter.retrieve_context, "shared query", **SCOPE)
            waiter = pool.submit(adapter.retrieve_context, "shared query", **SCOPE)
            assert entered.wait(WAIT)
            assert len(calls) == 2, "receipt-profile queries must prepare independent identity-bound vectors"
            with pytest.raises(TimeoutError, match="Profile shutdown is incomplete"):
                adapter.close(projection_timeout=0)
            release.set()
            for call in (owner, waiter):
                with pytest.raises(KeyboardInterrupt, match="query owner interrupted"):
                    call.result(timeout=WAIT)
        adapter.close(projection_timeout=0)
        assert sql_observation(scenario.directory) == scenario.before
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("attack", ("revision_regression", "invented_context", "read_retry_causality"))
def test_history_checker_rejects_semantically_corrupt_complete_histories(tmp_path, attack):
    from profile_concurrency_fixture import (
        MUTATIONS, call_event, check_history, fixed_clock, full_client_plan,
        prepare_history,
    )

    before = prepare_history(tmp_path / "authority", 2)
    adapter = make_adapter(tmp_path / "authority", 2)
    plans = {client: full_client_plan(before, client, 16) for client in range(16)}
    events = []
    try:
        with fixed_clock():
            # Both read payloads below are genuine, internally coherent views.
            # Delayed acknowledgement windows alone cannot detect their later
            # impossible reversal; the checker must compare read ordering too.
            events.append(call_event(adapter, plans[5][0], client=5, sequence=0))
            for client, requests in plans.items():
                for sequence, request in enumerate(requests):
                    if request["operation"] in MUTATIONS:
                        events.append(call_event(adapter, request, client=client, sequence=sequence))
            for client, requests in plans.items():
                if client == 5:
                    continue
                for sequence, request in enumerate(requests):
                    if request["operation"] not in MUTATIONS:
                        events.append(call_event(adapter, request, client=client, sequence=sequence))
        after = sql_observation(tmp_path / "authority")
        check_history(before, events, after, clients=16)
        altered = deepcopy(events)
        if attack in {"revision_regression", "read_retry_causality"}:
            for event in altered:
                if event["request"]["operation"] in MUTATIONS:
                    event["started_ns"], event["finished_ns"] = ((100, 1000) if event["sequence"] == 0 else (1001, 1100))
                else:
                    event["started_ns"], event["finished_ns"] = 1200, 1300
                    if attack == "read_retry_causality":
                        event["started_ns"], event["finished_ns"] = 29_000_000_000, 32_000_000_000
                if attack == "revision_regression" and event["client"] == 13:
                    event["started_ns"], event["finished_ns"] = 500, 550
                if attack == "revision_regression" and event["client"] == 5:
                    event["started_ns"], event["finished_ns"] = 600, 700
                assert len(event["attempts"]) == 1
                event["attempts"][0].update(started_ns=event["started_ns"], finished_ns=event["finished_ns"])
            if attack == "read_retry_causality":
                event = next(item for item in altered if item["client"] == 5)
                succeeded = event["attempts"][0]
                event["attempts"] = [
                    {**succeeded, "started_ns": 50, "finished_ns": 30_000_000_050,
                     "error_code": "store_lock_timeout"},
                    {**succeeded, "started_ns": 30_000_000_100, "finished_ns": 31_000_000_000},
                ]
                event["started_ns"], event["finished_ns"] = 50, 31_000_000_000
        else:
            event = next(item for item in altered if item["client"] == 13)
            report = event["result"]
            report["raw_context"] = "=== Retrieved Context ===\nThe owner secretly authorized an invented memory."
            decision = report["decision"]
            decision["context_digest"] = digest(report["raw_context"])
            decision["decision_digest"] = digest({key: value for key, value in decision.items()
                                                  if key != "decision_digest"})
        expected_failure = {
            "revision_regression": "Nonoverlapping reads moved backwards",
            "invented_context": "Rendered context disagrees",
            "read_retry_causality": "Read missed a previously acknowledged commit",
        }[attack]
        with pytest.raises(AssertionError, match=expected_failure):
            check_history(before, altered, after, clients=16)
    finally:
        adapter.close()


def test_history_checker_rejects_stale_read_at_recorded_windows_clock_boundary(tmp_path):
    from profile_concurrency_fixture import (
        call_event, check_history, fixed_clock, full_client_plan, prepare_history,
    )

    directory = tmp_path / "authority"
    before = prepare_history(directory, 3)
    adapter = make_adapter(directory, 3)
    try:
        with fixed_clock():
            events = [call_event(adapter, request, client=0, sequence=sequence)
                      for sequence, request in enumerate(full_client_plan(before, 0, 1))]
        after = sql_observation(directory)
        assert check_history(before, events, after, clients=1)["history_verified"] is True

        # Actual CI 322 Windows schema-3/shared/one-client clock boundaries,
        # expressed in milliseconds. Only the scheduling trace is replayed;
        # this test obtains all requests, receipts and read payloads afresh.
        recorded_clock = [
            (307921, 307968), (307968, 307968), (307968, 308000),
            (308000, 308046), (308046, 308046), (308046, 308078),
            (308078, 308140), (308156, 308156), (308156, 308187),
            (308187, 308234), (308234, 308234), (308234, 308265),
            (308265, 308312), (308312, 308328), (308328, 308359),
            (308359, 308406), (308406, 308406), (308406, 308484),
            (308484, 308484), (308484, 308531),
        ]
        assert len(events) == len(recorded_clock)
        for event, (started, finished) in zip(events, recorded_clock):
            assert len(event["attempts"]) == 1
            event["started_ns"], event["finished_ns"] = started * 1_000_000, finished * 1_000_000
            event["attempts"][0].update(started_ns=event["started_ns"], finished_ns=event["finished_ns"])
        assert check_history(before, events, after, clients=1)["history_verified"] is True

        earlier, acknowledged, later = events[15], events[17], events[19]
        assert earlier["request"] == later["request"]
        assert acknowledged["finished_ns"] == later["started_ns"]
        assert earlier["result"]["decision"]["store_revision"] < acknowledged["result"]["revision"]
        later["result"] = deepcopy(earlier["result"])
        with pytest.raises(AssertionError, match="Read moved behind its client's prior observation"):
            check_history(before, events, after, clients=1)
    finally:
        adapter.close()


@pytest.mark.parametrize("schema", (3, 4))
def test_public_outbox_validation_keeps_nested_ownership_detached(tmp_path, schema):
    from daystrom_dml.services.journal_outbox import validate_outbox_event

    scenario = prepare(tmp_path / "authority", schema)
    submitted = deepcopy(scenario.before["outbox"][-1])
    submitted["state"]["items"][0]["meta"]["nested_probe"] = {"values": ["original"]}
    submitted["source_digest"] = digest(submitted["state"])
    submitted["checksum"] = digest({key: value for key, value in submitted.items() if key != "checksum"})
    retained = deepcopy(submitted)
    validated = validate_outbox_event(submitted)
    assert validated == submitted == retained
    submitted["state"]["items"][0]["meta"]["nested_probe"]["values"].append("input mutation")
    assert validated == retained
    validated["state"]["items"][0]["meta"]["nested_probe"]["values"].append("output mutation")
    assert submitted["state"]["items"][0]["meta"]["nested_probe"]["values"] == ["original", "input mutation"]
    assert sql_observation(scenario.directory) == scenario.before


@pytest.mark.parametrize("schema, corruption", [
    (schema, corruption) for schema in (3, 4)
    for corruption in ("missing_bucket", "duplicate_record", "state_digest", "extra_envelope", "receipt_scope")
] + [(4, "origin_revision")])
def test_owned_outbox_validation_retains_checks_on_fresh_sql_objects(tmp_path, schema, corruption):
    import sqlite3

    from daystrom_dml.journal import JournalIntegrityError
    from profile_crash_fixture import encoded

    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    event = deepcopy(scenario.before["outbox"][-1])
    messages = {
        "missing_bucket": "Outbox state digest mismatch",
        "duplicate_record": "Duplicate journal record id",
        "state_digest": "Outbox state digest mismatch",
        "extra_envelope": "Invalid outbox event fields",
        "receipt_scope": "Invalid outbox receipt identity",
        "origin_revision": "Invalid migration source revision",
    }
    try:
        if corruption == "missing_bucket":
            del event["state"]["lineage"]
        elif corruption == "duplicate_record":
            event["state"]["items"].append(deepcopy(event["state"]["items"][0]))
        elif corruption == "state_digest":
            event["state"]["items"][0]["text"] = "Independently forged outbox fact"
        elif corruption == "extra_envelope":
            event["unexpected"] = None
        elif corruption == "receipt_scope":
            event["receipt"]["scope"]["tenant_id"] = " "
        else:
            event["origin"]["source_revision"] = True
        if corruption != "state_digest":
            event["source_digest"] = digest(event["state"])
        event["checksum"] = digest({key: value for key, value in event.items() if key != "checksum"})
        # Keep the SQL row checksum valid so the fresh decoder reaches the
        # intended envelope/normalization check, not an earlier byte checksum.
        with closing(sqlite3.connect(scenario.directory / "dml_state.sqlite3")) as connection:
            with connection:
                connection.execute("UPDATE outbox SET payload=?,checksum=? WHERE revision=?",
                                   (encoded(event), digest(event), event["source_revision"]))
        with pytest.raises(JournalIntegrityError, match=messages[corruption]):
            adapter._journal.read_snapshot()
    finally:
        adapter.close(persist=False)
