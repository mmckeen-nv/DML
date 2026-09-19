"""Measured live callers and independently checked selected-profile histories."""
from __future__ import annotations

from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
from queue import Queue
import threading
from time import monotonic, monotonic_ns, sleep
from types import SimpleNamespace

import pytest

from profile_concurrency_fixture import (
    PROGRESS_SECONDS, WAIT_SECONDS, _assert_retention, _assert_retrieval,
    call_event, check_history, client_plan, digest, fixed_clock,
    full_client_plan, prepare_history, requested_scales, write_evidence, write_history,
)
from profile_crash_fixture import Request, assert_transition, make_adapter, sql_observation


@pytest.fixture(autouse=True, scope="module")
def isolated_environment():
    # Test selection variables are not production configuration fields. Scale
    # parametrization is resolved during collection, before this fixture runs.
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key in list(os.environ):
            if key.startswith("DML_"):
                monkeypatch.delenv(key)
        yield


def _client(adapter, before, client, clients, ready, release, results, stop, deadline):
    try:
        ready.put(client)
        if not release.wait(WAIT_SECONDS):
            raise AssertionError("Caller release gate timed out")
        events = []
        for sequence, request in enumerate(full_client_plan(before, client, clients)):
            if stop.is_set():
                return
            event = call_event(adapter, request, client=client, sequence=sequence, deadline_ns=deadline.value)
            events.append(event)
            results.put({"kind": "progress", "finished_ns": event["finished_ns"]})
        results.put({"kind": "done", "client": client, "events": events, "pid": os.getpid()})
    except BaseException as exc:
        results.put({"kind": "error", "client": client, "error": f"{type(exc).__name__}: {exc}"})


def _process_worker(directory, schema, before, indices, clients, ready, release, results, stop, deadline):
    try:
        with fixed_clock():
            adapter = make_adapter(directory, schema)
            try:
                workers = [threading.Thread(target=_client, args=(adapter, before, client, clients,
                           ready, release, results, stop, deadline), daemon=True) for client in indices]
                for worker in workers:
                    worker.start()
                cleanup_deadline = monotonic() + WAIT_SECONDS + 30
                for worker in workers:
                    worker.join(max(0, cleanup_deadline - monotonic()))
                assert not any(worker.is_alive() for worker in workers), "Process retained an active caller"
            finally:
                adapter.close()
    except BaseException as exc:
        results.put({"kind": "error", "error": f"Process {type(exc).__name__}: {exc}"})


def run_campaign(directory, schema, clients, transport):
    before = prepare_history(directory, schema)
    processes = []
    workers = []
    adapters = []
    if transport == "processes":
        context = multiprocessing.get_context("spawn")
        ready, results = context.Queue(), context.Queue()
        release, stop = context.Event(), context.Event()
        operation_deadline = context.Value("q", 0)
        count = min(clients, 4)
        for index in range(count):
            process = context.Process(target=_process_worker, args=(
                directory, schema, before, list(range(index, clients, count)), clients,
                ready, release, results, stop, operation_deadline))
            processes.append(process)
    else:
        ready, results = Queue(), Queue()
        release, stop = threading.Event(), threading.Event()
        operation_deadline = SimpleNamespace(value=0)
        adapters = [make_adapter(directory, schema) for _ in range(clients if transport == "threads-separate" else 1)]
        for client in range(clients):
            adapter = adapters[client] if transport == "threads-separate" else adapters[0]
            workers.append(threading.Thread(target=_client, args=(adapter, before, client, clients,
                           ready, release, results, stop, operation_deadline), daemon=True, name=f"profile-client-{client}"))
    events = []
    pids = set()
    first_progress = None
    with fixed_clock():
        try:
            for worker in [*processes, *workers]:
                worker.start()
            startup_deadline = monotonic() + WAIT_SECONDS
            arrived = set()
            while len(arrived) < clients:
                remaining = startup_deadline - monotonic()
                assert remaining > 0, "Caller startup exceeded its bounded deadline"
                client = ready.get(timeout=remaining)
                assert client not in arrived, "A caller announced readiness twice"
                arrived.add(client)
            assert arrived == set(range(clients)), "Ready gate did not contain every real caller"
            released_ns = monotonic_ns()
            operation_deadline.value = released_ns + WAIT_SECONDS * 1_000_000_000
            release.set()
            deadline = monotonic() + WAIT_SECONDS
            completed = set()
            while len(completed) < clients:
                remaining = deadline - monotonic()
                assert remaining > 0, "Campaign exceeded its 90 second bound"
                message = results.get(timeout=min(remaining, PROGRESS_SECONDS) if first_progress is None else remaining)
                assert message["kind"] != "error", message.get("error")
                if message["kind"] == "progress":
                    progress = (message["finished_ns"] - released_ns) / 1e9
                    first_progress = progress if first_progress is None else min(first_progress, progress)
                elif message["kind"] == "done":
                    assert message["client"] not in completed
                    completed.add(message["client"])
                    pids.add(message["pid"])
                    events.extend(message["events"])
            duration = (monotonic_ns() - released_ns) / 1e9
            assert first_progress is not None and first_progress <= PROGRESS_SECONDS
            assert duration <= WAIT_SECONDS
        finally:
            stop.set()
            release.set()
            cleanup_deadline = monotonic() + 35
            for worker in workers:
                if worker.ident is not None:
                    worker.join(max(0, cleanup_deadline - monotonic()))
            for process in processes:
                if process.pid is None:
                    continue
                process.join(max(0, cleanup_deadline - monotonic()))
                if process.is_alive():
                    process.kill()
                    process.join(5)
            assert not any(worker.is_alive() for worker in workers), "Caller failed to release ownership"
            for adapter in adapters:
                adapter.close()
            if transport == "processes":
                for channel in (ready, results):
                    channel.close()
                    channel.join_thread()
    assert all(process.exitcode == 0 for process in processes)
    assert len(pids) == (min(clients, 4) if processes else 1)
    after = sql_observation(directory)
    checked = check_history(before, events, after, clients=clients)
    reopened = make_adapter(directory, schema)
    try:
        assert sql_observation(directory) == after
        # A fresh adapter must render the same observed revision and scoped state.
        request = full_client_plan(before, 5 if clients > 5 else 0, clients)[-1]
        if request["operation"] == "retrieve":
            response = call_event(reopened, request, client=0, sequence=0)
            assert response["result"]["decision"]["store_revision"] == after["revision"]
    finally:
        reopened.close()
    evidence = {"schema_version": "dml-concurrency-evidence-v1", "schema": schema,
                "transport": transport, "clients": clients, "measured_ready_clients": len(arrived),
                "process_count": len(pids), "committed_revisions": checked.pop("unique_commits"),
                "duration_seconds": duration, "first_progress_seconds": first_progress, **checked}
    return before, events, after, evidence


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("clients", requested_scales())
@pytest.mark.parametrize("transport", ["threads-shared", "threads-separate", "processes"])
def test_profile_mixed_history(tmp_path, record_property, schema, clients, transport):
    before, events, after, evidence = run_campaign(tmp_path / "authority", schema, clients, transport)
    evidence.update(write_history(before, events, after, clients=clients, transport=transport))
    write_evidence(record_property, evidence)


@pytest.fixture(scope="module")
def accepted_history(tmp_path_factory):
    return run_campaign(tmp_path_factory.mktemp("oracle") / "authority", 3, 1, "threads-shared")


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "receipt", "revision", "dirty-read",
                                      "scope-leak", "retention", "request", "realtime"])
def test_history_oracle_rejects_corruption(accepted_history, corruption):
    before, events, after, _ = deepcopy(accepted_history)
    mutations = [event for event in events if event["request"]["operation"] == "ingest"]
    read = next(event for event in events if event["request"]["operation"] == "retrieve" and event["result"]["items"])
    if corruption == "missing":
        events.pop()
    elif corruption == "duplicate":
        events[-1] = deepcopy(events[0])
    elif corruption == "receipt":
        mutations[0]["result"]["result"]["memory"]["text"] = "Invented memory"
    elif corruption == "revision":
        mutations[0]["result"]["revision"] += 100
    elif corruption == "dirty-read":
        read["result"]["items"][0]["text"] = "Uncommitted dirty value"
    elif corruption == "scope-leak":
        read["result"]["items"][0]["meta"]["tenant_id"] = "foreign"
    elif corruption == "retention":
        next(event for event in events if event["request"]["operation"] == "retention")["result"]["source"]["state_digest"] = "0" * 64
    elif corruption == "request":
        mutations[0]["request"]["kwargs"]["idempotency_key"] = "unsubmitted"
    else:
        read["result"]["decision"]["store_revision"] = before["revision"]
    with pytest.raises(AssertionError):
        check_history(before, events, after, clients=1)


@pytest.mark.parametrize("corruption", ["erased", "retyped", "request", "overlap", "too-many", "deadline"])
def test_history_oracle_rejects_invalid_retry_attempts(accepted_history, corruption):
    before, events, after, _ = deepcopy(accepted_history)
    event = events[0]
    rejected = {"started_ns": event["started_ns"] - 2, "finished_ns": event["started_ns"] - 1,
                "request_digest": digest(event["request"]), "error_code": "store_lock_timeout"}
    event["started_ns"] = rejected["started_ns"]
    event["attempts"].insert(0, rejected)
    # This synthetic history exercises the checker only; it is not claimed as
    # an actual thirty-second OS timeout (qualified independently elsewhere).
    assert check_history(before, events, after, clients=1)["ownership_rejections"] == 1
    if corruption == "erased":
        event["attempts"].pop(0)
    elif corruption == "retyped":
        rejected["error_code"] = "embedding_failure"
    elif corruption == "request":
        rejected["request_digest"] = "0" * 64
    elif corruption == "overlap":
        rejected["finished_ns"] = event["attempts"][1]["started_ns"] + 1
    elif corruption == "too-many":
        event["attempts"] = [deepcopy(rejected)] * 3 + [event["attempts"][-1]]
    else:
        rejected["started_ns"] -= 91_000_000_000
        event["started_ns"] = rejected["started_ns"]
    with pytest.raises(AssertionError):
        check_history(before, events, after, clients=1)


@pytest.mark.parametrize("detail", [
    {"code": "retrieval_outcome_unavailable"},
    {"code": "retrieval_outcome_unavailable", "reason": "backend_timeout"},
    {"code": "retrieval_outcome_unavailable", "reason": True},
    {"code": "retrieval_outcome_unavailable", "reason": 1},
    {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout", "retry_same_key": True},
    {"code": "receipt_ownership_unavailable", "retry_same_key": True},
    {"code": "retrieval_ownership_unavailable", "reason": "store_ownership_timeout"},
    ["retrieval_outcome_unavailable", "store_ownership_timeout"],
])
def test_history_oracle_rejects_unclassified_http_read_retries(accepted_history, detail):
    before, events, after, _ = deepcopy(accepted_history)
    event = next(event for event in events if event["request"]["operation"] == "retrieve")
    rejected = {"started_ns": event["started_ns"] - 2, "finished_ns": event["started_ns"] - 1,
                "request_digest": digest(event["request"]), "error_code": "store_lock_timeout", "status_code": 503,
                "error_detail": {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}}
    event["started_ns"] = rejected["started_ns"]
    event["attempts"][-1]["status_code"] = 200
    event["attempts"].insert(0, rejected)
    assert check_history(before, events, after, clients=1)["ownership_rejections"] == 1
    rejected["error_detail"] = detail
    with pytest.raises(AssertionError, match="not positively identified"):
        check_history(before, events, after, clients=1)


def test_retry_helper_only_retries_typed_ownership_rejection(monkeypatch):
    import profile_concurrency_fixture as fixture
    from daystrom_dml.store_lock import StoreLockTimeout

    request = {"operation": "ingest", "args": ["same"], "kwargs": {"idempotency_key": "same"}}
    calls = []

    def rejected_once(_adapter, submitted):
        calls.append(deepcopy(submitted))
        if len(calls) == 1:
            raise StoreLockTimeout("test-only typed rejection")
        return {"confirmed": True}

    monkeypatch.setattr(fixture, "invoke_operation", rejected_once)
    event = call_event(None, request, client=0, sequence=0, deadline_ns=monotonic_ns() + 1_000_000_000)
    assert calls == [request, request]
    assert [attempt["error_code"] for attempt in event["attempts"]] == ["store_lock_timeout", None]
    calls.clear()

    def backend_timeout(_adapter, submitted):
        calls.append(submitted)
        raise TimeoutError("arbitrary backend timeout")

    monkeypatch.setattr(fixture, "invoke_operation", backend_timeout)
    with pytest.raises(TimeoutError, match="arbitrary backend"):
        call_event(None, request, client=0, sequence=0, deadline_ns=monotonic_ns() + 1_000_000_000)
    assert calls == [request]


def test_retry_helper_keeps_original_deadline_and_attempt_limit(monkeypatch):
    import profile_concurrency_fixture as fixture
    from daystrom_dml.store_lock import StoreLockTimeout

    request = {"operation": "ingest", "args": [], "kwargs": {}}
    calls = []

    def reject(_adapter, submitted):
        calls.append(submitted)
        raise StoreLockTimeout("bounded typed rejection")

    monkeypatch.setattr(fixture, "invoke_operation", reject)
    with pytest.raises(StoreLockTimeout):
        call_event(None, request, client=0, sequence=0, deadline_ns=monotonic_ns() + 1_000_000_000)
    assert len(calls) == 3
    calls.clear()
    clock = iter([10, 110, 120])
    monkeypatch.setattr(fixture, "monotonic_ns", lambda: next(clock))
    with pytest.raises(StoreLockTimeout):
        call_event(None, request, client=0, sequence=0, deadline_ns=100)
    assert len(calls) == 1


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_profile_checkpoint_backup_and_restart_overlap(tmp_path, monkeypatch, schema):
    from daystrom_dml.services.authority_backup import backup_authority, restore_backup
    import daystrom_dml.store_lock as lock_module

    directory = tmp_path / "authority"
    before = prepare_history(directory, schema)
    adapters = [make_adapter(directory, schema) for _ in range(3)]
    writer, reader, checkpoint = adapters
    held, release, checkpoint_done = threading.Event(), threading.Event(), threading.Event()
    backup_blocked, reader_blocked = threading.Event(), threading.Event()
    outcomes = {}
    errors = []
    workers = []
    original = lock_module.acquire_file_lock
    request = client_plan(before, 1)[0]
    backup = tmp_path / "concurrent-backup"

    def boundary(point):
        if point == "before_commit":
            held.set()
            assert release.wait(PROGRESS_SECONDS), "Writer release was not bounded"

    def observed(handle):
        try:
            return original(handle)
        except BlockingIOError:
            if threading.current_thread().name == "overlap-backup":
                backup_blocked.set()
            if threading.current_thread().name == "overlap-reader":
                reader_blocked.set()
            raise

    def start(name, action):
        def run():
            try:
                outcomes[name] = action()
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run, name=name, daemon=True)
        workers.append(worker)
        worker.start()

    def do_checkpoint():
        # This real SQLite maintenance runs while the writer's transaction is
        # uncommitted. It must finish without observing that incomplete state.
        checkpoint._journal.checkpoint()
        checkpoint_done.set()

    monkeypatch.setattr(lock_module, "acquire_file_lock", observed)
    writer._journal._fault_hook = boundary
    with fixed_clock():
        try:
            start("overlap-writer", lambda: call_event(writer, request, client=1, sequence=0))
            assert held.wait(PROGRESS_SECONDS)
            start("overlap-checkpoint", do_checkpoint)
            start("overlap-backup", lambda: backup_authority(directory / "dml_state.sqlite3", backup))
            read_request = client_plan(before, 5)[0]
            start("overlap-reader", lambda: call_event(reader, read_request, client=5, sequence=0))
            # Retention uses a pinned SQLite read; it can safely finish while
            # the cooperating writer owns its uncommitted mutation.
            retention = call_event(reader, client_plan(before, 6)[0], client=6, sequence=0)
            _assert_retention(retention, before)
            assert checkpoint_done.wait(PROGRESS_SECONDS)
            assert backup_blocked.wait(PROGRESS_SECONDS), "Backup never overlapped writer ownership"
            assert reader_blocked.wait(PROGRESS_SECONDS), "Read never overlapped writer ownership"
            assert sql_observation(directory) == before, "Uncommitted state became visible"
        finally:
            release.set()
            deadline = monotonic() + PROGRESS_SECONDS
            for worker in workers:
                worker.join(max(0, deadline - monotonic()))
            assert not any(worker.is_alive() for worker in workers), "Overlap operation failed to release"
            for adapter in adapters:
                adapter.close()
    assert not errors, errors
    after = sql_observation(directory)
    expected_receipt = assert_transition(before, after, Request(request["operation"], tuple(request["args"]), request["kwargs"]))
    assert outcomes["overlap-writer"]["result"] == expected_receipt
    _assert_retrieval(outcomes["overlap-reader"], after)
    restored = tmp_path / "restored"
    restore_backup(backup, restored)
    recovered = sql_observation(restored)
    assert recovered == after
    reopened = make_adapter(restored, schema)
    reopened.close()
    assert sql_observation(restored) == recovered


def test_profile_os_lock_wait_is_bounded_under_mixed_contention(tmp_path, monkeypatch, record_property):
    import daystrom_dml.store_lock as lock_module

    original = lock_module.acquire_file_lock
    guard = threading.Lock()
    starts = {}
    waits = []
    blocked = []

    def measured(handle):
        if (not threading.current_thread().name.startswith("profile-client-")
                or Path(handle.name).name != ".dml_store.lock"):
            return original(handle)
        key = (threading.get_ident(), id(handle))
        with guard:
            starts.setdefault(key, monotonic_ns())
        try:
            result = original(handle)
        except BlockingIOError:
            with guard:
                blocked.append(key)
            raise
        else:
            finished = monotonic_ns()
            with guard:
                waits.append((finished - starts.pop(key)) / 1e9)
            # A fixed bounded delay while genuinely holding OS ownership creates
            # contention without replacing the production lock/retry algorithm.
            sleep(0.002)
            return result

    monkeypatch.setattr(lock_module, "acquire_file_lock", measured)
    _, _, _, evidence = run_campaign(tmp_path / "authority", 3, 16, "threads-shared")
    assert evidence["history_verified"] is True
    assert waits and blocked and not starts
    assert max(waits) <= 32
    record_property("concurrency_lock_evidence", json.dumps({
        "schema_version": "dml-ownership-wait-evidence-v1", "schema": 3, "clients": 16,
        "timing_domain": "first_os_lock_attempt_to_acquired", "successful_acquisitions": len(waits),
        "blocked_attempts": len(blocked), "unfinished_acquisitions": len(starts),
        "max_wait_seconds": max(waits), "injected_owner_delay_seconds": 0.002,
    }, sort_keys=True))
