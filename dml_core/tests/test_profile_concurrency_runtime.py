"""Selected-profile lifetime, real ownership waits and related-state histories."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from copy import deepcopy
import importlib
import json
import multiprocessing
import os
import sqlite3
from threading import Event, get_ident, local
from time import monotonic

import pytest

from daystrom_dml.journal import IdempotencyConflict, JournalIntegrityError
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict
from daystrom_dml.store_lock import StoreLockTimeout, store_write_lock
from profile_crash_fixture import (
    SCOPE, expected_transition, make_adapter, prepare, request_for, sql_observation,
)

WAIT = 10
SERVICE_MODULES = (
    "receipt_ingestion", "receipt_lifecycle", "receipt_supersession",
    "receipt_update", "receipt_promotion",
)


@pytest.fixture(autouse=True, scope="module")
def isolated_environment():
    # Qualification selection/output variables are not production settings.
    # Keep the strict profile configuration boundary intact during these tests.
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key in list(os.environ):
            if key.startswith("DML_"):
                monkeypatch.delenv(key)
        yield


def invoke(adapter, request):
    # No process-global clock replacement while concurrent operations run.
    return getattr(adapter, request.method)(*request.args, **deepcopy(request.kwargs))


def calls(adapter, scenario):
    return [lambda request=request_for(operation, scenario.records): invoke(adapter, request)
            for operation in ("ingest", "update", "retire", "supersede", "promote")] + [
        lambda: adapter.retrieve_context("notebook", **SCOPE),
        lambda: adapter.inspect_memory_retention(0, **SCOPE),
    ]


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_close_timeout_fences_every_profile_operation_without_cancelling_admitted_write(tmp_path, schema):
    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    entered, release = Event(), Event()

    def block(_point):
        entered.set()
        assert release.wait(WAIT)

    adapter.embedder.barrier = block
    request = request_for("ingest", scenario.records)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(invoke, adapter, request)
            try:
                assert entered.wait(WAIT)
                with pytest.raises(TimeoutError, match="Profile shutdown is incomplete"):
                    adapter.close(projection_timeout=0)
                assert not adapter.store._stop_event.is_set()
                for call in [*calls(adapter, scenario), adapter.durability_status]:
                    with pytest.raises(RuntimeError, match="closing or closed"):
                        call()
                assert sql_observation(scenario.directory) == scenario.before
            finally:
                release.set()
            receipt = pending.result(WAIT)
        adapter.close(projection_timeout=1)
        adapter.close(projection_timeout=0)
        assert adapter.store._stop_event.is_set()
        assert sql_observation(scenario.directory)["revision"] == scenario.before["revision"] + 1
        reopened = make_adapter(scenario.directory, schema, unavailable=True)
        try:
            assert invoke(reopened, request) == receipt
        finally:
            reopened.close()
    finally:
        release.set()
        adapter.close()


def test_close_from_embedding_callback_fails_without_fencing_owner(tmp_path):
    adapter = make_adapter(tmp_path / "authority", 2)
    errors = []

    def attempt_close(_point):
        with pytest.raises(RuntimeError, match="from an admitted operation") as caught:
            adapter.close(projection_timeout=0)
        errors.append(str(caught.value))

    adapter.embedder.barrier = attempt_close
    try:
        receipt = adapter.ingest_memory_receipted("synthetic callback", idempotency_key="one", **SCOPE)
        assert receipt["revision"] == 1 and len(errors) == 1
        assert adapter.inspect_memory_retention(0, **SCOPE)["source"]["revision"] == 1
    finally:
        adapter.close()


def _forked_calls(adapter, scenario, channel):
    try:
        outcomes = []
        for call in [*calls(adapter, scenario), adapter.durability_status,
                     lambda: adapter.close(projection_timeout=0)]:
            try:
                call()
            except Exception as exc:
                outcomes.append((type(exc).__name__, str(exc)))
            else:
                outcomes.append(("unexpected_success", ""))
        channel.send(outcomes)
    finally:
        channel.close()


def test_inherited_profile_rejects_all_authority_paths_before_work(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("POSIX fork inheritance is unavailable on this platform")
    scenario = prepare(tmp_path / "authority", 2)
    adapter = make_adapter(scenario.directory, 2)
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_forked_calls, args=(adapter, scenario, child))
    try:
        process.start()
        child.close()
        assert parent.poll(WAIT), "inherited adapter blocked instead of rejecting"
        outcomes = parent.recv()
        assert len(outcomes) == 9
        assert all(kind == "RuntimeError" and "after fork" in message for kind, message in outcomes)
        process.join(WAIT)
        assert process.exitcode == 0
        assert sql_observation(scenario.directory) == scenario.before
    finally:
        if process.is_alive():
            process.kill()
        process.join(WAIT)
        parent.close()
        child.close()
        adapter.close()


def test_closed_profile_health_reports_degraded(tmp_path):
    from fastapi.testclient import TestClient

    from daystrom_dml.services.profile_http import build_profile_app

    adapter = make_adapter(tmp_path / "authority", 2)
    app = build_profile_app(adapter, tokens=("synthetic-token",))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        adapter.close()
        assert client.get("/health").json()["status"] == "degraded"
        with pytest.raises(RuntimeError, match="closing or closed"):
            adapter.durability_status()


@contextmanager
def http_client(adapter):
    from fastapi.testclient import TestClient

    from daystrom_dml.services.profile_http import build_profile_app

    with TestClient(build_profile_app(adapter, tokens=("synthetic-token",))) as client:
        yield client


def http_append(client):
    return client.post("/api/remember/receipt", headers={"Authorization": "Bearer synthetic-token"},
                       json={**SCOPE, "text": "Synthetic timeout request", "idempotency_key": "timeout-key"})


def test_real_zero_budget_ownership_rejection_is_distinct_from_http_execution_timeout(tmp_path, monkeypatch):
    """Shortened fixture budget; the separate six-caller test verifies real 30s."""
    from daystrom_dml.services import receipt_ingestion

    adapter = make_adapter(tmp_path / "authority", 2)

    @contextmanager
    def zero_budget(*args, **kwargs):
        kwargs["timeout_ms"] = 0
        with store_write_lock(*args, **kwargs) as ownership:
            yield ownership

    monkeypatch.setattr(receipt_ingestion, "store_write_lock", zero_budget)
    with http_client(adapter) as client:
        with store_write_lock(adapter.storage_dir, operation="real-http-owner", timeout_ms=0):
            response = http_append(client)
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True}}
        assert adapter._journal.read_snapshot()[0] == 0
        assert http_append(client).status_code == 200


def test_real_zero_budget_recall_ownership_timeout_has_explicit_http_reason(tmp_path, monkeypatch):
    """Keep the read code stable while proving the actual ownership boundary."""
    import daystrom_dml.dml_adapter as adapter_module

    scenario = prepare(tmp_path / "authority", 3)
    adapter = make_adapter(scenario.directory, 3)

    @contextmanager
    def zero_budget(*args, **kwargs):
        kwargs["timeout_ms"] = 0
        with store_write_lock(*args, **kwargs) as ownership:
            yield ownership

    monkeypatch.setattr(adapter_module, "store_write_lock", zero_budget)
    with http_client(adapter) as client:
        def recall():
            return client.post("/api/recall", headers={"Authorization": "Bearer synthetic-token"},
                               json={**SCOPE, "query": "notebooks"})

        with store_write_lock(adapter.storage_dir, operation="real-http-recall-owner", timeout_ms=0):
            response = recall()
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "retrieval_outcome_unavailable",
                                               "reason": "store_ownership_timeout"}}
        assert sql_observation(scenario.directory) == scenario.before
        assert recall().status_code == 200


def test_generic_http_recall_timeout_has_no_ownership_reason(tmp_path, monkeypatch):
    scenario = prepare(tmp_path / "authority", 3)
    adapter = make_adapter(scenario.directory, 3)

    def timeout(*_args, **_kwargs):
        raise TimeoutError("private retrieval callback information")

    monkeypatch.setattr(adapter, "retrieve_context", timeout)
    with http_client(adapter) as client:
        response = client.post("/api/recall", headers={"Authorization": "Bearer synthetic-token"},
                               json={**SCOPE, "query": "notebooks"})
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "retrieval_outcome_unavailable"}}
        assert "private" not in response.text
        assert sql_observation(scenario.directory) == scenario.before


@pytest.mark.parametrize("after_commit", [False, True])
def test_arbitrary_http_callback_timeout_does_not_claim_ownership_rejection(tmp_path, monkeypatch, after_commit):
    adapter = make_adapter(tmp_path / "authority", 2)
    original = adapter.ingest_memory_receipted
    acknowledged = []

    def timeout(*args, **kwargs):
        if after_commit:
            acknowledged.append(original(*args, **kwargs))
        raise TimeoutError("private callback information")

    monkeypatch.setattr(adapter, "ingest_memory_receipted", timeout)
    with http_client(adapter) as client:
        response = http_append(client)
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "receipt_outcome_unavailable", "retry_same_key": True}}
        assert "private" not in response.text
        assert adapter._journal.read_snapshot()[0] == int(after_commit)
        monkeypatch.setattr(adapter, "ingest_memory_receipted", original)
        replay = http_append(client)
        assert replay.status_code == 200
        if after_commit:
            assert replay.json() == acknowledged[0]
        assert adapter._journal.read_snapshot()[0] == 1


def test_profile_embedding_timeout_keeps_its_specific_failure_code(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path / "authority", 2)

    def timeout(_text):
        raise TimeoutError("private embedding timeout")

    monkeypatch.setattr(adapter.embedder, "embed", timeout)
    with http_client(adapter) as client:
        response = http_append(client)
        assert response.status_code == 503
        assert response.json() == {"detail": {"code": "embedding_unavailable", "retry_same_key": True}}
        assert adapter._journal.read_snapshot()[0] == 0


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_shared_stale_read_cannot_relabel_newly_loaded_context_revision(tmp_path, monkeypatch, schema):
    scenario = prepare(tmp_path / "authority", schema)
    reader = make_adapter(scenario.directory, schema)
    writer = make_adapter(scenario.directory, schema)
    stale_started, release_stale, stale_finished = Event(), Event(), Event()
    owner = get_ident()
    raw_snapshot = reader._journal._read_snapshot
    read_snapshot = reader._journal.read_snapshot

    def pause_old_snapshot(connection):
        result = raw_snapshot(connection)
        if get_ident() != owner:
            stale_started.set()
            assert release_stale.wait(WAIT)
        return result

    def finish_old_read_after_new_load():
        result = read_snapshot()
        if get_ident() == owner and getattr(reader._mutation_local, "depth", 0):
            # The loader already captured the newer (revision, payload) pair.
            # A concurrent preparation read then overwrites the journal's
            # advisory last-read property before that pair reaches the loader.
            release_stale.set()
            assert stale_finished.wait(WAIT)
        return result

    def stale_preparation():
        try:
            return reader._journal.read_snapshot()
        finally:
            stale_finished.set()

    monkeypatch.setattr(reader._journal, "_read_snapshot", pause_old_snapshot)
    monkeypatch.setattr(reader._journal, "read_snapshot", finish_old_read_after_new_load)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            stale = pool.submit(stale_preparation)
            try:
                assert stale_started.wait(WAIT)
                receipt = writer.ingest_memory_receipted("A fact committed after the paused snapshot",
                    idempotency_key="new-revision", meta={"source_trust": "trusted"}, **SCOPE)
                report = reader.retrieve_context("notebooks", **SCOPE)
                assert stale.result(WAIT)[0] == scenario.before["revision"]
                assert report["decision"]["store_revision"] == receipt["revision"]
                assert any(int(item["id"]) == receipt["result"]["memory"]["id"] for item in report["items"])
                assert reader._last_observed_state[0] == receipt["revision"]
                assert sql_observation(scenario.directory)["revision"] == receipt["revision"]
            finally:
                release_stale.set()
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_owned_profile_refresh_validates_one_snapshot_even_when_unchanged(tmp_path, monkeypatch, schema):
    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    peer = make_adapter(scenario.directory, schema)
    original = adapter._journal._read_snapshot
    reads = []

    def checked_read(connection):
        result = original(connection)
        reads.append(result[0])
        return result

    monkeypatch.setattr(adapter._journal, "_read_snapshot", checked_read)
    try:
        revision = scenario.before["revision"]
        for iteration in range(3):
            if iteration == 1:
                receipt = peer.ingest_memory_receipted("New independently committed snapshot",
                    idempotency_key="refresh-once", **SCOPE)
                revision = receipt["revision"]
            previous_reads = len(reads)
            # This is the same refresh/ownership boundary entered by recall.
            with adapter._mutation_transaction("qualify-single-snapshot-refresh"):
                assert adapter._last_observed_state[0] == revision
            assert reads[previous_reads:] == [revision]
            assert [item.id for item in adapter.store.items()] == [
                item["id"] for item in sql_observation(scenario.directory)["state"]["items"]]
        assert len(reads) == 3, "Unchanged calls must perform fresh full validation"
    finally:
        peer.close()
        adapter.close()


@pytest.mark.parametrize("schema,table", [(2, "records"), (3, "records"), (4, "records"),
                                         (3, "outbox"), (4, "outbox")])
def test_owned_refresh_detects_corruption_without_head_or_mtime_change(tmp_path, schema, table):
    scenario = prepare(tmp_path / "authority", schema)
    adapter = make_adapter(scenario.directory, schema)
    try:
        with adapter._mutation_transaction("before-hidden-corruption"):
            pass
        observed = adapter._last_observed_state
        adapter.query_cache.get("must-remain-on-failed-import", lambda _text: [1., 0., 0., 0.])
        path = adapter._journal.path
        stamp = path.stat()
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                # Table names are the fixed local parameter inventory above.
                connection.execute(f"UPDATE {table} SET checksum=? WHERE rowid=(SELECT MIN(rowid) FROM {table})",
                                   ("0" * 64,))
            assert connection.execute("SELECT revision FROM state WHERE id=1").fetchone()[0] == observed[0]
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert path.stat().st_mtime_ns == observed[1]
        for _ in range(2):
            with pytest.raises(JournalIntegrityError, match="checksum"):
                with adapter._mutation_transaction("after-hidden-corruption"):
                    pytest.fail("Corrupt unchanged authority reached the operation body")
        assert adapter._last_observed_state == observed
        assert "must-remain-on-failed-import" in adapter.query_cache.values
    finally:
        adapter.close()


@pytest.mark.parametrize("operation", ["recall", "retention"])
def test_successful_close_drains_admitted_read_before_dependency_cleanup(tmp_path, monkeypatch, operation):
    scenario = prepare(tmp_path / "authority", 3)
    adapter = make_adapter(scenario.directory, 3)
    reading, release, draining = Event(), Event(), Event()

    def pause_read(point):
        if point == ("during_embedding" if operation == "recall" else "retention_after_snapshot"):
            reading.set()
            assert release.wait(WAIT)

    adapter.embedder.barrier = pause_read
    adapter._journal._fault_hook = pause_read
    condition = adapter._profile_operation_lifetime._condition
    original_wait = condition.wait

    def observe_drain(timeout=None):
        draining.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observe_drain)
    call = (lambda: adapter.retrieve_context("notebook", **SCOPE)) if operation == "recall" else (
        lambda: adapter.inspect_memory_retention(0, **SCOPE))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(call)
            closing = None
            try:
                assert reading.wait(WAIT)
                closing = pool.submit(adapter.close, projection_timeout=WAIT)
                assert draining.wait(WAIT), "close did not wait for admitted read"
                assert not closing.done() and not adapter.store._stop_event.is_set()
                with pytest.raises(RuntimeError, match="closing or closed"):
                    adapter.durability_status()
            finally:
                release.set()
            result = pending.result(WAIT)
            assert isinstance(result, dict)
            assert closing is not None and closing.result(WAIT) is None
        assert adapter.store._stop_event.is_set()
        assert sql_observation(scenario.directory) == scenario.before
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("shared", [False, True], ids=["separate-adapters", "shared-adapter"])
@pytest.mark.parametrize("winner,contender", [
    ("update", "retire"), ("retire", "update"),
    ("update", "supersede"), ("supersede", "update"),
    ("update", "promote"), ("promote", "update"),
    ("ingest", "ingest"),
])
def test_related_state_history_has_only_compatible_commits(tmp_path, monkeypatch, schema, shared, winner, contender):
    scenario = prepare(tmp_path / "authority", schema)
    first = make_adapter(scenario.directory, schema)
    second = first if shared else make_adapter(scenario.directory, schema)
    before_commit, contender_waiting, release = Event(), Event(), Event()
    role = local()
    winning_request = request_for(winner, scenario.records, key="winner")
    competing_request = request_for(contender, scenario.records, key="contender")
    if winner == contender == "ingest":
        competing_request.kwargs["idempotency_key"] = "winner"
        competing_request.args = ("Different request with identical scoped key",)

    def barrier(point):
        if point == "after_begin" and getattr(role, "name", None) == "winner":
            before_commit.set()
            assert release.wait(WAIT)

    first._journal._fault_hook = barrier
    second._journal._fault_hook = barrier

    @contextmanager
    def observe_lock(*args, **kwargs):
        if getattr(role, "name", None) == "contender":
            contender_waiting.set()
        with store_write_lock(*args, **kwargs) as ownership:
            yield ownership

    for name in SERVICE_MODULES:
        monkeypatch.setattr(importlib.import_module("daystrom_dml.services." + name),
                            "store_write_lock", observe_lock)

    def run(adapter, request, name):
        role.name = name
        return invoke(adapter, request)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            successful = pool.submit(run, first, winning_request, "winner")
            pending = None
            try:
                assert before_commit.wait(WAIT)
                pending = pool.submit(run, second, competing_request, "contender")
                assert contender_waiting.wait(WAIT), "contender did not reach actual ownership acquisition"
                assert not pending.done()
                assert sql_observation(scenario.directory) == scenario.before, "uncommitted writer leaked state"
            finally:
                release.set()
            acknowledged = successful.result(WAIT)
            assert pending is not None
            if winner == "promote":
                # Promotion stores a derived record with its old source snapshot;
                # it does not mutate that source, so a later source edit is valid.
                later = pending.result(WAIT)
                expected = expected_transition(scenario.before, winning_request)
                assert acknowledged == expected["receipt"]
                intermediate = {**scenario.before, "state": expected["state"],
                                "revision": acknowledged["revision"]}
                final = expected_transition(intermediate, competing_request)
                assert later == final["receipt"]
                assert sql_observation(scenario.directory)["state"] == final["state"]
                assert sql_observation(scenario.directory)["revision"] == scenario.before["revision"] + 2
            else:
                error = IdempotencyConflict if winner == contender == "ingest" else ReceiptLifecycleConflict
                with pytest.raises(error):
                    pending.result(WAIT)
                after = sql_observation(scenario.directory)
                assert after["revision"] == scenario.before["revision"] + 1
                assert after["receipts"] == [*scenario.before["receipts"], acknowledged]
                if winner != "ingest":
                    expected = expected_transition(scenario.before, winning_request)
                    assert acknowledged == expected["receipt"]
                    assert after["state"] == expected["state"]
                else:
                    assert after["state"]["items"] == [*scenario.records, acknowledged["result"]["memory"]]
                    assert after["state"]["items"][-1]["text"] == winning_request.args[0]
            # Success must remain historical even after the related-state writer.
            assert invoke(first, winning_request) == acknowledged
    finally:
        release.set()
        if second is not first:
            second.close()
        first.close()


def test_real_default_ownership_wait_times_out_all_mutations_and_recall_without_commit(tmp_path, record_property):
    """Actual 30-second OS contention; no shortened/injected timeout budget."""
    scenario = prepare(tmp_path / "authority", 3)
    adapter = make_adapter(scenario.directory, 3)
    operations = calls(adapter, scenario)[:-1]

    def blocked(call):
        start = monotonic()
        with pytest.raises(StoreLockTimeout, match="Timed out waiting for DML store lock"):
            call()
        return monotonic() - start

    try:
        with ThreadPoolExecutor(max_workers=len(operations)) as pool:
            with store_write_lock(scenario.directory, operation="qualification-holder", timeout_ms=0):
                futures = [pool.submit(blocked, call) for call in operations]
                # Retention reads its SQLite snapshot and needs no store ownership.
                assert adapter.inspect_memory_retention(0, **SCOPE)["source"]["revision"] == scenario.before["revision"]
                durations = [future.result(35) for future in futures]
                assert all(29.8 <= duration <= 32 for duration in durations), durations
                assert sql_observation(scenario.directory) == scenario.before
            start = monotonic()
            receipt = invoke(adapter, request_for("ingest", scenario.records))
            released_progress = monotonic() - start
            assert released_progress < WAIT
            assert receipt["revision"] == scenario.before["revision"] + 1
        record_property("actual_ownership_timeout_seconds", json.dumps(durations))
        record_property("ownership_release_progress_seconds", json.dumps(released_progress))
    finally:
        adapter.close()
