"""Concurrent authenticated profile operations through actual TCP connections."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
import socket
import time

import httpx
import pytest

from daystrom_dml.provider_client import ProviderClient
from profile_concurrency_fixture import (
    check_history, prepare_history, requested_scales, write_evidence, write_history,
)
from profile_concurrency_http_fixture import (
    CLIENT_KEEPALIVE_SECONDS, HEADERS, PROGRESS_DEADLINE, SERVER_KEEPALIVE_SECONDS,
    TOKEN, LoopbackProvider, http_history, invoke_http, wire_request,
)
from profile_crash_fixture import (
    SCOPE, Request, assert_transition, digest, encoded, sql_observation,
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("clients", requested_scales())
def test_real_http_mixed_history(tmp_path, schema, clients, record_property):
    directory = tmp_path / "authority"
    before = prepare_history(directory, schema)
    with LoopbackProvider(directory, schema) as provider:
        events, transport = asyncio.run(http_history(provider, before, clients))
        after = sql_observation(directory)
        checked = check_history(before, events, after, clients=clients)
        assert transport["arrivals"] == clients
        assert transport["max_active_http"] >= clients
        assert transport["active_workers"] == 0
        assert transport["http_requests"] == checked["history_events"] + checked["ownership_rejections"]
        expected_statuses = {"200": checked["history_events"]}
        if checked["ownership_rejections"]:
            expected_statuses["503"] = checked["ownership_rejections"]
        assert transport["response_status_counts"] == expected_statuses
        assert 1 <= transport["max_active_workers"] <= transport["worker_threads"] <= 40
        evidence = {
            "schema_version": "dml-concurrency-evidence-v1", "schema": schema, "transport": "http",
            "clients": clients, "measured_ready_clients": transport["arrivals"], "process_count": 1,
            "committed_revisions": checked["unique_commits"], "history_verified": checked["history_verified"],
            "history_events": checked["history_events"], "max_operation_seconds": checked["max_operation_seconds"],
            "ownership_rejections": checked["ownership_rejections"], "max_attempts": checked["max_attempts"],
            "duration_seconds": transport["duration_seconds"], "first_progress_seconds": transport["first_progress_seconds"],
            "max_http_in_flight": transport["max_active_http"],
            "max_active_workers": transport["max_active_workers"], "worker_threads": transport["worker_threads"],
            "client_keepalive_expiry_seconds": CLIENT_KEEPALIVE_SECONDS,
            "server_keepalive_timeout_seconds": SERVER_KEEPALIVE_SECONDS,
        }
        evidence.update(write_history(before, events, after, clients=clients, transport="http"))
        write_evidence(record_property, evidence)
    # Lifespan shutdown leaves exactly the same durable acknowledged history.
    assert encoded(sql_observation(directory)) == encoded(after)


def append_request(key, text="Synthetic disconnected request"):
    return {"operation": "ingest", "args": [text],
            "kwargs": {**SCOPE, "idempotency_key": key, "kind": "memory", "meta": {"source_trust": "trusted"}}}


def assert_one_commit(before, after, request, receipt):
    expected = assert_transition(before, after, Request(request["operation"], tuple(request["args"]), request["kwargs"]))
    assert receipt == expected
    assert after["revision"] == before["revision"] + 1


async def cancel_client_at_barrier(provider, route, body):
    async with httpx.AsyncClient(base_url=provider.url, headers=HEADERS, timeout=10, trust_env=False) as client:
        task = asyncio.create_task(client.post(route, json=body))
        await asyncio.to_thread(provider.receive, "blocked")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("schema", [2, 3, 4])
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
@pytest.mark.parametrize("loss", ["disconnect", "read_timeout", "task_cancel"])
def test_real_http_uncertain_ack_replays_once(tmp_path, schema, point, loss, record_property):
    directory = tmp_path / "authority"
    before = prepare_history(directory, schema)
    request = append_request(f"uncertain-{point}-{loss}")
    route, body = wire_request(request)
    with LoopbackProvider(directory, schema) as provider:
        provider.command("arm", point=point)
        if loss == "disconnect":
            content = json.dumps(body).encode("utf-8")
            connection = socket.create_connection(("127.0.0.1", provider.port), timeout=10)
            try:
                connection.sendall((f"POST {route} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                    f"Authorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n"
                                    f"Content-Length: {len(content)}\r\nConnection: close\r\n\r\n").encode() + content)
                provider.receive("blocked")
                connection.shutdown(socket.SHUT_RDWR)
            finally:
                connection.close()
        elif loss == "read_timeout":
            def timed_request():
                with provider.client(timeout=httpx.Timeout(10, read=0.2)) as client:
                    return client.post(route, json=body)
            with ThreadPoolExecutor(max_workers=1) as executor:
                attempt = executor.submit(timed_request)
                provider.receive("blocked")
                with pytest.raises(httpx.ReadTimeout):
                    attempt.result(timeout=PROGRESS_DEADLINE)
        else:
            asyncio.run(cancel_client_at_barrier(provider, route, body))

        at_loss = sql_observation(directory)
        assert at_loss["revision"] == before["revision"] + (point == "after_commit")
        # Client-side loss remains uncertain in both cases; only the independent
        # fixture knows which commit boundary was paused.
        started = time.monotonic()
        provider.command("release")
        first_retry = provider.post(request)
        assert first_retry.status_code == 200, first_retry.text
        assert time.monotonic() - started <= PROGRESS_DEADLINE
        after = sql_observation(directory)
        assert_one_commit(before, after, request, first_retry.json())
        assert provider.post(request).json() == first_retry.json()
        assert encoded(sql_observation(directory)) == encoded(after)
        with provider.client() as client:
            assert client.get("/health").json()["status"] == "ok"
        assert provider.command("metrics")["active_workers"] == 0
        record_property("http_uncertain_ack", json.dumps({"schema": schema, "loss": loss,
            "point": point, "client_outcome": "uncertain", "resolved_commits": 1,
            "replay_equal": True, "released_progress_seconds": time.monotonic() - started}))
    # A new provider process also resolves the same historical request unchanged.
    with LoopbackProvider(directory, schema) as restarted:
        assert restarted.post(request).json() == first_retry.json()
    assert encoded(sql_observation(directory)) == encoded(after)


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_real_provider_recall_auth_and_scope_progress_while_embedding_blocked(tmp_path, schema, record_property):
    directory = tmp_path / "authority"
    before = prepare_history(directory, schema)
    blocked = append_request("blocked-embedding", "HTTP_BLOCKED_EMBEDDING")
    progress = append_request("unrelated-progress", "Another HTTP caller completes")
    with LoopbackProvider(directory, schema) as provider, ThreadPoolExecutor(max_workers=1) as executor:
        provider.command("arm", point="embedding", embedding_text=blocked["args"][0])
        outstanding = executor.submit(provider.post, blocked)
        provider.receive("blocked")
        try:
            started = time.monotonic()
            with provider.client() as client:
                assert client.get("/health").json()["status"] == "ok"
                unauthorized = client.post("/api/recall", json={**SCOPE, "query": "facts"},
                                           headers={"Authorization": "Bearer invalid-synthetic-token"})
                assert unauthorized.status_code == 401
                wrong_scope = client.post("/api/memory/update/receipt", json={**SCOPE, "tenant_id": "foreign",
                    "memory_id": 0, "expected_memory_digest": digest(before["state"]["items"][0]),
                    "text": "Forbidden scope rewrite", "reason": "Synthetic rejection",
                    "idempotency_key": "wrong-scope"})
                assert wrong_scope.status_code == 404
                assert "notebooks" not in wrong_scope.text
            # Exercise the actual public SDK method; profile writes use the
            # admitted HTTP routes because the SDK's writes are legacy routes.
            with ProviderClient(provider.url, token=TOKEN, timeout=PROGRESS_DEADLINE) as sdk:
                recalled = sdk.recall("Concurrent facts", **SCOPE, top_k=10)
            expected = {str(record["id"]) for record in before["state"]["items"]
                        if all(record["meta"].get(key) == value for key, value in SCOPE.items())}
            assert {entry["id"] for entry in recalled["items"]} == expected
            assert recalled["decision"]["store_revision"] == before["revision"]
            assert "PRIVATE_FOREIGN_RECOVERY_SENTINEL" not in recalled["raw_context"]
            committed = provider.post(progress)
            assert committed.status_code == 200, committed.text
            after_progress = sql_observation(directory)
            assert_one_commit(before, after_progress, progress, committed.json())
            assert time.monotonic() - started <= PROGRESS_DEADLINE
            assert not outstanding.done()
            assert provider.command("metrics")["max_active_workers"] >= 2
        finally:
            provider.command("release")
        completed = outstanding.result(timeout=PROGRESS_DEADLINE)
        assert completed.status_code == 200, completed.text
        assert_one_commit(after_progress, sql_observation(directory), blocked, completed.json())
        record_property("http_blocked_embedding_progress", json.dumps({"schema": schema,
            "provider_client_recall": True, "unrelated_commit": True, "auth_rejected": True,
            "wrong_scope_rejected": True, "max_active_workers": provider.command("metrics")["max_active_workers"]}))


class RecordedResponses:
    """Classifier unit fixture; never counted as real-network qualification."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    async def post(self, route, *, json):
        from copy import deepcopy

        self.sent.append((route, deepcopy(json)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def rejection_response():
    return httpx.Response(503, json={"detail": {
        "code": "receipt_ownership_unavailable", "retry_same_key": True}})


def test_http_retry_preserves_every_attempt_and_exact_request():
    request = append_request("retry-classifier")
    client = RecordedResponses([rejection_response(), rejection_response(), httpx.Response(200, json={"ok": True})])
    event = asyncio.run(invoke_http(client, request, deadline=time.monotonic() + 10))
    attempts = event["attempts"]
    assert len(client.sent) == len(attempts) == 3
    assert client.sent == [wire_request(request)] * 3
    assert [attempt["status_code"] for attempt in attempts] == [503, 503, 200]
    assert [attempt["error_code"] for attempt in attempts] == ["store_lock_timeout", "store_lock_timeout", None]
    assert {attempt["request_digest"] for attempt in attempts} == {digest(request)}
    assert all(previous["finished_ns"] <= following["started_ns"] for previous, following in zip(attempts, attempts[1:]))
    assert event["started_ns"] == attempts[0]["started_ns"]
    assert event["finished_ns"] == attempts[-1]["finished_ns"]
    assert event["result"] == {"ok": True}


@pytest.mark.parametrize("status,body", [
    (502, {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True}}),
    (409, {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True}}),
    (503, {"detail": {"code": "receipt_outcome_unavailable", "retry_same_key": True}}),
    (503, {"detail": {"code": "retrieval_outcome_unavailable"}}),
    (503, {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": False}}),
    (503, {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": 1}}),
    (503, {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True, "extra": True}}),
    (503, {"detail": {"code": "receipt_ownership_unavailable"}}),
    (503, {"detail": None}),
    (503, None),
])
def test_http_retry_rejects_unapproved_status_or_error_shape(status, body):
    client = RecordedResponses([httpx.Response(status, json=body), httpx.Response(200, json={"ok": True})])
    with pytest.raises((AssertionError, ValueError)):
        asyncio.run(invoke_http(client, append_request("strict-classifier"), deadline=time.monotonic() + 10))
    assert len(client.sent) == 1


@pytest.mark.parametrize("error", [httpx.ReadError("synthetic transport loss"), asyncio.CancelledError()])
def test_http_retry_does_not_hide_transport_loss_or_cancellation(error):
    client = RecordedResponses([error, httpx.Response(200, json={"ok": True})])
    with pytest.raises(type(error)):
        asyncio.run(invoke_http(client, append_request("no-transport-retry"), deadline=time.monotonic() + 10))
    assert len(client.sent) == 1


def test_http_retry_exhausts_three_attempts_without_resetting_budget():
    client = RecordedResponses([rejection_response() for _ in range(4)])
    with pytest.raises(AssertionError, match="three bounded attempts"):
        asyncio.run(invoke_http(client, append_request("bounded-attempts"), deadline=time.monotonic() + 10))
    assert len(client.sent) == 3


def test_http_retry_respects_original_absolute_deadline():
    class DelayedSecondResponse(RecordedResponses):
        async def post(self, route, *, json):
            if self.sent:
                self.sent.append((route, json))
                await asyncio.Event().wait()
            return await super().post(route, json=json)

    client = DelayedSecondResponse([rejection_response()])
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        asyncio.run(invoke_http(client, append_request("one-deadline"), deadline=time.monotonic() + 0.2))
    assert len(client.sent) == 2


@pytest.mark.parametrize("operation", ["retrieve", "retention"])
def test_http_retry_does_not_apply_receipt_rejection_to_read_routes(operation):
    request = {"operation": operation, "args": ["query" if operation == "retrieve" else 0], "kwargs": dict(SCOPE)}
    client = RecordedResponses([rejection_response(), httpx.Response(200, json={"ok": True})])
    with pytest.raises(AssertionError):
        asyncio.run(invoke_http(client, request, deadline=time.monotonic() + 10))
    assert len(client.sent) == 1


@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_http_campaign_cancels_and_drains_siblings_before_client_close(monkeypatch, diagnostic_failure):
    import profile_concurrency_fixture
    import profile_concurrency_http_fixture

    original_error = httpx.ReadError("synthetic first caller failure")
    state = {"sibling_drained": False, "client_closed": False}
    captures = []

    def rejected_history(before, events, **arguments):
        assert state == {"sibling_drained": True, "client_closed": True}
        captures.append({"before": before, "events": events, **arguments})
        if diagnostic_failure:
            raise OSError("synthetic diagnostic storage failure")
        return {}

    class FailingAndBlockedClient:
        def __init__(self, **_kwargs):
            self.started = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_error):
            assert state["sibling_drained"]
            state["client_closed"] = True

        async def post(self, _route, *, json):
            if json["idempotency_key"] == "caller-0":
                await self.started.wait()
                raise original_error
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                state["sibling_drained"] = True

    class ControlledProvider:
        url = "http://synthetic-no-network"

        def command(self, name, **_arguments):
            assert name == "arrival"
            return {}

    monkeypatch.setattr(profile_concurrency_fixture, "full_client_plan",
                        lambda _before, client, _clients: [append_request(f"caller-{client}")])
    monkeypatch.setattr(profile_concurrency_fixture, "write_rejected_history", rejected_history)
    monkeypatch.setattr(profile_concurrency_http_fixture.httpx, "AsyncClient", FailingAndBlockedClient)
    with pytest.raises(httpx.ReadError) as caught:
        asyncio.run(http_history(ControlledProvider(), {}, 2))
    assert caught.value is original_error
    assert state == {"sibling_drained": True, "client_closed": True}
    assert len(captures) == 1 and captures[0]["error"] is original_error
    assert captures[0]["diagnostics"]["failure_stage"] == "campaign"
    assert "counter_capture_error" in captures[0]["diagnostics"]
    assert captures[0]["diagnostics"]["failure_observed_phase"] == "before-sibling-drain"
    assert captures[0]["diagnostics"]["failure_observed_ns"] < captures[0]["diagnostics"]["campaign_deadline_ns"]


def test_http_progress_failure_retains_completed_history_and_exact_observations(monkeypatch, tmp_path):
    import profile_concurrency_fixture
    import profile_concurrency_http_fixture

    released_ns = 1_000_000_000_000
    first_response_ns = released_ns + 11_250_000_000
    counters = {"released_ns": released_ns, "first_response_started_ns": first_response_ns - 1_000_000,
                "arrivals": 1, "active_workers": 0, "http_requests": 2, "response_status_counts": {"200": 2}}
    requests = [append_request(f"captured-{index}") for index in range(2)]

    class UnusedClient:
        def __init__(self, **_arguments):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_arguments):
            pass

    class ObservedProvider:
        url = "http://synthetic-no-network"

        def command(self, name, **arguments):
            if name == "arrival":
                assert arguments == {"clients": 1}
                return {}
            assert name == "metrics" and arguments == {"timeout": 1}
            return dict(counters)

        def receive(self, name):
            assert name == "arrivals_released"
            return {"released_ns": released_ns}

    async def completed_call(_client, request, *, deadline):
        assert deadline > time.monotonic()
        index = requests.index(request)
        return {"request": request, "started_ns": released_ns - 10_000_000,
                "finished_ns": first_response_ns + index * 1_000_000, "result": {"synthetic": index}}

    monkeypatch.setattr(profile_concurrency_fixture, "HISTORY_DIRECTORY", tmp_path / "histories")
    monkeypatch.setattr(profile_concurrency_fixture, "full_client_plan",
                        lambda _before, _client, _clients: requests)
    monkeypatch.setattr(profile_concurrency_http_fixture.httpx, "AsyncClient", UnusedClient)
    monkeypatch.setattr(profile_concurrency_http_fixture, "invoke_http", completed_call)
    with pytest.raises(AssertionError, match=r"11\.250000000s.*\[0, 10\]s.*completed_events=2"):
        asyncio.run(http_history(ObservedProvider(), {"schema": 4}, 1))
    paths = list((tmp_path / "diagnostics").glob("*.rejected.json.gz"))
    assert len(paths) == 1
    diagnostic = json.loads(gzip.decompress(paths[0].read_bytes()))
    assert diagnostic["schema_version"] == "dml-concurrency-rejected-diagnostic-v1"
    assert diagnostic["accepted"] is False
    assert len(diagnostic["events"]) == 2
    assert [event["request"] for event in diagnostic["events"]] == requests
    observed = diagnostic["diagnostics"]
    assert observed.pop("failure_observed_ns") < observed.pop("campaign_deadline_ns")
    assert observed.pop("failure_observed_phase") == "outside-active-callers"
    assert observed == {"failure_stage": "first-progress", "released_ns": released_ns,
                        "first_response_ns": first_response_ns, "first_progress_seconds": 11.25,
                        "transport_counters": counters}
    assert not (tmp_path / "histories").exists()


def test_http_retry_allows_only_positively_typed_read_ownership_rejection():
    request = {"operation": "retrieve", "args": ["Synthetic exact query"], "kwargs": {**SCOPE, "top_k": 10}}
    detail = {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}
    client = RecordedResponses([httpx.Response(503, json={"detail": detail}), httpx.Response(200, json={"items": []})])
    event = asyncio.run(invoke_http(client, request, deadline=time.monotonic() + 10))
    assert client.sent == [wire_request(request)] * 2
    assert event["attempts"][0]["error_detail"] == detail
    assert event["attempts"][0]["error_code"] == "store_lock_timeout"
    assert event["attempts"][1]["error_code"] is None
    assert {attempt["request_digest"] for attempt in event["attempts"]} == {digest(request)}


@pytest.mark.parametrize("status,detail", [
    (503, {"code": "retrieval_outcome_unavailable"}),
    (503, {"code": "retrieval_outcome_unavailable", "reason": "backend_timeout"}),
    (503, {"code": "retrieval_outcome_unavailable", "reason": True}),
    (503, {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout", "retry_same_key": True}),
    (503, {"code": "retention_outcome_unavailable", "reason": "store_ownership_timeout"}),
    (503, {"code": "receipt_ownership_unavailable", "retry_same_key": True}),
    (504, {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}),
])
def test_http_retry_rejects_untyped_or_mismatched_read_failures(status, detail):
    request = {"operation": "retrieve", "args": ["Synthetic exact query"], "kwargs": dict(SCOPE)}
    client = RecordedResponses([httpx.Response(status, json={"detail": detail}), httpx.Response(200, json={"items": []})])
    with pytest.raises(AssertionError):
        asyncio.run(invoke_http(client, request, deadline=time.monotonic() + 10))
    assert len(client.sent) == 1


@pytest.mark.parametrize("client_expiry", [SERVER_KEEPALIVE_SECONDS, CLIENT_KEEPALIVE_SECONDS])
def test_real_idle_timer_race_and_declared_client_expiry(tmp_path, monkeypatch, client_expiry, record_property):
    # Private hooks are confined to this protocol regression. No error is
    # fabricated: the actual server idle timer closes a real TCP socket after
    # the client has selected it. The declared policy instead selects a fresh
    # connection before the server's five-second expiry.
    from httpcore._async.http11 import AsyncHTTP11Connection

    directory = tmp_path / "authority"
    before = prepare_history(directory, 2)
    observations = {"client_expiry_seconds": client_expiry, "server_expiry_seconds": SERVER_KEEPALIVE_SECONDS}
    with LoopbackProvider(directory, 2) as provider:
        provider.command("arm", point="idle_close")

        async def exercise():
            limits = httpx.Limits(keepalive_expiry=client_expiry)
            async with httpx.AsyncClient(base_url=provider.url, limits=limits, timeout=10, trust_env=False) as client:
                first = await client.get("/health")
                assert first.status_code == 200
                first_peer = list(first.extensions["network_stream"].get_extra_info("client_addr"))
                observations["first_peer"] = first_peer
                original = AsyncHTTP11Connection._send_request_headers

                async def selected_headers(connection, request):
                    peer = list(connection._network_stream.get_extra_info("client_addr"))
                    observations.update(second_peer=peer, second_selected_ns=time.monotonic_ns())
                    if peer == first_peer:
                        blocked = await asyncio.to_thread(provider.receive, "blocked")
                        assert blocked["point"] == "idle_close"
                        assert blocked["peer"] == peer
                        assert blocked["server_timeout_seconds"] == SERVER_KEEPALIVE_SECONDS
                        observations["timer_fired_ns"] = blocked["timer_fired_ns"]
                        await original(connection, request)
                        observations["second_written_ns"] = time.monotonic_ns()
                        # The loopback bytes enter the receive buffer while the
                        # timer callback is paused; its normal close then resets
                        # the same socket with unread request data.
                        await asyncio.sleep(0.01)
                        await asyncio.to_thread(provider.command, "release")
                        return
                    return await original(connection, request)

                monkeypatch.setattr(AsyncHTTP11Connection, "_send_request_headers", selected_headers)
                await asyncio.sleep(CLIENT_KEEPALIVE_SECONDS + 0.1)
                try:
                    if client_expiry == SERVER_KEEPALIVE_SECONDS:
                        with pytest.raises((httpx.ReadError, httpx.RemoteProtocolError)) as lost:
                            await client.get("/health")
                        observations.update(outcome=type(lost.value).__name__, finished_ns=time.monotonic_ns())
                        chain, seen, cause = [], set(), lost.value
                        while cause is not None and id(cause) not in seen:
                            seen.add(id(cause))
                            chain.append({"type": type(cause).__name__, "representation": repr(cause)})
                            cause = cause.__cause__ or cause.__context__
                        observations["exception_chain"] = chain
                        closed = await asyncio.to_thread(provider.receive, "idle_closed")
                        assert closed["peer"] == first_peer
                        observations["close_started_ns"] = closed["close_started_ns"]
                        assert observations["second_peer"] == first_peer
                        assert (observations["second_selected_ns"] <= observations["timer_fired_ns"]
                                <= observations["second_written_ns"] <= observations["close_started_ns"]
                                <= observations["finished_ns"])
                    else:
                        second = await client.get("/health")
                        assert second.status_code == 200
                        observations["outcome"] = "http_200"
                        assert observations["second_peer"] != first_peer
                        assert "timer_fired_ns" not in observations
                finally:
                    await asyncio.to_thread(provider.command, "release")

        asyncio.run(exercise())
    assert encoded(sql_observation(directory)) == encoded(before)
    record_property("http_idle_close_boundary", json.dumps(observations, sort_keys=True))
