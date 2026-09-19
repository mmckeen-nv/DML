"""Real loopback provider process with test-only control over scheduling.

Control messages use child stdin/stdout, never production HTTP endpoints. Arrival
and worker counters measure different things; a queued HTTP request is not a
running Python worker. Only synthetic stores and credentials are used.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
import socket
import subprocess
import sys
from tempfile import TemporaryFile
import threading
import time
from types import SimpleNamespace

import httpx

TOKEN = "synthetic-concurrency-http-credential"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
PREFIX = "DML-HTTP-CONTROL "
DEADLINE = 90
PROGRESS_DEADLINE = 10
CLIENT_KEEPALIVE_SECONDS = 1.0
SERVER_KEEPALIVE_SECONDS = 5.0
ROUTES = {
    "ingest": "/api/remember/receipt", "update": "/api/memory/update/receipt",
    "promote": "/api/memory/promote/receipt", "retire": "/api/memory/retire/receipt",
    "supersede": "/api/memory/supersede/receipt", "retrieve": "/api/recall",
    "retention": "/api/memory/retention/inspect",
}


def wire_request(request):
    arguments = deepcopy(request["kwargs"])
    operation, args = request["operation"], request["args"]
    if args:
        key = ("text" if operation == "ingest" else "sources" if operation == "promote"
               else "query" if operation == "retrieve" else "memory_id")
        arguments[key] = args[0]
    if operation == "retrieve":
        if "prompt" in arguments:
            arguments["query"] = arguments.pop("prompt")
    return ROUTES[operation], arguments


class LoopbackProvider:
    def __init__(self, directory, schema):
        self.directory, self.schema = Path(directory), schema
        self.messages = Queue()
        self.pending = []
        self.process = None
        self.stderr = None

    def __enter__(self):
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith("DML_"):
                environment.pop(name)
        environment.update(DML_API_TOKEN=TOKEN, DML_SKIP_VENV_REEXEC="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        tests = Path(__file__).resolve().parent
        environment["PYTHONPATH"] = os.pathsep.join((str(tests), str(tests.parent)))
        self.stderr = TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(self.directory), str(self.schema)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1, env=environment,
        )

        def read_messages():
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    self.messages.put(json.loads(line[len(PREFIX):]))
            self.messages.put({"event": "eof"})

        self.reader = threading.Thread(target=read_messages, daemon=True)
        self.reader.start()
        try:
            ready = self.receive("ready", timeout=30)
            self.port = ready["port"]
            self.url = f"http://127.0.0.1:{self.port}"
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def diagnostics(self):
        self.stderr.seek(0)
        return self.stderr.read().decode("utf-8", errors="replace")[-12000:]

    def receive(self, event, *, timeout=PROGRESS_DEADLINE):
        deadline = time.monotonic() + timeout
        while True:
            for index, message in enumerate(self.pending):
                if message["event"] == event:
                    return self.pending.pop(index)
            try:
                message = self.messages.get(timeout=max(0.001, deadline - time.monotonic()))
            except Empty as exc:
                raise AssertionError(f"Provider control deadline waiting for {event}: {self.diagnostics()}") from exc
            if message["event"] == "eof":
                raise AssertionError(f"Provider process exited waiting for {event}: {self.diagnostics()}")
            self.pending.append(message)
            if time.monotonic() > deadline:
                raise AssertionError(f"Provider control deadline waiting for {event}")

    def command(self, command, *, timeout=PROGRESS_DEADLINE, **arguments):
        self.process.stdin.write(json.dumps({"command": command, **arguments}) + "\n")
        self.process.stdin.flush()
        return self.receive(command, timeout=timeout)

    @contextmanager
    def client(self, *, timeout=PROGRESS_DEADLINE):
        with httpx.Client(base_url=self.url, headers=HEADERS, timeout=timeout, trust_env=False) as client:
            yield client

    def post(self, request):
        route, body = wire_request(request)
        with self.client() as client:
            return client.post(route, json=body)

    def __exit__(self, exc_type, _exc, _tb):
        if self.process is None:
            return
        failure = None
        try:
            if self.process.poll() is None:
                try:
                    self.command("stop")
                    self.process.wait(timeout=PROGRESS_DEADLINE)
                except BaseException as error:
                    failure = error
                    self.process.kill()
                    self.process.wait(timeout=PROGRESS_DEADLINE)
            if self.process.returncode != 0 and failure is None:
                failure = AssertionError(self.diagnostics())
        finally:
            if exc_type is not None:
                print(f"Synthetic HTTP worker exit={self.process.returncode}: {self.diagnostics()}", file=sys.stderr)
            self.reader.join(timeout=PROGRESS_DEADLINE)
            self.process.stdin.close()
            self.process.stdout.close()
            self.stderr.close()
        if exc_type is None and failure is not None:
            raise failure


async def invoke_http(client, request, *, deadline):
    """Retry only explicit pre-commit ownership rejection within one deadline."""
    from profile_concurrency_fixture import digest

    route, body = wire_request(request)
    attempts = []
    for _index in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Original HTTP campaign deadline expired")
        started = time.monotonic_ns()
        # Transport failures remain failures. Only exact typed ownership
        # rejections may retry: writes did not commit and reads never acquired
        # the authority needed to produce their snapshot.
        response = await asyncio.wait_for(client.post(route, json=body), remaining)
        finished = time.monotonic_ns()
        payload = response.json()
        attempt = {"started_ns": started, "finished_ns": finished,
                   "request_digest": digest(request), "status_code": response.status_code,
                   "error_code": None}
        if response.status_code == 200:
            attempts.append(attempt)
            return {"request": deepcopy(request), "started_ns": attempts[0]["started_ns"],
                    "finished_ns": finished, "result": payload, "attempts": attempts}
        if request["operation"] in {"ingest", "update", "promote", "retire", "supersede"}:
            expected = {"detail": {"code": "receipt_ownership_unavailable", "retry_same_key": True}}
            assert (response.status_code == 503 and payload == expected
                    and type(payload["detail"]["retry_same_key"]) is bool), response.text
        else:
            expected = {"detail": {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}}
            assert request["operation"] == "retrieve" and response.status_code == 503 and payload == expected, response.text
        attempt.update(error_code="store_lock_timeout", error_detail=payload["detail"])
        attempts.append(attempt)
    raise AssertionError("HTTP request did not acknowledge within three bounded attempts")


async def http_history(provider, before, clients):
    from profile_concurrency_fixture import full_client_plan, write_rejected_history

    events = []
    diagnostics = {"failure_stage": "arrival"}
    try:
        provider.command("arrival", clients=clients)
        deadline = time.monotonic() + DEADLINE
        diagnostics["campaign_deadline_ns"] = int(deadline * 1e9)
        diagnostics["failure_stage"] = "campaign"
        limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients,
                             keepalive_expiry=CLIENT_KEEPALIVE_SECONDS)
        async with httpx.AsyncClient(base_url=provider.url, headers=HEADERS, limits=limits,
                                     timeout=DEADLINE, trust_env=False) as client:
            async def caller(index):
                for sequence, request in enumerate(full_client_plan(before, index, clients)):
                    event = await invoke_http(client, request, deadline=deadline)
                    events.append({"client": index, "sequence": sequence, **event})

            callers = [asyncio.create_task(caller(index)) for index in range(clients)]
            try:
                await asyncio.wait_for(asyncio.gather(*callers), max(0, deadline - time.monotonic()))
            except BaseException:
                diagnostics["failure_observed_ns"] = time.monotonic_ns()
                diagnostics["failure_observed_phase"] = "before-sibling-drain"
                # gather propagates a first error without cancelling its siblings.
                # Drain them while the client remains open, retaining the original
                # failure rather than manufacturing client-closed transport errors.
                for task in callers:
                    task.cancel()
                await asyncio.gather(*callers, return_exceptions=True)
                raise
        diagnostics["failure_stage"] = "transport-observation"
        released_ns = provider.receive("arrivals_released")["released_ns"]
        diagnostics["released_ns"] = released_ns
        first_response_ns = min(event["finished_ns"] for event in events)
        first_progress = (first_response_ns - released_ns) / 1e9
        diagnostics.update(first_response_ns=first_response_ns, first_progress_seconds=first_progress,
                           failure_stage="first-progress")
        assert 0 <= first_progress <= PROGRESS_DEADLINE, (
            f"HTTP first progress {first_progress:.9f}s is outside [0, {PROGRESS_DEADLINE}]s; "
            f"released_ns={released_ns}, first_response_ns={first_response_ns}, completed_events={len(events)}"
        )
        # Logical operation timing includes pre-release HTTP arrival waiting; this
        # shorter release-to-completion metric must not bound max_operation_seconds.
        # The original absolute campaign deadline bounds both timing domains.
        diagnostics["failure_stage"] = "transport-observation"
        return events, {**provider.command("metrics"), "first_progress_seconds": first_progress,
                        "duration_seconds": (max(event["finished_ns"] for event in events) - released_ns) / 1e9}
    except BaseException as error:
        diagnostics.setdefault("failure_observed_ns", time.monotonic_ns())
        diagnostics.setdefault("failure_observed_phase", "outside-active-callers")
        # Diagnostics are rejected evidence. Capture happens after sibling drain,
        # outside the campaign deadline, and may never replace its original error.
        try:
            counters = provider.command("metrics", timeout=1)
            diagnostics["transport_counters"] = counters
            if counters.get("released_ns") is not None:
                diagnostics["released_ns"] = counters["released_ns"]
        except Exception as capture_error:
            diagnostics["counter_capture_error"] = f"{type(capture_error).__name__}: {capture_error}"[:2000]
        if events:
            diagnostics["first_response_ns"] = min(event["finished_ns"] for event in events)
            if "released_ns" in diagnostics:
                diagnostics["first_progress_seconds"] = (
                    diagnostics["first_response_ns"] - diagnostics["released_ns"]
                ) / 1e9
        try:
            written = write_rejected_history(before, events, clients=clients, transport="http", error=error,
                                             diagnostics=diagnostics)
            if written:
                print(f"Rejected HTTP campaign diagnostic: {json.dumps(written, sort_keys=True)}", file=sys.stderr)
        except Exception as capture_error:
            print(f"Rejected HTTP diagnostic capture failed: {type(capture_error).__name__}: {capture_error}",
                  file=sys.stderr)
        raise


def worker(directory, schema):
    import uvicorn
    from uvicorn.protocols.http.h11_impl import H11Protocol
    from daystrom_dml.provider_server import create_app
    from daystrom_dml.services import receipt_ingestion
    from profile_crash_fixture import CLOCK, make_adapter, supported_config

    # One process-wide test clock, installed before workers start.
    receipt_ingestion.time = SimpleNamespace(time=lambda: CLOCK)
    os.environ["DML_API_TOKEN"] = TOKEN
    adapter = make_adapter(directory, schema)
    config_path = Path(directory).parent / (Path(directory).name + "-http-config.json")
    config_path.write_text(json.dumps(supported_config(directory, schema)), encoding="utf-8")
    app = create_app(config_path=str(config_path), adapter_factory=lambda: adapter)
    output_lock, counters_lock, barrier_lock = threading.Lock(), threading.Lock(), threading.Lock()
    release = threading.Event()
    barrier = {"point": None, "embedding_text": None}
    counters = {"active_workers": 0, "max_active_workers": 0, "worker_threads": set(),
                "active_http": 0, "max_active_http": 0, "arrivals": 0, "target_clients": 0,
                "http_requests": 0, "response_status_counts": {},
                "released_ns": None, "first_response_started_ns": None}

    def publish(event, **data):
        with output_lock:
            print(PREFIX + json.dumps({"event": event, **data}), flush=True)

    def pause(point, **measurements):
        with barrier_lock:
            should_pause = barrier["point"] == point
            if should_pause:
                barrier["point"] = None
        if should_pause:
            publish("blocked", point=point, **measurements)
            if not release.wait(DEADLINE):
                raise TimeoutError("Synthetic HTTP scheduling barrier expired")
        return should_pause

    adapter._journal._fault_hook = pause
    original_embed = adapter.embedder.embed

    def embed(text):
        with barrier_lock:
            blocked_text = barrier["embedding_text"]
        if text == blocked_text:
            pause("embedding")
        return original_embed(text)

    adapter.embedder.embed = embed
    methods = ("ingest_memory_receipted", "update_memory_receipted", "promote_memories_receipted",
               "retire_memory_receipted", "supersede_memory_receipted", "retrieve_context",
               "inspect_memory_retention")

    def count_worker(method):
        def counted(*args, **kwargs):
            with counters_lock:
                counters["active_workers"] += 1
                counters["max_active_workers"] = max(counters["max_active_workers"], counters["active_workers"])
                counters["worker_threads"].add(threading.get_ident())
            try:
                return method(*args, **kwargs)
            finally:
                with counters_lock:
                    counters["active_workers"] -= 1
        return counted

    for name in methods:
        setattr(adapter, name, count_worker(getattr(adapter, name)))

    class ArrivalGate:
        def __init__(self, application):
            self.application = application
            self.target = 0
            self.gate = None
            self.loop = None
            self.shutdown_complete = False

        async def arm(self, clients):
            self.target = clients
            self.gate = asyncio.Event()
            with counters_lock:
                counters.update(arrivals=0, target_clients=clients, max_active_http=0,
                                max_active_workers=0, http_requests=0, response_status_counts={},
                                released_ns=None, first_response_started_ns=None)
                counters["worker_threads"].clear()

        async def __call__(self, scope, receive, send):
            self.loop = asyncio.get_running_loop()
            if scope["type"] != "http":
                async def lifespan_send(message):
                    if message["type"] == "lifespan.shutdown.complete":
                        self.shutdown_complete = True
                    await send(message)
                return await self.application(scope, receive, lifespan_send)
            with counters_lock:
                counters["active_http"] += 1
                counters["http_requests"] += 1
                counters["max_active_http"] = max(counters["active_http"], counters["max_active_http"])
            try:
                if self.target and not self.gate.is_set():
                    with counters_lock:
                        counters["arrivals"] += 1
                        all_arrived = counters["arrivals"] == self.target
                    if all_arrived:
                        released_ns = time.monotonic_ns()
                        with counters_lock:
                            counters["released_ns"] = released_ns
                        publish("arrivals_released", released_ns=released_ns)
                        self.gate.set()
                    await asyncio.wait_for(self.gate.wait(), DEADLINE)
                async def counted_send(message):
                    if message["type"] == "http.response.start":
                        with counters_lock:
                            counts = counters["response_status_counts"]
                            code = str(message["status"])
                            counts[code] = counts.get(code, 0) + 1
                            if counters["first_response_started_ns"] is None:
                                counters["first_response_started_ns"] = time.monotonic_ns()
                    await send(message)
                await self.application(scope, receive, counted_send)
            finally:
                with counters_lock:
                    counters["active_http"] -= 1

    class ControlledIdleProtocol(H11Protocol):
        """Test-only scheduling hook; the real five-second timer owns closing."""

        def timeout_keep_alive_handler(self):
            paused = pause("idle_close", peer=list(self.client), timer_fired_ns=time.monotonic_ns(),
                           server_timeout_seconds=self.timeout_keep_alive)
            close_ns = time.monotonic_ns()
            super().timeout_keep_alive_handler()
            if paused:
                publish("idle_closed", peer=list(self.client), close_started_ns=close_ns)

    gated = ArrivalGate(app)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2048)
    server = uvicorn.Server(uvicorn.Config(gated, log_level="error", access_log=False, http=ControlledIdleProtocol,
                                         timeout_keep_alive=SERVER_KEEPALIVE_SECONDS,
                                         timeout_graceful_shutdown=PROGRESS_DEADLINE, lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Loopback provider did not start")
        threading.Event().wait(0.005)
    publish("ready", port=listener.getsockname()[1])
    try:
        for raw in sys.stdin:
            command = json.loads(raw)
            name = command["command"]
            if name == "arm":
                release.clear()
                with barrier_lock:
                    barrier.update(point=command["point"], embedding_text=command.get("embedding_text"))
                publish(name)
            elif name == "release":
                release.set()
                publish(name)
            elif name == "arrival":
                asyncio.run_coroutine_threadsafe(gated.arm(command["clients"]), gated.loop).result(timeout=10)
                publish(name)
            elif name == "metrics":
                with counters_lock:
                    values = {**counters, "worker_threads": len(counters["worker_threads"]),
                              "response_status_counts": dict(counters["response_status_counts"])}
                publish(name, **values)
            elif name == "stop":
                release.set()
                if gated.gate is not None:
                    gated.loop.call_soon_threadsafe(gated.gate.set)
                server.should_exit = True
                thread.join(timeout=PROGRESS_DEADLINE)
                if thread.is_alive() or not gated.shutdown_complete:
                    raise RuntimeError("Provider failed bounded graceful shutdown")
                publish(name)
                break
            else:
                raise ValueError("Unknown synthetic control message")
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=PROGRESS_DEADLINE)
        listener.close()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "worker":
        raise SystemExit("Test helper requires worker DIRECTORY SCHEMA")
    worker(Path(sys.argv[2]), int(sys.argv[3]))
