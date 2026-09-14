# Bounded projection retry scheduling and worker ownership

This serial gate adds explicit background delivery to the accepted coalesced
projection service. The receipt journal remains authoritative. SQLite remains the
qualified projection backend; neither persisted schema changes. No backend work
is added to receipt ingestion, and no worker starts implicitly.

## Scheduling contract

`ProjectionWorker(source, backend, poll_interval=1.0, retry_initial=0.1,
retry_max=30.0)` is configured with keyword-only timing arguments. Call `start()`
to begin immediate reconciliation, then periodic reconciliation. One worker admits
at most one reconciliation at a time. `request_sync()` coalesces notifications into
one pending flag. There is no unbounded task queue or retained mutation payload.
Intervals must be finite numbers from 1 ms through `threading.TIMEOUT_MAX`, with
the retry cap at least the initial delay. Shutdown accepts zero through that same
maximum. Invalid values are rejected before changing lifecycle state.

Each attempt regenerates a delta from verified source and target state. Recoverable
I/O or contention failures use capped exponential backoff. Wake requests cannot
bypass that delay. A successful reconciliation resets the failure delay. Requests
arriving during an attempt can cause one subsequent pass; periodic polling also
discovers source commits from other adapters and processes. The worker does not
require ingestion callbacks or in-memory notifications for eventual catch-up.

Integrity, schema and compatibility failures halt delivery with `state="faulted"`.
Unknown errors also halt conservatively. The worker does not delete damaged state,
reset identity markers or infer that unreadable authority is empty. Preserve and
repair the cause, then create a new worker. Journal errors already classified as
recovery-required retain that classification.

The retry policy accepts `OSError` (including timeout/connection errors), revision
conflicts, stale projection bases, and SQLite resource/contention operational
errors. SQLite syntax/schema/unknown operational errors halt. On Python 3.10,
where native error codes are unavailable, only exact known resource/contention
messages qualify for SQLite retry. The worker never exposes those messages in
telemetry. Backend implementations must use the documented error types; an
`OSError` caused by a persistent configuration problem still retries until close.

Backoff bounds retry frequency and scheduling memory, not attempt count, backend
latency, resource usage inside a backend, or aggregate retries across processes.
An unavailable backend may be retried indefinitely at the capped frequency until
the worker closes. Independent workers still use accepted publication CAS and
cursor checks. There is no fleet-wide rate limiter or distributed ownership lease.

## Terminal shutdown

`start()` is idempotent for an active worker. `close(timeout=5.0)` permanently closes
admission and interrupts scheduled waits. It returns `True` only when the worker
has drained; `False` means an admitted reconciliation is still running. Retry close
after the backend returns. A closed or faulted worker cannot restart, and rejected
wake requests return `False`.

Python cannot cancel an arbitrary blocked backend. One already admitted
reconciliation, including its remaining backend reads and publication, can finish
after close starts. Use backend I/O deadlines where bounded completion is required.
No subsequent reconciliation is admitted. The service never holds source mutation
ownership or its scheduling condition during backend I/O.
Backend exception classification also runs outside that condition and remains part
of the admitted attempt. A raising classifier faults the worker; a blocked
classifier can delay drain but cannot prevent a bounded close from reporting it.

## Adapter integration

Receipt-mode adapters expose `start_projection_worker(backend, ...)`, returning the
owned worker for status and wake requests. Only one backend instance can be owned
by an adapter. Repeated starts for that instance retain the initial settings;
switching backends requires a new adapter. Starting from a nested adapter mutation
or a non-receipt configuration is rejected.

`adapter.close(persist=False, projection_timeout=5.0)` closes worker admission before
draining it and before closing its dependencies. A blocked worker raises
`TimeoutError`; dependencies remain available, and close can be retried after drain.
Concurrent start cannot install a worker after adapter shutdown begins. Directly
constructed workers are caller-owned and must be closed separately.

```python
from daystrom_dml.services.projection import SQLiteProjection

backend = SQLiteProjection(projection_path)  # separate directory from authority
worker = adapter.start_projection_worker(
    backend, poll_interval=1.0, retry_initial=0.1, retry_max=30.0,
)
receipt = adapter.ingest_memory_receipted(
    "My preferred timezone is UTC", tenant_id="example", idempotency_key="tz-1",
)
worker.request_sync()  # optional; periodic polling also sees the committed receipt
status = worker.status()
adapter.close(persist=False, projection_timeout=5.0)
```

## Observability and recovery

`status()` returns a detached snapshot of lifecycle state, in-flight/pending flags,
attempt/success/failure counters, sanitized exception type, retry countdown and the
last completed reconciliation report. That report describes historical pinned
source/target observations. It is not a promise that the target is currently fresh;
use `projection_status()` or the fail-closed query API for a new verified comparison.
Monotonic completion timestamps are process-local. Runtime counters and schedules
are not durable audit records, and raw backend exception messages are not exposed.

A process death loses only the schedule and pending flag. A new worker compares
durable verified cursors on its first pass. The crash harness kills actual worker
processes at all seven journal publication boundaries, verifies unchanged authority
and atomic old-or-new target state, then restarts delivery and replays the original
receipt without creating another source revision.

This gate does not implement an operation outbox, qualify FAISS, route model context
automatically, compact history, prove growing-store performance, or complete the
ten-area production plan. Full snapshot/history verification remains a cost of
every reconciliation. The maturity inventory remains candidate-only.


## Independent acceptance and retained harness

The independent grader rejected the initial revision at **9.3/10**: exception
formatting could raise and leave a dead worker appearing active, or block bounded
close while holding the lifecycle condition. Exception classification and type
introspection now run outside that condition while the attempt remains in flight.
Failures fault the worker with sanitized telemetry; a blocked classifier leaves
close and status responsive. Timeout validation also precedes lifecycle changes.
Final independent grade: **9.5/10**, with no unresolved blocking findings.

The focused harness has **105 passing cases**: 79 worker scheduling/lifecycle
cases, 15 adapter integrations, seven real process-kill/recovery cases and four
independent adversarial cases. It covers 256 coalesced start/wake requests, sustained
concurrent retry storms, simultaneous close/start, actual authority corruption,
receipt replay after outage/restart, historical status isolation, and raising or
blocking exception formatters, properties and metaclasses. CI includes all four
worker test files in portability and production-evidence selections.

Final local integration: **1,318 passed**, nine optional-dependency skips, two
known warnings, in 56.55 seconds. Ruff, 48 maintained mypy files (local Python
target 3.12) and Hermes hygiene passed. The
[review record](artifacts/projection-worker-review-2026-09-14.json) pins accepted
source hashes and preserves the rejected/accepted review history. Exact-commit
remote CI remains a separate integration gate recorded in the PR.
