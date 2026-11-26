"""Independent submitted-intent histories for the supported local profile.

This is test infrastructure, not a production history/replay implementation.
The small fixed working set separates concurrency qualification from the later
growing-store campaign. No runtime digest, selector, or receipt builder defines
an expected result.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace
from unittest.mock import patch

from profile_crash_fixture import (
    CLOCK, SCOPE, Request, assert_transition, digest, encoded,
    expected_transition, invoke, prepare, sql_observation,
)

WAIT_SECONDS = 90
PROGRESS_SECONDS = 10
_HISTORY_SETTING = os.environ.get("DML_CONCURRENCY_HISTORY_DIR")
HISTORY_DIRECTORY = Path(_HISTORY_SETTING).resolve() if _HISTORY_SETTING is not None else None
MUTATIONS = frozenset({"ingest", "update", "promote", "supersede", "retire"})
LATTICE_DERIVED_FIELDS = frozenset({
    "lattice_row", "lattice_col", "lattice_layer", "lattice_id", "lattice_policy",
    "lattice_index", "lattice_size_hint", "lattice_neighbors", "lattice_degree",
})


def requested_scales():
    values = tuple(int(part) for part in os.environ.get("DML_CONCURRENCY_SCALES", "1,16").split(","))
    if not values or len(set(values)) != len(values) or any(value not in (1, 16, 64, 256) for value in values):
        raise ValueError("DML_CONCURRENCY_SCALES must select distinct values from 1,16,64,256")
    return values


@contextmanager
def fixed_clock():
    """Install once before starting threads; per-call patch stacks would race."""
    with patch("daystrom_dml.services.receipt_ingestion.time", SimpleNamespace(time=lambda: CLOCK)):
        yield


def prepare_history(directory, schema):
    from profile_crash_fixture import make_adapter

    scenario = prepare(directory, schema)
    before = scenario.before
    adapter = make_adapter(directory, schema)
    try:
        # IDs 0/1/2 are the independently checked source/replacement/foreign
        # records from the accepted crash fixture. Each mutable target is disjoint.
        for name in ("replacement", "promotion source", "retirement target", "retention target"):
            scope = dict(SCOPE)
            if name != "replacement":
                scope["client_id"] = name.split()[0]
            request = Request("ingest", (f"Concurrent {name}",), {
                **scope, "idempotency_key": f"seed-{name}", "meta": {"source_trust": "trusted"},
            })
            receipt = invoke(adapter, request)
            after = sql_observation(directory)
            assert receipt == assert_transition(before, after, request)
            before = after
    finally:
        adapter.close()
    return before


def _request(operation, args=(), **kwargs):
    return {"operation": operation, "args": list(args), "kwargs": kwargs}


def client_plan(before, index):
    records = {record["id"]: record for record in before["state"]["items"]}
    family = index % 8
    scope = dict(SCOPE)
    if family in (2, 3, 6):
        scope["client_id"] = {2: "promotion", 3: "retirement", 6: "retention"}[family]
    common = {**scope, "reason": "Synthetic concurrency qualification"}
    if family == 0:
        scope["client_id"] = f"unique-client-{index}"
        request = _request("ingest", [f"Unique concurrent fact {index}"], **scope,
                           idempotency_key=f"unique-{index}", kind="memory", meta={"source_trust": "trusted"})
    elif family == 1:
        request = _request("update", [0], **common, idempotency_key="shared-update",
                           expected_memory_digest=digest(records[0]), text="Updated concurrent fact")
    elif family == 2:
        request = _request("promote", [[{"memory_id": 4, "expected_memory_digest": digest(records[4])}]],
                           **common, idempotency_key="shared-promotion", text="Approved concurrent interpretation")
    elif family == 3:
        request = _request("retire", [5], **common, idempotency_key="shared-retirement",
                           expected_memory_digest=digest(records[5]))
    elif family == 4:
        request = _request("supersede", [1], **common, idempotency_key="shared-supersession",
                           expected_memory_digest=digest(records[1]), replacement_memory_id=3,
                           expected_replacement_digest=digest(records[3]))
    elif family == 5:
        return [_request("retrieve", ["Concurrent facts"], **scope, top_k=10)]
    elif family == 6:
        return [_request("retention", [6], **scope)]
    else:
        request = _request("ingest", ["Shared concurrent fact"], **scope,
                           idempotency_key="shared-append", kind="memory", meta={"source_trust": "trusted"})
    return [request, deepcopy(request),
            _request("retrieve", ["Concurrent facts"], **scope, top_k=10)]


def full_client_plan(before, client, clients):
    """One caller at scale one still executes every operation family."""
    return [request for index in range(client, max(clients, 8), clients)
            for request in client_plan(before, index)]


def invoke_operation(adapter, request):
    operation = request["operation"]
    if operation == "retrieve":
        method = adapter.retrieve_context
    elif operation == "retention":
        method = adapter.inspect_memory_retention
    else:
        method = getattr(adapter, Request(operation, (), {}).method)
    return method(*deepcopy(request["args"]), **deepcopy(request["kwargs"]))


def call_event(adapter, request, *, client, sequence, deadline_ns=None):
    from daystrom_dml.store_lock import StoreLockTimeout

    attempts = []
    for number in range(3 if deadline_ns is not None else 1):
        started = monotonic_ns()
        if deadline_ns is not None and started >= deadline_ns:
            raise TimeoutError("Qualification campaign deadline exhausted")
        try:
            result = invoke_operation(adapter, request)
        except StoreLockTimeout:
            attempts.append({"started_ns": started, "finished_ns": monotonic_ns(),
                             "request_digest": digest(request), "error_code": "store_lock_timeout"})
            if deadline_ns is None or number == 2 or monotonic_ns() >= deadline_ns:
                raise
        else:
            attempts.append({"started_ns": started, "finished_ns": monotonic_ns(),
                             "request_digest": digest(request), "error_code": None})
            return {"client": client, "sequence": sequence, "request": deepcopy(request),
                    "started_ns": attempts[0]["started_ns"], "finished_ns": attempts[-1]["finished_ns"],
                    "attempts": attempts, "result": result}
    raise AssertionError("Unreachable retry state")


def _binding(request):
    kwargs = request["kwargs"]
    return encoded([{key: kwargs.get(key) for key in SCOPE}, kwargs["idempotency_key"]])


def _within_revision(event, revision, windows, initial):
    assert initial <= revision <= initial + len(windows), "Read named an unknown revision"
    for committed_revision, (start, finish) in windows.items():
        if finish < event["started_ns"]:
            assert committed_revision <= revision, "Read missed a previously acknowledged commit"
        if start > event["finished_ns"]:
            assert committed_revision > revision, "Read included an uninvoked future commit"


def _assert_retrieval(event, view):
    report = event["result"]
    request = event["request"]
    scope = {key: request["kwargs"].get(key) for key in SCOPE}
    expected = {str(record["id"]): record for record in view["state"]["items"]
                if all(record["meta"].get(key) == value for key, value in scope.items())
                and record["meta"].get("memory_state", "active") == "active"}
    assert len(expected) <= 4, "Working set exceeded exact-read oracle bound"
    assert len(report["items"]) == len(expected), "Missing or duplicate retrieved memory"
    assert {entry["id"] for entry in report["items"]} == set(expected), "Dirty read or scope leakage"
    # All synthetic vectors, timestamps, salience and fidelity are identical:
    # the documented exact-ranking tie break is ascending memory ID.
    ordered_ids = sorted(expected, key=int)
    assert [entry["id"] for entry in report["items"]] == ordered_ids
    for entry in report["items"]:
        record = expected[entry["id"]]
        assert entry["text"] == record["text"], "Read content disagrees with its pinned revision"
        assert entry["summary"] == record["text"]
        # Hydration adds a derived in-memory lattice layout. Durable metadata,
        # including every scope/provenance/lifecycle field, must remain exact.
        assert {key: value for key, value in entry["meta"].items()
                if key not in LATTICE_DERIVED_FIELDS} == record["meta"]
        for field in ("timestamp", "level", "fidelity", "salience"):
            assert entry[field] == record[field]
    decision = report["decision"]
    expected_context = "\n".join(["=== Retrieved Context ===", *[
        f"- (2023-11-14) [source=unknown]\n  {expected[ident]['text']}" for ident in ordered_ids
    ]]) if expected else ""
    assert report["raw_context"] == expected_context, "Rendered context disagrees with committed records"
    assert report["top_k"] == request["kwargs"]["top_k"]
    assert decision["scope"] == scope
    assert decision["suppressed"] == [
        {"id": str(record["id"]), "reason": "state_" + record["meta"]["memory_state"]}
        for record in view["state"]["items"]
        if all(record["meta"].get(key) == value for key, value in scope.items())
        and record["meta"].get("memory_state") in {"deleted", "superseded"}
    ]
    assert decision["returned_ids"] == [entry["id"] for entry in report["items"]]
    assert decision["source_digests"] == [digest(entry["meta"]) for entry in report["items"]]
    assert decision["context_digest"] == digest(report["raw_context"])
    assert decision["decision_digest"] == digest({key: value for key, value in decision.items()
                                                if key != "decision_digest"})
    assert "PRIVATE_FOREIGN_RECOVERY_SENTINEL" not in report["raw_context"]


def _assert_retention(event, view):
    report = event["result"]
    request = event["request"]
    scope = {key: request["kwargs"].get(key) for key in SCOPE}
    ident = request["args"][0]
    assert report["schema_version"] == "dml-retention-report-v1"
    assert report["request"] == {"schema_version": "dml-retention-request-v1", "memory_id": ident, "scope": scope}
    assert report["coverage"] == "known_structured_references_in_one_journal"
    assert report["source"] == {"store_id": view["store_id"], "journal_schema_version": view["schema"],
                                "revision": view["revision"], "state_digest": digest(view["state"])}
    groups = {"current_items": view["state"]["items"], "current_lineage": view["state"]["lineage"],
              "journal_snapshot": [item for bucket in ("items", "lineage") for item in view["snapshot"][bucket]],
              "receipts": [receipt["result"]["memory"] for receipt in view["receipts"]],
              "outbox_states": [item for event in view["outbox"] for bucket in ("items", "lineage")
                                for item in event["state"][bucket]]}
    expected = {}
    for surface, records in groups.items():
        scoped = [record for record in records if all(record["meta"].get(key) == value
                                                      for key, value in scope.items())]
        expected[surface] = {
            "direct_records": sum(record["id"] == ident for record in scoped),
            "embedded_source_records": sum(source["memory"]["id"] == ident for record in scoped
                for source in record["meta"].get("promotion_decision", {}).get("sources", [])),
        }
    assert report["surfaces"] == expected
    assert report["known_reference_count"] == sum(sum(counts.values()) for counts in expected.values())
    assert report["erasure_proven"] is False
    assert report["physical_erasure_supported"] is False
    assert report["retirement_is_erasure"] is False


def _assert_recorded_causality(events, initial_revision):
    """Preserve recorded program order even when clock readings are equal.

    A receipt may be a historical replay. Its lower revision must not be
    mistaken for a new commit, but it still provides a floor for later reads.
    A commit is necessarily later only if every successful attempt that could
    have created it is known to follow the observation being checked.
    """
    commits = {}
    clients = {}

    def revision(event):
        operation = event["request"]["operation"]
        if operation == "retrieve":
            return event["result"]["decision"]["store_revision"]
        if operation == "retention":
            return event["result"]["source"]["revision"]
        return event["result"]["revision"]

    for event in events:
        clients.setdefault(event["client"], []).append(event)
        if event["request"]["operation"] in MUTATIONS:
            commits.setdefault(revision(event), []).append(event)
    for event in events:
        observed_revision = revision(event)
        for committed_revision, candidates in commits.items():
            if all((candidate["client"] == event["client"] and candidate["sequence"] > event["sequence"])
                   or candidate["attempts"][-1]["started_ns"] > event["finished_ns"]
                   for candidate in candidates):
                assert observed_revision < committed_revision, "Observed revision includes a causally later commit"
    for history in clients.values():
        floor = initial_revision
        for event in sorted(history, key=lambda event: event["sequence"]):
            observed_revision = revision(event)
            if event["request"]["operation"] not in MUTATIONS:
                assert observed_revision >= floor, "Read moved behind its client's prior observation"
            floor = max(floor, observed_revision)


def check_history(before, events, after, *, clients):
    """Check a finite completed history against submitted intent and direct SQL.

    Receipt revision fixes commit order. Earliest invocation and earliest
    successful acknowledgement bound a first commit; later historical retries
    must not incorrectly move that commit's interval. This checks the recorded
    schedule, not arbitrary-schedule formal linearizability.
    """
    assert type(clients) is int and clients in (1, 16, 64, 256)
    expected_requests = {(client, sequence): request for client in range(clients)
                         for sequence, request in enumerate(full_client_plan(before, client, clients))}
    assert len(events) == len(expected_requests), "Missing or duplicated acknowledgement"
    observed = {}
    mutations = {}
    ownership_rejections = 0
    max_attempts = 0
    read_events = []
    for event in events:
        key = (event["client"], event["sequence"])
        assert key not in observed, "Duplicated client event"
        assert key in expected_requests and event["request"] == expected_requests[key], "Request ledger mismatch"
        assert type(event["started_ns"]) is int and type(event["finished_ns"]) is int
        assert 0 < event["started_ns"] <= event["finished_ns"]
        attempts = event["attempts"]
        assert type(attempts) is list and 1 <= len(attempts) <= 3, "Invalid bounded attempt count"
        assert event["started_ns"] == attempts[0]["started_ns"], "An earlier attempt was erased"
        assert event["finished_ns"] == attempts[-1]["finished_ns"]
        for number, attempt in enumerate(attempts):
            assert type(attempt["started_ns"]) is int and type(attempt["finished_ns"]) is int
            assert 0 < attempt["started_ns"] <= attempt["finished_ns"]
            if number:
                assert attempts[number - 1]["finished_ns"] <= attempt["started_ns"], "Retry attempts overlap"
            assert attempt["request_digest"] == digest(event["request"]), "Retry changed the submitted request"
            final = number == len(attempts) - 1
            assert attempt["error_code"] == (None if final else "store_lock_timeout"), "Unapproved retry outcome"
            fields = {"started_ns", "finished_ns", "request_digest", "error_code"}
            if "status_code" in attempt:
                fields.add("status_code")
                assert type(attempt["status_code"]) is int
                assert attempt["status_code"] == (200 if final else 503)
                if not final:
                    fields.add("error_detail")
                    operation = event["request"]["operation"]
                    if operation in MUTATIONS:
                        expected_detail = {"code": "receipt_ownership_unavailable", "retry_same_key": True}
                    else:
                        assert operation == "retrieve", "Endpoint has no qualified HTTP ownership rejection"
                        expected_detail = {"code": "retrieval_outcome_unavailable", "reason": "store_ownership_timeout"}
                    assert type(attempt["error_detail"]) is dict and attempt["error_detail"] == expected_detail, \
                        "HTTP ownership rejection is not positively identified"
                    if operation in MUTATIONS:
                        assert attempt["error_detail"]["retry_same_key"] is True
            assert set(attempt) == fields, "Attempt envelope has unsupported or missing fields"
        ownership_rejections += len(attempts) - 1
        max_attempts = max(max_attempts, len(attempts))
        observed[key] = event
        if event["request"]["operation"] in MUTATIONS:
            mutations.setdefault(_binding(event["request"]), []).append(event)
    assert set(observed) == set(expected_requests)
    assert (max(event["finished_ns"] for event in events) - min(event["started_ns"] for event in events)) / 1e9 <= WAIT_SECONDS, \
        "History exceeded the original campaign deadline"
    for client in range(clients):
        ordered = [observed[(client, sequence)] for sequence in range(len(full_client_plan(before, client, clients)))]
        assert all(first["finished_ns"] <= second["started_ns"] for first, second in zip(ordered, ordered[1:])), \
            "One client returned an impossible program order"
    unique = []
    windows = {}
    for attempts in mutations.values():
        receipt = attempts[0]["result"]
        assert all(attempt["result"] == receipt for attempt in attempts), "Historical retry changed receipt"
        revision = receipt["revision"]
        assert revision not in windows, "Different requests share a commit revision"
        windows[revision] = (min(event["attempts"][-1]["started_ns"] for event in attempts),
                             min(event["attempts"][-1]["finished_ns"] for event in attempts))
        unique.append((revision, attempts[0]["request"], receipt))
    unique.sort(key=lambda item: item[0])
    assert [item[0] for item in unique] == list(range(before["revision"] + 1,
                                                    before["revision"] + 1 + len(unique)))
    assert len(unique) <= 40, "Concurrency campaign exceeded frozen mutation budget"
    earliest = 0
    for revision, _, _ in unique:
        start, finish = windows[revision]
        earliest = max(earliest, start)
        assert earliest <= finish, "Commit revisions violate real-time precedence"
    views = {before["revision"]: deepcopy(before)}
    previous = views[before["revision"]]
    for revision, raw, receipt in unique:
        request = Request(raw["operation"], tuple(raw["args"]), raw["kwargs"])
        expected = expected_transition(previous, request)
        assert receipt == expected["receipt"], "Receipt is not the submitted state transition"
        view = {**deepcopy(previous), "revision": revision,
                "state": expected["state"], "snapshot": deepcopy(expected["state"]),
                "receipts": [*previous["receipts"], receipt],
                "decisions": [entry for entry in after["decisions"] if entry["revision"] <= revision],
                "outbox": [entry for entry in after["outbox"] if entry["source_revision"] <= revision]}
        assert_transition(previous, view, request)
        views[revision] = view
        previous = view
    assert encoded(after) == encoded(previous), "Final authority diverged from acknowledged history"
    for event in events:
        event = {**event, "started_ns": event["attempts"][-1]["started_ns"]}
        operation = event["request"]["operation"]
        if operation == "retrieve":
            revision = event["result"]["decision"]["store_revision"]
            _within_revision(event, revision, windows, before["revision"])
            _assert_retrieval(event, views[revision])
            read_events.append((event, revision))
        elif operation == "retention":
            revision = event["result"]["source"]["revision"]
            _within_revision(event, revision, windows, before["revision"])
            _assert_retention(event, views[revision])
            read_events.append((event, revision))
    for earlier, earlier_revision in read_events:
        for later, later_revision in read_events:
            if earlier["finished_ns"] < later["started_ns"]:
                assert earlier_revision <= later_revision, "Nonoverlapping reads moved backwards in revision"
    _assert_recorded_causality(events, before["revision"])
    # Find a time-ordered placement of all observed commits and pinned reads.
    # Reads at one revision can commute; earliest deadline ordering gives a
    # feasible placement if one exists. A later revision cannot precede them.
    earliest = 0
    for revision in range(before["revision"], after["revision"] + 1):
        if revision in windows:
            start, finish = windows[revision]
            earliest = max(earliest, start)
            assert earliest <= finish, "Commit/read intervals have no consistent revision order"
        same_revision = sorted((event for event, observed_revision in read_events
                                if observed_revision == revision), key=lambda event: event["finished_ns"])
        for event in same_revision:
            earliest = max(earliest, event["started_ns"])
            assert earliest <= event["finished_ns"], "Read intervals have no consistent revision order"
    return {"history_verified": True, "history_events": len(events), "unique_commits": len(unique),
            "ownership_rejections": ownership_rejections, "max_attempts": max_attempts,
            "max_operation_seconds": max(event["finished_ns"] - event["started_ns"] for event in events) / 1e9}


def write_evidence(record_property, evidence):
    """JUnit carries only synthetic metrics; full events remain test-local."""
    record_property("concurrency_evidence", json.dumps(evidence, sort_keys=True, separators=(",", ":")))


def write_history(before, events, after, *, clients, transport):
    """Save bounded synthetic qualification evidence only when requested."""
    if HISTORY_DIRECTORY is None:
        return {}
    assert transport in {"threads-shared", "threads-separate", "processes", "http"}
    filename = f"schema{before['schema']}-{transport}-{clients}.json.gz"
    envelope = {"schema_version": "dml-concurrency-history-v1", "before": before,
                "events": events, "after": after, "clients": clients}
    raw = encoded(envelope).encode("utf-8")
    assert len(raw) <= 32 * 1024 * 1024, "History exceeded qualification artifact bound"
    compressed = gzip.compress(raw, mtime=0)
    assert len(compressed) <= 8 * 1024 * 1024
    directory = Path(HISTORY_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(compressed)
    return {"history_file": filename, "history_sha256": hashlib.sha256(compressed).hexdigest()}


def write_rejected_history(before, events, *, clients, transport, error, diagnostics, after=None):
    """Preserve bounded failure observations without producing acceptance evidence.

    Callers retain and re-raise the primary failure even if this best-effort
    writer fails. Existing files are never overwritten or followed as symlinks.
    """
    if HISTORY_DIRECTORY is None:
        return {}
    if (transport not in {"threads-shared", "threads-separate", "processes", "http"}
            or type(clients) is not int or clients not in (1, 16, 64, 256)
            or type(before.get("schema")) is not int or before["schema"] not in (2, 3, 4)):
        raise ValueError("Invalid rejected diagnostic identity")
    if not isinstance(error, BaseException) or type(diagnostics) is not dict:
        raise ValueError("Rejected diagnostics require a failure and observed fields")
    envelope = {"schema_version": "dml-concurrency-rejected-diagnostic-v1", "accepted": False,
                "clients": clients, "transport": transport, "before": before, "events": events,
                "after": after, "failure_type": type(error).__name__, "failure_message": str(error)[:2000],
                "diagnostics": diagnostics}
    raw = encoded(envelope).encode("utf-8")
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Rejected history exceeds the 32 MiB raw bound")
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > 8 * 1024 * 1024:
        raise ValueError("Rejected history exceeds the 8 MiB compressed bound")
    root = Path(HISTORY_DIRECTORY).parent.resolve()
    directory = root / "diagnostics"
    if directory.is_symlink():
        raise ValueError("Rejected diagnostic directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve().parent != root:
        raise ValueError("Rejected diagnostic path escapes its artifact directory")
    filename = f"schema{before['schema']}-{transport}-{clients}.rejected.json.gz"
    with (directory / filename).open("xb") as output:
        output.write(compressed)
    return {"diagnostic_file": f"diagnostics/{filename}",
            "diagnostic_sha256": hashlib.sha256(compressed).hexdigest()}
