"""Seeded public-profile recovery histories with a parent-owned state oracle.

The fixed seeds select interleavings, not the expected outcome. Every campaign
contains acknowledged operations, lost acknowledgements on both sides of commit,
rejected requests and historical retries after later lifecycle transitions.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from copy import deepcopy
import hashlib
import json
import os
import random
import sqlite3

import pytest

from daystrom_dml.contracts.profile import ProductionProfileError
from daystrom_dml.journal import IdempotencyConflict, JournalIntegrityError
from daystrom_dml.services.receipt_ingestion import ReceiptCommitRejected, ReceiptEmbeddingError
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict
from profile_crash_fixture import (
    Request, SCOPE, expected_transition, invoke, make_adapter, prepare,
    request_for, run_interrupted, sql_observation,
)


SEEDS = (1729, 314159, 7)
SCHEMAS = (2, 3, 4)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _request(operation, *args, key, **kwargs):
    return Request(operation, args, {**SCOPE, "idempotency_key": key, **kwargs})


def _reject_without_change(directory, schema, request, exception):
    before = sql_observation(directory)
    instance = make_adapter(directory, schema)
    try:
        with pytest.raises(exception):
            invoke(instance, request)
    finally:
        instance.close(persist=False)
    assert sql_observation(directory) == before


def _assert_history(observed, baseline, state, receipts, operations):
    """Compare the cumulative history without adopting recovered records."""
    assert observed["store_id"] == baseline["store_id"]
    assert observed["schema"] == baseline["schema"]
    assert observed["revision"] == baseline["revision"] + len(receipts)
    assert observed["state"] == state
    assert observed["snapshot"] == state
    assert observed["receipts"] == baseline["receipts"] + receipts
    assert observed["decisions"][:len(baseline["decisions"])] == baseline["decisions"]
    decisions = observed["decisions"][len(baseline["decisions"]):]
    assert len(decisions) == len(receipts)
    for decision, receipt, operation in zip(decisions, receipts, operations):
        assert decision["operation"] == operation
        assert decision["revision"] == receipt["revision"]
        assert decision["receipt"] == {
            "scope": receipt["scope"], "key": receipt["key"],
            "request_digest": receipt["request_digest"], "digest": digest(receipt),
        }
    if decisions:
        assert decisions[-1]["state_digest"] == digest(state)
    assert observed["outbox"][:len(baseline["outbox"])] == baseline["outbox"]
    events = observed["outbox"][len(baseline["outbox"]):]
    assert len(events) == (len(receipts) if baseline["schema"] in (3, 4) else 0)
    for event, receipt, operation in zip(events, receipts, operations):
        assert event["source_revision"] == receipt["revision"]
        assert event["operation"] == operation
        assert event["receipt"] == {
            "scope": receipt["scope"], "key": receipt["key"],
            "request_digest": receipt["request_digest"], "digest": digest(receipt),
        }
    if events:
        assert events[-1]["state"] == state


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_seeded_mixed_profile_history_survives_repeated_process_death(tmp_path, schema, seed):
    rng = random.Random(seed)
    scenario = prepare(tmp_path / "authority", schema)
    baseline = deepcopy(scenario.before)
    state = deepcopy(baseline["state"])
    receipts, operations, requests = [], [], []
    # All five operations participate in the kill schedule across the fixed
    # seeds. Each run has two uncommitted and two committed lost responses.
    crash_indices = sorted(rng.sample(range(12), 4))
    precommit = ["after_records", "after_state", "after_snapshot", "after_receipt", "before_commit"]
    if schema in (3, 4):
        precommit.append("after_outbox")
    crash_points = dict(zip(crash_indices, [rng.choice(precommit), "after_commit",
                                          rng.choice(precommit), "before_response"]))
    named = {}

    def submit(request, *, remember=None):
        nonlocal state
        index = len(receipts)
        expected = expected_transition(scenario.before, request)
        before = deepcopy(scenario.before)
        point = crash_points.get(index)
        if point is not None:
            assert run_interrupted(scenario, request, point) is None
            recovered = sql_observation(scenario.directory)
            committed = point in {"after_commit", "before_response"}
            if not committed:
                assert recovered == before
            else:
                _assert_history(recovered, baseline, expected["state"],
                                receipts + [expected["receipt"]],
                                operations + [expected["operation"]])
            instance = make_adapter(scenario.directory, schema, unavailable=committed)
        else:
            instance = make_adapter(scenario.directory, schema)
        try:
            actual = invoke(instance, request)
            assert actual == expected["receipt"]
        finally:
            instance.close(persist=False)
        state = expected["state"]
        receipts.append(deepcopy(expected["receipt"]))
        operations.append(expected["operation"])
        requests.append(deepcopy(request))
        observed = sql_observation(scenario.directory)
        _assert_history(observed, baseline, state, receipts, operations)
        scenario.before = observed
        if remember is not None:
            named[remember] = actual["result"]["memory"]["id"]
        return actual["result"]["memory"]

    def record(name):
        return next(item for item in state["items"] if item["id"] == named[name])

    def key(label):
        return f"campaign-{seed}-{label}"

    def ingest(name):
        return submit(_request("ingest", f"seed {seed}: remembered {name}", key=key(name),
                               meta={"source_trust": "trusted"}), remember=name)

    def update(name):
        return submit(_request("update", named[name], text=f"seed {seed}: corrected {name}",
                               expected_memory_digest=digest(record(name)),
                               reason="Owner correction", key=key(f"update-{name}")))

    def promote(name):
        return submit(_request("promote", [{"memory_id": named[name],
                               "expected_memory_digest": digest(record(name))}],
                               text=f"seed {seed}: derived {name}", reason="Owner approved summary",
                               key=key(f"promote-{name}")), remember=f"derived-{name}")

    def retire(name):
        return submit(_request("retire", named[name], expected_memory_digest=digest(record(name)),
                               reason="Owner withdrew fact", key=key(f"retire-{name}")))

    ingest("a")
    ingest("b")
    ingest("c")
    original_a = deepcopy(record("a"))
    update("a")
    _reject_without_change(scenario.directory, schema, _request("retire", named["a"],
        expected_memory_digest=digest(original_a), reason="Stale decision",
        key=key("stale-cas")), ReceiptLifecycleConflict)
    promote("a")
    submit(_request("supersede", named["a"], replacement_memory_id=named["b"],
        expected_memory_digest=digest(record("a")), expected_replacement_digest=digest(record("b")),
        reason="Owner selected replacement", key=key("supersede-a")))
    retire("b")
    ingest("d")
    update("c")
    promote("c")
    retire("d")
    retire("c")
    assert len(receipts) == 12
    assert set(operations) == {"append-receipt-v1", "update-receipt-v1", "promote-receipt-v1",
                               "supersede-receipt-v1", "retire-receipt-v1"}
    _reject_without_change(scenario.directory, schema,
        _request("ingest", "conflicting replacement text", key=key("a")), IdempotencyConflict)

    # Old append/update/promote receipts remain historical successes even after
    # their source is superseded or retired, with the backend unavailable.
    instance = make_adapter(scenario.directory, schema, unavailable=True)
    try:
        order = list(range(len(requests)))
        rng.shuffle(order)
        for index in order:
            assert invoke(instance, requests[index]) == receipts[index]
    finally:
        instance.close(persist=False)
    _assert_history(sql_observation(scenario.directory), baseline, state, receipts, operations)
    instance = make_adapter(scenario.directory, schema)
    try:
        visible = instance.retrieve_context("remembered", top_k=10, **SCOPE)
        expected_ids = {item["id"] for item in state["items"]
                        if all(item["meta"].get(name) == value for name, value in SCOPE.items())
                        and item["meta"].get("memory_state") not in {"deleted", "superseded"}}
        assert {int(item["id"]) for item in visible["items"]} == expected_ids
        assert instance.durability_status() == {"status": "ok", "failures": {}}
    finally:
        instance.close(persist=False)
    _assert_history(sql_observation(scenario.directory), baseline, state, receipts, operations)


def _authority_bytes(directory):
    names = ("dml_state.sqlite3", "dml_state.sqlite3.identity.json", "dml_state.sqlite3.migration.json")
    return {name: (directory / name).read_bytes() for name in names if (directory / name).exists()}


CORRUPTIONS = ("header", "record-checksum", "snapshot-checksum", "missing-receipt",
               "missing-decision", "identity-mismatch", "missing-authority", "migration-marker")


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_profile_reopen_preserves_damaged_authority_and_fails_explicitly(tmp_path, schema, corruption):
    scenario = prepare(tmp_path / "authority", schema)
    path = scenario.directory / "dml_state.sqlite3"
    if corruption == "header":
        raw = bytearray(path.read_bytes())
        raw[:16] = b"broken authority"
        path.write_bytes(raw)
    elif corruption == "identity-mismatch":
        path.with_name(path.name + ".identity.json").write_text(
            json.dumps({"schema_version": 1, "store_id": "f" * 32}), encoding="utf-8")
    elif corruption == "missing-authority":
        path.unlink()
    elif corruption == "migration-marker":
        path.with_name(path.name + ".migration.json").write_text("{}", encoding="utf-8")
    else:
        statements = {
            "record-checksum": "UPDATE records SET checksum='" + "0" * 64 + "' WHERE bucket='items'",
            "snapshot-checksum": "UPDATE snapshot SET checksum='" + "0" * 64 + "'",
            "missing-receipt": "DELETE FROM receipts WHERE revision=(SELECT MAX(revision) FROM receipts)",
            "missing-decision": "DELETE FROM decisions WHERE revision=(SELECT MAX(revision) FROM decisions)",
        }
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                cursor = connection.execute(statements[corruption])
                assert cursor.rowcount > 0
    damaged = _authority_bytes(scenario.directory)
    for _ in range(2):
        with pytest.raises((ProductionProfileError, JournalIntegrityError)):
            make_adapter(scenario.directory, schema)
        assert _authority_bytes(scenario.directory) == damaged


@pytest.mark.parametrize("schema", (2, 3))
@pytest.mark.parametrize("incomplete", ("zero-byte", "uncommitted-schema"))
def test_profile_does_not_silently_reinitialize_incomplete_creation(tmp_path, schema, incomplete):
    directory = tmp_path / "authority"
    directory.mkdir()
    path = directory / "dml_state.sqlite3"
    if incomplete == "zero-byte":
        path.touch()
    else:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE records (payload TEXT)")
            connection.rollback()
    before = _authority_bytes(directory)
    with pytest.raises(ProductionProfileError):
        make_adapter(directory, schema)
    assert _authority_bytes(directory) == before
    assert not path.with_name(path.name + ".identity.json").exists()


@pytest.mark.parametrize("schema", SCHEMAS)
def test_temporary_embedding_outage_preserves_history_and_allows_retry(tmp_path, monkeypatch, schema):
    scenario = prepare(tmp_path / "authority", schema)
    source = next(record for record in scenario.records
                  if all(record["meta"].get(name) == value for name, value in SCOPE.items()))
    requests = [
        _request("ingest", "provider outage memory", key="outage-ingest", meta={"source_trust": "trusted"}),
        _request("update", source["id"], text="provider outage correction",
                 expected_memory_digest=digest(source), reason="Correction", key="outage-update"),
        _request("promote", [{"memory_id": source["id"], "expected_memory_digest": digest(source)}],
                 text="provider outage summary", reason="Summary", key="outage-promote"),
    ]
    instance = make_adapter(scenario.directory, schema)
    original_embed = instance.embedder.embed

    def unavailable(_text):
        raise RuntimeError("temporary embedding provider outage")

    try:
        monkeypatch.setattr(instance.embedder, "embed", unavailable)
        for request in requests:
            with pytest.raises(ReceiptEmbeddingError):
                invoke(instance, request)
            assert sql_observation(scenario.directory) == scenario.before
        with pytest.raises(RuntimeError, match="temporary embedding provider outage"):
            instance.retrieve_context("uncached outage query", **SCOPE)
        assert sql_observation(scenario.directory) == scenario.before
        monkeypatch.setattr(instance.embedder, "embed", original_embed)
        # The exact rejected append key remains available once the same provider
        # returns; no invented placeholder or hidden fallback vector was stored.
        expected = expected_transition(scenario.before, requests[0])
        receipt = invoke(instance, requests[0])
        assert receipt == expected["receipt"]
        monkeypatch.setattr(instance.embedder, "embed", unavailable)
        assert invoke(instance, requests[0]) == receipt
    finally:
        instance.close(persist=False)
    _assert_history(sql_observation(scenario.directory), scenario.before, expected["state"],
                    [receipt], [expected["operation"]])


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("operation", ("ingest", "update", "promote", "supersede", "retire"))
def test_sqlite_page_quota_rejects_complete_mutation_and_same_key_recovers(
        tmp_path, monkeypatch, schema, operation):
    """A real SQLite SQLITE_FULL quota result; this is not host disk ENOSPC."""
    scenario = prepare(tmp_path / "authority", schema)
    instance = make_adapter(scenario.directory, schema)
    try:
        # A large immutable source means even lifecycle-only mutations need a
        # fresh, substantial receipt. Seed it through the admitted public API.
        large = invoke(instance, _request("ingest", "Large provenance source", key="quota-source",
            meta={"source_trust": "trusted", "large_provenance": "q" * 200_000}))["result"]["memory"]
        replacement = next(item for item in scenario.records
                           if all(item["meta"].get(name) == value for name, value in SCOPE.items()))
        request = request_for(operation, [large, replacement], key="quota-target")
        if operation == "ingest":
            request.args = ("q" * 200_000,)
        before = sql_observation(scenario.directory)
        connect = instance._journal._connect

        @contextmanager
        def limited():
            with connect() as connection:
                pages = connection.execute("PRAGMA page_count").fetchone()[0]
                assert connection.execute(f"PRAGMA max_page_count={pages}").fetchone()[0] == pages
                yield connection

        monkeypatch.setattr(instance._journal, "_connect", limited)
        with pytest.raises(ReceiptCommitRejected) as rejected:
            invoke(instance, request)
        cause = rejected.value.__cause__
        assert isinstance(cause, sqlite3.OperationalError)
        # Python 3.10 does not expose sqlite_errorcode; the SQLite exception and
        # canonical error text still distinguish the actual quota result.
        assert "full" in str(cause).lower()
        if hasattr(cause, "sqlite_errorcode"):
            assert cause.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert sql_observation(scenario.directory) == before
        monkeypatch.setattr(instance._journal, "_connect", connect)
    finally:
        instance.close(persist=False)

    restarted = make_adapter(scenario.directory, schema)
    try:
        assert sql_observation(scenario.directory) == before
        expected = expected_transition(before, request)
        receipt = invoke(restarted, request)
        assert receipt == expected["receipt"]
        restarted.embedder.unavailable = True
        assert invoke(restarted, request) == receipt
    finally:
        restarted.close(persist=False)
    _assert_history(sql_observation(scenario.directory), before, expected["state"],
                    [receipt], [expected["operation"]])
