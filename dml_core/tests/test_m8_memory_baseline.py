"""No-model controls for the independently durable M8 memory wrappers."""
from dataclasses import replace
import json
import sqlite3

import pytest

from daystrom_dml.services.m8_memory_baseline import (
    ContextBuffer, MemoryEvent, PersistentRAGBaseline, present_records, validate_event,
)
from scripts.production_baseline import SQLiteBaseline

SCOPE = dict(tenant_id="tenant", client_id="project", session_id="story", instance_id=None)
OTHER = {**SCOPE, "client_id": "private"}
VEC = [1., 0.]


def event(index, **changes):
    return replace(MemoryEvent(index, f"fact {index}", SCOPE, f"source {index}", "setting", 1, 10.), **changes)


def render(records):
    return json.dumps({"records": records}, sort_keys=True)


def count(text):
    # Injected deterministic token counter for wrapper mechanics only. Actual
    # inference integration must pass the pinned tokenizer, never this fixture.
    return len(text)


def query(store, **kwargs):
    return store.retrieve(VEC, scope=SCOPE, now=100., token_budget=20000,
                          count_tokens=count, render_records=render, **kwargs)


def store_at(path):
    return PersistentRAGBaseline(path, embedding_identity="fixture-2d-v1", dimensions=2)


def test_durable_history_correction_and_independent_source(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = store_at(path)
    store.ingest(event(1, text="old", source_id="authority"), VEC)
    store.ingest(event(2, text="new", source_id="authority", version=2, corrects=1, timestamp=20), VEC)
    store.ingest(event(3, text="independent conflict", timestamp=20), VEC)
    store.ingest(event(4, scope=OTHER, text="private"), VEC)
    store.close()
    reopened = store_at(path)
    result = query(reopened)
    assert [r["record_id"] for r in result.records] == [2, 3]
    assert [r["text"] for r in reopened.history(scope=SCOPE)] == ["old", "new", "independent conflict"]
    assert "private" not in result.context
    assert reopened.store.connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert reopened.store.connection.execute("PRAGMA synchronous").fetchone() == (2,)
    reopened.close()


def test_current_view_filters_before_ranking_and_topk(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    store.ingest(event(1, source_id="a"), VEC)
    store.ingest(event(2, source_id="a", corrects=1, version=2, timestamp=20), [0., 1.])
    store.ingest(event(3, timestamp=20), [0.6, 0.8])
    assert [r["record_id"] for r in query(store, top_k=1).records] == [3]
    store.close()


def test_ranking_matches_original_sqlite_baseline(tmp_path):
    original = SQLiteBaseline(tmp_path / "original.sqlite")
    store = store_at(tmp_path / "new.sqlite")
    for index, vector in enumerate(([0., 1.], [0.6, 0.8], VEC, VEC), 1):
        item = event(index, timestamp=float(index * 10))
        original.remember(index, item.text, vector, scope=SCOPE, timestamp=item.timestamp)
        store.ingest(item, vector)
    assert [r["record_id"] for r in query(store).records] == original.recall(VEC, scope=SCOPE, now=100., top_k=8)["ids"]
    original.close()
    store.close()


@pytest.mark.parametrize("changes", [
    dict(record_id=True), dict(timestamp=float("nan")), dict(timestamp=-1),
    dict(version=0), dict(scope={"tenant_id": "tenant"}), dict(text=""),
    dict(metadata={"bad": float("inf")}), dict(corrects=88),
    dict(source_id="other", corrects=1, version=2),
    dict(scope=OTHER, source_id="authority", corrects=1, version=2),
    dict(source_id="authority", corrects=1, version=1),
    dict(source_id="authority", corrects=1, version=2, timestamp=0),
    dict(source_id="authority"),
])
def test_invalid_ingest_leaves_both_tables_unchanged(tmp_path, changes):
    store = store_at(tmp_path / "memory.sqlite")
    store.ingest(event(1, source_id="authority"), VEC)
    before = store.history(scope=SCOPE)
    with pytest.raises((ValueError, TypeError)):
        store.ingest(event(2, **changes), VEC)
    assert store.history(scope=SCOPE) == before
    assert store.store.connection.execute("SELECT count(*) FROM memory").fetchone() == (1,)
    assert store.store.connection.execute("SELECT count(*) FROM m8_events").fetchone() == (1,)
    store.close()


def test_sql_failure_rolls_back_memory_insert(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    store.store.connection.execute("CREATE TRIGGER fail_metadata BEFORE INSERT ON m8_events BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.ingest(event(1), VEC)
    assert store.history(scope=SCOPE) == []
    assert store.store.connection.execute("SELECT count(*) FROM memory").fetchone() == (0,)
    store.close()


def test_stale_predecessor_rejects_without_duplicate_effect(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    store.ingest(event(1, source_id="a"), VEC)
    store.ingest(event(2, source_id="a", corrects=1, version=2), VEC)
    with pytest.raises(ValueError, match="stale"):
        store.ingest(event(3, source_id="a", corrects=1, version=3), VEC)
    assert len(store.history(scope=SCOPE)) == 2
    store.close()


@pytest.mark.parametrize("vector", [[], [1.], [float("nan"), 0.], [float("inf"), 0.], [0., 0.], [2., 0.]])
def test_invalid_vectors_rejected_for_ingest_and_query(tmp_path, vector):
    store = store_at(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError):
        store.ingest(event(1), vector)
    with pytest.raises(ValueError):
        store.retrieve(vector, scope=SCOPE, now=100., count_tokens=count, render_records=render)
    assert store.history(scope=SCOPE) == []
    store.close()


def test_identity_attestation_and_future_time(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = store_at(path)
    store.ingest(event(1, timestamp=101), VEC)
    with pytest.raises(ValueError, match="future"):
        query(store)
    store.close()
    with pytest.raises(ValueError, match="identity"):
        PersistentRAGBaseline(path, embedding_identity="changed", dimensions=2)


def test_exact_token_hook_preserves_long_whole_records_and_ids(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    item = event(1, text="long fact " * 100)
    store.ingest(item, VEC)
    tokens = len(render([validate_event(item)]))
    result = store.retrieve(VEC, scope=SCOPE, now=100, token_budget=tokens, count_tokens=count, render_records=render)
    assert result.records[0]["text"] == item.text
    assert result.token_count == tokens
    shorter = store.retrieve(VEC, scope=SCOPE, now=100, token_budget=tokens - 1, count_tokens=count, render_records=render)
    assert shorter.records == ()
    store.close()


def test_context_only_newest_whole_chronology_no_hidden_lookup():
    one, two, three = (event(i, timestamp=i) for i in (1, 2, 3))
    budget = len(render([validate_event(two), validate_event(three)]))
    buffer = ContextBuffer(scope=SCOPE, token_budget=budget, count_tokens=count, render_records=render)
    for item in (one, two, three):
        buffer.ingest(item)
    assert [r["record_id"] for r in buffer.present().records] == [2, 3]
    with pytest.raises(ValueError, match="scope"):
        buffer.ingest(event(4, scope=OTHER))
    assert [r["record_id"] for r in buffer.present().records] == [2, 3]


def test_presentation_rejects_invalid_counter_or_envelope_budget():
    with pytest.raises(ValueError, match="token count"):
        present_records([], 10, lambda text: True, render)
    with pytest.raises(ValueError, match="envelope"):
        present_records([], 1, count, render)


def test_context_returned_envelopes_cannot_mutate_retained_state():
    item = event(1, metadata={"nested": {"value": "original"}})
    buffer = ContextBuffer(scope=SCOPE, token_budget=2000, count_tokens=count, render_records=render)
    buffer.ingest(item)
    item.metadata["nested"]["value"] = "caller mutation"
    first = buffer.present()
    assert first.records[0]["metadata"]["nested"]["value"] == "original"
    first.records[0]["metadata"]["nested"]["value"] = "returned mutation"
    assert buffer.present().records[0]["metadata"]["nested"]["value"] == "original"


@pytest.mark.parametrize("changes", [
    {"scope": {**SCOPE, "tenant_id": None}},
    {"scope": {**SCOPE, "tenant_id": "  "}},
    {"scope": {**SCOPE, "client_id": " "}},
    {"scope": {**SCOPE, "client_id": "é" * 129}},
    {"record_id": 2**63}, {"version": 2**63}, {"corrects": 2**63},
    {"version": True}, {"corrects": False},
])
def test_scope_and_integer_contract_boundaries(tmp_path, changes):
    store = store_at(tmp_path / "memory.sqlite")
    store.ingest(event(1), VEC)
    with pytest.raises(ValueError):
        store.ingest(event(2, **changes), VEC)
    assert len(store.history(scope=SCOPE)) == 1
    assert store.store.connection.execute("SELECT count(*) FROM memory").fetchone() == (1,)
    store.close()


def test_unrelated_event_chronology_is_scoped_and_atomic(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    store.ingest(event(1, timestamp=20), VEC)
    with pytest.raises(ValueError, match="nonchronological"):
        store.ingest(event(2, timestamp=19), VEC)
    store.ingest(event(3, scope=OTHER, timestamp=1), VEC)
    assert [item["record_id"] for item in store.history(scope=SCOPE)] == [1]
    assert [item["record_id"] for item in store.history(scope=OTHER)] == [3]
    store.close()


def test_maximum_supported_int64_and_scope_bytes(tmp_path):
    store = store_at(tmp_path / "memory.sqlite")
    scoped = {**SCOPE, "client_id": "é" * 128}
    item = event(2**63 - 1, version=2**63 - 1, scope=scoped)
    store.ingest(item, VEC)
    assert store.history(scope=scoped)[0]["record_id"] == 2**63 - 1
    store.close()
