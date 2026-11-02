"""Event-time ingestion API/profile controls; no model involvement.

Covers the proposed dml-append-request-v2 boundary: digest binding, strict
validation, idempotency, restart, CAS retry, replay integrity and the
separately versioned DMLDevelopmentEventMemory bridge. The historical
commit-time API is pinned unchanged.
"""

from dataclasses import replace
import json
import os
import time

import pytest

from daystrom_dml.journal import IdempotencyConflict, RevisionConflict
from daystrom_dml.services.agent_episode import _fixture_adapter
from daystrom_dml.services.m8_dml_memory import DMLDevelopmentEventMemory, DMLDevelopmentMemory
from daystrom_dml.services.m8_memory_baseline import MemoryEvent
from daystrom_dml.services.receipt_ingestion import (
    append_receipted,
    canonical_event_request,
    canonical_request,
)

SCOPE = {"tenant_id": "m8-dev", "client_id": None, "session_id": "project", "instance_id": None}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    adapter = _fixture_adapter(tmp_path / "authority")
    bridge = DMLDevelopmentEventMemory(adapter, scope=SCOPE, evidence_path=tmp_path / "evidence.jsonl")
    yield adapter, bridge, tmp_path
    adapter.close()


def event(i, text="Project status is amber.", **kw):
    return MemoryEvent(
        record_id=i,
        text=text,
        scope=SCOPE,
        source_id="notice",
        fact_key="status",
        version=kw.pop("version", 1),
        timestamp=1000 + i,
        metadata={"source_trust": "trusted"},
        **kw,
    )


def read(bridge):
    return bridge.retrieve(
        "Project status",
        now=time.time() + 100,
        top_k=8,
        token_budget=100000,
        count_tokens=len,
        render_records=lambda rows: json.dumps(rows),
    )


def test_event_time_bound_into_canonical_request_and_digest():
    first, digest_first = canonical_event_request("fact", event_time=1000.0, **SCOPE)
    second, digest_second = canonical_event_request("fact", event_time=1001.0, **SCOPE)
    integer, digest_integer = canonical_event_request("fact", event_time=1000, **SCOPE)
    assert first["schema_version"] == "dml-append-request-v2"
    assert first["event_time"] == 1000.0
    assert digest_first != digest_second
    assert digest_first == digest_integer
    historical, _ = canonical_request("fact", **SCOPE)
    assert historical["schema_version"] == "dml-append-request-v1"
    assert "event_time" not in historical


@pytest.mark.parametrize("bad", [True, "1000", None, float("nan"), float("inf"), -1, -0.5])
def test_event_time_validation_rejects_nonfinite_negative_and_text(bad):
    with pytest.raises(ValueError, match="Event time"):
        canonical_event_request("fact", event_time=bad, **SCOPE)


def test_event_time_boundary_zero_and_large_values():
    zero, _ = canonical_event_request("fact", event_time=0, **SCOPE)
    assert zero["event_time"] == 0.0
    large, _ = canonical_event_request("fact", event_time=2**53, **SCOPE)
    assert large["event_time"] == float(2**53)


def test_record_authority_timestamp_is_event_time_commit_time_retained(setup):
    adapter, bridge, _ = setup
    start = time.time()
    receipt = adapter.ingest_memory_event_receipted(
        "Historical claim",
        event_time=1000.0,
        idempotency_key="evt-1",
        meta={"source_trust": "trusted"},
        **SCOPE,
    )
    stop = time.time()
    record = receipt["result"]["memory"]
    assert record["timestamp"] == 1000.0
    assert start <= record["meta"]["commit_time"] <= stop


def test_historical_commit_time_api_unchanged(setup):
    adapter, _, _ = setup
    start = time.time()
    receipt = adapter.ingest_memory_receipted(
        "Live claim", idempotency_key="live-1", meta={"source_trust": "trusted"}, **SCOPE
    )
    stop = time.time()
    record = receipt["result"]["memory"]
    assert start <= record["timestamp"] <= stop
    assert "commit_time" not in record["meta"]


def test_idempotent_event_retry_returns_original_receipt(setup):
    adapter, _, _ = setup
    kwargs = dict(event_time=1000.0, idempotency_key="dup", meta={"source_trust": "trusted"}, **SCOPE)
    first = adapter.ingest_memory_event_receipted("same", **kwargs)
    second = adapter.ingest_memory_event_receipted("same", **kwargs)
    assert first == second
    _, _, snapshot = adapter._journal.verified_snapshot()
    assert len(snapshot["items"]) == 1


def test_conflicting_key_with_different_event_time_rejected(setup):
    adapter, _, _ = setup
    adapter.ingest_memory_event_receipted(
        "same", event_time=1000.0, idempotency_key="dup", meta={"source_trust": "trusted"}, **SCOPE
    )
    with pytest.raises(IdempotencyConflict):
        adapter.ingest_memory_event_receipted(
            "same", event_time=1001.0, idempotency_key="dup", meta={"source_trust": "trusted"}, **SCOPE
        )
    _, _, snapshot = adapter._journal.verified_snapshot()
    assert len(snapshot["items"]) == 1


def test_append_consistency_guards(setup):
    adapter, _, _ = setup
    event_request, event_digest = canonical_event_request("guard", event_time=5.0, **SCOPE)
    plain_request, plain_digest = canonical_request("guard", **SCOPE)
    common = dict(
        key="guard",
        embed=adapter.embedder.embed,
        capacity=adapter.store.capacity,
        embedding_space=adapter._receipt_embedding_space,
        hydrate=lambda revision, payload: None,
        degraded=lambda exc: None,
    )
    with pytest.raises(ValueError, match="bound event_time"):
        append_receipted(adapter._journal, request=event_request, request_digest=event_digest, **common)
    with pytest.raises(ValueError, match="event-time request schema"):
        append_receipted(adapter._journal, request=plain_request, request_digest=plain_digest, event_time=5.0, **common)
    with pytest.raises(ValueError, match="differs"):
        append_receipted(adapter._journal, request=event_request, request_digest=event_digest, event_time=6.0, **common)
    _, _, snapshot = adapter._journal.verified_snapshot()
    assert snapshot["items"] == []


def test_revision_conflict_retries_then_commits_event_time(setup, monkeypatch):
    adapter, _, _ = setup
    original = adapter._journal.save_with_receipt
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RevisionConflict("injected race")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter._journal, "save_with_receipt", flaky)
    receipt = adapter.ingest_memory_event_receipted(
        "race", event_time=100.0, idempotency_key="race", meta={"source_trust": "trusted"}, **SCOPE
    )
    assert calls["count"] == 2
    assert receipt["result"]["memory"]["timestamp"] == 100.0


def test_v2_bridge_event_times_supersession_and_restart(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(10))
    bridge.ingest(event(11, "Project status is green.", version=2, corrects=10))
    bridge.ingest(replace(event(12, "Project status is red."), source_id="independent"))
    result = read(bridge)
    assert {r["record_id"] for r in result["evidence"].records} == {11, 12}
    assert result["event_clock_qualified"] is False
    assert result["event_time_profile"] is True
    for record_id, record in bridge.records.items():
        assert record["timestamp"] == bridge.events[record_id]["timestamp"]
        assert isinstance(record["meta"]["commit_time"], float)
    assert bridge.records[10]["meta"]["memory_state"] == "superseded"
    restored = DMLDevelopmentEventMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")
    assert restored.records == bridge.records
    assert {r["record_id"] for r in read(restored)["evidence"].records} == {11, 12}


def test_v2_bridge_preserves_validation_and_no_clock_patching(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    before = (path / "evidence.jsonl").read_bytes()
    for bad in (
        event(1),
        replace(event(2), timestamp=999),  # nonchronological
        event(3, version=2),  # existing source/fact without explicit correction
        replace(event(4), metadata={"source_trust": "untrusted"}),
    ):
        with pytest.raises(ValueError):
            bridge.ingest(bad)
    assert (path / "evidence.jsonl").read_bytes() == before
    assert not bridge.failed
    record = bridge.records[1]
    assert record["timestamp"] == 1001.0


def test_v1_and_v2_chains_are_version_isolated(setup):
    adapter, bridge, path = setup
    scope_v1 = {**SCOPE, "session_id": "project-v1"}
    bridge.ingest(event(1))
    with pytest.raises(ValueError, match="chain"):
        DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")
    historical = DMLDevelopmentMemory(adapter, scope=scope_v1, evidence_path=path / "v1.jsonl")
    historical.ingest(replace(event(50), scope=scope_v1, fact_key="other"))
    with pytest.raises(ValueError, match="chain"):
        DMLDevelopmentEventMemory(adapter, scope=scope_v1, evidence_path=path / "v1.jsonl")


def test_adapter_restart_preserves_event_time_authority(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    adapter = _fixture_adapter(tmp_path / "event-authority")
    bridge = DMLDevelopmentEventMemory(adapter, scope=SCOPE, evidence_path=tmp_path / "restart.jsonl")
    bridge.ingest(event(1))
    adapter.close()
    reopened = _fixture_adapter(tmp_path / "event-authority")
    try:
        report = reopened.retrieve_context("Project status", top_k=8, as_of=time.time() + 100, **SCOPE)
        assert report["items"][0]["timestamp"] == 1001.0
        assert isinstance(report["items"][0]["meta"]["commit_time"], float)
    finally:
        reopened.close()
