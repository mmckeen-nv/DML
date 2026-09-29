"""No-model controls against actual DML receipt authority, not a RAG substitute."""

from dataclasses import replace
import json
import os
import time
import pytest
from daystrom_dml.services.agent_episode import _fixture_adapter
from daystrom_dml.services.m8_dml_memory import DMLDevelopmentMemory
from daystrom_dml.services.m8_memory_baseline import MemoryEvent

SCOPE = {"tenant_id": "m8-dev", "client_id": None, "session_id": "project", "instance_id": None}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    adapter = _fixture_adapter(tmp_path / "authority")
    bridge = DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=tmp_path / "evidence.jsonl")
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


def test_real_dml_supersession_projection_and_restart(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(10))
    bridge.ingest(event(11, "Project status is green.", version=2, corrects=10))
    bridge.ingest(replace(event(12, "Project status is red."), source_id="independent"))
    result = read(bridge)
    assert {r["record_id"] for r in result["evidence"].records} == {11, 12}
    assert result["event_clock_qualified"] is False
    assert len(bridge.receipts) == 4  # three appends plus genuine supersession
    assert bridge.records[10]["meta"]["memory_state"] == "superseded"
    assert bridge.records[10]["timestamp"] != bridge.events[10]["timestamp"]
    restored = DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")
    assert restored.records == bridge.records
    assert {r["record_id"] for r in read(restored)["evidence"].records} == {11, 12}


def test_failed_second_mutation_blocks_read_and_restart_retains_receipt(setup, monkeypatch):
    adapter, bridge, path = setup
    bridge.ingest(event(1))

    def failure(*a, **k):
        raise RuntimeError("deliberate dispatch failure")

    monkeypatch.setattr(adapter, "supersede_memory_receipted", failure)
    with pytest.raises(RuntimeError, match="deliberate"):
        bridge.ingest(event(2, version=2, corrects=1))
    with pytest.raises(ValueError, match="unresolved"):
        read(bridge)
    with pytest.raises(ValueError, match="unresolved"):
        bridge.ingest(event(3))
    rows = [json.loads(line) for line in (path / "evidence.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "receipt"
    assert len([r for r in rows if r["kind"] == "receipt"]) == 2
    with pytest.raises(ValueError, match="unresolved"):
        DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")


def test_duplicate_cross_scope_and_cross_source_reject_before_dispatch(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    before = (path / "evidence.jsonl").read_bytes()
    for bad in (
        event(1),
        replace(event(2), scope={**SCOPE, "tenant_id": "foreign"}),
        replace(event(3, version=2, corrects=1), source_id="foreign"),
    ):
        with pytest.raises(ValueError):
            bridge.ingest(bad)
    assert (path / "evidence.jsonl").read_bytes() == before
    assert len(bridge.receipts) == 1


def test_tampered_evidence_rejected(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    evidence = path / "evidence.jsonl"
    evidence.write_bytes(evidence.read_bytes().replace(b"amber", b"green"))
    with pytest.raises(ValueError, match="chain"):
        DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=evidence)


def test_external_authority_change_rejects_restart(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    from daystrom_dml.services.episode_tools import _digest

    record = bridge.records[1]
    adapter.update_memory_receipted(
        record["id"],
        text="external change",
        reason="external",
        expected_memory_digest=_digest(record),
        idempotency_key="external",
        **SCOPE,
    )
    with pytest.raises(ValueError, match="authority"):
        DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")


def test_extra_owned_row_blocks_live_read_and_restart(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    adapter.ingest_memory_receipted(
        "Unselected unrelated external row",
        idempotency_key="extra",
        meta={"source_trust": "trusted", "source": "external"},
        **SCOPE,
    )
    with pytest.raises(ValueError, match="authority"):
        read(bridge)
    with pytest.raises(ValueError, match="authority"):
        DMLDevelopmentMemory(adapter, scope=SCOPE, evidence_path=path / "evidence.jsonl")


def test_external_retirement_blocks_live_read(setup):
    adapter, bridge, path = setup
    bridge.ingest(event(1))
    from daystrom_dml.services.episode_tools import _digest

    old = bridge.records[1]
    adapter.retire_memory_receipted(
        old["id"], expected_memory_digest=_digest(old), idempotency_key="retire", reason="external", **SCOPE
    )
    with pytest.raises(ValueError, match="authority"):
        read(bridge)


def test_untrusted_replacement_remains_authentically_rejected(setup):
    from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict

    adapter, bridge, path = setup
    bridge.ingest(event(1))
    with pytest.raises(ReceiptLifecycleConflict):
        bridge.ingest(replace(event(2, version=2, corrects=1), metadata={"source_trust": "untrusted"}))
    assert bridge.failed
    assert len(bridge.receipts) == 2  # Both append acknowledgements retained.
    with pytest.raises(ValueError, match="unresolved"):
        read(bridge)
