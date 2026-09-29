"""Receipt-backed supersession display, without model generation or extra reads."""
from copy import deepcopy
import hashlib
import json
import os

import pytest

from daystrom_dml.contracts.agent_episode import (
    EXECUTION_PROTOCOL_V2, AgentEpisodeError, canonical_json, evidence_digest,
    presented_record_identities, validate_episode_events,
)
from daystrom_dml.contracts.model_input import ModelInputRequest
from daystrom_dml.services.agent_episode import (
    EpisodeLimits, _fixture_adapter, _observed, run_episode_with_test_dependencies,
)
from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
from scripts.agent_campaign_evidence import _observations as campaign_observations
from test_agent_episode_runtime import ValidationScriptedConsumer, final, tool


SCOPE = {"tenant_id": "design-team", "client_id": "reader",
         "session_id": "review", "instance_id": "test-agent"}


@pytest.fixture
def authority(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    adapter = _fixture_adapter(tmp_path / "authority")
    seeds = [{"alias": alias, "text": text, "meta": {
        "source": alias, "source_trust": "trusted", "no_merge": True,
        "claim_key": "pipeline.batch_size", "claim_value": value,
    }} for alias, text, value in (
        ("draft", "The draft design uses batches of 48 entries.", 48),
        ("approved", "The approved design uses batches of 96 entries.", 96),
    )]
    receipts = [adapter.ingest_memory_receipted(seed["text"],
        idempotency_key=seed["alias"], meta=seed["meta"], **SCOPE) for seed in seeds]
    bridge = SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id="projection-test",
        seed_receipts=receipts, allowed_tools=("retrieve", "supersede"),
        execution_protocol=EXECUTION_PROTOCOL_V2)
    try:
        yield adapter, bridge, seeds, receipts
    finally:
        adapter.close()


def supersede(authority):
    _, bridge, _, receipts = authority
    raw, shown = bridge.execute(bridge.prepare("retrieve",
        {"query": "design batches", "top_k": 10}, call_id="read"))
    records = {item["id"]: item for item in json.loads(shown)["records"]}
    source, replacement = [receipt["result"]["memory"] for receipt in receipts]
    prepared = bridge.prepare("supersede", {
        "record_ref": records[source["id"]]["record_ref"],
        "replacement_ref": records[replacement["id"]]["record_ref"],
        "reason": "The design review approved the replacement.",
    }, call_id="replace")
    return prepared, records, raw


def payload(raw, shown):
    return {"name": "supersede", "result": raw, "model_result": shown}


def test_supersession_displays_only_the_committed_source_and_replacement_id(authority, monkeypatch):
    adapter, bridge, _, receipts = authority
    prepared, records, _ = supersede(authority)
    source, replacement = [receipt["result"]["memory"] for receipt in receipts]
    before_ledger = deepcopy(bridge._presentation_ledger)
    before_references = dict(bridge._references)

    def no_extra_retrieval(*args, **kwargs):
        pytest.fail("Supersession must not expand the requested retrieval")

    monkeypatch.setattr(adapter, "retrieve_context", no_extra_retrieval)
    raw, shown = bridge.execute(prepared)
    committed = raw["receipt"]["result"]["memory"]
    displayed = json.loads(shown)
    assert set(displayed) == {"records"}
    assert len(displayed["records"]) == 1
    changed = displayed["records"][0]
    assert type(changed["superseded_by"]) is int
    assert changed["superseded_by"] == committed["meta"]["superseded_by"] == replacement["id"]
    assert changed == {**records[source["id"]], "record_ref": changed["record_ref"],
                      "memory_state": "superseded", "superseded_by": replacement["id"]}
    assert changed["record_ref"] != records[source["id"]]["record_ref"]
    assert raw["observed_records"] == [committed]
    assert replacement["text"] not in shown
    state = {item["id"]: item for item in adapter._journal.load()["items"]}
    assert state == {source["id"]: committed, replacement["id"]: replacement}
    assert len(bridge._references) == len(before_references) + 1
    assert all(bridge._references[key] == value for key, value in before_references.items())
    assert bridge._presentation_ledger == {
        **before_ledger, changed["record_ref"]: (committed["id"], canonical_json(committed))}

    # Re-executing the same owned operation does not mint another record/ref or write.
    snapshot = adapter._journal.verified_snapshot()
    ledger = deepcopy(bridge._presentation_ledger)
    assert bridge.execute(prepared) == (raw, shown)
    assert adapter._journal.verified_snapshot() == snapshot
    assert bridge._presentation_ledger == ledger


def alter_relationship(shown, mutation):
    displayed = json.loads(shown)
    record = displayed["records"][0]
    if mutation == "dropped":
        del record["superseded_by"]
    elif mutation == "different-id":
        record["superseded_by"] = record["id"]
    elif mutation == "boolean":
        record["superseded_by"] = True
    else:
        record["superseded_by"] = str(record["superseded_by"])
    return canonical_json(displayed).decode("utf-8")


@pytest.mark.parametrize("mutation", ["dropped", "different-id", "boolean", "string"])
def test_v2_presentation_rejects_missing_or_forged_receipt_relationship(authority, mutation):
    _, bridge, _, _ = authority
    prepared, _, _ = supersede(authority)
    raw, shown = bridge.execute(prepared)
    committed = raw["receipt"]["result"]["memory"]
    reference = json.loads(shown)["records"][0]["record_ref"]
    assert presented_record_identities(payload(raw, shown), SCOPE) == {
        reference: (committed["id"], canonical_json(committed))}
    altered = alter_relationship(shown, mutation)
    before = deepcopy(bridge._presentation_ledger)
    with pytest.raises(AgentEpisodeError, match="unique full record identity"):
        presented_record_identities(payload(raw, altered), SCOPE)
    with pytest.raises(AgentEpisodeError, match="unique full record identity"):
        bridge._present("supersede", raw, altered)
    assert bridge._presentation_ledger == before


@pytest.fixture
def report(authority):
    adapter, bridge, seeds, receipts = authority
    seed_records = {seed["alias"]: receipt["result"]["memory"]
                    for seed, receipt in zip(seeds, receipts)}
    task = {"id": "design-review", "prompt": "Apply the approved design after reading both notes.",
            "truth": {"pipeline.batch_size": {"value": 96, "evidence_aliases": ["approved"]}},
            "state_expectations": [{"alias": "draft", "replacement_alias": "approved"}]}
    scenario = {"id": "projection-test", "scope": SCOPE, "seeds": seeds,
                "expected_memory_count": 2, "tasks": [task]}

    def replace_source(request):
        records = json.loads(request["messages"][-1]["content"])["records"]
        refs = {item["id"]: item["record_ref"] for item in records}
        return tool("supersede", record_ref=refs[seed_records["draft"]["id"]],
                    replacement_ref=refs[seed_records["approved"]["id"]], reason="Approved design")

    consumer = ValidationScriptedConsumer([
        tool(query="design batches", top_k=10), replace_source,
        final({"claims": [{"key": "pipeline.batch_size", "value": 96,
                           "evidence_ids": [seed_records["approved"]["id"]]}]}),
    ])
    result = run_episode_with_test_dependencies(consumer=consumer, toolbox=bridge,
        task=task, scenario=scenario, seed_records=seed_records,
        current_records=lambda: adapter._journal.load()["items"],
        consumer_profile="qwen2-action-json-validation-v2", episode_id="projection-test",
        limits=EpisodeLimits(output_tokens=16))
    return result, consumer


@pytest.mark.parametrize("mutation", ["dropped", "different-id", "boolean", "string"])
def test_live_observation_and_replay_share_the_receipt_backed_relationship(report, mutation):
    result, consumer = report
    assert result["terminal"]["success"] is True
    assert result["terminal"]["execution_path"] == "test_injected"
    events = result["events"]
    validate_episode_events(events)
    mutations = [event for event in events if event["kind"] == "tool_completed"
                 and event["payload"]["name"] == "supersede"]
    assert len(mutations) == 1
    completed = mutations[0]
    assert consumer.requests[-1]["messages"][-1]["content"] == completed["payload"]["model_result"]
    committed = completed["payload"]["result"]["receipt"]["result"]["memory"]
    assert [item["record"] for item in _observed(events) if item["operation"] == "supersede"] == [committed]
    assert campaign_observations(events) == _observed(events)
    changed = deepcopy(completed)
    changed["payload"]["model_result"] = alter_relationship(changed["payload"]["model_result"], mutation)
    assert _observed([changed]) == []
    assert campaign_observations([changed]) == []


@pytest.mark.parametrize("mutation", ["dropped", "different-id", "boolean", "string"])
def test_replay_rejects_changed_relationship_even_after_transcript_rehash(report, mutation):
    result, _ = report
    events = deepcopy(result["events"])
    completed = next(event for event in events if event["kind"] == "tool_completed"
                     and event["payload"]["name"] == "supersede")
    altered = alter_relationship(completed["payload"]["model_result"], mutation)
    completed["payload"]["model_result"] = altered
    # Rewrite the later visible transcript and every affected public hash, leaving
    # the genuine authority receipt intact. Receipt identity must still reject it.
    for event in events:
        if event["kind"] != "model_requested" or event["sequence"] <= completed["sequence"]:
            continue
        request_payload = event["payload"]
        for message in request_payload["request"]["messages"]:
            if message.get("role") == "tool" and message.get("tool_call_id") == completed["call_id"]:
                message["content"] = altered
        request = ModelInputRequest.from_payload(request_payload["request"])
        request_payload["compiled"]["request_digest"] = request.request_digest
        digest = hashlib.sha256(canonical_json(request_payload["compiled"])).hexdigest()
        request_payload["artifact_digest"] = digest
        for output in events:
            if output["kind"] == "model_completed" and output["call_id"] == event["call_id"]:
                output["payload"]["artifact_digest"] = digest
    events[-1]["payload"]["evidence_digest"] = evidence_digest(events[:-1])
    with pytest.raises(AgentEpisodeError, match="unique full record identity"):
        validate_episode_events(events)
