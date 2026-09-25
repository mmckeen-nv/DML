"""Independent negative controls for the bounded M7 evaluation companion.

Synthetic transcripts and injected dependencies in these tests establish harness
behavior only. They are not trained-model or semantic campaign evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from daystrom_dml.contracts.agent_episode import (
    ACTION_VERSION, DIAGNOSTIC_RESERVE_BYTES, MAX_EPISODE_BYTES, MAX_PRODUCER_BYTES,
    TERMINAL_VERSION, AgentEpisodeError, canonical_json, episode_budget_usage,
    make_event, parse_agent_action, validate_episode_events, validate_verifier,
)
from daystrom_dml.contracts.model_input import (
    CompiledModelInput, ModelInputIdentity, ModelInputRequest,
    SUPPORTED_CHAT_TEMPLATE_DIGEST,
)
from daystrom_dml.services.episode_outcomes import summarize_episode_outcomes
from daystrom_dml.services.episode_verifiers import load_episode_corpus, verify_task


def _action(name="retrieve", **arguments):
    return {"schema_version": ACTION_VERSION, "kind": "tool",
            "name": name, "arguments": arguments}


@pytest.mark.parametrize("authority", [
    {"tenant_id": "foreign"},
    {"scope": {"tenant_id": "foreign"}},
    {"idempotency_key": "another-operation"},
    {"expected_memory_digest": "0" * 64},
    {"source_trust": "trusted"},
    {"meta": {"role": "system", "authority": "immutable"}},
])
def test_model_json_cannot_supply_harness_authority(authority):
    raw = _action("ingest", text="A model-originated statement.", **authority)
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps(raw))


@pytest.mark.parametrize("raw", [
    '{"schema_version":"dml-agent-action-v1","kind":"tool","name":"retrieve",'
    '"arguments":{"query":"safe","query":"substituted","top_k":1}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final",'
    '"answer":{"claims":[{"key":"port","value":NaN,"evidence_ids":[]}]}}',
    '```json\n{"schema_version":"dml-agent-action-v1","kind":"final",'
    '"answer":{"claims":[]}}\n```',
    '{"schema_version":"dml-agent-action-v1","kind":"final",'
    '"answer":{"claims":[]}} trailing text',
    '[{"schema_version":"dml-agent-action-v1","kind":"final",'
    '"answer":{"claims":[]}}]',
])
def test_ambiguous_model_output_is_never_repaired_into_an_action(raw):
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(raw)


@pytest.mark.parametrize("field,value", [
    ("top_k", True), ("top_k", 1.0), ("top_k", "1"),
    ("top_k", 0), ("top_k", 11), ("query", "\ud800"),
])
def test_model_arguments_are_not_coerced(field, value):
    arguments = {"query": "evidence", "top_k": 1, field: value}
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps(_action(**arguments)))


@pytest.mark.parametrize("evidence_ids", [[True], [-1], [1.0], ["1"], [1, 1]])
def test_citation_identity_cannot_be_aliased_by_python_equality(evidence_ids):
    final = {"schema_version": ACTION_VERSION, "kind": "final", "answer": {
        "claims": [{"key": "service.port", "value": 8000,
                    "evidence_ids": evidence_ids}]}}
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps(final))


def test_claims_cannot_redefine_the_same_output_key():
    final = {"schema_version": ACTION_VERSION, "kind": "final", "answer": {
        "claims": [{"key": "port", "value": 8000, "evidence_ids": [1]},
                   {"key": "port", "value": 9000, "evidence_ids": [2]}]}}
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps(final))


def _terminal(task_id, *, success=False, unknown=False):
    """A literal accounting fixture, without any claimed producer authenticity."""
    report = {"verifier_version": "dml-episode-verifier-v1", "success": success,
              "reasons": [] if success else ["No verified final answer"],
              "contradictions": None, "factual_outputs": None,
              "false_memory_claims": None, "recalled_claims": None,
              "repeat_errors": None, "repeat_opportunities": None}
    return {
        "schema_version": TERMINAL_VERSION, "episode_id": "independent-accounting",
        "task_id": task_id, "execution_path": "test_injected",
        "status": "completed" if success else "timeout", "success": success,
        "answer": {"claims": []} if success else None, "verifier": report,
        "evidence_digest": "a" * 64,
        "input_tokens": None if unknown else 10,
        "output_tokens": None if unknown else 5,
        "maintenance_tokens": 0, "known_input_tokens": 10,
        "known_output_tokens": 5, "unknown_input_calls": int(unknown),
        "unknown_output_calls": int(unknown), "usage_unknown": unknown,
        "effects_unknown": unknown, "latency_ms": 100,
        "retrieval_ms": None if unknown else 2, "ttft_ms": None,
    }


def test_failed_attempts_charge_known_tokens_to_successful_task_denominator():
    report = summarize_episode_outcomes([
        _terminal("failed"), _terminal("succeeded", success=True),
    ])
    assert report["attempted_tasks"] == 2
    assert report["completed_tasks"] == 1
    assert report["task_success_rate"] == 0.5
    assert report["total_tokens"] == 30
    assert report["tokens_per_completed_task"] == 30
    assert report["false_memory_rate"] is None
    assert report["ttft_ms"]["samples"] == 0


def test_unknown_failed_work_cannot_improve_an_exact_cost_or_quality_result():
    success = _terminal("success", success=True)
    success["verifier"].update(contradictions=0, factual_outputs=1,
                               false_memory_claims=0, recalled_claims=1)
    report = summarize_episode_outcomes([success, _terminal("killed", unknown=True)])
    assert report["task_success_rate"] == 0.5
    assert report["total_tokens"] is None
    assert report["tokens_per_completed_task"] is None
    assert report["known_total_tokens"] == 30
    assert report["unknown_usage_tasks"] == 1
    assert report["unknown_effect_tasks"] == 1
    assert report["contradiction_rate"] is None
    assert report["false_memory_rate"] is None
    assert report["false_memory_coverage"]["known_denominator"] == 1
    assert report["false_memory_coverage"]["unknown_tasks"] == 1
    assert report["retrieval_overhead_ms"]["unknown_tasks"] == 1


def test_zero_success_cost_is_undefined_and_repeated_terminals_are_rejected():
    terminal = _terminal("failed", unknown=True)
    report = summarize_episode_outcomes([terminal])
    assert report["tokens_per_completed_task"] is None
    assert report["known_tokens_per_completed_task"] is None
    with pytest.raises(AgentEpisodeError):
        summarize_episode_outcomes([terminal, terminal])


def test_injected_results_cannot_be_pooled_with_concrete_local_execution():
    live = _terminal("live")
    live["execution_path"] = "live_local"
    with pytest.raises(AgentEpisodeError):
        summarize_episode_outcomes([_terminal("injected"), live])


@pytest.mark.parametrize("version", [
    "dml-episode-verifier-v2", "dml-agent-verifier-v1", "custom-verifier", "", None,
])
def test_terminal_v1_rejects_unknown_nested_verifier_semantics(version):
    report = _terminal("bad-version")["verifier"]
    report["verifier_version"] = version
    with pytest.raises(AgentEpisodeError):
        validate_verifier(report)


def _verification_case():
    """Independent literal state for the two-source verifier, without a runner."""
    corpus = load_episode_corpus()
    scenario = next(value for value in corpus["scenarios"]
                    if value["id"] == "near_duplicates")
    task = scenario["tasks"][0]
    seeds = {seed["alias"]: {
        "id": ident, "text": seed["text"], "timestamp": 1900000000.0,
        "level": 0, "salience": 1.0, "fidelity": 1.0,
        "meta": {**seed["meta"], **seed.get("scope", scenario["scope"]),
                 "kind": "memory"},
    } for ident, seed in enumerate(scenario["seeds"])}
    observed = [{"sequence": index + 1, "task_id": task["id"],
                 "operation": "retrieve", "record": deepcopy(record)}
                for index, record in enumerate(seeds.values())
                if record["meta"]["tenant_id"] == scenario["scope"]["tenant_id"]]
    answer = {"claims": [{"key": "service.port", "value": 8000,
                           "evidence_ids": [0, 1]}]}
    kwargs = {"seed_records": seeds, "observed_records": observed,
              "current_records": deepcopy(list(seeds.values())),
              "final_sequence": 5, "effective_time": corpus["effective_time"]}
    return scenario, task, answer, kwargs


def test_independent_typed_two_source_oracle_has_a_positive_control():
    scenario, task, answer, kwargs = _verification_case()
    report = verify_task(scenario, task, answer, **kwargs)
    assert report["success"] is True
    assert report["factual_outputs"] == 1
    assert report["false_memory_claims"] == 0


@pytest.mark.parametrize("mutation", [
    "wrong_value", "numeric_coercion", "foreign_citation", "missing_source",
    "future_observation", "other_task_observation", "unseen_source",
    "modified_snapshot", "retired_snapshot", "wrong_seed_alias",
])
def test_oracle_rejects_plausible_answers_with_bad_provenance_or_state(mutation):
    scenario, task, answer, kwargs = _verification_case()
    claim = answer["claims"][0]
    if mutation == "wrong_value":
        claim["value"] = 9000
    elif mutation == "numeric_coercion":
        claim["value"] = 8000.0
    elif mutation == "foreign_citation":
        claim["evidence_ids"] = [0, 2]
    elif mutation == "missing_source":
        claim["evidence_ids"] = [0]
    elif mutation == "future_observation":
        kwargs["observed_records"][1]["sequence"] = kwargs["final_sequence"]
    elif mutation == "other_task_observation":
        kwargs["observed_records"][1]["task_id"] = "a-prior-task"
    elif mutation == "unseen_source":
        kwargs["observed_records"].pop()
    elif mutation == "modified_snapshot":
        kwargs["current_records"][1]["text"] = "Changed after the observation."
    elif mutation == "retired_snapshot":
        kwargs["current_records"][1]["meta"]["memory_state"] = "retired"
    else:
        seeds = kwargs["seed_records"]
        seeds["configuration_a"], seeds["configuration_b"] = (
            seeds["configuration_b"], seeds["configuration_a"])
    report = verify_task(scenario, task, answer, **kwargs)
    assert report["success"] is False
    assert report["reasons"]


@pytest.mark.parametrize("answer", [None, {}, {"claims": "unparsed"},
                                    {"claims": [{"success": True}]}])
def test_unverifiable_final_outputs_keep_all_quality_denominators_unknown(answer):
    scenario, task, _, kwargs = _verification_case()
    report = verify_task(scenario, task, answer, **kwargs)
    assert report["success"] is False
    for field in ("contradictions", "factual_outputs", "false_memory_claims",
                  "recalled_claims", "repeat_errors", "repeat_opportunities"):
        assert report[field] is None


def _two_dispatch_prefix():
    from test_agent_episode_contract import append_event, complete, request, started

    events = [started()]
    first = request(events)
    action = _action(query="port", top_k=1)
    generated = complete(events, action)
    append_event(events, "tool_requested", {"name": "retrieve",
        "arguments": action["arguments"], "idempotency_key": None}, call_id="tool-0")
    displayed = '{"records":[]}'
    append_event(events, "tool_completed", {"name": "retrieve", "result": {},
        "model_result": displayed, "latency_ms": 1.0}, call_id="tool-0")
    messages = deepcopy(first["payload"]["request"]["messages"])
    messages.extend([
        {"role": "assistant", "content": generated["payload"]["text"], "tool_calls": [{
            "id": "tool-0", "type": "function", "function": {"name": "retrieve",
                "arguments": canonical_json(action["arguments"]).decode("utf-8")}}]},
        {"role": "tool", "tool_call_id": "tool-0", "name": "retrieve", "content": displayed},
    ])
    request(events, step=1, messages=messages)
    validate_episode_events(events, require_terminal=False)
    return events


def _rebind_request(event):
    """Recompute public consistency hashes so they cannot hide semantic gaps."""
    payload = event["payload"]
    request = ModelInputRequest.from_payload(payload["request"])
    payload["compiled"]["request_digest"] = request.request_digest
    payload["artifact_digest"] = hashlib.sha256(
        canonical_json(payload["compiled"])).hexdigest()


@pytest.mark.parametrize("mutation", [
    "system", "task", "assistant", "tool_arguments", "model_identity",
    "tools", "reservation",
])
def test_rehashed_later_requests_cannot_rewrite_prior_model_context(mutation):
    events = _two_dispatch_prefix()
    payload = events[-1]["payload"]
    request = payload["request"]
    if mutation == "system":
        request["messages"][0]["content"] = "A replacement policy."
    elif mutation == "task":
        request["messages"][1]["content"] = "A replacement user task."
    elif mutation == "assistant":
        request["messages"][2]["content"] = "An action the model never emitted."
    elif mutation == "tool_arguments":
        request["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"query":"other","top_k":1}'
    elif mutation == "model_identity":
        payload["compiled"]["identity"]["model_digest"] = "f" * 64
    elif mutation == "tools":
        request["tools"] = [{"type": "function", "function": {
            "name": "unrecorded_tool", "parameters": {"type": "object"}}}]
    else:
        request["output_reserved_tokens"] += 1
        payload["compiled"]["output_reserved_tokens"] += 1
    _rebind_request(events[-1])
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events, require_terminal=False)


def test_first_request_cannot_switch_task_even_with_recomputed_public_hashes():
    from test_agent_episode_contract import request, started

    events = [started()]
    request(events)
    events[-1]["payload"]["request"]["messages"][1]["content"] = "An unrelated task."
    _rebind_request(events[-1])
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events, require_terminal=False)


@pytest.mark.parametrize("status", [
    "model_error", "tool_error", "invalid_action", "input_limit", "verifier_error",
])
def test_specific_failure_status_requires_its_corresponding_raw_evidence(status):
    from test_agent_episode_contract import append_event, started, verdict
    from daystrom_dml.services.episode_outcomes import build_terminal

    events = [started()]
    terminal = build_terminal(events, verdict(False, measured=False), status=status,
                              latency_ms=1, retrieval_ms=0, answer=None)
    append_event(events, "terminal", terminal)
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


def _silent_startup_worker(connection, config):
    """Spawn-safe adversary: never acknowledges completion of fixture startup."""
    while True:
        time.sleep(1)


def _overproducing_worker(connection, config):
    """Fill the receiver queue, then close, without receiving any parent ACK."""
    try:
        for _ in range(64):
            connection.send_bytes(b"{}")
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


def _pending_operation_worker(connection, config):
    """Emit admitted synthetic operations, then disappear before their result."""
    from daystrom_dml.services.agent_episode import EpisodeLimits, build_episode_request

    def send(value):
        connection.send_bytes(canonical_json(value))
        assert connection.recv_bytes(128) == b"ack"

    send({"kind": "prepared", "prepared": {
        "seed_receipts": [], "seed_records": {}, "setup_records": []}})
    request = ModelInputRequest.from_payload(build_episode_request(
        config["task"], limits=EpisodeLimits(**config["limits"])))
    identity = ModelInputIdentity("1" * 64, "2" * 64,
        SUPPORTED_CHAT_TEMPLATE_DIGEST, "independent-synthetic-dispatch", 4096)
    artifact = CompiledModelInput(identity, request.request_digest, (1, 2, 3), (1, 1, 1),
        request.output_reserved_tokens, 4096, "synthetic-consumer", "synthetic-nonce", "0" * 64)
    sequence = 1

    def event(kind, call_id, payload):
        nonlocal sequence
        send({"kind": "event", "event": make_event(
            episode_id=config["episode_id"], task_id=config["task"]["id"],
            sequence=sequence, kind=kind, call_id=call_id, payload=payload)})
        sequence += 1

    event("model_requested", "model-0", {"step": 0, "request": request.to_payload(),
        "compiled": artifact.signing_payload(), "artifact_digest": artifact.artifact_digest})
    if config["pending_kind"] == "tool":
        action = _action("supersede", record_ref="r0", replacement_ref="r1", reason="Apply current preference")
        event("model_completed", "model-0", {"step": 0,
            "artifact_digest": artifact.artifact_digest, "input_token_count": 3,
            "output_ids": [4, 5], "output_token_count": 2,
            "text": canonical_json(action).decode("utf-8"), "latency_ms": 0,
            "ttft_ms": None})
        event("tool_requested", "tool-0", {"name": "supersede", "arguments": action["arguments"],
            "idempotency_key": "episode:" + config["episode_id"] + ":tool-0"})
    Path(config["admitted_marker"]).write_text("parent acknowledged", encoding="utf-8")
    while True:
        time.sleep(1)


def _supervisor_config(tmp_path, *, seconds=1.0):
    from daystrom_dml.services.agent_episode import EpisodeLimits

    corpus = load_episode_corpus()
    scenario = corpus["scenarios"][0]
    return {"episode_id": "independent-supervisor", "execution_path": "test_injected",
            "scenario": scenario, "task": scenario["tasks"][0],
            "limits": asdict(EpisodeLimits(wall_time_seconds=seconds)),
            "effective_time": corpus["effective_time"], "previous_answers": None,
            "snapshot_directory": str(tmp_path / "absent-model"),
            "authority_directory": str(tmp_path / "absent-authority")}


def test_supervisor_counts_and_terminates_stalled_startup(tmp_path):
    from daystrom_dml.services.agent_episode import _supervise

    start = time.monotonic()
    result = _supervise(_supervisor_config(tmp_path, seconds=0.5),
                        worker_target=_silent_startup_worker)
    assert time.monotonic() - start < 5.0
    terminal = result["terminal"]
    assert terminal["status"] == "timeout"
    assert terminal["success"] is False
    assert terminal["input_tokens"] is None
    assert terminal["output_tokens"] is None
    assert terminal["effects_unknown"] is True
    assert terminal["execution_path"] == "test_injected"
    assert result["live_qualified"] is False
    validate_episode_events(result["events"])


def test_process_creation_failure_still_produces_one_failed_terminal(tmp_path, monkeypatch):
    import multiprocessing.process
    from daystrom_dml.services import agent_episode

    removed = []
    remove = agent_episode._remove_worker_scratch

    def record_cleanup(directory, identity):
        remove(directory, identity)
        removed.append(directory)

    def cannot_start(process):
        raise OSError("Independent simulated process admission failure")

    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", cannot_start)
    monkeypatch.setattr(agent_episode, "_remove_worker_scratch", record_cleanup)
    result = agent_episode._supervise(_supervisor_config(tmp_path), worker_target=_silent_startup_worker)
    assert result["terminal"]["status"] == "runner_error"
    assert result["terminal"]["success"] is False
    assert sum(event["kind"] == "terminal" for event in result["events"]) == 1
    assert len(removed) == 1 and not Path(removed[0]).exists()
    validate_episode_events(result["events"])


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_worker_scratch_cleanup_preserves_symlink_target_and_readonly_copy(tmp_path, monkeypatch, symlink_parent):
    import os
    from daystrom_dml.services.agent_episode import _remove_worker_scratch

    owned = tmp_path / "owned"
    copy = owned / "verified"
    copy.mkdir(parents=True)
    (copy / "weights").write_bytes(b"private-copy")
    (copy / "weights").chmod(0o400)
    copy.chmod(0o500)
    outside = tmp_path / "unrelated-source"
    outside.mkdir()
    sentinel = outside / "weights"
    sentinel.write_bytes(b"must survive")
    try:
        (owned / "external").symlink_to(outside, target_is_directory=True)
        if symlink_parent:
            alias = tmp_path / "parent-alias"
            alias.symlink_to(tmp_path, target_is_directory=True)
            owned = alias / owned.name
    except OSError:
        pytest.skip("Creating directory symlinks requires platform permission")
    unlink = os.unlink
    denied = []

    def readonly_once(path, *args, **kwargs):
        if Path(path).name == "weights" and not denied:
            denied.append(path)
            raise PermissionError("Exercise read-only cleanup even as a privileged user")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", readonly_once)
    identity = owned.lstat()
    _remove_worker_scratch(owned, (identity.st_dev, identity.st_ino))
    assert len(denied) == 1
    assert not owned.exists()
    assert sentinel.read_bytes() == b"must survive"


def test_worker_scratch_cleanup_refuses_replaced_directory(tmp_path):
    from daystrom_dml.services.agent_episode import _remove_worker_scratch

    owned = tmp_path / "owned"
    owned.mkdir()
    identity = owned.lstat()
    owned.rename(tmp_path / "original")
    owned.mkdir()
    sentinel = owned / "not-owned"
    sentinel.write_bytes(b"must survive")
    with pytest.raises(OSError, match="ownership"):
        _remove_worker_scratch(owned, (identity.st_dev, identity.st_ino))
    assert sentinel.read_bytes() == b"must survive"


@pytest.mark.parametrize("pending_kind", ["model", "tool"])
def test_killed_admitted_operation_has_one_matching_unknown_failure(tmp_path, pending_kind):
    from daystrom_dml.services.agent_episode import _supervise

    config = _supervisor_config(tmp_path, seconds=3)
    if pending_kind == "tool":
        config["scenario"] = load_episode_corpus()["scenarios"][1]
        config["task"] = config["scenario"]["tasks"][0]
    config["pending_kind"] = pending_kind
    config["admitted_marker"] = str(tmp_path / "admitted.txt")
    result = _supervise(config, worker_target=_pending_operation_worker)
    assert Path(config["admitted_marker"]).read_text(encoding="utf-8") == "parent acknowledged"
    failed = [event for event in result["events"] if event["kind"] == pending_kind + "_failed"]
    assert len(failed) == 1
    assert failed[0]["call_id"] == pending_kind + "-0"
    terminal = result["terminal"]
    assert terminal["status"] == "timeout"
    assert terminal["success"] is False
    assert terminal["input_tokens"] is None
    assert terminal["output_tokens"] is None
    if pending_kind == "model":
        assert failed[0]["payload"]["input_token_count"] is None
        assert failed[0]["payload"]["output_token_count"] is None
    else:
        assert failed[0]["payload"]["effects"] == "unknown"
        assert terminal["known_input_tokens"] == 3
        assert terminal["known_output_tokens"] == 2
    validate_episode_events(result["events"])


def test_repeated_malformed_worker_frames_do_not_leave_reader_threads(tmp_path):
    from daystrom_dml.services.agent_episode import _supervise

    initial = set(threading.enumerate())
    for iteration in range(3):
        result = _supervise(_supervisor_config(tmp_path / str(iteration)),
                            worker_target=_overproducing_worker)
        assert result["terminal"]["success"] is False
        assert result["terminal"]["status"] in {"runner_error", "killed"}
        validate_episode_events(result["events"])
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and any(
            thread.is_alive() for thread in set(threading.enumerate()) - initial):
        time.sleep(0.01)
    assert not [thread.name for thread in set(threading.enumerate()) - initial
                if thread.is_alive()], "Each failed episode must release its receiver thread"


def test_high_salience_expired_fixture_is_suppressed_at_the_pinned_task_time(tmp_path):
    from daystrom_dml.services.agent_episode import _prepare_fixture
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools

    corpus = load_episode_corpus()
    scenario = next(case for case in corpus["scenarios"]
                    if case["id"] == "stale_high_salience")
    adapter, prepared = _prepare_fixture(tmp_path / "clock-authority", scenario, "clock-test")
    try:
        bridge = SelectedProfileEpisodeTools(
            adapter, scope=scenario["scope"], episode_id="clock-test",
            seed_receipts=prepared["seed_receipts"],
            observation_records=list(prepared["seed_records"].values()),
            allowed_tools=("retrieve",), effective_time=corpus["effective_time"])
        raw, displayed = bridge.execute(bridge.prepare(
            "retrieve", {"query": "endpoint", "top_k": 10}, call_id="tool-0"))
        records = json.loads(displayed)["records"]
        assert prepared["seed_records"]["stale"]["salience"] == 1000
        assert prepared["seed_records"]["stale"]["id"] not in [
            record["id"] for record in records]
        assert prepared["seed_records"]["current_endpoint"]["id"] in [
            record["id"] for record in records]
        assert "old.example.invalid" not in displayed
        assert raw["observed_records"]
    finally:
        adapter.close()


def test_producer_byte_boundary_is_exact_and_one_byte_over_is_rejected():
    from test_agent_episode_contract import final_events

    events = final_events()
    usage = episode_budget_usage(events)
    assert usage["total_bytes"] == len(canonical_json(events))
    expected_producer = sum(len(canonical_json(event)) + 1 for event in events[1:])
    assert usage["producer_bytes"] == expected_producer
    limits = events[0]["payload"]["limits"]
    limits["max_episode_bytes"] = expected_producer
    limits["max_event_bytes"] = max(len(canonical_json(event)) for event in events[1:])
    validate_episode_events(events, require_terminal=False)
    limits["max_episode_bytes"] -= 1
    with pytest.raises(AgentEpisodeError, match="episode byte budget"):
        validate_episode_events(events, require_terminal=False)


def test_individual_event_cap_cannot_be_hidden_inside_larger_episode_budget():
    from test_agent_episode_contract import final_events

    events = final_events()
    events[0]["payload"]["limits"]["max_event_bytes"] = max(
        len(canonical_json(event)) for event in events[1:]) - 1
    with pytest.raises(AgentEpisodeError, match="event byte budget"):
        validate_episode_events(events, require_terminal=False)


def test_tiny_producer_budget_still_has_bounded_diagnostic_failure_evidence():
    from test_agent_episode_contract import append_event, started, verdict
    from daystrom_dml.services.episode_outcomes import build_terminal

    events = [started()]
    events[0]["payload"]["limits"].update(max_episode_bytes=1, max_event_bytes=1)
    terminal = build_terminal(events, verdict(False, measured=False), status="runner_error",
        latency_ms=1, retrieval_ms=None, usage_unknown=True)
    append_event(events, "terminal", terminal)
    validate_episode_events(events)
    usage = episode_budget_usage(events)
    assert usage["producer_bytes"] == 0
    assert 1 < usage["diagnostic_bytes"] <= DIAGNOSTIC_RESERVE_BYTES
    assert usage["total_bytes"] == len(canonical_json(events))
    assert MAX_PRODUCER_BYTES + DIAGNOSTIC_RESERVE_BYTES == MAX_EPISODE_BYTES


def test_diagnostic_reserve_is_a_real_cap_even_for_escaped_control_characters():
    from test_agent_episode_contract import started

    event = started()
    event["payload"]["prompt"] = "\x00" * 400000
    assert len(event["payload"]["prompt"].encode("utf-8")) < 1024 * 1024
    assert len(canonical_json(event)) > DIAGNOSTIC_RESERVE_BYTES
    with pytest.raises(AgentEpisodeError, match="diagnostics"):
        validate_episode_events([event], require_terminal=False)


def test_only_exact_synthetic_failure_shape_uses_diagnostic_reserve():
    from test_agent_episode_contract import append_event, request, started

    events = [started()]
    request(events)
    append_event(events, "model_failed", {"step": 0, "phase": "execute",
        "error_code": "worker_timeout", "input_token_count": None,
        "output_token_count": None, "latency_ms": None, "ttft_ms": None}, call_id="model-0")
    synthetic = episode_budget_usage(events)
    events[-1]["payload"]["latency_ms"] = 0
    measured = episode_budget_usage(events)
    assert measured["producer_bytes"] == synthetic["producer_bytes"] + len(canonical_json(events[-1])) + 1
    assert measured["total_bytes"] == len(canonical_json(events))


@pytest.mark.parametrize("mutation", ["float_fact", "integer_boolean"])
def test_pinned_corpus_identity_uses_typed_json_equality_before_writes(tmp_path, mutation):
    from daystrom_dml.services.agent_episode import run_local_episode

    scenario = deepcopy(load_episode_corpus()["scenarios"][0])
    if mutation == "float_fact":
        scenario["seeds"][0]["meta"]["claim_value"] = 8000.0
        scenario["tasks"][0]["truth"]["runbook_a.port"]["value"] = 8000.0
    else:
        scenario["seeds"][0]["meta"]["no_merge"] = 1
    with pytest.raises(ValueError, match="pinned"):
        run_local_episode(snapshot_directory=tmp_path / "absent-model",
            work_directory=tmp_path / "run", scenario=scenario, task=scenario["tasks"][0])
    assert not (tmp_path / "run").exists()


def test_actual_prior_wrong_answer_reaches_followup_model_as_untrusted_context(tmp_path):
    from test_agent_episode_runtime import ScriptedConsumer, final, tool
    from daystrom_dml.services.agent_episode import (
        _prepare_fixture, _read_records, run_episode_with_test_dependencies,
    )
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools

    corpus = load_episode_corpus()
    scenario = next(case for case in corpus["scenarios"]
                    if case["id"] == "self_reinforcing_error")
    directory = tmp_path / "feedback-authority"
    adapter, prepared = _prepare_fixture(directory, scenario, "feedback")
    try:
        bridge = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"],
            episode_id="feedback", seed_receipts=prepared["seed_receipts"],
            observation_records=list(prepared["seed_records"].values()),
            allowed_tools=("retrieve",), effective_time=corpus["effective_time"])
        evidence = prepared["seed_records"]["operator_verification"]["id"]
        wrong = {"claims": [{"key": "service.port", "value": 7000, "evidence_ids": [evidence]}]}
        first = run_episode_with_test_dependencies(
            consumer=ScriptedConsumer([tool(query="operator verification", top_k=10), final(wrong)]),
            toolbox=bridge, scenario=scenario, task=scenario["tasks"][0],
            seed_records=prepared["seed_records"], current_records=lambda: _read_records(directory),
            effective_time=corpus["effective_time"], episode_id="feedback-first")
        assert first["terminal"]["status"] == "completed"
        assert first["terminal"]["success"] is False
        context = {field: first["terminal"][field]
                   for field in ("episode_id", "task_id", "evidence_digest", "answer")}
        correct = {"claims": [{"key": "service.port", "value": 8000, "evidence_ids": [evidence]}]}
        consumer = ScriptedConsumer([tool(query="operator verification", top_k=10), final(correct)])
        second = run_episode_with_test_dependencies(
            consumer=consumer, toolbox=bridge, scenario=scenario, task=scenario["tasks"][1],
            seed_records=prepared["seed_records"], current_records=lambda: _read_records(directory),
            effective_time=corpus["effective_time"], episode_id="feedback-second", prior_context=context)
        initial = consumer.requests[0]["messages"]
        assert any("untrusted" in message["content"].lower()
                   and canonical_json(context).decode("utf-8") in message["content"] for message in initial)
        assert initial[-1] == {"role": "user", "content": scenario["tasks"][1]["prompt"]}
        assert second["events"][0]["payload"]["prior_context"] == context
        assert second["terminal"]["success"] is True
        assert second["terminal"]["verifier"]["repeat_opportunities"] == 1
        assert second["terminal"]["verifier"]["repeat_errors"] == 0
        assert second["terminal"]["execution_path"] == "test_injected"
        validate_episode_events(second["events"])
    finally:
        adapter.close()


def _validation_interruption_worker(connection, config):
    connection.send_bytes(canonical_json({'kind': 'prepared', 'prepared': config['test_prepared']}))
    assert connection.recv_bytes(128) == b'ack'
    for event in config['test_events']:
        connection.send_bytes(canonical_json({'kind': 'event', 'event': event}))
        assert connection.recv_bytes(128) == b'ack'
    time.sleep(10)


@pytest.mark.parametrize('boundary', ['before_rejection', 'after_rejection', 'actual_dispatch'])
def test_v2_supervisor_interruptions_never_invent_recovery_or_erase_dispatch(tmp_path, boundary):
    from daystrom_dml.services.agent_episode import _supervise
    from test_agent_episode_runtime import validation_case
    from daystrom_dml.contracts.agent_episode import VALIDATION_CONSUMER_PROFILE
    report, _, _ = validation_case(tmp_path)
    events = report['events']
    rejection = next(i for i, e in enumerate(events) if e['kind'] == 'tool_validation_rejected')
    end = rejection - 1 if boundary == 'before_rejection' else rejection
    if boundary == 'actual_dispatch':
        end = next(i for i, e in enumerate(events) if e['kind'] == 'tool_requested' and e['payload']['name'] == 'supersede')
    corpus = load_episode_corpus()
    scenario = next(s for s in corpus['scenarios'] if s['id'] == report['scenario_id'])
    config = {'episode_id': 'validation-test', 'execution_path': 'test_injected', 'scenario': scenario,
        'task': scenario['tasks'][0], 'limits': {**events[0]['payload']['limits'], 'wall_time_seconds': 2.0},
        'effective_time': corpus['effective_time'], 'previous_answers': None,
        'snapshot_directory': str(tmp_path / 'absent-model'), 'authority_directory': str(tmp_path / 'validation-authority'),
        'consumer_profile': VALIDATION_CONSUMER_PROFILE, 'test_prepared': report['prepared'], 'test_events': events[1:end + 1]}
    observed = _supervise(config, worker_target=_validation_interruption_worker)
    assert observed['terminal']['status'] == 'timeout'
    assert observed['terminal']['usage_unknown'] and observed['terminal']['effects_unknown']
    assert observed['terminal']['schema_version'] == 'dml-agent-terminal-v2'
    retained = [e for e in observed['events'] if e['kind'] == 'tool_validation_rejected']
    assert len(retained) == (0 if boundary == 'before_rejection' else 1)
    failed = [e for e in observed['events'] if e['kind'] == 'tool_failed']
    if boundary == 'actual_dispatch':
        assert len(failed) == 1 and failed[0]['payload']['effects'] == 'unknown'
    else:
        assert failed == []
    assert not observed['terminal']['success']


@pytest.mark.parametrize('boundary', ['before_rejection', 'after_rejection', 'actual_dispatch', 'foreign_v2'])
def test_v3_supervisor_interruptions_preserve_explicit_profile_and_uncertainty(tmp_path, boundary):
    from daystrom_dml.services.agent_episode import _supervise
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE
    from test_agent_episode_runtime import recovery_case, validation_case
    report, _, _ = (validation_case if boundary == 'foreign_v2' else recovery_case)(tmp_path)
    events = report['events']
    rejection = next(i for i, e in enumerate(events) if e['kind'] == 'tool_validation_rejected')
    end = rejection - 1 if boundary == 'before_rejection' else rejection
    if boundary == 'actual_dispatch':
        end = next(i for i, e in enumerate(events) if e['kind'] == 'tool_requested' and e['payload']['name'] == 'supersede')
    corpus = load_episode_corpus()
    scenario = next(s for s in corpus['scenarios'] if s['id'] == report['scenario_id'])
    config = {'episode_id': 'validation-test', 'execution_path': 'test_injected', 'scenario': scenario,
        'task': scenario['tasks'][0], 'limits': {**events[0]['payload']['limits'], 'wall_time_seconds': 2.0},
        'effective_time': corpus['effective_time'], 'previous_answers': None,
        'snapshot_directory': str(tmp_path / 'absent-model'), 'authority_directory': str(tmp_path / 'validation-authority'),
        'consumer_profile': RECOVERY_CONSUMER_PROFILE, 'test_prepared': report['prepared'], 'test_events': events[1:end + 1]}
    observed = _supervise(config, worker_target=_validation_interruption_worker)
    terminal = observed['terminal']
    assert terminal['consumer_profile'] == observed['events'][0]['payload']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
    assert terminal['status'] == ('runner_error' if boundary == 'foreign_v2' else 'timeout')
    assert terminal['usage_unknown'] and terminal['effects_unknown'] and not terminal['success']
    retained = [e for e in observed['events'] if e['kind'] == 'tool_validation_rejected']
    assert len(retained) == (0 if boundary in ('before_rejection', 'foreign_v2') else 1)
    failed = [e for e in observed['events'] if e['kind'] == 'tool_failed']
    assert len(failed) == (1 if boundary == 'actual_dispatch' else 0)
    validate_episode_events(observed['events'])
