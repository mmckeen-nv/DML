"""Focused producer evidence, exact dispatch, budgets and real-consumer plumbing."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import canonical_json, validate_episode_events
from daystrom_dml.contracts.model_input import (
    CompiledModelInput, ModelInputBudgetError, ModelInputIdentity, ModelInputRequest,
    SUPPORTED_CHAT_TEMPLATE_DIGEST,
)
from daystrom_dml.services.agent_episode import (
    EpisodeLimits, _prepare_fixture, _read_records,
    build_episode_request, run_episode_with_test_dependencies, run_local_episode,
)
from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools, canonical
from daystrom_dml.services.episode_verifiers import load_episode_corpus


class ScriptedConsumer:
    """Explicit fake outputs; they cannot enter the concrete live entry point."""
    def __init__(self, actions, *, compile_error=None, execute_error=None):
        self.actions = iter(actions)
        self.compile_error = compile_error
        self.execute_error = execute_error
        self.requests = []
        self.dispatched = []

    def compile(self, messages, tools, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload({"messages": messages, "tools": tools,
                                                  "output_reserved_tokens": output_reserved_tokens})
        self.requests.append(request.to_payload())
        if self.compile_error:
            raise self.compile_error
        identity = ModelInputIdentity(model_digest="1" * 64, tokenizer_digest="2" * 64,
            chat_template_digest=SUPPORTED_CHAT_TEMPLATE_DIGEST,
            runtime_identity="dml-model-input-runtime-v1:test-injected", model_window_tokens=32768)
        return CompiledModelInput(identity=identity, request_digest=request.request_digest,
            input_ids=(1, 2, 3, 4), attention_mask=(1, 1, 1, 1),
            output_reserved_tokens=output_reserved_tokens, model_window_tokens=32768,
            consumer_id="test-consumer", nonce="compile-" + str(len(self.requests)), auth_tag="0" * 64)

    def execute(self, artifact):
        self.dispatched.append(artifact)
        if self.execute_error:
            raise self.execute_error
        action = next(self.actions)
        text = action if type(action) is str else canonical(action)
        return SimpleNamespace(artifact_digest=artifact.artifact_digest, input_token_count=artifact.input_tokens,
            output_token_count=2, output_ids=(5, 6), text=text)


def tool(name="retrieve", **arguments):
    return {"schema_version": "dml-agent-action-v1", "kind": "tool", "name": name, "arguments": arguments}


def final(answer):
    return {"schema_version": "dml-agent-action-v1", "kind": "final", "answer": answer}


@pytest.fixture
def prepared(tmp_path):
    scenario = load_episode_corpus()["scenarios"][0]
    adapter, values = _prepare_fixture(tmp_path / "authority", scenario, "test")
    toolbox = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"], episode_id="test",
        seed_receipts=values["seed_receipts"], observation_records=list(values["seed_records"].values()),
        allowed_tools=("retrieve",))
    yield scenario, adapter, toolbox, values, tmp_path / "authority"
    adapter.close()


def run(prepared, consumer, **kwargs):
    scenario, _, toolbox, values, directory = prepared
    return run_episode_with_test_dependencies(consumer=consumer, toolbox=toolbox, scenario=scenario,
        task=scenario["tasks"][0], seed_records=values["seed_records"],
        current_records=lambda: _read_records(directory), **kwargs)


def test_actual_output_alone_selects_the_tool_and_entire_next_request_is_bound(prepared):
    consumer = ScriptedConsumer([tool(query="service port", top_k=10), final({"claims": []})])
    result = run(prepared, consumer)
    validate_episode_events(result["events"])
    assert result["terminal"]["execution_path"] == "test_injected"
    assert result["terminal"]["status"] == "completed"
    assert result["terminal"]["success"] is False
    assert result["terminal"]["input_tokens"] == 8
    assert result["terminal"]["output_tokens"] == 4
    assert result["terminal"]["ttft_ms"] is None
    requested = [e for e in result["events"] if e["kind"] == "model_requested"]
    completed = next(e for e in result["events"] if e["kind"] == "tool_completed")
    assert requested[0]["payload"]["request"] == consumer.requests[0]
    assert requested[1]["payload"]["request"]["messages"][-1]["content"] == completed["payload"]["model_result"]
    assert requested[1]["payload"]["request"]["messages"][:2] == requested[0]["payload"]["request"]["messages"]
    assert "PRIVATE OTHER TENANT" not in completed["payload"]["model_result"]
    assert all(artifact.input_ids == (1, 2, 3, 4) for artifact in consumer.dispatched)


@pytest.mark.parametrize("text", ["not JSON", "```json\n{}\n```", '{"kind":"tool"}'])
def test_invalid_model_output_is_preserved_and_charged(prepared, text):
    result = run(prepared, ScriptedConsumer([text]))
    assert result["terminal"]["status"] == "invalid_action"
    assert result["terminal"]["input_tokens"] == 4
    assert result["terminal"]["output_tokens"] == 2
    assert not any(e["kind"] == "tool_requested" for e in result["events"])
    assert next(e for e in result["events"] if e["kind"] == "model_completed")["payload"]["text"] == text


def test_compile_failure_has_proven_no_dispatch_and_execute_failure_has_unknown_output(prepared):
    consumer = ScriptedConsumer([], compile_error=ModelInputBudgetError("too large"))
    compile_report = run(prepared, consumer)
    assert compile_report["terminal"]["status"] == "input_limit"
    assert compile_report["terminal"]["input_tokens"] == compile_report["terminal"]["output_tokens"] == 0
    assert consumer.dispatched == []
    execute_report = run(prepared, ScriptedConsumer([], execute_error=RuntimeError("failed")))
    assert execute_report["terminal"]["status"] == "model_error"
    assert execute_report["terminal"]["input_tokens"] == 4
    assert execute_report["terminal"]["output_tokens"] is None
    assert execute_report["terminal"]["known_output_tokens"] == 0


def test_cumulative_input_refusal_is_replayable_before_dispatch(prepared):
    consumer = ScriptedConsumer([])
    report = run(prepared, consumer, limits=replace(EpisodeLimits(), max_input_tokens=3))
    assert report["terminal"]["status"] == "token_limit"
    assert consumer.dispatched == []
    rejection = next(e for e in report["events"] if e["kind"] == "admission_rejected")
    assert rejection["payload"]["limit"] == "input_tokens"
    assert rejection["payload"]["observed"] == 4
    assert rejection["payload"]["maximum"] == 3


def test_transcript_refusal_is_recorded_without_silent_truncation(prepared):
    consumer = ScriptedConsumer([])
    report = run(prepared, consumer, limits=replace(EpisodeLimits(), max_transcript_bytes=10))
    assert report["terminal"]["status"] == "transcript_limit"
    assert consumer.requests == []
    rejection = next(e for e in report["events"] if e["kind"] == "admission_rejected")
    assert rejection["payload"]["request"]["messages"][1]["content"] == prepared[0]["tasks"][0]["prompt"]


def test_allowed_tool_definitions_match_task_authority(prepared):
    corpus = load_episode_corpus()
    read = build_episode_request(corpus["scenarios"][0]["tasks"][0])
    change = build_episode_request(corpus["scenarios"][1]["tasks"][0])
    assert [t["function"]["name"] for t in read["tools"]] == ["retrieve"]
    assert [t["function"]["name"] for t in change["tools"]] == ["retrieve", "supersede"]
    report = run(prepared, ScriptedConsumer([tool("ingest", text="Unauthorized write")]))
    assert report["terminal"]["status"] == "invalid_action"
    assert not any(e["kind"] == "tool_requested" for e in report["events"])


def test_fixture_storage_ignores_environment_and_marks_offline_salience(tmp_path, monkeypatch):
    monkeypatch.setenv("DML_STORAGE_DIR", str(tmp_path / "forbidden"))
    monkeypatch.setenv("DML_CONFIG_PATH", str(tmp_path / "forbidden.yaml"))
    corpus = load_episode_corpus()
    scenario = next(s for s in corpus["scenarios"] if any(x["operation"] == "set_salience" for x in s["setup"]))
    adapter, values = _prepare_fixture(tmp_path / "owned", scenario, "test")
    try:
        assert adapter.storage_dir == tmp_path / "owned"
        assert not (tmp_path / "forbidden").exists()
        assert values["setup_records"][0]["source"] == "offline_fresh_fixture_initialization"
        assert values["seed_records"]["stale"]["salience"] == 1000
        original = next(r for r in values["seed_receipts"] if r["result"]["memory"]["id"] == values["seed_records"]["stale"]["id"])
        assert original["result"]["memory"]["salience"] == 1
    finally:
        adapter.close()


def test_concrete_path_attempts_missing_snapshot_and_keeps_unknown_startup_cost(tmp_path):
    scenario = load_episode_corpus()["scenarios"][0]
    report = run_local_episode(snapshot_directory=tmp_path / "missing", work_directory=tmp_path / "run",
        scenario=scenario, task=scenario["tasks"][0], limits=replace(EpisodeLimits(), wall_time_seconds=10))
    assert report["terminal"]["execution_path"] == "live_local"
    assert report["terminal"]["status"] == "runner_error"
    assert report["terminal"]["input_tokens"] is None
    assert report["terminal"]["output_tokens"] is None
    assert report["live_qualified"] is False
    assert report["prepared"]["seed_receipts"]
    assert report["events"][0]["payload"]["seed_receipts_digest"]


def test_real_local_consumer_generates_recorded_ids_without_claiming_trained_quality(tmp_path):
    from model_input_fixture import create_snapshot
    snapshot = create_snapshot(tmp_path / "snapshot", context_window=4096)
    scenario = load_episode_corpus()["scenarios"][0]
    report = run_local_episode(snapshot_directory=snapshot.path, work_directory=tmp_path / "run",
        scenario=scenario, task=scenario["tasks"][0],
        limits=replace(EpisodeLimits(), max_steps=1, output_tokens=4, wall_time_seconds=30))
    completed = [event for event in report["events"] if event["kind"] == "model_completed"]
    assert completed, report["terminal"]
    assert completed[0]["payload"]["output_token_count"] == len(completed[0]["payload"]["output_ids"])
    assert completed[0]["payload"]["input_token_count"] > 0
    assert report["terminal"]["execution_path"] == "live_local"
    assert report["live_qualified"] is False
    assert report["terminal"]["success"] is False
    validate_episode_events(report["events"])


def test_concrete_api_rejects_unpinned_or_oversized_scenario_before_writes(tmp_path):
    scenario = deepcopy(load_episode_corpus()["scenarios"][0])
    scenario["seeds"] *= 100
    with pytest.raises(ValueError, match="pinned"):
        run_local_episode(snapshot_directory=tmp_path / "missing", work_directory=tmp_path / "run",
            scenario=scenario, task=scenario["tasks"][0])
    assert not (tmp_path / "run").exists()


def test_injected_actions_drive_receipted_supersession_and_independent_verifier(tmp_path):
    scenario = load_episode_corpus()["scenarios"][1]
    adapter, values = _prepare_fixture(tmp_path / "authority", scenario, "supersession")
    try:
        toolbox = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"], episode_id="supersession",
            seed_receipts=values["seed_receipts"], observation_records=list(values["seed_records"].values()),
            allowed_tools=("retrieve", "supersede"))
        consumer = ScriptedConsumer([
            tool(query="user reply settings", top_k=2),
            tool("supersede", record_ref="r0", replacement_ref="r1", reason="Apply the current user setting."),
            final({"claims": [{"key": "reply.style", "value": "concise", "evidence_ids": [1]}]}),
        ])
        report = run_episode_with_test_dependencies(consumer=consumer, toolbox=toolbox, scenario=scenario,
            task=scenario["tasks"][0], seed_records=values["seed_records"],
            current_records=lambda: _read_records(tmp_path / "authority"))
        assert report["terminal"]["success"] is True, report["terminal"]["verifier"]
        assert report["terminal"]["execution_path"] == "test_injected"
        assert [e["payload"]["name"] for e in report["events"] if e["kind"] == "tool_completed"] == ["retrieve", "supersede"]
        assert report["terminal"]["input_tokens"] == 12
        assert report["terminal"]["output_tokens"] == 6
    finally:
        adapter.close()


def test_actual_wrong_prior_output_reaches_followup_as_untrusted_context(tmp_path):
    scenario = next(s for s in load_episode_corpus()["scenarios"] if any("repeat_from" in t for t in s["tasks"]))
    adapter, values = _prepare_fixture(tmp_path / "authority", scenario, "repeat")
    try:
        toolbox = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"], episode_id="repeat",
            seed_receipts=values["seed_receipts"], observation_records=list(values["seed_records"].values()),
            allowed_tools=("retrieve",))
        evidence_id = values["seed_records"]["operator_verification"]["id"]
        def execute(task, consumer, **kwargs):
            return run_episode_with_test_dependencies(consumer=consumer, toolbox=toolbox, scenario=scenario,
                task=task, seed_records=values["seed_records"],
                current_records=lambda: _read_records(tmp_path / "authority"), **kwargs)
        wrong = {"claims": [{"key": "service.port", "value": 9000, "evidence_ids": [evidence_id]}]}
        first = execute(scenario["tasks"][0], ScriptedConsumer([
            tool(query="operator verified service port", top_k=10), final(wrong)]), episode_id="first-episode")
        assert first["terminal"]["answer"] == wrong
        assert first["terminal"]["success"] is False
        context = {key: first["terminal"][key] for key in ("episode_id", "task_id", "evidence_digest", "answer")}
        corrected = {"claims": [{"key": "service.port", "value": 8000, "evidence_ids": [evidence_id]}]}
        consumer = ScriptedConsumer([tool(query="operator verified service port", top_k=10), final(corrected)])
        second = execute(scenario["tasks"][1], consumer, prior_context=context, episode_id="second-episode")
        assert second["events"][0]["payload"]["prior_context"] == context
        messages = consumer.requests[0]["messages"]
        assert messages[-1]["content"] == scenario["tasks"][1]["prompt"]
        assert "Untrusted" in messages[1]["content"]
        assert canonical_json(context).decode("utf-8") in messages[1]["content"]
        assert second["terminal"]["success"] is True, second["terminal"]["verifier"]
        assert second["terminal"]["verifier"]["repeat_opportunities"] == 1
        assert second["terminal"]["verifier"]["repeat_errors"] == 0
        with pytest.raises(ValueError, match="differ"):
            execute(scenario["tasks"][1], ScriptedConsumer([]), prior_context=context,
                    previous_answers={scenario["tasks"][0]["id"]: corrected})
    finally:
        adapter.close()


def test_live_fixture_identity_distinguishes_integer_and_float(tmp_path):
    scenario = deepcopy(load_episode_corpus()["scenarios"][0])
    scenario["seeds"][0]["meta"]["claim_value"] = 8000.0
    with pytest.raises(ValueError, match="pinned"):
        run_local_episode(snapshot_directory=tmp_path / "missing", work_directory=tmp_path / "run",
            scenario=scenario, task=scenario["tasks"][0])
    assert not (tmp_path / "run").exists()
