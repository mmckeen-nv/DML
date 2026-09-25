"""Focused producer evidence, exact dispatch, budgets and real-consumer plumbing."""
from copy import deepcopy
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import time
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


class ValidationScriptedConsumer(ScriptedConsumer):
    """Synthetic protocol control: every decision is explicitly supplied by the test."""

    def compile(self, messages, tools, *, output_reserved_tokens):
        artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
        return replace(artifact, identity=replace(artifact.identity,
            runtime_identity="dml-qwen-action-runtime-v2:" + "a" * 64))

    def execute(self, artifact):
        self.dispatched.append(artifact)
        action = next(self.actions)
        if callable(action):
            action = action(self.requests[-1])
        return SimpleNamespace(artifact_digest=artifact.artifact_digest, input_token_count=artifact.input_tokens,
            output_token_count=2, output_ids=(5, 6), text=canonical(action))


def validation_case(tmp_path, *, repeated=False, limits=None, prepare_hook=None, execute_hook=None,
                    consumer_profile="qwen2-action-json-validation-v2", consumer_factory=ValidationScriptedConsumer):
    from daystrom_dml.contracts.agent_episode import EXECUTION_PROTOCOL_V2
    scenario = next(s for s in load_episode_corpus()["scenarios"] if s["id"] == "superseded_preference")
    task = scenario["tasks"][0]
    directory = tmp_path / "validation-authority"
    adapter, values = _prepare_fixture(directory, scenario, "validation-test")
    bridge = SelectedProfileEpisodeTools(adapter, scope=scenario["scope"], episode_id="validation-test",
        seed_receipts=values["seed_receipts"], observation_records=list(values["seed_records"].values()),
        allowed_tools=("retrieve", "supersede"), execution_protocol=EXECUTION_PROTOCOL_V2)
    first_reference = []

    def invalid(request):
        if not first_reference:
            first_reference.append(json.loads(request["messages"][-1]["content"])["records"][0]["record_ref"])
        return tool("supersede", record_ref=first_reference[0], replacement_ref=first_reference[0], reason="test rejection")

    def valid(request):
        records = json.loads(request["messages"][-1]["content"])["records"]
        refs = {record["id"]: record["record_ref"] for record in records}
        return tool("supersede", record_ref=refs[values["seed_records"]["old"]["id"]],
            replacement_ref=refs[values["seed_records"]["current"]["id"]], reason="test selected replacement")

    answer = {"claims": [{"key": key, "value": fact["value"],
        "evidence_ids": [values["seed_records"][alias]["id"] for alias in fact["evidence_aliases"]]}
        for key, fact in task["truth"].items()]}
    actions = [tool(query="preference", top_k=1), invalid]
    actions += [invalid] * 5 if repeated else [tool(query="preference", top_k=10), valid, final(answer)]
    consumer = consumer_factory(actions)
    if prepare_hook:
        original = bridge.prepare
        bridge.prepare = lambda name, arguments, **kw: prepare_hook(bridge, original, name, arguments, kw)
    if execute_hook:
        original_execute = bridge.execute
        bridge.execute = lambda prepared: execute_hook(bridge, original_execute, prepared)
    try:
        report = run_episode_with_test_dependencies(consumer=consumer, toolbox=bridge, task=task, scenario=scenario,
            seed_records=values["seed_records"], current_records=lambda: _read_records(directory),
            consumer_profile=consumer_profile, episode_id="validation-test",
            limits=limits or EpisodeLimits(output_tokens=16))
        report.update(prepared=values, current_records=_read_records(directory), scenario_id=scenario["id"],
                      consumer_profile=consumer_profile)
        return report, consumer, bridge
    finally:
        adapter.close()


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


@pytest.mark.parametrize("profile", [None, True, [], "auto", "Qwen2-instruct-v1"])
def test_concrete_api_refuses_unknown_consumer_before_any_work(tmp_path, profile):
    scenario = load_episode_corpus()["scenarios"][0]
    with pytest.raises(ValueError, match="consumer profile"):
        run_local_episode(snapshot_directory=tmp_path / "missing", work_directory=tmp_path / "run",
            scenario=scenario, task=scenario["tasks"][0], consumer_profile=profile)
    assert not (tmp_path / "run").exists()


def test_qwen_profile_does_not_fall_back_to_gpt2_snapshot(tmp_path):
    from model_input_fixture import create_snapshot
    snapshot = create_snapshot(tmp_path / "snapshot", context_window=4096)
    scenario = load_episode_corpus()["scenarios"][0]
    report = run_local_episode(snapshot_directory=snapshot.path, work_directory=tmp_path / "run",
        scenario=scenario, task=scenario["tasks"][0], consumer_profile="qwen2-instruct-v1",
        limits=replace(EpisodeLimits(), max_steps=1, output_tokens=4, wall_time_seconds=30))
    assert report["consumer_profile"] == "qwen2-instruct-v1"
    assert report["terminal"]["status"] == "runner_error"
    assert report["terminal"]["execution_path"] == "live_local"
    assert not any(event["kind"] in ("model_requested", "model_completed") for event in report["events"])
    assert report["live_qualified"] is False
    validate_episode_events(report["events"])


def test_real_qwen_profile_generates_through_spawned_worker_without_quality_claim(tmp_path):
    from qwen_model_input_fixture import create_qwen_snapshot
    snapshot = create_qwen_snapshot(tmp_path / "snapshot", context_window=4096)
    scenario = load_episode_corpus()["scenarios"][0]
    report = run_local_episode(snapshot_directory=snapshot.path, work_directory=tmp_path / "run",
        scenario=scenario, task=scenario["tasks"][0], consumer_profile="qwen2-instruct-v1",
        limits=replace(EpisodeLimits(), max_steps=1, output_tokens=4, wall_time_seconds=30))
    requested = [event for event in report["events"] if event["kind"] == "model_requested"]
    completed = [event for event in report["events"] if event["kind"] == "model_completed"]
    assert len(requested) == len(completed) == 1, report["terminal"]
    assert requested[0]["payload"]["compiled"]["identity"]["model_window_tokens"] == 4096
    assert completed[0]["payload"]["output_token_count"] == len(completed[0]["payload"]["output_ids"]) > 0
    assert completed[0]["payload"]["input_token_count"] > 0
    assert report["consumer_profile"] == "qwen2-instruct-v1"
    assert report["terminal"]["execution_path"] == "live_local"
    assert report["live_qualified"] is report["terminal"]["success"] is False
    validate_episode_events(report["events"])


def test_default_profile_does_not_infer_qwen_from_snapshot(tmp_path):
    from qwen_model_input_fixture import create_qwen_snapshot
    snapshot = create_qwen_snapshot(tmp_path / "snapshot", context_window=4096)
    scenario = load_episode_corpus()["scenarios"][0]
    report = run_local_episode(snapshot_directory=snapshot.path, work_directory=tmp_path / "run",
        scenario=scenario, task=scenario["tasks"][0],
        limits=replace(EpisodeLimits(), max_steps=1, output_tokens=4, wall_time_seconds=30))
    assert report["consumer_profile"] == "gpt2-v1"
    assert report["terminal"]["status"] == "runner_error"
    assert not any(event["kind"] in ("model_requested", "model_completed") for event in report["events"])
    assert report["live_qualified"] is False


def _copy_snapshot_worker(connection, config):
    """Real snapshot admission, then a controlled worker exit; no learned quality."""
    from daystrom_dml.services.agent_episode import _run_loop
    from daystrom_dml.contracts.agent_episode import make_event
    from daystrom_dml.services.model_input_snapshot import verify_local_snapshot
    from daystrom_dml.services.qwen_model_snapshot import verify_qwen_snapshot

    verify = verify_local_snapshot if config["consumer_profile"] == "gpt2-v1" else verify_qwen_snapshot
    snapshot = verify(config["snapshot_directory"])
    snapshot.validate_integrity()
    if config["exit_mode"] == "timeout":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(config["marker"]).write_text(json.dumps({
        "scratch": tempfile.gettempdir(), "private_copy": str(snapshot.path), "pid": os.getpid(),
        "environment": {name: os.environ[name] for name in ("TMPDIR", "TEMP", "TMP")},
    }), encoding="utf-8")
    if config["exit_mode"] == "killed":
        if hasattr(signal, "SIGKILL"):
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(73)
    if config["exit_mode"] == "timeout":
        while True:
            time.sleep(1)
    sequence = 1

    def send(value):
        connection.send_bytes(canonical_json(value))
        assert connection.recv_bytes(128) == b"ack"

    def emit(kind, call_id, payload):
        nonlocal sequence
        send({"kind": "event", "event": make_event(episode_id=config["episode_id"],
            task_id=config["task"]["id"], sequence=sequence, kind=kind, call_id=call_id, payload=payload)})
        sequence += 1

    send({"kind": "prepared", "prepared": config["prepared"]})
    outcome = _run_loop(ScriptedConsumer([final({"claims": []})]), SimpleNamespace(allowed_tools=("retrieve",)),
        task=config["task"], limits=EpisodeLimits(**config["limits"]), emit=emit)
    snapshot.close()
    send({"kind": "finished", "outcome": outcome})
    connection.close()


@pytest.fixture(params=["gpt2-v1", "qwen2-instruct-v1"])
def cleanup_config(tmp_path, request):
    from model_input_fixture import create_snapshot
    from qwen_model_input_fixture import create_qwen_snapshot
    from test_agent_episode_adversarial import _supervisor_config

    create = create_snapshot if request.param == "gpt2-v1" else create_qwen_snapshot
    snapshot = create(tmp_path / "source", context_window=4096)
    config = _supervisor_config(tmp_path, seconds=10)
    config.update(snapshot_directory=str(snapshot.path), consumer_profile=request.param,
                  marker=str(tmp_path / "copy-admitted.json"))
    adapter, config["prepared"] = _prepare_fixture(
        config["authority_directory"], config["scenario"], config["episode_id"])
    adapter.close()
    return config


@pytest.mark.parametrize("exit_mode", ["timeout", "killed", "finished"])
def test_parent_reclaims_only_owned_scratch_after_real_snapshot_worker_exit(
        tmp_path, monkeypatch, cleanup_config, exit_mode):
    from daystrom_dml.services import agent_episode

    config = {**cleanup_config, "exit_mode": exit_mode}
    parent_tempdir = tempfile.tempdir
    parent_environment = {name: os.environ.get(name) for name in ("TMPDIR", "TEMP", "TMP")}
    preserved = []
    for directory in (tmp_path / "shared-snapshot", Path(config["authority_directory"]),
                      tmp_path / "evidence", tmp_path / "other-worker"):
        directory.mkdir(exist_ok=True)
        sentinel = directory / "sentinel"
        sentinel.write_bytes(directory.name.encode())
        preserved.append((sentinel, sentinel.read_bytes()))
    source = Path(config["snapshot_directory"])
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    remove = agent_episode._remove_worker_scratch
    removed = []

    def after_confirmed_exit(directory, identity):
        marker = json.loads(Path(config["marker"]).read_text(encoding="utf-8"))
        assert not any(child.pid == marker["pid"] for child in multiprocessing.active_children())
        assert Path(directory) == Path(marker["scratch"])
        assert Path(marker["private_copy"]).parent == Path(directory)
        assert set(marker["environment"].values()) == {directory}
        if exit_mode != "finished":
            assert Path(marker["private_copy"]).is_dir()
            assert (Path(marker["private_copy"]) / "model.safetensors").read_bytes() == original["model.safetensors"]
        remove(directory, identity)
        removed.append(directory)

    monkeypatch.setattr(agent_episode, "_remove_worker_scratch", after_confirmed_exit)
    report = agent_episode._supervise(config, worker_target=_copy_snapshot_worker)
    assert report["terminal"]["status"] == {"timeout": "timeout", "killed": "killed", "finished": "completed"}[exit_mode]
    assert report["live_qualified"] is report["terminal"]["success"] is False
    assert len(removed) == 1 and not Path(removed[0]).exists()
    assert tempfile.tempdir == parent_tempdir
    assert {name: os.environ.get(name) for name in parent_environment} == parent_environment
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert all(path.read_bytes() == payload for path, payload in preserved)
    validate_episode_events(report["events"])


@pytest.mark.parametrize("failure", ["cleanup", "termination"])
def test_supervisor_refuses_success_when_cleanup_or_death_is_uncertain(
        tmp_path, monkeypatch, cleanup_config, failure):
    from daystrom_dml.services import agent_episode
    from multiprocessing.process import BaseProcess

    config = {**cleanup_config, "exit_mode": "finished"}
    remove = agent_episode._remove_worker_scratch
    attempted = []

    def cannot_remove(directory, identity):
        attempted.append(directory)
        raise PermissionError("Injected cleanup refusal")

    with monkeypatch.context() as patch:
        patch.setattr(agent_episode, "_remove_worker_scratch", cannot_remove)
        if failure == "termination":
            patch.setattr(BaseProcess, "is_alive", lambda self: True)
        report = agent_episode._supervise(config, worker_target=_copy_snapshot_worker)
    marker = json.loads(Path(config["marker"]).read_text(encoding="utf-8"))
    scratch = Path(marker["scratch"])
    try:
        assert report["terminal"]["status"] == "runner_error"
        assert report["terminal"]["success"] is False
        assert any(event["kind"] == "model_completed" for event in report["events"])
        assert scratch.is_dir()
        assert bool(attempted) is (failure == "cleanup")
        assert not any(child.pid == marker["pid"] for child in multiprocessing.active_children())
        validate_episode_events(report["events"])
    finally:
        identity = scratch.lstat()
        remove(scratch, (identity.st_dev, identity.st_ino))


def test_v2_rejection_then_new_model_decision_commits_once_with_full_transcript(tmp_path):
    from daystrom_dml.contracts.agent_episode import VALIDATION_MODEL_RESULT
    report, consumer, bridge = validation_case(tmp_path)
    assert report['terminal']['success'] is True
    assert report['terminal']['status'] == 'completed'
    assert report['terminal']['input_tokens'] == 20 and report['terminal']['output_tokens'] == 10
    rejected = [e for e in report['events'] if e['kind'] == 'tool_validation_rejected']
    assert len(rejected) == 1 and rejected[0]['call_id'] == 'tool-1'
    assert rejected[0]['payload']['model_result'] == VALIDATION_MODEL_RESULT
    assert consumer.requests[2]['messages'][-1]['content'] == VALIDATION_MODEL_RESULT
    assert consumer.requests[2]['messages'][-2]['content'] == report['events'][6]['payload']['text']
    mutations = [e for e in report['events'] if e['kind'] == 'tool_completed' and e['payload']['name'] == 'supersede']
    assert len(mutations) == 1
    assert set(bridge._keys) == {'episode:validation-test:tool-3'}
    assert not any(e['kind'] == 'tool_requested' and e['call_id'] == 'tool-1' for e in report['events'])
    validate_episode_events(report['events'])


def test_v2_repeated_invalid_decisions_exhaust_existing_steps_without_retry(tmp_path):
    report, consumer, bridge = validation_case(tmp_path, repeated=True)
    assert report['terminal']['status'] == 'step_limit'
    assert len(consumer.dispatched) == 6
    assert sum(e['kind'] == 'tool_validation_rejected' for e in report['events']) == 5
    assert sum(e['kind'] == 'tool_requested' for e in report['events']) == 1
    assert bridge._keys == {} and len(bridge._prepared) == 1
    assert report['terminal']['effects_unknown'] is False


@pytest.mark.parametrize('kind', ['generic', 'forged', 'subclass'])
def test_v2_only_owned_exact_preparation_rejection_can_recover(tmp_path, kind):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    class Derived(EpisodeToolValidationRejected):
        pass
    def reject(bridge, original, name, arguments, kw):
        if name != 'supersede':
            return original(name, arguments, **kw)
        if kind == 'generic':
            raise ValueError('not whitelisted')
        cls = Derived if kind == 'subclass' else EpisodeToolValidationRejected
        raise cls(bridge._rejection_owner if kind == 'subclass' else object(), arguments, kw['call_id'])
    report, consumer, _ = validation_case(tmp_path, prepare_hook=reject)
    assert report['terminal']['status'] == 'invalid_action'
    assert len(consumer.dispatched) == 2
    assert not any(e['kind'] == 'tool_validation_rejected' for e in report['events'])


def test_v2_exact_trusted_type_from_execution_is_terminal_with_unknown_effects(tmp_path):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    def fail(bridge, original, prepared):
        if prepared.name == 'supersede':
            raise EpisodeToolValidationRejected(bridge._rejection_owner, prepared.arguments, 'tool-3')
        return original(prepared)
    report, consumer, _ = validation_case(tmp_path, execute_hook=fail)
    assert report['terminal']['status'] == 'tool_error' and report['terminal']['effects_unknown'] is True
    assert len(consumer.dispatched) == 4
    assert sum(e['kind'] == 'tool_validation_rejected' for e in report['events']) == 1


@pytest.mark.parametrize('budget', ['input', 'output', 'transcript'])
def test_v2_recovery_next_request_respects_existing_budget(tmp_path, budget):
    baseline, consumer, _ = validation_case(tmp_path / 'baseline')
    limits = EpisodeLimits(output_tokens=16)
    if budget == 'input':
        limits = replace(limits, max_input_tokens=8)
    elif budget == 'output':
        limits = replace(limits, max_output_tokens=19)
    else:
        size = len(canonical_json(consumer.requests[2]))
        assert size > len(canonical_json(consumer.requests[1]))
        limits = replace(limits, max_transcript_bytes=size - 1)
    report, admitted, _ = validation_case(tmp_path / 'bounded', limits=limits)
    assert report['terminal']['status'] == ('transcript_limit' if budget == 'transcript' else 'token_limit')
    assert len(admitted.dispatched) == 2
    assert sum(e['kind'] == 'tool_validation_rejected' for e in report['events']) == 1
    assert report['events'][-2]['kind'] == 'admission_rejected'


class RecoveryScriptedConsumer(ValidationScriptedConsumer):
    """Synthetic two-byte token accounting; real tokenizer controls are separate."""
    def compile(self, messages, tools, *, output_reserved_tokens):
        artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
        ids = tuple(canonical_json(self.requests[-1])[::2])
        return replace(artifact, input_ids=ids, attention_mask=(1,) * len(ids),
            identity=replace(artifact.identity, runtime_identity='dml-qwen-action-runtime-v3:' + 'b' * 64))


def recovery_case(tmp_path, **options):
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE
    return validation_case(tmp_path, consumer_profile=RECOVERY_CONSUMER_PROFILE,
                           consumer_factory=RecoveryScriptedConsumer, **options)


def test_v3_guidance_is_in_every_charged_request_and_model_still_owns_recovery(tmp_path):
    from daystrom_dml.contracts import agent_episode as contract
    report, consumer, bridge = recovery_case(tmp_path)
    assert report['terminal']['success'] is True
    expected = contract.AGENT_POLICY + '\n\n' + contract.RECOVERY_GUIDANCE
    assert all(request['messages'][0] == {'role': 'system', 'content': expected} for request in consumer.requests)
    assert len(consumer.dispatched) == 5
    assert report['terminal']['input_tokens'] == sum((len(canonical_json(request)) + 1) // 2 for request in consumer.requests)
    assert set(bridge._keys) == {'episode:validation-test:tool-3'}
    assert consumer.requests[2]['messages'][-1]['content'] == contract.VALIDATION_MODEL_RESULT
    assert consumer.requests[2]['messages'][:4] == consumer.requests[1]['messages']
    validate_episode_events(report['events'])


@pytest.mark.parametrize('budget', ['input', 'transcript'])
def test_v3_guidance_counts_against_original_limits_without_allowance(tmp_path, budget):
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE
    task = next(s for s in load_episode_corpus()['scenarios'] if s['id'] == 'superseded_preference')['tasks'][0]
    limits = EpisodeLimits(output_tokens=16)
    old = build_episode_request(task, limits=limits)
    new = build_episode_request(task, limits=limits, consumer_profile=RECOVERY_CONSUMER_PROFILE)
    assert len(canonical_json(new)) > len(canonical_json(old))
    maximum = (len(canonical_json(old)) + 1) // 2 if budget == 'input' else len(canonical_json(old))
    limits = replace(limits, **{('max_input_tokens' if budget == 'input' else 'max_transcript_bytes'): maximum})
    report, consumer, bridge = recovery_case(tmp_path, limits=limits)
    assert report['terminal']['status'] == ('token_limit' if budget == 'input' else 'transcript_limit')
    assert not consumer.dispatched and not bridge._keys
    rejected = report['events'][-2]
    assert rejected['kind'] == 'admission_rejected'
    expected = (len(canonical_json(new)) + 1) // 2 if budget == 'input' else len(canonical_json(new))
    assert rejected['payload']['observed'] == expected
    assert rejected['payload']['maximum'] == maximum
    assert report['terminal']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE


@pytest.mark.parametrize('error', ['compile', 'cross_profile'])
def test_v3_compile_failure_or_cross_profile_consumer_retains_selected_boundary(tmp_path, error):
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE
    factory = (lambda actions: RecoveryScriptedConsumer(actions, compile_error=ValueError('test refusal'))) if error == 'compile' else ValidationScriptedConsumer
    report, consumer, bridge = validation_case(tmp_path, consumer_profile=RECOVERY_CONSUMER_PROFILE,
                                               consumer_factory=factory)
    assert report['events'][0]['payload']['consumer_profile'] == report['terminal']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
    assert report['terminal']['status'] == ('model_error' if error == 'compile' else 'runner_error')
    assert report['terminal']['usage_unknown'] is (error == 'cross_profile')
    assert not consumer.dispatched and not bridge._keys
    validate_episode_events(report['events'])


def test_v3_model_can_still_finish_unsuccessfully_after_rejection_without_forced_work(tmp_path):
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE
    def factory(actions):
        return RecoveryScriptedConsumer([*actions[:2], final({'claims': []})])
    report, consumer, bridge = validation_case(tmp_path, consumer_profile=RECOVERY_CONSUMER_PROFILE,
                                               consumer_factory=factory)
    assert report['terminal']['status'] == 'completed' and not report['terminal']['success']
    assert len(consumer.dispatched) == 3 and not bridge._keys
    assert sum(e['kind'] == 'tool_validation_rejected' for e in report['events']) == 1
    assert not any(e['kind'] == 'tool_requested' and e['payload']['name'] == 'supersede' for e in report['events'])
