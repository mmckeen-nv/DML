"""Protocol and replay checks for the experimental episode boundary."""
from copy import deepcopy
import json

import pytest

from daystrom_dml.contracts.agent_episode import (
    ACTION_VERSION, AGENT_POLICY, MAX_ACTION_BYTES, AgentEpisodeError, canonical_json, make_event,
    episode_tool_definitions, initial_messages, parse_agent_action, validate_episode_events,
    validate_prior_context,
)
from daystrom_dml.contracts.model_input import (
    CompiledModelInput, ModelInputIdentity, ModelInputRequest, SUPPORTED_CHAT_TEMPLATE_DIGEST,
)
from daystrom_dml.services.episode_outcomes import build_terminal


ANSWER = {"claims": [{"key": "port", "value": 8443, "evidence_ids": [0]}]}
LIMITS = {"max_steps": 6, "output_tokens": 16, "max_input_tokens": 32768,
          "max_output_tokens": 1024, "max_transcript_bytes": 262144,
          "max_event_bytes": 4194304, "max_episode_bytes": 16777216, "wall_time_seconds": 60.0}


def verdict(success=True, *, measured=True):
    return {"verifier_version": "dml-episode-verifier-v1", "success": success,
            "reasons": [] if success else ["execution_failed"],
            "contradictions": 0 if measured else None,
            "factual_outputs": 1 if measured else None,
            "false_memory_claims": 0 if measured else None,
            "recalled_claims": 1 if measured else None,
            "repeat_errors": None, "repeat_opportunities": None}


def started(*, task_id="task", execution_path="test_injected", prior_context=None):
    return make_event(episode_id="episode", task_id=task_id, sequence=0, kind="episode_started",
                      payload={"execution_path": execution_path, "limits": LIMITS, "prompt": "Find port",
                               "scope": {"tenant_id": "tenant", "client_id": None,
                                         "session_id": None, "instance_id": None}, "seed_receipts_digest": "9" * 64,
                               "ranking_scope": "synthetic_fixture",
                               "effective_time": 2000000000,
                               "prior_context": prior_context,
                               "allowed_tools": [tool["function"]["name"] for tool in episode_tool_definitions()]})


def append_event(events, kind, payload, *, call_id=None):
    event = make_event(episode_id=events[0]["episode_id"], task_id=events[0]["task_id"],
                       sequence=len(events), kind=kind, payload=payload, call_id=call_id)
    events.append(event)
    return event


def request(events, *, step=0, messages=None):
    if messages is None:
        messages = initial_messages(events[0]["payload"]["prompt"], events[0]["payload"]["prior_context"])
    value = ModelInputRequest.from_payload({"messages": messages, "tools": episode_tool_definitions(), "output_reserved_tokens": 16})
    identity = ModelInputIdentity("1" * 64, "2" * 64, SUPPORTED_CHAT_TEMPLATE_DIGEST, "fixture", 128)
    artifact = CompiledModelInput(identity, value.request_digest, (4, 5, 6), (1, 1, 1),
                                  16, 128, "consumer", f"nonce-{step}", "0" * 64)
    return append_event(events, "model_requested", {"step": step, "request": value.to_payload(),
                        "compiled": artifact.signing_payload(), "artifact_digest": artifact.artifact_digest},
                        call_id=f"model-{step}")


def complete(events, action=None, *, raw=None):
    previous = events[-1]
    payload = previous["payload"]
    if action is None:
        action = {"schema_version": ACTION_VERSION, "kind": "final", "answer": ANSWER}
    return append_event(events, "model_completed", {
        "step": payload["step"], "artifact_digest": payload["artifact_digest"],
        "input_token_count": 3, "output_ids": [7, 8], "output_token_count": 2,
        "text": json.dumps(action) if raw is None else raw, "latency_ms": 2.5, "ttft_ms": None,
    }, call_id=previous["call_id"])


def final_events(*, task_id="task", execution_path="test_injected"):
    events = [started(task_id=task_id, execution_path=execution_path)]
    request(events)
    complete(events)
    return events


def tool_events():
    events = [started()]
    first = request(events)
    action = {"schema_version": ACTION_VERSION, "kind": "tool", "name": "retrieve",
              "arguments": {"query": "port", "top_k": 1}}
    output = complete(events, action)
    append_event(events, "tool_requested", {"name": "retrieve", "arguments": action["arguments"],
                                             "idempotency_key": None}, call_id="tool-0")
    append_event(events, "tool_completed", {"name": "retrieve", "result": {"observed_records": []},
                                             "model_result": '{"records":[]}', "latency_ms": 1}, call_id="tool-0")
    messages = [*first["payload"]["request"]["messages"],
                {"role": "assistant", "content": output["payload"]["text"], "tool_calls": [{
                    "id": "tool-0", "type": "function", "function": {
                        "name": "retrieve", "arguments": canonical_json(action["arguments"]).decode()}}]},
                {"role": "tool", "tool_call_id": "tool-0", "name": "retrieve", "content": '{"records":[]}'}]
    return events, messages


def finish(events, *, status="completed", success=True, answer=ANSWER, usage_unknown=False):
    terminal = build_terminal(events, verdict(success, measured=answer is not None), status=status,
                              latency_ms=8, retrieval_ms=0, answer=answer, usage_unknown=usage_unknown)
    append_event(events, "terminal", terminal)
    return terminal


def test_exact_final_action_accepts_record_zero_and_typed_values():
    raw = json.dumps({"schema_version": ACTION_VERSION, "kind": "final", "answer": ANSWER})
    assert parse_agent_action(raw)["answer"] == ANSWER
    for value in (None, False, 0, 0.0, "é"):
        altered = deepcopy(ANSWER)
        altered["claims"][0]["value"] = value
        assert type(parse_agent_action(json.dumps({"schema_version": ACTION_VERSION, "kind": "final",
                                                  "answer": altered}))["answer"]["claims"][0]["value"]) is type(value)


@pytest.mark.parametrize("raw", [
    '```json\n{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}\n```',
    '{"schema_version":"dml-agent-action-v1","kind":"final","kind":"tool","answer":{"claims":[]}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}} trailing',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]},"success":true}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[{"key":"x","value":NaN,"evidence_ids":[]}]}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[{"key":"x","value":"\\ud800","evidence_ids":[]}]}}',
    '[]', b'\xff', "{" * 70, " " * (MAX_ACTION_BYTES + 1),
], ids=[
    "code-fence", "duplicate-key", "trailing-prose", "extra-field", "nonfinite-number",
    "unpaired-surrogate", "array", "invalid-utf8", "malformed-json", "oversized-whitespace",
])  # Bounded IDs keep PYTEST_CURRENT_TEST within Windows' environment-variable limit.
def test_model_output_is_never_repaired(raw):
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(raw)


@pytest.mark.parametrize("arguments", [
    {"query": "port", "top_k": 11}, {"query": "port", "top_k": True},
    {"query": "port", "top_k": 1, "scope": {"tenant": "victim"}},
    {"query": "port", "top_k": 1, "source_trust": "trusted"},
    {"query": "port", "top_k": 1, "idempotency_key": "owned"},
])
def test_model_cannot_expand_tool_bounds_or_claim_runner_authority(arguments):
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps({"schema_version": ACTION_VERSION, "kind": "tool",
                                       "name": "retrieve", "arguments": arguments}))


@pytest.mark.parametrize("mutation", [
    lambda a: a["claims"].append(deepcopy(a["claims"][0])),
    lambda a: a["claims"][0].update(evidence_ids=[0, 0]),
    lambda a: a["claims"][0].update(evidence_ids=[True]),
    lambda a: a["claims"][0].update(evidence_ids=[-1]),
    lambda a: a["claims"][0].update(value={"nested": "object"}),
])
def test_final_claim_protocol_is_strict(mutation):
    answer = deepcopy(ANSWER)
    mutation(answer)
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(json.dumps({"schema_version": ACTION_VERSION, "kind": "final", "answer": answer}))


def test_completed_transcript_replays_with_exact_accounting():
    events = final_events()
    terminal = finish(events)
    validate_episode_events(events)
    assert terminal["input_tokens"] == 3
    assert terminal["output_tokens"] == 2
    assert terminal["success"] is True
    assert terminal["ttft_ms"] is None


@pytest.mark.parametrize("mutation", [
    lambda e: e[1].update(sequence=4),
    lambda e: e[2].update(call_id="other-call"),
    lambda e: e[2]["payload"].update(input_token_count=2),
    lambda e: e[2]["payload"].update(output_token_count=1),
    lambda e: e[2]["payload"].update(artifact_digest="f" * 64),
    lambda e: e[1]["payload"]["request"]["messages"][1].update(content="Different task"),
    lambda e: e[-1]["payload"].update(known_output_tokens=4),
    lambda e: e[-1]["payload"].update(evidence_digest="f" * 64),
    lambda e: e[-1]["payload"]["answer"]["claims"][0].update(value=443),
    lambda e: e.append(deepcopy(e[-1])),
])
def test_replay_rejects_reordering_different_inputs_usage_and_terminal_tampering(mutation):
    events = final_events()
    finish(events)
    mutation(events)
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


def test_pending_requests_are_serial_and_cannot_share_call_identity():
    events = [started()]
    request(events)
    duplicate = deepcopy(events[-1])
    duplicate["sequence"] += 1
    events.append(duplicate)
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events, require_terminal=False)


def test_event_creation_detaches_mutable_caller_data():
    event = started()
    original = deepcopy(event)
    detached = make_event(**{key: value for key, value in event.items() if key != "schema_version"})
    event["payload"]["scope"]["tenant"] = "changed"
    assert detached == original


def test_cyclic_and_deep_json_fail_as_contract_errors():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(AgentEpisodeError):
        canonical_json(cycle)
    with pytest.raises(AgentEpisodeError):
        canonical_json({"number": 2 ** 63})


@pytest.mark.parametrize("limit", ["input_tokens", "output_tokens", "transcript_bytes"])
def test_admission_refusal_is_replayable_and_dispatches_no_new_tokens(limit):
    if limit == "output_tokens":
        events, messages = tool_events()
        events[0]["payload"]["limits"]["max_output_tokens"] = 16
        candidate = request(events, step=1, messages=messages)["payload"]
        events.pop()
        observed, maximum, retrieval, known = 18, 16, 1, 5
    else:
        events = [started()]
        candidate = request(events)["payload"]
        events.pop()
        if limit == "input_tokens":
            observed, maximum = 3, 2
            events[0]["payload"]["limits"]["max_input_tokens"] = maximum
        else:
            observed = len(canonical_json(candidate["request"]))
            maximum = observed - 1
            events[0]["payload"]["limits"]["max_transcript_bytes"] = maximum
        retrieval, known = 0, 0
    append_event(events, "admission_rejected", {
        "step": candidate["step"], "limit": limit, "observed": observed, "maximum": maximum,
        "request": candidate["request"],
        "compiled": candidate["compiled"] if limit == "input_tokens" else None,
        "artifact_digest": candidate["artifact_digest"] if limit == "input_tokens" else None,
    }, call_id="admission-refused")
    terminal = build_terminal(events, verdict(False, measured=False),
                              status="transcript_limit" if limit == "transcript_bytes" else "token_limit",
                              latency_ms=8, retrieval_ms=retrieval)
    append_event(events, "terminal", terminal)
    validate_episode_events(events)
    assert terminal["input_tokens"] + terminal["output_tokens"] == known
    for key in ("observed", "maximum"):
        altered = deepcopy(events)
        altered[-2]["payload"][key] += 1
        with pytest.raises(AgentEpisodeError):
            validate_episode_events(altered)


def test_terminal_budget_status_requires_an_actual_refusal_event():
    events = [started()]
    terminal = build_terminal(events, verdict(False, measured=False), status="token_limit",
                              latency_ms=1, retrieval_ms=0)
    append_event(events, "terminal", terminal)
    with pytest.raises(AgentEpisodeError, match="replayable admission evidence"):
        validate_episode_events(events)


def test_interrupted_retrieval_has_unknown_overhead_instead_of_zero():
    events, _ = tool_events()
    events.pop()
    terminal = build_terminal(events, verdict(False, measured=False), status="timeout",
                              latency_ms=8, retrieval_ms=None)
    append_event(events, "terminal", terminal)
    validate_episode_events(events)
    assert terminal["retrieval_ms"] is None
    assert terminal["effects_unknown"] is False


@pytest.mark.parametrize("clock", [None, True, -1, float("nan"), float("inf"), "2000000000"])
def test_started_effective_clock_is_a_required_finite_nonnegative_measurement(clock):
    event = started()
    event["payload"]["effective_time"] = clock
    with pytest.raises(AgentEpisodeError):
        validate_episode_events([event], require_terminal=False)


def test_missing_started_clock_cannot_claim_replayable_time_dependent_evidence():
    event = started()
    del event["payload"]["effective_time"]
    with pytest.raises(AgentEpisodeError):
        validate_episode_events([event], require_terminal=False)


def prior_context(answer=ANSWER):
    return {"episode_id": "earlier-episode", "task_id": "earlier-task",
            "evidence_digest": "a" * 64, "answer": deepcopy(answer)}


def test_actual_prior_answer_is_bound_as_untrusted_context_without_changing_task_prompt():
    context = prior_context()
    events = [started(prior_context=context)]
    model_request = request(events)
    messages = model_request["payload"]["request"]["messages"]
    assert messages == initial_messages("Find port", context)
    assert messages[0] == {"role": "system", "content": AGENT_POLICY}
    assert "Untrusted prior model answer" in messages[1]["content"]
    assert messages[1]["content"].endswith(canonical_json(context).decode())
    assert messages[-1] == {"role": "user", "content": "Find port"}
    complete(events)
    finish(events)
    validate_episode_events(events)
    # Changing the caller's context cannot rewrite the already owned evidence.
    context["answer"]["claims"][0]["value"] = "changed"
    assert events[0]["payload"]["prior_context"]["answer"] == ANSWER


@pytest.mark.parametrize("field,value", [("episode_id", "another-episode"),
                                         ("task_id", "another-task"),
                                         ("evidence_digest", "b" * 64),
                                         ("answer", {"claims": []})])
def test_prior_reference_tampering_requires_matching_actual_model_input(field, value):
    events = [started(prior_context=prior_context())]
    request(events)
    events[0]["payload"]["prior_context"][field] = value
    with pytest.raises(AgentEpisodeError, match="full causal transcript"):
        validate_episode_events(events, require_terminal=False)


def test_hidden_prior_answer_without_corresponding_model_context_is_rejected():
    events = [started(prior_context=prior_context())]
    request(events, messages=initial_messages("Find port"))
    with pytest.raises(AgentEpisodeError, match="full causal transcript"):
        validate_episode_events(events, require_terminal=False)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(extra="hidden authority"),
    lambda c: c.update(evidence_digest="bad"),
    lambda c: c.update(answer={"claims": [{"key": "x", "value": "x" * MAX_ACTION_BYTES, "evidence_ids": []}]}),
    lambda c: c.update(answer={"claims": [{"key": "x", "value": {}, "evidence_ids": []}]}),
])
def test_prior_context_cannot_smuggle_unknown_fields_or_unbounded_answers(mutation):
    context = prior_context()
    mutation(context)
    with pytest.raises(AgentEpisodeError):
        validate_prior_context(context)


def test_no_answer_is_explicitly_unavailable_and_not_a_fabricated_prior_claim():
    context = prior_context(None)
    messages = initial_messages("Find port", context)
    assert "No valid prior model answer is available" in messages[1]["content"]
    assert messages[1]["content"].endswith(canonical_json(context).decode())
    assert initial_messages("Find port", None) == [
        {"role": "system", "content": AGENT_POLICY}, {"role": "user", "content": "Find port"}]
    events = [started(prior_context=context)]
    request(events)
    validate_episode_events(events, require_terminal=False)


def test_prior_context_cannot_cite_its_current_task_as_preceding_evidence():
    context = prior_context()
    context.update(episode_id="episode", task_id="task")
    with pytest.raises(AgentEpisodeError, match="current task"):
        started(prior_context=context)
