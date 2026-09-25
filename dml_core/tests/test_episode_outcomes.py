"""Failure-inclusive totals and honest coverage, including interruption paths."""
from copy import deepcopy

import pytest

from daystrom_dml.contracts.agent_episode import AgentEpisodeError, validate_episode_events
from daystrom_dml.services.episode_outcomes import build_terminal, summarize_episode_outcomes
from daystrom_dml.services.evaluation import TASK_OUTCOME_VERSION, summarize_outcomes
from test_agent_episode_contract import (
    ANSWER, append_event, final_events, finish, request, started, verdict,
)


def failed_generation(*, task_id="failure", input_tokens=3):
    events = [started(task_id=task_id)]
    request(events)
    append_event(events, "model_failed", {
        "step": 0, "phase": "execute", "error_code": "generation_failed",
        "input_token_count": input_tokens, "output_token_count": None,
        "latency_ms": 4, "ttft_ms": None,
    }, call_id="model-0")
    terminal = build_terminal(events, verdict(False, measured=False), status="model_error",
                              latency_ms=9, retrieval_ms=0)
    append_event(events, "terminal", terminal)
    return events, terminal


def test_failed_generation_is_attempted_and_retains_known_input_with_unknown_output():
    events, terminal = failed_generation()
    validate_episode_events(events)
    assert terminal["success"] is False
    assert terminal["input_tokens"] == terminal["known_input_tokens"] == 3
    assert terminal["output_tokens"] is None
    assert terminal["known_output_tokens"] == 0
    assert terminal["unknown_output_calls"] == 1
    assert terminal["ttft_ms"] is None


def test_parent_kill_before_acknowledged_dispatch_never_becomes_zero_usage():
    events = [started()]
    request(events)
    terminal = build_terminal(events, verdict(False, measured=False), status="killed",
                              latency_ms=10, retrieval_ms=None, usage_unknown=True)
    append_event(events, "terminal", terminal)
    validate_episode_events(events)
    assert terminal["input_tokens"] is terminal["output_tokens"] is None
    assert terminal["known_input_tokens"] == terminal["known_output_tokens"] == 0
    assert terminal["unknown_input_calls"] == terminal["unknown_output_calls"] == 1
    assert terminal["effects_unknown"] is True
    assert terminal["retrieval_ms"] is None


def test_supervisor_unknown_dispatch_retains_prior_acknowledged_lower_bounds():
    events = final_events()
    terminal = build_terminal(events, verdict(False, measured=False), status="runner_error",
                              latency_ms=10, retrieval_ms=None, usage_unknown=True)
    assert terminal["input_tokens"] is terminal["output_tokens"] is None
    assert terminal["known_input_tokens"] == 3
    assert terminal["known_output_tokens"] == 2


def test_proven_compile_failure_has_zero_generation_and_full_uncompiled_request():
    reference = [started()]
    exact = request(reference)["payload"]["request"]
    events = [started()]
    append_event(events, "model_failed", {
        "step": 0, "phase": "compile", "error_code": "input_budget",
        "input_token_count": 0, "output_token_count": 0, "request": exact,
        "latency_ms": 1, "ttft_ms": None,
    }, call_id="model-0")
    terminal = build_terminal(events, verdict(False, measured=False), status="input_limit",
                              latency_ms=2, retrieval_ms=0)
    append_event(events, "terminal", terminal)
    validate_episode_events(events)
    assert terminal["input_tokens"] == terminal["output_tokens"] == 0


def test_mixed_success_and_failure_cost_never_excludes_failed_attempt():
    success = finish(final_events(task_id="success"))
    _, failure = failed_generation()
    summary = summarize_episode_outcomes([success, failure])
    assert summary["attempted_tasks"] == 2
    assert summary["completed_tasks"] == 1
    assert summary["task_success_rate"] == 0.5
    assert summary["input_tokens"] == 6
    assert summary["output_tokens"] is None
    assert summary["known_total_tokens"] == 8
    assert summary["total_tokens"] is None
    assert summary["tokens_per_completed_task"] is None
    assert summary["known_tokens_per_completed_task"] == 8
    assert summary["unknown_usage_tasks"] == 1
    assert summary["contradiction_rate"] is None
    assert summary["contradiction_coverage"] == {"rate": None, "known_numerator": 0,
                                                  "known_denominator": 1, "measured_tasks": 1,
                                                  "unknown_tasks": 1}
    assert summary["ttft_ms"] == {"samples": 0, "unknown_tasks": 2,
                                   "p50": None, "p95": None, "p99": None}


def test_no_success_never_produces_per_completed_task_cost():
    _, failure = failed_generation()
    summary = summarize_episode_outcomes([failure])
    assert summary["completed_tasks"] == 0
    assert summary["tokens_per_completed_task"] is None
    assert summary["known_tokens_per_completed_task"] is None


def test_measured_failure_tokens_included_in_complete_cost():
    success = finish(final_events(task_id="success"))
    failure = finish(final_events(task_id="wrong-answer"), success=False)
    summary = summarize_episode_outcomes([success, failure])
    assert summary["total_tokens"] == 10
    assert summary["tokens_per_completed_task"] == 10
    assert summary["task_success_rate"] == 0.5


def test_unknown_input_call_counts_are_distinct_from_known_lower_bound():
    _, failure = failed_generation(input_tokens=None)
    summary = summarize_episode_outcomes([failure])
    assert summary["known_input_tokens"] == 0
    assert summary["input_tokens"] is None
    assert summary["unknown_input_calls"] == 1


def test_duplicate_terminals_and_mixed_execution_paths_cannot_be_pooled():
    terminal = finish(final_events())
    with pytest.raises(AgentEpisodeError, match="exactly one terminal"):
        summarize_episode_outcomes([terminal, deepcopy(terminal)])
    concrete = finish(final_events(task_id="concrete", execution_path="live_local"))
    with pytest.raises(AgentEpisodeError, match="Cannot pool"):
        summarize_episode_outcomes([terminal, concrete])


@pytest.mark.parametrize("field,value", [
    ("input_tokens", False), ("output_tokens", -1), ("known_input_tokens", 3.0),
    ("latency_ms", float("nan")), ("retrieval_ms", float("inf")),
    ("ttft_ms", 0), ("success", 1), ("maintenance_tokens", 1),
    ("unknown_output_calls", 1), ("usage_unknown", True),
])
def test_invalid_or_falsely_complete_terminal_counters_rejected(field, value):
    terminal = finish(final_events())
    terminal[field] = value
    with pytest.raises(AgentEpisodeError):
        summarize_episode_outcomes([terminal])


def test_latency_must_cover_measured_serial_operations_and_retrieval_is_derived():
    events = final_events()
    with pytest.raises(AgentEpisodeError, match="serial measured"):
        build_terminal(events, verdict(), status="completed", answer=ANSWER,
                       latency_ms=1, retrieval_ms=0)
    with pytest.raises(AgentEpisodeError, match="Retrieval latency"):
        build_terminal(events, verdict(), status="completed", answer=ANSWER,
                       latency_ms=8, retrieval_ms=1)


def test_quality_numerator_and_denominator_must_be_jointly_measured():
    terminal = finish(final_events())
    terminal["verifier"]["contradictions"] = None
    with pytest.raises(AgentEpisodeError):
        summarize_episode_outcomes([terminal])
    terminal["verifier"]["contradictions"] = 2
    with pytest.raises(AgentEpisodeError):
        summarize_episode_outcomes([terminal])


def test_empty_run_has_no_success_or_quality_denominator():
    summary = summarize_episode_outcomes([])
    assert summary["attempted_tasks"] == 0
    assert summary["task_success_rate"] is None
    assert summary["contradiction_rate"] is None
    assert summary["execution_path"] is None


def test_original_outcome_v1_remains_unchanged_and_does_not_accept_nullable_required_costs():
    legacy = {"schema_version": TASK_OUTCOME_VERSION, "episode_id": "old", "task_id": "task",
              "success": False, "input_tokens": 2, "output_tokens": 3, "maintenance_tokens": 0,
              "latency_ms": 8, "retrieval_ms": 0}
    summary = summarize_outcomes([legacy])
    assert summary["schema_version"] == TASK_OUTCOME_VERSION
    assert summary["attempted_tasks"] == 1
    assert summary["total_tokens"] == 5
    legacy["output_tokens"] = None
    with pytest.raises(ValueError, match="output_tokens"):
        summarize_outcomes([legacy])


def test_v2_rejection_latency_is_costed_without_retrieval_or_effect_credit(tmp_path):
    from copy import deepcopy
    from test_agent_episode_runtime import validation_case
    from daystrom_dml.contracts.agent_episode import AgentEpisodeError
    report, _, _ = validation_case(tmp_path)
    terminal = report['terminal']
    rejected = next(e for e in report['events'] if e['kind'] == 'tool_validation_rejected')
    assert terminal['schema_version'] == 'dml-agent-terminal-v2'
    assert terminal['effects_unknown'] is False
    events = deepcopy(report['events'][:-1])
    target = next(e for e in events if e['kind'] == 'tool_validation_rejected')
    target['payload']['latency_ms'] = terminal['latency_ms'] + 1
    with pytest.raises(AgentEpisodeError, match='latency'):
        build_terminal(events, terminal['verifier'], status=terminal['status'], latency_ms=terminal['latency_ms'],
                       retrieval_ms=terminal['retrieval_ms'], answer=terminal['answer'])
    assert rejected['payload']['latency_ms'] >= 0


def test_v2_and_v1_terminal_summaries_cannot_be_pooled(tmp_path):
    from test_agent_episode_runtime import validation_case
    from daystrom_dml.contracts.agent_episode import AgentEpisodeError
    report, _, _ = validation_case(tmp_path)
    old = dict(report['terminal'], schema_version='dml-agent-terminal-v1', episode_id='old')
    old.pop('execution_protocol')
    old.pop('consumer_profile')
    with pytest.raises(AgentEpisodeError, match='protocol'):
        summarize_episode_outcomes([old, report['terminal']])


def test_v3_summary_preserves_profile_and_rejects_v2_pooling(tmp_path):
    from copy import deepcopy
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE, VALIDATION_CONSUMER_PROFILE
    from test_agent_episode_runtime import recovery_case
    report, _, _ = recovery_case(tmp_path)
    terminal = report['terminal']
    summary = summarize_episode_outcomes([terminal])
    assert terminal['consumer_profile'] == summary['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
    other = deepcopy(terminal)
    other.update(episode_id='another-episode', consumer_profile=VALIDATION_CONSUMER_PROFILE)
    with pytest.raises(AgentEpisodeError, match='consumer profiles'):
        summarize_episode_outcomes([terminal, other])
