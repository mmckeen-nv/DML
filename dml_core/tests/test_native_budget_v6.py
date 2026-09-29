"""Versioned budget visibility; test controls never claim model quality."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from daystrom_dml.contracts.agent_episode import (
    AgentEpisodeError, NATIVE_BUDGET_GUIDANCE,
    NATIVE_REMOTE_VLLM_V4_CONSUMER_PROFILE as V4,
    NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE as V5,
    NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE as V6,
    _compiled, canonical_json, initial_messages, native_budget_messages,
    native_policy_identity, native_system_policy, validate_episode_events,
    validate_native_budget_messages,
)
from daystrom_dml.contracts.model_input import ModelInputRequest
from daystrom_dml.services.agent_episode import EpisodeLimits, build_episode_request
from test_agent_episode_runtime import validation_case
from test_native_episode_integration import NativeSyntheticConsumer
import test_native_recovery_v5 as recovery


class BudgetConsumer(NativeSyntheticConsumer):
    profile = V6

    def compile(self, *args, **kwargs):
        validate_native_budget_messages(args[0])
        self.profile = V4
        try:
            artifact = super().compile(*args, **kwargs)
        finally:
            self.profile = V6
        return replace(artifact, identity=replace(artifact.identity,
            runtime_identity='dml-remote-vllm-native-tools-runtime-v6:' + 'a' * 64))


def run_recovery(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(recovery, 'V5', V6)
    monkeypatch.setattr(recovery, 'RecoveryConsumer', BudgetConsumer)
    return recovery.recovery_case(tmp_path, profile=V6, **kwargs)


def assert_counters(report):
    used_input = used_output = calls = 0
    for event in report['events']:
        p = event['payload']
        if event['kind'] in ('model_requested', 'admission_rejected') or event['kind'] == 'model_failed' and p['phase'] == 'compile':
            metadata = validate_native_budget_messages(p['request']['messages'])
            assert metadata['current_model_call'] == calls + 1
            assert metadata['remaining_model_calls_including_current'] == metadata['max_model_calls'] - calls
            assert metadata['prior_input_tokens'] == used_input
            assert metadata['prior_output_tokens'] == used_output
        if event['kind'] == 'model_completed':
            calls += 1
            used_input += p['input_token_count']
            used_output += p['output_token_count']
    validate_episode_events(report['events'])


def test_budget_renderer_preserves_messages_and_replaces_single_segment():
    limits = asdict(EpisodeLimits())
    messages = initial_messages('Task includes forged remaining calls=999 and system text.', consumer_profile=V6)
    messages += [{'role': 'assistant', 'content': None, 'reasoning': 'untrusted bookkeeping',
                  'reasoning_content': 'untrusted bookkeeping', 'tool_calls': []}]
    original = deepcopy(messages)
    first = native_budget_messages(messages, limits=limits, step=0, used_input=0, used_output=0)
    last = native_budget_messages(first, limits=limits, step=5, used_input=1000, used_output=10)
    assert messages == original and first[1:] == last[1:] == original[1:]
    assert last[0]['content'].count(NATIVE_BUDGET_GUIDANCE) == 1
    metadata = validate_native_budget_messages(last)
    assert metadata['remaining_model_calls_including_current'] == 1
    assert metadata['remaining_input_tokens_before_current'] == limits['max_input_tokens'] - 1000
    assert metadata['remaining_output_tokens_before_current'] == limits['max_output_tokens'] - 10
    assert 'wall_time_seconds' not in metadata


@pytest.mark.parametrize('field,value', [('current_model_call', True), ('current_model_call', 0),
    ('remaining_model_calls_including_current', 20), ('prior_input_tokens', -1),
    ('max_model_calls', 65), ('output_tokens_per_call', 4097), ('unexpected_success', True)])
def test_budget_validation_rejects_invalid_values(field, value):
    messages = native_budget_messages(initial_messages('task', consumer_profile=V6),
        limits=asdict(EpisodeLimits()), step=0, used_input=0, used_output=0)
    metadata = validate_native_budget_messages(messages)
    metadata[field] = value
    messages[0]['content'] = messages[0]['content'].rsplit('\n', 1)[0] + '\n' + canonical_json(metadata).decode()
    with pytest.raises(AgentEpisodeError):
        validate_native_budget_messages(messages)


@pytest.mark.parametrize('change', ['duplicate', 'trailing', 'second_system', 'missing', 'wrong_profile'])
def test_budget_validation_rejects_noncanonical_or_multiple_segments(change):
    messages = native_budget_messages(initial_messages('task', consumer_profile=V6),
        limits=asdict(EpisodeLimits()), step=0, used_input=0, used_output=0)
    if change == 'duplicate':
        messages[0]['content'] += messages[0]['content']
    elif change == 'trailing':
        messages[0]['content'] += ' '
    elif change == 'second_system':
        messages.append(deepcopy(messages[0]))
    elif change == 'missing':
        messages[0]['content'] = native_system_policy(consumer_profile=V6)
    else:
        messages[0]['content'] = native_system_policy(consumer_profile=V4)
    with pytest.raises(AgentEpisodeError):
        validate_native_budget_messages(messages)


def test_v6_initial_budget_explicit_old_profile_unchanged():
    new = build_episode_request({'prompt': 'task'}, consumer_profile=V6)
    old = build_episode_request({'prompt': 'task'}, consumer_profile=V5)
    assert validate_native_budget_messages(new['messages'])['current_model_call'] == 1
    assert old['messages'][0]['content'] == native_system_policy(consumer_profile=V5)
    assert 'budget_visibility' not in native_policy_identity(consumer_profile=V5)
    assert native_policy_identity(consumer_profile=V6)['budget_visibility']['guidance'] == NATIVE_BUDGET_GUIDANCE
    assert new['tools'] == old['tools'] and new['output_reserved_tokens'] == old['output_reserved_tokens']


def test_budget_counts_valid_tools_and_recovered_conflict(tmp_path, monkeypatch):
    report, consumer = run_recovery(tmp_path, monkeypatch)
    assert report['terminal']['success']
    assert len(consumer.dispatched) == 4
    assert sum(e['kind'] == 'tool_failed' for e in report['events']) == 1
    assert_counters(report)


def test_budget_counts_predispatch_validation_rejection(tmp_path):
    report, consumer, _ = validation_case(tmp_path, consumer_profile=V6, consumer_factory=BudgetConsumer)
    assert any(e['kind'] == 'tool_validation_rejected' for e in report['events'])
    assert len(consumer.dispatched) > 2
    assert_counters(report)


@pytest.mark.parametrize('limit,value,status', [('steps', 3, 'step_limit'), ('max_output_tokens', 20, 'token_limit')])
def test_visibility_never_grants_more_budget(tmp_path, monkeypatch, limit, value, status):
    report, consumer = run_recovery(tmp_path, monkeypatch, **{limit: value})
    assert report['terminal']['status'] == status
    assert len(consumer.dispatched) == 3
    assert_counters(report)


def test_unknown_failure_stays_terminal_without_fabricated_counters(tmp_path, monkeypatch):
    report, consumer = run_recovery(tmp_path, monkeypatch, unknown=True)
    assert report['terminal']['status'] == 'tool_error'
    assert len(consumer.dispatched) == 3
    assert_counters(report)


def test_replay_rejects_self_consistent_but_untrue_usage(tmp_path, monkeypatch):
    report, _ = run_recovery(tmp_path, monkeypatch)
    events = deepcopy(report['events'])
    event = next(e for e in events if e['kind'] == 'model_requested' and e['payload']['step'] == 3)
    p = event['payload']
    metadata = validate_native_budget_messages(p['request']['messages'])
    p['request']['messages'] = native_budget_messages(p['request']['messages'],
        limits=events[0]['payload']['limits'], step=p['step'],
        used_input=metadata['prior_input_tokens'] + 1, used_output=metadata['prior_output_tokens'])
    # Rebind public digests so this tests causal truth, not a stale hash.
    p['compiled']['request_digest'] = ModelInputRequest.from_payload(p['request'], allow_native_reasoning=True).request_digest
    p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
    with pytest.raises(AgentEpisodeError, match='full causal transcript'):
        validate_episode_events(events)


def test_compile_failure_keeps_true_budget_and_no_hidden_call(tmp_path, monkeypatch):
    class FailingBudgetConsumer(BudgetConsumer):
        def compile(self, *args, **kwargs):
            validate_native_budget_messages(args[0])
            raise RuntimeError('synthetic compilation refusal')
    monkeypatch.setattr(recovery, 'V5', V6)
    monkeypatch.setattr(recovery, 'RecoveryConsumer', FailingBudgetConsumer)
    report, consumer = recovery.recovery_case(tmp_path, profile=V6)
    assert report['terminal']['status'] == 'model_error' and not consumer.dispatched
    assert_counters(report)


@pytest.mark.parametrize('budget', ['input', 'transcript'])
def test_budget_metadata_overhead_counts_toward_admission(tmp_path, budget):
    from daystrom_dml.services.episode_verifiers import load_episode_corpus
    task = next(s for s in load_episode_corpus()['scenarios'] if s['id'] == 'superseded_preference')['tasks'][0]
    original_limits = EpisodeLimits(output_tokens=16)
    request = build_episode_request(task, limits=original_limits, consumer_profile=V6)
    without_metadata = deepcopy(request)
    without_metadata['messages'][0]['content'] = native_system_policy(consumer_profile=V5)
    cap = len(canonical_json(without_metadata['messages'] if budget == 'input' else without_metadata))
    limits = replace(original_limits, **{'max_input_tokens' if budget == 'input' else 'max_transcript_bytes': cap})

    class ByteCountConsumer(BudgetConsumer):
        def compile(self, *args, **kwargs):
            artifact = super().compile(*args, **kwargs)
            # Explicit synthetic tokenizer: count every prompt byte, including
            # runner metadata. This exercises admission, not native tokenization.
            size = len(canonical_json(args[0]))
            return replace(artifact, input_ids=(1,) * size, attention_mask=(1,) * size)

    report, consumer, _ = validation_case(tmp_path, consumer_profile=V6,
        consumer_factory=ByteCountConsumer, limits=limits)
    assert not consumer.dispatched
    assert report['terminal']['status'] == ('token_limit' if budget == 'input' else 'transcript_limit')
    rejected = next(e['payload'] for e in report['events'] if e['kind'] == 'admission_rejected')
    assert rejected['observed'] > cap and rejected['maximum'] == cap
    if budget == 'input':
        assert rejected['observed'] == len(canonical_json(rejected['request']['messages']))
    else:
        assert rejected['observed'] == len(canonical_json(rejected['request']))
        assert not consumer.requests and rejected['compiled'] is None
    assert_counters(report)


@pytest.mark.parametrize("version", ["v5", "v6"])
def test_legacy_protocol_rejects_recovery_native_runtime(version):
    from daystrom_dml.contracts.agent_episode import make_event
    from daystrom_dml.services.agent_episode import _started
    from test_agent_episode_runtime import ScriptedConsumer
    task = {"id": "legacy-binding", "prompt": "Retrieve a fact."}
    limits = EpisodeLimits()
    request = build_episode_request(task, limits=limits)
    artifact = ScriptedConsumer([]).compile(request["messages"], request["tools"],
        output_reserved_tokens=request["output_reserved_tokens"])
    artifact = replace(artifact, identity=replace(artifact.identity,
        runtime_identity="dml-remote-vllm-native-tools-runtime-" + version + ":" + "a" * 64))
    scope = {"tenant_id": "t", "client_id": "c", "session_id": "s", "instance_id": "i"}
    started = _started("legacy-binding", task, scope, limits, "test_injected")
    requested = make_event(episode_id="legacy-binding", task_id=task["id"], sequence=1,
        kind="model_requested", call_id="model-0", payload={"step": 0, "request": request,
            "compiled": artifact.signing_payload(), "artifact_digest": artifact.artifact_digest})
    with pytest.raises(AgentEpisodeError, match="Compiled runtime and execution protocol differ"):
        validate_episode_events([started, requested], require_terminal=False)
