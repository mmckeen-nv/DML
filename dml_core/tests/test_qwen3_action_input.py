"""New explicit action profile: tiny constrained runtime and legacy boundaries."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE, VALIDATION_CONSUMER_PROFILE,
    initial_messages, parse_agent_action,
)
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputRequest
from daystrom_dml.services.qwen3_action_input import LocalQwen3ActionInputConsumer, constrained_identity
from daystrom_dml.services.qwen_action_input import constrained_identity as old_identity
from daystrom_dml.services.agent_action_grammar import MAX_BOUND_REQUESTS
from scripts import agent_campaign_evidence as checker
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_qwen3_model_input import snapshot as snapshot
from test_qwen_action_input import _steered_logits, _tools, _action


@pytest.fixture
def consumer(snapshot, grammar_runtime):
    with LocalQwen3ActionInputConsumer(snapshot.path, consumer_profile=QWEN3_CONSUMER_PROFILE) as instance:
        yield instance


def messages():
    return initial_messages('Read the requested memory.', consumer_profile=QWEN3_CONSUMER_PROFILE)


def test_qwen3_explicit_identity_and_exact_old_policy_bytes(consumer):
    assert consumer._identity == constrained_identity(consumer._base_action_identity, consumer_profile=QWEN3_CONSUMER_PROFILE)
    assert consumer._identity.runtime_identity.startswith('dml-qwen3-action-runtime-v1:')
    assert messages()[0] == initial_messages('irrelevant', consumer_profile=RECOVERY_CONSUMER_PROFILE)[0]
    with pytest.raises(ModelInputError):
        old_identity(consumer._base_action_identity, consumer_profile=QWEN3_CONSUMER_PROFILE)
    with pytest.raises(TypeError):
        constrained_identity(consumer._base_action_identity)
    with pytest.raises(TypeError):
        LocalQwen3ActionInputConsumer(consumer._snapshot.path)


@pytest.mark.parametrize('profile', [None, '', 'auto', RECOVERY_CONSUMER_PROFILE, VALIDATION_CONSUMER_PROFILE])
def test_qwen3_profile_cannot_fall_back_or_accept_legacy_selection(snapshot, profile):
    with pytest.raises(ModelInputError, match='profile'):
        LocalQwen3ActionInputConsumer(snapshot.path, consumer_profile=profile)


def test_qwen3_identity_refuses_other_architecture_even_with_selected_profile(consumer):
    base = replace(consumer._base_action_identity, runtime_identity='dml-qwen-model-input-runtime-v1:' + 'a' * 64)
    with pytest.raises(ModelInputError, match='base runtime'):
        constrained_identity(base, consumer_profile=QWEN3_CONSUMER_PROFILE)


@pytest.mark.parametrize('change', ['omit', 'alter', 'double', 'move'])
def test_qwen3_compile_never_inserts_or_repairs_policy(consumer, change):
    request = messages()
    if change == 'omit':
        request.pop(0)
    elif change == 'alter':
        request[0]['content'] += ' '
    elif change == 'double':
        request[0]['content'] *= 2
    else:
        request.reverse()
    with pytest.raises(ModelInputError, match='first system message'):
        consumer.compile(request, _tools('retrieve'), output_reserved_tokens=16)
    assert not consumer._bound_requests


def test_qwen3_a_b_a_dispatch_uses_request_owned_grammar_and_ids(consumer, monkeypatch):
    a = consumer.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    b = consumer.compile(messages(), _tools('retire'), output_reserved_tokens=128)
    expected = ModelInputRequest.from_payload({'messages': messages(), 'tools': _tools('retrieve'), 'output_reserved_tokens': 128})
    assert a.request_digest == expected.request_digest
    assert json.loads(consumer._bound_requests[a.artifact_digest]) == expected.to_payload()
    calls = _steered_logits(consumer, monkeypatch, {a.input_ids: _action('retrieve'), b.input_ids: _action('retire')})
    results = [consumer.execute(x) for x in (a, b, a)]
    assert [parse_agent_action(x.text)['name'] for x in results] == ['retrieve', 'retire', 'retrieve']
    assert calls == [a.input_ids, b.input_ids, a.input_ids]
    assert all(x.output_token_count == len(x.output_ids) for x in results)


@pytest.mark.parametrize('mutation', ['request', 'foreign', 'artifact', 'guidance', 'template', 'grammar'])
def test_qwen3_forgery_or_runtime_drift_refuses_before_model(consumer, snapshot, monkeypatch, mutation):
    artifact = consumer.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    calls = []
    monkeypatch.setattr(consumer._model, 'generate', lambda **kw: calls.append(kw))
    if mutation == 'request':
        request = json.loads(consumer._bound_requests[artifact.artifact_digest])
        request['messages'][0]['content'] += ' '
        consumer._bound_requests[artifact.artifact_digest] = json.dumps(request).encode()
    elif mutation == 'foreign':
        with LocalQwen3ActionInputConsumer(snapshot.path, consumer_profile=QWEN3_CONSUMER_PROFILE) as other:
            artifact = other.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    elif mutation == 'artifact':
        artifact = replace(artifact, request_digest='f' * 64)
    elif mutation == 'guidance':
        from daystrom_dml.contracts import agent_episode as contract
        monkeypatch.setattr(contract, 'RECOVERY_GUIDANCE', contract.RECOVERY_GUIDANCE + ' ')
    elif mutation == 'template':
        consumer._tokenizer.chat_template += ' '
    else:
        from daystrom_dml.services import agent_action_grammar as grammar
        monkeypatch.setattr(grammar, 'GRAMMAR_POLICY_VERSION', 'altered', raising=False)
        monkeypatch.setattr(consumer, '_grammar_policy', {})
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert calls == []


def test_qwen3_grammar_replay_checks_exact_input_output_ids(consumer, monkeypatch):
    artifact = consumer.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    _steered_logits(consumer, monkeypatch, {artifact.input_ids: _action('retrieve')})
    result = consumer.execute(artifact)
    events = [{'kind': 'model_requested', 'payload': {'compiled': artifact.signing_payload(),
               'request': json.loads(consumer._bound_requests[artifact.artifact_digest])}},
              {'kind': 'model_completed', 'payload': {'output_ids': list(result.output_ids), 'text': result.text}}]
    assert checker._replay_model(events, consumer._identity.to_payload(), consumer._tokenizer, QWEN3_CONSUMER_PROFILE) == 1
    changed = deepcopy(events)
    changed[0]['payload']['compiled']['input_ids'][-1] = consumer._tokenizer.eos_token_id
    with pytest.raises(ValueError, match='input tokens'):
        checker._replay_model(changed, consumer._identity.to_payload(), consumer._tokenizer, QWEN3_CONSUMER_PROFILE)
    changed = deepcopy(events)
    changed[1]['payload']['output_ids'] = consumer._tokenizer.encode('<think>', add_special_tokens=False)
    changed[1]['payload']['text'] = '<think>'
    with pytest.raises(ValueError, match='grammar'):
        checker._replay_model(changed, consumer._identity.to_payload(), consumer._tokenizer, QWEN3_CONSUMER_PROFILE)


def test_qwen3_capacity_is_bounded_and_close_drops_bindings(consumer):
    for _ in range(MAX_BOUND_REQUESTS):
        consumer.compile(messages(), [], output_reserved_tokens=1)
    with pytest.raises(ModelInputError, match='capacity'):
        consumer.compile(messages(), [], output_reserved_tokens=1)
    assert len(consumer._bound_requests) == MAX_BOUND_REQUESTS
    private = consumer._snapshot.path
    consumer.close()
    assert not consumer._bound_requests and not private.exists()


def test_qwen3_addition_preserves_literal_attempt11_identity_and_legacy_profile_set():
    from daystrom_dml.contracts import agent_episode as contract
    from daystrom_dml.contracts.model_input import ModelInputIdentity, SUPPORTED_CHAT_TEMPLATE_DIGEST
    base = ModelInputIdentity('1' * 64, '2' * 64, SUPPORTED_CHAT_TEMPLATE_DIGEST, 'fixed-baseline-runtime', 32768)
    # Retained before attempt12 edits from source e08c38e3, legacy-before.json.
    assert old_identity(base, consumer_profile=RECOVERY_CONSUMER_PROFILE).runtime_identity == 'dml-qwen-action-runtime-v3:8461911fa599cd107656b8d517f3e71fab221cc290efbce2ddbac201d988ace9'
    assert contract.VALIDATION_PROFILES == (VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE)
    assert contract.execution_protocol_for_profile(QWEN3_CONSUMER_PROFILE) == contract.EXECUTION_PROTOCOL_V2
