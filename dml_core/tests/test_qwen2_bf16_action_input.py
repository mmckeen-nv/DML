"""Explicit Qwen2 BF16 sampled actions; tiny mechanics, never model-quality evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from threading import Event, Thread

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_CONSUMER_PROFILE, QWEN2_BF16_SAMPLED_CONSUMER_PROFILE, initial_messages, parse_agent_action,
)
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputIdentity
from daystrom_dml.services.agent_action_grammar import (
    ActionLogitsProcessor, MAX_BOUND_REQUESTS, action_schema, compile_action_grammar, policy_identity,
)
from daystrom_dml.services.model_input import ModelInputExecutionError, _json_bytes
from daystrom_dml.services.qwen2_bf16_action_input import (
    LocalQwen2BF16ActionInputConsumer, _sampled_cpu_rng, constrained_identity, sampling_policy_identity,
)
from scripts import agent_campaign_evidence as checker
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_qwen2_bf16_model_input import snapshot as snapshot
from daystrom_dml.services.qwen2_bf16_model_input import LocalQwen2BF16InputConsumer
from test_qwen_action_input import _steered_logits, _tools


@pytest.fixture
def consumer(snapshot, grammar_runtime):
    with LocalQwen2BF16ActionInputConsumer(snapshot.path, consumer_profile=QWEN2_BF16_SAMPLED_CONSUMER_PROFILE) as instance:
        yield instance


def _messages(prompt='Read the requested memory.'):
    return initial_messages(prompt, consumer_profile=QWEN2_BF16_SAMPLED_CONSUMER_PROFILE)


@pytest.mark.parametrize('profile', [None, '', 'auto', QWEN3_CONSUMER_PROFILE, 'qwen2-instruct-bf16-v1'])
def test_action_profile_rejects_implicit_or_other_profile_selection(snapshot, profile):
    with pytest.raises(ModelInputError, match='profile'):
        LocalQwen2BF16ActionInputConsumer(snapshot.path, consumer_profile=profile)


@pytest.mark.parametrize('change', ['omit', 'alter', 'double', 'move'])
def test_compile_requires_exact_visible_policy_without_insertion_or_repair(consumer, change):
    request = _messages()
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


def test_compiled_request_capacity_and_close_release_are_explicit(consumer):
    for _ in range(MAX_BOUND_REQUESTS):
        consumer.compile(_messages(), [], output_reserved_tokens=1)
    with pytest.raises(ModelInputError, match='capacity'):
        consumer.compile(_messages(), [], output_reserved_tokens=1)
    private = consumer._snapshot.path
    consumer.close()
    assert not consumer._bound_requests and not private.exists()


@pytest.mark.parametrize('boundary', ['policy', 'template', 'grammar', 'cache'])
def test_runtime_boundary_drift_refuses_before_generation(consumer, monkeypatch, boundary):
    artifact = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    calls = []
    monkeypatch.setattr(consumer._model, 'generate', lambda **kw: calls.append(kw))
    if boundary == 'policy':
        consumer._recovery_guidance = {}
    elif boundary == 'template':
        consumer._tokenizer.chat_template += ' '
    elif boundary == 'grammar':
        consumer._grammar_policy = {}
    else:
        consumer._model._cache = object()
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert calls == []


def test_sampled_identity_is_explicit_and_old_literal_identity_is_unchanged():
    from daystrom_dml.contracts.agent_episode import recovery_guidance_identity
    from daystrom_dml.services.qwen3_action_input import constrained_identity as prior_identity

    prior = ModelInputIdentity('1' * 64, '2' * 64, '3' * 64,
                               'dml-qwen3-model-input-runtime-v1:' + 'a' * 64, 32768)
    assert prior_identity(prior, consumer_profile=QWEN3_CONSUMER_PROFILE).runtime_identity == (
        'dml-qwen3-action-runtime-v1:da3f43217de5b657a70a390b33663bed1d37d1e65aae27cdd8f2385887bbe420')
    base = replace(prior, runtime_identity='dml-qwen2-bf16-model-input-runtime-v1:' + 'a' * 64)
    sampled = constrained_identity(base, consumer_profile=QWEN2_BF16_SAMPLED_CONSUMER_PROFILE)
    payload = {'consumer_profile': QWEN2_BF16_SAMPLED_CONSUMER_PROFILE, 'base_runtime_identity': base.runtime_identity,
               **policy_identity(), 'inherited_guidance_policy': recovery_guidance_identity(),
               'sampling_policy': sampling_policy_identity()}
    assert sampled.runtime_identity == 'dml-qwen2-bf16-action-runtime-v1:' + hashlib.sha256(_json_bytes(payload)).hexdigest()
    assert _messages()[0] == initial_messages('irrelevant', consumer_profile=QWEN3_CONSUMER_PROFILE)[0]
    with pytest.raises(ModelInputError, match='base runtime'):
        constrained_identity(prior, consumer_profile=QWEN2_BF16_SAMPLED_CONSUMER_PROFILE)


def test_exact_effective_config_has_only_declared_sampling_changes(consumer, snapshot):
    with LocalQwen2BF16InputConsumer(snapshot.path) as base:
        expected = base._generation_config(31).to_dict()
    expected.update(sampling_policy_identity()['generation_overrides'])
    assert consumer._generation_config(31).to_dict() == expected
    assert consumer._model.generation_config.to_dict() == consumer._generation_config(1).to_dict()
    config = consumer._generation_config(31)
    assert config.num_beams == config.num_beam_groups == config.num_return_sequences == 1
    assert config.typical_p == config.repetition_penalty == 1.0
    assert config.epsilon_cutoff == config.eta_cutoff == 0.0
    assert config.renormalize_logits is False and config.forced_eos_token_id is None
    assert config.eos_token_id == snapshot.tokenizer.eos_token_id
    assert type(config.eos_token_id) is int
    assert config.use_cache is True and config.cache_implementation == 'dynamic'
    assert sampling_policy_identity()['seed'] == 0


def test_actual_sampled_generation_matches_independent_tiny_model_and_processor_order(consumer, snapshot):
    import torch
    from transformers import GenerationConfig

    tools = _tools('retrieve')
    artifact = consumer.compile(_messages(), tools, output_reserved_tokens=64)
    before = torch.random.get_rng_state().clone()
    actual = consumer.execute(artifact)
    assert torch.equal(torch.random.get_rng_state(), before)
    grammar = compile_action_grammar(snapshot.tokenizer, snapshot.model.config.vocab_size, tools)
    processor = ActionLogitsProcessor(grammar, snapshot.tokenizer, artifact.input_ids, 64)
    generation = GenerationConfig(
        max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
        num_beams=1, num_return_sequences=1, use_cache=True, cache_implementation='dynamic', disable_compile=True,
        bos_token_id=snapshot.tokenizer.bos_token_id, eos_token_id=snapshot.tokenizer.eos_token_id,
        pad_token_id=snapshot.tokenizer.pad_token_id,
    )
    processors = snapshot.model._get_logits_processor(
        generation_config=generation, input_ids_seq_length=len(artifact.input_ids),
        logits_processor=[processor], device='cpu')
    assert [type(item).__name__ for item in processors] == [
        'ActionLogitsProcessor', 'TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper', 'MinPLogitsWarper']
    with torch.random.fork_rng(devices=[]), torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
        torch.random.default_generator.manual_seed(0)
        expected = snapshot.model.generate(
            input_ids=torch.tensor([artifact.input_ids], dtype=torch.long),
            attention_mask=torch.tensor([artifact.attention_mask], dtype=torch.long),
            generation_config=generation, use_model_defaults=False, logits_processor=[processor])
    complete = tuple(expected[0].tolist())
    processor.finish(complete)
    assert complete[:len(artifact.input_ids)] == artifact.input_ids
    assert complete[len(artifact.input_ids):] == actual.output_ids
    assert actual.output_token_count == len(actual.output_ids)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_actual_a_b_a_sampling_ignores_nonce_ambient_rng_and_consumer_order(consumer, snapshot):
    import torch

    a = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    b = consumer.compile(_messages('Read the other memory.'), _tools('retrieve'), output_reserved_tokens=24)
    repeated_a = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    assert repeated_a.nonce != a.nonce and repeated_a.artifact_digest != a.artifact_digest
    assert repeated_a.input_ids == a.input_ids
    outputs = []
    for artifact in (a, b, repeated_a, a):
        torch.rand(7)
        before = torch.random.get_rng_state().clone()
        outputs.append(consumer.execute(artifact))
        assert torch.equal(torch.random.get_rng_state(), before)
    assert outputs[0].output_ids == outputs[2].output_ids == outputs[3].output_ids
    with LocalQwen2BF16ActionInputConsumer(snapshot.path, consumer_profile=QWEN2_BF16_SAMPLED_CONSUMER_PROFILE) as other:
        foreign_a = other.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
        assert other.execute(foreign_a).output_ids == outputs[0].output_ids


def test_generation_failure_restores_cpu_rng_without_retry_or_accelerator_seed(consumer, monkeypatch):
    import torch

    artifact = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    before = torch.random.get_rng_state().clone()
    calls = []

    def failing(**kwargs):
        calls.append(kwargs)
        torch.rand(19)
        raise RuntimeError('private backend details')

    def forbidden(*args, **kwargs):
        pytest.fail('Sampling touched a global or accelerator seed API')

    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    monkeypatch.setattr(consumer._model, 'generate', failing)
    with pytest.raises(ModelInputExecutionError) as error:
        consumer.execute(artifact)
    assert 'private' not in str(error.value)
    assert len(calls) == 1 and len(consumer._bound_requests) == 1
    assert torch.equal(torch.random.get_rng_state(), before)
    # A second context can acquire the module lock after the exception.
    with _sampled_cpu_rng(torch, 0):
        torch.rand(3)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_sampled_rng_lock_serializes_profile_contexts_and_restores_caller_state():
    import torch

    first_entered, second_attempting, release_first, second_entered = (Event() for _ in range(4))
    values, failures = [], []
    before = torch.random.get_rng_state().clone()

    def run(first):
        try:
            if not first:
                second_attempting.set()
            with _sampled_cpu_rng(torch, 0):
                values.append(torch.rand(4))
                if first:
                    first_entered.set()
                    assert release_first.wait(5)
                else:
                    second_entered.set()
        except BaseException as error:
            failures.append(error)

    first, second = Thread(target=run, args=(True,)), Thread(target=run, args=(False,))
    first.start()
    try:
        assert first_entered.wait(5)
        second.start()
        assert second_attempting.wait(5)
        assert not second_entered.wait(0.05)
    finally:
        release_first.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive() and not failures
    assert second_entered.is_set() and len(values) == 2
    assert torch.equal(values[0], values[1])
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize('field,value', [('seed', 1), ('selection', 'retry')])
def test_sampling_policy_drift_refuses_before_generation(consumer, monkeypatch, field, value):
    artifact = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    consumer._sampling_policy[field] = value
    calls = []
    monkeypatch.setattr(consumer._model, 'generate', lambda **kw: calls.append(kw))
    with pytest.raises(ModelInputError, match='sampling policy'):
        consumer.execute(artifact)
    assert calls == []


@pytest.mark.parametrize('direction', ['old_to_sampled', 'sampled_to_old'])
def test_profile_crossing_refuses_authenticated_artifacts(consumer, snapshot, monkeypatch, direction):
    with LocalQwen2BF16InputConsumer(snapshot.path) as old:
        source, target = (old, consumer) if direction == 'old_to_sampled' else (consumer, old)
        artifact = source.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
        calls = []
        monkeypatch.setattr(target._model, 'generate', lambda **kw: calls.append(kw))
        with pytest.raises(ModelInputError, match='authenticated'):
            target.execute(artifact)
        assert calls == []


@pytest.mark.parametrize('mutation', ['identity', 'request', 'generation_config'])
def test_sampled_artifact_and_config_forgery_refuses_before_model(consumer, monkeypatch, mutation):
    artifact = consumer.compile(_messages(), _tools('retrieve'), output_reserved_tokens=24)
    if mutation == 'identity':
        artifact = replace(artifact, identity=consumer._base_action_identity)
    elif mutation == 'request':
        request = json.loads(consumer._bound_requests[artifact.artifact_digest])
        request['messages'][1]['content'] += ' changed'
        consumer._bound_requests[artifact.artifact_digest] = json.dumps(request).encode()
    else:
        consumer._model.generation_config.temperature = 0.9
    calls = []
    monkeypatch.setattr(consumer._model, 'generate', lambda **kw: calls.append(kw))
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert calls == []


_ACTIONS = [
    ('retrieve', {'query': 'arbitrary', 'top_k': 10}),
    ('ingest', {'text': 'arbitrary'}),
    ('update', {'record_ref': 'r77', 'text': 'arbitrary', 'reason': 'requested'}),
    ('promote', {'record_refs': ['r77'], 'text': 'arbitrary', 'reason': 'requested'}),
    ('supersede', {'record_ref': 'r77', 'replacement_ref': 'r88', 'reason': 'requested'}),
    ('retire', {'record_ref': 'r77', 'reason': 'requested'}),
    (None, {'claims': []}),
]


@pytest.mark.parametrize('name,arguments', _ACTIONS)
def test_sampled_profile_keeps_all_actions_expressible_and_exactly_replayable(consumer, monkeypatch, name, arguments):
    tools = [] if name is None else _tools(name)
    action = {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': arguments} if name is None else {
        'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': name, 'arguments': arguments}
    artifact = consumer.compile(_messages(), tools, output_reserved_tokens=192)
    _steered_logits(consumer, monkeypatch, {artifact.input_ids: action})
    result = consumer.execute(artifact)
    assert parse_agent_action(result.text) == action
    assert len(action_schema(tools)['anyOf']) == len(tools) + 1
    events = [{'kind': 'model_requested', 'payload': {'compiled': artifact.signing_payload(),
               'request': json.loads(consumer._bound_requests[artifact.artifact_digest])}},
              {'kind': 'model_completed', 'payload': {'output_ids': list(result.output_ids), 'text': result.text}}]
    assert checker._replay_model(events, consumer._identity.to_payload(), consumer._tokenizer,
                                 QWEN2_BF16_SAMPLED_CONSUMER_PROFILE) == 1
    changed = deepcopy(events)
    changed[1]['payload']['text'] += ' '
    with pytest.raises(ValueError, match='output'):
        checker._replay_model(changed, consumer._identity.to_payload(), consumer._tokenizer,
                              QWEN2_BF16_SAMPLED_CONSUMER_PROFILE)
