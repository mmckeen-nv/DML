"""CUDA profile evidence isolation without GPU or trained-model execution."""
from copy import deepcopy
from dataclasses import replace

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_GGUF_CUDA_CONSUMER_PROFILE as GPU, QWEN3_GGUF_ARM64_CONSUMER_PROFILE as CPU,
    QWEN3_GGUF_CONSUMER_PROFILES, _compiled, initial_messages, validate_episode_events,
)
from daystrom_dml.services import agent_episode as runner
from daystrom_dml.services import qwen3_gguf_action_input as consumer
from scripts import agent_campaign_evidence as checker
from scripts import agent_episodes as producer
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_agent_campaign_evidence import _synthetic_campaign_impl
from test_qwen3_gguf_profile import SyntheticGGUFCompiler
from test_qwen_chat_template import tokenizer as tokenizer

PREFIX = 'dml-qwen3-gguf-cuda-action-runtime-v1:'


def test_cuda_dispatch_and_source_inventory_are_explicit(monkeypatch):
    calls, sentinel = [], object()
    def construct(path, *, consumer_profile):
        calls.append((path, consumer_profile))
        return sentinel
    monkeypatch.setattr(consumer, 'LocalQwen3GGUFActionInputConsumer', construct)
    assert GPU not in QWEN3_GGUF_CONSUMER_PROFILES
    assert runner._open_consumer('gpu-only-bundle', GPU) is sentinel
    assert calls == [('gpu-only-bundle', GPU)]
    gpu_sources = producer._source_digests(consumer_profile=GPU)
    cpu_sources = producer._source_digests(consumer_profile=CPU)
    assert set(gpu_sources) - set(cpu_sources) == {'daystrom_dml.services.qwen3_gguf_cuda_model_input'}
    assert all(gpu_sources[k] == v for k, v in cpu_sources.items())
    old = runner.build_episode_request({'prompt': 'unchanged task'}, consumer_profile=CPU)
    new = runner.build_episode_request({'prompt': 'unchanged task'}, consumer_profile=GPU)
    assert old['tools'] == new['tools'] and old['output_reserved_tokens'] == new['output_reserved_tokens']
    assert old['messages'][1:] == new['messages'][1:]


class CUDACompiler(SyntheticGGUFCompiler):
    _consumer_profile = GPU

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._identity = replace(self._identity, runtime_identity=PREFIX + 'd' * 64)


@pytest.fixture(scope='module')
def gpu_campaign(tmp_path_factory, tokenizer, grammar_runtime):
    return _synthetic_campaign_impl(tmp_path_factory.mktemp('synthetic-cuda-gguf'),
        validation_consumer=CUDACompiler(tokenizer))


def test_cuda_synthetic_replay_retains_nine_tasks_and_original_gates(gpu_campaign):
    campaign, spec, identity, tokenizer = gpu_campaign
    result = checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    assert result['predeclared_gates_passed']
    assert spec['acceptance'] == checker.GATES and len(campaign['episodes']) == 9
    assert not result['execution_authenticity_verified']
    for episode in campaign['episodes']:
        for event in episode['events']:
            if event['kind'] == 'model_requested':
                assert event['payload']['request']['messages'][0] == initial_messages('same', consumer_profile=GPU)[0]


@pytest.mark.parametrize('target', ['selected_profile', 'compiled_runtime', 'replay_identity', 'guidance'])
def test_cuda_evidence_cannot_relabel_runtime_or_strip_guidance(gpu_campaign, target):
    campaign, spec, identity, tokenizer = gpu_campaign
    campaign, spec, identity = deepcopy(campaign), deepcopy(spec), deepcopy(identity)
    if target == 'selected_profile':
        campaign['consumer_profile'] = spec['consumer_profile'] = CPU
    elif target == 'compiled_runtime':
        events = campaign['episodes'][0]['events']
        p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
        p['compiled']['identity']['runtime_identity'] = 'dml-qwen3-gguf-arm64-action-runtime-v1:' + 'd' * 64
        p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
        with pytest.raises(ValueError, match='runtime'):
            validate_episode_events(events[:2], require_terminal=False)
        return
    elif target == 'guidance':
        events = campaign['episodes'][0]['events']
        p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
        p['request']['messages'][0] = initial_messages('same', consumer_profile=CPU)[0]
    else:
        identity['runtime_identity'] = 'dml-qwen3-gguf-arm64-action-runtime-v1:' + 'd' * 64
    with pytest.raises(ValueError):
        checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)


def test_legacy_event_protocol_rejects_cuda_identity(tmp_path):
    campaign, _, _, _ = _synthetic_campaign_impl(tmp_path / 'legacy-campaign')
    events = deepcopy(campaign['episodes'][0]['events'])
    p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
    p['compiled']['identity']['runtime_identity'] = PREFIX + 'd' * 64
    p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
    with pytest.raises(ValueError, match='runtime'):
        validate_episode_events(events[:2], require_terminal=False)
