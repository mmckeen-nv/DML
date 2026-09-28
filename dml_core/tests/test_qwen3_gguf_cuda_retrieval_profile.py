"""New retrieval policy keeps GPU runtime, corpus and acceptance gates explicit."""
from copy import deepcopy
from dataclasses import replace

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_GGUF_CUDA_CONSUMER_PROFILE as OLD,
    QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE as NEW,
    QWEN3_GGUF_CONSUMER_PROFILES, QWEN3_GGUF_CUDA_CONSUMER_PROFILES,
    _compiled, initial_messages, validate_episode_events,
)
from daystrom_dml.services import agent_episode as runner
from daystrom_dml.services import qwen3_gguf_action_input as consumer
from scripts import agent_campaign_evidence as checker
from scripts import agent_episodes as producer
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_agent_campaign_evidence import _synthetic_campaign_impl
from test_qwen3_gguf_profile import SyntheticGGUFCompiler
from test_qwen_chat_template import tokenizer as tokenizer

PREFIX = 'dml-qwen3-gguf-cuda-action-runtime-v2:'


def test_retrieval_profile_keeps_explicit_gpu_dispatch_and_inventory(monkeypatch):
    calls, sentinel = [], object()
    def construct(path, *, consumer_profile):
        calls.append((path, consumer_profile))
        return sentinel
    monkeypatch.setattr(consumer, 'LocalQwen3GGUFActionInputConsumer', construct)
    assert NEW not in QWEN3_GGUF_CONSUMER_PROFILES
    assert QWEN3_GGUF_CUDA_CONSUMER_PROFILES == (OLD, NEW)
    assert runner._open_consumer('same-gpu-runtime', NEW) is sentinel
    assert calls == [('same-gpu-runtime', NEW)]
    assert producer._source_digests(consumer_profile=NEW) == producer._source_digests(consumer_profile=OLD)
    old = runner.build_episode_request({'prompt': 'same task'}, consumer_profile=OLD)
    new = runner.build_episode_request({'prompt': 'same task'}, consumer_profile=NEW)
    assert old['tools'] == new['tools'] and old['output_reserved_tokens'] == new['output_reserved_tokens']
    assert old['messages'][1:] == new['messages'][1:]
    assert new['messages'][0]['content'].startswith(old['messages'][0]['content'] + '\n\n')


class RetrievalCompiler(SyntheticGGUFCompiler):
    _consumer_profile = NEW

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._identity = replace(self._identity, runtime_identity=PREFIX + 'd' * 64)


@pytest.fixture(scope='module')
def retrieval_campaign(tmp_path_factory, tokenizer, grammar_runtime):
    return _synthetic_campaign_impl(tmp_path_factory.mktemp('synthetic-cuda-retrieval'),
        validation_consumer=RetrievalCompiler(tokenizer))


def test_retrieval_profile_replay_preserves_nine_tasks_and_original_gates(retrieval_campaign):
    campaign, spec, identity, tokenizer = retrieval_campaign
    result = checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    assert result['predeclared_gates_passed']
    assert spec['acceptance'] == checker.GATES and len(campaign['episodes']) == 9
    assert not result['execution_authenticity_verified']
    for episode in campaign['episodes']:
        for event in episode['events']:
            if event['kind'] == 'model_requested':
                assert event['payload']['request']['messages'][0] == initial_messages('same', consumer_profile=NEW)[0]


@pytest.mark.parametrize('target', ['profile', 'compiled_runtime', 'replay_identity', 'guidance'])
def test_retrieval_candidate_cannot_reinterpret_previous_cuda_profile(retrieval_campaign, target):
    campaign, spec, identity, tokenizer = retrieval_campaign
    campaign, spec, identity = deepcopy(campaign), deepcopy(spec), deepcopy(identity)
    if target == 'profile':
        campaign['consumer_profile'] = spec['consumer_profile'] = OLD
    elif target == 'compiled_runtime':
        events = campaign['episodes'][0]['events']
        p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
        p['compiled']['identity']['runtime_identity'] = 'dml-qwen3-gguf-cuda-action-runtime-v1:' + 'd' * 64
        p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
        with pytest.raises(ValueError, match='runtime'):
            validate_episode_events(events[:2], require_terminal=False)
        return
    elif target == 'guidance':
        p = next(e['payload'] for e in campaign['episodes'][0]['events'] if e['kind'] == 'model_requested')
        p['request']['messages'][0] = initial_messages('same', consumer_profile=OLD)[0]
    else:
        identity['runtime_identity'] = 'dml-qwen3-gguf-cuda-action-runtime-v1:' + 'd' * 64
    with pytest.raises(ValueError):
        checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)


def test_legacy_event_protocol_rejects_retrieval_runtime(tmp_path):
    campaign, _, _, _ = _synthetic_campaign_impl(tmp_path / 'legacy-campaign')
    events = deepcopy(campaign['episodes'][0]['events'])
    p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
    p['compiled']['identity']['runtime_identity'] = PREFIX + 'd' * 64
    p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
    with pytest.raises(ValueError, match='runtime'):
        validate_episode_events(events[:2], require_terminal=False)
