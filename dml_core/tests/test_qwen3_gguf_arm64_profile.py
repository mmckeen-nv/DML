"""Explicit ARM64 qualification routing; synthetic actions never prove quality."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from daystrom_dml.contracts.agent_episode import (
    EXECUTION_PROTOCOL_V2, QWEN3_GGUF_CONSUMER_PROFILE as OLD,
    QWEN3_GGUF_ARM64_CONSUMER_PROFILE as ARM, _compiled,
    execution_protocol_for_profile, initial_messages, recovery_guidance_identity,
    validate_episode_events,
)
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputIdentity
from daystrom_dml.services import agent_episode as runner
from daystrom_dml.services import qwen3_gguf_action_input as consumer
from daystrom_dml.services.agent_action_grammar import policy_identity
from daystrom_dml.services.model_input import _json_bytes
from scripts import agent_campaign_evidence as checker
from scripts import agent_episodes as producer
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_agent_campaign_evidence import _synthetic_campaign_impl
from test_qwen3_gguf_profile import SyntheticGGUFCompiler
from test_qwen_chat_template import tokenizer as tokenizer

PREFIX = 'dml-qwen3-gguf-arm64-action-runtime-v1:'


@pytest.mark.parametrize('system,machine', [('Linux', 'x86_64'), ('Darwin', 'arm64'), ('Windows', 'aarch64')])
def test_arm_profile_rejects_wrong_host_before_loading(monkeypatch, system, machine):
    monkeypatch.setattr(consumer.platform, 'system', lambda: system)
    monkeypatch.setattr(consumer.platform, 'machine', lambda: machine)
    monkeypatch.setattr(consumer.LocalQwen3GGUFInputConsumer, '__init__',
        lambda *a, **k: pytest.fail('wrong platform reached model loading'))
    with pytest.raises(ModelInputError, match='Linux aarch64'):
        consumer.LocalQwen3GGUFActionInputConsumer('unused', consumer_profile=ARM)


def test_old_identity_formula_preserved_arm_identity_explicit(monkeypatch):
    backend = {'version': '0.3.35', 'files': {'cpu.so': 'd' * 64}, 'system_info': 'synthetic CPU'}
    monkeypatch.setattr(consumer, 'backend_identity', lambda: backend)
    base = ModelInputIdentity('1' * 64, '2' * 64, '3' * 64,
        'dml-qwen3-gguf-model-input-runtime-v1:' + '4' * 64, 32768)
    original_policy = {'consumer_profile': OLD, 'base_runtime_identity': base.runtime_identity,
        **policy_identity(), 'inherited_guidance_policy': recovery_guidance_identity(),
        'sampling_policy': consumer.sampling_policy_identity(), 'backend_identity': backend}
    old = consumer.constrained_identity(base, consumer_profile=OLD)
    assert old.runtime_identity == 'dml-qwen3-gguf-action-runtime-v1:' + hashlib.sha256(_json_bytes(original_policy)).hexdigest()
    monkeypatch.setattr(consumer.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(consumer.platform, 'machine', lambda: 'aarch64')
    arm = consumer.constrained_identity(base, consumer_profile=ARM)
    assert arm.runtime_identity.startswith(PREFIX) and arm != old
    arm_policy = {**original_policy, 'consumer_profile': ARM,
        'host_architecture': {'system': 'Linux', 'machine': 'aarch64', 'device': 'cpu'}}
    assert arm.runtime_identity == PREFIX + hashlib.sha256(_json_bytes(arm_policy)).hexdigest()
    assert replace(arm, runtime_identity=old.runtime_identity) == old


def test_arm_dispatch_keeps_tools_policy_sampling_and_source_inventory(monkeypatch):
    sentinel = object()
    calls = []
    def construct(path, *, consumer_profile):
        calls.append((path, consumer_profile))
        return sentinel
    monkeypatch.setattr(consumer, 'LocalQwen3GGUFActionInputConsumer', construct)
    assert runner._open_consumer('arm-bundle', ARM) is sentinel
    assert calls == [('arm-bundle', ARM)]
    assert execution_protocol_for_profile(ARM) == EXECUTION_PROTOCOL_V2
    assert initial_messages('unchanged', consumer_profile=ARM) == initial_messages('unchanged', consumer_profile=OLD)
    assert runner.build_episode_request({'prompt': 'unchanged'}, consumer_profile=ARM) == runner.build_episode_request({'prompt': 'unchanged'}, consumer_profile=OLD)
    assert producer._source_digests(consumer_profile=ARM) == producer._source_digests(consumer_profile=OLD)
    sampling = consumer.sampling_policy_identity()
    assert sampling['seed'] == 0 and sampling['processor_order'] == ['authenticated_action_grammar', 'temperature', 'top_k', 'top_p', 'min_p']
    assert sampling['generation_overrides']['temperature'] == 0.7
    assert sampling['generation_overrides']['top_p'] == 0.8
    assert sampling['generation_overrides']['top_k'] == 20


class ARMCompiler(SyntheticGGUFCompiler):
    _consumer_profile = ARM

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._identity = replace(self._identity, runtime_identity=PREFIX + 'd' * 64)


@pytest.fixture(scope='module')
def arm_campaign(tmp_path_factory, tokenizer, grammar_runtime):
    return _synthetic_campaign_impl(tmp_path_factory.mktemp('synthetic-arm-gguf'),
        validation_consumer=ARMCompiler(tokenizer))


def test_arm_synthetic_replay_preserves_original_gates(arm_campaign):
    campaign, spec, identity, tokenizer = arm_campaign
    result = checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    assert result['predeclared_gates_passed']
    assert spec['acceptance'] == checker.GATES and len(campaign['episodes']) == 9
    assert not result['execution_authenticity_verified']


@pytest.mark.parametrize('target', ['selected_profile', 'compiled_runtime', 'replay_identity'])
def test_arm_evidence_cannot_be_relabelled_as_old_profile(arm_campaign, target):
    campaign, spec, identity, tokenizer = arm_campaign
    campaign, spec, identity = deepcopy(campaign), deepcopy(spec), deepcopy(identity)
    if target == 'selected_profile':
        campaign['consumer_profile'] = spec['consumer_profile'] = OLD
    elif target == 'compiled_runtime':
        events = campaign['episodes'][0]['events']
        p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
        p['compiled']['identity']['runtime_identity'] = 'dml-qwen3-gguf-action-runtime-v1:' + 'd' * 64
        p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
        with pytest.raises(ValueError, match='runtime'):
            validate_episode_events(events[:2], require_terminal=False)
        return
    else:
        identity['runtime_identity'] = 'dml-qwen3-gguf-action-runtime-v1:' + 'd' * 64
    with pytest.raises(ValueError):
        checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)


def test_legacy_event_protocol_rejects_arm_runtime(tmp_path):
    from test_agent_campaign_evidence import _synthetic_campaign_impl
    campaign, _, _, _ = _synthetic_campaign_impl(tmp_path / 'legacy-campaign')
    events = deepcopy(campaign['episodes'][0]['events'])
    p = next(e['payload'] for e in events if e['kind'] == 'model_requested')
    p['compiled']['identity']['runtime_identity'] = PREFIX + 'd' * 64
    p['artifact_digest'] = _compiled(p['compiled']).artifact_digest
    with pytest.raises(ValueError, match='runtime'):
        validate_episode_events(events[:2], require_terminal=False)
