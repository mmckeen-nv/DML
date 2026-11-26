"""Separately versioned query guidance without changing transport or authority."""
from dataclasses import replace
import hashlib

import pytest

from daystrom_dml.contracts.agent_episode import (
    AGENT_POLICY, RECOVERY_GUIDANCE, QWEN3_GGUF_COMPLETION_GUIDANCE, QWEN3_GGUF_RETRIEVAL_GUIDANCE,
    QWEN3_GGUF_CUDA_CONSUMER_PROFILE as V1,
    QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE as V2,
    QWEN3_GGUF_CUDA_CONSUMER_PROFILES, QWEN3_GGUF_CONSUMER_PROFILES,
    initial_messages, qwen3_gguf_completion_policy_identity, qwen3_gguf_retrieval_policy_identity,
)
from daystrom_dml.contracts.model_input import ModelInputIdentity, ModelInputError
from daystrom_dml.services import qwen3_gguf_action_input as action
from daystrom_dml.services import qwen3_gguf_cuda_model_input as cuda
from daystrom_dml.services.model_input import _json_bytes


def test_v2_appends_exact_guidance_preserves_v1_and_original_task():
    old = AGENT_POLICY + '\n\n' + RECOVERY_GUIDANCE + '\n\n' + QWEN3_GGUF_COMPLETION_GUIDANCE
    assert initial_messages('same task', consumer_profile=V1)[0]['content'] == old
    new = initial_messages('same task', consumer_profile=V2)
    assert new[0]['content'] == old + '\n\n' + QWEN3_GGUF_RETRIEVAL_GUIDANCE
    assert new[1:] == initial_messages('same task', consumer_profile=V1)[1:]
    assert qwen3_gguf_completion_policy_identity()['system_message_sha256'] == hashlib.sha256(old.encode()).hexdigest()
    policy = qwen3_gguf_retrieval_policy_identity()
    assert policy['base_completion_policy'] == qwen3_gguf_completion_policy_identity()
    assert policy['guidance'] == QWEN3_GGUF_RETRIEVAL_GUIDANCE
    assert policy['system_message_sha256'] == hashlib.sha256(new[0]['content'].encode()).hexdigest()
    assert V2 not in QWEN3_GGUF_CONSUMER_PROFILES and QWEN3_GGUF_CUDA_CONSUMER_PROFILES == (V1, V2)


def test_v1_runtime_composition_remains_exact_and_v2_binds_new_policy(monkeypatch):
    backend = {'test_only_cuda_backend': 'unchanged'}
    monkeypatch.setattr(cuda, 'cuda_backend_identity', lambda: backend)
    monkeypatch.setattr(action, 'policy_identity', lambda: {'test_grammar': 'unchanged'})
    base = ModelInputIdentity(model_digest='1'*64, tokenizer_digest='2'*64, chat_template_digest='3'*64,
        runtime_identity='dml-qwen3-gguf-model-input-runtime-v1:'+'4'*64, model_window_tokens=32768)
    historical = {'consumer_profile': V1,
        'base_runtime_identity': 'dml-qwen3-gguf-cuda-model-input-runtime-v1:' + hashlib.sha256(_json_bytes({
            'snapshot_provenance_identity': base.to_payload(), 'active_cuda_backend': backend})).hexdigest(),
        'test_grammar': 'unchanged', 'inherited_guidance_policy': action.recovery_guidance_identity(),
        'sampling_policy': action.sampling_policy_identity(),
        'completion_policy': qwen3_gguf_completion_policy_identity(), 'backend_identity': backend}
    expected = replace(base, runtime_identity='dml-qwen3-gguf-cuda-action-runtime-v1:' + hashlib.sha256(_json_bytes(historical)).hexdigest())
    assert action.constrained_identity(base, consumer_profile=V1) == expected
    v2 = action.constrained_identity(base, consumer_profile=V2)
    assert v2.runtime_identity.startswith('dml-qwen3-gguf-cuda-action-runtime-v2:') and v2 != expected
    original = action.qwen3_gguf_retrieval_policy_identity
    monkeypatch.setattr(action, 'qwen3_gguf_retrieval_policy_identity', lambda: {**original(), 'guidance': 'tampered'})
    assert action.constrained_identity(base, consumer_profile=V2) != v2
    assert action.constrained_identity(base, consumer_profile=V1) == expected


@pytest.mark.parametrize('selected, supplied', [(V2, V1), (V1, V2)])
def test_compile_rejects_cross_version_system_policy_before_dispatch(selected, supplied):
    consumer = object.__new__(action.LocalQwen3GGUFActionInputConsumer)
    consumer._consumer_profile = selected
    with pytest.raises(ModelInputError, match='exact first system message'):
        consumer.compile(initial_messages('same task', consumer_profile=supplied), [], output_reserved_tokens=256)


@pytest.mark.parametrize('profile', [V1, V2])
def test_gpu_versions_choose_identical_backend_mechanics(monkeypatch, profile):
    calls = []
    marker = object()
    monkeypatch.setattr(cuda, 'cuda_backend_identity', lambda: {'test': 'same backend'})
    monkeypatch.setattr(cuda, '_CUDAGGUFBackend', lambda path, count: calls.append((path, count)) or marker)
    consumer = object.__new__(action.LocalQwen3GGUFActionInputConsumer)
    consumer._consumer_profile = profile
    assert consumer._active_backend_identity() == {'test': 'same backend'}
    assert consumer._create_backend('unused-test-path', 151936) is marker
    assert calls == [('unused-test-path', 151936)]
