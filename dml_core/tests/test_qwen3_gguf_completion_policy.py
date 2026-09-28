"""Generic completion salience is new-profile-only, never new action authority."""
import hashlib
from copy import deepcopy

import pytest

from daystrom_dml.contracts.agent_episode import (
    AGENT_POLICY, RECOVERY_GUIDANCE, QWEN3_GGUF_COMPLETION_GUIDANCE,
    QWEN3_GGUF_CONSUMER_PROFILES, QWEN3_GGUF_ALL_CONSUMER_PROFILES,
    QWEN3_GGUF_CUDA_CONSUMER_PROFILE, initial_messages,
    qwen3_gguf_completion_policy_identity, recovery_guidance_identity,
    parse_agent_action, AgentEpisodeError,
)


def test_historical_cpu_system_messages_and_policy_unchanged():
    historical = AGENT_POLICY + '\n\n' + RECOVERY_GUIDANCE
    for profile in QWEN3_GGUF_CONSUMER_PROFILES:
        assert initial_messages('unmodified task', consumer_profile=profile) == [
            {'role': 'system', 'content': historical}, {'role': 'user', 'content': 'unmodified task'}]
    assert recovery_guidance_identity()['system_message_sha256'] == hashlib.sha256(historical.encode()).hexdigest()
    assert QWEN3_GGUF_CUDA_CONSUMER_PROFILE not in QWEN3_GGUF_CONSUMER_PROFILES
    assert QWEN3_GGUF_ALL_CONSUMER_PROFILES == (*QWEN3_GGUF_CONSUMER_PROFILES, QWEN3_GGUF_CUDA_CONSUMER_PROFILE)


def test_gpu_guidance_identity_binds_exact_appended_message_and_task():
    prompt = 'Read an unrelated project fact without changing stored records.'
    messages = initial_messages(prompt, consumer_profile=QWEN3_GGUF_CUDA_CONSUMER_PROFILE)
    expected = AGENT_POLICY + '\n\n' + RECOVERY_GUIDANCE + '\n\n' + QWEN3_GGUF_COMPLETION_GUIDANCE
    assert messages == [{'role': 'system', 'content': expected}, {'role': 'user', 'content': prompt}]
    policy = qwen3_gguf_completion_policy_identity()
    assert policy['system_message_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
    assert policy['guidance_sha256'] == hashlib.sha256(QWEN3_GGUF_COMPLETION_GUIDANCE.encode()).hexdigest()
    assert policy['base_recovery_policy'] == recovery_guidance_identity()
    assert policy['guidance'] == QWEN3_GGUF_COMPLETION_GUIDANCE
    changed = deepcopy(policy)
    changed['guidance'] += 'changed'
    assert changed != qwen3_gguf_completion_policy_identity()


def test_guidance_does_not_extend_original_final_contract():
    for raw in ('I changed it.', '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"completed":true}}'):
        with pytest.raises(AgentEpisodeError):
            parse_agent_action(raw)
    assert 'If the task requests readout only, do not introduce a mutation.' in QWEN3_GGUF_COMPLETION_GUIDANCE
    assert 'within the existing limits' in QWEN3_GGUF_COMPLETION_GUIDANCE
