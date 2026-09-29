"""V3 enters the same strict token/failure replay, without changing V2."""
import pytest
from scripts import agent_campaign_evidence as checker
from scripts.agent_episodes import _source_digests
from daystrom_dml.contracts.agent_episode import (
    LLAMA3_SFT_CONSUMER_PROFILE, LLAMA3_SFT_V3_CONSUMER_PROFILE,
)
from test_llama3_sft_campaign_evidence import Consumer, events


@pytest.mark.parametrize('profile', [LLAMA3_SFT_CONSUMER_PROFILE, LLAMA3_SFT_V3_CONSUMER_PROFILE])
def test_registered_profiles_replay_incomplete_prefix_without_repair(profile):
    consumer = Consumer()
    assert checker._replay_model(events(), {}, consumer, profile) == 1
    assert consumer.released


def test_v3_binds_new_adapter_in_addition_to_shared_runtime():
    old = _source_digests(consumer_profile=LLAMA3_SFT_CONSUMER_PROFILE)
    new = _source_digests(consumer_profile=LLAMA3_SFT_V3_CONSUMER_PROFILE)
    assert set(new) - set(old) == {'daystrom_dml.services.llama3_sft_v3_action_input'}
    assert all(new[key] == value for key, value in old.items())
    assert checker.GATES['all_task_successes_required'] is False


def test_v3_rejects_token_mask_tampering():
    trace = events()
    trace[1]['payload']['output_ids'] = [9]
    with pytest.raises(ValueError, match='grammar mask'):
        checker._replay_model(trace, {}, Consumer(), LLAMA3_SFT_V3_CONSUMER_PROFILE)
