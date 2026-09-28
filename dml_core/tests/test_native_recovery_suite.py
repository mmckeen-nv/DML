"""The live recovery declaration is fixed before outcomes and cannot be reused."""
from copy import deepcopy

import pytest

from daystrom_dml.services.agent_episode import EpisodeLimits
from scripts import native_recovery_suite as suite


def test_declaration_has_two_real_peer_cases_without_answer_hints(tmp_path):
    path = tmp_path / 'suite.json'
    declared = suite.prepare_suite(path, identity={'synthetic': True}, limits=EpisodeLimits(**suite.QUALIFICATION_LIMITS))
    assert tuple(case['id'] for case in declared['cases']) == suite.CASE_IDS
    assert suite.validate_suite(declared) == EpisodeLimits(**suite.QUALIFICATION_LIMITS)
    assert declared['run_policy']['live_retirement_qualified'] is False
    assert declared['run_policy']['automatic_retry'] is False
    assert declared['run_policy']['forced_tool_choice'] is False
    assert 'scripts.native_recovery_synthetic' in declared['source_digests']
    for case in declared['cases']:
        assert case['expected']['value'] not in case['prompt']
        assert case['peer_operation']['text'] not in case['prompt']
        target = next(seed for seed in case['seeds'] if seed['alias'] == case['peer_operation']['target_alias'])
        assert case['peer_operation']['text'].startswith(target['text'])
        assert case['peer_operation']['operation'] == 'update'
        assert case['requested_operation'] == 'supersede'
    with pytest.raises(FileExistsError):
        suite.prepare_suite(path, identity={'synthetic': True}, limits=EpisodeLimits(**suite.QUALIFICATION_LIMITS))


@pytest.mark.parametrize('tamper', ['peer_key', 'target', 'forced_tool_choice', 'limit'])
def test_self_consistent_digest_does_not_override_declared_recovery_policy(tmp_path, tamper):
    body = deepcopy(suite.prepare_suite(tmp_path / 'suite.json', identity={}, limits=EpisodeLimits(**suite.QUALIFICATION_LIMITS)))
    if tamper == 'peer_key':
        body['cases'][0]['peer_operation']['idempotency_key'] = 'another-operation'
    elif tamper == 'target':
        body['cases'][0]['peer_operation']['target_alias'] = 'current'
    elif tamper == 'forced_tool_choice':
        body['run_policy']['forced_tool_choice'] = True
    else:
        body['limits']['max_steps'] = 7
    body['suite_digest'] = suite._digest({key: value for key, value in body.items() if key != 'suite_digest'})
    with pytest.raises(ValueError):
        suite.validate_suite(body)


def test_profile_and_budget_cannot_be_changed_at_declaration(tmp_path):
    with pytest.raises(ValueError):
        suite.prepare_suite(tmp_path / 'suite.json', identity={}, limits=EpisodeLimits(**{**suite.QUALIFICATION_LIMITS, 'max_steps': 7}))
    with pytest.raises(ValueError):
        suite.prepare_suite(tmp_path / 'suite.json', identity={}, limits=EpisodeLimits(**suite.QUALIFICATION_LIMITS), consumer_profile='legacy')
