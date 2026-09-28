"""Controlled real-journal recovery fixtures; scripted outputs are not live proof."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import canonical_json, decode_json, validate_episode_events
from daystrom_dml.services.agent_episode import EpisodeLimits
from scripts import native_recovery_synthetic as diagnostic
from scripts.native_recovery_suite import prepare_suite
from scripts.native_dml_synthetic import QUALIFICATION_LIMITS
from test_native_recovery_v5 import RecoveryConsumer


@pytest.fixture
def suite(tmp_path):
    return prepare_suite(tmp_path / 'suite.json', identity={'synthetic': True},
        limits=EpisodeLimits(**QUALIFICATION_LIMITS))


@pytest.mark.parametrize('index', [0, 1])
def test_real_peer_conflict_model_recovery_and_original_verifier(tmp_path, suite, monkeypatch, index):
    case = suite['cases'][index]
    events = []

    def choose(request):
        records = decode_json(request['messages'][-1]['content'])['records']
        refs = {record['source']: record['record_ref'] for record in records}
        return {'kind': 'tool', 'name': 'supersede', 'arguments': {
            'record_ref': refs['synthetic-stale'], 'replacement_ref': refs['synthetic-current'],
            'reason': 'controlled model-selected replacement'}}

    def final(request):
        records = decode_json(request['messages'][-1]['content'])['records']
        current = next(record for record in records if record['source'] == 'synthetic-current')
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': {'claims': [{
            'key': case['expected']['key'], 'value': case['expected']['value'], 'evidence_ids': [current['id']]}]}}

    retrieve = {'kind': 'tool', 'name': 'retrieve', 'arguments': {'query': ' '.join(seed['text'] for seed in case['seeds']), 'top_k': 10}}

    class Consumer(RecoveryConsumer):
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])

        def __init__(self, *args, **kwargs):
            super().__init__([deepcopy(retrieve), choose, deepcopy(retrieve), choose, deepcopy(retrieve), final])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(diagnostic, 'NativeRemoteVLLMActionInputConsumer', Consumer)
    directory = tmp_path / case['id']
    directory.mkdir()
    diagnostic._worker(SimpleNamespace(put=events.append), snapshot='unused-synthetic',
        directory=directory, suite=suite, case=case)
    errors = [event for event in events if event['kind'] == 'diagnostic_error']
    assert not errors, errors
    episode = [event for event in events if 'episode_id' in event]
    validate_episode_events(episode)
    authority = diagnostic.validate_recovery_authority(case, events)
    assert authority['peer_operations'] == 1 and not authority['pending_dispatch']
    result = diagnostic.assess_case(case, events)
    assert result['synthetic_case_pass'], result
    assert result['qualified_conflicts'] == 1
    assert result['model_calls_completed'] == 6
    fixture = next(event['payload'] for event in events if event['kind'] == 'fixture_ready')
    peer = next(event['payload'] for event in events if event['kind'] == 'external_peer_completed')
    baseline = next(event['payload'] for event in events if event['kind'] == 'verifier_baseline')
    assert peer['model_owned'] is False
    target = case['peer_operation']['target_alias']
    assert fixture['seed_records'][target]['text'] != baseline['seed_records'][target]['text']
    assert canonical_json(diagnostic.verifier_baseline(case, fixture, peer)) == canonical_json(baseline)
    requests = [event for event in episode if event['kind'] == 'model_requested']
    assert peer['registered_reference'] not in requests[1]['payload']['request']['messages'][-1]['content']
    assert peer['registered_reference'] in requests[3]['payload']['request']['messages'][-1]['content']
    forged = deepcopy(peer)
    forged['result']['receipt']['result']['memory']['meta']['claim_value'] = 'forged'
    with pytest.raises(ValueError):
        diagnostic.verifier_baseline(case, fixture, forged)
    # A model final before conflict never qualifies, even if text is correct.
    assert not diagnostic.assess_case(case, [event for event in events if event['kind'] != 'tool_failed'])['synthetic_case_pass']
    assert not diagnostic.assess_case(case, [event for event in events if event['kind'] != 'external_peer_completed'])['synthetic_case_pass']
    final_read = max(i for i, event in enumerate(events) if event['kind'] == 'tool_completed' and event['payload']['name'] == 'retrieve')
    assert not diagnostic.assess_case(case, events[:final_read] + events[final_read + 1:])['synthetic_case_pass']

    no_readback = deepcopy(events)
    no_readback[final_read]['payload']['result']['observed_records'] = []
    no_readback[final_read]['payload']['model_result'] = '{"records":[]}'
    assert not diagnostic.assess_case(case, no_readback)['synthetic_case_pass']
    unrelated_readback = deepcopy(events)
    unrelated_readback[final_read]['payload']['result']['observed_records'] = [fixture['seed_records']['stale']]
    unrelated_readback[final_read]['payload']['model_result'] = canonical_json({'records': [
        {'id': fixture['seed_records']['stale']['id'], 'record_ref': 'unrelated'}]}).decode()
    assert not diagnostic.assess_case(case, unrelated_readback)['synthetic_case_pass']
    wrong_feedback = deepcopy(events)
    next(event for event in wrong_feedback if event['kind'] == 'model_requested' and event['call_id'] == 'model-2')['payload']['request']['messages'][-1]['tool_call_id'] = 'wrong-native-call'
    assert not diagnostic.assess_case(case, wrong_feedback)['synthetic_case_pass']
