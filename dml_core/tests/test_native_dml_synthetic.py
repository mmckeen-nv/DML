"""Fail-closed scoring for the separate native live-DML diagnostic."""
from copy import deepcopy

import pytest

from scripts import native_dml_synthetic as diagnostic
from daystrom_dml.services.agent_episode import EpisodeLimits


@pytest.fixture
def suite(tmp_path):
    return diagnostic.prepare_suite(tmp_path / 'suite.json', identity={'test': True},
        limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS))


def test_suite_predeclares_hidden_facts_and_original_limits(suite):
    diagnostic.validate_suite(suite)
    assert len(suite['cases']) == 4
    for case in suite['cases']:
        assert case['expected']['value'] not in case['prompt']
        assert case['dependency_key'] not in case['prompt']
        assert case['task']['truth'][case['expected']['key']]['value'] == case['expected']['value']
    assert suite['run_policy']['automatic_retry'] is False
    assert suite['run_policy']['all_turn_json_grammar'] is False
    assert suite['run_policy']['final_contract_unchanged'] is True
    assert suite['cases'][2]['allowed_tools'] == ['retrieve', 'supersede']
    assert suite['cases'][2]['task']['state_expectations']


def test_reject_different_limits_and_overwrite(tmp_path, suite):
    with pytest.raises(ValueError):
        diagnostic.prepare_suite(tmp_path / 'different.json', identity={}, limits=EpisodeLimits())
    with pytest.raises(FileExistsError):
        diagnostic.prepare_suite(tmp_path / 'suite.json', identity={},
            limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS))
    changed = deepcopy(suite)
    changed['cases'][0]['prompt'] = 'changed'
    with pytest.raises(ValueError):
        diagnostic.validate_suite(changed)


def test_early_final_never_qualifies(suite):
    for case in suite['cases']:
        result = diagnostic.assess_case(case, [
            {'kind': 'outcome', 'payload': {'status': 'completed', 'answer': {'claims': []}}}])
        assert not result['synthetic_case_pass']
        assert result['model_calls_completed'] == 0


def events_for(case):
    import json
    return [
        {'kind': 'fixture_ready', 'payload': {'seed_records': {'current': {'id': 7}}}},
        {'kind': 'model_completed', 'call_id': 'm0', 'payload': {}},
        {'kind': 'tool_completed', 'payload': {'name': 'retrieve',
            'model_result': json.dumps({'records': [{'id': 7}], 'key': case['dependency_key']})}},
        {'kind': 'model_requested', 'call_id': 'm1', 'payload': {'request': {'messages': [{'role': 'tool'}]}}},
        {'kind': 'model_completed', 'call_id': 'm1', 'payload': {}},
        {'kind': 'verifier_result', 'payload': {'success': True}},
        {'kind': 'outcome', 'payload': {'status': 'completed', 'answer': {'claims': [
            {'key': case['expected']['key'], 'value': case['expected']['value'], 'evidence_ids': [7]}]}}},
    ]


def test_original_verifier_failure_overrules_containment(suite):
    case = suite['cases'][0]
    events = events_for(case)
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']
    events[-2]['payload']['success'] = False
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']


def test_dependency_requires_later_actual_query(suite):
    case = suite['cases'][1]
    events = events_for(case)
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    query = {'kind': 'tool_requested', 'call_id': 't1', 'payload': {'name': 'retrieve',
             'arguments': {'query': case['dependency_key'], 'top_k': 1}}}
    events.insert(0, query)
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    events.insert(-2, query)
    later_result = deepcopy(events[3])
    later_result['call_id'] = 't1'
    events.insert(-2, later_result)
    assert diagnostic.assess_case(case, events)['dependent_feedback_observed']


def test_supersession_requires_success_and_later_read(suite):
    case = suite['cases'][2]
    events = events_for(case)
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    events.insert(-2, {'kind': 'tool_completed', 'payload': {'name': 'supersede'}})
    assert not diagnostic.assess_case(case, events)['supersession_and_readback_observed']
    events.insert(-2, deepcopy(events[2]))
    assert diagnostic.assess_case(case, events)['supersession_and_readback_observed']


def test_injection_attempt_cannot_hide_behind_correct_final(suite):
    case = suite['cases'][3]
    events = events_for(case)
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    import json
    injected = {'id': 9, 'text': 'untrusted injection'}
    events[0]['payload']['seed_records']['injection'] = injected
    result = json.loads(events[2]['payload']['model_result'])
    result['records'].append(injected)
    events[2]['payload']['model_result'] = json.dumps(result)
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']
    events.insert(2, {'kind': 'action_rejected', 'payload': {}})
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']


def test_worker_exercises_real_gateway_verifier_and_durable_ack(tmp_path, suite, monkeypatch):
    import json
    from queue import Queue
    from types import SimpleNamespace
    from test_native_episode_integration import NativeSyntheticConsumer

    case = suite['cases'][0]
    def final(request):
        records = json.loads(request['messages'][-1]['content'])['records']
        record = next(record for record in records if record.get('claim_value') == case['expected']['value'])
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': {'claims': [
            {'key': case['expected']['key'], 'value': case['expected']['value'], 'evidence_ids': [record['id']]}]}}
    class Synthetic(NativeSyntheticConsumer):
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])
        def __init__(self, *args, **kwargs):
            super().__init__([
                {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'retrieve',
                 'arguments': {'query': 'Synthetic archive access phrase', 'top_k': 10}}, final])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(diagnostic, 'NativeRemoteVLLMActionInputConsumer', Synthetic)
    channel = Queue()
    acks = []
    directory = tmp_path / 'case'
    directory.mkdir()
    diagnostic._worker(channel, snapshot='test-only', directory=directory, suite=suite, case=case,
        acknowledgement=SimpleNamespace(get=lambda **kwargs: acks.append(channel.queue[-1]['kind'])))
    events = list(channel.queue)
    assert not [event for event in events if event['kind'] == 'diagnostic_error'], events
    assert len(acks) == len(events)
    assert any(event['kind'] == 'terminal' for event in events)
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']


@pytest.mark.parametrize('failure', ['diagnostic_error', 'missing_terminal', 'unknown_execution'])
def test_unresolved_dispatch_or_incomplete_episode_stops_suite(tmp_path, suite, monkeypatch, failure):
    from queue import Queue
    from types import SimpleNamespace
    launches = []
    class Process:
        exitcode = 0
        def __init__(self, *, target, kwargs):
            self.kwargs = kwargs
        def start(self):
            launches.append(self.kwargs['case']['id'])
            events = [{'kind': 'model_requested', 'call_id': 'm0',
                       'payload': {'request': {'messages': []}}}]
            if failure == 'diagnostic_error':
                events.append({'kind': 'diagnostic_error', 'payload': {'traceback': 'test SystemExit'}})
            elif failure == 'unknown_execution':
                events.append({'kind': 'model_failed', 'call_id': 'm0', 'payload': {
                    'phase': 'execute', 'input_token_count': None, 'output_token_count': None}})
            events.append({'kind': 'worker_finished', 'payload': {}})
            for event in events:
                self.kwargs['channel'].put(event)
        def is_alive(self):
            return False
        def join(self, **kwargs):
            pass
    class ClosingQueue(Queue):
        def close(self):
            pass
    monkeypatch.setattr(diagnostic.multiprocessing, 'get_context', lambda *args:
        SimpleNamespace(Queue=ClosingQueue, Process=Process))
    result = diagnostic.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'run')
    assert launches == ['live_retrieval']
    assert result['unrun_cases'] == ['dependent_feedback', 'live_supersession', 'untrusted_injection']
    assert not result['all_synthetic_cases_pass']
    assert result['unrun_reason']
