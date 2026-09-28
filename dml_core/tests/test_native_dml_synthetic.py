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


def test_dependency_requires_separation_then_later_actual_query(suite):
    import json
    case = suite['cases'][1]
    events = events_for(case)
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    first = events[2]
    first['payload']['model_result'] = json.dumps({'records': [{'id': 2, 'text': case['dependency_key']}]})
    query = {'kind': 'tool_requested', 'call_id': 't1', 'payload': {'name': 'retrieve',
             'arguments': {'query': case['dependency_key'], 'top_k': 1}}}
    later_result = {'kind': 'tool_completed', 'call_id': 't1', 'payload': {'name': 'retrieve',
                    'model_result': json.dumps({'records': [{'id': 7, 'text': case['expected']['value']}]})}}
    events[-2:-2] = [query, later_result]
    assert diagnostic.assess_case(case, events)['dependent_feedback_observed']
    first['payload']['model_result'] = later_result['payload']['model_result'] + ' '
    assert not diagnostic.assess_case(case, events)['dependent_feedback_observed']
    first['payload']['model_result'] = json.dumps({'records': [{'id': 2, 'text': case['dependency_key'] + case['expected']['value']}]})
    assert not diagnostic.assess_case(case, events)['dependent_feedback_observed']


def test_dependency_declaration_uses_unchanged_real_retrieval(tmp_path, suite):
    from daystrom_dml.services.agent_episode import _prepare_fixture
    case = suite['cases'][1]
    scenario = {'scope': case['scope'], 'seeds': case['seeds'], 'setup': [], 'expected_memory_count': len(case['seeds'])}
    adapter, fixture = _prepare_fixture(tmp_path / 'dependency-authority', scenario, 'dependency-test')
    try:
        result = diagnostic.dependency_preflight(adapter, fixture, case, suite['effective_time'])
        assert result['passed'], result
        assert len(case['seeds']) == 11
    finally:
        adapter.close()


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
    injected = {'id': 7, 'text': 'Quoted data ' + case['injection_text']}
    events[0]['payload']['seed_records'][case['injection_alias']] = injected
    result = json.loads(events[2]['payload']['model_result'])
    result['records'] = [injected]
    events[2]['payload']['model_result'] = json.dumps(result)
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']
    events.insert(2, {'kind': 'action_rejected', 'payload': {}})
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']


@pytest.mark.parametrize('execute_failure', [False, True])
def test_worker_exercises_real_gateway_verifier_and_durable_ack(tmp_path, suite, monkeypatch, execute_failure):
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
        profile = diagnostic.CONSUMER_PROFILE
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])
        def __init__(self, *args, **kwargs):
            super().__init__([
                {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'retrieve',
                 'arguments': {'query': 'Synthetic archive access phrase', 'top_k': 10}}, final])
        def execute(self, artifact):
            if execute_failure:
                raise RuntimeError('synthetic execution failure after dispatch')
            return super().execute(artifact)
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
    result = diagnostic.assess_case(case, events)
    assert result['synthetic_case_pass'] is (not execute_failure)
    if execute_failure:
        terminal = next(event['payload'] for event in events if event['kind'] == 'terminal')
        assert terminal['status'] == 'model_error'
        assert terminal['retrieval_ms'] is None


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


def test_injection_fixture_reaches_model_through_unchanged_retrieval(tmp_path, suite):
    import json
    from daystrom_dml.contracts.agent_episode import EXECUTION_PROTOCOL_V2
    from daystrom_dml.services.agent_episode import _prepare_fixture
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
    case = suite['cases'][3]
    scenario = {'scope': case['scope'], 'seeds': case['seeds'], 'setup': []}
    adapter, fixture = _prepare_fixture(tmp_path / 'authority', scenario, 'exposure-control')
    try:
        toolbox = SelectedProfileEpisodeTools(adapter, scope=case['scope'], episode_id='exposure-control',
            seed_receipts=fixture['seed_receipts'], allowed_tools=('retrieve',),
            effective_time=suite['effective_time'], execution_protocol=EXECUTION_PROTOCOL_V2)
        prepared = toolbox.prepare('retrieve', {'query': case['prompt'], 'top_k': 10}, call_id='tool-0')
        _, model_result = toolbox.execute(prepared)
        record = fixture['seed_records'][case['injection_alias']]
        shown = json.loads(model_result)['records']
        assert record['meta']['source_trust'] == 'trusted'
        assert any(item['id'] == record['id'] and item['text'] == record['text']
                   and case['injection_text'] in item['text'] for item in shown)
        assert case['injection_text'] not in case['prompt']
    finally:
        adapter.close()


def test_original_four_support_v6_without_changing_default(tmp_path):
    from daystrom_dml.contracts.agent_episode import NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
    old = diagnostic.prepare_suite(tmp_path / 'v4.json', identity={},
        limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS))
    new = diagnostic.prepare_suite(tmp_path / 'v6.json', identity={},
        limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS), consumer_profile=NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE)
    assert old['consumer_profile'] == diagnostic.CONSUMER_PROFILE
    assert new['consumer_profile'] == NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
    assert old['limits'] == new['limits'] and old['sampling'] == new['sampling']
    assert old['run_policy'] == new['run_policy']
    assert [case['id'] for case in new['cases']] == list(diagnostic.CASE_IDS)
    assert 'daystrom_dml.services.receipt_conflict_boundary' in new['source_digests']
    diagnostic.validate_suite(new)


@pytest.mark.parametrize('recovery', [False, True])
@pytest.mark.parametrize('unknown_effects', [False, True])
def test_v6_unknown_effects_stop_each_suite_but_known_failure_does_not(tmp_path, monkeypatch, recovery, unknown_effects):
    from queue import Queue
    from types import SimpleNamespace
    from daystrom_dml.contracts.agent_episode import NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
    from scripts import native_recovery_synthetic, native_recovery_suite
    target = native_recovery_synthetic if recovery else diagnostic
    prepare = native_recovery_suite.prepare_suite if recovery else diagnostic.prepare_suite
    prepared = prepare(tmp_path / 'suite.json', identity={}, limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS),
                       consumer_profile=NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE)
    launches = []
    class Process:
        exitcode = 0
        def __init__(self, *, target, kwargs):
            self.kwargs = kwargs
        def start(self):
            launches.append(self.kwargs['case']['id'])
            for event in [
                {'kind': 'tool_failed', 'payload': {'name': 'retrieve', 'effects': 'unknown' if unknown_effects else 'none'}},
                {'kind': 'terminal', 'payload': {'effects_unknown': unknown_effects, 'usage_unknown': False}},
                {'kind': 'worker_finished', 'payload': {}}]:
                self.kwargs['channel'].put(event)
        def is_alive(self):
            return False
        def join(self, **kwargs):
            pass
    class ClosingQueue(Queue):
        def close(self):
            pass
    monkeypatch.setattr(target.multiprocessing, 'get_context', lambda *args: SimpleNamespace(Queue=ClosingQueue, Process=Process))
    result = target.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'run')
    assert launches == [case['id'] for case in prepared['cases'][:1 if unknown_effects else len(prepared['cases'])]]
    assert (result['unrun_reason'] == 'tool_effects_or_usage_unknown') is unknown_effects
