"""Fail-closed scoring for the separate local Qwen GGUF live-DML diagnostic."""
from copy import deepcopy

import pytest

from scripts import qwen3_gguf_synthetic as diagnostic
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
    assert suite['run_policy']['all_turn_json_grammar'] is True
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
    from test_agent_episode_runtime import ValidationScriptedConsumer
    from dataclasses import replace

    case = suite['cases'][0]
    def final(request):
        records = json.loads(request['messages'][-1]['content'])['records']
        record = next(record for record in records if record.get('claim_value') == case['expected']['value'])
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': {'claims': [
            {'key': case['expected']['key'], 'value': case['expected']['value'], 'evidence_ids': [record['id']]}]}}
    class Synthetic(ValidationScriptedConsumer):
        profile = diagnostic.CONSUMER_PROFILE
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])
        def __init__(self, *args, **kwargs):
            super().__init__([
                {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'retrieve',
                 'arguments': {'query': 'Synthetic archive access phrase', 'top_k': 10}}, final])
        def compile(self, *args, **kwargs):
            artifact = super().compile(*args, **kwargs)
            return replace(artifact, identity=replace(artifact.identity,
                runtime_identity='dml-qwen3-gguf-arm64-action-runtime-v1:' + 'a' * 64))
        def execute(self, artifact):
            if execute_failure:
                raise RuntimeError('synthetic execution failure after dispatch')
            return super().execute(artifact)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(diagnostic, 'LocalQwen3GGUFActionInputConsumer', Synthetic)
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



def test_reject_native_profiles_even_if_suite_digest_rebound(tmp_path, suite):
    from daystrom_dml.contracts.agent_episode import NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
    with pytest.raises(ValueError, match='Only the declared Qwen'):
        diagnostic.prepare_suite(tmp_path / 'wrong.json', identity={},
            limits=EpisodeLimits(**diagnostic.QUALIFICATION_LIMITS),
            consumer_profile=NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE)
    altered = deepcopy(suite)
    altered['consumer_profile'] = NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
    altered['suite_digest'] = diagnostic._digest({k: v for k, v in altered.items() if k != 'suite_digest'})
    with pytest.raises(ValueError):
        diagnostic.validate_suite(altered)


def test_dependency_fixture_real_gateway_separates_lookup(tmp_path, suite):
    from daystrom_dml.services.agent_episode import _prepare_fixture
    case = suite['cases'][1]
    scenario = {'scope': case['scope'], 'seeds': case['seeds'], 'setup': []}
    adapter, fixture = _prepare_fixture(tmp_path / 'dependency', scenario, 'control')
    try:
        assert diagnostic.dependency_preflight(adapter, fixture, case, suite['effective_time'])['passed']
    finally:
        adapter.close()


def test_legacy_same_record_rejection_is_model_owned_and_charged(tmp_path, suite, monkeypatch):
    """Scripted noncorpus gateway control, never trained-model qualification."""
    import json
    from queue import Queue
    from dataclasses import replace
    from types import SimpleNamespace
    from test_agent_episode_runtime import ValidationScriptedConsumer
    from daystrom_dml.contracts.agent_episode import VALIDATION_MODEL_RESULT, validate_episode_events
    case = suite['cases'][2]
    refs = {}
    def retrieve(_request):
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'retrieve',
                'arguments': {'query': case['prompt'], 'top_k': 10}}
    def invalid(request):
        records = json.loads(request['messages'][-1]['content'])['records']
        refs.update({record['claim_value']: record['record_ref'] for record in records})
        ref = refs[case['stale_value']]
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'supersede',
                'arguments': {'record_ref': ref, 'replacement_ref': ref, 'reason': 'controlled same-record rejection'}}
    def valid(request):
        assert request['messages'][-1]['content'] == VALIDATION_MODEL_RESULT
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'supersede',
                'arguments': {'record_ref': refs[case['stale_value']], 'replacement_ref': refs[case['expected']['value']],
                              'reason': 'controlled subsequent independent decision'}}
    def final(request):
        records = json.loads(request['messages'][-1]['content'])['records']
        record = next(record for record in records if record['claim_value'] == case['expected']['value'])
        return {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': {'claims': [
            {'key': case['expected']['key'], 'value': case['expected']['value'], 'evidence_ids': [record['id']]}]}}
    class Synthetic(ValidationScriptedConsumer):
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])
        def __init__(self, *args, **kwargs):
            super().__init__([retrieve, invalid, valid, retrieve, final])
        def compile(self, *args, **kwargs):
            artifact = super().compile(*args, **kwargs)
            return replace(artifact, identity=replace(artifact.identity,
                runtime_identity='dml-qwen3-gguf-arm64-action-runtime-v1:' + 'a' * 64))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(diagnostic, 'LocalQwen3GGUFActionInputConsumer', Synthetic)
    channel = Queue()
    directory = tmp_path / 'legacy'
    directory.mkdir()
    diagnostic._worker(channel, snapshot='scripted-control-only', directory=directory, suite=suite, case=case)
    events = list(channel.queue)
    assert not [e for e in events if e['kind'] == 'diagnostic_error'], events
    assert len([e for e in events if e['kind'] == 'model_completed']) == 5
    assert len([e for e in events if e['kind'] == 'tool_validation_rejected']) == 1
    assert len([e for e in events if e['kind'] == 'tool_completed' and e['payload']['name'] == 'supersede']) == 1
    assert not [e for e in events if e['kind'] == 'tool_failed']
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']
    validate_episode_events([e for e in events if 'episode_id' in e])


@pytest.mark.parametrize('mode', ['deadline', 'termination_unknown', 'cleanup_failure'])
def test_local_supervisor_retains_interruption_and_stops(tmp_path, suite, monkeypatch, mode):
    from queue import Queue, Empty
    from types import SimpleNamespace
    from daystrom_dml.services import agent_episode
    ticks = iter([0.0, 0.0, 301.0])
    monkeypatch.setattr(diagnostic.time, 'monotonic', lambda: next(ticks, 301.0))
    launches = []
    class Process:
        exitcode = -15
        alive = True
        def __init__(self, *, target, kwargs):
            self.kwargs = kwargs
        def start(self):
            launches.append(self.kwargs['case']['id'])
            assert 'scratch_directory' in self.kwargs
        def is_alive(self):
            return self.alive
        def join(self, **kwargs):
            pass
        def terminate(self):
            if mode != 'termination_unknown':
                self.alive = False
        def kill(self):
            self.terminate()
    class ClosingQueue(Queue):
        def get(self, **kwargs):
            raise Empty
        def close(self):
            pass
    if mode == 'cleanup_failure':
        def fail(*args):
            raise OSError('controlled cleanup refusal')
        monkeypatch.setattr(agent_episode, '_remove_worker_scratch', fail)
    monkeypatch.setattr(diagnostic.multiprocessing, 'get_context', lambda *args:
        SimpleNamespace(Queue=ClosingQueue, Process=Process))
    result = diagnostic.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'run')
    assert launches == ['live_retrieval']
    assert result['unrun_reason'] == {'deadline': 'deadline', 'termination_unknown': 'worker_termination_unconfirmed',
                                     'cleanup_failure': 'worker_scratch_cleanup_failed'}[mode]
    import json
    events = [json.loads(line) for line in (tmp_path / 'run/live_retrieval/events.jsonl').read_text().splitlines()]
    assert any(event['kind'] == 'supervisor_interrupted' for event in events)
    with pytest.raises(FileExistsError):
        diagnostic.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'another-run')
