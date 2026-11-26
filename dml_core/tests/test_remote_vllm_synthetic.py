"""Synthetic controls do not run live generation or the evaluation corpus."""
from copy import deepcopy
import json
from queue import Queue
from types import SimpleNamespace

import pytest

from scripts import remote_vllm_synthetic as diagnostic
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.model_input import ModelInputResult


@pytest.fixture
def suite(tmp_path):
    return diagnostic.prepare_suite(tmp_path / 'suite.json', identity={'test': True}, limits=EpisodeLimits())


def test_predeclared_secret_not_in_prompt_and_complete_policy(suite):
    diagnostic.validate_suite(suite)
    assert len(suite['cases']) == 3
    for case in suite['cases']:
        assert case['expected']['value'] not in case['prompt']
        assert any(case['expected']['value'] in seed['text'] for seed in case['seeds'])
    assert suite['run_policy']['forced_tool_choice'] is False
    assert suite['run_policy']['final_branch_available'] is True
    assert suite['run_policy']['automatic_retry'] is False
    assert suite['run_policy']['runs_per_case'] == 1


def test_suite_tampering_and_overwrite_rejected(suite, tmp_path):
    changed = deepcopy(suite)
    changed['cases'][0]['prompt'] = 'changed'
    with pytest.raises(ValueError):
        diagnostic.validate_suite(changed)
    with pytest.raises(FileExistsError):
        diagnostic.prepare_suite(tmp_path / 'suite.json', identity={}, limits=EpisodeLimits())


def test_early_final_does_not_fake_tool_use(suite):
    case = suite['cases'][0]
    events = [{'kind': 'outcome', 'payload': {'status': 'completed', 'answer': {'claims': []}}}]
    result = diagnostic.assess_case(case, events)
    assert result['model_selected_retrieve_calls'] == 0
    assert result['calls_after_real_tool_feedback'] == 0
    assert result['synthetic_case_pass'] is False


def test_actual_bridge_feedback_reaches_second_call(tmp_path, suite, monkeypatch):
    """Scripted model control exercises the real trusted bridge and production loop."""
    case = suite['cases'][0]
    compiled_messages = []
    acknowledged = []
    class Scripted:
        last_exchange = {'test_only': True}
        identity = SimpleNamespace(to_payload=lambda: suite['model_identity'])
        def __init__(self, *args, **kwargs):
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def compile(self, messages, tools, *, output_reserved_tokens):
            compiled_messages.append(deepcopy(messages))
            assert [tool['function']['name'] for tool in tools] == ['retrieve']
            return SimpleNamespace(input_tokens=3, output_reserved_tokens=output_reserved_tokens,
                artifact_digest='a' * 64, signing_payload=lambda: {'test_only': True})
        def execute(self, artifact):
            assert acknowledged[-1] == 'model_requested'
            if not self.calls:
                action = {'schema_version': 'dml-agent-action-v1', 'kind': 'tool', 'name': 'retrieve',
                          'arguments': {'query': 'Synthetic cabinet current access phrase', 'top_k': 10}}
            else:
                result = json.loads(compiled_messages[-1][-1]['content'])
                record_id = next(record['id'] for record in result['records']
                                 if case['expected']['value'] in record['text'])
                action = {'schema_version': 'dml-agent-action-v1', 'kind': 'final',
                          'answer': {'claims': [{'key': case['expected']['key'],
                              'value': case['expected']['value'], 'evidence_ids': [record_id]}]}}
            self.calls += 1
            return ModelInputResult(artifact.artifact_digest, 3, 1, (1,), json.dumps(action))
    monkeypatch.setattr(diagnostic, 'RemoteVLLMActionInputConsumer', Scripted)
    directory = tmp_path / 'case'
    directory.mkdir()
    channel = Queue()
    acknowledgement = SimpleNamespace(get=lambda **kwargs: acknowledged.append(channel.queue[-1]['kind']))
    diagnostic._worker(channel, snapshot='unused', directory=directory, suite=suite, case=case,
                       acknowledgement=acknowledgement)
    events = []
    while not channel.empty():
        events.append(channel.get())
    assert not [event for event in events if event['kind'] == 'diagnostic_error'], events
    assert len(compiled_messages) == 2
    assert compiled_messages[1][-2]['role'] == 'assistant'
    assert compiled_messages[1][-2]['tool_calls'][0]['function']['name'] == 'retrieve'
    assert compiled_messages[1][-1]['role'] == 'tool'
    result = diagnostic.assess_case(case, events)
    assert result['synthetic_case_pass']
    assert result['model_selected_retrieve_calls'] == result['calls_after_real_tool_feedback'] == 1


def test_unknown_remote_failure_stops_remaining_cases(tmp_path, suite, monkeypatch):
    launches = []
    class Process:
        exitcode = 0
        def __init__(self, *, target, kwargs):
            self.kwargs = kwargs
        def start(self):
            launches.append(self.kwargs['case']['id'])
            for event in [
                {'kind': 'model_failed', 'payload': {'phase': 'execute', 'input_token_count': None,
                                                   'output_token_count': None}},
                {'kind': 'outcome', 'payload': {'status': 'model_error', 'answer': None}},
                {'kind': 'worker_finished', 'payload': {}}]:
                self.kwargs['channel'].put(event)
        def is_alive(self):
            return False
        def join(self, **kwargs):
            pass
    class ClosingQueue(Queue):
        def close(self):
            pass
    monkeypatch.setattr(diagnostic.multiprocessing, 'get_context',
                        lambda *args: SimpleNamespace(Queue=ClosingQueue, Process=Process))
    result = diagnostic.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'run')
    assert launches == ['hidden_lookup']
    assert result['unrun_cases'] == ['stale_user_correction', 'stored_correction']
    assert result['unrun_reason'] == 'remote_execution_failed_usage_or_cancellation_unknown'
    with pytest.raises(FileExistsError):
        diagnostic.run_suite(tmp_path / 'suite.json', snapshot='unused', output=tmp_path / 'repeat')


def test_partial_followup_or_contradictory_answer_not_pass(suite):
    case = suite['cases'][0]
    expected = case['expected']
    claims = [{'key': expected['key'], 'value': expected['value'], 'evidence_ids': [7]}]
    events = [
        {'kind': 'fixture_ready', 'payload': {'seed_records': {'current': {'id': 7}}}},
        {'kind': 'tool_completed', 'payload': {'name': 'retrieve', 'model_result': '{"records":[{"id":7}]}'}},
        {'kind': 'model_requested', 'call_id': 'model-1', 'payload': {'request': {'messages': [{'role': 'tool'}]}}},
        {'kind': 'outcome', 'payload': {'status': 'completed', 'answer': {'claims': claims}}}]
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    events.insert(3, {'kind': 'model_completed', 'call_id': 'model-1'})
    assert diagnostic.assess_case(case, events)['synthetic_case_pass']
    claims.append({'key': expected['key'], 'value': case['stale_value'], 'evidence_ids': [7]})
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
