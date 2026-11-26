"""Planning observations never replace original verifier, citations or authority."""
from copy import deepcopy
import json

import pytest

from daystrom_dml.services.agent_episode import EpisodeLimits, _prepare_fixture
from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
from daystrom_dml.contracts.agent_episode import EXECUTION_PROTOCOL_V2
from scripts import qwen3_gguf_retrieval_diagnostic as diagnostic


@pytest.fixture
def suite(tmp_path):
    return diagnostic.prepare_suite(tmp_path / 'planning.json', identity={'test': True},
        limits=EpisodeLimits(**diagnostic.shared.QUALIFICATION_LIMITS))


def test_predeclared_readonly_hidden_facts_and_no_forced_query(suite):
    diagnostic.validate_suite(suite)
    assert len(suite['cases']) == 2
    assert [len(c['seeds']) for c in suite['cases']] == [2, 5]
    for case in suite['cases']:
        assert case['allowed_tools'] == ['retrieve']
        assert case['task']['state_expectations'] == []
        for value in case['task']['truth'].values():
            assert value['value'] not in case['prompt']
            assert all(value['value'] not in query for query in case['preflight_queries'])
    assert suite['run_policy']['forced_initial_query'] is False
    assert suite['run_policy']['automatic_retry'] is False


def test_tamper_and_overwrite_rejected(suite, tmp_path):
    altered = deepcopy(suite)
    altered['cases'][0]['allowed_tools'].append('supersede')
    altered['suite_digest'] = diagnostic.shared._digest({k: v for k, v in altered.items() if k != 'suite_digest'})
    with pytest.raises(ValueError):
        diagnostic.validate_suite(altered)
    with pytest.raises(FileExistsError):
        diagnostic.prepare_suite(tmp_path / 'planning.json', identity={},
            limits=EpisodeLimits(**diagnostic.shared.QUALIFICATION_LIMITS))


def evidence(case, *, queries=('broad description',), results=((10, 11),)):
    events = [{'kind': 'fixture_ready', 'payload': {'seed_records': {'old': {'id': 10}, 'current': {'id': 11}}}}]
    for index, (query, ids) in enumerate(zip(queries, results)):
        call = 't' + str(index)
        events.extend([
            {'kind': 'tool_requested', 'call_id': call, 'payload': {'name': 'retrieve', 'arguments': {'query': query, 'top_k': 10}}},
            {'kind': 'tool_completed', 'call_id': call, 'payload': {'name': 'retrieve', 'model_result': json.dumps({'records': [{'id': i} for i in ids]})}},
        ])
    claims = [{'key': key, 'value': value['value'], 'evidence_ids': [10 if value['evidence_aliases'] == ['old'] else 11]}
              for key, value in case['task']['truth'].items()]
    events.extend([
        {'kind': 'model_requested', 'call_id': 'final', 'payload': {'request': {'messages': [{'role': 'tool'}]}}},
        {'kind': 'model_completed', 'call_id': 'final', 'payload': {}},
        {'kind': 'verifier_result', 'payload': {'success': True}},
        {'kind': 'outcome', 'payload': {'status': 'completed', 'answer': {'claims': claims}}},
    ])
    return events


def test_direct_complete_lookup_passes_without_reformulation_credit(suite):
    case = suite['cases'][0]
    result = diagnostic.assess_case(case, evidence(case))
    assert result['synthetic_case_pass']
    assert not result['query_reformulation_observed']


@pytest.mark.parametrize('second_query,expected', [('changed missing role', True), ('first query', False)])
def test_reformulation_requires_changed_query_and_new_missing_evidence(suite, second_query, expected):
    case = suite['cases'][0]
    events = evidence(case, queries=('first query', second_query), results=((11,), (10, 11)))
    result = diagnostic.assess_case(case, events)
    assert result['synthetic_case_pass']
    assert result['query_reformulation_observed'] is expected
    events = evidence(case, queries=('first query', second_query), results=((11,), (11,)))
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']
    assert not diagnostic.assess_case(case, events)['query_reformulation_observed']


@pytest.mark.parametrize('failure', ['verifier', 'missing_claim', 'wrong_citation', 'mutation', 'rejection'])
def test_original_failures_cannot_be_reclassified_as_success(suite, failure):
    case = suite['cases'][0]
    events = evidence(case)
    if failure == 'verifier':
        events[-2]['payload']['success'] = False
    elif failure == 'missing_claim':
        events[-1]['payload']['answer']['claims'].pop()
    elif failure == 'wrong_citation':
        events[-1]['payload']['answer']['claims'][0]['evidence_ids'] = [11]
    elif failure == 'mutation':
        events.insert(1, {'kind': 'tool_requested', 'payload': {'name': 'retire'}})
    else:
        events.insert(1, {'kind': 'tool_validation_rejected', 'payload': {}})
    assert not diagnostic.assess_case(case, events)['synthetic_case_pass']


def test_actual_unchanged_retrieval_can_find_both_declared_records(suite, tmp_path):
    for case in suite['cases']:
        adapter, fixture = _prepare_fixture(tmp_path / case['id'],
            {'scope': case['scope'], 'seeds': case['seeds'], 'setup': []}, 'planning-control')
        try:
            tools = SelectedProfileEpisodeTools(adapter, scope=case['scope'], episode_id='planning-control',
                seed_receipts=fixture['seed_receipts'], allowed_tools=('retrieve',),
                effective_time=suite['effective_time'], execution_protocol=EXECUTION_PROTOCOL_V2)
            observed = set()
            for index, query in enumerate(case['preflight_queries']):
                prepared = tools.prepare('retrieve', {'query': query, 'top_k': 10}, call_id=str(index))
                _, public = tools.execute(prepared)
                observed.update(record['id'] for record in json.loads(public)['records'])
            assert {fixture['seed_records'][role]['id'] for role in ('old', 'current')}.issubset(observed)
        finally:
            adapter.close()


@pytest.mark.parametrize('condition', ['pass', 'known_failure', 'unknown', 'readiness_unknown'])
def test_stages_advance_only_on_complete_success_once(suite, tmp_path, monkeypatch, condition):
    readiness = tmp_path / 'readiness.json'
    diagnostic.shared.prepare_suite(readiness, identity={'test': True},
        limits=EpisodeLimits(**diagnostic.shared.QUALIFICATION_LIMITS), consumer_profile=diagnostic.PROFILE)
    calls = []
    def planning_run(path, *, snapshot, output, suite_validator, case_assessor):
        calls.append('planning')
        output.mkdir()
        result = {'all_synthetic_cases_pass': condition in ('pass', 'readiness_unknown'),
            'cases': [{'synthetic_case_pass': True}, {'synthetic_case_pass': condition in ('pass', 'readiness_unknown')}],
            'unrun_cases': [], 'unrun_reason': 'unknown_usage' if condition == 'unknown' else None}
        diagnostic.shared._write_exclusive(output / 'summary.json', result)
        return result
    def readiness_run(*args, **kwargs):
        calls.append('readiness')
        return {'all_synthetic_cases_pass': condition == 'pass',
            'unrun_cases': list(diagnostic.shared.CASE_IDS[1:]) if condition == 'readiness_unknown' else [],
            'unrun_reason': 'tool_effects_or_usage_unknown' if condition == 'readiness_unknown' else None}
    monkeypatch.setattr(diagnostic.shared, 'run_declared_suite', planning_run)
    monkeypatch.setattr(diagnostic.shared, 'run_suite', readiness_run)
    result = diagnostic.run_staged(tmp_path / 'planning.json', readiness, snapshot='unused', output=tmp_path / 'run')
    assert calls == (['planning', 'readiness'] if condition in ('pass', 'readiness_unknown') else ['planning'])
    assert result['readiness_started'] is (condition in ('pass', 'readiness_unknown'))
    if condition == 'readiness_unknown':
        assert result['readiness_unrun_cases'] == list(diagnostic.shared.CASE_IDS[1:])
        assert result['readiness_unrun_reason'] == 'tool_effects_or_usage_unknown'
        assert not result['readiness_passed']
    else:
        assert result['readiness_unrun_cases'] == ([] if condition == 'pass' else list(diagnostic.shared.CASE_IDS))
    with pytest.raises(FileExistsError):
        diagnostic.run_staged(tmp_path / 'planning.json', readiness, snapshot='unused', output=tmp_path / 'run')
