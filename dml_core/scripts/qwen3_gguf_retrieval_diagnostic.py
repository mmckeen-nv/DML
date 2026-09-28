"""Two predeclared, read-only retrieval-planning controls; never M7 acceptance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import time

from daystrom_dml.contracts.agent_episode import (
    canonical_json, QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE as PROFILE,
)
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.qwen3_gguf_action_input import (
    LocalQwen3GGUFActionInputConsumer, sampling_policy_identity,
)
from scripts import qwen3_gguf_synthetic as shared

SCHEMA = 'dml-qwen3-gguf-retrieval-planning-v1'
CASE_IDS = ('paired_records', 'paired_records_with_distractors')


def source_digests():
    result = shared.source_digests(consumer_profile=PROFILE)
    result['scripts/qwen3_gguf_retrieval_diagnostic.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def prepare_suite(path, *, identity, limits):
    if type(limits) is not EpisodeLimits or canonical_json(asdict(limits)) != canonical_json(shared.QUALIFICATION_LIMITS):
        raise ValueError('Retrieval planning uses the original qualification limits')
    cases = []
    for ordinal, case_id in enumerate(CASE_IDS):
        subject = 'Synthetic archive ' + secrets.token_hex(6)
        truth, seeds = {}, []
        for role in ('old', 'current'):
            value = secrets.token_hex(12)
            key = 'archive.' + role + '_access_phrase'
            truth[key] = {'value': value, 'evidence_aliases': [role]}
            seeds.append({'alias': role,
                'text': subject + ' ' + role.upper() + ' entry: access phrase is ' + value + '.',
                'meta': {'source': 'planning-' + role, 'source_trust': 'trusted',
                         'claim_key': key, 'claim_value': value, 'no_merge': True}})
        if ordinal:
            for index in range(3):
                value = secrets.token_hex(12)
                seeds.append({'alias': 'distractor' + str(index),
                    'text': 'Unrelated archive ' + secrets.token_hex(6) + ' current access phrase is ' + value + '.',
                    'meta': {'source': 'planning-distractor', 'source_trust': 'trusted',
                             'claim_key': 'unrelated.access_phrase', 'claim_value': value, 'no_merge': True}})
        prompt = ('For ' + subject + ', report both the OLD entry access phrase and the current entry access phrase. '
                  'Use claim keys archive.old_access_phrase and archive.current_access_phrase respectively, '
                  'with exact stored values and citations to each entry. Read only; do not change memory.')
        cases.append({'id': case_id, 'ordinal': ordinal, 'prompt': prompt,
            'scope': dict(shared.SCOPE), 'seeds': seeds,
            'task': {'id': case_id, 'prompt': prompt, 'truth': truth, 'state_expectations': []},
            'allowed_tools': ['retrieve'], 'dependency_preflight': None,
            'preflight_queries': [subject + ' OLD entry access phrase', subject + ' current entry access phrase', prompt]})
    suite = {'schema_version': SCHEMA, 'classification': 'synthetic-noncorpus-retrieval-planning',
        'created_ns': time.time_ns(), 'consumer_profile': PROFILE, 'model_identity': identity,
        'sampling': sampling_policy_identity(), 'limits': asdict(limits), 'effective_time': 2000000000,
        'source_digests': source_digests(), 'cases': cases,
        'run_policy': {'runs_per_case': 1, 'automatic_retry': False, 'repair': False,
            'forced_tool_choice': False, 'forced_initial_query': False, 'final_contract_unchanged': True,
            'all_turn_json_grammar': True, 'read_only': True,
            'proceed_to_readiness_requires': 'both cases pass',
            'direct_complete_retrieval': 'valid success; no query-reformulation credit'}}
    suite['suite_digest'] = shared._digest(suite)
    shared._write_exclusive(path, suite)
    return suite


def validate_suite(suite):
    body = {key: value for key, value in suite.items() if key != 'suite_digest'}
    if (suite.get('schema_version') != SCHEMA or suite.get('suite_digest') != shared._digest(body)
            or suite['consumer_profile'] != PROFILE
            or [case['id'] for case in suite['cases']] != list(CASE_IDS)
            or canonical_json(suite['limits']) != canonical_json(shared.QUALIFICATION_LIMITS)
            or suite['sampling'] != sampling_policy_identity() or suite['source_digests'] != source_digests()
            or any(case['allowed_tools'] != ['retrieve'] or case['task']['state_expectations']
                   for case in suite['cases'])):
        raise ValueError('Retrieval planning declaration, limits or source changed')
    return EpisodeLimits(**suite['limits'])


def assess_case(case, events):
    fixture = next((e['payload'] for e in events if e['kind'] == 'fixture_ready'), None)
    targets = {fixture['seed_records'][alias]['id'] for alias in ('old', 'current')} if fixture else set()
    requests = {e.get('call_id'): e['payload'] for e in events if e['kind'] == 'tool_requested'}
    observed, prior_query, incomplete = set(), None, False
    reformulated = False
    retrieved = []
    for event in events:
        if event['kind'] != 'tool_completed' or event['payload']['name'] != 'retrieve':
            continue
        retrieved.append(event)
        ids = {r['id'] for r in json.loads(event['payload']['model_result'])['records']}
        query = requests.get(event.get('call_id'), {}).get('arguments', {}).get('query')
        if (incomplete and prior_query is not None and query is not None and query != prior_query
                and bool((ids & targets) - observed)):
            reformulated = True
        observed.update(ids)
        incomplete = not targets.issubset(observed)
        prior_query = query
    outcome = next((e['payload'] for e in reversed(events) if e['kind'] == 'outcome'), {})
    verdict = next((e['payload'] for e in events if e['kind'] == 'verifier_result'), {})
    claims = (outcome.get('answer') or {}).get('claims', [])
    complete = bool(fixture) and len(claims) == len(case['task']['truth']) and all(
        any(c['key'] == key and c['value'] == expected['value']
            and fixture['seed_records'][expected['evidence_aliases'][0]]['id'] in c['evidence_ids']
            and fixture['seed_records'][expected['evidence_aliases'][0]]['id'] in observed
            for c in claims) for key, expected in case['task']['truth'].items())
    completed = [e for e in events if e['kind'] == 'model_completed']
    completed_ids = {e.get('call_id') for e in completed}
    feedback = any(e['kind'] == 'model_requested' and e.get('call_id') in completed_ids and any(
        m['role'] == 'tool' for m in e['payload']['request']['messages']) for e in events)
    rejected = [e for e in events if e['kind'] in ('action_rejected', 'tool_validation_rejected')]
    mutation = any(e['kind'] in ('tool_requested', 'tool_completed') and e['payload']['name'] != 'retrieve'
                   for e in events)
    passed = bool(retrieved and feedback and complete and targets.issubset(observed)
        and outcome.get('status') == 'completed' and verdict.get('success') is True and not rejected and not mutation)
    return {'id': case['id'], 'status': outcome.get('status', 'interrupted'),
        'model_calls_completed': len(completed), 'model_selected_retrieve_calls': len(retrieved),
        'both_distinct_facts_cited': complete, 'original_verifier': verdict,
        'query_reformulation_observed': reformulated, 'rejection_events': len(rejected),
        'synthetic_case_pass': passed}


def run_staged(planning_path, readiness_path, *, snapshot, output):
    """Execute the two frozen stages once; failed or unknown planning never advances."""
    planning = json.loads(Path(planning_path).read_text())
    readiness = json.loads(Path(readiness_path).read_text())
    validate_suite(planning)
    shared.validate_suite(readiness)
    if (readiness['consumer_profile'] != PROFILE or readiness['model_identity'] != planning['model_identity']
            or readiness['limits'] != planning['limits'] or readiness['sampling'] != planning['sampling']):
        raise ValueError('Both stages must bind the same candidate and original limits')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shared._write_exclusive(output / 'stages.json', {
        'planning_suite_digest': planning['suite_digest'], 'readiness_suite_digest': readiness['suite_digest'],
        'runs_per_case': 1, 'retry': False, 'resume': False, 'planning_required_passes': 2})
    first = shared.run_declared_suite(planning_path, snapshot=snapshot, output=output / 'planning',
        suite_validator=validate_suite, case_assessor=assess_case)
    admitted = (first['all_synthetic_cases_pass'] is True and len(first['cases']) == 2
        and all(c['synthetic_case_pass'] is True for c in first['cases'])
        and not first['unrun_cases'] and first['unrun_reason'] is None)
    decision = {'planning_passed': admitted, 'readiness_started': admitted,
        'reason': 'planning_2_of_2' if admitted else 'planning_failed_or_incomplete',
        'planning_summary_sha256': hashlib.sha256((output / 'planning' / 'summary.json').read_bytes()).hexdigest(),
        'automatic_retry': False, 'evaluation_campaign_run': False}
    shared._write_exclusive(output / 'stage-decision.json', decision)
    second = shared.run_suite(readiness_path, snapshot=snapshot, output=output / 'readiness') if admitted else None
    result = {**decision, 'readiness_passed': bool(second and second['all_synthetic_cases_pass']),
        'readiness_unrun_cases': second['unrun_cases'] if second else list(shared.CASE_IDS),
        'readiness_unrun_reason': second['unrun_reason'] if second else 'planning_failed_or_incomplete'}
    shared._write_exclusive(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--snapshot', required=True)
    prepare.add_argument('--limits-file', required=True)
    prepare.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--snapshot', required=True)
    run.add_argument('--suite', required=True)
    run.add_argument('--output', required=True)
    staged = sub.add_parser('staged-run')
    staged.add_argument('--snapshot', required=True)
    staged.add_argument('--planning-suite', required=True)
    staged.add_argument('--readiness-suite', required=True)
    staged.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        with LocalQwen3GGUFActionInputConsumer(args.snapshot, consumer_profile=PROFILE) as consumer:
            suite = prepare_suite(args.output, identity=consumer._identity.to_payload(),
                limits=EpisodeLimits(**json.loads(Path(args.limits_file).read_text())))
        print(json.dumps({'suite_digest': suite['suite_digest'], 'cases': list(CASE_IDS)}))
    elif args.command == 'run':
        print(json.dumps(shared.run_declared_suite(args.suite, snapshot=args.snapshot, output=args.output,
            suite_validator=validate_suite, case_assessor=assess_case)))
    else:
        print(json.dumps(run_staged(args.planning_suite, args.readiness_suite,
            snapshot=args.snapshot, output=args.output)))


if __name__ == '__main__':
    main()
