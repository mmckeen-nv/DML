"""Manifest-bound Llama SFT qualification using unchanged planning/readiness controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from daystrom_dml.contracts.agent_episode import LLAMA3_SFT_V3_CONSUMER_PROFILE as PROFILE, canonical_json
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.llama3_sft_v3_action_input import LocalLlama3SFTV3ActionInputConsumer, sampling_policy_identity
from scripts import qwen3_gguf_synthetic as shared
from scripts import qwen3_gguf_retrieval_diagnostic as planning
from scripts.agent_episodes import _source_digests

SCHEMA = 'dml-llama3-sft-v3-qualification-v1'
STAGES = {'planning': planning, 'readiness': shared}


def source_digests():
    result = _source_digests(consumer_profile=PROFILE)
    for module in (shared, planning):
        result['scripts/' + Path(module.__file__).name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    result['scripts/llama3_sft_v3_synthetic.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def prepare_suite(path, *, stage, identity):
    if stage not in STAGES:
        raise ValueError('Unknown qualification stage')
    suite = dict(schema_version=SCHEMA, stage=stage, classification='synthetic-development-qualification',
                 created_ns=time.time_ns(), consumer_profile=PROFILE, model_identity=identity,
                 sampling=sampling_policy_identity(), limits=dict(shared.QUALIFICATION_LIMITS),
                 effective_time=2000000000, source_digests=source_digests(),
                 cases=STAGES[stage].build_cases(),
                 run_policy={'runs_per_case': 1, 'automatic_retry': False, 'repair': False,
                             'forced_tool_choice': False, 'all_turn_json_grammar': True,
                             'final_contract_unchanged': True})
    suite['suite_digest'] = shared._digest(suite)
    validate_suite(suite)
    shared._write_exclusive(path, suite)
    return suite


def validate_suite(suite):
    stage = suite.get('stage')
    body = {key: value for key, value in suite.items() if key != 'suite_digest'}
    if (stage not in STAGES or suite.get('schema_version') != SCHEMA
            or suite.get('suite_digest') != shared._digest(body)
            or suite.get('consumer_profile') != PROFILE
            or suite.get('limits') != shared.QUALIFICATION_LIMITS
            or suite.get('sampling') != sampling_policy_identity()
            or suite.get('source_digests') != source_digests()
            or [case['id'] for case in suite['cases']] != list(STAGES[stage].CASE_IDS)
            or suite.get('run_policy') != {'runs_per_case': 1, 'automatic_retry': False, 'repair': False,
                'forced_tool_choice': False, 'all_turn_json_grammar': True, 'final_contract_unchanged': True}):
        raise ValueError('Qualification declaration changed')
    if stage == 'planning' and any(case['allowed_tools'] != ['retrieve'] or case['task']['state_expectations']
                                   for case in suite['cases']):
        raise ValueError('Planning controls must remain read-only')
    return EpisodeLimits(**suite['limits'])


def run_suite(suite_path, *, snapshot, output):
    suite = json.loads(Path(suite_path).read_bytes())
    validate_suite(suite)
    return shared.run_declared_suite(suite_path, snapshot=snapshot, output=output,
                                    suite_validator=validate_suite, case_assessor=STAGES[suite['stage']].assess_case)


def run_staged(planning_suite, readiness_suite, *, snapshot, output):
    """Both declarations exist before the first generation; no retry or selective continuation."""
    declarations = [json.loads(Path(path).read_bytes()) for path in (planning_suite, readiness_suite)]
    for expected, suite in zip(('planning', 'readiness'), declarations):
        validate_suite(suite)
        if suite['stage'] != expected:
            raise ValueError('Qualification stage order differs')
    if canonical_json(declarations[0]['model_identity']) != canonical_json(declarations[1]['model_identity']):
        raise ValueError('Qualification candidates differ')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    first = run_suite(planning_suite, snapshot=snapshot, output=root / 'planning')
    second = None
    if first['all_synthetic_cases_pass']:
        second = run_suite(readiness_suite, snapshot=snapshot, output=root / 'readiness')
    result = {'schema_version': SCHEMA, 'consumer_profile': PROFILE,
              'model_identity': declarations[0]['model_identity'],
              'planning': first, 'readiness': second,
              'readiness_unrun_reason': None if second else 'planning gate failed',
              'qualified': bool(second and second['all_synthetic_cases_pass']),
              'production_qualified': False}
    shared._write_exclusive(root / 'summary.json', result)
    return result


def replay_suite(suite_path, *, snapshot, output):
    """Data-only replay of complete qualification attempts; quality failure is retained."""
    from daystrom_dml.contracts.agent_episode import validate_episode_events
    from daystrom_dml.services.episode_verifiers import verify_task
    from daystrom_dml.services.agent_episode import _read_records
    from scripts.agent_campaign_evidence import _replay_llama3_sft_model, _replay_presented_authority, _observations
    suite_path, output = Path(suite_path), Path(output)
    suite = json.loads(suite_path.read_bytes())
    validate_suite(suite)
    summary = json.loads((output / 'summary.json').read_bytes())
    if summary['suite_digest'] != suite['suite_digest']:
        raise ValueError('Result belongs to another declaration')
    count = len(summary['cases'])
    if count > len(suite['cases']):
        raise ValueError('Extra undeclared qualification rows')
    if summary['unrun_cases'] != [case['id'] for case in suite['cases'][count:]]:
        raise ValueError('Attempted/unrun selection differs')
    rows = []
    with LocalLlama3SFTV3ActionInputConsumer(snapshot, offline=True) as consumer:
        if consumer.identity.to_payload() != suite['model_identity']:
            raise ValueError('Replay candidate differs')
        for case, retained in zip(suite['cases'], summary['cases']):
            if retained.get('id') != case['id']:
                raise ValueError('Qualification case identity differs')
            events = [json.loads(line) for line in (output / case['id'] / 'events.jsonl').read_bytes().splitlines()]
            previous = '0' * 64
            for index, event in enumerate(events):
                body = {key:value for key,value in event.items() if key != 'event_digest'}
                if event['sequence'] != index or event['previous_digest'] != previous or event['event_digest'] != shared._digest(body):
                    raise ValueError('Qualification event hash chain differs')
                previous = event['event_digest']
            if previous != retained['evidence_digest']:
                raise ValueError('Summary evidence digest differs')
            model_events = []
            for event in events:
                if 'episode_sequence' in event:
                    model_events.append({**{key:value for key,value in event.items()
                        if key not in ('episode_sequence','received_ns','previous_digest','event_digest')},
                        'sequence':event['episode_sequence']})
            validate_episode_events(model_events)
            start = model_events[0]
            if start['task_id'] != case['task']['id'] or start['episode_id'] != 'synthetic-' + case['id']:
                raise ValueError('Qualification episode identity differs')
            for key, value in {'prompt':case['task']['prompt'],'scope':case['scope'],
                    'limits':suite['limits'],'effective_time':suite['effective_time'],
                    'allowed_tools':case['allowed_tools'],'consumer_profile':PROFILE,
                    'execution_path':'live_local','prior_context':None}.items():
                if start['payload'].get(key) != value:
                    raise ValueError('Qualification start differs: ' + key)
            fixture = next(event['payload'] for event in events if event['kind'] == 'fixture_ready')
            _replay_presented_authority(model_events, fixture)
            generated = _replay_llama3_sft_model(model_events, suite['model_identity'], consumer)
            records = next(event['payload']['records'] for event in events if event['kind'] == 'authority_snapshot')
            if canonical_json(records) != canonical_json(_read_records(output / case['id'] / 'authority')):
                raise ValueError('Final authority snapshot differs from retained database')
            outcome = next(event['payload'] for event in events if event['kind'] == 'outcome')
            verifier = next(event['payload'] for event in events if event['kind'] == 'verifier_result')
            scenario = {'scope':case['scope'],'seeds':case['seeds'],'setup':[],
                        'expected_memory_count':len(case['seeds'])}
            verdict = verify_task(scenario, case['task'], outcome.get('answer'),
                seed_records=fixture['seed_records'], observed_records=_observations(model_events),
                current_records=records, final_sequence=len(model_events)-1, effective_time=suite['effective_time'])
            if canonical_json(verdict) != canonical_json(verifier):
                raise ValueError('Original verifier outcome differs')
            from daystrom_dml.services.episode_outcomes import build_terminal
            from daystrom_dml.services.agent_episode import _failure_verdict
            terminal = model_events[-1]['payload']
            if terminal['status'] != outcome['status'] or terminal['answer'] != outcome.get('answer'):
                raise ValueError('Terminal and retained outcome differ')
            expected_verifier = verdict if outcome['status'] == 'completed' else _failure_verdict(outcome['status'])
            if terminal['verifier'] != expected_verifier:
                raise ValueError('Terminal verifier differs')
            rebuilt = build_terminal(model_events[:-1],expected_verifier,status=terminal['status'],
                latency_ms=terminal['latency_ms'],retrieval_ms=terminal['retrieval_ms'],
                answer=terminal['answer'],usage_unknown=terminal['usage_unknown'])
            if rebuilt != terminal:
                raise ValueError('Terminal accounting differs')
            import sqlite3
            from daystrom_dml.journal import JournalStateStore, _checked
            db_path = output / case['id'] / 'authority' / 'dml_state.sqlite3'
            with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.execute('BEGIN')
                journal = object.__new__(JournalStateStore)
                journal._schema_version = 2
                journal._identity = {'schema_version':1,'store_id':db.execute('SELECT store_id FROM identity').fetchone()[0]}
                revision,state,_ = journal._read_snapshot(db)
                receipts = [_checked(raw,checksum) for raw,checksum in
                            db.execute('SELECT payload,checksum FROM receipts ORDER BY revision')]
            acknowledged = list(fixture['seed_receipts']) + [event['payload']['result']['receipt']
                for event in model_events if event['kind'] == 'tool_completed' and event['payload']['name'] != 'retrieve']
            if not terminal['effects_unknown']:
                ordered = sorted(acknowledged,key=lambda receipt:receipt['revision'])
                final = {receipt['result']['memory']['id']:receipt['result']['memory'] for receipt in ordered}
                if (receipts != ordered or revision != max(receipt['revision'] for receipt in ordered)
                        or final != {record['id']:record for record in state['items']}):
                    raise ValueError('Recorded receipts differ from retained authority journal')
            if canonical_json(state['items']) != canonical_json(records):
                raise ValueError('Journal state differs from recorded final authority')
            assessed = STAGES[suite['stage']].assess_case(case,events)
            if any(retained.get(key) != value for key,value in assessed.items()):
                raise ValueError('Original qualification scorer differs')
            rows.append({'id':case['id'],'generated':generated,'quality_passed':assessed['synthetic_case_pass']})
    qualified = len(rows) == len(suite['cases']) and all(row['quality_passed'] for row in rows)
    if summary['all_synthetic_cases_pass'] != qualified:
        raise ValueError('Qualification summary gate differs')
    return {'passed':True,'consumer_profile':PROFILE,'model_identity':suite['model_identity'],
            'suite_sha256':hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            'result_sha256':hashlib.sha256((output/'summary.json').read_bytes()).hexdigest(),
            'cases':rows,'qualification_passed':qualified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run-staged', 'replay'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--planning-suite', type=Path, required=True)
    parser.add_argument('--readiness-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        with LocalLlama3SFTV3ActionInputConsumer(args.snapshot, offline=True) as consumer:
            identity = consumer.identity.to_payload()
        prepare_suite(args.planning_suite, stage='planning', identity=identity)
        prepare_suite(args.readiness_suite, stage='readiness', identity=identity)
        return 0
    if args.command == 'replay':
        if args.output is None:
            parser.error('--output is required for replay')
        for path, stage in ((args.planning_suite,'planning'),(args.readiness_suite,'readiness')):
            directory = args.output / stage
            if directory.exists():
                shared._write_exclusive(directory/'primary-replay.json', replay_suite(path,snapshot=args.snapshot,output=directory))
        return 0
    if args.output is None:
        parser.error('--output is required for run-staged')
    return 0 if run_staged(args.planning_suite, args.readiness_suite,
                          snapshot=args.snapshot, output=args.output)['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
