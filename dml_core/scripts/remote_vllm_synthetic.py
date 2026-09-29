"""Predeclared noncorpus diagnostics using DML's unchanged action/tool loop.

This is not the nine-task evaluation or a replacement acceptance gate. Prepare
once before generation; run each case once, retaining early finals and failures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import secrets
import time
import traceback

from daystrom_dml.contracts.agent_episode import canonical_json, EXECUTION_PROTOCOL_V2
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.remote_vllm_action_input import (
    CONSUMER_PROFILE, RemoteVLLMActionInputConsumer, sampling_policy_identity,
)

SCHEMA = 'dml-remote-synthetic-diagnostic-v1'
CASE_IDS = ('hidden_lookup', 'stale_user_correction', 'stored_correction')
SCOPE = {'tenant_id': 'synthetic-diagnostic', 'client_id': 'noncorpus',
         'session_id': 'synthetic-session', 'instance_id': 'agent'}


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def source_digests():
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/remote_vllm_synthetic.py',
             'daystrom_dml/services/remote_vllm_action_input.py',
             'daystrom_dml/services/agent_episode.py',
             'daystrom_dml/services/episode_tools.py',
             'daystrom_dml/services/agent_action_grammar.py',
             'daystrom_dml/contracts/agent_episode.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _write_exclusive(path, value):
    path = Path(path)
    with path.open('xb') as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_suite(path, *, identity, limits, consumer_profile=CONSUMER_PROFILE):
    """Create secrets before any model call, with none in the answer-bearing prompt."""
    if type(limits) is not EpisodeLimits:
        raise ValueError('Use the explicit unchanged episode limits')
    cases = []
    for index, case_id in enumerate(CASE_IDS):
        subject = 'Synthetic cabinet ' + secrets.token_hex(6)
        current, stale = secrets.token_hex(12), secrets.token_hex(12)
        key = 'cabinet.access_phrase'
        seeds = [{'alias': 'current',
                  'text': subject + ' current access phrase is ' + current + '. This is the current verified entry.',
                  'meta': {'source': 'synthetic-current', 'source_trust': 'trusted',
                           'claim_key': key, 'claim_value': current, 'no_merge': True}}]
        prompt = ('What is the current access phrase for ' + subject + '? '
                  'Return claim key ' + key + ' and cite the stored evidence supporting the answer.')
        if case_id == 'stale_user_correction':
            prompt += ' An unverified older note reported ' + stale + '; it may be wrong.'
        if case_id == 'stored_correction':
            seeds.insert(0, {'alias': 'stale',
                'text': subject + ' old access phrase was ' + stale + '. This entry was withdrawn; use the corrected current entry.',
                'meta': {'source': 'synthetic-withdrawn', 'source_trust': 'trusted',
                         'claim_key': key, 'claim_value': stale, 'no_merge': True}})
            seeds[-1]['text'] += ' It corrects the withdrawn old entry.'
        cases.append({'id': case_id, 'prompt': prompt, 'scope': dict(SCOPE), 'seeds': seeds,
                      'expected': {'key': key, 'value': current, 'evidence_alias': 'current'},
                      'stale_value': stale, 'ordinal': index})
    suite = {'schema_version': SCHEMA, 'classification': 'synthetic-noncorpus-not-acceptance',
             'created_ns': time.time_ns(), 'consumer_profile': consumer_profile,
             'model_identity': identity, 'sampling': sampling_policy_identity(),
             'limits': asdict(limits), 'effective_time': 2000000000,
             'source_digests': source_digests(), 'cases': cases,
             'run_policy': {'runs_per_case': 1, 'automatic_retry': False, 'repair': False,
                            'forced_tool_choice': False, 'final_branch_available': True,
                            'allowed_tools': ['retrieve'], 'case_order': list(CASE_IDS)}}
    suite['suite_digest'] = _digest(suite)
    _write_exclusive(path, suite)
    return suite


def validate_suite(suite):
    body = {key: value for key, value in suite.items() if key != 'suite_digest'}
    if suite.get('schema_version') != SCHEMA or suite.get('suite_digest') != _digest(body):
        raise ValueError('Synthetic suite digest/version mismatch')
    if ([case['id'] for case in suite['cases']] != list(CASE_IDS)
            or suite['sampling'] != sampling_policy_identity()
            or suite['source_digests'] != source_digests()):
        raise ValueError('Synthetic suite policy/source changed')
    return EpisodeLimits(**suite['limits'])


def _worker(channel, *, snapshot, directory, suite, case, acknowledgement=None):
    from daystrom_dml.services.agent_episode import _prepare_fixture, _run_loop
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
    adapter = None
    def publish(event):
        channel.put(event)
        if acknowledgement is not None:
            acknowledgement.get(timeout=suite['limits']['wall_time_seconds'])
    try:
        episode_id = 'synthetic-' + case['id']
        scenario = {'scope': case['scope'], 'seeds': case['seeds'], 'setup': []}
        adapter, fixture = _prepare_fixture(Path(directory) / 'authority', scenario, episode_id)
        publish({'kind': 'fixture_ready', 'payload': fixture})
        toolbox = SelectedProfileEpisodeTools(adapter, scope=case['scope'], episode_id=episode_id,
            seed_receipts=fixture['seed_receipts'], allowed_tools=('retrieve',),
            effective_time=suite['effective_time'], execution_protocol=EXECUTION_PROTOCOL_V2)
        def emit(kind, call_id, payload):
            publish({'kind': kind, 'call_id': call_id, 'payload': payload})
        with RemoteVLLMActionInputConsumer(snapshot, consumer_profile=suite['consumer_profile']) as consumer:
            if consumer.identity.to_payload() != suite['model_identity']:
                raise ValueError('Synthetic candidate identity changed')
            outcome = _run_loop(consumer, toolbox, task={'id': case['id'], 'prompt': case['prompt']},
                limits=EpisodeLimits(**suite['limits']), emit=emit,
                execution_protocol=EXECUTION_PROTOCOL_V2, consumer_profile=suite['consumer_profile'])
        publish({'kind': 'outcome', 'payload': outcome})
    except BaseException:
        publish({'kind': 'diagnostic_error', 'payload': {'traceback': traceback.format_exc()}})
    finally:
        if adapter is not None:
            adapter.close()
        publish({'kind': 'worker_finished', 'payload': {}})


def assess_case(case, events):
    """Quality observations; early final is retained, not converted into tool use."""
    completed = [event for event in events if event['kind'] == 'model_completed']
    retrieved = [event for event in events if event['kind'] == 'tool_completed'
                 and event['payload']['name'] == 'retrieve']
    requested = [event for event in events if event['kind'] == 'model_requested']
    completed_ids = {event.get('call_id') for event in completed}
    later_feedback = [event for event in requested if event.get('call_id') in completed_ids and any(
        message['role'] == 'tool' for message in event['payload']['request']['messages'])]
    outcomes = [event['payload'] for event in events if event['kind'] == 'outcome']
    outcome = outcomes[-1] if outcomes else {'status': 'interrupted', 'answer': None}
    expected = case['expected']
    fixture = next((event['payload'] for event in events if event['kind'] == 'fixture_ready'), None)
    expected_id = fixture['seed_records'][expected['evidence_alias']]['id'] if fixture else None
    observed_ids = set()
    for event in retrieved:
        model_result = json.loads(event['payload']['model_result'])
        observed_ids.update(record['id'] for record in model_result.get('records', []))
    claims = (outcome.get('answer') or {}).get('claims', [])
    target_claims = [claim for claim in claims if claim['key'] == expected['key']]
    correct = bool(target_claims) and all(claim['value'] == expected['value'] for claim in target_claims) and any(claim['key'] == expected['key'] and claim['value'] == expected['value']
                  and expected_id in claim['evidence_ids'] and expected_id in observed_ids
                  for claim in claims)
    return {'id': case['id'], 'status': outcome['status'], 'model_calls_completed': len(completed),
            'model_selected_retrieve_calls': len(retrieved), 'calls_after_real_tool_feedback': len(later_feedback),
            'correct_cited_current_value': correct,
            'synthetic_case_pass': bool(retrieved and later_feedback and correct and outcome['status'] == 'completed')}


def run_suite(suite_path, *, snapshot, output):
    suite = json.loads(Path(suite_path).read_text())
    limits = validate_suite(suite)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite or resume a selective subset.
    _write_exclusive(Path(suite_path).with_suffix('.started.json'), {
        'suite_digest': suite['suite_digest'], 'output': str(output.resolve()), 'started_ns': time.time_ns()})
    _write_exclusive(output / 'suite.json', suite)
    results = []
    unrun_reason = None
    context = multiprocessing.get_context('spawn')
    for case in suite['cases']:
        directory = output / case['id']
        directory.mkdir()
        channel = context.Queue()
        acknowledgement = context.Queue()
        process = context.Process(target=_worker, kwargs={'channel': channel, 'snapshot': str(snapshot),
            'directory': str(directory), 'suite': suite, 'case': case, 'acknowledgement': acknowledgement})
        events, previous, evidence_bytes = [], '0' * 64, 0
        def retain(event):
            nonlocal previous, evidence_bytes
            event = {**event, 'sequence': len(events), 'received_ns': time.time_ns(), 'previous_digest': previous}
            next_digest = _digest(event)
            event['event_digest'] = next_digest
            raw = canonical_json(event, limit=limits.max_event_bytes) + b'\n'
            if evidence_bytes + len(raw) > limits.max_episode_bytes:
                raise ValueError('Synthetic evidence exceeds unchanged episode budget')
            evidence_bytes += len(raw)
            previous = next_digest
            events.append(event)
            with (directory / 'events.jsonl').open('ab') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        retain({'kind': 'case_started', 'payload': {'case_id': case['id'], 'suite_digest': suite['suite_digest']}})
        started = time.monotonic()
        process.start()
        finished = False
        while time.monotonic() - started < limits.wall_time_seconds:
            try:
                event = channel.get(timeout=min(0.2, max(0.01, limits.wall_time_seconds - (time.monotonic() - started))))
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            try:
                retain(event)
            except ValueError:
                unrun_reason = 'evidence_budget_rejected'
                break
            acknowledgement.put(True)
            if event['kind'] == 'worker_finished':
                finished = True
                break
        if not finished:
            unrun_reason = unrun_reason or ('deadline' if process.is_alive() else 'worker_exit')
            # Final supervisor accounting is saved separately if raw evidence budget is exhausted.
            if unrun_reason != 'evidence_budget_rejected':
                retain({'kind': 'supervisor_interrupted', 'payload': {
                'cause': 'deadline' if process.is_alive() else 'worker_exit',
                'remote_cancellation_confirmed': False, 'automatic_retry': False}})
        if finished:
            process.join(timeout=2)
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        # Every worker event required a durable ACK; no unacknowledged tail is used as accepted evidence.
        summary = assess_case(case, events)
        summary.update(exitcode=process.exitcode, evidence_digest=previous)
        _write_exclusive(directory / 'summary.json', summary)
        results.append(summary)
        channel.close()
        acknowledgement.close()
        unknown_dispatch = any(event['kind'] == 'model_failed'
            and event['payload'].get('phase') == 'execute'
            and (event['payload'].get('input_token_count') is None
                 or event['payload'].get('output_token_count') is None) for event in events)
        # A timed-out request may still execute remotely: stop, retain unrun cases.
        if not finished or unknown_dispatch:
            unrun_reason = unrun_reason or 'remote_execution_failed_usage_or_cancellation_unknown'
            break
    summary = {'schema_version': SCHEMA, 'classification': suite['classification'],
               'suite_digest': suite['suite_digest'], 'cases': results,
               'unrun_cases': [case['id'] for case in suite['cases'][len(results):]],
               'unrun_reason': unrun_reason,
               'all_synthetic_cases_pass': len(results) == len(CASE_IDS) and all(r['synthetic_case_pass'] for r in results),
               'evaluation_campaign_run': False}
    _write_exclusive(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--snapshot', required=True)
    prepare.add_argument('--consumer-profile', required=True)
    prepare.add_argument('--limits-file', required=True, help='JSON containing exact EpisodeLimits field values')
    prepare.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--suite', required=True)
    run.add_argument('--snapshot', required=True)
    run.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        with RemoteVLLMActionInputConsumer(args.snapshot, consumer_profile=args.consumer_profile, offline=True) as consumer:
            suite = prepare_suite(args.output, identity=consumer.identity.to_payload(),
                                  limits=EpisodeLimits(**json.loads(Path(args.limits_file).read_text())),
                                  consumer_profile=args.consumer_profile)
        print(json.dumps({'suite_digest': suite['suite_digest'], 'cases': list(CASE_IDS)}))
    else:
        print(json.dumps(run_suite(args.suite, snapshot=args.snapshot, output=args.output)))


if __name__ == '__main__':
    main()
