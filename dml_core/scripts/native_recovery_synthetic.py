"""Once-only noncorpus lifecycle recovery qualification with authentic peer writes.

External peer work is retained separately and is never model-owned authority.
This module does not change the frozen nine-task acceptance corpus or gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback

from daystrom_dml.contracts.agent_episode import canonical_json, EXECUTION_PROTOCOL_V2, NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.native_remote_vllm_action_input import NativeRemoteVLLMActionInputConsumer
from scripts.native_dml_synthetic import _digest, _write_exclusive
from scripts.native_recovery_suite import CASE_IDS, BASELINE_POLICY, prepare_suite, validate_suite


def verifier_baseline(case, fixture, peer):
    """Explicit predeclared peer baseline; initial fixture remains immutable."""
    from daystrom_dml.journal import _validated_receipt
    from daystrom_dml.services.receipt_lifecycle import memory_digest
    from daystrom_dml.services.receipt_update import canonical_update_request, _update_receipt
    declaration = case['peer_operation']
    original = fixture['seed_records'][declaration['target_alias']]
    receipt = peer['result']['receipt']
    _validated_receipt(receipt)
    request, digest = canonical_update_request(original['id'], text=declaration['text'],
        reason=declaration['reason'], expected_memory_digest=memory_digest(original), **case['scope'])
    if (receipt['key'] != declaration['idempotency_key'] or receipt['request_digest'] != digest
            or receipt['scope'] != case['scope']
            or receipt['store_id'] not in {value['store_id'] for value in fixture['seed_receipts']}):
        raise ValueError('Peer baseline receipt differs from declared original update')
    _update_receipt(receipt, request)
    updated = receipt['result']['memory']
    restored = deepcopy(updated)
    restored['text'], restored['embedding'] = original['text'], deepcopy(original['embedding'])
    restored['meta'].pop('content_update_decision', None)
    if 'content_update_decision' in original['meta']:
        restored['meta']['content_update_decision'] = deepcopy(original['meta']['content_update_decision'])
    if canonical_json(restored) != canonical_json(original):
        raise ValueError('Peer changed undeclared identity, lifecycle, claim or provenance')
    records = deepcopy(fixture['seed_records'])
    records[declaration['target_alias']] = deepcopy(updated)
    return {'policy': BASELINE_POLICY, 'initial_seed_records_digest': _digest(fixture['seed_records']),
        'peer_receipt_digest': _digest(receipt), 'seed_records': records,
        'setup': [{'operation': 'related_writes', 'updates': [{'alias': declaration['target_alias'],
            'text': declaration['text'], 'reason': declaration['reason']}]}]}


def perform_peer_update(directory, case, fixture, toolbox, episode_id, effective_time):
    """Real separately owned gateway dispatch; never rewrite a model proposal."""
    from daystrom_dml.services.agent_episode import _fixture_adapter
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
    from daystrom_dml.journal import _validated_receipt
    declaration = case['peer_operation']
    peer = _fixture_adapter(directory)
    try:
        bridge = SelectedProfileEpisodeTools(peer, scope=case['scope'],
            episode_id=declaration['episode_id'], seed_receipts=fixture['seed_receipts'],
            allowed_tools=('retrieve', 'update'), effective_time=effective_time,
            execution_protocol=EXECUTION_PROTOCOL_V2)
        target = fixture['seed_records'][declaration['target_alias']]
        read = bridge.prepare('retrieve', {'query': target['text'], 'top_k': 10}, call_id='peer-observe')
        read_raw, read_visible = bridge.execute(read)
        matches = [record for record in json.loads(read_visible)['records']
                   if record['id'] == target['id'] and record.get('record_ref') is not None]
        if len(matches) != 1:
            raise ValueError('Predeclared peer target was not independently retrieved')
        update = bridge.prepare('update', {'record_ref': matches[0]['record_ref'],
            'text': declaration['text'], 'reason': declaration['reason']}, call_id=declaration['call_id'])
        if update.idempotency_key != declaration['idempotency_key']:
            raise ValueError('Peer dispatch differs from predeclared idempotency key')
        raw, visible = bridge.execute(update)
        receipt = raw['receipt']
        _validated_receipt(receipt)
        member = peer._journal.lookup_receipt(receipt['scope'], receipt['key'], receipt['request_digest'])
        stores = {value['store_id'] for value in fixture['seed_receipts']}
        if (canonical_json(member) != canonical_json(receipt) or receipt['scope'] != case['scope']
                or receipt['store_id'] not in stores or receipt['result']['memory']['id'] != target['id']):
            raise ValueError('External peer receipt lacks prepared scoped journal authority')
        _, current = peer._journal.read_snapshot()
        if not any(canonical_json(record) == canonical_json(receipt['result']['memory']) for record in current['items']):
            raise ValueError('External receipt is not the current committed record')
        verifier_baseline(case, fixture, {"result": raw})
        reference = toolbox._register_receipt(receipt)
        if reference is None:
            raise ValueError('External peer receipt could not be registered in scope')
        return {'actor': 'predeclared-external-peer-not-model', 'model_owned': False,
            'trigger': 'after-first-model-retrieval-exposing-required-original-references',
            'model_episode_id': episode_id, 'peer_episode_id': declaration['episode_id'],
            'observation_request': read.request_payload(), 'observation_result': read_raw,
            'observation_model_result': read_visible, 'request': update.request_payload(),
            'result': raw, 'model_result': visible, 'registered_reference': reference}
    finally:
        peer.close()


def peer_trigger(case, fixture, payload):
    if payload['name'] != 'retrieve':
        return False
    exposed = {record['id'] for record in json.loads(payload['model_result']).get('records', [])
               if record.get('record_ref') is not None}
    needed = {fixture['seed_records'][alias]['id'] for alias in case['peer_required_aliases']}
    return needed <= exposed


def _worker(channel, *, snapshot, directory, suite, case, acknowledgement=None):
    from daystrom_dml.services.agent_episode import _prepare_fixture, _run_loop, _observed, _read_records, _started, _finish
    from daystrom_dml.services.episode_verifiers import verify_task
    from daystrom_dml.contracts.agent_episode import make_event, validate_episode_events
    from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
    adapter = None

    def publish(event):
        channel.put(event)
        if acknowledgement is not None:
            acknowledgement.get(timeout=suite['limits']['wall_time_seconds'])
    try:
        episode_id = 'recovery-' + case['id']
        scenario = {'scope': case['scope'], 'seeds': case['seeds'], 'setup': [],
                    'expected_memory_count': len(case['seeds'])}
        authority = Path(directory) / 'authority'
        adapter, fixture = _prepare_fixture(authority, scenario, episode_id)
        publish({'kind': 'fixture_ready', 'payload': fixture})
        toolbox = SelectedProfileEpisodeTools(adapter, scope=case['scope'], episode_id=episode_id,
            seed_receipts=fixture['seed_receipts'], allowed_tools=tuple(case['allowed_tools']),
            effective_time=suite['effective_time'], execution_protocol=EXECUTION_PROTOCOL_V2,
            recover_precommit_conflicts=True)
        observed_events = [_started(episode_id, case['task'], case['scope'],
            EpisodeLimits(**suite['limits']), 'live_remote',
            seed_receipts_digest=_digest(fixture['seed_receipts']), allowed_tools=case['allowed_tools'],
            effective_time=suite['effective_time'], consumer_profile=suite['consumer_profile'])]
        publish(observed_events[0])
        started = time.monotonic()
        peer_done = False
        baseline = None

        def emit(kind, call_id, payload):
            nonlocal peer_done, baseline
            event = make_event(episode_id=episode_id, task_id=case['id'], sequence=len(observed_events),
                kind=kind, call_id=call_id, payload=payload, execution_protocol=EXECUTION_PROTOCOL_V2)
            validate_episode_events([*observed_events, event], require_terminal=False)
            observed_events.append(event)
            publish(event)
            if not peer_done and kind == 'tool_completed' and peer_trigger(case, fixture, payload):
                peer_done = True  # Any peer failure is fatal; never repeat a possible write.
                publish({'kind': 'peer_dispatch_started', 'payload': deepcopy(case['peer_operation'])})
                peer = perform_peer_update(authority, case, fixture, toolbox, episode_id, suite['effective_time'])
                publish({'kind': 'external_peer_completed', 'payload': peer})
                baseline = verifier_baseline(case, fixture, peer)
                publish({'kind': 'verifier_baseline', 'payload': baseline})
        with NativeRemoteVLLMActionInputConsumer(snapshot, consumer_profile=suite['consumer_profile']) as consumer:
            if consumer.identity.to_payload() != suite['model_identity']:
                raise ValueError('Synthetic candidate identity changed')
            outcome = _run_loop(consumer, toolbox, task=case['task'],
                limits=EpisodeLimits(**suite['limits']), emit=emit,
                execution_protocol=EXECUTION_PROTOCOL_V2, consumer_profile=suite['consumer_profile'])
        records = _read_records(authority)
        publish({'kind': 'authority_snapshot', 'payload': {'records': records}})
        verifier_scenario = {**scenario, 'setup': baseline['setup']} if baseline is not None else scenario
        verifier_seeds = baseline['seed_records'] if baseline is not None else fixture['seed_records']
        verdict = verify_task(verifier_scenario, case['task'], outcome.get('answer'),
            seed_records=verifier_seeds, observed_records=_observed(observed_events),
            current_records=records, final_sequence=len(observed_events), effective_time=suite['effective_time'])
        usage_unknown = any(event['kind'] == 'model_failed' and event['payload']['phase'] == 'execute'
                            for event in observed_events)
        if usage_unknown:
            outcome = {**outcome, 'retrieval_ms': None}
        _finish(observed_events, outcome, scenario=verifier_scenario, task=case['task'],
            seed_records=verifier_seeds, current_records=records,
            effective_time=suite['effective_time'], previous_answers={},
            elapsed_ms=(time.monotonic() - started) * 1000, usage_unknown=usage_unknown)
        publish(observed_events[-1])
        publish({'kind': 'verifier_result', 'payload': verdict})
        publish({'kind': 'outcome', 'payload': outcome})
    except BaseException:
        publish({'kind': 'diagnostic_error', 'payload': {'traceback': traceback.format_exc()}})
    finally:
        if adapter is not None:
            adapter.close()
        publish({'kind': 'worker_finished', 'payload': {}})


def validate_recovery_authority(case, events):
    """Offline causal authority replay including separately labeled external work."""
    from daystrom_dml.contracts.agent_episode import (
        decode_json, presented_record_identities, validate_precommit_conflict_proof,
    )
    from daystrom_dml.journal import _validated_receipt
    from daystrom_dml.services.receipt_lifecycle import memory_digest
    from daystrom_dml.services.receipt_supersession import canonical_supersession_request, _supersession_receipt
    fixture = next(event['payload'] for event in events if event['kind'] == 'fixture_ready')
    scope = case['scope']
    references, owned, ledger = {}, set(), {}
    for receipt in fixture['seed_receipts']:
        _validated_receipt(receipt)
        if receipt['scope'] != scope:
            raise ValueError('Seed receipt scope differs')
        raw = canonical_json(receipt['result']['memory'])
        owned.add(raw)
        references.setdefault(raw, 'r' + str(len(references)))
    pending, last_retrieval, peer_count, baseline = None, None, 0, None
    for event in events:
        payload = event['payload']
        if event['kind'] == 'external_peer_completed':
            peer_count += 1
            if peer_count != 1 or last_retrieval is None or not peer_trigger(case, fixture, last_retrieval):
                raise ValueError('Peer operation lacks its original-reference retrieval trigger')
            if payload['model_owned'] is not False or payload['actor'] != 'predeclared-external-peer-not-model':
                raise ValueError('External work was relabeled as model-owned')
            baseline = verifier_baseline(case, fixture, payload)
            raw = canonical_json(payload['result']['receipt']['result']['memory'])
            owned.add(raw)
            reference = references.setdefault(raw, 'r' + str(len(references)))
            if payload['registered_reference'] != reference:
                raise ValueError('External receipt reference was rebound')
        elif event['kind'] == 'verifier_baseline':
            if baseline is None or canonical_json(payload) != canonical_json(baseline):
                raise ValueError('Verifier baseline is not derived from authentic declared peer work')
        elif event['kind'] == 'tool_requested':
            pending = event
        elif event['kind'] == 'tool_completed':
            if pending is None or pending['call_id'] != event['call_id']:
                raise ValueError('Result lacks dispatched request')
            if payload['name'] != 'retrieve':
                receipt = payload['result']['receipt']
                _validated_receipt(receipt)
                args = pending['payload']['arguments']
                source = decode_json(ledger[args['record_ref']][1])
                replacement = decode_json(ledger[args['replacement_ref']][1])
                request, digest = canonical_supersession_request(source['id'],
                    replacement_memory_id=replacement['id'], expected_memory_digest=memory_digest(source),
                    expected_replacement_digest=memory_digest(replacement), reason=args['reason'], **scope)
                if (payload['name'] != 'supersede' or receipt['request_digest'] != digest
                        or receipt['key'] != pending['payload']['idempotency_key']
                        or receipt['scope'] != scope or receipt['store_id'] not in {r['store_id'] for r in fixture['seed_receipts']}):
                    raise ValueError('Model mutation lacks original scoped proposal authority')
                _supersession_receipt(receipt, request)
                raw = canonical_json(receipt['result']['memory'])
                owned.add(raw)
                references.setdefault(raw, 'r' + str(len(references)))
            else:
                last_retrieval = payload
            additions = presented_record_identities(payload, scope)
            if any(raw not in owned or references.get(raw) != reference for reference, (_, raw) in additions.items()):
                raise ValueError('Presented record lacks prior initial, peer, or model receipt authority')
            ledger.update(additions)
            pending = None
        elif event['kind'] == 'tool_failed':
            if 'recovery_proof' in payload:
                if pending is None:
                    raise ValueError('Conflict lacks dispatched proposal')
                validate_precommit_conflict_proof(payload['recovery_proof'], request=pending['payload'], scope=scope, ledger=ledger)
            pending = None
    return {'peer_operations': peer_count, 'presented_references': len(ledger),
            'baseline': baseline, 'pending_dispatch': pending is not None}


def assess_case(case, events):
    """Require an exercised conflict and model-chosen recovery, never peer credit."""
    completed = [event for event in events if event['kind'] == 'model_completed']
    peers = [event for event in events if event['kind'] == 'external_peer_completed']
    conflicts = [(i, event) for i, event in enumerate(events) if event['kind'] == 'tool_failed'
                 and event['payload'].get('recovery_proof', {}).get('record_role') == case['required_conflict_role']]
    outcomes = [event['payload'] for event in events if event['kind'] == 'outcome']
    outcome = outcomes[-1] if outcomes else {'status': 'interrupted', 'answer': None}
    verdict = next((event['payload'] for event in events if event['kind'] == 'verifier_result'), {})
    retrieved = [(i, event) for i, event in enumerate(events) if event['kind'] == 'tool_completed'
                 and event['payload']['name'] == 'retrieve']
    mutations = [(i, event) for i, event in enumerate(events) if event['kind'] == 'tool_completed'
                 and event['payload']['name'] == case['requested_operation']]
    baseline = next((event['payload'] for event in events if event['kind'] == 'verifier_baseline'), None)
    expected_target = baseline['seed_records'][case['peer_operation']['target_alias']] if baseline else None
    expected_current = baseline['seed_records']['current'] if baseline else None
    dispatched = {event.get('call_id'): event['payload'] for event in events if event['kind'] == 'tool_requested'}

    def contains_record(event, record, reference=None):
        if record is None or not any(canonical_json(value) == canonical_json(record)
                                    for value in event['payload']['result'].get('observed_records', [])):
            return False
        return any(value.get('id') == record['id'] and value.get('record_ref') is not None
                   and (reference is None or value.get('record_ref') == reference)
                   for value in json.loads(event['payload']['model_result']).get('records', []))

    recovery = False
    for ci, _ in conflicts:
        for mi, mutation in mutations:
            proposed = dispatched.get(mutation.get('call_id'), {}).get('arguments', {})
            target_field = 'record_ref' if case['required_conflict_role'] == 'source' else 'replacement_ref'
            used_ref = proposed.get(target_field)
            refreshed = used_ref is not None and any(ci < ri < mi and contains_record(read, expected_target, used_ref)
                                                     for ri, read in retrieved)
            read_back = any(mi < ri and contains_record(read, expected_current) for ri, read in retrieved)
            recovery = recovery or bool(refreshed and read_back)
    feedback_completed = False
    for ci, failed in conflicts:
        raw = next((event['payload'] for event in completed
                    if event['payload'].get('dml_tool_call_id') == failed.get('call_id')), None)
        if raw is None:
            continue
        for i, event in enumerate(events):
            if i <= ci or event['kind'] != 'model_requested':
                continue
            messages = event['payload']['request']['messages']
            latest = messages[-1] if messages else {}
            if (latest.get('role') == 'tool' and latest.get('tool_call_id') == raw['native_tool_call_id']
                    and latest.get('content') == failed['payload']['model_result']
                    and any(done.get('call_id') == event.get('call_id') for done in completed)):
                feedback_completed = True
    passed = (len(peers) == 1 and bool(conflicts) and recovery and feedback_completed
              and outcome['status'] == 'completed' and verdict.get('success') is True)
    return {'id': case['id'], 'status': outcome['status'], 'model_calls_completed': len(completed),
        'external_peer_operations': len(peers), 'qualified_conflicts': len(conflicts),
        'recovery_feedback_completed': feedback_completed, 'model_retrieved_mutated_and_read_back_after_conflict': recovery,
        'model_selected_retrieve_calls': len(retrieved), 'model_owned_lifecycle_calls': len(mutations),
        'original_verifier': verdict, 'branch_unexercised': not conflicts, 'synthetic_case_pass': bool(passed)}


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
            event = {**event, **({'episode_sequence': event['sequence']} if 'sequence' in event else {}), 'sequence': len(events), 'received_ns': time.time_ns(), 'previous_digest': previous}
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
        requested_ids = {event.get('call_id') for event in events if event['kind'] == 'model_requested'}
        resolved_ids = {event.get('call_id') for event in events if event['kind'] in ('model_completed', 'model_failed')}
        unresolved = bool(requested_ids - resolved_ids)
        diagnostic_failed = any(event['kind'] == 'diagnostic_error' for event in events)
        missing_terminal = not any(event['kind'] == 'terminal' for event in events)
        unknown_effects = suite['consumer_profile'] == NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE and any(
            (event['kind'] == 'tool_failed' and event['payload'].get('effects') == 'unknown')
            or (event['kind'] == 'terminal' and (event['payload'].get('effects_unknown')
                                                or event['payload'].get('usage_unknown')))
            for event in events)
        if unknown_effects:
            unrun_reason = unrun_reason or 'tool_effects_or_usage_unknown'
        # Worker exit/diagnostic completion never proves a dispatched request stopped.
        if not finished or unknown_dispatch or unresolved or diagnostic_failed or missing_terminal or unknown_effects:
            unrun_reason = unrun_reason or ('diagnostic_error_or_incomplete_episode'
                if diagnostic_failed or missing_terminal else 'remote_execution_failed_usage_or_cancellation_unknown')
            break
    summary = {'schema_version': suite['schema_version'], 'classification': suite['classification'],
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
        with NativeRemoteVLLMActionInputConsumer(args.snapshot, consumer_profile=args.consumer_profile, offline=True) as consumer:
            suite = prepare_suite(args.output, identity=consumer.identity.to_payload(),
                                  limits=EpisodeLimits(**json.loads(Path(args.limits_file).read_text())),
                                  consumer_profile=args.consumer_profile)
        print(json.dumps({'suite_digest': suite['suite_digest'], 'cases': list(CASE_IDS)}))
    else:
        print(json.dumps(run_suite(args.suite, snapshot=args.snapshot, output=args.output)))


if __name__ == '__main__':
    main()
