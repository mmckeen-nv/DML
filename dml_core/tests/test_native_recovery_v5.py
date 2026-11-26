"""Synthetic controls of v5 recovery; no model or live qualification claims."""
from copy import deepcopy
from dataclasses import replace

import pytest

from daystrom_dml.contracts.agent_episode import (
    AgentEpisodeError, EXECUTION_PROTOCOL_V2, NATIVE_REMOTE_VLLM_V4_CONSUMER_PROFILE as V4,
    NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE as V5, PRECOMMIT_CONFLICT_MODEL_RESULT,
    canonical_json, decode_json, native_policy_identity, native_system_policy, validate_episode_events,
)
from daystrom_dml.services.agent_episode import (
    EpisodeLimits, _prepare_fixture, _read_records, run_episode_with_test_dependencies,
)
from daystrom_dml.services.episode_tools import SelectedProfileEpisodeTools
from daystrom_dml.services.episode_verifiers import load_episode_corpus
from test_native_episode_integration import NativeSyntheticConsumer


class RecoveryConsumer(NativeSyntheticConsumer):
    profile = V5

    def compile(self, *args, **kwargs):
        # Reuse the synthetic v4 tokenizer shim, restoring v5 before execution.
        self.profile = V4
        try:
            artifact = super().compile(*args, **kwargs)
        finally:
            self.profile = V5
        return replace(artifact, identity=replace(artifact.identity,
            runtime_identity='dml-remote-vllm-native-tools-runtime-v5:' + 'a' * 64))


def recovery_case(tmp_path, *, steps=5, unknown=False, profile=V5, max_output_tokens=1024):
    scenario = next(s for s in load_episode_corpus()['scenarios'] if s['id'] == 'superseded_preference')
    task = scenario['tasks'][0]
    directory = tmp_path / 'authority'
    adapter, prepared = _prepare_fixture(directory, scenario, 'recovery-test')
    bridge = SelectedProfileEpisodeTools(adapter, scope=scenario['scope'], episode_id='recovery-test',
        seed_receipts=prepared['seed_receipts'], observation_records=list(prepared['seed_records'].values()),
        allowed_tools=('retrieve', 'supersede'), execution_protocol=EXECUTION_PROTOCOL_V2,
        recover_precommit_conflicts=profile == V5)
    stale = {}

    def mutation(request):
        records = decode_json(request['messages'][-1]['content'])['records']
        refs = {record['id']: record['record_ref'] for record in records}
        stale.update(kind='tool', name='supersede', arguments={
            'record_ref': refs[prepared['seed_records']['old']['id']],
            'replacement_ref': refs[prepared['seed_records']['current']['id']], 'reason': 'synthetic stale proposal'})
        return deepcopy(stale)

    answer = {'schema_version': 'dml-agent-action-v1', 'kind': 'final', 'answer': {'claims': [{
        'key': key, 'value': fact['value'],
        'evidence_ids': [prepared['seed_records'][alias]['id'] for alias in fact['evidence_aliases']]}
        for key, fact in task['truth'].items()]}}
    actions = [{'kind': 'tool', 'name': 'retrieve', 'arguments': {'query': 'preference', 'top_k': 10}},
               mutation, lambda request: deepcopy(stale), answer]
    consumer = RecoveryConsumer(actions) if profile == V5 else NativeSyntheticConsumer(actions)
    if profile != V5:
        consumer.profile = profile
    if unknown:
        from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict
        original = bridge.execute
        calls = []

        def execute(value):
            calls.append(value)
            if len(calls) == 3:
                raise ReceiptLifecycleConflict('unregistered late conflict')
            return original(value)
        bridge.execute = execute
    try:
        report = run_episode_with_test_dependencies(consumer=consumer, toolbox=bridge,
            scenario=scenario, task=task, seed_records=prepared['seed_records'],
            current_records=lambda: _read_records(directory), episode_id='recovery-test',
            consumer_profile=profile, limits=EpisodeLimits(max_steps=steps, output_tokens=16, max_output_tokens=max_output_tokens))
        return report, consumer
    finally:
        adapter.close()


def test_v5_owned_conflict_preserves_failure_and_authentic_feedback(tmp_path):
    report, consumer = recovery_case(tmp_path)
    failures = [event for event in report['events'] if event['kind'] == 'tool_failed']
    assert len(failures) == 1
    failure = failures[0]['payload']
    assert failure['error_code'] == 'ReceiptPreconditionConflict'
    assert failure['error_message']
    assert failure['effects'] == 'none'
    assert failure['model_result'] == PRECOMMIT_CONFLICT_MODEL_RESULT
    assert failure['recovery_proof']['save_entered'] is False
    assert consumer.requests[-1]['messages'][-1]['tool_call_id'] == 'native-call-3'
    assert consumer.requests[-1]['messages'][-1]['content'] == PRECOMMIT_CONFLICT_MODEL_RESULT
    assert len(consumer.dispatched) == 4
    assert report['terminal']['success']
    validate_episode_events(report['events'])


def test_v5_recovery_does_not_grant_extra_step(tmp_path):
    report, consumer = recovery_case(tmp_path, steps=3)
    assert len(consumer.dispatched) == 3
    assert report['terminal']['status'] == 'step_limit'
    assert any(event['kind'] == 'tool_failed' for event in report['events'])
    validate_episode_events(report['events'])


def test_v5_unowned_lifecycle_failure_unknown_and_terminal(tmp_path):
    report, consumer = recovery_case(tmp_path, unknown=True)
    failure = next(event['payload'] for event in report['events'] if event['kind'] == 'tool_failed')
    assert failure['effects'] == 'unknown'
    assert 'recovery_proof' not in failure and 'model_result' not in failure
    assert failure['error_message'] == 'unregistered late conflict'
    assert report['terminal']['status'] == 'tool_error'
    assert len(consumer.dispatched) == 3


@pytest.mark.parametrize('field,value', [('save_entered', True), ('expected_memory_digest', '0' * 64),
    ('request_digest', '0' * 64), ('snapshot_digest', '0' * 64), ('record_role', 'target'),
    ('operation', 'update'), ('key', 'different')])
def test_v5_replay_rejects_tampered_conflict(tmp_path, field, value):
    report, _ = recovery_case(tmp_path)
    events = deepcopy(report['events'])
    failed = next(event for event in events if event['kind'] == 'tool_failed')
    failed['payload']['recovery_proof'][field] = value
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


def test_v5_system_and_native_authority_unchanged_identity_distinct():
    assert native_system_policy(consumer_profile=V5) == native_system_policy(consumer_profile=V4)
    old, new = native_policy_identity(consumer_profile=V4), native_policy_identity(consumer_profile=V5)
    assert 'precommit_conflict_recovery' not in old
    assert new['precommit_conflict_recovery']['model_result'] == PRECOMMIT_CONFLICT_MODEL_RESULT
    assert old['system_message_sha256'] == new['system_message_sha256']
    assert canonical_json(old) != canonical_json(new)


@pytest.mark.parametrize('change', ['missing', 'scope', 'wrong_branch'])
def test_v5_counterpart_and_branch_tampering_rejected(tmp_path, change):
    import hashlib
    import json
    from daystrom_dml.services.receipt_lifecycle import memory_digest
    report, _ = recovery_case(tmp_path)
    events = deepcopy(report['events'])
    proof = next(event['payload']['recovery_proof'] for event in events if event['kind'] == 'tool_failed')
    other_id = proof['references']['replacement_ref']['record_id']
    other = next(record for record in proof['snapshot']['items'] if record['id'] == other_id)
    if change == 'missing':
        proof['snapshot']['items'].remove(other)
    elif change == 'scope':
        other['meta']['tenant_id'] = 'foreign'
    else:
        other['text'] += ' changed replacement'
        proof.update(record_role='replacement', record_id=other_id, observed_record=deepcopy(other),
            expected_memory_digest=proof['references']['replacement_ref']['memory_digest'],
            observed_memory_digest=memory_digest(other))
    proof['snapshot_digest'] = hashlib.sha256(json.dumps(proof['snapshot'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


def test_v4_same_conflict_remains_terminal_legacy_payload(tmp_path):
    report, consumer = recovery_case(tmp_path, profile=V4)
    failed = next(event['payload'] for event in report['events'] if event['kind'] == 'tool_failed')
    assert set(failed) == {'name', 'error_code', 'effects', 'latency_ms'}
    assert failed['error_code'] == 'ReceiptLifecycleConflict'
    assert report['terminal']['status'] == 'tool_error'
    assert len(consumer.dispatched) == 3


def test_v5_failure_feedback_cannot_be_rewritten(tmp_path):
    report, _ = recovery_case(tmp_path)
    events = deepcopy(report['events'])
    failed = next(event for event in events if event['kind'] == 'tool_failed')
    failed['payload']['model_result'] = '{"effects":"none","result":"success"}'
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


def test_v5_recovery_does_not_grant_extra_output_budget(tmp_path):
    report, consumer = recovery_case(tmp_path, max_output_tokens=20)
    assert len(consumer.dispatched) == 3
    assert report['terminal']['status'] == 'token_limit'
    assert any(event['kind'] == 'tool_failed' for event in report['events'])
    assert sum(event['payload']['output_token_count'] for event in report['events'] if event['kind'] == 'model_completed') == 6
    validate_episode_events(report['events'])
