"""Predeclare two noncorpus, real-peer stale-reference recovery cases.

No model outcome participates in construction. Initial fixtures stay immutable;
the worker separately attests the declared peer update and verifier baseline.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
from pathlib import Path
import secrets
import time

from daystrom_dml.contracts.agent_episode import (NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE,
    NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE, NATIVE_RECOVERY_CONSUMER_PROFILES)
from daystrom_dml.services.agent_episode import EpisodeLimits
from daystrom_dml.services.remote_vllm_action_input import sampling_policy_identity
from scripts.agent_episodes import _source_digests
from scripts.native_dml_synthetic import QUALIFICATION_LIMITS, _digest, _write_exclusive

CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE
SCHEMA = 'dml-native-recovery-synthetic-v1'
SCHEMA_V6 = 'dml-native-recovery-synthetic-v2-budget-guidance'
CASE_IDS = ('stale_source_recovery', 'stale_replacement_recovery')
SCOPE = {'tenant_id': 'synthetic-recovery', 'client_id': 'noncorpus',
         'session_id': 'peer-update-session', 'instance_id': 'agent'}
BASELINE_POLICY = 'validated-declared-peer-update-separate-baseline-v1'
RUN_POLICY = {'runs_per_case': 1, 'automatic_retry': False, 'repair': False,
    'forced_tool_choice': False, 'all_turn_json_grammar': False,
    'final_contract_unchanged': True, 'case_order': list(CASE_IDS),
    'peer_after': 'first successful retrieval',
    'peer_actor': 'declared-external-update-not-model-owned',
    'verifier_baseline': BASELINE_POLICY,
    'live_retirement_qualified': False,
    'no_conflict_is_unqualified': True}


def source_digests(*, consumer_profile=CONSUMER_PROFILE):
    files = _source_digests(consumer_profile=consumer_profile)
    for name in ('native_recovery_suite.py', 'native_recovery_synthetic.py', 'native_dml_synthetic.py'):
        files['scripts.' + name.removesuffix('.py')] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return files


def prepare_suite(path, *, identity, limits, consumer_profile=CONSUMER_PROFILE):
    if consumer_profile not in NATIVE_RECOVERY_CONSUMER_PROFILES or type(limits) is not EpisodeLimits or asdict(limits) != QUALIFICATION_LIMITS:
        raise ValueError('Recovery declaration requires an explicit recovery profile and unchanged episode limits')
    cases = []
    for index, case_id in enumerate(CASE_IDS):
        subject = 'Synthetic registry ' + secrets.token_hex(6)
        stale, current, marker = (secrets.token_hex(12) for _ in range(3))
        claim_key = 'registry.access_phrase'
        def seed(alias, text, value):
            return {'alias': alias, 'text': text, 'meta': {'source': 'synthetic-' + alias,
                'source_trust': 'trusted', 'claim_key': claim_key, 'claim_value': value, 'no_merge': True}}
        seeds = [seed('stale', subject + ' OLD access phrase is ' + stale + '.', stale),
                 seed('current', subject + ' CURRENT access phrase is ' + current + '.', current)]
        target = 'stale' if index == 0 else 'current'
        peer_episode = 'synthetic-peer-' + secrets.token_hex(12)
        peer_call = 'declared-update'
        original = next(value for value in seeds if value['alias'] == target)
        peer_text = original['text'] + ' Registry annotation: ' + marker + '.'
        prompt = ('For ' + subject + ', mark the OLD entry superseded by the CURRENT entry using stored references. '
                  'After the change, retrieve again to check the current evidence. '
                  'Report the current access phrase as claim key ' + claim_key + ' with its exact stored value and cite its evidence.')
        task = {'id': case_id, 'prompt': prompt,
                'truth': {claim_key: {'value': current, 'evidence_aliases': ['current']}},
                'state_expectations': [{'alias': 'stale', 'state': 'superseded', 'replacement_alias': 'current'}]}
        cases.append({'id': case_id, 'ordinal': index, 'prompt': prompt,
            'scope': {**SCOPE, 'session_id': SCOPE['session_id'] + '-' + str(index)},
            'seeds': seeds, 'task': task, 'allowed_tools': ['retrieve', 'supersede'],
            'peer_after': 'first successful retrieval',
            'peer_required_aliases': ['stale', 'current'], 'requested_operation': 'supersede',
            'peer_operation': {'operation': 'update', 'target_alias': target, 'text': peer_text,
                'reason': 'Predeclared synthetic concurrent registry annotation',
                'episode_id': peer_episode, 'call_id': peer_call,
                'idempotency_key': 'episode:' + peer_episode + ':' + peer_call},
            'required_conflict_role': 'source' if index == 0 else 'replacement',
            'verifier_baseline_policy': BASELINE_POLICY,
            'expected': {'key': claim_key, 'value': current, 'evidence_alias': 'current'},
            'required_recovery': {'authenticated_conflict': True, 'model_chosen_next_action': True,
                'model_owned_supersession': True, 'subsequent_readback': True, 'original_cited_final': True}})
    suite = {'schema_version': SCHEMA_V6 if consumer_profile == NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE else SCHEMA, 'classification': 'synthetic-noncorpus-development-not-acceptance',
        'created_ns': time.time_ns(), 'consumer_profile': consumer_profile,
        'model_identity': identity, 'sampling': sampling_policy_identity(), 'limits': asdict(limits),
        'effective_time': 2000000000, 'source_digests': source_digests(consumer_profile=consumer_profile), 'cases': cases,
        'run_policy': deepcopy(RUN_POLICY),
        'development_scope': 'Designed after prior failures; not held-out generalization; no evaluation corpus used'}
    suite['suite_digest'] = _digest(suite)
    validate_suite(suite)
    _write_exclusive(path, suite)
    return suite


def validate_suite(suite):
    body = {key: value for key, value in suite.items() if key != 'suite_digest'}
    expected_schema = SCHEMA_V6 if suite.get('consumer_profile') == NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE else SCHEMA
    if suite.get('schema_version') != expected_schema or suite.get('suite_digest') != _digest(body):
        raise ValueError('Recovery suite digest/version mismatch')
    if (suite['consumer_profile'] not in NATIVE_RECOVERY_CONSUMER_PROFILES or suite['limits'] != QUALIFICATION_LIMITS
            or suite['sampling'] != sampling_policy_identity() or suite['source_digests'] != source_digests(consumer_profile=suite['consumer_profile'])
            or suite['run_policy'] != RUN_POLICY or [case['id'] for case in suite['cases']] != list(CASE_IDS)):
        raise ValueError('Recovery declaration policy/source changed')
    for index, case in enumerate(suite['cases']):
        target = 'stale' if index == 0 else 'current'
        seed = next(value for value in case['seeds'] if value['alias'] == target)
        if (case['peer_operation']['operation'] != 'update' or case['peer_operation']['target_alias'] != target
                or not case['peer_operation']['text'].startswith(seed['text'] + ' Registry annotation: ')
                or case['required_conflict_role'] != ('source' if index == 0 else 'replacement')
                or case['allowed_tools'] != ['retrieve', 'supersede']
                or case['peer_required_aliases'] != ['stale', 'current'] or case['requested_operation'] != 'supersede'
                or case['peer_operation']['idempotency_key'] != 'episode:' + case['peer_operation']['episode_id'] + ':' + case['peer_operation']['call_id']
                or case['peer_after'] != RUN_POLICY['peer_after']
                or case['verifier_baseline_policy'] != BASELINE_POLICY):
            raise ValueError('Recovery case differs from declared peer-update construction')
    return EpisodeLimits(**suite['limits'])
