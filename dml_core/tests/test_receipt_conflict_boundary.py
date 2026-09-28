"""Owned precommit classification; no exception-name recovery or hidden retries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json

import pytest

from daystrom_dml.journal import RevisionConflict
from daystrom_dml.services.receipt_conflict_boundary import ReceiptConflictBoundary, ReceiptPreconditionConflict
from daystrom_dml.services.receipt_lifecycle import (
    ReceiptLifecycleConflict, ReceiptMemoryNotFound, canonical_retirement_request,
    memory_digest, retire_receipted,
)
from test_receipt_supersession import SCOPE, intent, seeded, supersede


def boundary(journal, request, digest, operation='supersede', key='conflict'):
    return ReceiptConflictBoundary(journal=journal, operation=operation,
        request_digest=digest, key=key, scope=request['scope'])


def change(journal, record_id):
    revision, snapshot = journal.read_snapshot()
    next(record for record in snapshot['items'] if record['id'] == record_id)['text'] += ' changed'
    journal.save(snapshot, expected_revision=revision)


@pytest.mark.parametrize('role', ['source', 'replacement', 'target'])
def test_scoped_snapshot_staleness_mints_exact_owned_no_write_proof(tmp_path, monkeypatch, role):
    journal, source, replacement = seeded(tmp_path)
    if role == 'target':
        request, digest = canonical_retirement_request(source['id'], expected_memory_digest=memory_digest(source), reason='retire', **SCOPE)
        operation = 'retire'
        def invoke():
            return retire_receipted(journal, request=request, request_digest=digest, key='conflict', hydrate=lambda *_: None, degraded=lambda _: None)
    else:
        request, digest = intent(source, replacement)
        operation = 'supersede'
        def invoke():
            return supersede(journal, request=request, digest=digest, key='conflict')
    change(journal, replacement['id'] if role == 'replacement' else source['id'])
    before = journal.read_snapshot()
    def forbidden(*args, **kwargs):
        pytest.fail('Stale record crossed write boundary')
    monkeypatch.setattr(journal, 'save_with_receipt', forbidden)
    owner = boundary(journal, request, digest, operation)
    with owner, pytest.raises(ReceiptPreconditionConflict) as caught:
        invoke()
    proof = owner.authenticate(caught.value)
    assert proof['record_role'] == role and proof['snapshot_revision'] == before[0]
    assert proof['snapshot'] == before[1] and proof['save_entered'] is False
    assert proof['expected_memory_digest'] != proof['observed_memory_digest']
    assert proof['observed_memory_digest'] == memory_digest(proof['observed_record'])
    assert proof['snapshot_digest'] == hashlib.sha256(json.dumps(before[1], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert journal.read_snapshot() == before
    assert journal.lookup_receipt(scope=SCOPE, key='conflict', request_digest=digest) is None
    assert owner.authenticate(ReceiptPreconditionConflict(str(caught.value))) is None
    assert boundary(journal, request, digest, operation).authenticate(caught.value) is None
    proof['snapshot']['items'].clear()
    assert owner.authenticate(caught.value) is None  # Proof is consumed exactly once.
    # The default API and post-context calls retain their exact original type.
    with pytest.raises(ReceiptLifecycleConflict) as legacy:
        invoke()
    assert type(legacy.value) is ReceiptLifecycleConflict


def test_entering_save_permanently_disables_recovery_after_revision_race(tmp_path, monkeypatch):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    attempts = []
    def raced_save(*args, **kwargs):
        attempts.append(True)
        change(journal, source['id'])
        raise RevisionConflict('simulated concurrent commit')
    monkeypatch.setattr(journal, 'save_with_receipt', raced_save)
    owner = boundary(journal, request, digest)
    with owner, pytest.raises(ReceiptLifecycleConflict) as caught:
        supersede(journal, request=request, digest=digest, key='conflict')
    assert type(caught.value) is ReceiptLifecycleConflict
    assert len(attempts) == 1 and owner.authenticate(caught.value) is None


@pytest.mark.parametrize('unsafe', ['scope', 'existing_receipt', 'postcommit_receipt', 'forged_subtype'])
def test_unsafe_paths_never_mint_owned_recovery(tmp_path, monkeypatch, unsafe):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    if unsafe == 'scope':
        revision, snapshot = journal.read_snapshot()
        snapshot['items'][0]['meta']['tenant_id'] = 'another'
        journal.save(snapshot, expected_revision=revision)
    elif unsafe == 'existing_receipt':
        monkeypatch.setattr(journal, 'lookup_receipt', lambda **_: {'result': {'memory': source}})
    elif unsafe == 'postcommit_receipt':
        save = journal.save_with_receipt
        def corrupted_ack(*args, **kwargs):
            receipt = deepcopy(save(*args, **kwargs))
            receipt['result']['memory']['meta']['memory_state'] = 'active'
            return receipt
        monkeypatch.setattr(journal, 'save_with_receipt', corrupted_ack)
    else:
        def forged():
            raise ReceiptPreconditionConflict('Source memory changed since the supersession decision')
        monkeypatch.setattr(journal, 'read_snapshot', forged)
    owner = boundary(journal, request, digest)
    with owner, pytest.raises((ReceiptLifecycleConflict, ReceiptMemoryNotFound)) as caught:
        supersede(journal, request=request, digest=digest, key='conflict')
    assert owner.authenticate(caught.value) is None
    if unsafe == 'postcommit_receipt':
        assert journal.lookup_receipt(scope=SCOPE, key='conflict', request_digest=digest) is not None


def test_invocation_context_is_thread_local_nonreentrant_and_one_shot(tmp_path):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    change(journal, source['id'])
    owner = boundary(journal, request, digest)
    with owner:
        with pytest.raises(RuntimeError, match='cannot nest'):
            with boundary(journal, request, digest):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(supersede, journal, request=request, digest=digest, key='conflict')
            with pytest.raises(ReceiptLifecycleConflict) as caught:
                future.result()
            assert type(caught.value) is ReceiptLifecycleConflict
            assert owner.authenticate(caught.value) is None
        with pytest.raises(ReceiptPreconditionConflict) as caught:
            supersede(journal, request=request, digest=digest, key='conflict')
    assert owner.authenticate(caught.value) is not None
    with pytest.raises(RuntimeError, match='one-shot'):
        with owner:
            pass


def test_gateway_owns_only_exact_prepared_invocation_and_keeps_prior_commit(tmp_path):
    from daystrom_dml.contracts.agent_episode import EXECUTION_PROTOCOL_V2
    from daystrom_dml.services.agent_episode import _prepare_fixture
    from daystrom_dml.services.episode_tools import EpisodeToolConflictRejected, SelectedProfileEpisodeTools
    scenario = {'scope': SCOPE, 'seeds': [
        {'alias': name, 'text': text, 'meta': {'source': name, 'source_trust': 'trusted', 'no_merge': True}}
        for name, text in [('old', 'Previous setting'), ('new', 'Replacement setting')]], 'setup': [], 'expected_memory_count': 2}
    adapter, fixture = _prepare_fixture(tmp_path / 'gateway', scenario, 'recovery')
    try:
        bridge = SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id='recovery', seed_receipts=fixture['seed_receipts'],
            allowed_tools=('retrieve', 'supersede'), execution_protocol=EXECUTION_PROTOCOL_V2, recover_precommit_conflicts=True)
        bridge.execute(bridge.prepare('retrieve', {'query': 'setting', 'top_k': 10}, call_id='read'))
        args = {'record_ref': 'r0', 'replacement_ref': 'r1', 'reason': 'apply replacement'}
        bridge.execute(bridge.prepare('supersede', args, call_id='first'))
        committed = adapter._journal.read_snapshot()
        prepared = bridge.prepare('supersede', args, call_id='second')
        with pytest.raises(EpisodeToolConflictRejected) as caught:
            bridge.execute(prepared)
        error = caught.value
        assert bridge.owns_conflict_rejection(error, prepared)
        assert not bridge.owns_conflict_rejection(error, prepared)
        assert error.original_error_code == 'ReceiptPreconditionConflict'
        assert error.proof['references']['record_ref']['record_ref'] == 'r0'
        assert adapter._journal.read_snapshot() == committed
        fake = EpisodeToolConflictRejected(object(), prepared, error.proof, ReceiptLifecycleConflict('fake'))
        assert not bridge.owns_conflict_rejection(fake, prepared)
        assert not bridge.owns_conflict_rejection(error, bridge.prepare('supersede', args, call_id='third'))
        error.original_error_code = 'forged'
        assert not bridge.owns_conflict_rejection(error, prepared)
    finally:
        adapter.close()


@pytest.mark.parametrize("write_kind", ["direct", "other_receipt"])
def test_nested_journal_write_on_another_operation_invalidates_proof(tmp_path, write_kind):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    owner = boundary(journal, request, digest)
    with owner:
        if write_kind == "direct":
            change(journal, source['id'])
        else:
            supersede(journal, source, replacement, key='other-operation')
        with pytest.raises(ReceiptLifecycleConflict) as caught:
            supersede(journal, request=request, digest=digest, key='conflict')
    assert type(caught.value) is ReceiptLifecycleConflict
    assert owner.authenticate(caught.value) is None


@pytest.mark.parametrize('race', ['matching_receipt', 'different_receipt', 'revision', 'lookup_failure'])
def test_atomic_absence_guard_rejects_raw_writer_race(tmp_path, monkeypatch, race):
    from daystrom_dml.journal import IdempotencyConflict
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    change(journal, source['id'])
    original_read = journal.read_snapshot
    original_guard = journal.observe_precommit_receipt_absence
    invoked = []

    def raw_writer():
        revision, payload = original_read()
        if race == 'revision':
            payload['items'][2]['text'] += ' external change'
            journal.save(payload, expected_revision=revision)
        else:
            journal.save_with_receipt(payload, scope=SCOPE, key='conflict',
                request_digest=digest if race == 'matching_receipt' else 'f' * 64,
                result={'memory': payload['items'][0]}, expected_revision=revision)

    def raced_guard(**kwargs):
        invoked.append(True)
        if race == 'lookup_failure':
            raise OSError('receipt authority unavailable')
        # A raw journal writer need not honor the service advisory lock. Its
        # separate thread also cannot inherit the invocation-local save latch.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(raw_writer).result()
        return original_guard(**kwargs)

    monkeypatch.setattr(journal, 'observe_precommit_receipt_absence', raced_guard)
    owner = boundary(journal, request, digest)
    expected = IdempotencyConflict if race == 'different_receipt' else OSError if race == 'lookup_failure' else ReceiptLifecycleConflict
    with owner, pytest.raises(expected) as caught:
        supersede(journal, request=request, digest=digest, key='conflict')
    assert invoked == [True]
    assert type(caught.value) is expected
    assert owner.authenticate(caught.value) is None


@pytest.mark.parametrize('entry', ['save', 'save_with_receipt', '_save'])
def test_all_journal_save_entries_latch_even_unrelated_write(tmp_path, entry):
    journal, source, replacement = seeded(tmp_path)
    request, digest = intent(source, replacement)
    change(journal, source['id'])
    revision, payload = journal.read_snapshot()
    owner = boundary(journal, request, digest)
    with owner:
        if entry == 'save_with_receipt':
            journal.save_with_receipt(payload, scope=SCOPE, key='unrelated', request_digest='a' * 64,
                result={'memory': payload['items'][2]}, expected_revision=revision)
        else:
            getattr(journal, entry)(payload, expected_revision=revision)
        with pytest.raises(ReceiptLifecycleConflict) as caught:
            supersede(journal, request=request, digest=digest, key='conflict')
    assert type(caught.value) is ReceiptLifecycleConflict
    assert owner.authenticate(caught.value) is None
