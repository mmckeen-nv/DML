"""Exercise the episode gateway against actual selected-profile receipt authority."""
from __future__ import annotations

from copy import deepcopy
import json
import os

import pytest

from daystrom_dml.services.agent_episode import _fixture_adapter
from daystrom_dml.services.episode_tools import (
    EpisodeToolError, PreparedEpisodeTool, SelectedProfileEpisodeTools,
)
from daystrom_dml.services.receipt_lifecycle import ReceiptLifecycleConflict


SCOPE = {"tenant_id": "owner", "client_id": "client", "session_id": "session", "instance_id": "agent"}


@pytest.fixture
def authority(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("DML_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    adapter = _fixture_adapter(tmp_path / "authority")
    seeds = []
    for key, text, scope in (
        ("previous", "The owner prefers verbose answers.", SCOPE),
        ("current", "The owner now prefers concise answers.", SCOPE),
        ("foreign", "OTHER TENANT PRIVATE MEMORY", {**SCOPE, "tenant_id": "other"}),
    ):
        seeds.append(adapter.ingest_memory_receipted(text, idempotency_key=key,
            meta={"source_trust": "trusted", "source": key}, **scope))
    bridge = SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id="tool-test", seed_receipts=seeds)
    try:
        yield adapter, bridge, seeds
    finally:
        adapter.close()


def invoke(bridge, name, arguments, call_id):
    return bridge.execute(bridge.prepare(name, arguments, call_id=call_id))


def retrieved(bridge):
    raw, model_text = invoke(bridge, "retrieve", {"query": "owner answers", "top_k": 10}, "read")
    return raw, json.loads(model_text)["records"]


def test_actual_retrieval_yields_original_receipt_refs_and_exact_presented_text(authority):
    _, bridge, seeds = authority
    raw, shown = retrieved(bridge)
    originals = {receipt["result"]["memory"]["id"]: receipt["result"]["memory"] for receipt in seeds[:2]}
    assert {item["id"] for item in shown} == set(originals)
    assert {record["id"] for record in raw["observed_records"]} == set(originals)
    assert "OTHER TENANT" not in json.dumps(shown)
    for item in shown:
        assert item["text"] == originals[item["id"]]["text"]
        assert item["record_ref"]
        assert next(record for record in raw["observed_records"] if record["id"] == item["id"]) == originals[item["id"]]


@pytest.mark.parametrize("scope, top_k, count, limit_reached", [
    (SCOPE, 1, 1, True),
    (SCOPE, 2, 2, True),  # Exactly all eligible records also reaches the requested limit.
    (SCOPE, 10, 2, False),
    ({**SCOPE, "tenant_id": "empty"}, 1, 0, False),
], ids=["capped", "exact-limit", "below-limit", "empty"])
def test_retrieval_metadata_describes_only_the_requested_read(
        authority, monkeypatch, scope, top_k, count, limit_reached):
    adapter, _, seeds = authority
    bridge = SelectedProfileEpisodeTools(adapter, scope=scope, episode_id="read-metadata", seed_receipts=seeds)
    retrieve = adapter.retrieve_context
    calls, reports = [], []

    def record_read(query, **kwargs):
        calls.append((query, kwargs))
        report = retrieve(query, **kwargs)
        reports.append(deepcopy(report))
        return report

    monkeypatch.setattr(adapter, "retrieve_context", record_read)
    before = adapter._journal.verified_snapshot()
    raw, model_text = invoke(bridge, "retrieve", {"query": "owner answers", "top_k": top_k}, "read")
    shown = json.loads(model_text)
    assert calls == [("owner answers", {"top_k": top_k, "as_of": bridge.effective_time, **scope})]
    assert raw["report"] == reports[0]
    assert raw["receipt"] is None
    assert adapter._journal.verified_snapshot() == before
    assert set(shown) == {"records", "requested_top_k", "returned_count", "limit_reached"}
    assert shown["requested_top_k"] == top_k
    assert shown["returned_count"] == count
    assert shown["limit_reached"] is limit_reached
    assert len(shown["records"]) == count
    returned_ids = {int(item["id"]) for item in reports[0]["items"]}
    assert {item["id"] for item in shown["records"]} == returned_ids
    assert {record["id"] for record in raw["observed_records"]} == returned_ids
    assert not any(key in model_text for key in SCOPE)
    for receipt in seeds:
        record = receipt["result"]["memory"]
        if record["id"] not in returned_ids:
            assert record["text"] not in model_text
        else:
            presented = next(item for item in shown["records"] if item["id"] == record["id"])
            assert presented["text"] == record["text"]
            assert presented["record_ref"]
            assert next(item for item in raw["observed_records"] if item["id"] == record["id"]) == record


def test_generated_memory_is_untrusted_and_receipt_replay_does_not_duplicate(authority):
    adapter, bridge, _ = authority
    prepared = bridge.prepare("ingest", {"text": "The model guessed a new fact."}, call_id="write")
    raw, shown = bridge.execute(prepared)
    before_replay = adapter._journal.verified_snapshot()
    assert bridge.execute(prepared) == (raw, shown)
    assert adapter._journal.verified_snapshot() == before_replay
    record = raw["receipt"]["result"]["memory"]
    assert set(json.loads(shown)) == {"records"}
    assert record["meta"]["source_trust"] == "untrusted"
    assert record["meta"]["source"] == "episode-generated:tool-test"
    assert all(record["meta"][key] == value for key, value in SCOPE.items())
    assert record["id"] not in {item["id"] for item in retrieved(bridge)[1]}
    with pytest.raises(EpisodeToolError):
        bridge.prepare("ingest", {"text": "Changed content under the same key."}, call_id="write")


def test_update_uses_immutable_cas_and_old_reference_stays_stale(authority):
    adapter, bridge, seeds = authority
    _, shown = retrieved(bridge)
    original = seeds[0]["result"]["memory"]
    reference = next(item["record_ref"] for item in shown if item["id"] == original["id"])
    raw, model_text = invoke(bridge, "update", {"record_ref": reference,
        "text": "The owner explicitly corrected this note.", "reason": "owner correction"}, "update")
    updated = raw["receipt"]["result"]["memory"]
    assert updated["id"] == original["id"]
    assert updated["text"] != original["text"]
    assert json.loads(model_text)["records"][0]["record_ref"] != reference
    before_rejection = adapter._journal.verified_snapshot()
    with pytest.raises(ReceiptLifecycleConflict):
        invoke(bridge, "retire", {"record_ref": reference, "reason": "stale view"}, "retire-stale")
    assert adapter._journal.verified_snapshot() == before_rejection
    assert next(record for record in retrieved(bridge)[0]["observed_records"] if record["id"] == original["id"]) == updated


@pytest.mark.parametrize("operation", ["promote", "retire", "supersede"])
def test_lifecycle_tools_return_committed_full_record_and_preserve_other_sources(authority, operation):
    adapter, bridge, seeds = authority
    _, shown = retrieved(bridge)
    ids = [receipt["result"]["memory"]["id"] for receipt in seeds]
    refs = {item["id"]: item["record_ref"] for item in shown}
    arguments = {"record_ref": refs[ids[0]], "reason": "explicit task decision"}
    if operation == "promote":
        arguments = {"record_refs": [refs[ids[0]]], "text": "Derived summary of the original preference.",
                     "reason": "explicit task decision"}
    elif operation == "supersede":
        arguments["replacement_ref"] = refs[ids[1]]
    raw, model_text = invoke(bridge, operation, arguments, "lifecycle")
    record = raw["receipt"]["result"]["memory"]
    state = {item["id"]: item for item in adapter._journal.load()["items"]}
    assert state[record["id"]] == record
    assert raw["observed_records"] == [record]
    assert json.loads(model_text)["records"][0]["text"] == record["text"]
    assert state[ids[1]] == seeds[1]["result"]["memory"]
    assert state[ids[2]] == seeds[2]["result"]["memory"]
    if operation == "promote":
        assert record["level"] == 1
        assert record["summary_of"] == [ids[0]]
        assert state[ids[0]] == seeds[0]["result"]["memory"]
    else:
        assert record["meta"]["memory_state"] == ("deleted" if operation == "retire" else "superseded")
        assert ids[0] not in {item["id"] for item in retrieved(bridge)[1]}


def test_gateway_rejects_forged_preparation_and_task_disallowed_tool_without_writes(authority):
    adapter, _, seeds = authority
    bridge = SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id="read-only",
        seed_receipts=seeds, allowed_tools=("retrieve",))
    before = adapter._journal.verified_snapshot()
    with pytest.raises(EpisodeToolError):
        bridge.prepare("ingest", {"text": "disallowed"}, call_id="write")
    with pytest.raises(EpisodeToolError):
        bridge.execute(PreparedEpisodeTool("ingest", '{"text":"forged"}', "forged-key"))
    assert adapter._journal.verified_snapshot() == before


def test_owned_scope_and_receipts_are_detached_from_caller_mutation(authority):
    adapter, _, seeds = authority
    owned_scope, owned_seeds = deepcopy(SCOPE), deepcopy(seeds)
    bridge = SelectedProfileEpisodeTools(adapter, scope=owned_scope, episode_id="detached", seed_receipts=owned_seeds)
    owned_scope["tenant_id"] = "other"
    owned_seeds[0]["result"]["memory"]["text"] = "caller changed receipt"
    exported_scope = bridge.scope
    exported_scope["tenant_id"] = "other"
    _, shown = retrieved(bridge)
    assert bridge.scope == SCOPE
    assert "caller changed receipt" not in json.dumps(shown)
    assert "OTHER TENANT" not in json.dumps(shown)


@pytest.mark.parametrize("scope", [{"tenant_id": "owner"}, {**SCOPE, "admin": True}, {**SCOPE, "tenant_id": None}])
def test_incomplete_or_extended_trusted_scope_is_rejected(authority, scope):
    adapter, _, seeds = authority
    with pytest.raises(EpisodeToolError):
        SelectedProfileEpisodeTools(adapter, scope=scope, episode_id="bad-scope", seed_receipts=seeds)


def test_model_generated_memory_cannot_be_configured_as_trusted(authority):
    adapter, _, seeds = authority
    with pytest.raises(EpisodeToolError):
        SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id="bad-trust",
            seed_receipts=seeds, source_trust="trusted")


def test_retrieval_uses_pinned_episode_time_for_expiry(authority):
    adapter, _, seeds = authority
    receipt = adapter.ingest_memory_receipted("The old endpoint expires at the recorded cutoff.",
        idempotency_key="expiring", meta={"source_trust": "trusted", "expires_at": 1999999999}, **SCOPE)
    memory_id = receipt["result"]["memory"]["id"]
    visible = []
    for timestamp in (1999999998, 2000000000):
        bridge = SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id="clock-test",
            seed_receipts=[*seeds, receipt], effective_time=timestamp)
        raw, _ = retrieved(bridge)
        visible.append(memory_id in {int(item["id"]) for item in raw["report"]["items"]})
    assert visible == [True, False]


def validation_bridge(authority):
    from daystrom_dml.contracts.agent_episode import EXECUTION_PROTOCOL_V2
    adapter, _, seeds = authority
    return SelectedProfileEpisodeTools(adapter, scope=SCOPE, episode_id='validation-tools',
        seed_receipts=seeds, execution_protocol=EXECUTION_PROTOCOL_V2)


def test_v2_presented_self_supersession_rejects_before_any_authority_or_preparation(authority, monkeypatch):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    adapter, _, _ = authority
    bridge = validation_bridge(authority)
    _, records = retrieved(bridge)
    reference = records[0]['record_ref']
    keys, prepared = deepcopy(bridge._keys), dict(bridge._prepared)
    def forbidden(*args, **kwargs):
        pytest.fail('Semantic validation must not access adapter or storage')
    for name in ('retrieve_context', 'supersede_memory_receipted', 'ingest_memory_receipted',
                 'update_memory_receipted', 'retire_memory_receipted', 'promote_memories_receipted'):
        monkeypatch.setattr(adapter, name, forbidden)
    for name in ('read_snapshot', 'lookup_receipt', 'verified_snapshot'):
        monkeypatch.setattr(adapter._journal, name, forbidden)
    with pytest.raises(EpisodeToolValidationRejected) as caught:
        bridge.prepare('supersede', {'record_ref': reference, 'replacement_ref': reference, 'reason': 'same'}, call_id='tool-1')
    assert type(caught.value) is EpisodeToolValidationRejected
    assert bridge._keys == keys and bridge._prepared == prepared
    assert bridge.owns_validation_rejection(caught.value,
        {'record_ref': reference, 'replacement_ref': reference, 'reason': 'same'}, 'tool-1')


def test_v2_distinct_immutable_versions_of_same_id_cannot_evade_rejection(authority):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    bridge = validation_bridge(authority)
    _, records = retrieved(bridge)
    old = records[0]
    raw, visible = invoke(bridge, 'update', {'record_ref': old['record_ref'], 'text': 'Updated test text',
                                           'reason': 'test version'}, 'update')
    fresh = json.loads(visible)['records'][0]
    assert fresh['id'] == old['id'] and fresh['record_ref'] != old['record_ref']
    assert bridge._presentation_ledger[fresh['record_ref']] != bridge._presentation_ledger[old['record_ref']]
    before = dict(bridge._keys)
    with pytest.raises(EpisodeToolValidationRejected):
        bridge.prepare('supersede', {'record_ref': old['record_ref'], 'replacement_ref': fresh['record_ref'],
                                    'reason': 'two aliases'}, call_id='reject')
    assert bridge._keys == before


def test_v2_private_seed_table_cannot_create_recoverable_reference(authority):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    bridge = validation_bridge(authority)
    assert bridge._presentation_ledger == {}
    with pytest.raises(EpisodeToolError) as caught:
        bridge.prepare('supersede', {'record_ref': 'r0', 'replacement_ref': 'r0', 'reason': 'hidden'}, call_id='bad')
    assert type(caught.value) is not EpisodeToolValidationRejected
    assert bridge._keys == {} and bridge._prepared == {}


def test_v2_presented_reference_rebind_is_rejected_without_partial_ledger_update(authority):
    from daystrom_dml.contracts.agent_episode import AgentEpisodeError
    bridge = validation_bridge(authority)
    raw, records = retrieved(bridge)
    before = dict(bridge._presentation_ledger)
    altered = deepcopy(records)
    altered[1]['record_ref'] = altered[0]['record_ref']
    with pytest.raises((AgentEpisodeError, EpisodeToolError)):
        bridge._present('retrieve', raw, json.dumps({'records': altered}))
    assert bridge._presentation_ledger == before


def test_v1_self_supersession_retains_original_terminal_adapter_rejection(authority):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    _, bridge, _ = authority
    _, records = retrieved(bridge)
    ref = records[0]['record_ref']
    prepared = bridge.prepare('supersede', {'record_ref': ref, 'replacement_ref': ref, 'reason': 'same'}, call_id='bad')
    with pytest.raises(ValueError) as caught:
        bridge.execute(prepared)
    assert type(caught.value) is not EpisodeToolValidationRejected
    assert 'episode:tool-test:bad' in bridge._keys


@pytest.mark.parametrize('shown', ['none', 'one'])
def test_v2_distinct_private_references_are_terminal_without_keys_or_dispatch(authority, monkeypatch, shown):
    from daystrom_dml.services.episode_tools import EpisodeToolValidationRejected
    adapter, _, _ = authority
    bridge = validation_bridge(authority)
    if shown == 'one':
        invoke(bridge, 'retrieve', {'query': 'owner answers', 'top_k': 1}, 'read-one')
        assert len(bridge._presentation_ledger) == 1
    before = dict(bridge._prepared)
    monkeypatch.setattr(adapter, 'supersede_memory_receipted', lambda *a, **kw: pytest.fail('No dispatch'))
    with pytest.raises(EpisodeToolError) as caught:
        bridge.prepare('supersede', {'record_ref': 'r0', 'replacement_ref': 'r1', 'reason': 'private'}, call_id='bad')
    assert type(caught.value) is not EpisodeToolValidationRejected
    assert bridge._keys == {} and bridge._prepared == before
