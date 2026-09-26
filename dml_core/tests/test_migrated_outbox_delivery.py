"""Migrated history has one honest baseline, then ordered operation delivery."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import threading

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.outbox_delivery import (
    MIGRATED_FORMAT, OutboxDeliveryError, SQLiteOutboxConsumer,
    deliver_outbox, outbox_status, validate_consumer,
)
from daystrom_dml.services.outbox_migration import upgrade_outbox_journal


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resign(event):
    event["checksum"] = digest({key: value for key, value in event.items() if key != "checksum"})
    return event


def save(source, value):
    source.save({"schema_version": 1, "items": [], "lineage": [], "value": value},
                expected_revision=source.read_snapshot()[0], operation="test-transition")


class Proxy:
    def __init__(self, consumer, apply=None, read=None):
        self.path = consumer.path
        self.apply_event = apply or consumer.apply_event
        self.read = read or consumer.read


@pytest.fixture(params=[0, 3], ids=["empty-legacy", "legacy-history"])
def pair(tmp_path, request):
    old = JournalStateStore(tmp_path / "legacy" / "journal.sqlite", receipt_mode=True)
    for revision in range(request.param):
        save(old, revision)
    upgrade_outbox_journal(old.path, tmp_path / "migrated" / "journal.sqlite")
    source = JournalStateStore(tmp_path / "migrated" / "journal.sqlite")
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "journal.sqlite")
    return source, consumer, old


def test_status_reports_only_deliverable_history_and_baseline(pair):
    source, consumer, old = pair
    before = consumer.journal.verified_snapshot()
    report = outbox_status(source, consumer)
    assert report["backlog"] == 1 and not report["bound"]
    assert not report["matches_observed_source"]
    assert report["history_coverage"]["first_deliverable_revision"] == old.read_snapshot()[0] + 1
    assert report["history_coverage"]["legacy_operations_delivered"] is False
    assert report["history_coverage"]["origin"] == source.outbox_events()["origin"]
    assert consumer.journal.verified_snapshot() == before
    delivered = deliver_outbox(source, consumer)
    envelope = consumer.read()
    assert delivered["delivered_count"] == 1 and delivered["backlog"] == 0
    assert envelope["schema_version"] == 1 and envelope["consumer_format"] == MIGRATED_FORMAT
    assert envelope["last_event"]["state"] == old.load()
    assert envelope["origin"] == report["history_coverage"]["origin"]
    assert len(envelope["event_checksums"]) == 1
    assert SQLiteOutboxConsumer(consumer.path).read() == envelope


def test_pages_updates_deletion_and_exact_baseline_duplicate(pair):
    source, consumer, old = pair
    save(source, {"memory": "updated"})
    save(source, {"memory": None})
    events = source.outbox_events()["events"]
    for index, expected in enumerate(events):
        report = deliver_outbox(source, consumer, limit=1)
        assert report["backlog"] == len(events) - index - 1
        assert consumer.read()["last_event"] == expected
    before = consumer.journal.verified_snapshot()
    assert consumer.apply_event(events[0])["source_revision"] == old.read_snapshot()[0] + 1
    assert consumer.journal.verified_snapshot() == before
    assert deliver_outbox(source, consumer)["delivered_count"] == 0


def test_future_event_cannot_bootstrap_even_with_baseline_operation(pair):
    source, consumer, _ = pair
    save(source, 90)
    event = source.outbox_events()["events"][-1]
    with pytest.raises(OutboxDeliveryError):
        consumer.apply_event(event)
    event["operation"] = "outbox-migration-baseline-v1"
    with pytest.raises(OutboxDeliveryError):
        consumer.apply_event(resign(event))
    assert consumer.read()["cursor"] is None


def test_baseline_state_tampering_cannot_bind_consumer(pair):
    source, consumer, _ = pair
    event = source.outbox_events()["events"][0]
    event["state"]["value"] = "invented"
    event["source_digest"] = digest(event["state"])
    with pytest.raises(JournalIntegrityError):
        consumer.apply_event(resign(event))
    assert consumer.read()["cursor"] is None


def test_foreign_origin_rejected_even_for_same_authority_revision(pair):
    source, consumer, _ = pair
    deliver_outbox(source, consumer)
    event = source.outbox_events()["events"][0]
    event["origin"]["history_digest"] = "a" * 64
    with pytest.raises(OutboxDeliveryError, match="origin"):
        consumer.apply_event(resign(event))


def test_bound_v1_consumer_never_reinterpreted_as_migrated(pair, tmp_path):
    source, consumer, _ = pair
    native = JournalStateStore(tmp_path / "native" / "journal.sqlite", receipt_mode=True, outbox_mode=True)
    save(native, 0)
    deliver_outbox(native, consumer)
    before = consumer.journal.verified_snapshot()
    with pytest.raises(OutboxDeliveryError):
        consumer.apply_event(source.outbox_events()["events"][0])
    with pytest.raises(ValueError):
        deliver_outbox(source, consumer)
    assert consumer.journal.verified_snapshot() == before


def test_v2_consumer_rejects_v1_event_even_with_matching_source_id(pair):
    source, consumer, _ = pair
    deliver_outbox(source, consumer)
    event = source.outbox_events()["events"][0]
    event.pop("origin")
    event.update(schema_version=1, event_format="dml-journal-outbox-v1")
    with pytest.raises(OutboxDeliveryError, match="format"):
        consumer.apply_event(resign(event))


def test_corrupted_earlier_prefix_cannot_hide_behind_correct_last_event(pair):
    source, consumer, _ = pair
    save(source, 91)
    deliver_outbox(source, consumer)
    envelope = consumer.read()
    envelope["event_checksums"][0] = "b" * 64
    # This simulates a backend claiming a consistent latest cursor over bad history.
    with pytest.raises(OutboxDeliveryError, match="prefix"):
        outbox_status(source, Proxy(consumer, read=lambda: envelope))
    with pytest.raises(OutboxDeliveryError, match="prefix"):
        deliver_outbox(source, Proxy(consumer, read=lambda: envelope))


@pytest.mark.parametrize("mutation", ["origin", "format", "count", "unbound"])
def test_consumer_envelope_cannot_disagree_with_event(pair, mutation):
    source, consumer, _ = pair
    deliver_outbox(source, consumer)
    envelope = consumer.read()
    if mutation == "origin":
        envelope["origin"]["history_digest"] = "c" * 64
    elif mutation == "format":
        envelope["consumer_format"] = "dml-sqlite-outbox-consumer-v1"
    elif mutation == "count":
        envelope["event_checksums"].append("d" * 64)
    else:
        envelope.update(cursor=None, event_checksums=[], last_event=None)
    with pytest.raises((OutboxDeliveryError, JournalIntegrityError)):
        validate_consumer(envelope)


def test_baseline_transition_rolls_back_on_precommit_failure(pair):
    source, consumer, _ = pair
    before = consumer.journal.verified_snapshot()
    def fail(point):
        if point == "before_commit":
            raise OSError("disk full")
    consumer.journal._fault_hook = fail
    with pytest.raises(OSError):
        deliver_outbox(source, consumer)
    assert consumer.journal.verified_snapshot() == before
    assert consumer.read()["schema_version"] == 1
    consumer.journal._fault_hook = lambda _: None
    assert deliver_outbox(source, consumer)["backlog"] == 0


def test_lost_ack_and_false_ack_remain_distinguishable(pair):
    source, consumer, _ = pair
    def false_ack(event):
        return {"source_store_id": event["source_store_id"], "source_revision": event["source_revision"], "event_checksum": event["checksum"]}
    with pytest.raises(OutboxDeliveryError):
        deliver_outbox(source, Proxy(consumer, apply=false_ack))
    def lost_ack(event):
        consumer.apply_event(event)
        raise ConnectionError("lost")
    assert deliver_outbox(source, Proxy(consumer, apply=lost_ack))["backlog"] == 0


def test_source_can_progress_during_migrated_delivery(pair):
    source, consumer, _ = pair
    entered, release = threading.Event(), threading.Event()
    def blocked(event):
        entered.set()
        assert release.wait(15)
        return consumer.apply_event(event)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(deliver_outbox, source, Proxy(consumer, apply=blocked))
        try:
            assert entered.wait(15)
            save(source, "progress")
        finally:
            release.set()
        report = future.result(timeout=15)
    assert report["backlog"] == 1 and not report["matches_observed_source"]
    assert deliver_outbox(source, consumer)["backlog"] == 0


def test_concurrent_baseline_binding_and_replay_commits_once(pair):
    source, consumer, _ = pair
    save(source, "post-migration")
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def delivery(_):
        barrier.wait(timeout=30)
        return deliver_outbox(source, consumer)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        reports = list(pool.map(delivery, range(clients)))
    assert all(report["backlog"] == 0 for report in reports)
    assert consumer.journal.verified_snapshot()[1] == 3
    assert len(consumer.read()["event_checksums"]) == 2


def test_deleted_legacy_receipt_is_not_resurrected_or_invented_as_delivery(tmp_path):
    old = JournalStateStore(tmp_path / "legacy" / "journal.sqlite", receipt_mode=True)
    scope = dict(tenant_id="owner", client_id=None, session_id=None, instance_id=None)
    memory = {"schema_version": 1, "id": 7, "text": "deleted private memory", "meta": dict(scope)}
    state = {"schema_version": 1, "items": [memory], "lineage": []}
    receipt = old.save_with_receipt(state, scope=scope, key="old", request_digest="1" * 64,
                                    result={"memory": memory}, expected_revision=0)
    old.save({"schema_version": 1, "items": [], "lineage": []}, expected_revision=1, operation="delete")
    upgrade_outbox_journal(old.path, tmp_path / "new" / "journal.sqlite")
    source = JournalStateStore(tmp_path / "new" / "journal.sqlite")
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "journal.sqlite")
    assert deliver_outbox(source, consumer)["delivered_count"] == 1
    event = consumer.read()["last_event"]
    assert event["receipt"] is None and event["state"]["items"] == []
    assert "deleted private memory" not in json.dumps(consumer.read())
    assert source.lookup_receipt(scope=scope, key="old", request_digest="1" * 64) == receipt


def test_postcommit_baseline_fault_recovers_atomic_format_binding(pair):
    source, consumer, _ = pair
    def fail(point):
        if point == "after_commit":
            raise OSError("lost commit acknowledgement")
    consumer.journal._fault_hook = fail
    assert deliver_outbox(source, consumer)["backlog"] == 0
    assert consumer.read()["consumer_format"] == MIGRATED_FORMAT
    assert consumer.journal.verified_snapshot()[1] == 2


def test_postmigration_records_preserve_scope_and_intermediate_updates(pair):
    source, consumer, _ = pair
    deliver_outbox(source, consumer)
    scope = dict(tenant_id="tenant-A", client_id="client-A", session_id="session-A", instance_id="instance-A")
    memory = {"schema_version": 1, "id": 11, "text": "private preference", "meta": dict(scope)}
    revision, state = source.read_snapshot()
    state["items"] = [memory]
    receipt = source.save_with_receipt(state, scope=scope, key="after-migration", request_digest="2" * 64,
                                      result={"memory": memory}, expected_revision=revision)
    state = source.load()
    state["items"][0]["text"] = "superseded preference"
    source.save(state, expected_revision=source.read_snapshot()[0], operation="update")
    state["items"] = []
    source.save(state, expected_revision=source.read_snapshot()[0], operation="delete")
    deliver_outbox(source, consumer, limit=1)
    event = consumer.read()["last_event"]
    assert event["receipt"]["scope"] == scope
    assert event["receipt"]["digest"] == digest(receipt)
    assert event["state"]["items"] == [memory]
    deliver_outbox(source, consumer, limit=1)
    item = consumer.read()["last_event"]["state"]["items"][0]
    assert item["text"] == "superseded preference" and item["meta"] == scope
    assert deliver_outbox(source, consumer, limit=1)["backlog"] == 0
    assert consumer.read()["last_event"]["state"]["items"] == []
