"""Ordered reference-consumer delivery, retry, restart and isolation oracles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import errno
import json
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from daystrom_dml.journal import JournalIntegrityError, JournalStateStore
from daystrom_dml.services.outbox_delivery import (
    OutboxDeliveryError, SQLiteOutboxConsumer, deliver_outbox, outbox_status,
)
from scripts.dml_outbox import main


@pytest.fixture
def pair(tmp_path):
    source = JournalStateStore(tmp_path / "source" / "journal.sqlite", receipt_mode=True, outbox_mode=True)
    consumer = SQLiteOutboxConsumer(tmp_path / "consumer" / "journal.sqlite")
    return source, consumer


def save(source, value):
    source.save({"schema_version": 1, "items": [], "lineage": [], "value": value},
                expected_revision=source.read_snapshot()[0], operation="test-transition")


class Proxy:
    def __init__(self, consumer, apply=None, read=None):
        self.path = consumer.path
        self.apply_event = apply or consumer.apply_event
        self.read = read or consumer.read


def test_empty_authority_is_current_but_unbound(pair):
    source, consumer = pair
    result = deliver_outbox(source, consumer)
    assert result["delivered_count"] == 0
    assert result["backlog"] == 0 and result["matches_observed_source"]
    assert result["bound"] is False and result["consumer_cursor"] is None


def test_bounded_pages_preserve_intermediate_states_and_restart(pair):
    source, consumer = pair
    for value in ("ingested", "updated", "deleted"):
        save(source, value)
    events = source.outbox_events()["events"]
    first = deliver_outbox(source, consumer, limit=1)
    assert first["delivered_count"] == 1 and first["backlog"] == 2
    assert consumer.read()["last_event"]["state"]["value"] == "ingested"
    consumer = SQLiteOutboxConsumer(consumer.path)
    second = deliver_outbox(source, consumer, limit=1)
    assert second["backlog"] == 1
    assert consumer.read()["last_event"]["state"]["value"] == "updated"
    third = deliver_outbox(source, consumer)
    assert third["backlog"] == 0 and third["matches_observed_source"]
    assert consumer.read()["last_event"]["state"]["value"] == "deleted"
    assert consumer.read()["event_checksums"] == [event["checksum"] for event in events]


def test_duplicate_old_event_is_acknowledged_without_rollback_or_write(pair):
    source, consumer = pair
    save(source, "old")
    save(source, "new")
    events = source.outbox_events()["events"]
    deliver_outbox(source, consumer)
    before = consumer.journal.verified_snapshot()
    assert consumer.apply_event(events[0])["source_revision"] == 1
    assert consumer.journal.verified_snapshot() == before


def test_gap_and_foreign_authority_refused(pair, tmp_path):
    source, consumer = pair
    save(source, 1)
    save(source, 2)
    events = source.outbox_events()["events"]
    with pytest.raises(OutboxDeliveryError):
        consumer.apply_event(events[1])
    assert consumer.read()["cursor"] is None
    consumer.apply_event(events[0])
    foreign = JournalStateStore(tmp_path / "other" / "journal.sqlite", receipt_mode=True, outbox_mode=True)
    save(foreign, 3)
    with pytest.raises(OutboxDeliveryError):
        consumer.apply_event(foreign.outbox_events()["events"][0])
    with pytest.raises(OutboxDeliveryError):
        outbox_status(foreign, consumer)


def test_noop_reconcile_does_not_write_consumer(pair):
    source, consumer = pair
    save(source, 1)
    deliver_outbox(source, consumer)
    before = consumer.journal.verified_snapshot()
    assert deliver_outbox(source, consumer)["delivered_count"] == 0
    assert consumer.journal.verified_snapshot() == before


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1001, 1.0, "1", None])
def test_invalid_limits_do_not_touch_backend(pair, limit):
    source, consumer = pair
    def forbidden():
        pytest.fail("Invalid request reached backend")
    with pytest.raises(ValueError):
        deliver_outbox(source, Proxy(consumer, read=forbidden), limit=limit)


def test_lost_transport_ack_recovers_from_durable_event(pair):
    source, consumer = pair
    save(source, "committed")
    def lost_ack(event):
        consumer.apply_event(event)
        raise ConnectionError("private backend endpoint")
    assert deliver_outbox(source, Proxy(consumer, apply=lost_ack))["backlog"] == 0


def test_lost_internal_postcommit_ack_recovers(pair):
    source, consumer = pair
    save(source, "committed")
    def fail(point):
        if point == "after_commit":
            raise OSError("lost acknowledgement")
    consumer.journal._fault_hook = fail
    assert deliver_outbox(source, consumer)["backlog"] == 0


def test_precommit_disk_full_keeps_source_and_consumer_retryable(pair):
    source, consumer = pair
    save(source, "durable authority")
    source_before = source.verified_snapshot()
    consumer_before = consumer.journal.verified_snapshot()
    def fail(point):
        if point == "before_commit":
            raise OSError(errno.ENOSPC, "disk full")
    consumer.journal._fault_hook = fail
    with pytest.raises(OSError):
        deliver_outbox(source, consumer)
    assert source.verified_snapshot() == source_before
    assert consumer.journal.verified_snapshot() == consumer_before
    consumer.journal._fault_hook = lambda _: None
    assert deliver_outbox(source, consumer)["backlog"] == 0


def test_false_ack_does_not_count_as_delivery(pair):
    source, consumer = pair
    save(source, 1)
    def false_ack(event):
        return {"source_store_id": event["source_store_id"], "source_revision": event["source_revision"], "event_checksum": event["checksum"]}
    with pytest.raises(OutboxDeliveryError):
        deliver_outbox(source, Proxy(consumer, apply=false_ack))
    assert consumer.read()["cursor"] is None


def test_wrong_ack_after_commit_is_rejected_but_retry_is_safe(pair):
    source, consumer = pair
    save(source, 1)
    def wrong_ack(event):
        consumer.apply_event(event)
        return {"source_revision": True}
    with pytest.raises(OutboxDeliveryError):
        deliver_outbox(source, Proxy(consumer, apply=wrong_ack))
    assert deliver_outbox(source, consumer)["backlog"] == 0


def test_mutation_of_backend_event_argument_cannot_change_expected_ack(pair):
    source, consumer = pair
    save(source, 1)
    def mutation(event):
        consumer.apply_event(event)
        event["checksum"] = "0" * 64
        return {"source_store_id": event["source_store_id"], "source_revision": event["source_revision"], "event_checksum": event["checksum"]}
    with pytest.raises(OutboxDeliveryError):
        deliver_outbox(source, Proxy(consumer, apply=mutation))


def test_source_progresses_while_backend_blocked_and_reports_lag(pair):
    source, consumer = pair
    save(source, 1)
    entered, release = threading.Event(), threading.Event()
    def blocked(event):
        entered.set()
        assert release.wait(10)
        return consumer.apply_event(event)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(deliver_outbox, source, Proxy(consumer, apply=blocked))
        try:
            assert entered.wait(10)
            save(source, 2)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result["page_head_revision"] == 1
    assert result["observed_head_revision"] == 2 and result["backlog"] == 1


def test_concurrent_duplicate_delivery_commits_each_event_once(pair):
    source, consumer = pair
    for value in range(3):
        save(source, value)
    clients = int(os.environ.get("DML_STRESS_CLIENTS", "32"))
    barrier = threading.Barrier(clients)
    def concurrent_delivery(_):
        barrier.wait(timeout=30)
        return deliver_outbox(source, consumer)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        results = list(pool.map(concurrent_delivery, range(clients)))
    assert all(result["matches_observed_source"] for result in results)
    assert consumer.journal.verified_snapshot()[1] == 4  # init plus three events
    assert len(consumer.read()["event_checksums"]) == 3


def test_eight_processes_deliver_each_operation_once(pair):
    source, consumer = pair
    for value in range(3):
        save(source, value)
    code = """
import sys
from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_delivery import SQLiteOutboxConsumer, deliver_outbox
source = JournalStateStore(sys.argv[1])
consumer = SQLiteOutboxConsumer(sys.argv[2])
assert deliver_outbox(source, consumer)['backlog'] == 0
"""
    children = []
    try:
        for _ in range(8):
            children.append(subprocess.Popen([sys.executable, "-c", code, str(source.path), str(consumer.path)],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            output, error = child.communicate(timeout=45)
            assert child.returncode == 0, output + error
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert consumer.journal.verified_snapshot()[1] == 4
    assert len(consumer.read()["event_checksums"]) == 3


def test_existing_unrelated_empty_journal_is_not_adopted(tmp_path):
    path = tmp_path / "generic.sqlite"
    unrelated = JournalStateStore(path)
    assert unrelated.verified_snapshot()[1] == 0
    with pytest.raises(OutboxDeliveryError):
        SQLiteOutboxConsumer(path)
    assert unrelated.verified_snapshot()[1] == 0


def test_live_consumer_rejects_replaced_database_identity(pair):
    source, consumer = pair
    save(source, 1)
    deliver_outbox(source, consumer)
    with closing(sqlite3.connect(consumer.path)) as connection:
        with connection:
            connection.execute("UPDATE identity SET store_id=?", ("a" * 32,))
    with pytest.raises(JournalIntegrityError):
        consumer.read()
    with pytest.raises(JournalIntegrityError):
        consumer.apply_event(source.outbox_events()["events"][0])


def test_corrupt_consumer_and_missing_initialized_consumer_fail_closed(pair):
    source, consumer = pair
    save(source, 1)
    deliver_outbox(source, consumer)
    with closing(sqlite3.connect(consumer.path)) as connection:
        with connection:
            connection.execute("UPDATE state SET checksum=?", ("0" * 64,))
    with pytest.raises(JournalIntegrityError):
        deliver_outbox(source, consumer)
    consumer.path.unlink()
    with pytest.raises(JournalIntegrityError):
        SQLiteOutboxConsumer(consumer.path)


def test_read_returns_detached_state(pair):
    source, consumer = pair
    save(source, {"nested": [1]})
    deliver_outbox(source, consumer)
    value = consumer.read()
    value["last_event"]["state"]["value"]["nested"].append(2)
    value["event_checksums"].clear()
    assert consumer.read()["last_event"]["state"]["value"] == {"nested": [1]}


def test_status_rejects_consumer_ahead_of_restored_authority(pair):
    source, consumer = pair
    save(source, 1)
    save(source, 2)
    deliver_outbox(source, consumer)
    # A source page interface claiming an older head cannot pass cursor checks.
    real = source.outbox_events
    def old_page(**kwargs):
        page = real(**kwargs)
        return {**page, "head_revision": 1}
    source.outbox_events = old_page
    with pytest.raises(OutboxDeliveryError):
        outbox_status(source, consumer)


def test_same_directory_rejected_before_backend_read(pair):
    source, consumer = pair
    backend = Proxy(consumer)
    backend.path = source.path.parent / "other.sqlite"
    backend.read = lambda: pytest.fail("Backend touched before directory guard")
    with pytest.raises(OutboxDeliveryError):
        deliver_outbox(source, backend)


def test_cli_sync_and_status(pair, capsys):
    source, consumer = pair
    save(source, 1)
    assert main([str(source.path), str(consumer.path), "sync", "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["delivered_count"] == 1
    assert main([str(source.path), str(consumer.path), "status"]) == 0
    assert json.loads(capsys.readouterr().out)["backlog"] == 0


def test_cli_status_does_not_create_target_and_sanitizes_errors(pair, tmp_path, capsys):
    source, _ = pair
    target = tmp_path / "absent" / "private.sqlite"
    assert main([str(source.path), str(target), "status"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {"ok": False, "error": "FileNotFoundError"}
    assert not target.parent.exists()


def test_cli_invalid_source_schema_does_not_create_target(tmp_path, capsys):
    source = JournalStateStore(tmp_path / "source.sqlite", receipt_mode=True)
    target = tmp_path / "consumer" / "target.sqlite"
    assert main([str(source.path), str(target), "sync"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "ValueError"
    assert not target.parent.exists()


def test_last_event_corruption_is_rejected_even_with_valid_outer_journal(pair):
    source, consumer = pair
    save(source, 1)
    deliver_outbox(source, consumer)
    revision, envelope = consumer.journal.read_snapshot()
    corrupted = deepcopy(envelope)
    corrupted["last_event"]["state"]["value"] = 2
    consumer.journal.save(corrupted, expected_revision=revision)
    with pytest.raises(JournalIntegrityError):
        consumer.read()
