"""Independent authority oracles for ordered transactional delivery."""
from __future__ import annotations

import copy

import pytest

from daystrom_dml.journal import JournalStateStore
from daystrom_dml.services.outbox_delivery import (
    OutboxDeliveryError, SQLiteOutboxConsumer, deliver_outbox, outbox_status,
)


@pytest.fixture
def histories(tmp_path):
    source = JournalStateStore(tmp_path / "source" / "source.sqlite3",
                               receipt_mode=True, outbox_mode=True)
    source.save({"items": [{"id": 1, "text": "original"}]}, expected_revision=0)
    source.save({"items": [{"id": 1, "text": "updated"}]}, expected_revision=1)
    target = SQLiteOutboxConsumer(tmp_path / "consumer" / "consumer.sqlite3")
    deliver_outbox(source, target)
    return source, target


class ReadProxy:
    def __init__(self, target, value):
        self.path = target.path
        self.value = value
        self.applied = []

    def read(self):
        return copy.deepcopy(self.value)

    def apply_event(self, event):
        self.applied.append(event)
        raise AssertionError("An invalid prefix must be rejected before delivery")


@pytest.mark.parametrize("operation", [outbox_status, deliver_outbox])
def test_consumer_with_false_historical_checksum_cannot_claim_current(histories, operation):
    source, target = histories
    envelope = target.read()
    envelope["event_checksums"][0] = "0" * 64
    proxy = ReadProxy(target, envelope)
    with pytest.raises(OutboxDeliveryError):
        operation(source, proxy)
    assert proxy.applied == []


@pytest.mark.parametrize("operation", [outbox_status, deliver_outbox])
def test_self_consistent_persisted_wrong_prefix_is_checked_against_authority(histories, operation):
    source, target = histories
    envelope = target.read()
    envelope["event_checksums"][0] = "0" * 64
    revision, _ = target.journal.read_snapshot()
    target.journal.save(envelope, expected_revision=revision, operation="untrusted-import")
    reopened = SQLiteOutboxConsumer(target.path)
    with pytest.raises(OutboxDeliveryError):
        operation(source, reopened)
