"""Ordered, replayable journal delivery to a local transactional consumer.

The consumer commits state and its event ledger together. It does not execute
external side effects or promise exactly-once behavior for remote callbacks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol

from ..journal import JournalStateStore, RevisionConflict
from ..store_lock import store_write_lock
from .journal_outbox import validate_outbox_event
from .receipt_ingestion import _strict_json

FORMAT = "dml-sqlite-outbox-consumer-v1"


class OutboxDeliveryError(ValueError):
    """Delivery cannot establish a valid ordered durable outcome."""


class OutboxConsumer(Protocol):
    path: Path

    def read(self) -> dict: ...

    def apply_event(self, event: dict) -> dict: ...


def _encode(value) -> str:
    _strict_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ack(event: dict) -> dict:
    return {"source_store_id": event["source_store_id"],
            "source_revision": event["source_revision"], "event_checksum": event["checksum"]}


def _empty() -> dict:
    return {"schema_version": 1, "consumer_format": FORMAT, "cursor": None,
            "event_checksums": [], "last_event": None, "items": [], "lineage": []}


def validate_consumer(value: dict) -> dict:
    """Freeze and check the complete consumer envelope, including its last state."""
    value = json.loads(_encode(value))
    if type(value) is not dict or set(value) != set(_empty()):
        raise OutboxDeliveryError("Invalid consumer envelope")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["consumer_format"] != FORMAT:
        raise OutboxDeliveryError("Unsupported consumer format")
    if value["items"] != [] or value["lineage"] != []:
        raise OutboxDeliveryError("Consumer records must be inside the event state")
    checksums = value["event_checksums"]
    if type(checksums) is not list or any(type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in checksums):
        raise OutboxDeliveryError("Invalid consumer event ledger")
    if value["cursor"] is None:
        if checksums or value["last_event"] is not None:
            raise OutboxDeliveryError("Invalid unbound consumer")
    else:
        event = validate_outbox_event(value["last_event"])
        if _encode(value["cursor"]) != _encode(_ack(event)) or len(checksums) != event["source_revision"] or checksums[-1] != event["checksum"]:
            raise OutboxDeliveryError("Consumer state and ledger disagree")
    return value


def _accepted(envelope: dict, event: dict) -> bool:
    cursor = envelope["cursor"]
    if cursor is None or cursor["source_store_id"] != event["source_store_id"] or cursor["source_revision"] < event["source_revision"]:
        return False
    if envelope["event_checksums"][event["source_revision"] - 1] != event["checksum"]:
        raise OutboxDeliveryError("Conflicting event at a previously accepted revision")
    if cursor["source_revision"] == event["source_revision"] and _encode(envelope["last_event"]) != _encode(event):
        raise OutboxDeliveryError("Consumer contents disagree with accepted event")
    return True


class SQLiteOutboxConsumer:
    def __init__(self, path: str | Path, *, fault_hook: Callable[[str], None] | None = None):
        self.path = Path(path).resolve()
        self.fault_hook = fault_hook or (lambda _point: None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with store_write_lock(self.path.parent, operation="outbox-consumer-initialize"):
            existed = self.path.exists()
            self.journal = JournalStateStore(self.path, fault_hook=self.fault_hook)
            if self.journal.schema_version != 1:
                raise OutboxDeliveryError("Consumer requires journal schema 1")
            _, revision, envelope = self.journal.verified_snapshot()
            if not existed:
                self.journal.save(_empty(), expected_revision=revision, operation="outbox-consumer-initialize")
            else:
                validate_consumer(envelope)

    def read(self) -> dict:
        if self.journal.schema_version != 1:
            raise OutboxDeliveryError("Unsupported consumer journal schema")
        return validate_consumer(self.journal.verified_snapshot()[2])

    def apply_event(self, event: dict) -> dict:
        event = validate_outbox_event(event)
        desired_ack = _ack(event)
        self.fault_hook("consumer_before_apply")
        for _ in range(16):
            _, revision, current = self.journal.verified_snapshot()
            current = validate_consumer(current)
            cursor = current["cursor"]
            if cursor is not None and cursor["source_store_id"] != event["source_store_id"]:
                raise OutboxDeliveryError("Consumer is bound to another authority")
            if _accepted(current, event):
                return desired_ack
            if event["source_revision"] != len(current["event_checksums"]) + 1:
                raise OutboxDeliveryError("Delivery would skip a source revision")
            proposed = {**current, "cursor": desired_ack, "last_event": event,
                        "event_checksums": [*current["event_checksums"], event["checksum"]]}
            try:
                self.journal.save(proposed, expected_revision=revision, operation="outbox-consumer-apply")
                self.fault_hook("consumer_after_apply")
                return desired_ack
            except RevisionConflict:
                continue
            except Exception:
                # A lost post-commit acknowledgement is successful only when
                # checked durable state proves this exact event was committed.
                if _accepted(self.read(), event):
                    return desired_ack
                raise
        raise RevisionConflict("Concurrent outbox delivery did not settle")


def _guard(source: JournalStateStore, consumer: OutboxConsumer) -> None:
    if source.path.resolve().parent == Path(consumer.path).resolve().parent:
        raise OutboxDeliveryError("Authority and consumer require separate directories")
    if not source.path.is_file() or source.schema_version != 3:
        raise OutboxDeliveryError("An existing schema-3 authority is required")


def _checked_position(source: JournalStateStore, consumer: OutboxConsumer) -> tuple[dict, dict]:
    current = validate_consumer(consumer.read())
    cursor = current["cursor"]
    revision = 0 if cursor is None else cursor["source_revision"]
    page = source.outbox_events(after_revision=max(0, revision - 1), limit=1)
    prefix = current["event_checksums"][:max(0, revision - 1)]
    if hashlib.sha256(_encode(prefix).encode("utf-8")).hexdigest() != page["prefix_digest"]:
        raise OutboxDeliveryError("Consumer history differs from the authoritative prefix")
    if cursor is not None:
        if cursor["source_store_id"] != page["source_store_id"] or revision > page["head_revision"] or not page["events"]:
            raise OutboxDeliveryError("Consumer cursor is not in the authoritative history")
        event = validate_outbox_event(page["events"][0])
        if event["source_revision"] != revision or not _accepted(current, event):
            raise OutboxDeliveryError("Consumer does not match the authoritative event")
    return current, page


def _report(current: dict, page: dict) -> dict:
    cursor = current["cursor"]
    revision = 0 if cursor is None else cursor["source_revision"]
    return {"source_store_id": page["source_store_id"],
            "observed_head_revision": page["head_revision"], "consumer_cursor": cursor,
            "backlog": page["head_revision"] - revision,
            "matches_observed_source": page["head_revision"] == revision,
            "bound": cursor is not None}


def outbox_status(source: JournalStateStore, consumer: OutboxConsumer) -> dict:
    """Report a pinned observation; concurrent commits may immediately add lag."""
    _guard(source, consumer)
    return _report(*_checked_position(source, consumer))


def deliver_outbox(source: JournalStateStore, consumer: OutboxConsumer, *, limit: int = 100) -> dict:
    """Deliver at most one bounded source page, in revision order, with readback."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Delivery limit must be an integer between 1 and 1000")
    _guard(source, consumer)
    initial, _ = _checked_position(source, consumer)
    revision = 0 if initial["cursor"] is None else initial["cursor"]["source_revision"]
    page = source.outbox_events(after_revision=revision, limit=limit)
    events = [validate_outbox_event(event) for event in page["events"]]
    for event in events:
        expected = _ack(event)
        frozen = json.loads(_encode(event))
        try:
            acknowledgement = consumer.apply_event(event)
        except Exception:
            # The original exception remains the outcome if no durable commit
            # can be established. Never retry a side effect within this call.
            if not _accepted(validate_consumer(consumer.read()), frozen):
                raise
        else:
            if _encode(acknowledgement) != _encode(expected):
                raise OutboxDeliveryError("Consumer returned a mismatching acknowledgement")
        observed = validate_consumer(consumer.read())
        if observed["event_checksums"][:len(initial["event_checksums"])] != initial["event_checksums"] or not _accepted(observed, frozen):
            raise OutboxDeliveryError("Consumer did not durably accept the delivered event")
        initial = observed
    final, observed_page = _checked_position(source, consumer)
    if final["event_checksums"][:len(initial["event_checksums"])] != initial["event_checksums"]:
        raise OutboxDeliveryError("Consumer history regressed after delivery")
    return {**_report(final, observed_page), "delivered_count": len(events),
            "page_head_revision": page["head_revision"]}
