"""Strict wire validation for immutable, full-state journal outbox events.

Checksums detect corruption and bind the event to its source decision history.
They do not authenticate an arbitrary sender; consumers must verify source pages.
"""
from __future__ import annotations

from ..journal import (
    OUTBOX_EVENT_FORMAT, JournalIntegrityError, JournalSchemaError,
    _decode, _digest, _encode, _is_digest, _normalized, _receipt_identity,
    _valid_identity,
)


def validate_outbox_event(value: dict) -> dict:
    """Return a detached, strictly validated version-1 event.

    Decision/history binding is additionally verified by ``outbox_events`` on
    the authoritative journal; this function validates the standalone envelope.
    """
    try:
        event = _decode(_encode(value))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise JournalIntegrityError("Invalid outbox event serialization") from exc
    fields = {"schema_version", "event_format", "source_store_id", "source_revision",
              "source_digest", "operation", "receipt", "state", "decision_digest", "checksum"}
    if not isinstance(event, dict) or set(event) != fields:
        raise JournalIntegrityError("Invalid outbox event fields")
    if type(event["schema_version"]) is not int or event["schema_version"] != 1 or event["event_format"] != OUTBOX_EVENT_FORMAT:
        raise JournalSchemaError("Unsupported outbox event version or format")
    _valid_identity({"schema_version": 1, "store_id": event["source_store_id"]})
    if type(event["source_revision"]) is not int or event["source_revision"] <= 0:
        raise JournalIntegrityError("Invalid outbox source revision")
    if not isinstance(event["operation"], str) or not event["operation"].strip():
        raise JournalIntegrityError("Invalid outbox operation")
    for key in ("source_digest", "decision_digest", "checksum"):
        if not _is_digest(event[key]):
            raise JournalIntegrityError("Invalid outbox digest")
    if event["checksum"] != _digest(_encode({key: value for key, value in event.items() if key != "checksum"})):
        raise JournalIntegrityError("Outbox event checksum mismatch")
    state = _normalized(event["state"])
    if _encode(state) != _encode(event["state"]) or _digest(_encode(state)) != event["source_digest"]:
        raise JournalIntegrityError("Outbox state digest mismatch")
    binding = event["receipt"]
    if binding is not None:
        if not isinstance(binding, dict) or set(binding) != {"scope", "key", "request_digest", "digest"} or not _is_digest(binding["digest"]):
            raise JournalIntegrityError("Invalid outbox receipt binding")
        try:
            _receipt_identity(binding["scope"], binding["key"], binding["request_digest"])
        except (TypeError, ValueError, UnicodeError) as exc:
            raise JournalIntegrityError("Invalid outbox receipt identity") from exc
    return event
