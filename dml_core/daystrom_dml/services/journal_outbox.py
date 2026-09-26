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


def validate_migration_origin(value: dict) -> dict:
    """Validate the honest boundary before which full-state events do not exist."""
    if not isinstance(value, dict) or set(value) != {"source_schema_version", "source_revision", "source_digest", "history_digest"}:
        raise JournalIntegrityError("Invalid migration origin fields")
    if type(value["source_schema_version"]) is not int or value["source_schema_version"] != 2:
        raise JournalSchemaError("Unsupported migration source schema")
    if type(value["source_revision"]) is not int or value["source_revision"] < 0:
        raise JournalIntegrityError("Invalid migration source revision")
    if not all(_is_digest(value[field]) for field in ("source_digest", "history_digest")):
        raise JournalIntegrityError("Invalid migration origin digest")
    return dict(value)


def validate_outbox_event(value: dict) -> dict:
    """Return a detached, strictly validated version-1 or version-2 event.

    Decision/history binding is additionally verified by ``outbox_events`` on
    the authoritative journal; this function validates the standalone envelope.
    """
    try:
        event = _decode(_encode(value))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise JournalIntegrityError("Invalid outbox event serialization") from exc
    return _validate_owned_outbox_event(event)


def _validate_owned_outbox_event(event: dict) -> dict:
    """Validate one call's private strict-JSON event without copying it again.

    Only the defensive public wrapper or a freshly ``_checked`` SQL decode may
    use this path. No object or validation result is retained between calls.
    """
    fields = {"schema_version", "event_format", "source_store_id", "source_revision",
              "source_digest", "operation", "receipt", "state", "decision_digest", "checksum"}
    if isinstance(event, dict) and type(event.get("schema_version")) is int and event["schema_version"] == 2:
        fields.add("origin")
    if not isinstance(event, dict) or set(event) != fields:
        raise JournalIntegrityError("Invalid outbox event fields")
    version = event["schema_version"]
    if type(version) is not int or version not in (1, 2) or event["event_format"] != (OUTBOX_EVENT_FORMAT if version == 1 else "dml-journal-outbox-v2"):
        raise JournalSchemaError("Unsupported outbox event version or format")
    _valid_identity({"schema_version": 1, "store_id": event["source_store_id"]})
    if type(event["source_revision"]) is not int or event["source_revision"] <= 0:
        raise JournalIntegrityError("Invalid outbox source revision")
    if version == 2:
        origin = validate_migration_origin(event["origin"])
        if event["source_revision"] <= origin["source_revision"]:
            raise JournalIntegrityError("Outbox event precedes migration boundary")
        if event["source_revision"] == origin["source_revision"] + 1 and (event["operation"] != "outbox-migration-baseline-v1" or event["receipt"] is not None or event["source_digest"] != origin["source_digest"]):
            raise JournalIntegrityError("Invalid migration baseline envelope")
    if not isinstance(event["operation"], str) or not event["operation"].strip():
        raise JournalIntegrityError("Invalid outbox operation")
    for key in ("source_digest", "decision_digest", "checksum"):
        if not _is_digest(event[key]):
            raise JournalIntegrityError("Invalid outbox digest")
    if event["checksum"] != _digest(_encode({key: value for key, value in event.items() if key != "checksum"})):
        raise JournalIntegrityError("Outbox event checksum mismatch")
    state = _normalized(event["state"])
    encoded_state = _encode(state)
    if encoded_state != _encode(event["state"]) or _digest(encoded_state) != event["source_digest"]:
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
