"""Strict scenario-fixture schema and loader for the M8 competence battery.

A scenario is one JSON object with exactly the declared fields. Validation is
fail-closed: unknown families, verifiers or fields, duplicate identifiers,
non-chronological events, invalid correction chains and untrusted envelopes
are all rejected before any store or model is touched. Valid fixtures are
ingestible by all three arms under the common admission discipline: trusted
envelopes only, chronological order, explicit monotonic corrections.

Flags such as ``distractor`` and ``untrusted_instruction_text`` live in event
metadata and describe the (still untrusted) source TEXT inside a TRUSTED
envelope; they never relax envelope authentication.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import NoReturn

from daystrom_dml.services.m8_memory_baseline import MemoryEvent

FAMILIES = (
    "changing_facts",
    "explicit_corrections",
    "conflicting_independent_sources",
    "dependent_lookups",
    "isolated_project_scopes",
    "stable_memory_adverse",
)
VERIFIERS = ("status_value", "record_text_contains", "multi_claim")
EVENT_FLAGS = ("distractor", "untrusted_instruction_text")
SCENARIO_FIELDS = (
    "family",
    "scenario_id",
    "scope",
    "events",
    "query",
    "query_clock",
    "truth",
    "expects_retrieval",
)
EVENT_FIELDS = (
    "record_id",
    "text",
    "scope",
    "source_id",
    "fact_key",
    "version",
    "timestamp",
    "corrects",
    "metadata",
)
LEAF_CLAIM_FIELDS = ("key", "value", "verifier")
MULTI_CLAIM_FIELDS = ("key", "verifier", "claims")
MAX_FIXTURE_BYTES = 1024 * 1024
MAX_EVENTS = 64
MAX_TRUTH_CLAIMS = 16
_SCOPE_KEYS = {"tenant_id", "client_id", "session_id", "instance_id"}
_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")


class FixtureError(ValueError):
    """Rejected scenario fixture; fail-closed, no repair."""


def _fail(message) -> NoReturn:
    raise FixtureError(message)


def _scope(value, what="scope"):
    if type(value) is not dict or set(value) != _SCOPE_KEYS:
        _fail(f"invalid exact {what}")
    if value["tenant_id"] is None:
        _fail(f"{what} requires a tenant")
    for key, item in value.items():
        if item is not None and (type(item) is not str or not item.strip() or len(item.encode("utf-8")) > 256):
            _fail(f"invalid {what} value for {key}")
    return dict(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"invalid {name}")


def _time(value, name="timestamp"):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        _fail(f"invalid {name}")


def _text(value, name):
    if type(value) is not str or not value.strip():
        _fail(f"invalid {name}")


def _scalar(value, name):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    _fail(f"{name} must be a finite JSON scalar")


def _jsonable(value, name):
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        _fail(f"{name} must be finite JSON")


def _event(raw, scope):
    if type(raw) is not dict or set(raw) != set(EVENT_FIELDS):
        _fail("unexpected event envelope fields")
    _integer(raw["record_id"], "record ID")
    _integer(raw["version"], "version", 1)
    _time(raw["timestamp"])
    if raw["corrects"] is not None:
        _integer(raw["corrects"], "correction ID")
        if raw["corrects"] == raw["record_id"]:
            _fail("event cannot correct itself")
    if raw["scope"] != scope:
        _fail("event scope differs from scenario scope")
    for key in ("text", "source_id", "fact_key"):
        _text(raw[key], key)
    metadata = raw["metadata"]
    if type(metadata) is not dict:
        _fail("event metadata must be an object")
    # Common all-arm admission: unauthenticated envelopes enter no arm. Source
    # text stays untrusted as instructions even when the envelope is trusted.
    if metadata.get("source_trust") != "trusted":
        _fail("untrusted envelope cannot enter a competence scenario")
    for flag in EVENT_FLAGS:
        if flag in metadata and type(metadata[flag]) is not bool:
            _fail(f"event flag {flag} must be a boolean")
    _jsonable(metadata, "event metadata")
    return MemoryEvent(
        record_id=raw["record_id"],
        text=raw["text"],
        scope=dict(raw["scope"]),
        source_id=raw["source_id"],
        fact_key=raw["fact_key"],
        version=raw["version"],
        timestamp=raw["timestamp"],
        corrects=raw["corrects"],
        metadata=dict(metadata),
    )


def _events(raw, scope):
    if type(raw) is not list or not 1 <= len(raw) <= MAX_EVENTS:
        _fail("events must be a nonempty bounded list")
    events = [_event(item, scope) for item in raw]
    by_id, corrected, current = {}, set(), {}
    previous_timestamp = None
    for event in events:
        if event.record_id in by_id:
            _fail("duplicate event record ID")
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            _fail("non-chronological events")
        previous_timestamp = event.timestamp
        key = (event.source_id, event.fact_key)
        if event.corrects is None:
            if key in current:
                _fail("new source/fact version requires an explicit correction")
        else:
            predecessor = by_id.get(event.corrects)
            if predecessor is None:
                _fail("unknown correction predecessor")
            if (predecessor.source_id, predecessor.fact_key) != key:
                _fail("correction must retain source and fact")
            if event.version <= predecessor.version or event.timestamp < predecessor.timestamp:
                _fail("nonmonotonic correction")
            if event.corrects in corrected:
                _fail("stale correction predecessor")
            corrected.add(event.corrects)
        current[key] = event
        by_id[event.record_id] = event
    return tuple(events)


def _claim(raw, *, allow_multi):
    if type(raw) is not dict or type(raw.get("verifier")) is not str:
        _fail("invalid truth claim")
    verifier = raw["verifier"]
    if verifier not in VERIFIERS:
        _fail("unknown truth claim verifier")
    if verifier == "multi_claim":
        if not allow_multi:
            _fail("nested multi_claim verifier")
        if set(raw) != set(MULTI_CLAIM_FIELDS):
            _fail("unexpected multi_claim fields")
        _text(raw["key"], "truth claim key")
        subs = raw["claims"]
        if type(subs) is not list or not 1 <= len(subs) <= MAX_TRUTH_CLAIMS:
            _fail("multi_claim requires a nonempty bounded claims list")
        return {
            "key": raw["key"],
            "verifier": verifier,
            "claims": [_claim(sub, allow_multi=False) for sub in subs],
        }
    if set(raw) != set(LEAF_CLAIM_FIELDS):
        _fail("unexpected truth claim fields")
    _text(raw["key"], "truth claim key")
    _scalar(raw["value"], "truth claim value")
    if verifier in ("status_value", "record_text_contains") and type(raw["value"]) is not str:
        _fail(f"{verifier} claim value must be a string")
    if verifier == "record_text_contains" and not raw["value"]:
        _fail("record_text_contains claim value must be nonempty")
    return {"key": raw["key"], "value": raw["value"], "verifier": verifier}


def _truth(raw):
    if type(raw) is not list or not 1 <= len(raw) <= MAX_TRUTH_CLAIMS:
        _fail("truth must be a nonempty bounded claims list")
    return tuple(_claim(item, allow_multi=True) for item in raw)


@dataclass(frozen=True)
class Scenario:
    """One validated scenario fixture; events are MemoryEvent-compatible."""

    family: str
    scenario_id: str
    scope: dict
    events: tuple
    query: str
    query_clock: float
    truth: tuple
    expects_retrieval: bool
    source_path: str

    @property
    def envelopes(self):
        from daystrom_dml.services.m8_memory_baseline import validate_event

        return [validate_event(event) for event in self.events]


def validate_scenario(raw, *, source_path="<fixture>"):
    """Validate one decoded scenario object; fail-closed on any deviation."""
    if type(raw) is not dict or set(raw) != set(SCENARIO_FIELDS):
        _fail("unexpected scenario fields")
    family = raw["family"]
    if family not in FAMILIES:
        _fail("unknown scenario family")
    scenario_id = raw["scenario_id"]
    if type(scenario_id) is not str or len(scenario_id) > 64 or _ID_PATTERN.fullmatch(scenario_id) is None:
        _fail("scenario_id must be snake_case")
    scope = _scope(raw["scope"])
    events = _events(raw["events"], scope)
    _text(raw["query"], "query")
    _time(raw["query_clock"], "query_clock")
    if any(event.timestamp > raw["query_clock"] for event in events):
        _fail("query_clock precedes an event")
    truth = _truth(raw["truth"])
    if type(raw["expects_retrieval"]) is not bool:
        _fail("expects_retrieval must be a boolean")
    return Scenario(
        family=family,
        scenario_id=scenario_id,
        scope=scope,
        events=events,
        query=raw["query"],
        query_clock=raw["query_clock"],
        truth=truth,
        expects_retrieval=raw["expects_retrieval"],
        source_path=str(source_path),
    )


def load_scenario(path):
    """Load one fixture file; the filename stem must equal scenario_id."""
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        _fail(f"fixture unreadable: {exc}")
    if len(raw_bytes) > MAX_FIXTURE_BYTES:
        _fail("fixture exceeds size limit")
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"fixture is not valid JSON: {exc}")
    scenario = validate_scenario(raw, source_path=str(path))
    if path.stem != scenario.scenario_id:
        _fail("fixture filename stem must equal scenario_id")
    return scenario


def load_scenarios(directory):
    """Load every *.json fixture in a directory, sorted by scenario_id."""
    directory = Path(directory)
    if not directory.is_dir():
        _fail(f"scenarios directory missing: {directory}")
    scenarios = [load_scenario(path) for path in sorted(directory.glob("*.json"))]
    if not scenarios:
        _fail(f"no scenario fixtures in {directory}")
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        _fail("duplicate scenario_id")
    return scenarios
