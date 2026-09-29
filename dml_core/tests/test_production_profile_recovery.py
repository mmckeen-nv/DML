"""Actual-profile deterministic process death; no device/power-loss claim."""
from __future__ import annotations

import os

import pytest

from daystrom_dml.journal import OUTBOX_FAULT_POINTS, RECEIPT_FAULT_POINTS
from profile_crash_fixture import (
    COMMITTED_POINTS, OPERATIONS, OUTBOX_POINTS, RECEIPT_POINTS, SCHEMAS,
    assert_recovered, expected_transition, prepare, request_for, run_interrupted,
)


CASES = [(schema, operation, point) for schema in SCHEMAS for operation in OPERATIONS
         for point in (*(RECEIPT_POINTS if schema == 2 else OUTBOX_POINTS),
                       *(("during_embedding",) if operation in ("ingest", "update", "promote") else ()),
                       "before_hydration", "before_response", "after_ack")]


@pytest.fixture(autouse=True)
def isolated_profile_environment(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("DML_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


def test_explicit_profile_mutation_inventory_matches_runtime_hooks():
    assert RECEIPT_POINTS == RECEIPT_FAULT_POINTS
    assert OUTBOX_POINTS == OUTBOX_FAULT_POINTS
    assert len(CASES) == 184


@pytest.mark.parametrize("schema,operation,point", CASES,
                         ids=[f"schema-{schema}-{operation}-{point}" for schema, operation, point in CASES])
def test_profile_process_death_preserves_parent_history(tmp_path, schema, operation, point):
    scenario = prepare(tmp_path / "authority", schema)
    request = request_for(operation, scenario.records)
    expected = expected_transition(scenario.before, request)
    acknowledgement = run_interrupted(scenario, request, point)
    # Absence of a returned receipt is uncertain to the caller, even if test
    # instrumentation knows the kill occurred before commit. Recovery resolves it.
    outcome = "acknowledged" if acknowledgement is not None else "uncertain"
    assert outcome == ("acknowledged" if point == "after_ack" else "uncertain")
    if acknowledgement is not None:
        assert acknowledgement == expected["receipt"]
    receipt = assert_recovered(scenario, request, committed=point in COMMITTED_POINTS)
    assert receipt == expected["receipt"]
