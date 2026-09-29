import pytest
from daystrom_dml.services.m8_accounting import BudgetExhausted, DevelopmentAccounting, tokens_per_correct_success


def test_six_attempts_consume_budget_including_repeated_searches():
    ledger = DevelopmentAccounting(clock=lambda: 0)
    for i in range(6):
        ledger.admit(str(i), [1] * 10)
        ledger.complete(str(i), [2] * 20)
    with pytest.raises(BudgetExhausted):
        ledger.admit("seventh", [1])
    assert ledger.snapshot()["input_tokens"] == 60
    assert ledger.snapshot()["output_tokens"] == 120


def test_failure_is_costed_and_no_retry_allowed():
    ledger = DevelopmentAccounting(clock=lambda: 0)
    ledger.admit("a", [1] * 5)
    ledger.complete("a", [2] * 3, error="authentic failure")
    with pytest.raises(BudgetExhausted):
        ledger.admit("retry", [1])
    row = ledger.snapshot()
    assert row["turns"][0]["error"] == "authentic failure"
    assert (
        tokens_per_correct_success([{"correct": False, "accounting": row}, {"correct": True, "accounting": row}]) == 16
    )


def test_unknown_pending_duplicate_and_mutation_cannot_hide_usage():
    ledger = DevelopmentAccounting(clock=lambda: 0)
    admitted = ledger.admit("a", [1, 2])
    admitted["input_ids"].clear()
    assert ledger.snapshot()["turns"][0]["input_ids"] == [1, 2]
    assert tokens_per_correct_success([{"correct": True, "accounting": ledger.snapshot()}]) is None
    with pytest.raises(ValueError):
        ledger.admit("a", [1])
    ledger.complete("a", [2], usage_known=False, error="dispatch outcome unresolved")
    with pytest.raises(BudgetExhausted):
        ledger.admit("b", [1])


def test_request_total_and_wall_boundaries():
    now = [0]
    ledger = DevelopmentAccounting(clock=lambda: now[0])
    with pytest.raises(BudgetExhausted):
        ledger.admit("too_big", [1] * 7937)
    for i in range(4):
        ledger.admit(str(i), [1] * 7936)
        ledger.complete(str(i), [2])
    with pytest.raises(BudgetExhausted):
        ledger.admit("aggregate", [1] * 1025)
    now[0] = 300
    with pytest.raises(BudgetExhausted):
        ledger.admit("time", [1])


def test_late_completion_retained_and_output_overrun_terminal():
    now = [0]
    ledger = DevelopmentAccounting(clock=lambda: now[0])
    ledger.admit("a", [1])
    now[0] = 301
    assert ledger.complete("a", [2])["status"] == "timeout"
    assert ledger.snapshot()["output_tokens"] == 1
    other = DevelopmentAccounting(clock=lambda: 0)
    other.admit("a", [1], reserved_output=1)
    assert other.complete("a", [2, 3])["status"] == "output_overrun"
    with pytest.raises(BudgetExhausted):
        other.admit("b", [1])


def test_embedding_costs_separate_and_zero_success_not_free():
    ledger = DevelopmentAccounting(clock=lambda: 0)
    ledger.record_stage("ingestion", wall_seconds=2, embedding_tokens=99, embedding_calls=1, bytes_written=4096)
    snap = ledger.snapshot()
    assert snap["input_tokens"] == 0
    assert snap["stages"][0]["embedding_tokens"] == 99
    assert snap["stages"][0]["gpu_seconds"] is None
    assert tokens_per_correct_success([{"correct": False, "accounting": snap}]) is None
    with pytest.raises(ValueError):
        ledger.record_stage("retrieval", wall_seconds=float("nan"))
