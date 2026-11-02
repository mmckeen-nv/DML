import os

import pytest
from daystrom_dml.services.m8_accounting import (
    BudgetExhausted,
    DeadlineExceeded,
    DeadlineSupervisor,
    DevelopmentAccounting,
    DurableLedger,
    EpisodeMetering,
    cost_per_correct_success,
    tokens_per_correct_success,
)


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


def test_durable_ledger_records_and_replays_across_restart(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = DurableLedger(path)
    ledger.record("input", {"call_id": "a", "input_ids": [1, 2, 3]})
    ledger.record("step", {"name": "retrieval", "wall_seconds": 0.5, "bytes_written": 4096})
    row = ledger.record("output", {"call_id": "a", "status": "error", "output_ids": [9], "error": "authentic failure"})
    assert row["sequence"] == 2
    replayed = DurableLedger(path)
    assert len(replayed) == 3
    assert replayed.records() == ledger.records()
    assert replayed.records()[2]["payload"]["error"] == "authentic failure"
    again = replayed.record("input", {"call_id": "b", "input_ids": [4]})
    assert again["sequence"] == 3
    assert again["previous"] == row["sha256"]
    assert DurableLedger(path).records()[-1]["payload"]["call_id"] == "b"


def test_durable_ledger_replay_detects_tampering_and_truncation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = DurableLedger(path)
    ledger.record("input", {"call_id": "a", "input_ids": [1]})
    path.write_bytes(path.read_bytes().replace(b"[1]", b"[2]"))
    with pytest.raises(ValueError):
        DurableLedger(path)
    path.write_bytes(path.read_bytes().replace(b"[2]", b"[1]") + b'{"bogus": 1}\n')
    with pytest.raises(ValueError):
        DurableLedger(path)
    path.write_bytes(path.read_bytes() + b'{"version":')
    with pytest.raises(ValueError):
        DurableLedger(path)


def test_durable_ledger_fail_closed_on_bad_record_and_io_failure(tmp_path, monkeypatch):
    ledger = DurableLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        ledger.record("stage", {"name": "x"})
    with pytest.raises(ValueError):
        ledger.record("output", {"call_id": "a", "status": "invented", "output_ids": [], "error": None})
    assert not ledger.failed
    ledger.record("input", {"call_id": "a", "input_ids": [1]})

    def boom(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        ledger.record("input", {"call_id": "b", "input_ids": [2]})
    assert ledger.failed
    with pytest.raises(ValueError):
        ledger.record("input", {"call_id": "c", "input_ids": [3]})


def test_durable_ledger_records_full_episode_failure_inclusive(tmp_path):
    accounting = DevelopmentAccounting(clock=lambda: 0)
    accounting.admit("a", [1] * 5)
    accounting.complete("a", [2] * 3, error="authentic failure")
    accounting.record_stage("ingestion", wall_seconds=2, embedding_tokens=99, embedding_calls=1, bytes_written=4096)
    ledger = DurableLedger(tmp_path / "episode.jsonl")
    ledger.record_episode(accounting.snapshot())
    assert [r["kind"] for r in ledger.records()] == ["input", "output", "step"]
    assert ledger.records()[1]["payload"]["status"] == "error"
    assert ledger.records()[2]["payload"]["embedding_tokens"] == 99
    assert DurableLedger(tmp_path / "episode.jsonl").records() == ledger.records()
    pending = DevelopmentAccounting(clock=lambda: 0)
    pending.admit("p", [1])
    other = DurableLedger(tmp_path / "pending.jsonl")
    with pytest.raises(ValueError):
        other.record_episode(pending.snapshot())
    assert len(other) == 0


def test_deadline_timeout_charges_full_declared_deadline():
    now = [0.0]
    with pytest.raises(DeadlineExceeded):
        with DeadlineSupervisor(300.0, clock=lambda: now[0]) as deadline:
            now[0] = 350.0
            deadline.check()
    (attempt,) = deadline.records()
    assert attempt["timed_out"] is True
    assert attempt["charged_seconds"] == 300.0
    assert attempt["observed_seconds"] == 350.0
    with pytest.raises(ValueError):
        with deadline:
            pass


def test_deadline_overrun_without_check_still_raises_and_charges_full():
    now = [0.0]
    with pytest.raises(DeadlineExceeded):
        with DeadlineSupervisor(300.0, clock=lambda: now[0]) as deadline:
            now[0] = 301.0
    assert deadline.records()[0]["charged_seconds"] == 300.0
    with pytest.raises(ValueError):
        DeadlineSupervisor(0.0)
    with pytest.raises(ValueError):
        DeadlineSupervisor(float("nan"))


def test_deadline_completed_attempt_charged_observed_wall():
    now = [0.0]
    with DeadlineSupervisor(300.0, clock=lambda: now[0]) as deadline:
        now[0] = 12.5
        assert deadline.check() == 287.5
    (attempt,) = deadline.records()
    assert attempt == {
        "deadline_seconds": 300.0,
        "observed_seconds": 12.5,
        "timed_out": False,
        "charged_seconds": 12.5,
    }


def test_metering_unknown_costs_stay_unknown_never_zero():
    samples = [(10.0, 1000), (12.5, 4096), (13.0, 8192)]
    calls = [0]

    def sample():
        pair = samples[min(calls[0], len(samples) - 1)]
        calls[0] += 1
        return pair

    metering = EpisodeMetering(sample=sample)
    snap = metering.snapshot()
    assert snap["embedding"] is None
    assert snap["stages"] == {}
    assert snap["disk_bytes_written"] is None
    assert snap["gpu_wall_seconds"] is None
    assert snap["gpu_wall_seconds_source"] is None
    assert snap["cpu_core_seconds"] == 2.5
    assert snap["peak_rss_bytes"] == 4096
    metering.record_embedding(input_tokens=99, calls=2, cpu_seconds=0.25)
    metering.record_stage("retrieval", wall_seconds=1.5)
    metering.record_stage("retrieval", wall_seconds=0.5, cpu_seconds=0.1)
    metering.record_stage("ingestion", wall_seconds=2.0, cpu_seconds=0.4)
    metering.record_disk(bytes_written=4096)
    metering.record_gpu(wall_seconds=3.25)
    snap = metering.snapshot()
    assert snap["embedding"] == {"input_tokens": 99, "calls": 2, "cpu_seconds": 0.25}
    assert snap["stages"]["retrieval"] == {"wall_seconds": 2.0, "cpu_seconds": 0.1}
    assert snap["stages"]["ingestion"] == {"wall_seconds": 2.0, "cpu_seconds": 0.4}
    assert snap["disk_bytes_written"] == 4096
    assert snap["gpu_wall_seconds"] == 3.25
    assert snap["gpu_wall_seconds_source"] == "caller_reported"
    assert snap["cpu_core_seconds"] == 3.0
    assert snap["peak_rss_bytes"] == 8192
    with pytest.raises(ValueError):
        metering.record_stage("nvml_probe", wall_seconds=1.0)
    with pytest.raises(ValueError):
        metering.record_embedding(input_tokens=-1, calls=1, cpu_seconds=0.1)
    with pytest.raises(ValueError):
        metering.record_gpu(wall_seconds=float("inf"))


def test_metering_live_rusage_sample_is_measurable():
    snap = EpisodeMetering().snapshot()
    assert snap["cpu_core_seconds"] >= 0
    assert snap["peak_rss_bytes"] > 0
    assert snap["execution_costs_qualified"] is False


def test_cost_per_correct_success_failure_inclusive_and_rejects_bad_denominators():
    episodes = [
        {"correct": False, "cost": 100.0},
        {"correct": False, "cost": 250.0},
        {"correct": True, "cost": 50.0},
    ]
    assert cost_per_correct_success(episodes) == 400.0
    assert cost_per_correct_success([{"correct": True, "cost": 0.0}]) == 0.0
    with pytest.raises(ValueError):
        cost_per_correct_success([{"correct": False, "cost": 10.0}])
    with pytest.raises(ValueError):
        cost_per_correct_success([{"correct": True, "cost": None}])
    with pytest.raises(ValueError):
        cost_per_correct_success([{"correct": False, "cost": 10.0}, {"correct": True, "cost": None}])
    with pytest.raises(ValueError):
        cost_per_correct_success([{"correct": True, "cost": -1.0}])
    with pytest.raises(ValueError):
        cost_per_correct_success([{"correct": "yes", "cost": 1.0}])
    with pytest.raises(ValueError):
        cost_per_correct_success([])
