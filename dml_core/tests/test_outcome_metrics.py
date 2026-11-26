import pytest
from daystrom_dml.services.evaluation import summarize_outcomes, TASK_OUTCOME_VERSION
from scripts.production_baseline import SQLiteBaseline
import numpy as np


def event(task, success):
    return {"schema_version": TASK_OUTCOME_VERSION, "episode_id": "episode", "task_id": task, "success": success,
            "input_tokens": 10, "output_tokens": 5, "maintenance_tokens": 5, "latency_ms": 10., "retrieval_ms": 2.}


def test_failed_attempts_and_maintenance_count_against_completed_tasks():
    result = summarize_outcomes([event("failed", False), event("succeeded", True)])
    assert result["tokens_per_completed_task"] == 40
    assert result["task_success_rate"] == .5
    assert result["false_memory_rate"] is None
    assert result["ttft_ms"]["samples"] == 0
    assert summarize_outcomes([event("failed", False)])["tokens_per_completed_task"] is None


def test_duplicate_terminal_events_and_unknown_schema_cannot_bias_metrics():
    with pytest.raises(ValueError, match="unique"):
        summarize_outcomes([event("task", True)] * 2)
    with pytest.raises(ValueError, match="schema"):
        summarize_outcomes([{**event("task", True), "schema_version": "future"}])
    with pytest.raises(ValueError, match="TTFT"):
        summarize_outcomes([{**event("task", True), "ttft_ms": 100}])


def test_baseline_durable_scope_filter_and_recency(tmp_path):
    path = tmp_path / "baseline.sqlite3"
    store = SQLiteBaseline(path)
    vec = np.ones(2)
    store.remember(1, "older", vec, scope={"tenant":"a"}, timestamp=0)
    store.remember(2, "newer", vec, scope={"tenant":"a"}, timestamp=3600)
    store.remember(3, "PRIVATE", vec, scope={"tenant":"b"}, timestamp=3600)
    store.close()
    reopened = SQLiteBaseline(path)
    result = reopened.recall(vec, scope={"tenant":"a"}, now=3600, top_k=1)
    assert result == {"ids": [2], "context": "newer"}
    reopened.close()
