import json

from daystrom_dml.cognition.evaluation import DCNEvalHarness, EvalCase, EvalMemoryItem, smoke_eval_cases


def test_eval_smoke_suite_passes_and_is_deterministic():
    harness = DCNEvalHarness(clock=lambda: 42.0)

    first = harness.run_suite(smoke_eval_cases())
    second = harness.run_suite(smoke_eval_cases())

    assert first.passed is True
    assert second.passed is True
    assert first.deterministic_hash == second.deterministic_hash
    assert first.summary["case_count"] == 7
    assert first.summary["max_pollution_score"] == 0.0
    assert first.summary["blocked_polluting_items"] == 2
    case_ids = {case.case_id for case in first.cases}
    assert {"code_verification_tool_policy", "setup_retrieval_semantic", "debugging_requires_verification"} <= case_ids


def test_eval_artifact_is_deterministic_sanitized_and_coverage_rich():
    harness = DCNEvalHarness(clock=lambda: 0.0)

    first = harness.run_suite(smoke_eval_cases())
    second = harness.run_suite(smoke_eval_cases())
    artifact = first.artifact()

    assert artifact == second.artifact()
    assert artifact["schema_version"] == "dcn-eval-artifact-v1"
    assert artifact["artifact_hash"]
    assert artifact["summary"]["case_count"] == 7
    assert artifact["coverage"]["case_ids"] == [case.case_id for case in first.cases]
    assert {"code_change", "debugging", "admin"} <= set(artifact["coverage"]["task_types"])
    assert {"none", "hybrid", "resume", "semantic"} <= set(artifact["coverage"]["retrieval_modes"])
    assert {"durable_signal_only", "preference_candidate", "none"} <= set(artifact["coverage"]["writeback_modes"])
    assert all(flag is False for flag in artifact["redaction_policy"].values())
    rendered = json.dumps(artifact, sort_keys=True).lower()
    assert "raw_transcript" not in rendered
    assert "tool_calls" not in rendered
    assert "prompt_scaffold" not in rendered
    assert "sk-" not in rendered
    assert "continue the phase eleven" not in rendered


def test_eval_report_excludes_raw_fixture_text_and_secret_like_values():
    case = EvalCase(
        case_id="secret_block",
        prompt="continue and build memory retrieval quality code",
        corpus=[
            EvalMemoryItem("unsafe", "raw_transcript token=sk-secret-fixture-123456789 tool_calls", ["unsafe"]),
            EvalMemoryItem("safe", "continue memory work retrieval quality", ["dcn"]),
        ],
        relevant_ids=["safe"],
        expected_task_type="code_change",
        expected_retrieval_mode="hybrid",
        expected_writeback_mode="durable_signal_only",
        min_precision_at_k=1.0,
        min_recall_at_k=1.0,
    )

    report = DCNEvalHarness(clock=lambda: 0.0).run_suite([case])
    rendered = json.dumps(report.to_dict(), sort_keys=True)

    assert report.passed is True
    assert "sk-secret-fixture" not in rendered
    assert "raw_transcript" not in rendered
    assert "tool_calls" not in rendered
    assert "continue memory work retrieval quality" not in rendered
    assert report.summary["blocked_polluting_items"] == 1


def test_eval_case_violates_expected_policy_and_thresholds():
    case = EvalCase(
        case_id="bad_expectation",
        prompt="hello again",
        corpus=[EvalMemoryItem("noise", "irrelevant content", [])],
        relevant_ids=["missing"],
        expected_task_type="code_change",
        expected_retrieval_mode="resume",
        min_recall_at_k=1.0,
    )

    result = DCNEvalHarness(clock=lambda: 0.0).run_case(case)

    assert result.passed is False
    assert any("task_type expected" in violation for violation in result.violations)
    assert any("retrieval_mode expected" in violation for violation in result.violations)
    assert any("recall_at_k" in violation for violation in result.violations)


def test_eval_memory_retrieval_metrics_are_exact_for_clean_fixture():
    case = EvalCase(
        case_id="retrieval_quality",
        prompt="continue dcn evaluation retrieval quality",
        corpus=[
            EvalMemoryItem("a", "dcn evaluation retrieval quality", []),
            EvalMemoryItem("b", "dcn evaluation retrieval quality extra", []),
            EvalMemoryItem("c", "unrelated groceries", []),
        ],
        relevant_ids=["a", "b"],
        expected_task_type="planning",
        expected_retrieval_mode="resume",
        min_precision_at_k=1.0,
        min_recall_at_k=1.0,
        top_k=2,
    )

    result = DCNEvalHarness(clock=lambda: 0.0).run_case(case)

    assert result.passed is True
    assert result.metrics["precision_at_k"] == 1.0
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["spurious_retrieval_rate"] == 0.0
