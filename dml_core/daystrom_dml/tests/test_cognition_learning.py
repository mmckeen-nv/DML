import pytest

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.cognition.learning import MAX_BUDGET_DELTA, ProceduralLearningPolicy
from daystrom_dml.cognition.policy import DeterministicCognitionPolicy
from daystrom_dml.cognition.schema import CognitionEvent, CognitionFeedback


def test_positive_feedback_increases_retrieval_for_similar_task_class():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)
    policy = DeterministicCognitionPolicy(learning=learning)

    learning.learn_from_feedback(
        CognitionFeedback(
            decision_id="d1",
            outcome="verified",
            signals={"task_type": "answer", "retrieval_helpful": True, "memory_mode": "hybrid"},
        )
    )

    plan = policy.plan(CognitionEvent(content="what is the status?"))

    assert plan.intent.task_type == "answer"
    assert plan.intent.needs_memory is True
    assert plan.retrieval_plan.mode == "hybrid"
    assert plan.policy_version == "dcn-policy-v0+procedural-v1"
    assert "procedural_learning_applied" in plan.reason_codes


def test_stale_context_feedback_decreases_trust_and_updates_template():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)
    policy = DeterministicCognitionPolicy(learning=learning)

    result = learning.learn_from_feedback(
        CognitionFeedback(
            decision_id="d2",
            outcome="rejected",
            signals={
                "task_type": "recall",
                "stale_context": True,
                "query_template": "fresh project state only",
            },
        )
    )
    plan = policy.plan(CognitionEvent(content="recall what we decided"))

    assert result["accepted"] == 2
    assert plan.retrieval_plan.mode == "none"
    assert plan.intent.needs_memory is False
    assert plan.retrieval_plan.queries == ["fresh project state only"]


def test_tool_failed_feedback_increases_verification_and_alternate_tool():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)
    policy = DeterministicCognitionPolicy(learning=learning)

    learning.learn_from_feedback(
        CognitionFeedback(
            decision_id="d3",
            outcome="tool_failed",
            signals={"task_type": "debugging", "tool_failed": True, "alternate_tool": "python-debugpy"},
        )
    )
    plan = policy.plan(CognitionEvent(content="debug this traceback failure"))

    assert plan.intent.task_type == "debugging"
    assert plan.intent.needs_verification is True
    assert "real_tool_output" in plan.tool_plan.verification_required
    assert "python-debugpy" in plan.tool_plan.recommended_tools


def test_forbidden_identity_or_preference_learning_is_rejected_and_audited_redacted():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)

    result = learning.apply_update(
        "answer",
        {"field": "identity", "value": {"api_key": "secret-value", "name": "Other Bot"}},
        source="test",
        decision_id="d4",
    )

    assert result["accepted"] is False
    assert result["reason"] == "forbidden_field"
    assert learning.profiles == {}
    audit = learning.audit_tail(1)[0]
    assert audit["action"] == "rejected"
    assert audit["attempted_value"] == "[REDACTED]"
    assert "secret-value" not in str(audit)
    assert "Other Bot" not in str(audit)


def test_budget_drift_ceiling_blocks_runaway_context_growth():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)

    rejected = learning.apply_update("code_change", {"field": "context_budget_adjustment", "value": MAX_BUDGET_DELTA + 1})
    accepted = learning.apply_update("code_change", {"field": "context_budget_adjustment", "value": MAX_BUDGET_DELTA})

    assert rejected["accepted"] is False
    assert rejected["reason"] == "ContractError"
    assert accepted["accepted"] is True


def test_export_import_learned_policy_roundtrip_applies_overlay():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)
    learning.apply_update("code_change", {"field": "tool_recommendation", "value": ["terminal", "file"]})
    learning.apply_update("code_change", {"field": "verification_requirement", "value": "strict"})

    snapshot = learning.export_policy()
    restored = ProceduralLearningPolicy(clock=lambda: 2.0)
    result = restored.import_policy(snapshot)
    policy = DeterministicCognitionPolicy(learning=restored)
    plan = policy.plan(CognitionEvent(content="implement and test the change"))

    assert result["imported"] is True
    assert snapshot["schema_version"] == "dcn-procedural-learning-v1"
    assert snapshot["base_policy_ref"] == "dcn-policy-v0"
    assert "code_change" in snapshot["mutable_overlay"]
    assert "terminal" in plan.tool_plan.recommended_tools
    assert "real_tool_output" in plan.tool_plan.verification_required


def test_rollback_to_deterministic_policy_removes_learning_overlay():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)
    policy = DeterministicCognitionPolicy(learning=learning)
    learning.apply_update("answer", {"field": "memory_mode_preference", "value": "hybrid"})

    learned_plan = policy.plan(CognitionEvent(content="what is the status?"))
    rollback = learning.rollback()
    deterministic_plan = policy.plan(CognitionEvent(content="what is the status?"))

    assert learned_plan.retrieval_plan.mode == "hybrid"
    assert rollback["rolled_back"] is True
    assert deterministic_plan.retrieval_plan.mode == "none"
    assert deterministic_plan.policy_version == "dcn-policy-v0"


def test_import_rejects_wrong_schema_or_base_ref():
    learning = ProceduralLearningPolicy(clock=lambda: 1.0)

    with pytest.raises(ContractError):
        learning.import_policy({"schema_version": "wrong", "base_policy_ref": "dcn-policy-v0"})
    with pytest.raises(ContractError):
        learning.import_policy({"schema_version": "dcn-procedural-learning-v1", "base_policy_ref": "identity-policy"})
