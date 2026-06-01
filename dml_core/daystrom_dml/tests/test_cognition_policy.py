from daystrom_dml.api_contracts import ReasonCode
from daystrom_dml.cognition.policy import DeterministicCognitionPolicy
from daystrom_dml.cognition.schema import CognitionEvent


def plan_for(text, **metadata):
    return DeterministicCognitionPolicy().plan(CognitionEvent(content=text, metadata=metadata))


def test_continue_work_produces_resume_or_hybrid_retrieval():
    plan = plan_for("continue the DML hardening work")

    assert plan.intent.needs_memory is True
    assert plan.retrieval_plan.mode in {"resume", "hybrid"}
    assert ReasonCode.RESUME_REQUEST.value in plan.reason_codes


def test_hello_again_does_not_retrieve_memory():
    plan = plan_for("hello again")

    assert plan.retrieval_plan.mode == "none"
    assert plan.intent.needs_memory is False
    assert ReasonCode.GREETING.value in plan.reason_codes


def test_run_tests_and_fix_failures_recommends_tools_and_verification():
    plan = plan_for("run tests and fix failures")

    assert plan.intent.needs_tools is True
    assert plan.intent.needs_verification is True
    assert "terminal" in plan.tool_plan.recommended_tools
    assert "tests" in plan.tool_plan.verification_required
    assert ReasonCode.CODE_TASK.value in plan.reason_codes


def test_remember_preference_creates_preference_candidate_not_dml_fact():
    plan = plan_for("remember I prefer concise updates")

    assert plan.writeback_plan.mode == "preference_candidate"
    assert "preference" in plan.writeback_plan.candidate_classes
    assert "raw_transcript" in plan.writeback_plan.forbidden_classes
    assert ReasonCode.PREFERENCE_SIGNAL.value in plan.reason_codes


def test_side_effect_action_sets_medium_risk_and_confirmation_when_needed():
    plan = plan_for("delete the stale branch")

    assert plan.risk.level == "medium"
    assert plan.risk.requires_confirmation is True
    assert "delete" in plan.risk.side_effect_classes
    assert ReasonCode.SIDE_EFFECT.value in plan.reason_codes


def test_compaction_resume_metadata_triggers_resume_retrieval():
    plan = plan_for("ok", compaction_resume=True)

    assert plan.retrieval_plan.mode == "resume"
    assert plan.intent.needs_memory is True
