from daystrom_dml.cognition.learning import ALLOWED_FIELDS
from daystrom_dml.cognition.seed_trial import run_seed_trial


def test_seed_trial_accepts_allowed_updates_and_records_policy_pressure():
    ticks = iter([1.0 + i for i in range(20)])
    artifact = run_seed_trial(
        {
            "seed_model": "llama3:8b",
            "embedding_model": "ollama:qwen3-embedding:0.6b",
            "candidate_updates": [
                {"task_type": "debugging", "field": "verification_requirement", "value": "strict", "decision_id": "d1"}
            ],
            "feedback": [
                {
                    "decision_id": "d2",
                    "outcome": "verified",
                    "signals": {"task_type": "code_change", "retrieval_helpful": True, "memory_mode": "resume"},
                    "unsupported_policy_pressure": [
                        {
                            "task_type": "debugging",
                            "needed_capability": "tool_sequence_policy",
                            "proposed_schema_extension": {
                                "field": "preferred_tool_sequence",
                                "type": "list[str]",
                                "bounds": {"max_items": 6, "allowed_tools": ["terminal", "file"]},
                            },
                            "raw_prompt": "must not persist",
                        }
                    ],
                }
            ],
        },
        clock=lambda: next(ticks),
    )

    assert artifact["schema_version"] == "dcn-seed-trial-artifact-v1"
    assert artifact["non_promoting"] is True
    assert artifact["seed_model"] == "llama3:8b"
    assert artifact["allowed_fields"] == sorted(ALLOWED_FIELDS)
    assert artifact["summary"]["feedback_count"] == 1
    assert artifact["summary"]["accepted_update_count"] == 2
    assert artifact["summary"]["rejected_update_count"] == 0
    assert artifact["summary"]["unsupported_policy_pressure_count"] == 1
    assert artifact["candidate_policy_snapshot"]["mutable_overlay"]["debugging"]["verification_requirement"] == "strict"
    assert artifact["candidate_policy_snapshot"]["mutable_overlay"]["code_change"]["memory_mode_preference"] == "resume"
    pressure = artifact["unsupported_policy_pressure"][0]
    assert pressure["needed_capability"] == "tool_sequence_policy"
    assert pressure["proposed_schema_extension"]["field"] == "preferred_tool_sequence"
    assert "raw_prompt" not in pressure
    assert "must not persist" not in str(artifact)


def test_seed_trial_rejects_forbidden_and_unknown_updates_without_snapshot_mutation():
    artifact = run_seed_trial(
        {
            "candidate_updates": [
                {"task_type": "answer", "field": "identity", "value": {"name": "Other Bot", "token": "secret"}},
                {"task_type": "answer", "field": "preferred_tool_sequence", "value": ["terminal", "file"]},
            ]
        },
        clock=lambda: 1.0,
    )

    assert artifact["summary"]["accepted_update_count"] == 0
    assert artifact["summary"]["rejected_update_count"] == 2
    assert artifact["candidate_policy_snapshot"]["mutable_overlay"] == {}
    reasons = {item["reason"] for item in artifact["rejected_updates"]}
    assert reasons == {"forbidden_field", "unknown_or_unclassified_field"}
    assert "Other Bot" not in str(artifact)
    assert "secret" not in str(artifact)


def test_seed_trial_preserves_shorthand_policy_pressure_without_raw_payload():
    artifact = run_seed_trial(
        {
            "feedback": [
                {
                    "decision_id": "d3",
                    "outcome": "needs_fix",
                    "signals": {"task_type": "debugging"},
                    "policy_pressure": {
                        "preferred_tool_sequence": ["terminal", "file"],
                        "raw_prompt": "do not persist this prompt",
                    },
                }
            ],
            "policy_pressure": {"prefer_retrieval_mode": "semantic", "task_type": "admin"},
        },
        clock=lambda: 1.0,
    )

    assert artifact["summary"]["unsupported_policy_pressure_count"] == 2
    capabilities = {item["needed_capability"] for item in artifact["unsupported_policy_pressure"]}
    assert capabilities == {"tool_sequence_policy", "retrieval_mode_policy"}
    fields = {item["field"] for item in artifact["unsupported_policy_pressure"]}
    assert fields == {"preferred_tool_sequence", "prefer_retrieval_mode"}
    assert "raw_prompt" not in str(artifact)
    assert "do not persist" not in str(artifact)
