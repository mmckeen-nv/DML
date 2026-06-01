import json

from daystrom_dml.cognition.seed_proposer import propose_seed_updates, run_seed_loop


def test_seed_proposer_normalizes_model_updates_and_policy_pressure():
    def fake_generate(base_url: str, model: str, prompt: str, timeout: float) -> str:
        assert base_url == "http://local"
        assert model == "llama3:8b"
        assert "allowed_fields" in prompt
        return json.dumps(
            {
                "candidate_updates": [
                    {
                        "task_type": "debugging",
                        "field": "verification_requirement",
                        "value": "strict",
                        "reason": "debugging needs explicit verification",
                    },
                    {"task_type": "answer", "field": "identity", "value": "Other Bot"},
                ],
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
        )

    proposal = propose_seed_updates(
        {
            "feedback": [{"decision_id": "d1", "outcome": "verified", "signals": {"task_type": "debugging"}}],
            "unsupported_policy_pressure": [
                {"task_type": "answer", "needed_capability": "operator_review_queue", "raw_prompt": "drop me"}
            ],
        },
        model="llama3:8b",
        ollama_base_url="http://local",
        generate_fn=fake_generate,
        clock=lambda: 1.0,
    )

    assert proposal["schema_version"] == "dcn-seed-proposal-v1"
    assert proposal["non_promoting"] is True
    assert proposal["candidate_updates"] == [
        {
            "task_type": "debugging",
            "field": "verification_requirement",
            "value": "strict",
            "reason": "debugging needs explicit verification",
        }
    ]
    assert proposal["rejected_model_items"][0]["reason"] == "field_not_allowed"
    assert proposal["unsupported_policy_pressure"][0]["needed_capability"] == "operator_review_queue"
    assert proposal["unsupported_policy_pressure"][1]["needed_capability"] == "tool_sequence_policy"
    assert "raw_prompt" not in str(proposal)
    assert "must not persist" not in str(proposal)


def test_seed_proposer_preserves_shorthand_policy_pressure():
    def fake_generate(base_url: str, model: str, prompt: str, timeout: float) -> str:
        return json.dumps({"candidate_updates": [], "unsupported_policy_pressure": []})

    proposal = propose_seed_updates(
        {
            "policy_pressure": {
                "preferred_tool_sequence": ["terminal", "file"],
                "raw_prompt": "must not persist",
            }
        },
        generate_fn=fake_generate,
        clock=lambda: 1.0,
    )

    assert proposal["unsupported_policy_pressure"] == [
        {
            "task_type": "*",
            "needed_capability": "tool_sequence_policy",
            "reason": "",
            "field": "preferred_tool_sequence",
            "evidence": {"source_keys": ["preferred_tool_sequence"]},
        }
    ]
    assert "raw_prompt" not in str(proposal)
    assert "must not persist" not in str(proposal)


def test_seed_loop_runs_proposal_through_trial_without_promotion():
    ticks = iter([1.0 + i for i in range(20)])

    def fake_generate(base_url: str, model: str, prompt: str, timeout: float) -> str:
        return json.dumps(
            {
                "candidate_updates": [
                    {"task_type": "debugging", "field": "verification_requirement", "value": "strict"}
                ],
                "unsupported_policy_pressure": [
                    {"task_type": "debugging", "needed_capability": "tool_sequence_policy"}
                ],
            }
        )

    artifact = run_seed_loop(
        {
            "feedback": [{"decision_id": "d1", "outcome": "verified", "signals": {"task_type": "debugging"}}],
            "unsupported_policy_pressure": [{"task_type": "debugging", "needed_capability": "operator_review_queue"}],
        },
        generate_fn=fake_generate,
        clock=lambda: next(ticks),
    )

    assert artifact["schema_version"] == "dcn-seed-loop-artifact-v1"
    assert artifact["non_promoting"] is True
    assert artifact["proposal_summary"]["candidate_update_count"] == 1
    assert artifact["proposal_summary"]["unsupported_policy_pressure_count"] == 2
    assert artifact["trial_summary"]["accepted_update_count"] == 1
    assert artifact["trial_summary"]["unsupported_policy_pressure_count"] == 2
    assert artifact["trial"]["candidate_policy_snapshot"]["mutable_overlay"]["debugging"]["verification_requirement"] == "strict"
