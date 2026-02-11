from __future__ import annotations

from datetime import datetime, timezone

from daystrom_dml.stm.controller import STMController
from daystrom_dml.stm.policy import LTMWritePolicy, MemoryWrite
from daystrom_dml.stm.schema import Commitment, STMState


def test_extract_structured_updates_success_and_failure() -> None:
    controller = STMController()

    def generator(prompt: str, max_new_tokens: int, **kwargs: object) -> str:
        return (
            "{\n"
            "  \"commitments\": [\n"
            "    {\"statement\": \"User prefers tea\", \"confidence\": 0.9, \"source\": \"user\"}\n"
            "  ],\n"
            "  \"goals\": [\"Brew tea\"],\n"
            "  \"constraints\": [],\n"
            "  \"entities\": [],\n"
            "  \"plan\": {\"steps\": [], \"current_step\": 0, \"status\": \"idle\"}\n"
            "}"
        )

    extraction = controller.extract_structured_updates(
        user_msg="I like tea",
        model_msg="Got it",
        generator=generator,
    )
    assert extraction.commitments
    assert extraction.commitments[0].statement == "User prefers tea"

    def bad_generator(prompt: str, max_new_tokens: int, **kwargs: object) -> str:
        return "not json"

    extraction = controller.extract_structured_updates(
        user_msg="I like tea",
        model_msg="Got it",
        generator=bad_generator,
    )
    assert extraction.commitments == []


def test_write_policy_gates_low_confidence() -> None:
    policy = LTMWritePolicy(confidence_threshold=0.75)
    low_conf = MemoryWrite(
        text="Maybe preference",
        meta={},
        confidence=0.4,
        source="user",
    )
    assert policy.filter_writes([low_conf]) == []

    hypothesis = MemoryWrite(
        text="Guess about preference",
        meta={},
        confidence=0.2,
        source="model",
        hypothesis=True,
    )
    writes = policy.filter_writes([hypothesis])
    assert writes
    assert writes[0].expires_at is not None


def test_contradiction_detector_flags_negation() -> None:
    controller = STMController()
    stm = STMState(
        commitments=[
            Commitment(
                id="1",
                statement="The sky is blue",
                confidence=0.9,
                source="user",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]
    )
    new_commitment = Commitment(
        id="2",
        statement="The sky is not blue",
        confidence=0.8,
        source="user",
    )
    contradictions = controller.detect_contradictions(stm, [new_commitment])
    assert contradictions
