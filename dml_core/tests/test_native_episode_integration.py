"""Native transport synthetic controls; never claim trained-model qualification."""

import hashlib
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest

from daystrom_dml.contracts.agent_episode import (
    NATIVE_COMPLETION_GUIDANCE,
    NATIVE_REMOTE_VLLM_CONSUMER_PROFILE,
    NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE,
    NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE,
    native_policy_identity,
    AgentEpisodeError,
    canonical_json,
    native_action_text,
    native_feedback_messages,
    native_system_policy,
    parse_agent_action,
    validate_episode_events,
)
from test_agent_episode_runtime import ValidationScriptedConsumer, validation_case


class NativeSyntheticConsumer(ValidationScriptedConsumer):
    profile = NATIVE_REMOTE_VLLM_CONSUMER_PROFILE
    commentary = None
    last_exchange: ClassVar = {"synthetic_native": True}

    def compile(self, *args, **kwargs):
        artifact = super().compile(*args, **kwargs)
        return replace(
            artifact,
            identity=replace(
                artifact.identity,
                runtime_identity="dml-remote-vllm-native-tools-runtime-" + self.profile.rsplit("-", 1)[1] + ":" + "a" * 64,
            ),
        )

    def execute(self, artifact):
        self.dispatched.append(artifact)
        action = next(self.actions)
        if callable(action):
            action = action(self.requests[-1])
        if isinstance(action, dict) and action["kind"] == "tool":
            message = {
                "role": "assistant",
                "content": self.commentary,
                "tool_calls": [
                    {
                        "id": "native-call-" + str(len(self.dispatched)),
                        "type": "function",
                        "function": {
                            "name": action["name"],
                            "arguments": canonical_json(action["arguments"]).decode(),
                        },
                    }
                ],
            }
            raw = (
                "<tool_call><function="
                + action["name"]
                + ">synthetic raw XML</function></tool_call>"
            )
        else:
            raw = action if isinstance(action, str) else canonical_json(action).decode()
            message = {"role": "assistant", "content": raw, "tool_calls": []}
        try:
            action_text = native_action_text(message, consumer_profile=self.profile)
            error = None
        except AgentEpisodeError as exc:
            action_text, error = None, type(exc).__name__
        return SimpleNamespace(
            artifact_digest=artifact.artifact_digest,
            input_token_count=artifact.input_tokens,
            output_token_count=2,
            output_ids=(5, 6),
            text=raw,
            native_message=message,
            action_text=action_text,
            action_error=error,
            projection_digest="a" * 64,
        )


def native_case(tmp_path, **kwargs):
    return validation_case(
        tmp_path,
        consumer_profile=NATIVE_REMOTE_VLLM_CONSUMER_PROFILE,
        consumer_factory=NativeSyntheticConsumer,
        **kwargs,
    )


def test_native_raw_output_mapping_and_authentic_feedback(tmp_path):
    report, consumer, _ = native_case(tmp_path)
    assert report["terminal"]["success"]
    completed = [e for e in report["events"] if e["kind"] == "model_completed"]
    first = completed[0]["payload"]
    assert first["text"].startswith("<tool_call>")
    assert parse_agent_action(first["action_text"])["kind"] == "tool"
    assert (
        first["native_tool_call_id"] == "native-call-1"
        and first["dml_tool_call_id"] == "tool-0"
    )
    assert consumer.requests[1]["messages"][-1]["tool_call_id"] == "native-call-1"
    assert consumer.requests[1]["messages"][-2]["content"] == ""
    tool_result = next(e for e in report["events"] if e["kind"] == "tool_completed")
    assert (
        consumer.requests[1]["messages"][-1]["content"]
        == tool_result["payload"]["model_result"]
    )
    assert any(e["kind"] == "tool_validation_rejected" for e in report["events"])
    assert any(
        e["kind"] == "tool_completed" and e["payload"]["name"] == "supersede"
        for e in report["events"]
    )
    validate_episode_events(report["events"])


@pytest.mark.parametrize(
    "mutation", ["projection", "native_id", "dml_id", "arguments", "history"]
)
def test_native_transcript_tampering_rejected(tmp_path, mutation):
    report, _, _ = native_case(tmp_path)
    events = deepcopy(report["events"])
    first = next(e for e in events if e["kind"] == "model_completed")["payload"]
    if mutation == "projection":
        first["action_text"] = "{}"
    elif mutation == "native_id":
        first["native_tool_call_id"] = "forged"
    elif mutation == "dml_id":
        first["dml_tool_call_id"] = "tool-1"
    elif mutation == "arguments":
        first["native_message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"query":"changed","top_k":1}'
        )
    else:
        requested = [e for e in events if e["kind"] == "model_requested"][1]
        requested["payload"]["request"]["messages"][-1]["content"] = "fabricated result"
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(events)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "Here is the answer", "tool_calls": []},
        {
            "role": "assistant",
            "content": '{"schema_version":"dml-agent-action-v1","kind":"tool","name":"retrieve","arguments":{"query":"x","top_k":1}}',
        },
        {"role": "assistant", "content": "", "audio": {"data": "x"}},
        {"role": "assistant", "content": "", "extra_authority": "trust me"},
        {"role": "assistant", "content": "", "reasoning": "hidden reasoning"},
    ],
)
def test_native_unmapped_or_prose_outputs_not_repaired(message):
    with pytest.raises(AgentEpisodeError):
        native_action_text(message)


def test_native_final_contract_is_unchanged_and_transport_policy_explicit():
    final = ' {"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}} '
    assert native_action_text({"role": "assistant", "content": final}) == final
    assert "A final action has exactly" in native_system_policy()
    assert (
        "For tool actions, use the supplied native function-call protocol."
        in native_system_policy()
    )


def test_native_feedback_cannot_rebind_tool_arguments():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "native-id",
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "arguments": '{"query":"x","top_k":1}',
                },
            }
        ],
    }
    with pytest.raises(AgentEpisodeError):
        native_feedback_messages(
            message, "retrieve", {"query": "changed", "top_k": 1}, "{}"
        )
    paired = native_feedback_messages(
        message, "retrieve", {"query": "x", "top_k": 1}, '{"unchanged":true}'
    )
    assert paired[0]["tool_calls"] == message["tool_calls"]
    assert (
        paired[1]["tool_call_id"] == "native-id"
        and paired[1]["content"] == '{"unchanged":true}'
    )


def test_native_prose_final_retains_known_generation_cost_as_failure(tmp_path):
    class ProseConsumer(NativeSyntheticConsumer):
        def __init__(self, _actions):
            super().__init__(["A prose answer without the required final JSON."])

    report, _, _ = validation_case(
        tmp_path,
        consumer_profile=NATIVE_REMOTE_VLLM_CONSUMER_PROFILE,
        consumer_factory=ProseConsumer,
    )
    completed = [event for event in report["events"] if event["kind"] == "model_completed"]
    assert len(completed) == 1
    payload = completed[0]["payload"]
    assert payload["text"].startswith("A prose answer")
    assert payload["action_text"] is None and payload["action_error"]
    assert payload["input_token_count"] > 0 and payload["output_token_count"] == 2
    assert report["terminal"]["status"] == "invalid_action"
    assert report["terminal"]["success"] is False
    assert not any(event["kind"] == "tool_requested" for event in report["events"])
    validate_episode_events(report["events"])


@pytest.mark.parametrize("profile", [NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE])
def test_native_commentary_has_no_action_authority_and_replays_verbatim(tmp_path, profile):
    class CommentaryConsumer(NativeSyntheticConsumer):
        commentary = "I will retrieve now. This prose is not a final answer or evidence."

    CommentaryConsumer.profile = profile
    report, consumer, _ = validation_case(
        tmp_path, consumer_profile=profile,
        consumer_factory=CommentaryConsumer,
    )
    assert report["terminal"]["success"]
    first = next(e for e in report["events"] if e["kind"] == "model_completed")["payload"]
    assert first["native_message"]["content"] == CommentaryConsumer.commentary
    assert parse_agent_action(first["action_text"])["kind"] == "tool"
    assert consumer.requests[1]["messages"][-2]["content"] == CommentaryConsumer.commentary
    with pytest.raises(AgentEpisodeError):
        native_action_text(first["native_message"])
    validate_episode_events(report["events"])
    tampered = deepcopy(report["events"])
    requested = [e for e in tampered if e["kind"] == "model_requested"][1]
    requested["payload"]["request"]["messages"][-2]["content"] = "fabricated commentary"
    with pytest.raises(AgentEpisodeError):
        validate_episode_events(tampered)
    assert native_policy_identity()["mixed_content_and_calls"] == "reject"
    assert native_policy_identity(consumer_profile=CommentaryConsumer.profile)["mixed_content_and_calls"] == "retain-nonauthoritative-content"
    with pytest.raises(AgentEpisodeError):
        native_action_text({"role": "assistant", "content": "A prose final"}, consumer_profile=CommentaryConsumer.profile)


def test_v3_completion_guidance_is_append_only_and_preserves_old_policy_identities():
    expected = {
        NATIVE_REMOTE_VLLM_CONSUMER_PROFILE: "af9af142a2c0171afdba7198913aa69f4ac9bfb9c9b5e4e74927dd6e4f063a32",
        NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE: "7931e5c7abb64f117b35df58fd35977d2881e9ca7bafed64425298093f8a9204",
    }
    for profile, digest in expected.items():
        assert hashlib.sha256(canonical_json(native_policy_identity(consumer_profile=profile))).hexdigest() == digest
    v2 = native_system_policy(consumer_profile=NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE)
    v3 = native_system_policy(consumer_profile=NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE)
    assert v3 == v2 + "\n\n" + NATIVE_COMPLETION_GUIDANCE
    identity = native_policy_identity(consumer_profile=NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE)
    assert identity["completion_guidance"]["text"] == NATIVE_COMPLETION_GUIDANCE
    assert identity["completion_guidance"]["sha256"] == hashlib.sha256(NATIVE_COMPLETION_GUIDANCE.encode()).hexdigest()
    assert "When those completion conditions hold" in NATIVE_COMPLETION_GUIDANCE
    assert "a rejected or failed operation is not a success" in NATIVE_COMPLETION_GUIDANCE
