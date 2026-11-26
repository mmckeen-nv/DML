"""GGUF routing and evidence boundaries; synthetic decisions, no model weights."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from daystrom_dml.contracts.agent_episode import (
    AgentEpisodeError, EXECUTION_PROTOCOL_V2, QWEN3_CONSUMER_PROFILE,
    QWEN3_GGUF_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE, _compiled,
    execution_protocol_for_profile, initial_messages, validate_episode_events,
)
from daystrom_dml.contracts.model_input import CompiledModelInput, ModelInputIdentity, ModelInputRequest
from daystrom_dml.services import agent_episode as runner
from daystrom_dml.services.episode_verifiers import load_episode_corpus
from daystrom_dml.services.qwen3_model_snapshot import QWEN3_CHAT_TEMPLATE, QWEN3_CHAT_TEMPLATE_DIGEST
from scripts import agent_campaign_evidence as checker
from scripts import agent_episodes as cli
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_agent_campaign_evidence import _synthetic_campaign_impl
from test_agent_episode_cli import arguments, failed_test_report
from test_agent_episode_runtime import Qwen3ScriptedConsumer, validation_case
from test_qwen_chat_template import tokenizer as tokenizer


PROFILE = QWEN3_GGUF_CONSUMER_PROFILE
PREFIX = "dml-qwen3-gguf-action-runtime-v1:"


class GGUFScriptedConsumer(Qwen3ScriptedConsumer):
    def compile(self, messages, tools, *, output_reserved_tokens):
        artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
        return replace(artifact, identity=replace(artifact.identity, runtime_identity=PREFIX + "d" * 64))


def test_explicit_gguf_dispatch_preserves_protocol_and_logical_policy(monkeypatch):
    from daystrom_dml.services import qwen3_gguf_action_input

    calls, sentinel = [], object()

    def construct(snapshot_directory, *, consumer_profile):
        calls.append((snapshot_directory, consumer_profile))
        return sentinel

    monkeypatch.setattr(qwen3_gguf_action_input, "LocalQwen3GGUFActionInputConsumer", construct)
    assert runner._open_consumer("explicit-gguf-snapshot", PROFILE) is sentinel
    assert calls == [("explicit-gguf-snapshot", PROFILE)]
    assert execution_protocol_for_profile(PROFILE) == EXECUTION_PROTOCOL_V2
    assert initial_messages("unchanged task", consumer_profile=PROFILE) == initial_messages(
        "unchanged task", consumer_profile=RECOVERY_CONSUMER_PROFILE)
    with pytest.raises(ValueError, match="profile"):
        runner._open_consumer("unused", PROFILE + "-auto")
    assert len(calls) == 1
    for scenario in load_episode_corpus()["scenarios"]:
        for task in scenario["tasks"]:
            assert runner.build_episode_request(task, consumer_profile=PROFILE) == runner.build_episode_request(
                task, consumer_profile=QWEN3_CONSUMER_PROFILE)


@pytest.mark.parametrize("mode", ["compile_error", "wrong_consumer", "input_limit", "repeated_rejection"])
def test_failures_retain_profile_and_inherited_bounds(tmp_path, mode):
    factory, options, expected = GGUFScriptedConsumer, {}, "step_limit"
    if mode == "compile_error":
        def factory(actions):
            return GGUFScriptedConsumer(actions, compile_error=ValueError("synthetic refusal"))
        expected = "model_error"
    elif mode == "wrong_consumer":
        factory, expected = Qwen3ScriptedConsumer, "runner_error"
    elif mode == "input_limit":
        options["limits"] = runner.EpisodeLimits(output_tokens=16, max_input_tokens=1)
        expected = "token_limit"
    else:
        options["repeated"] = True
    report, consumer, _ = validation_case(tmp_path, consumer_profile=PROFILE,
        consumer_factory=factory, **options)
    assert report["terminal"]["status"] == expected
    assert report["terminal"]["consumer_profile"] == report["events"][0]["payload"]["consumer_profile"] == PROFILE
    assert consumer.requests[0]["messages"][0] == initial_messages("same", consumer_profile=PROFILE)[0]
    assert len(consumer.dispatched) == (6 if mode == "repeated_rejection" else 0)
    validate_episode_events(report["events"])


def test_cli_preserves_all_nine_failures_and_binds_gguf_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_local_episode", failed_test_report)
    assert cli.main(arguments(tmp_path, "--consumer-profile", PROFILE)) == 1
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    sources = cli._source_digests(consumer_profile=PROFILE)
    base = cli._source_digests()
    assert set(sources) - set(base) == {"daystrom_dml.services." + name for name in (
        "qwen3_gguf_pretrained_snapshot", "qwen3_gguf_model_snapshot",
        "qwen3_gguf_model_input", "qwen3_gguf_action_input")}
    assert all(sources[name] == digest for name, digest in base.items())
    assert campaign["source_sha256"] == sources
    assert campaign["consumer_profile"] == campaign["summary"]["consumer_profile"] == PROFILE
    assert len(campaign["episodes"]) == 9
    assert all(report["terminal"]["consumer_profile"] == PROFILE for report in campaign["episodes"])
    assert not campaign["raw_evidence_complete"] and not campaign["live_qualified"]


class SyntheticGGUFCompiler:
    """Only tokenize test messages; synthetic outputs are supplied by the fixture."""
    _consumer_profile = PROFILE

    def __init__(self, tokenizer):
        # Reuse the established tiny Qwen fixture vocabulary without creating weights.
        self._tokenizer = tokenizer.train_new_from_iterator([
            "system user assistant tool content tools name function arguments tool_calls tool_call_id",
            "Read the green notebook. Preserve café memory and every Unicode 雨 field.",
            '{"type":"function","function":{"name":"lookup_note","arguments":{"subject":"green"}}}',
            "<|im_start|> <|im_end|> <|endoftext|> \\u003c & \\\" </script> \n",
        ], vocab_size=384)
        self._tokenizer.chat_template = QWEN3_CHAT_TEMPLATE
        self._identity = ModelInputIdentity("1" * 64, "2" * 64, QWEN3_CHAT_TEMPLATE_DIGEST,
                                           PREFIX + "d" * 64, 32768)
        self.count = 0

    def compile(self, messages, tools, *, output_reserved_tokens):
        self.count += 1
        request = ModelInputRequest.from_payload({"messages": messages, "tools": tools,
                                                  "output_reserved_tokens": output_reserved_tokens})
        encoded = self._tokenizer.apply_chat_template(request.messages, tools=request.tools,
            chat_template=QWEN3_CHAT_TEMPLATE, tokenize=True, add_generation_prompt=False,
            continue_final_message=False, truncation=False, padding=False, return_dict=True,
            tokenizer_kwargs={"return_attention_mask": True})
        return CompiledModelInput(self._identity, request.request_digest, tuple(encoded["input_ids"]),
            tuple(encoded["attention_mask"]), output_reserved_tokens, 32768,
            "synthetic-gguf", str(self.count), "0" * 64)


@pytest.fixture(scope="module")
def gguf_campaign(tmp_path_factory, tokenizer, grammar_runtime):
    return _synthetic_campaign_impl(tmp_path_factory.mktemp("synthetic-gguf-campaign"),
                                    validation_consumer=SyntheticGGUFCompiler(tokenizer))


def test_synthetic_replay_preserves_corpus_policy_tools_and_acceptance(gguf_campaign):
    campaign, spec, identity, tokenizer = gguf_campaign
    evidence = checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    assert evidence["predeclared_gates_passed"] is True
    assert spec["acceptance"] == checker.GATES
    assert not evidence["execution_authenticity_verified"] and not evidence["source_ci_qualified"]
    assert len(campaign["episodes"]) == 9
    for report in campaign["episodes"]:
        for event in report["events"]:
            if event["kind"] != "model_requested":
                continue
            payload = event["payload"]
            assert payload["request"]["messages"][0] == initial_messages(
                "same", consumer_profile=QWEN3_CONSUMER_PROFILE)[0]
            rendered = tokenizer.decode(payload["compiled"]["input_ids"], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            assert payload["compiled"]["output_reserved_tokens"] == 256


@pytest.mark.parametrize("boundary", ["spec", "report", "start", "terminal", "summary", "compiled"])
def test_campaign_cannot_be_relabelled_as_previous_qwen3(gguf_campaign, boundary):
    campaign, spec, identity, tokenizer = gguf_campaign
    campaign, spec = deepcopy(campaign), deepcopy(spec)
    if boundary == "spec":
        campaign["consumer_profile"] = spec["consumer_profile"] = QWEN3_CONSUMER_PROFILE
    elif boundary == "report":
        campaign["episodes"][0]["consumer_profile"] = QWEN3_CONSUMER_PROFILE
    elif boundary == "start":
        campaign["episodes"][0]["events"][0]["payload"]["consumer_profile"] = QWEN3_CONSUMER_PROFILE
    elif boundary == "terminal":
        campaign["episodes"][0]["terminal"]["consumer_profile"] = QWEN3_CONSUMER_PROFILE
    elif boundary == "summary":
        campaign["summary"]["consumer_profile"] = QWEN3_CONSUMER_PROFILE
    else:
        payload = campaign["episodes"][0]["events"][1]["payload"]
        payload["compiled"]["identity"]["runtime_identity"] = "dml-qwen3-action-runtime-v1:" + "a" * 64
        payload["artifact_digest"] = _compiled(payload["compiled"]).artifact_digest
    with pytest.raises(ValueError):
        checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)


@pytest.mark.parametrize("corruption", ["runtime", "input_token", "input_mask", "alternate_framing", "output_token"])
def test_replay_rejects_wrong_identity_framing_and_tokens(gguf_campaign, corruption):
    campaign, _, identity, tokenizer = gguf_campaign
    events = deepcopy(campaign["episodes"][0]["events"])
    identity, tokenizer = deepcopy(identity), deepcopy(tokenizer)
    request = next(e["payload"] for e in events if e["kind"] == "model_requested")
    if corruption == "runtime":
        identity["runtime_identity"] = "dml-qwen2-bf16-action-runtime-v1:" + "b" * 64
        request["compiled"]["identity"] = identity
        request["artifact_digest"] = _compiled(request["compiled"]).artifact_digest
        with pytest.raises(AgentEpisodeError, match="runtime"):
            validate_episode_events(events[:2], require_terminal=False)
    elif corruption == "input_token":
        request["compiled"]["input_ids"][0] += 1
    elif corruption == "input_mask":
        request["compiled"]["attention_mask"][0] = 0
    elif corruption == "alternate_framing":
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE + "alternate framing marker"
    else:
        output = next(e["payload"] for e in events if e["kind"] == "model_completed")
        output["output_ids"][0] = max(tokenizer.get_vocab().values()) + 1
    with pytest.raises(ValueError, match="runtime|tokenization|vocabulary"):
        checker._replay_model(events[1:], identity, tokenizer, PROFILE)
