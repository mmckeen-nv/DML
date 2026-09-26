"""Sampled-profile routing and replay controls; no learned live generation."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from daystrom_dml.contracts.agent_episode import (
    AgentEpisodeError, EXECUTION_PROTOCOL_V2, QWEN3_CONSUMER_PROFILE,
    QWEN3_CONSUMER_PROFILES, QWEN3_SAMPLED_CONSUMER_PROFILE,
    RECOVERY_CONSUMER_PROFILE, _compiled, execution_protocol_for_profile,
    initial_messages, validate_episode_events,
)
from daystrom_dml.services import agent_episode as runner
from scripts import agent_campaign_evidence as checker
from scripts import agent_episodes as cli
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_agent_campaign_evidence import _synthetic_campaign_impl
from test_agent_episode_cli import arguments, failed_test_report
from test_agent_episode_runtime import Qwen3ScriptedConsumer, validation_case


SAMPLED = QWEN3_SAMPLED_CONSUMER_PROFILE


class SampledScriptedConsumer(Qwen3ScriptedConsumer):
    """Synthetic boundary events, not the sampled runtime's generated output."""

    def compile(self, messages, tools, *, output_reserved_tokens):
        artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
        return replace(artifact, identity=replace(artifact.identity,
            runtime_identity="dml-qwen3-action-runtime-v2:" + "d" * 64))


@pytest.mark.parametrize("profile", QWEN3_CONSUMER_PROFILES)
def test_both_qwen3_profiles_route_to_explicit_consumer(monkeypatch, profile):
    from daystrom_dml.services import qwen3_action_input

    calls, sentinel = [], object()

    def construct(snapshot_directory, *, consumer_profile):
        calls.append((snapshot_directory, consumer_profile))
        return sentinel

    monkeypatch.setattr(qwen3_action_input, "LocalQwen3ActionInputConsumer", construct)
    assert runner._open_consumer("explicit-snapshot", profile) is sentinel
    assert calls == [("explicit-snapshot", profile)]
    assert execution_protocol_for_profile(profile) == EXECUTION_PROTOCOL_V2
    assert initial_messages("same task", consumer_profile=profile) == initial_messages(
        "same task", consumer_profile=RECOVERY_CONSUMER_PROFILE)


@pytest.mark.parametrize("profile", [None, "", "auto", SAMPLED + "-other", "qwen3-action-runtime-v2"])
def test_unknown_profile_never_dispatches_or_replays(monkeypatch, profile):
    from daystrom_dml.services import qwen3_action_input

    calls = []
    monkeypatch.setattr(qwen3_action_input, "LocalQwen3ActionInputConsumer", lambda *a, **kw: calls.append(kw))
    with pytest.raises(ValueError, match="profile"):
        runner._open_consumer("unused", profile)
    with pytest.raises(ValueError, match="profile"):
        checker._replay_model([], {}, None, profile)
    assert not calls


@pytest.mark.parametrize("mode", ["compile_error", "wrong_consumer", "input_limit", "repeated_rejection"])
def test_sampled_episode_failure_retains_profile_and_original_bounds(tmp_path, mode):
    factory, options = SampledScriptedConsumer, {}
    expected = "step_limit"
    if mode == "compile_error":
        def factory(actions):
            return SampledScriptedConsumer(actions, compile_error=ValueError("synthetic refusal"))
        expected = "model_error"
    elif mode == "wrong_consumer":
        factory, expected = Qwen3ScriptedConsumer, "runner_error"
    elif mode == "input_limit":
        options["limits"] = runner.EpisodeLimits(output_tokens=16, max_input_tokens=1)
        expected = "token_limit"
    else:
        options["repeated"] = True
    report, consumer, _ = validation_case(tmp_path, consumer_profile=SAMPLED,
        consumer_factory=factory, **options)
    assert report["terminal"]["status"] == expected
    assert report["terminal"]["consumer_profile"] == report["events"][0]["payload"]["consumer_profile"] == SAMPLED
    assert consumer.requests[0]["messages"][0] == initial_messages("same", consumer_profile=SAMPLED)[0]
    assert len(consumer.dispatched) == (6 if mode == "repeated_rejection" else 0)
    validate_episode_events(report["events"])


def test_sampled_cli_preserves_every_failure_and_binds_same_qwen3_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_local_episode", failed_test_report)
    assert cli.main(arguments(tmp_path, "--consumer-profile", SAMPLED)) == 1
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    sources = cli._source_digests(consumer_profile=SAMPLED)
    assert sources == cli._source_digests(consumer_profile=QWEN3_CONSUMER_PROFILE)
    assert len(sources) == 19 and len(cli._source_digests()) == 15
    assert all("daystrom_dml.services." + name in sources for name in (
        "qwen3_pretrained_snapshot", "qwen3_model_snapshot", "qwen3_model_input", "qwen3_action_input"))
    assert campaign["source_sha256"] == sources
    assert campaign["consumer_profile"] == campaign["summary"]["consumer_profile"] == SAMPLED
    assert len(campaign["episodes"]) == 9
    assert all(report["terminal"]["consumer_profile"] == SAMPLED for report in campaign["episodes"])
    assert not campaign["raw_evidence_complete"] and not campaign["live_qualified"]


@pytest.fixture(scope="module")
def sampled_campaign(tmp_path_factory, grammar_runtime):
    from daystrom_dml.services.qwen3_action_input import LocalQwen3ActionInputConsumer
    from qwen3_model_input_fixture import create_qwen3_snapshot

    directory = tmp_path_factory.mktemp("sampled-synthetic-campaign")
    snapshot = create_qwen3_snapshot(directory / "snapshot", context_window=8192)
    with LocalQwen3ActionInputConsumer(snapshot.path, consumer_profile=SAMPLED) as compiler:
        case = _synthetic_campaign_impl(directory, validation_consumer=compiler)
    return case, snapshot.path


def test_sampled_scripted_campaign_replays_original_corpus_policy_and_gates(sampled_campaign):
    (campaign, spec, identity, tokenizer), _ = sampled_campaign
    evidence = checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)
    assert evidence["predeclared_gates_passed"] is True
    assert evidence["summary"]["consumer_profile"] == SAMPLED
    assert not evidence["execution_authenticity_verified"] and not evidence["source_ci_qualified"]
    assert spec["acceptance"] == checker.GATES
    requests = [event["payload"] for report in campaign["episodes"] for event in report["events"]
                if event["kind"] == "model_requested"]
    assert all(request["request"]["messages"][0] == initial_messages(
        "same", consumer_profile=QWEN3_CONSUMER_PROFILE)[0] for request in requests)
    assert all(tokenizer.decode(request["compiled"]["input_ids"], skip_special_tokens=False,
        clean_up_tokenization_spaces=False).endswith("<think>\n\n</think>\n\n") for request in requests)


@pytest.mark.parametrize("boundary", ["spec", "report", "start", "terminal", "summary", "compiled"])
def test_sampled_campaign_cannot_be_relabelled_greedy(sampled_campaign, boundary):
    (campaign, spec, identity, tokenizer), _ = sampled_campaign
    campaign, spec, identity = deepcopy(campaign), deepcopy(spec), deepcopy(identity)
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
        event = campaign["episodes"][0]["events"][1]
        event["payload"]["compiled"]["identity"]["runtime_identity"] = "dml-qwen3-action-runtime-v1:" + "a" * 64
        event["payload"]["artifact_digest"] = _compiled(event["payload"]["compiled"]).artifact_digest
    with pytest.raises(ValueError):
        checker.replay_campaign(campaign, spec, identity=identity, tokenizer=tokenizer)


@pytest.mark.parametrize("profile,prefix", [
    (SAMPLED, "dml-qwen3-action-runtime-v1:"),
    (QWEN3_CONSUMER_PROFILE, "dml-qwen3-action-runtime-v2:"),
    (SAMPLED, "dml-qwen3-action-runtime-v3:"),
])
def test_cross_profile_runtime_is_rejected_by_both_event_and_token_replay(sampled_campaign, profile, prefix):
    (campaign, _, identity, tokenizer), _ = sampled_campaign
    events = deepcopy(campaign["episodes"][0]["events"][:2])
    events[0]["payload"]["consumer_profile"] = profile
    payload = events[1]["payload"]
    payload["compiled"]["identity"]["runtime_identity"] = prefix + "b" * 64
    payload["artifact_digest"] = _compiled(payload["compiled"]).artifact_digest
    with pytest.raises(AgentEpisodeError, match="runtime"):
        validate_episode_events(events, require_terminal=False)
    identity = {**identity, "runtime_identity": prefix + "b" * 64}
    with pytest.raises(ValueError, match="runtime"):
        checker._replay_model(events[1:], identity, tokenizer, profile)


def test_sampled_file_replay_binds_shards_identity_and_exact_sources(sampled_campaign, tmp_path):
    from daystrom_dml.services.qwen3_model_snapshot import REQUIRED_FILES

    (campaign, spec, identity, _), snapshot = sampled_campaign
    campaign, spec = deepcopy(campaign), deepcopy(spec)
    source = cli._source_digests(consumer_profile=SAMPLED)
    spec.update(producer_source_sha256=source, source_sha256={
        "dml_core/" + name.replace(".", "/") + ".py": digest for name, digest in source.items()},
        snapshot_sha256={name: checker.file_digest(snapshot / name) for name in REQUIRED_FILES | {"snapshot.json"}},
        model_identity=identity)
    campaign["source_sha256"] = source
    spec_path, campaign_path = tmp_path / "spec.json", tmp_path / "campaign.json"
    spec_path.write_text(json.dumps(spec))
    campaign_path.write_text(json.dumps(campaign))
    arguments = dict(spec_path=spec_path, campaign_path=campaign_path, snapshot_directory=snapshot,
                     source_root=Path(__file__).resolve().parents[2])
    evidence = checker.verify_files(spec_sha256=checker.file_digest(spec_path), **arguments)
    assert evidence["predeclared_gates_passed"] is True
    assert evidence["summary"]["consumer_profile"] == SAMPLED
    spec["model_identity"]["runtime_identity"] = "dml-qwen3-action-runtime-v1:" + "a" * 64
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="identity"):
        checker.verify_files(spec_sha256=checker.file_digest(spec_path), **arguments)
