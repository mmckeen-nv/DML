"""CLI persistence/accounting tests. Patched producer results are test-only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import agent_episodes as cli


def arguments(tmp_path, *extra):
    return ["--snapshot-directory", str(tmp_path / "missing-model"),
            "--work-directory", str(tmp_path / "work"), "--output", str(tmp_path / "campaign.json"),
            *extra]


def failed_test_report(**kwargs):
    return cli._interrupted_report(kwargs["scenario"], kwargs["task"], kwargs["limits"], RuntimeError(), 1.0,
        prior_context=kwargs.get("prior_context"), consumer_profile=kwargs.get("consumer_profile", "gpt2-v1"))


def test_full_corpus_failures_are_retained_and_exit_nonzero(tmp_path, monkeypatch):
    calls = []
    def fake_producer(**kwargs):
        calls.append(deepcopy(kwargs))
        return failed_test_report(**kwargs)
    monkeypatch.setattr(cli, "run_local_episode", fake_producer)
    assert cli.main(arguments(tmp_path)) == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert artifact["schema_version"] == "dml-agent-campaign-v1"
    assert artifact["summary"]["attempted_tasks"] == len(artifact["episodes"]) == 9
    assert artifact["summary"]["completed_tasks"] == 0
    assert artifact["summary"]["tokens_per_completed_task"] is None
    assert artifact["summary"]["unknown_usage_tasks"] == 9
    assert artifact["summary"]["contradiction_rate"] is None
    assert artifact["live_qualified"] is artifact["source_ci_qualified"] is False
    assert artifact["raw_evidence_complete"] is False
    assert len(artifact["source_sha256"]) == 15
    assert {"daystrom_dml.services.model_input", "daystrom_dml.services.model_input_snapshot",
            "daystrom_dml.services.pretrained_snapshot", "daystrom_dml.services.qwen_model_input",
            "daystrom_dml.services.qwen_model_snapshot", "daystrom_dml.services.qwen_pretrained_snapshot",
            "daystrom_dml.services.agent_action_grammar", "daystrom_dml.services.qwen_action_input",
            "scripts.agent_campaign_evidence"}.issubset(artifact["source_sha256"])
    assert artifact["consumer_profile"] == "gpt2-v1"
    assert all(call["consumer_profile"] == "gpt2-v1" for call in calls)
    assert all(report["consumer_profile"] == "gpt2-v1" for report in artifact["episodes"])
    assert len({str(call["work_directory"]) for call in calls}) == 9
    assert len(list((tmp_path / "work").rglob("report.json"))) == 9
    repeat = next(call for call in calls if call["task"]["id"] == "recall_after_correction")
    assert repeat["previous_answers"] == {"first_recall": None}
    prior = next(report["terminal"] for report in artifact["episodes"] if report["terminal"]["task_id"] == "first_recall")
    assert repeat["prior_context"] == {key: prior[key]
                                       for key in ("episode_id", "task_id", "evidence_digest", "answer")}
    assert repeat["prior_context"]["answer"] is None
    assert not (tmp_path / "missing-model").exists()


def test_selected_followup_without_prior_attempt_has_null_context(tmp_path, monkeypatch):
    calls = []
    def producer(**kwargs):
        calls.append(deepcopy(kwargs))
        return failed_test_report(**kwargs)
    monkeypatch.setattr(cli, "run_local_episode", producer)
    assert cli.main(arguments(tmp_path, "--scenario", "self_reinforcing_error", "--task", "recall_after_correction")) == 1
    assert len(calls) == 1
    assert calls[0]["prior_context"] is None
    assert calls[0]["previous_answers"] == {}


def test_cli_hands_off_actual_failed_answer_and_terminal_provenance_verbatim(tmp_path, monkeypatch):
    """The producer is patched only to test handoff, not establish live origin."""
    from daystrom_dml.contracts.model_input import (
        CompiledModelInput, ModelInputIdentity, ModelInputRequest, SUPPORTED_CHAT_TEMPLATE_DIGEST,
    )
    from daystrom_dml.services.agent_episode import build_episode_request

    calls, prior_terminals = [], []
    wrong_answer = {"claims": [{"key": "service.port", "value": 7000, "evidence_ids": [0]}]}
    def producer(**kwargs):
        calls.append(deepcopy(kwargs))
        if kwargs["task"]["id"] != "first_recall":
            return failed_test_report(**kwargs)
        events = [cli._started("prior-actual-terminal", kwargs["task"], kwargs["scenario"]["scope"],
                               kwargs["limits"], "live_local", prior_context=kwargs["prior_context"])]
        request = ModelInputRequest.from_payload(build_episode_request(kwargs["task"], limits=kwargs["limits"]))
        identity = ModelInputIdentity("1" * 64, "2" * 64, SUPPORTED_CHAT_TEMPLATE_DIGEST,
                                      "cli-handoff-fixture", 32768)
        compiled = CompiledModelInput(identity, request.request_digest, (4, 5, 6), (1, 1, 1),
                                      kwargs["limits"].output_tokens, 32768, "fixture-consumer", "nonce", "0" * 64)
        def append(kind, payload, call_id=None):
            events.append(cli.make_event(episode_id=events[0]["episode_id"], task_id=kwargs["task"]["id"],
                sequence=len(events), kind=kind, payload=payload, call_id=call_id))
        append("model_requested", {"step": 0, "request": request.to_payload(),
            "compiled": compiled.signing_payload(), "artifact_digest": compiled.artifact_digest}, "model-0")
        append("model_completed", {"step": 0, "artifact_digest": compiled.artifact_digest,
            "input_token_count": 3, "output_ids": [7, 8], "output_token_count": 2,
            "text": json.dumps({"schema_version": "dml-agent-action-v1", "kind": "final", "answer": wrong_answer}),
            "latency_ms": 1.0, "ttft_ms": None}, "model-0")
        verdict = {"verifier_version": cli.VERIFIER_VERSION, "success": False, "reasons": ["wrong_fact"],
            "contradictions": 1, "factual_outputs": 1, "false_memory_claims": 1, "recalled_claims": 1,
            "repeat_errors": None, "repeat_opportunities": None}
        terminal = cli.build_terminal(events, verdict, status="completed", latency_ms=2.0,
                                      retrieval_ms=0.0, answer=wrong_answer)
        append("terminal", terminal)
        prior_terminals.append(deepcopy(terminal))
        return {"events": events, "terminal": terminal, "prepared": {}, "current_records": [],
                "live_qualified": False, "consumer_profile": kwargs["consumer_profile"]}
    monkeypatch.setattr(cli, "run_local_episode", producer)
    assert cli.main(arguments(tmp_path, "--scenario", "self_reinforcing_error")) == 1
    assert len(prior_terminals) == 1
    previous = prior_terminals[0]
    assert previous["success"] is False
    assert calls[0]["prior_context"] is None
    assert calls[1]["prior_context"] == {key: previous[key]
                                        for key in ("episode_id", "task_id", "evidence_digest", "answer")}
    assert calls[1]["previous_answers"] == {"first_recall": wrong_answer}
    assert "truth" not in calls[1]["prior_context"]
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert artifact["episodes"][0]["terminal"] == previous
    assert artifact["episodes"][1]["events"][0]["payload"]["prior_context"] == calls[1]["prior_context"]


def test_unexpected_runner_escape_is_accounted_and_later_tasks_continue(tmp_path, monkeypatch):
    count = 0
    def broken(**kwargs):
        nonlocal count
        count += 1
        raise RuntimeError("private error contents")
    monkeypatch.setattr(cli, "run_local_episode", broken)
    assert cli.main(arguments(tmp_path, "--scenario", "self_reinforcing_error")) == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert artifact["summary"]["attempted_tasks"] == count == 2
    assert "private error contents" not in (tmp_path / "campaign.json").read_text()
    assert all(report["runner_error_code"] == "RuntimeError" for report in artifact["episodes"])


def test_retained_observed_events_do_not_imply_complete_execution_evidence(tmp_path, monkeypatch):
    def incomplete_startup(**kwargs):
        report = failed_test_report(**kwargs)
        del report["raw_evidence_incomplete"]
        return report
    monkeypatch.setattr(cli, "run_local_episode", incomplete_startup)
    assert cli.main(arguments(tmp_path, "--scenario", "near_duplicates")) == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert artifact["raw_evidence_complete"] is False


def test_cli_rejects_injected_execution_and_preserves_rejected_report(tmp_path, monkeypatch):
    def fake(**kwargs):
        report = failed_test_report(**kwargs)
        report["terminal"]["execution_path"] = "test_injected"
        return report
    monkeypatch.setattr(cli, "run_local_episode", fake)
    assert cli.main(arguments(tmp_path, "--scenario", "near_duplicates")) == 1
    report = json.loads((tmp_path / "campaign.json").read_text())["episodes"][0]
    assert report["terminal"]["status"] == "runner_error"
    assert report["raw_evidence_incomplete"] is True
    assert report["rejected_raw_report"]["terminal"]["execution_path"] == "test_injected"


@pytest.mark.parametrize("existing", ["work", "campaign.json"])
def test_existing_output_or_authority_refused_before_model_work(tmp_path, monkeypatch, existing):
    path = tmp_path / existing
    if existing == "work":
        path.mkdir()
        (path / "preserve.txt").write_text("original")
    else:
        path.write_text("original")
    monkeypatch.setattr(cli, "run_local_episode", lambda **kwargs: pytest.fail("must not dispatch"))
    assert cli.main(arguments(tmp_path, "--scenario", "near_duplicates")) == 2
    assert (path / "preserve.txt" if path.is_dir() else path).read_text() == "original"


@pytest.mark.parametrize("extra", [
    ["--max-steps", "0"], ["--max-steps", "65"], ["--output-tokens", "4097"],
    ["--wall-time-seconds", "nan"], ["--wall-time-seconds", "301"],
    ["--max-event-bytes", "16777217"], ["--task", "first_recall"],
    ["--scenario", "near_duplicates", "--task", "missing"],
    ["--max-st", "2"],
    ["--consumer-profile", "auto"], ["--consumer-profile", "Qwen2-instruct-v1"],
])
def test_invalid_or_unbounded_flags_refused_before_work(tmp_path, extra):
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments(tmp_path, *extra))
    assert caught.value.code == 2
    assert not (tmp_path / "work").exists()


def test_atomic_artifact_no_overwrite_or_partial_oversized_output(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="byte limit"):
        cli._atomic_json(path, {"large": "x" * 100}, max_bytes=20)
    assert not path.exists()
    assert not list(tmp_path.glob(".episode-*"))
    cli._atomic_json(path, {"valid": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli._atomic_json(path, {"overwrite": True})
    assert path.read_bytes() == before


def test_atomic_artifact_race_cannot_replace_competing_output(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    real_link = cli.os.link
    def competing_link(source, destination):
        Path(destination).write_text("competing artifact")
        return real_link(source, destination)
    monkeypatch.setattr(cli.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        cli._atomic_json(path, {"new": True})
    assert path.read_text() == "competing artifact"
    assert not list(tmp_path.glob(".episode-*"))


def test_actual_missing_snapshot_attempt_retains_failure_artifact(tmp_path):
    # No model weights, scripted actions or injected consumer qualify this
    # worker bootstrap failure as a live model run.
    result = cli.main(arguments(tmp_path, "--scenario", "near_duplicates", "--wall-time-seconds", "5"))
    assert result == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert artifact["summary"]["attempted_tasks"] == 1
    assert artifact["summary"]["completed_tasks"] == 0
    assert artifact["live_qualified"] is False
    assert artifact["episodes"][0]["terminal"]["status"] in ("runner_error", "timeout")
    assert len(list((tmp_path / "work").rglob("report.json"))) == 1


def test_explicit_qwen_profile_is_forwarded_and_retained_without_qualification(tmp_path, monkeypatch):
    calls = []
    def producer(**kwargs):
        calls.append(kwargs)
        return failed_test_report(**kwargs)
    monkeypatch.setattr(cli, "run_local_episode", producer)
    assert cli.main(arguments(tmp_path, "--scenario", "near_duplicates",
                              "--consumer-profile", "qwen2-instruct-v1")) == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    assert len(calls) == 1
    assert calls[0]["consumer_profile"] == artifact["consumer_profile"] == "qwen2-instruct-v1"
    assert artifact["episodes"][0]["consumer_profile"] == "qwen2-instruct-v1"
    assert artifact["live_qualified"] is artifact["source_ci_qualified"] is False


def test_cli_refuses_report_from_another_consumer_and_preserves_evidence(tmp_path, monkeypatch):
    def producer(**kwargs):
        report = failed_test_report(**kwargs)
        report["consumer_profile"] = "gpt2-v1"
        return report
    monkeypatch.setattr(cli, "run_local_episode", producer)
    assert cli.main(arguments(tmp_path, "--scenario", "near_duplicates",
                              "--consumer-profile", "qwen2-instruct-v1")) == 1
    artifact = json.loads((tmp_path / "campaign.json").read_text())
    report = artifact["episodes"][0]
    assert report["terminal"]["status"] == "runner_error"
    assert report["runner_error_code"] == "ValueError"
    assert report["consumer_profile"] == "qwen2-instruct-v1"
    assert report["rejected_raw_report"]["consumer_profile"] == "gpt2-v1"
    assert artifact["raw_evidence_complete"] is False


@pytest.mark.parametrize("profile", [None, True, [], "auto"])
def test_campaign_api_rejects_unknown_profile_before_any_work(tmp_path, profile, monkeypatch):
    monkeypatch.setattr(cli, "run_local_episode", lambda **kwargs: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="consumer profile"):
        cli.run_campaign(snapshot_directory=tmp_path / "missing", work_directory=tmp_path / "work",
                         output=tmp_path / "campaign.json", limits=cli.EpisodeLimits(), consumer_profile=profile)
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "campaign.json").exists()


def test_v2_runner_escape_retains_selected_protocol_without_known_zero_effects():
    from daystrom_dml.contracts.agent_episode import VALIDATION_CONSUMER_PROFILE, EXECUTION_PROTOCOL_V2
    from daystrom_dml.services.agent_episode import EpisodeLimits
    from scripts.agent_episodes import _interrupted_report
    corpus = cli.load_episode_corpus()
    scenario = corpus['scenarios'][0]
    report = _interrupted_report(scenario, scenario['tasks'][0], EpisodeLimits(), RuntimeError('lost'), 1,
                                 consumer_profile=VALIDATION_CONSUMER_PROFILE)
    assert report['events'][0]['schema_version'] == 'dml-agent-event-v2'
    assert report['events'][0]['payload']['execution_protocol'] == EXECUTION_PROTOCOL_V2
    assert report['terminal']['schema_version'] == 'dml-agent-terminal-v2'
    assert report['terminal']['usage_unknown'] and report['terminal']['effects_unknown']


def test_v2_campaign_cli_freezes_protocol_and_preserves_all_failed_attempts(tmp_path, monkeypatch):
    from daystrom_dml.contracts.agent_episode import VALIDATION_CONSUMER_PROFILE, EXECUTION_PROTOCOL_V2
    monkeypatch.setattr(cli, 'run_local_episode', failed_test_report)
    assert cli.main(arguments(tmp_path, '--consumer-profile', VALIDATION_CONSUMER_PROFILE)) == 1
    campaign = json.loads((tmp_path / 'campaign.json').read_text())
    assert campaign['schema_version'] == cli.CAMPAIGN_VERSION_V2
    assert campaign['execution_protocol'] == EXECUTION_PROTOCOL_V2
    assert len(campaign['episodes']) == 9
    assert all(e['terminal']['schema_version'] == 'dml-agent-terminal-v2' for e in campaign['episodes'])
    assert campaign['summary']['unknown_effect_tasks'] == 9
    assert not campaign['live_qualified'] and not campaign['raw_evidence_complete']


def test_v3_campaign_retains_explicit_profile_in_every_error_boundary(tmp_path, monkeypatch):
    from daystrom_dml.contracts.agent_episode import RECOVERY_CONSUMER_PROFILE, EXECUTION_PROTOCOL_V2
    calls = []
    def producer(**kwargs):
        calls.append(kwargs)
        return failed_test_report(**kwargs)
    monkeypatch.setattr(cli, 'run_local_episode', producer)
    assert cli.main(arguments(tmp_path, '--consumer-profile', RECOVERY_CONSUMER_PROFILE)) == 1
    campaign = json.loads((tmp_path / 'campaign.json').read_text())
    assert campaign['schema_version'] == cli.CAMPAIGN_VERSION_V2
    assert campaign['execution_protocol'] == EXECUTION_PROTOCOL_V2
    assert campaign['consumer_profile'] == campaign['summary']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
    assert len(calls) == len(campaign['episodes']) == 9
    assert all(call['consumer_profile'] == RECOVERY_CONSUMER_PROFILE for call in calls)
    for episode in campaign['episodes']:
        assert episode['consumer_profile'] == episode['terminal']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
        assert episode['events'][0]['payload']['consumer_profile'] == RECOVERY_CONSUMER_PROFILE
        assert episode['terminal']['usage_unknown'] and episode['terminal']['effects_unknown']
    assert not campaign['raw_evidence_complete']


@pytest.mark.parametrize('requested', ['qwen2-action-json-recovery-v3', 'qwen2-action-json-validation-v2'])
def test_v3_cli_rejects_cross_profile_error_report_even_with_matching_outer_label(tmp_path, monkeypatch, requested):
    other = 'qwen2-action-json-validation-v2' if requested.endswith('v3') else 'qwen2-action-json-recovery-v3'
    def producer(**kwargs):
        kwargs['consumer_profile'] = other
        report = failed_test_report(**kwargs)
        report['consumer_profile'] = requested
        return report
    monkeypatch.setattr(cli, 'run_local_episode', producer)
    assert cli.main(arguments(tmp_path, '--scenario', 'near_duplicates', '--consumer-profile', requested)) == 1
    episode = json.loads((tmp_path / 'campaign.json').read_text())['episodes'][0]
    assert episode['consumer_profile'] == episode['terminal']['consumer_profile'] == requested
    assert episode['rejected_raw_report']['terminal']['consumer_profile'] == other
    assert episode['runner_error_code'] == 'ValueError'


def test_qwen3_cli_selects_new_source_inventory_and_preserves_all_failed_attempts(tmp_path, monkeypatch):
    from daystrom_dml.contracts.agent_episode import QWEN3_CONSUMER_PROFILE
    monkeypatch.setattr(cli, 'run_local_episode', failed_test_report)
    assert cli.main(arguments(tmp_path, '--consumer-profile', QWEN3_CONSUMER_PROFILE)) == 1
    campaign = json.loads((tmp_path / 'campaign.json').read_text())
    assert campaign['consumer_profile'] == campaign['summary']['consumer_profile'] == QWEN3_CONSUMER_PROFILE
    assert campaign['schema_version'] == cli.CAMPAIGN_VERSION_V2
    assert len(campaign['source_sha256']) == 19 and len(cli._source_digests()) == 15
    assert len(campaign['episodes']) == 9
    assert all(r['terminal']['consumer_profile'] == QWEN3_CONSUMER_PROFILE for r in campaign['episodes'])
    assert not campaign['raw_evidence_complete'] and not campaign['live_qualified']
