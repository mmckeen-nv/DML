"""Run the packaged bounded corpus through the actual local model consumer.

Every returned attempt is checkpointed, including failures. This entry point has
no injected backend or download path and does not establish trained-model or
live semantic qualification from a successful process exit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from daystrom_dml.atomic_io import _sync_directory
from daystrom_dml.contracts.agent_episode import (
    make_event, validate_episode_events, execution_protocol_for_profile, EXECUTION_PROTOCOL_V2,
)
from daystrom_dml.services.agent_episode import (
    CONSUMER_PROFILES, EpisodeLimits, _started, run_local_episode, validate_consumer_profile,
)
from daystrom_dml.services.episode_outcomes import build_terminal, summarize_episode_outcomes
from daystrom_dml.services.episode_verifiers import INTENTS, VERIFIER_VERSION, load_episode_corpus

CAMPAIGN_VERSION = "dml-agent-campaign-v1"
CAMPAIGN_VERSION_V2 = "dml-agent-campaign-v2"
MAX_CAMPAIGN_BYTES = 256 * 1024 * 1024


def _atomic_json(path, value, *, max_bytes=MAX_CAMPAIGN_BYTES):
    """Publish a complete bounded JSON file atomically, never replacing a file."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".episode-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            written = 0
            encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
            for chunk in encoder.iterencode(value):
                raw = chunk.encode("utf-8")
                written += len(raw)
                if written + 1 > max_bytes:
                    raise ValueError("Campaign artifact exceeds its byte limit")
                stream.write(raw)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() atomically refuses an existing destination,
        # including a race after the preliminary path check.
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_digests():
    import daystrom_dml.contracts.agent_episode as contract
    import daystrom_dml.services.agent_episode as runner
    import daystrom_dml.services.episode_tools as gateway
    import daystrom_dml.services.episode_verifiers as verifier
    import daystrom_dml.services.episode_outcomes as reducer
    files = {module.__name__: Path(module.__file__) for module in (contract, runner, gateway, verifier, reducer)}
    files["scripts.agent_episodes"] = Path(__file__)
    for name in ("model_input", "model_input_snapshot", "pretrained_snapshot", "qwen_model_input",
                 "qwen_model_snapshot", "qwen_pretrained_snapshot", "qwen_action_input", "agent_action_grammar"):
        files["daystrom_dml.services." + name] = Path(runner.__file__).with_name(name + ".py")
    files["scripts.agent_campaign_evidence"] = Path(__file__).with_name("agent_campaign_evidence.py")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def _interrupted_report(scenario, task, limits, exc, elapsed_ms, *, prior_context=None,
                        consumer_profile="gpt2-v1"):
    """Account for an unexpected runner escape without inventing lost events."""
    ident = "episode-" + uuid.uuid4().hex
    protocol = execution_protocol_for_profile(consumer_profile)
    events = [_started(ident, task, scenario["scope"], limits, "live_local", prior_context=prior_context,
                       consumer_profile=consumer_profile)]
    verdict = {"verifier_version": VERIFIER_VERSION, "success": False,
               "reasons": ["runner_escaped_before_report"], "contradictions": None,
               "factual_outputs": None, "false_memory_claims": None, "recalled_claims": None,
               "repeat_errors": None, "repeat_opportunities": None}
    terminal = build_terminal(events, verdict, status="runner_error", latency_ms=elapsed_ms,
                              retrieval_ms=None, usage_unknown=True)
    events.append(make_event(episode_id=ident, task_id=task["id"], sequence=len(events),
                             kind="terminal", payload=terminal, execution_protocol=protocol))
    return {"events": events, "terminal": terminal, "prepared": None, "current_records": None,
            "live_qualified": False, "raw_evidence_incomplete": True,
            "runner_error_code": type(exc).__name__, "consumer_profile": consumer_profile}


def _selected(corpus, scenario_id=None, task_id=None):
    if task_id is not None and scenario_id is None:
        raise ValueError("--task requires --scenario")
    selected = [(scenario, task) for scenario in corpus["scenarios"]
                if scenario_id is None or scenario["id"] == scenario_id
                for task in scenario["tasks"] if task_id is None or task["id"] == task_id]
    if not selected:
        raise ValueError("No matching corpus task")
    return selected


def run_campaign(*, snapshot_directory, work_directory, output, limits,
                 scenario_id=None, task_id=None, consumer_profile="gpt2-v1"):
    """Run the selected finite tasks; all code paths use run_local_episode."""
    validate_consumer_profile(consumer_profile)
    protocol = execution_protocol_for_profile(consumer_profile)
    corpus = load_episode_corpus()
    selected = _selected(corpus, scenario_id, task_id)
    output = Path(output).expanduser().absolute()
    directory = Path(work_directory).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Work directory must be fresh")
    directory.mkdir(parents=True, exist_ok=False)
    source_digests = _source_digests()
    episodes, previous, prior_terminals = [], {}, {}
    for scenario, task in selected:
        attempt_directory = directory / scenario["id"] / task["id"]
        start = time.monotonic()
        raw_report = None
        prior_terminal = prior_terminals.get(scenario["id"], {}).get(task.get("repeat_from"))
        prior_context = None
        if prior_terminal is not None:
            prior_context = deepcopy({key: prior_terminal[key]
                                      for key in ("episode_id", "task_id", "evidence_digest", "answer")})
        try:
            report = run_local_episode(snapshot_directory=snapshot_directory,
                work_directory=attempt_directory, scenario=scenario, task=task, limits=limits,
                effective_time=corpus["effective_time"], previous_answers=previous.get(scenario["id"], {}),
                prior_context=prior_context, consumer_profile=consumer_profile)
            raw_report = report
            validate_episode_events(report["events"])
            if report["events"][-1]["payload"] != report["terminal"]:
                raise ValueError("Returned terminal differs from raw event")
            if report["terminal"]["execution_path"] != "live_local":
                raise ValueError("CLI requires concrete local execution")
            if report.get("consumer_profile") != consumer_profile:
                raise ValueError("Returned consumer profile differs from requested implementation")
            if protocol == EXECUTION_PROTOCOL_V2 and any(value.get("consumer_profile") != consumer_profile
                    for value in (report["events"][0]["payload"], report["terminal"])):
                raise ValueError("Returned boundary consumer profile differs from requested implementation")
            if report["events"][0]["payload"].get("execution_protocol") != (
                    protocol if protocol == EXECUTION_PROTOCOL_V2 else None):
                raise ValueError("Returned execution protocol differs from requested implementation")
        except Exception as exc:
            report = _interrupted_report(scenario, task, limits, exc,
                max(0.0, (time.monotonic() - start) * 1000), prior_context=prior_context,
                consumer_profile=consumer_profile)
            if raw_report is not None:
                report["rejected_raw_report"] = raw_report
        report = {"scenario_id": scenario["id"], **report}
        episodes.append(report)
        # Retain prior actual structured output, including independently wrong
        # answers, so the correction task never receives a scripted answer.
        previous.setdefault(scenario["id"], {})[task["id"]] = report["terminal"].get("answer")
        prior_terminals.setdefault(scenario["id"], {})[task["id"]] = deepcopy(report["terminal"])
        _atomic_json(attempt_directory / "report.json", report)
    summary = summarize_episode_outcomes([episode["terminal"] for episode in episodes])
    corpus_raw = json.dumps(corpus, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    artifact = {"schema_version": CAMPAIGN_VERSION_V2 if protocol == EXECUTION_PROTOCOL_V2 else CAMPAIGN_VERSION,
                "consumer_profile": consumer_profile,
                "corpus_digest": hashlib.sha256(corpus_raw).hexdigest(),
                "source_sha256": source_digests, "limits": asdict(limits),
                "selection": [{"scenario_id": scenario["id"], "task_id": task["id"]}
                              for scenario, task in selected],
                "snapshot_directory": str(Path(snapshot_directory).expanduser().absolute()),
                "ranking_scope": "synthetic_fixture", "live_qualified": False,
                "source_ci_qualified": False,
                "raw_evidence_complete": all(
                    not episode.get("raw_evidence_incomplete") and bool(episode.get("prepared"))
                    and episode["events"][0]["payload"].get("seed_receipts_digest") is not None
                    and not episode["terminal"]["usage_unknown"]
                    and not episode["terminal"]["effects_unknown"]
                    and not episode["terminal"]["unknown_input_calls"]
                    and not episode["terminal"]["unknown_output_calls"] for episode in episodes),
                "episodes": episodes, "summary": summary}
    if protocol == EXECUTION_PROTOCOL_V2:
        artifact["execution_protocol"] = protocol
    _atomic_json(output, artifact)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--snapshot-directory", required=True, type=Path)
    parser.add_argument("--work-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--consumer-profile", choices=CONSUMER_PROFILES, default="gpt2-v1")
    parser.add_argument("--scenario", choices=INTENTS)
    parser.add_argument("--task")
    defaults = EpisodeLimits()
    for name in ("max_steps", "output_tokens", "max_input_tokens", "max_output_tokens",
                 "max_transcript_bytes", "max_event_bytes", "max_episode_bytes"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--wall-time-seconds", type=float, default=defaults.wall_time_seconds)
    args = parser.parse_args(argv)
    try:
        limits = EpisodeLimits(**{name: getattr(args, name) for name in asdict(defaults)})
        _selected(load_episode_corpus(), args.scenario, args.task)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        artifact = run_campaign(snapshot_directory=args.snapshot_directory,
            work_directory=args.work_directory, output=args.output, limits=limits,
            scenario_id=args.scenario, task_id=args.task, consumer_profile=args.consumer_profile)
    except (OSError, ValueError) as exc:
        print("Agent campaign failed: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "attempted_tasks": artifact["summary"]["attempted_tasks"],
                      "completed_tasks": artifact["summary"]["completed_tasks"], "live_qualified": False}))
    return 0 if all(episode["terminal"]["success"] for episode in artifact["episodes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
