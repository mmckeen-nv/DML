"""Remote protocol controls: synthetic actions never establish live quality."""
from copy import deepcopy
from dataclasses import replace
import json
import os

import pytest

from daystrom_dml.contracts.agent_episode import (
    AgentEpisodeError, REMOTE_VLLM_CONSUMER_PROFILE, validate_episode_events,
)
from daystrom_dml.services.agent_episode import _persist_frame
from test_agent_episode_runtime import ValidationScriptedConsumer, validation_case


class RemoteSyntheticConsumer(ValidationScriptedConsumer):
    last_exchange = {"test_only": True}

    def compile(self, *args, **kwargs):
        artifact = super().compile(*args, **kwargs)
        return replace(artifact, identity=replace(artifact.identity,
            runtime_identity="dml-remote-vllm-action-runtime-v1:" + "a" * 64))


def remote_case(tmp_path, **kwargs):
    return validation_case(tmp_path, consumer_profile=REMOTE_VLLM_CONSUMER_PROFILE,
                           consumer_factory=RemoteSyntheticConsumer, **kwargs)


def test_remote_preserves_same_record_rejection_and_model_supersession(tmp_path):
    report, _, _ = remote_case(tmp_path)
    assert report["terminal"]["success"]
    assert any(e["kind"] == "tool_validation_rejected" for e in report["events"])
    assert any(e["kind"] == "tool_completed" and e["payload"]["name"] == "supersede"
               for e in report["events"])
    validate_episode_events(report["events"])
    forged = deepcopy(report["events"])
    next(e for e in forged if e["kind"] == "model_completed")["payload"].pop("remote_evidence")
    with pytest.raises(AgentEpisodeError, match="exchange evidence"):
        validate_episode_events(forged)


def test_remote_transport_failure_preserves_unknown_input_and_output(tmp_path):
    class FailedRemote(RemoteSyntheticConsumer):
        def execute(self, artifact):
            self.last_exchange = {"error": "TimeoutError", "timeout": True}
            raise TimeoutError("test transport")
    report, _, _ = validation_case(tmp_path, consumer_profile=REMOTE_VLLM_CONSUMER_PROFILE,
                                    consumer_factory=FailedRemote)
    terminal = report["terminal"]
    assert terminal["status"] == "model_error"
    assert terminal["input_tokens"] is None and terminal["output_tokens"] is None
    assert terminal["unknown_input_calls"] == terminal["unknown_output_calls"] == 1
    failure = next(e for e in report["events"] if e["kind"] == "model_failed")
    assert failure["payload"]["remote_evidence"]["timeout"]
    validate_episode_events(report["events"])


def test_frame_journal_fsync_and_no_symlink(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    path = tmp_path / "events.jsonl"
    _persist_frame({"journal_path": str(path)}, {"kind": "event", "event": {"sequence": 1}})
    assert json.loads(path.read_text())["event"]["sequence"] == 1
    assert calls
    assert path.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "alias"
    link.symlink_to(path)
    with pytest.raises(OSError):
        _persist_frame({"journal_path": str(link)}, {"kind": "event"})


def test_remote_replay_rejects_changed_prompt_ids(tmp_path):
    from types import SimpleNamespace
    from scripts.agent_campaign_evidence import _replay_remote_model
    report, _, _ = remote_case(tmp_path)
    requested = next(e for e in report["events"] if e["kind"] == "model_requested")
    identity = requested["payload"]["compiled"]["identity"]
    completed = next(e for e in report["events"] if e["kind"] == "model_completed")
    class Replay:
        def compile(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1, 2, 3, 4], attention_mask=[1, 1, 1, 1])
        def decode_output(self, ids):
            return completed["payload"]["text"]
        def validate_exchange(self, compiled, request, exchange, result):
            assert exchange == {"test_only": True}
    assert _replay_remote_model([requested, completed], identity, Replay()) == 1
    changed = deepcopy(requested)
    changed["payload"]["compiled"]["input_ids"][0] = 100
    with pytest.raises(ValueError, match="independent frozen tokenization"):
        _replay_remote_model([changed, completed], identity, Replay())


def test_remote_replay_retains_schema_invalid_model_failure(tmp_path):
    from types import SimpleNamespace
    from scripts.agent_campaign_evidence import _replay_remote_model
    report, _, _ = remote_case(tmp_path)
    requested = next(e for e in report["events"] if e["kind"] == "model_requested")
    completed = deepcopy(next(e for e in report["events"] if e["kind"] == "model_completed"))
    completed["payload"]["text"] = '{"schema_version":"dml-agent-action-v1","kind":"arbitrary"}'
    class Replay:
        def compile(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1, 2, 3, 4], attention_mask=[1, 1, 1, 1])
        def decode_output(self, ids):
            return completed["payload"]["text"]
        def validate_exchange(self, *args):
            pass
    assert _replay_remote_model([requested, completed], requested["payload"]["compiled"]["identity"], Replay()) == 1
