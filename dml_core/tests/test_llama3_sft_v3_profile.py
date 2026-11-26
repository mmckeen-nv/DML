"""The integrated SFT profile cannot reinterpret historical runtime evidence."""

from copy import deepcopy
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import (
    EVENT_VERSION_V2,
    EXECUTION_PROTOCOL_V2,
    LLAMA3_SFT_V3_CONSUMER_PROFILE as PROFILE,
    QWEN3_GGUF_CUDA_RETRIEVAL_CONSUMER_PROFILE as POLICY_SOURCE,
    execution_protocol_for_profile,
    initial_messages,
    make_event,
    validate_episode_events,
)
from daystrom_dml.contracts.model_input import CompiledModelInput, ModelInputIdentity, ModelInputRequest
from daystrom_dml.services import agent_episode as runner

PREFIX = "dml-llama3-sft-action-runtime-v2:"


def test_explicit_profile_dispatch_preserves_historical_profiles(monkeypatch):
    calls, sentinel = [], object()

    def construct(path, *, consumer_profile):
        calls.append((path, consumer_profile))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "daystrom_dml.services.llama3_sft_v3_action_input",
        SimpleNamespace(LocalLlama3SFTV3ActionInputConsumer=construct),
    )
    assert runner._open_consumer("candidate", PROFILE) is sentinel
    assert calls == [("candidate", PROFILE)]
    assert execution_protocol_for_profile(PROFILE) == EXECUTION_PROTOCOL_V2
    with pytest.raises(ValueError, match="Unknown"):
        runner._open_consumer("candidate", PROFILE + "-disabled")


def test_profile_preserves_trained_policy_tools_and_original_limits():
    task = {"id": "case", "prompt": "Read two facts", "state_expectations": []}
    actual = runner.build_episode_request(task, consumer_profile=PROFILE)
    original = runner.build_episode_request(task, consumer_profile=POLICY_SOURCE)
    assert actual == original
    assert initial_messages(task["prompt"], consumer_profile=PROFILE) == initial_messages(
        task["prompt"], consumer_profile=POLICY_SOURCE
    )
    assert runner.EpisodeLimits().max_steps == 6


def _event_prefix(profile=PROFILE, runtime=PREFIX + "a" * 64):
    limits = runner.EpisodeLimits(output_tokens=16)
    task = {"id": "case", "prompt": "Read two facts"}
    start = runner._started(
        "episode",
        task,
        {"tenant_id": "tenant", "client_id": None, "session_id": None, "instance_id": None},
        limits,
        "test_injected",
        "9" * 64,
        consumer_profile=profile,
    )
    request = ModelInputRequest.from_payload(
        runner.build_episode_request(task, limits=limits, consumer_profile=profile)
    )
    identity = ModelInputIdentity("1" * 64, "2" * 64, "3" * 64, runtime, 8192)
    artifact = CompiledModelInput(
        identity, request.request_digest, (4, 5, 6), (1, 1, 1), 16, 8192, "consumer", "nonce", "0" * 64
    )
    call = make_event(
        episode_id="episode",
        task_id="case",
        sequence=1,
        kind="model_requested",
        call_id="model-0",
        execution_protocol=execution_protocol_for_profile(profile),
        payload={
            "step": 0,
            "request": request.to_payload(),
            "compiled": artifact.signing_payload(),
            "artifact_digest": artifact.artifact_digest,
        },
    )
    return [start, call]


def test_standard_v2_accepts_only_selected_runtime():
    events = _event_prefix()
    assert all(e["schema_version"] == EVENT_VERSION_V2 for e in events)
    validate_episode_events(events, require_terminal=False)
    for runtime in (
        "diagnostic-meta-llama3-8b-bf16-lora-sft-grammar-v2:" + "a" * 64,
        "dml-llama3-sft-action-runtime-v1:" + "a" * 64,
        "dml-qwen3-gguf-cuda-action-runtime-v2:" + "a" * 64,
    ):
        with pytest.raises(ValueError, match="runtime"):
            validate_episode_events(_event_prefix(runtime=runtime), require_terminal=False)


@pytest.mark.parametrize("profile", ["gpt2-v1", POLICY_SOURCE, "llama3-8b-instruct-sft-v2-bf16-action-json-v1"])
def test_historical_protocols_reject_new_sft_identity(profile):
    with pytest.raises(ValueError, match="runtime"):
        validate_episode_events(_event_prefix(profile=profile), require_terminal=False)


def test_profile_cannot_silently_lose_retrieval_policy():
    events = deepcopy(_event_prefix())
    events[1]["payload"]["request"]["messages"][0] = initial_messages("unused")[0]
    request = ModelInputRequest.from_payload(events[1]["payload"]["request"])
    from daystrom_dml.contracts.agent_episode import _compiled

    artifact = replace(_compiled(events[1]["payload"]["compiled"]), request_digest=request.request_digest)
    events[1]["payload"]["compiled"] = artifact.signing_payload()
    events[1]["payload"]["artifact_digest"] = artifact.artifact_digest
    with pytest.raises(ValueError):
        validate_episode_events(events, require_terminal=False)


def _failed_prefix(*, profile=PROFILE):
    events = _event_prefix(profile=profile)
    request = events[1]['payload']
    retained = {'artifact_digest': request['artifact_digest'],
        'exception_type': 'ModelInputExecutionError', 'exception_message': 'authentic failure',
        'runtime_result': {'input_ids': request['compiled']['input_ids'],
            'model_identity': request['compiled']['identity'], 'output_ids': [42],
            'raw_text': '{', 'input_token_count': None, 'output_token_count': None,
            'usage_unknown': True, 'execution_error': 'RuntimeError: device failure'}}
    events.append(make_event(episode_id='episode', task_id='case', sequence=2, kind='model_failed',
        call_id='model-0', execution_protocol=execution_protocol_for_profile(profile),
        payload={'step': 0, 'phase': 'execute', 'error_code': 'ModelInputExecutionError',
            'input_token_count': None, 'output_token_count': None, 'latency_ms': 1, 'ttft_ms': None,
            'local_execution': retained}))
    return events


def test_failed_partial_output_is_retained_without_consumption_credit():
    events = _failed_prefix()
    validate_episode_events(events, require_terminal=False)
    from daystrom_dml.services.episode_outcomes import build_terminal
    from test_agent_episode_contract import verdict
    terminal = build_terminal(events, verdict(False), status='model_error', latency_ms=2, retrieval_ms=0)
    assert terminal['input_tokens'] is None and terminal['output_tokens'] is None
    assert terminal['unknown_input_calls'] == terminal['unknown_output_calls'] == 1
    assert events[-1]['payload']['local_execution']['runtime_result']['output_ids'] == [42]


@pytest.mark.parametrize('change', ['missing', 'artifact', 'identity', 'input', 'prefix_limit', 'known_count'])
def test_failed_local_prefix_cannot_be_detached_or_counted_as_completion(change):
    events = _failed_prefix()
    payload = events[-1]['payload']
    raw = payload['local_execution']['runtime_result']
    if change == 'missing':
        del payload['local_execution']
    elif change == 'artifact':
        payload['local_execution']['artifact_digest'] = 'f' * 64
    elif change == 'identity':
        raw['model_identity']['model_digest'] = 'e' * 64
    elif change == 'input':
        raw['input_ids'] = [4]
    elif change == 'prefix_limit':
        raw['output_ids'] = [42] * 17
    else:
        payload['output_token_count'] = 1
    with pytest.raises(ValueError):
        validate_episode_events(events, require_terminal=False)


def test_old_profile_cannot_use_new_failed_local_payload():
    events = _failed_prefix(profile=POLICY_SOURCE)
    with pytest.raises(ValueError):
        validate_episode_events(events, require_terminal=False)


@pytest.mark.parametrize('status', ['timeout', 'killed', 'runner_error'])
def test_parent_interrupt_retains_unavailable_runtime_and_unknown_usage(status):
    events = _event_prefix()
    assert runner._interrupt_pending(events, status=status)
    assert not runner._interrupt_pending(events, status=status)
    validate_episode_events(events, require_terminal=False)
    retained = events[-1]['payload']['local_execution']
    assert retained['runtime_result'] is None
    assert retained['artifact_digest'] == events[1]['payload']['artifact_digest']
    assert events[-1]['payload']['input_token_count'] is None


def test_post_return_integrity_failure_retains_raw_result_without_success_credit():
    events = _failed_prefix()
    retained = events[-1]['payload']['local_execution']
    retained['runtime_result'].update(execution_error=None, usage_unknown=False,
        input_token_count=3, output_token_count=1)
    retained['exception_type'] = events[-1]['payload']['error_code'] = 'ValueError'
    retained['exception_message'] = 'Model result differs from the dispatched artifact'
    validate_episode_events(events, require_terminal=False)
    assert events[-1]['payload']['input_token_count'] is None
    assert events[-1]['payload']['output_token_count'] is None


def test_decode_failure_retains_ids_without_fabricated_text():
    events = _failed_prefix()
    raw = events[-1]['payload']['local_execution']['runtime_result']
    raw.update(raw_text=None, decode_error='ValueError: tokenizer failed')
    validate_episode_events(events, require_terminal=False)
    assert raw['output_ids'] == [42] and raw['raw_text'] is None
    del raw['decode_error']
    with pytest.raises(ValueError):
        validate_episode_events(events, require_terminal=False)


def _acknowledged_stalled_sft_worker(connection, config):
    """Dispatch-boundary control only: no model constructor or inference."""
    import time
    from daystrom_dml.contracts.agent_episode import canonical_json
    def send(value):
        connection.send_bytes(canonical_json(value))
        assert connection.recv_bytes(128) == b'ack'
    send({'kind': 'prepared', 'prepared': config['prepared']})
    request = ModelInputRequest.from_payload(runner.build_episode_request(
        config['task'], limits=runner.EpisodeLimits(**config['limits']), consumer_profile=PROFILE))
    identity = ModelInputIdentity('1' * 64, '2' * 64, '3' * 64, PREFIX + 'a' * 64, 8192)
    artifact = CompiledModelInput(identity, request.request_digest, (4, 5, 6), (1, 1, 1),
        request.output_reserved_tokens, 8192, 'consumer', 'nonce', '0' * 64)
    event = make_event(episode_id=config['episode_id'], task_id=config['task']['id'], sequence=1,
        kind='model_requested', call_id='model-0', execution_protocol=EXECUTION_PROTOCOL_V2,
        payload={'step': 0, 'request': request.to_payload(), 'compiled': artifact.signing_payload(),
                 'artifact_digest': artifact.artifact_digest})
    send({'kind': 'event', 'event': event})
    time.sleep(30)


def test_real_supervisor_timeout_retains_sft_dispatch_and_unavailable_response(tmp_path):
    from test_agent_episode_adversarial import _supervisor_config
    config = _supervisor_config(tmp_path, seconds=3)
    config['consumer_profile'] = PROFILE
    adapter, config['prepared'] = runner._prepare_fixture(
        config['authority_directory'], config['scenario'], config['episode_id'])
    adapter.close()
    report = runner._supervise(config, worker_target=_acknowledged_stalled_sft_worker)
    assert report['terminal']['status'] == 'timeout'
    assert report['terminal']['input_tokens'] is None
    assert report['terminal']['output_tokens'] is None
    failures = [e for e in report['events'] if e['kind'] == 'model_failed']
    assert len(failures) == 1
    assert failures[0]['payload']['local_execution']['runtime_result'] is None
    validate_episode_events(report['events'])
    assert report['live_qualified'] is False
