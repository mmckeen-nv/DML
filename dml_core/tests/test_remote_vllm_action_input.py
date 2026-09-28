"""Network-free adversarial adapter checks; synthetic tokenizer never claims model evidence."""
import copy
from dataclasses import replace
import hashlib
import json
import sys
import types

import pytest

from daystrom_dml.contracts.agent_episode import initial_messages, episode_tool_definitions
from daystrom_dml.contracts.model_input import ModelInputError, ModelInputBudgetError
from daystrom_dml.services.model_input import ModelInputExecutionError
from daystrom_dml.services.remote_vllm_action_input import (
    CONSUMER_PROFILE, RemoteVLLMActionInputConsumer, sampling_policy_identity, verify_remote_manifest, client_runtime_versions,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, ids, **kwargs):
        return '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}'

    def __len__(self):
        return 100


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer())))
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat-template.jinja'):
        (tmp_path / name).write_text('{}')
        files[name] = hashlib.sha256(b'{}').hexdigest()
    manifest = {'schema_version': 'dml-remote-vllm-manifest-v1',
        'endpoint': 'http://192.168.50.91:8000/v1', 'model': 'nvidia/nemotron-3-super',
        'client_runtime_versions': client_runtime_versions(), 'model_revision': 'synthetic-unattested', 'model_provenance': {'test': True},
        'server_configuration': {'test': True}, 'sampling': sampling_policy_identity(),
        'model_window_tokens': 100, 'timeout_seconds': 2, 'files': files,
        'evidence_directory': str(tmp_path / 'http')}
    (tmp_path / 'remote-vllm-manifest.json').write_text(json.dumps(manifest))
    with RemoteVLLMActionInputConsumer(tmp_path, offline=True) as instance:
        yield instance


def compile_input(consumer):
    return consumer.compile(initial_messages('Synthetic request', consumer_profile=CONSUMER_PROFILE),
        episode_tool_definitions(), output_reserved_tokens=10)


def response():
    return {'model': 'nvidia/nemotron-3-super', 'choices': [{'index': 0,
        'prompt_token_ids': [1, 2, 3], 'token_ids': [4, 5], 'text': '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}', 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


def test_exact_ids_and_unchanged_schema(consumer, monkeypatch):
    artifact = compile_input(consumer)
    sent = []
    monkeypatch.setattr(consumer, '_post', lambda path, payload: sent.append(payload) or response())
    result = consumer.execute(artifact)
    assert result.output_ids == (4, 5)
    assert sent[0]['prompt'] == [1, 2, 3]
    assert sent[0]['max_tokens'] == 10
    assert sent[0]['structured_outputs']['json']['anyOf']
    assert sent[0]['temperature'] == 0.7


@pytest.mark.parametrize('mutation', ['missing_output', 'prefix', 'usage', 'text', 'budget', 'bool_token', 'model', 'finish'])
def test_rejects_false_evidence(consumer, monkeypatch, mutation):
    artifact = compile_input(consumer)
    reply = response()
    choice = reply['choices'][0]
    if mutation == 'missing_output':
        del choice['token_ids']
    elif mutation == 'prefix':
        choice['prompt_token_ids'] = [9, 2, 3]
    elif mutation == 'usage':
        reply['usage']['total_tokens'] = 99
    elif mutation == 'text':
        choice['text'] = 'fabricated'
    elif mutation == 'budget':
        choice['token_ids'] = [4] * 11
    elif mutation == 'bool_token':
        choice['token_ids'] = [True]
    elif mutation == 'model':
        reply['model'] = 'other'
    else:
        choice['finish_reason'] = 'error'
    monkeypatch.setattr(consumer, '_post', lambda *args: reply)
    with pytest.raises(ModelInputExecutionError):
        consumer.execute(artifact)


def test_ownership_and_budget(consumer):
    artifact = compile_input(consumer)
    with pytest.raises(ModelInputError):
        consumer.execute(replace(artifact, nonce='forged'))
    with pytest.raises(ModelInputBudgetError):
        consumer.compile(initial_messages('x', consumer_profile=CONSUMER_PROFILE), [], output_reserved_tokens=99)
    with pytest.raises(ModelInputExecutionError):
        consumer.execute(artifact)  # Offline mode must never contact HTTP.


def test_manifest_mutation_fails_closed(consumer):
    artifact = compile_input(consumer)
    (consumer._directory / 'tokenizer.json').write_text('changed')
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    with pytest.raises(ModelInputError):
        verify_remote_manifest(consumer._directory)


def test_server_tokenizer_disagreement(consumer, monkeypatch):
    consumer._offline = False
    monkeypatch.setattr(consumer, '_post', lambda *args: {'tokens': [9], 'count': 1, 'max_model_len': 100})
    with pytest.raises(ModelInputError):
        compile_input(consumer)


def test_timeout_logged_no_retry(consumer, monkeypatch):
    from daystrom_dml.services import remote_vllm_action_input as module
    consumer._offline = False
    calls = []
    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError('synthetic timeout')
    monkeypatch.setattr(module.http, 'urlopen', timeout)
    with pytest.raises(TimeoutError):
        consumer._post('/v1/completions', {'synthetic': True})
    assert len(calls) == 1
    assert consumer.last_exchange['timed_out'] is True
    assert json.loads(__import__('pathlib').Path(consumer.last_exchange['evidence_path']).read_text())['error']


def test_replay_exchange_rejects_sampling_tamper(consumer):
    artifact = compile_input(consumer)
    request = consumer._bound_requests[artifact.artifact_digest].to_payload()
    exchange = {'path': '/v1/completions', 'error': None, 'timed_out': False, 'status': 200,
        'timeout_seconds': 2, 'request': consumer.request_payload(artifact.input_ids, 10, request['tools']),
        'response': response(), 'response_raw': json.dumps(response())}
    completed = {'output_ids': [4, 5], 'text': '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}', 'input_token_count': 3, 'output_token_count': 2}
    consumer.validate_exchange(artifact.signing_payload(), request, exchange, completed)
    bad = copy.deepcopy(exchange)
    bad['request']['temperature'] = 1.0
    with pytest.raises(ValueError):
        consumer.validate_exchange(artifact.signing_payload(), request, bad, completed)


def test_stale_exchange_not_attached_to_auth_failure(consumer):
    artifact = compile_input(consumer)
    consumer.last_exchange = {'stale': True}
    with pytest.raises(ModelInputError):
        consumer.execute(replace(artifact, nonce='forged'))
    assert consumer.last_exchange is None


def test_stale_exchange_not_attached_to_compile_failure(consumer):
    consumer.last_exchange = {'stale': True}
    with pytest.raises(ModelInputError):
        consumer.compile([], [], output_reserved_tokens=3)
    assert consumer.last_exchange is None


def test_replay_retains_authentic_rejected_output(consumer, monkeypatch):
    artifact = compile_input(consumer)
    request = consumer._bound_requests[artifact.artifact_digest].to_payload()
    reply = response()
    reply['choices'][0]['text'] = '{"ok":true}'
    monkeypatch.setattr(consumer, 'decode_output', lambda ids: '{"ok":true}')
    exchange = {'path': '/v1/completions', 'error': None, 'timed_out': False, 'status': 200,
        'timeout_seconds': 2, 'request': consumer.request_payload(artifact.input_ids, 10, request['tools']),
        'response': reply, 'response_raw': json.dumps(reply)}
    completed = {'output_ids': [4, 5], 'text': '{"ok":true}', 'input_token_count': 3, 'output_token_count': 2}
    consumer.validate_exchange(artifact.signing_payload(), request, exchange, completed)
