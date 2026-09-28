"""Native transport controls: no HTTP or model generation in these tests."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import initial_messages, episode_tool_definitions
from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services.model_input import ModelInputExecutionError
from daystrom_dml.services import native_remote_vllm_action_input as module


class Tokenizer:
    eos_token_id = 11
    all_special_ids = [10, 11, 12, 13]
    output = ''

    def __len__(self):
        return 100

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages = messages
        return [1, 2, 3]

    def decode(self, ids, *, skip_special_tokens, **kwargs):
        return self.output if skip_special_tokens else self.output + '<|im_end|>'


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer())))
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat-template.jinja'):
        (tmp_path / name).write_text('{}')
        files[name] = hashlib.sha256(b'{}').hexdigest()
    monkeypatch.setattr(module, 'NATIVE_TEMPLATE_SHA256', files['chat-template.jinja'])
    manifest = {'schema_version': 'dml-remote-vllm-manifest-v1',
        'endpoint': 'http://192.168.50.91:8000/v1', 'model': 'nvidia/nemotron-3-super',
        'model_revision': 'synthetic-test', 'model_provenance': {'test_only': True},
        'server_configuration': {'test_only': True}, 'sampling': module.sampling_policy_identity(),
        'model_window_tokens': 1024, 'timeout_seconds': 3, 'files': files,
        'evidence_directory': str(tmp_path / 'raw'), 'client_runtime_versions': module.client_runtime_versions(),
        'consumer_profile': module.CONSUMER_PROFILE, 'rendering_policy': module.NATIVE_RENDERING_POLICY,
        'action_projection_policy': module.NATIVE_ACTION_POLICY}
    (tmp_path / 'remote-vllm-manifest.json').write_text(json.dumps(manifest))
    with module.NativeRemoteVLLMActionInputConsumer(tmp_path, offline=True) as instance:
        yield instance


def request(consumer, *, reserve=16):
    return consumer.compile(initial_messages('Synthetic native request', consumer_profile=module.CONSUMER_PROFILE),
                            episode_tool_definitions(), output_reserved_tokens=reserve)


def tool_message(name='retrieve', arguments=None):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call-native-actual', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments or {'query': 'synthetic', 'top_k': 2})}}]}


def raw_call(name='retrieve', arguments=None):
    arguments = arguments or {'query': 'synthetic', 'top_k': 2}
    return '\n<tool_call>\n<function=' + name + '>\n' + ''.join(
        '<parameter=' + key + '>\n' + (json.dumps(value) if isinstance(value, (list, dict)) else str(value)) +
        '\n</parameter>\n' for key, value in arguments.items()) + '</function>\n</tool_call>\n'


def response(message=None, *, finish='tool_calls', output=None):
    return {'model': 'nvidia/nemotron-3-super', 'prompt_token_ids': [1, 2, 3],
            'choices': [{'index': 0, 'message': message or tool_message(), 'finish_reason': finish,
                         'token_ids': [4, 11] if output is None else output}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2 if output is None else len(output),
                      'total_tokens': 5 if output is None else 3 + len(output)}}


def mock_endpoint(consumer, monkeypatch, reply):
    def post(path, payload):
        current = ({'token_ids': [1, 2, 3], 'sampling_params': {
            **module.sampling_policy_identity(), 'max_tokens': payload['max_tokens'], 'structured_outputs': None}}
            if path.endswith('/render') else deepcopy(reply))
        consumer.last_exchange = {'path': path, 'request': deepcopy(payload), 'response': current,
            'response_raw': json.dumps(current), 'status': 200, 'error': None, 'timed_out': False,
            'timeout_seconds': consumer.manifest['timeout_seconds']}
        return current
    monkeypatch.setattr(consumer, '_post', post)
    consumer._offline = False


def test_raw_native_tool_output_is_not_overwritten_by_projection(consumer, monkeypatch):
    consumer._tokenizer.output = raw_call()
    mock_endpoint(consumer, monkeypatch, response())
    artifact = request(consumer)
    result = consumer.execute(artifact)
    assert result.text == raw_call() + '<|im_end|>'
    assert json.loads(result.action_text)['arguments'] == {'query': 'synthetic', 'top_k': 2}
    assert result.action_text != result.text
    assert result.native_tool_call_id == 'call-native-actual' and result.action_error is None
    payload = consumer.last_exchange['request']
    assert payload['tool_choice'] == 'auto' and payload['parallel_tool_calls'] is False
    assert 'structured_outputs' not in payload and 'response_format' not in payload
    assert payload['max_tokens'] == artifact.output_reserved_tokens
    assert result.output_ids == (4, 11)
    consumer.validate_exchange(artifact.signing_payload(), consumer._bound_requests[artifact.artifact_digest].to_payload(),
                               consumer.last_exchange, json.loads(json.dumps(asdict(result))))


def test_original_final_json_is_preserved_exactly(consumer, monkeypatch):
    text = '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}'
    consumer._tokenizer.output = text
    mock_endpoint(consumer, monkeypatch, response({'role': 'assistant', 'content': text, 'tool_calls': []}, finish='stop'))
    artifact = request(consumer)
    result = consumer.execute(artifact)
    assert result.action_text == text and result.action_error is None and result.native_tool_call_id is None
    assert result.text == text + '<|im_end|>'


@pytest.mark.parametrize('case', ['prose', 'parallel', 'mixed', 'reasoning', 'refusal', 'duplicate_arguments',
                                  'nonfinite', 'duplicate_xml', 'unknown_xml', 'truncated_xml'])
def test_invalid_native_actions_retain_known_usage_no_projection(consumer, monkeypatch, case):
    message = tool_message()
    raw, finish = raw_call(), 'tool_calls'
    if case == 'prose':
        message, raw, finish = {'role': 'assistant', 'content': 'Here is the answer.', 'tool_calls': []}, 'Here is the answer.', 'stop'
    elif case == 'parallel':
        message['tool_calls'].append(deepcopy(message['tool_calls'][0]))
        message['tool_calls'][1]['id'] = 'second-id'
        raw += raw_call()
    elif case == 'mixed':
        message['content'] = 'I will do it.'
        raw = message['content'] + raw
    elif case in ('reasoning', 'refusal'):
        message[case] = 'nonempty forbidden metadata'
    elif case == 'duplicate_arguments':
        message['tool_calls'][0]['function']['arguments'] = '{"query":"a","query":"b","top_k":2}'
    elif case == 'nonfinite':
        message['tool_calls'][0]['function']['arguments'] = '{"query":"a","top_k":1e999}'
    elif case == 'duplicate_xml':
        raw = raw.replace('</function>', '<parameter=query>synthetic</parameter></function>')
    elif case == 'unknown_xml':
        raw = raw.replace('</function>', '<parameter=extra>text</parameter></function>')
    else:
        raw = raw.removesuffix('</tool_call>\n')
    consumer._tokenizer.output = raw
    mock_endpoint(consumer, monkeypatch, response(message, finish=finish))
    result = consumer.execute(request(consumer))
    assert result.action_text is None and result.action_error
    assert result.input_token_count == 3 and result.output_token_count == 2
    assert result.native_message == message and result.native_tool_call_id is None


@pytest.mark.parametrize('ids', [[10, 4, 11], [4, 12, 11], [11, 4, 11], [4, 11, 11], [4]])
def test_control_tokens_never_disappear_into_valid_action(consumer, monkeypatch, ids):
    consumer._tokenizer.output = raw_call()
    mock_endpoint(consumer, monkeypatch, response(output=ids))
    result = consumer.execute(request(consumer))
    assert result.action_text is None and 'special tokens' in result.action_error
    assert result.output_ids == tuple(ids)


@pytest.mark.parametrize('mutation', ['parser_arguments', 'prompt_ids', 'usage', 'no_token_ids', 'too_many', 'content'])
def test_contradictory_native_evidence_fails_closed(consumer, monkeypatch, mutation):
    consumer._tokenizer.output = raw_call()
    reply = response()
    if mutation == 'parser_arguments':
        reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"query":"different","top_k":2}'
    elif mutation == 'prompt_ids':
        reply['prompt_token_ids'] = [9, 2, 3]
    elif mutation == 'usage':
        reply['usage']['completion_tokens'] = 99
    elif mutation == 'no_token_ids':
        del reply['choices'][0]['token_ids']
    elif mutation == 'too_many':
        reply['choices'][0]['token_ids'] = [4] * 17
    else:
        consumer._tokenizer.output = 'actual raw prose'
        reply['choices'][0]['message'] = {'role': 'assistant', 'content': 'different parsed prose', 'tool_calls': []}
        reply['choices'][0]['finish_reason'] = 'stop'
    mock_endpoint(consumer, monkeypatch, reply)
    with pytest.raises(ModelInputExecutionError):
        consumer.execute(request(consumer))


def test_replay_rejects_forged_projection(consumer, monkeypatch):
    consumer._tokenizer.output = raw_call()
    mock_endpoint(consumer, monkeypatch, response())
    artifact = request(consumer)
    result = asdict(consumer.execute(artifact))
    logical = consumer._bound_requests[artifact.artifact_digest].to_payload()
    result['action_text'] = '{"kind":"final"}'
    with pytest.raises(ModelInputError):
        consumer.validate_exchange(artifact.signing_payload(), logical, consumer.last_exchange, result)


def test_authentication_and_runtime_mutation_fail_before_dispatch(consumer):
    artifact = request(consumer)
    with pytest.raises(ModelInputError):
        consumer.execute(replace(artifact, nonce='forged'))
    consumer.manifest['endpoint'] = 'http://different.invalid/v1'
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)


def test_native_render_checks_default_n_and_forbids_grammar():
    sampling = {**module.sampling_policy_identity(), 'max_tokens': 16}
    sampling.pop('n')
    rendered = {'token_ids': [1], 'sampling_params': sampling}
    module.NativeRemoteVLLMActionInputConsumer._validate_render(rendered, [1], 16)
    sampling['structured_outputs'] = {'json': {}}
    with pytest.raises(ModelInputError):
        module.NativeRemoteVLLMActionInputConsumer._validate_render(rendered, [1], 16)


def test_array_argument_is_strict_json_without_python_literal_fallback():
    tools = episode_tool_definitions()
    args = {'record_refs': ['ref-a', 'ref-b'], 'text': 'summary', 'reason': 'synthetic'}
    name, parsed = module._xml_call(raw_call('promote', args), tools)
    assert name == 'promote' and parsed == args
    malformed = raw_call('promote', args).replace('["ref-a", "ref-b"]', "['ref-a', 'ref-b']")
    with pytest.raises(module.NativeProjectionError):
        module._xml_call(malformed, tools)


@pytest.mark.parametrize('prefix', ['', '\n', 'I will retrieve the stored facts.\n\n'])
def test_v2_binds_prefix_prose_without_granting_action_authority(prefix):
    message = tool_message()
    message['content'] = prefix if prefix.strip() else None
    raw = prefix + raw_call().lstrip('\n')
    action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                  consumer_profile=module.V2_CONSUMER_PROFILE)
    assert error is None
    assert json.loads(action)['name'] == 'retrieve'
    if prefix.strip():
        assert module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls')[0] is None


@pytest.mark.parametrize('prefix,suffix', [
    ('<tool_call>', ''), ('</function>', ''), ('<parameter=x>', ''),
    ('', '<tool_call>'), ('', '</parameter>'), ('', 'Now do a second action.'),
    ('<Tool_call x>', ''), ('', '<function=retire>'),
])
def test_v2_rejects_unbound_or_ambiguous_segments(prefix, suffix):
    message = tool_message()
    message['content'] = prefix
    raw = prefix + raw_call().lstrip('\n') + suffix
    action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                  consumer_profile=module.V2_CONSUMER_PROFILE)
    assert action is None and error


def test_v2_prefix_parser_mismatch_is_integrity_failure():
    message = tool_message()
    message['content'] = 'Looking up.\n'
    with pytest.raises(ModelInputError, match='prose contradicts'):
        module.project_native_message(message, 'Looking up.\n\n' + raw_call().lstrip('\n'),
            episode_tool_definitions(), 'tool_calls', consumer_profile=module.V2_CONSUMER_PROFILE)


def test_v2_final_prose_still_rejected():
    message = {'role': 'assistant', 'content': 'The answer is seven.', 'tool_calls': []}
    action, error = module.project_native_message(message, message['content'], episode_tool_definitions(),
                                                  'stop', consumer_profile=module.V2_CONSUMER_PROFILE)
    assert action is None and error


def test_v2_observed_supersession_preamble_is_only_metadata():
    prefix = ('I need to retrieve the current access phrase, then mark the old entry as superseded '
              'by the current entry, and finally check the current evidence again.\n\n'
              'Let me start by retrieving the archive information:\n')
    message = tool_message()
    message['content'] = prefix
    raw = prefix + raw_call().lstrip('\n')
    action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                  consumer_profile=module.V2_CONSUMER_PROFILE)
    assert error is None and json.loads(action)['name'] == 'retrieve'
    assert json.loads(action)['arguments'] == {'query': 'synthetic', 'top_k': 2}


def test_v2_nested_control_in_argument_rejected():
    message = tool_message(arguments={'query': '< /function>', 'top_k': 2})
    message['content'] = None
    action, error = module.project_native_message(message,
        raw_call(arguments={'query': '< /function>', 'top_k': 2}), episode_tool_definitions(),
        'tool_calls', consumer_profile=module.V2_CONSUMER_PROFILE)
    assert action is None and error


@pytest.mark.parametrize('profile,policy,runtime_prefix', [
    (module.V2_CONSUMER_PROFILE, module.NATIVE_V2_ACTION_POLICY, 'dml-remote-vllm-native-tools-runtime-v2:'),
    (module.V3_CONSUMER_PROFILE, module.NATIVE_V3_ACTION_POLICY, 'dml-remote-vllm-native-tools-runtime-v3:'),
    (module.V4_CONSUMER_PROFILE, module.NATIVE_V4_ACTION_POLICY, 'dml-remote-vllm-native-tools-runtime-v4:'),
])
def test_prose_consumer_roundtrip_and_identity_remain_distinct(consumer, monkeypatch, profile, policy, runtime_prefix):
    directory = consumer._directory
    manifest = deepcopy(consumer.manifest)
    manifest['consumer_profile'] = profile
    manifest['action_projection_policy'] = policy
    (directory / 'remote-vllm-manifest.json').write_text(json.dumps(manifest))
    with module.NativeRemoteVLLMActionInputConsumer(directory, consumer_profile=profile,
                                                   offline=True) as v2:
        assert v2.identity.runtime_identity.startswith(runtime_prefix)
        assert v2.identity != consumer.identity
        prefix = 'I will retrieve the stored value.\n'
        message = tool_message()
        message['content'] = prefix
        v2._tokenizer.output = prefix + raw_call().lstrip('\n')
        mock_endpoint(v2, monkeypatch, response(message))
        artifact = v2.compile(initial_messages('Synthetic native request', consumer_profile=profile),
                              episode_tool_definitions(), output_reserved_tokens=16)
        result = v2.execute(artifact)
        assert result.action_error is None and result.text.startswith(prefix)
        assert result.native_message['content'] == prefix
        v2.validate_exchange(artifact.signing_payload(), v2._bound_requests[artifact.artifact_digest].to_payload(),
                             v2.last_exchange, json.loads(json.dumps(asdict(result))))
        assert result.projection_digest != module.native_projection_digest(message, result.action_text, None)


@pytest.mark.parametrize('prefix', ['\n', ' ', '\t', '\n \t\r\n'])
@pytest.mark.parametrize('api_content', [None, ''])
def test_v2_observed_whitespace_prefix_normalization(prefix, api_content):
    message = tool_message()
    message['content'] = api_content
    raw = prefix + raw_call().lstrip('\n')
    if api_content is None:
        action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                      consumer_profile=module.V2_CONSUMER_PROFILE)
        assert error is None and json.loads(action)['name'] == 'retrieve'
        assert raw.startswith(prefix)  # Prefix remains raw evidence, never repaired.
    else:
        with pytest.raises(ModelInputError, match='prose contradicts'):
            module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                          consumer_profile=module.V2_CONSUMER_PROFILE)


@pytest.mark.parametrize('prefix,content', [(' factual claim ', 'factual claim'), ('fact', None), ('fact', '')])
def test_v2_never_normalizes_substantive_prefix(prefix, content):
    message = tool_message()
    message['content'] = content
    with pytest.raises(ModelInputError, match='prose contradicts'):
        module.project_native_message(message, prefix + raw_call().lstrip('\n'), episode_tool_definitions(),
                                      'tool_calls', consumer_profile=module.V2_CONSUMER_PROFILE)


@pytest.mark.parametrize('prefix,content', [('', None), ('\n', None), ('I have evidence.\n', 'I have evidence.\n')])
def test_v3_transport_projection_matches_corrected_v2(prefix, content):
    message = tool_message()
    message['content'] = content
    raw = prefix + raw_call().lstrip('\n')
    results = [module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                            consumer_profile=profile)
               for profile in (module.V2_CONSUMER_PROFILE, module.V3_CONSUMER_PROFILE)]
    assert results[0] == results[1] and results[0][1] is None


def test_v3_identity_requires_explicit_profile_manifest(consumer):
    manifest = deepcopy(consumer.manifest)
    v2 = module.native_identity(manifest, consumer_profile=module.V2_CONSUMER_PROFILE)
    v3 = module.native_identity(manifest, consumer_profile=module.V3_CONSUMER_PROFILE)
    assert v2 != v3
    assert v2.model_digest == v3.model_digest and v2.tokenizer_digest == v3.tokenizer_digest
    with pytest.raises(ModelInputError, match='pinned template and explicit'):
        module.verify_native_manifest(consumer._directory, consumer_profile=module.V3_CONSUMER_PROFILE)


@pytest.mark.parametrize('opener', ['', '<think>'])
def test_v4_reasoning_is_bound_metadata_not_action_authority(opener):
    message = tool_message()
    message['reasoning'] = 'I need the stored facts.\n'
    raw = opener + message['reasoning'] + '</think>' + raw_call()
    action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                  consumer_profile=module.V4_CONSUMER_PROFILE)
    assert error is None and json.loads(action)['name'] == 'retrieve'
    assert module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                         consumer_profile=module.V3_CONSUMER_PROFILE)[0] is None


@pytest.mark.parametrize('raw', [
    'reasoning<think>x</think>', '<think>x</think></think>', '<think><think>x</think>',
    '<think>x', 'x</think>', 'x</think>  ', 'x</Think>', 'x</think><think>',
    '<tool_call>x</think>',
])
def test_v4_rejects_ambiguous_or_incomplete_reasoning(raw):
    message = tool_message()
    message['reasoning'] = 'x'
    action, error = module.project_native_message(message, raw, episode_tool_definitions(), 'tool_calls',
                                                  consumer_profile=module.V4_CONSUMER_PROFILE)
    assert action is None and error


def test_v4_reasoning_mismatch_is_integrity_failure():
    message = tool_message()
    message['reasoning'] = 'different'
    with pytest.raises(ModelInputError, match='reasoning contradicts'):
        module.project_native_message(message, 'original</think>' + raw_call(), episode_tool_definitions(),
                                      'tool_calls', consumer_profile=module.V4_CONSUMER_PROFILE)


@pytest.mark.parametrize('final', ['The answer is seven.', '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}'])
def test_v4_reasoning_final_requires_original_json(final):
    message = {'role': 'assistant', 'content': final, 'reasoning': 'metadata', 'tool_calls': []}
    action, error = module.project_native_message(message, 'metadata</think>' + final,
        episode_tool_definitions(), 'stop', consumer_profile=module.V4_CONSUMER_PROFILE)
    if final.startswith('{'):
        assert action == final and error is None
    else:
        assert action is None and error


def test_v4_refusal_is_never_reasoning_admission():
    message = tool_message()
    message.update(reasoning='metadata', refusal='refused')
    action, error = module.project_native_message(message, 'metadata</think>' + raw_call(),
        episode_tool_definitions(), 'tool_calls', consumer_profile=module.V4_CONSUMER_PROFILE)
    assert action is None and error


@pytest.mark.parametrize('ids,admitted', [([4, 13, 11], True), ([4, 11], False), ([10, 13, 11], False)])
def test_v4_control_ids_independently_bind_reasoning(consumer, ids, admitted):
    consumer._consumer_profile = module.V4_CONSUMER_PROFILE
    # Pinned tokenizer lists only EOS/UNK/BOS as special: think tokens are ordinary IDs.
    consumer._tokenizer.all_special_ids = [0, 1, 11]
    consumer._tokenizer.output = 'metadata</think>' + raw_call()
    message = tool_message()
    message['reasoning'] = 'metadata'
    result = consumer._validate_native_response(response(message, output=ids), [1, 2, 3], 16,
                                                episode_tool_definitions())
    assert (result[3] is not None) is admitted
