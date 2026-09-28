"""Separately versioned native-tool transport; no all-turn action grammar.

Raw token output remains evidence. A distinct, strict projection can request an
existing DML action; the unchanged gateway alone owns effects and authority.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from threading import RLock

from ..contracts.agent_episode import (
    AgentEpisodeError, NATIVE_REMOTE_VLLM_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE,
    NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_V4_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE, canonical_json,
    initial_messages, native_action_text, native_policy_identity, parse_agent_action, validate_native_budget_messages,
)
from ..contracts.model_input import (
    CompiledModelInput, ModelInputError, ModelInputRequest, _decode,
)
from .agent_action_grammar import MAX_BOUND_REQUESTS, action_schema
from .model_input import ModelInputExecutionError, ModelInputResult, _json_bytes
from .remote_vllm_action_input import (
    CHAT_TEMPLATE_KWARGS, EXACT_TOKEN_LIMITATIONS, RemoteVLLMActionInputConsumer,
    client_runtime_versions, remote_identity, rendering_messages,
    sampling_policy_identity, verify_remote_manifest,
)

CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_CONSUMER_PROFILE
NATIVE_RENDERING_POLICY = 'native-template-auto-tools-v1'
NATIVE_ACTION_POLICY = 'single-native-call-or-original-final-json-v1'
V2_CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V2_CONSUMER_PROFILE
V3_CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V3_CONSUMER_PROFILE
V4_CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V4_CONSUMER_PROFILE
V5_CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V5_CONSUMER_PROFILE
V6_CONSUMER_PROFILE = NATIVE_REMOTE_VLLM_V6_CONSUMER_PROFILE
REASONING_CONSUMER_PROFILES = (V4_CONSUMER_PROFILE, V5_CONSUMER_PROFILE, V6_CONSUMER_PROFILE)
NATIVE_V5_ACTION_POLICY = "single-native-call-or-original-final-json-bound-native-reasoning-precommit-recovery-v5"
NATIVE_V6_ACTION_POLICY = "single-native-call-or-original-final-json-bound-native-reasoning-precommit-recovery-budget-guidance-v6"
PROSE_CONSUMER_PROFILES = (V2_CONSUMER_PROFILE, V3_CONSUMER_PROFILE, V4_CONSUMER_PROFILE, V5_CONSUMER_PROFILE, V6_CONSUMER_PROFILE)
NATIVE_V2_ACTION_POLICY = 'single-native-call-with-bound-nonauthoritative-prose-or-original-final-json-serving-whitespace-null-v2.1'
NATIVE_V3_ACTION_POLICY = 'single-native-call-with-bound-nonauthoritative-prose-or-original-final-json-serving-whitespace-null-completion-v3'
NATIVE_V4_ACTION_POLICY = 'single-native-call-or-original-final-json-bound-native-reasoning-v4'
NATIVE_TEMPLATE_SHA256 = '575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a'
NATIVE_EXACT_TOKEN_LIMITATIONS = [
    *EXACT_TOKEN_LIMITATIONS,
    'Native automatic generation has no all-turn grammar guarantee; complete actions are validated after generation',
    'Only the explicitly versioned single-call native XML subset is admitted; broader parser syntax is rejected without repair',
]
NATIVE_PROTOCOL_POLICY = {
    'rendering': NATIVE_RENDERING_POLICY, 'projection': NATIVE_ACTION_POLICY,
    'tool_choice': 'auto', 'parallel_tool_calls': False, 'structured_outputs': None,
    'raw_text': 'decode_all_output_ids_skip_special_tokens_false',
    'parser_input': 'decode_after_requiring_only_one_terminal_eos11_special_token',
    'output_special_tokens': 'only_one_terminal_eos11_optional_for_length',
    'parser': 'qwen3_coder', 'reasoning_parser': 'super_v3',
    'xml_admission': 'single_complete_call_no_unknown_duplicate_or_unparsed_parameters',
    'parameter_newlines': 'remove_exactly_one_leading_and_trailing_newline_as_pinned_parser',
    'parameter_conversion': 'declared_string_integer_or_strict_json_array_object_only',
    'render_default': {'n': 1}, 'render_default_source': 'vllm-0.20.0-GenerateRequest-schema',
    'invalid_projection': 'known_raw_usage_invalid_action_no_effects',
    'contradictory_parser_evidence': 'fail_closed_execution_error',
}


@dataclass(frozen=True, slots=True)
class NativeModelInputResult(ModelInputResult):
    native_message: dict
    action_text: str | None
    action_error: str | None
    native_tool_call_id: str | None
    projection_digest: str


class NativeProjectionError(ValueError):
    """Recorded native generation cannot become an admitted DML action."""


def native_projection_digest(native_message, action_text, action_error, *, consumer_profile=CONSUMER_PROFILE):
    return hashlib.sha256(canonical_json({'policy': native_policy_identity(consumer_profile=consumer_profile),
        'native_message': native_message, 'action_text': action_text,
        'action_error': action_error})).hexdigest()


def verify_native_manifest(directory, *, consumer_profile=CONSUMER_PROFILE):
    manifest = verify_remote_manifest(directory)
    if (manifest.get('consumer_profile') != consumer_profile
            or manifest.get('rendering_policy') != NATIVE_RENDERING_POLICY
            or manifest.get('action_projection_policy') != (NATIVE_V6_ACTION_POLICY if consumer_profile == V6_CONSUMER_PROFILE else NATIVE_V5_ACTION_POLICY if consumer_profile == V5_CONSUMER_PROFILE else NATIVE_V4_ACTION_POLICY if consumer_profile == V4_CONSUMER_PROFILE else NATIVE_V3_ACTION_POLICY if consumer_profile == V3_CONSUMER_PROFILE else NATIVE_V2_ACTION_POLICY if consumer_profile == V2_CONSUMER_PROFILE else NATIVE_ACTION_POLICY)
            or manifest['files']['chat-template.jinja'] != NATIVE_TEMPLATE_SHA256):
        raise ModelInputError('Native candidate requires its pinned template and explicit native protocol')
    return manifest


def native_identity(manifest, *, consumer_profile=CONSUMER_PROFILE):
    base = remote_identity(manifest)
    policy = {'base_identity': base.to_payload(), 'native_action_policy': native_policy_identity(consumer_profile=consumer_profile),
              'native_protocol': ({**NATIVE_PROTOCOL_POLICY, 'projection': NATIVE_V6_ACTION_POLICY if consumer_profile == V6_CONSUMER_PROFILE else NATIVE_V5_ACTION_POLICY if consumer_profile == V5_CONSUMER_PROFILE else NATIVE_V4_ACTION_POLICY if consumer_profile == V4_CONSUMER_PROFILE else NATIVE_V3_ACTION_POLICY if consumer_profile == V3_CONSUMER_PROFILE else NATIVE_V2_ACTION_POLICY,
                  'accompanying_prose': 'exact_prefix_matches_native_content_only_whitespace_suffix_no_authority',
                  'prefix_normalization': 'vllm020_engine_serving_whitespace_only_prefix_to_null_v1',
                  'xml_delimiters': 'exactly_one_complete_call_no_stray_or_nested_control_delimiters'}
                  if consumer_profile in PROSE_CONSUMER_PROFILES else NATIVE_PROTOCOL_POLICY), 'exact_token_limitations': NATIVE_EXACT_TOKEN_LIMITATIONS,
              'all_turn_grammar_removed': True}
    if consumer_profile in REASONING_CONSUMER_PROFILES:
        policy['reasoning_admission'] = {'no_thinking_request': 'advisory',
            'segmentation': 'optional_exact_leading_opener_single_closer_nonempty_suffix',
            'metadata': 'exact_raw_prefix_equals_api_reasoning_never_action_authority',
            'controls': 'think_ids12_13_exact_text_counts_role_id10_forbidden',
            'fallback': 'no_empty_suffix_promotion'}
    prefix = ('dml-remote-vllm-native-tools-runtime-v6:' if consumer_profile == V6_CONSUMER_PROFILE else
              'dml-remote-vllm-native-tools-runtime-v5:' if consumer_profile == V5_CONSUMER_PROFILE else
              'dml-remote-vllm-native-tools-runtime-v4:' if consumer_profile == V4_CONSUMER_PROFILE else
              'dml-remote-vllm-native-tools-runtime-v3:' if consumer_profile == V3_CONSUMER_PROFILE else
              'dml-remote-vllm-native-tools-runtime-v2:' if consumer_profile == V2_CONSUMER_PROFILE else
              'dml-remote-vllm-native-tools-runtime-v1:')
    return replace(base, runtime_identity=prefix + hashlib.sha256(
        canonical_json(policy)).hexdigest())


def _xml_call(text, tools, *, strict_controls=False):
    """Strict subset of the pinned parser, independently refusing ambiguous XML.

    Public DML argument types are strings, integers and JSON arrays. Python
    literal fallbacks and duplicate/unknown XML parameters are never admitted.
    """
    match = re.fullmatch(r'\s*<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>\s*</tool_call>\s*',
                         text, flags=re.DOTALL)
    if match is None:
        raise NativeProjectionError('Native output is not exactly one complete unmixed XML call')
    name, body = match.groups()
    definitions = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
    if name not in definitions:
        raise NativeProjectionError('Native function was not advertised')
    properties = definitions[name]['properties']
    arguments = {}
    cursor = 0
    for parameter in re.finditer(r'<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>', body, flags=re.DOTALL):
        if body[cursor:parameter.start()].strip():
            raise NativeProjectionError('Native call contains unparsed parameter content')
        key, value = parameter.groups()
        if key in arguments or key not in properties:
            raise NativeProjectionError('Native call contains duplicate or unknown parameters')
        if strict_controls and re.search(r'<\s*/?\s*(?:tool_call|function|parameter)\b', value, re.IGNORECASE):
            raise NativeProjectionError('Native parameter contains ambiguous control delimiters')
        if '<parameter=' in value or '</parameter>' in value or '<function=' in value or '<tool_call>' in value:
            raise NativeProjectionError('Nested native delimiters are ambiguous')
        if value.startswith('\n'):
            value = value[1:]
        if value.endswith('\n'):
            value = value[:-1]
        kind = properties[key]['type']
        try:
            if value.lower() == 'null':
                converted = None
            elif kind == 'string':
                converted = value
            elif kind == 'integer':
                converted = int(value)
            elif kind in ('array', 'object'):
                converted = _decode(value.encode())
            else:
                raise NativeProjectionError('Unsupported native parameter type')
        except (ValueError, UnicodeError) as exc:
            raise NativeProjectionError('Native parameter cannot be converted without repair') from exc
        arguments[key] = converted
        cursor = parameter.end()
    if body[cursor:].strip():
        raise NativeProjectionError('Native call contains trailing unparsed content')
    return name, arguments


def _bound_native_call_text(text, content):
    """Bind one XML action and all non-authoritative prose to parser evidence."""
    # Control-like fragments outside the one complete block must not disappear
    # through a permissive parser. Full strings remain in raw evidence/history.
    opening, closing = '<tool_call>', '</tool_call>'
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise NativeProjectionError('Native output requires exactly one complete XML call')
    start, end = text.index(opening), text.index(closing) + len(closing)
    if end <= start:
        raise NativeProjectionError('Native XML delimiters are out of order')
    prefix, suffix, block = text[:start], text[end:], text[start:end]
    controls = r'<\s*/?\s*(?:tool_call|function|parameter)\b'
    if re.search(controls, prefix + suffix, re.IGNORECASE):
        raise NativeProjectionError('Native prose contains stray control delimiters')
    if (block.count('<function=') != 1 or block.count('</function>') != 1
            or block.count('<parameter=') != block.count('</parameter>')):
        raise NativeProjectionError('Native XML contains duplicate or unmatched delimiters')
    if suffix.strip():
        raise NativeProjectionError('Native parser does not attest substantive trailing prose')
    # vLLM 0.20 engine/serving.py converts a nonempty whitespace-only
    # tool-parser content prefix to None. Raw text is never rewritten.
    if prefix and not prefix.strip():
        matches = content is None
    else:
        matches = (content or '') == prefix
    if not matches:
        raise ModelInputError('Native prose contradicts independently decoded output segments')
    return block


def _bound_reasoning_suffix(message, text):
    """Admit a strict subset of pinned super_v3/DeepSeek delimiter parsing."""
    if type(message) is not dict:
        raise NativeProjectionError('Native message is not an object')
    if message.get('reasoning_content') not in (None, ''):
        raise NativeProjectionError('Native alternate reasoning field is unsupported')
    reasoning = message.get('reasoning')
    control = re.compile(r'<\s*/?\s*think\b', re.IGNORECASE)
    if not control.search(text):
        if reasoning not in (None, ''):
            raise ModelInputError('Native reasoning lacks independently decoded delimiter evidence')
        return text
    if text.count('</think>') != 1 or text.count('<think>') > 1:
        raise NativeProjectionError('Native reasoning requires one closing delimiter')
    body = text
    if '<think>' in body:
        if not body.startswith('<think>'):
            raise NativeProjectionError('Native reasoning opener has an unbound prefix')
        body = body[len('<think>'):]
    prefix, suffix = body.split('</think>', 1)
    if control.search(prefix) or control.search(suffix):
        raise NativeProjectionError('Native reasoning contains stray or nested delimiters')
    if re.search(r'<\s*/?\s*(?:tool_call|function|parameter)\b', prefix, re.IGNORECASE):
        raise NativeProjectionError('Native reasoning contains ambiguous action delimiters')
    if not suffix.strip():
        raise NativeProjectionError('Native reasoning has no independently actionable suffix')
    if reasoning != prefix:
        raise ModelInputError('Native reasoning contradicts independently decoded output segments')
    return suffix


def project_native_message(message, visible_text, tools, finish_reason, *, consumer_profile=CONSUMER_PROFILE):
    """Keep malformed native actions as quality failures with known raw usage."""
    if consumer_profile in REASONING_CONSUMER_PROFILES:
        try:
            visible_text = _bound_reasoning_suffix(message, visible_text)
        except NativeProjectionError as exc:
            return None, type(exc).__name__ + ": " + str(exc)
    if (type(message) is dict and not message.get('tool_calls')
            and message.get('reasoning') in (None, '') and message.get('reasoning_content') in (None, '')
            and type(message.get('content')) in (str, type(None))
            and (message.get('content') or '') != visible_text):
        raise ModelInputError('Native content contradicts independently decoded output tokens')
    try:
        projected = native_action_text(message, consumer_profile=consumer_profile)
        action = parse_agent_action(projected)
        calls = message.get('tool_calls') or []
        if calls:
            call_text = visible_text
            if consumer_profile in PROSE_CONSUMER_PROFILES:
                call_text = _bound_native_call_text(visible_text, message.get('content'))
            name, raw_arguments = _xml_call(call_text, tools, strict_controls=consumer_profile in PROSE_CONSUMER_PROFILES)
            native_arguments = _decode(calls[0]['function']['arguments'].encode())
            if name != calls[0]['function']['name'] or canonical_json(raw_arguments) != canonical_json(native_arguments):
                raise ModelInputError('Native parser fields contradict independently decoded output tokens')
            if finish_reason != 'tool_calls':
                raise ModelInputError('Native tool call contradicts finish reason')
            if action['name'] not in {tool['function']['name'] for tool in tools}:
                raise NativeProjectionError('Native action uses an unadvertised tool')
        else:
            if (message.get('content') or '') != visible_text:
                raise ModelInputError('Native final content contradicts independently decoded output tokens')
            if finish_reason != 'stop':
                raise NativeProjectionError('Native final was not normally terminated')
        return projected, None
    except (AgentEpisodeError, NativeProjectionError) as exc:
        return None, type(exc).__name__ + ': ' + str(exc)


class NativeRemoteVLLMActionInputConsumer(RemoteVLLMActionInputConsumer):
    """Authenticated local native rendering with independently checked HTTP IDs."""

    def __init__(self, snapshot_directory, *, consumer_profile=CONSUMER_PROFILE, offline=False):
        if consumer_profile not in (CONSUMER_PROFILE, *PROSE_CONSUMER_PROFILES):
            raise ModelInputError('Unknown native consumer profile')
        from transformers import AutoTokenizer
        self._directory = Path(snapshot_directory)
        self.manifest = verify_native_manifest(self._directory, consumer_profile=consumer_profile)
        if self.manifest['client_runtime_versions'] != client_runtime_versions():
            raise ModelInputError('Native tokenizer runtime differs from frozen candidate')
        self._identity = native_identity(self.manifest, consumer_profile=consumer_profile)
        self._manifest_bytes = _json_bytes(self.manifest)
        self._consumer_profile, self._offline = consumer_profile, offline
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._directory), local_files_only=True, trust_remote_code=False)
        self._template = (self._directory / 'chat-template.jinja').read_text()
        self._tokenizer.chat_template = self._template
        if self._tokenizer.eos_token_id != 11:
            raise ModelInputError('Pinned native tokenizer requires EOS token 11')
        self._lock, self._closed = RLock(), False
        self._consumer_id, self._auth_key = secrets.token_hex(24), secrets.token_bytes(32)
        self._bound_requests, self._bound_render_exchanges = {}, {}
        self.last_exchange = None
        self._tokenizer_state = self._tokenizer_fingerprint()

    def _validate_runtime(self):
        super()._validate_runtime()
        if (_json_bytes(self.manifest) != self._manifest_bytes
                or verify_native_manifest(self._directory, consumer_profile=self._consumer_profile) != self.manifest
                or native_identity(self.manifest, consumer_profile=self._consumer_profile) != self._identity):
            raise ModelInputError('Native manifest or projection policy changed')

    def close(self):
        with self._lock:
            self._bound_render_exchanges.clear()
            super().close()

    def request_payload(self, messages, tools, output_reserved_tokens):
        return {'model': self.manifest['model'], 'messages': deepcopy(messages), 'tools': deepcopy(tools),
                'tool_choice': 'auto', 'parallel_tool_calls': False,
                'chat_template_kwargs': deepcopy(CHAT_TEMPLATE_KWARGS),
                'add_generation_prompt': True, 'continue_final_message': False,
                'add_special_tokens': False, 'return_token_ids': True,
                'max_tokens': output_reserved_tokens, **sampling_policy_identity()}

    @staticmethod
    def _validate_render(response, input_ids, reserved):
        if response.get('token_ids') != list(input_ids):
            raise ModelInputError('Native server rendering differs from frozen local tokens')
        sampling = response['sampling_params']
        if sampling.get('structured_outputs') is not None:
            raise ModelInputError('Native automatic profile forbids a decoding grammar')
        expected = {**sampling_policy_identity(), 'max_tokens': reserved}
        for key in ('temperature', 'top_p', 'top_k', 'min_p', 'seed', 'max_tokens', 'n',
                    'repetition_penalty', 'frequency_penalty', 'presence_penalty'):
            actual = sampling.get('n', 1) if key == 'n' else sampling[key]
            if actual != expected[key]:
                raise ModelInputError('Native resolved sampling differs from frozen request')

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        self.last_exchange = None
        request = ModelInputRequest.from_payload({'messages': messages, 'tools': [] if tools is None else tools,
            'output_reserved_tokens': output_reserved_tokens,
            **({'message_policy': 'native-reasoning-metadata-v1'} if self._consumer_profile in REASONING_CONSUMER_PROFILES else {})},
            allow_native_reasoning=self._consumer_profile in REASONING_CONSUMER_PROFILES)
        if self._consumer_profile == V6_CONSUMER_PROFILE:
            budget = validate_native_budget_messages(request.messages)
            if budget['output_tokens_per_call'] != output_reserved_tokens:
                raise ModelInputError('Native budget metadata differs from actual output reservation')
        elif request.messages[0] != initial_messages('profile binding', consumer_profile=self._consumer_profile)[0]:
            raise ModelInputError('Native profile requires its exact system policy')
        action_schema(request.tools)
        with self._lock:
            self._validate_runtime()
            if len(self._bound_requests) >= MAX_BOUND_REQUESTS:
                raise ModelInputError('Native compiled-request capacity exhausted')
            ids = tuple(self._tokenizer.apply_chat_template(rendering_messages(request.messages), tools=request.tools,
                chat_template=self._template, tokenize=True, add_generation_prompt=True,
                continue_final_message=False, truncation=False, padding=False, **CHAT_TEMPLATE_KWARGS))
            artifact = CompiledModelInput(identity=self._identity, request_digest=request.request_digest,
                input_ids=ids, attention_mask=(1,) * len(ids), output_reserved_tokens=output_reserved_tokens,
                model_window_tokens=self._identity.model_window_tokens, consumer_id=self._consumer_id,
                nonce=secrets.token_hex(24), auth_tag='0' * 64)
            if not self._offline:
                try:
                    rendered = self._post('/v1/chat/completions/render', self.request_payload(
                        request.messages, request.tools, output_reserved_tokens))
                    self._validate_render(rendered, ids, output_reserved_tokens)
                except ModelInputError:
                    raise
                except Exception as exc:
                    raise ModelInputError('Native rendering failed') from exc
            artifact = replace(artifact, auth_tag=self._auth_tag(artifact))
            self._bound_requests[artifact.artifact_digest] = request
            self._bound_render_exchanges[artifact.artifact_digest] = deepcopy(self.last_exchange)
            return artifact

    def decode_output(self, output_ids):
        return self._tokenizer.decode(list(output_ids), skip_special_tokens=False,
            clean_up_tokenization_spaces=False, spaces_between_special_tokens=False)

    def _validate_native_response(self, response, input_ids, reserved, tools):
        if response['model'] != self.manifest['model'] or len(response['choices']) != 1:
            raise ModelInputError('Unexpected native model or choice count')
        choice = response['choices'][0]
        output = choice.get('token_ids')
        if (type(choice.get('index')) is not int or choice['index'] != 0
                or response.get('prompt_token_ids') != list(input_ids)
                or type(output) is not list or not 0 < len(output) <= reserved
                or any(type(token) is not int or not 0 <= token < len(self._tokenizer) for token in output)):
            raise ModelInputError('Missing, changed or out-of-budget native token evidence')
        usage = response['usage']
        if (any(type(usage.get(key)) is not int for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                or usage['prompt_tokens'] != len(input_ids) or usage['completion_tokens'] != len(output)
                or usage['total_tokens'] != len(input_ids) + len(output)):
            raise ModelInputError('Native usage differs from returned IDs')
        if choice.get('finish_reason') not in ('stop', 'length', 'tool_calls'):
            raise ModelInputError('Native response has an unsupported finish reason')
        raw_text = self.decode_output(output)
        visible = self._tokenizer.decode(output, skip_special_tokens=True,
            clean_up_tokenization_spaces=False, spaces_between_special_tokens=False)
        message = deepcopy(choice['message'])
        specials = set(self._tokenizer.all_special_ids)
        special_positions = [index for index, token in enumerate(output) if token in specials]
        allowed_positions = [len(output) - 1] if output[-1] == self._tokenizer.eos_token_id else []
        if (special_positions != allowed_positions
                or (choice['finish_reason'] != 'length' and not allowed_positions)):
            action_text, action_error = None, 'NativeProjectionError: Native output contains forbidden special tokens or lacks terminal EOS'
        elif self._consumer_profile in REASONING_CONSUMER_PROFILES and (10 in output
                or output.count(12) != raw_text.count('<think>')
                or output.count(13) != raw_text.count('</think>')):
            action_text, action_error = None, 'NativeProjectionError: Native reasoning or role control IDs contradict raw delimiters'
        else:
            action_text, action_error = project_native_message(message, visible, tools, choice['finish_reason'], consumer_profile=self._consumer_profile)
        native_id = (message['tool_calls'][0]['id'] if action_text is not None
                     and parse_agent_action(action_text)['kind'] == 'tool' else None)
        return tuple(output), raw_text, message, action_text, action_error, native_id

    def execute(self, artifact):
        with self._lock:
            self.last_exchange = None
            self._validate_runtime()
            if type(artifact) is not CompiledModelInput:
                raise ModelInputError('Authenticated native compiled input required')
            artifact.validate()
            if (artifact.consumer_id != self._consumer_id or artifact.identity != self._identity
                    or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))):
                raise ModelInputError('Native input authentication failed')
            request = self._bound_requests.get(artifact.artifact_digest)
            if request is None or request.request_digest != artifact.request_digest:
                raise ModelInputError('Native input lacks an immutable request binding')
            try:
                response = self._post('/v1/chat/completions', self.request_payload(
                    request.messages, request.tools, artifact.output_reserved_tokens))
                self.last_exchange['render_exchange'] = deepcopy(self._bound_render_exchanges[artifact.artifact_digest])
                output, text, message, action, error, native_id = self._validate_native_response(
                    response, artifact.input_ids, artifact.output_reserved_tokens, request.tools)
                self._validate_runtime()
                return NativeModelInputResult(artifact.artifact_digest, len(artifact.input_ids), len(output), output, text,
                    message, action, error, native_id, native_projection_digest(message, action, error, consumer_profile=self._consumer_profile))
            except Exception as exc:
                raise ModelInputExecutionError('Native generation failed or returned inconsistent token evidence') from exc

    def validate_exchange(self, compiled_payload, logical_request, exchange, completed_payload):
        ids, reserved = compiled_payload['input_ids'], compiled_payload['output_reserved_tokens']
        payload = self.request_payload(logical_request['messages'], logical_request['tools'], reserved)
        render = exchange['render_exchange']
        for current, path in ((exchange, '/v1/chat/completions'), (render, '/v1/chat/completions/render')):
            if (current['path'] != path or current['error'] is not None or current['timed_out'] is not False
                    or current['status'] != 200 or current['timeout_seconds'] != self.manifest['timeout_seconds']
                    or current['request'] != payload or json.loads(current['response_raw']) != current['response']):
                raise ModelInputError('Native exchange differs from authenticated logical request')
        self._validate_render(render['response'], ids, reserved)
        output, text, message, action, error, native_id = self._validate_native_response(
            exchange['response'], ids, reserved, logical_request['tools'])
        expected = {'output_ids': list(output), 'text': text, 'input_token_count': len(ids),
                    'output_token_count': len(output), 'native_message': message, 'action_text': action,
                    'action_error': error, 'native_tool_call_id': native_id,
                    'projection_digest': native_projection_digest(message, action, error, consumer_profile=self._consumer_profile)}
        if any(completed_payload.get(key) != value for key, value in expected.items()):
            raise ModelInputError('Native raw output or action projection differs from recorded evidence')
