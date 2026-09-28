"""Remote candidate. Echoed token IDs do not attest engine tensors or weights."""
from dataclasses import replace
from copy import deepcopy
import hashlib
import hmac
import json
from importlib import metadata
import os
from pathlib import Path
import secrets
from threading import RLock
import time
from urllib import request as http, error as http_error

from ..contracts.agent_episode import initial_messages, recovery_guidance_identity, episode_tool_definitions
from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputRequest, ModelInputIdentity, _decode
from .agent_action_grammar import MAX_BOUND_REQUESTS, action_schema, schema_bytes
from .model_input import ModelInputExecutionError, ModelInputResult, _json_bytes

CONSUMER_PROFILE = 'nemotron-remote-vllm-action-v1'
MANIFEST_FILENAME = 'remote-vllm-manifest.json'
CHAT_TEMPLATE_KWARGS = {'enable_thinking': False}
HISTORY_RENDERING_POLICY = 'local-json-object-view-server-openai-string-wire-v1'
EXACT_TOKEN_LIMITATIONS = [
    'HTTP token IDs do not independently attest actual engine tensors or attention masks',
    'HTTP cannot attest loaded weights or live server/tokenizer configuration',
    'Request-owned KV isolation, sampler order and deterministic GPU output are not guaranteed',
    'The endpoint provides no per-token grammar masks or proof of prefix membership; complete accepted actions are checked independently',
    'Client timeout does not prove server cancellation; automatic retries are forbidden',
]


def sampling_policy_identity():
    return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
            'seed': 0, 'repetition_penalty': 1.0, 'frequency_penalty': 0.0,
            'presence_penalty': 0.0, 'n': 1, 'stream': False, 'ignore_eos': False,
            'skip_special_tokens': True, 'spaces_between_special_tokens': False}


def client_runtime_versions():
    return {name: metadata.version(name) for name in ('transformers', 'tokenizers', 'jinja2')}


def rendering_messages(messages):
    """Adapt the model template's argument type without rewriting DML history.

    DML stores function arguments as canonical JSON strings. Nemotron's template
    iterates argument mappings. Parse a private rendering view strictly; the
    original request, action text, tool results and authentication stay intact.
    """
    view = deepcopy(messages)
    for message in view:
        for call in message.get('tool_calls', []):
            try:
                raw = call['function']['arguments']
                if type(raw) is not str:
                    raise ModelInputError('Tool arguments must be JSON text')
                arguments = _decode(raw.encode('utf-8'))
                if type(arguments) is not dict:
                    raise ModelInputError('Tool arguments must encode an object')
                call['function']['arguments'] = arguments
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelInputError('Invalid tool arguments in rendering view') from exc
    return view


def verify_remote_manifest(directory):
    directory = Path(directory)
    try:
        value = json.loads((directory / MANIFEST_FILENAME).read_text())
        if value['schema_version'] != 'dml-remote-vllm-manifest-v1':
            raise ValueError('manifest version')
        if value['endpoint'] != 'http://192.168.50.91:8000/v1' or value['model'] != 'nvidia/nemotron-3-super':
            raise ValueError('candidate endpoint/model')
        if not value['model_revision'] or not isinstance(value['model_provenance'], dict):
            raise ValueError('revision provenance required')
        if not isinstance(value['server_configuration'], dict) or not value['server_configuration']:
            raise ValueError('server configuration required')
        if value['sampling'] != sampling_policy_identity():
            raise ValueError('sampling policy changed')
        if type(value['model_window_tokens']) is not int or value['model_window_tokens'] < 1:
            raise ValueError('model window')
        if type(value['timeout_seconds']) not in (int, float) or not 0 < value['timeout_seconds'] <= 3600:
            raise ValueError('timeout')
        files = value['files']
        if not {'tokenizer.json', 'tokenizer_config.json', 'chat-template.jinja'} <= files.keys():
            raise ValueError('tokenizer files required')
        for name, digest in files.items():
            if Path(name).name != name or (directory / name).is_symlink():
                raise ValueError('unsafe manifest path')
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError('frozen file changed')
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelInputError('Remote candidate manifest admission failed') from exc


def remote_identity(manifest):
    model = {'revision': manifest['model_revision'], 'provenance': manifest['model_provenance']}
    policy = {'manifest': manifest, 'consumer_profile': CONSUMER_PROFILE,
              'guidance': recovery_guidance_identity(), 'chat_template_kwargs': CHAT_TEMPLATE_KWARGS,
              'history_rendering_policy': HISTORY_RENDERING_POLICY,
              'grammar_sha256': hashlib.sha256(schema_bytes(episode_tool_definitions())).hexdigest(),
              'exact_token_limitations': EXACT_TOKEN_LIMITATIONS}
    return ModelInputIdentity(
        model_digest=hashlib.sha256(_json_bytes(model)).hexdigest(),
        tokenizer_digest=hashlib.sha256(_json_bytes({k: v for k, v in manifest['files'].items()
                                                   if k != 'chat-template.jinja'})).hexdigest(),
        chat_template_digest=manifest['files']['chat-template.jinja'],
        runtime_identity='dml-remote-vllm-action-runtime-v1:' + hashlib.sha256(_json_bytes(policy)).hexdigest(),
        model_window_tokens=manifest['model_window_tokens'])


class RemoteVLLMActionInputConsumer:
    def __init__(self, snapshot_directory, *, consumer_profile=CONSUMER_PROFILE, offline=False):
        if consumer_profile != CONSUMER_PROFILE:
            raise ModelInputError('Unknown remote consumer profile')
        from transformers import AutoTokenizer
        self._directory = Path(snapshot_directory)
        self.manifest = verify_remote_manifest(self._directory)
        if self.manifest['client_runtime_versions'] != client_runtime_versions():
            raise ModelInputError('Client tokenizer runtime versions differ from frozen candidate')
        self._identity = remote_identity(self.manifest)
        self._manifest_bytes = _json_bytes(self.manifest)
        self._consumer_profile = consumer_profile
        self._offline = offline
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._directory), local_files_only=True,
                                                        trust_remote_code=False)
        self._template = (self._directory / 'chat-template.jinja').read_text()
        self._tokenizer.chat_template = self._template
        self._lock = RLock()
        self._closed = False
        self._consumer_id = secrets.token_hex(24)
        self._auth_key = secrets.token_bytes(32)
        self._bound_requests = {}
        self.last_exchange = None
        self._tokenizer_state = self._tokenizer_fingerprint()

    @property
    def identity(self):
        return self._identity

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True
            self._bound_requests.clear()
            self._auth_key = b''

    def _require_open(self):
        if self._closed:
            raise ModelInputError('Remote consumer is closed')

    def _tokenizer_fingerprint(self):
        backend = getattr(self._tokenizer, 'backend_tokenizer', None)
        return (backend.to_str() if backend is not None else None, self._tokenizer.chat_template)

    def _validate_runtime(self):
        self._require_open()
        if (self.manifest['client_runtime_versions'] != client_runtime_versions()
                or self._tokenizer_fingerprint() != self._tokenizer_state
                or _json_bytes(verify_remote_manifest(self._directory)) != self._manifest_bytes):
            raise ModelInputError('Remote candidate manifest changed')

    def _auth_tag(self, artifact):
        return hmac.new(self._auth_key, _json_bytes(artifact.signing_payload()), hashlib.sha256).hexdigest()

    def _post(self, path, payload):
        if self._offline:
            raise ModelInputError('Offline replay cannot contact endpoint')
        evidence = Path(self.manifest['evidence_directory'])
        evidence.mkdir(parents=True, exist_ok=True)
        ident = secrets.token_hex(16)
        entry = {'schema_version': 'dml-remote-http-v1', 'id': ident, 'path': path, 'request': payload,
                 'timeout_seconds': self.manifest['timeout_seconds'], 'started_ns': time.time_ns(),
                 'response': None, 'error': None, 'timed_out': False}
        target = evidence / (ident + '.json')
        def persist():
            temporary = evidence / (ident + '.tmp')
            with temporary.open('wb') as handle:
                handle.write(_json_bytes(entry))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            fd = os.open(evidence, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        persist()
        started = time.monotonic()
        try:
            req = http.Request(self.manifest['endpoint'].removesuffix('/v1') + path,
                               data=_json_bytes(payload), headers={'Content-Type': 'application/json'})
            with http.urlopen(req, timeout=self.manifest['timeout_seconds']) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                entry['status'] = response.status
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError('Remote response exceeds evidence limit')
                entry['response_raw'] = raw.decode('utf-8')
                result = json.loads(raw)
                entry['response'] = result
                return result
        except BaseException as exc:
            entry['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            entry['timed_out'] = isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError)
            if isinstance(exc, http_error.HTTPError):
                entry['status'] = exc.code
                entry['response_raw'] = exc.read(1024 * 1024).decode('utf-8', errors='replace')
            raise
        finally:
            entry['duration_seconds'] = time.monotonic() - started
            entry['finished_ns'] = time.time_ns()
            persist()
            self.last_exchange = {**entry, 'evidence_path': str(target)}

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        self.last_exchange = None
        request = ModelInputRequest.from_payload({'messages': messages, 'tools': [] if tools is None else tools,
                                                  'output_reserved_tokens': output_reserved_tokens})
        expected = initial_messages('profile binding', consumer_profile=self._consumer_profile)[0]
        if request.messages[0] != expected:
            raise ModelInputError('Remote profile requires exact recovery system message')
        action_schema(request.tools)
        with self._lock:
            self._validate_runtime()
            if len(self._bound_requests) >= MAX_BOUND_REQUESTS:
                raise ModelInputError('Remote compiled-request capacity exhausted')
            rendered_messages = rendering_messages(request.messages)
            ids = tuple(self._tokenizer.apply_chat_template(rendered_messages, tools=request.tools,
                chat_template=self._template, tokenize=True, add_generation_prompt=True,
                continue_final_message=False, truncation=False, padding=False, **CHAT_TEMPLATE_KWARGS))
            artifact = CompiledModelInput(identity=self._identity, request_digest=request.request_digest,
                input_ids=ids, attention_mask=(1,) * len(ids), output_reserved_tokens=output_reserved_tokens,
                model_window_tokens=self._identity.model_window_tokens, consumer_id=self._consumer_id,
                nonce=secrets.token_hex(24), auth_tag='0' * 64)
            if not self._offline:
                try:
                    # vLLM accepts OpenAI string arguments and parses them before rendering.
                    # Compare all resulting IDs against our explicit private object view.
                    reply = self._post('/tokenize', {'model': self.manifest['model'], 'messages': request.messages,
                        'tools': request.tools, 'add_generation_prompt': True, 'continue_final_message': False,
                        'add_special_tokens': False, 'chat_template_kwargs': CHAT_TEMPLATE_KWARGS})
                    if (reply.get('tokens') != list(ids) or reply.get('count') != len(ids)
                            or reply.get('max_model_len') != self._identity.model_window_tokens):
                        raise ModelInputError('Server tokenization/window differs from frozen candidate')
                except ModelInputError:
                    raise
                except Exception as exc:
                    raise ModelInputError('Remote compilation/token evidence failed') from exc
            artifact = replace(artifact, auth_tag=self._auth_tag(artifact))
            self._bound_requests[artifact.artifact_digest] = request
            return artifact

    def decode_output(self, output_ids):
        return self._tokenizer.decode(list(output_ids), skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False,
                                      spaces_between_special_tokens=False)

    def request_payload(self, input_ids, output_reserved_tokens, tools):
        return {'model': self.manifest['model'], 'prompt': list(input_ids),
                'max_tokens': output_reserved_tokens, 'add_special_tokens': False,
                'return_token_ids': True, **sampling_policy_identity(),
                'structured_outputs': {'json': action_schema(tools)}}

    def _validate_response(self, response, input_ids, reserved):
        choices = response['choices']
        if response['model'] != self.manifest['model'] or len(choices) != 1:
            raise ValueError('Unexpected model or choice count')
        choice = choices[0]
        output = choice['token_ids']
        if (choice.get('index') != 0 or choice.get('prompt_token_ids') != list(input_ids)
                or type(output) is not list or not 0 < len(output) <= reserved
                or any(type(t) is not int or not 0 <= t < len(self._tokenizer) for t in output)):
            raise ValueError('Missing, substituted or out-of-budget token evidence')
        usage = response['usage']
        if (any(type(usage.get(k)) is not int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                or usage['prompt_tokens'] != len(input_ids) or usage['completion_tokens'] != len(output)
                or usage['total_tokens'] != len(input_ids) + len(output)):
            raise ValueError('Token usage differs from returned IDs')
        if choice['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Unexpected finish reason')
        text = self.decode_output(output)
        if text != choice['text']:
            raise ValueError('Returned text differs from independently decoded output IDs')
        return tuple(output), text

    def validate_exchange(self, compiled_payload, logical_request, exchange, completed_payload):
        ids = compiled_payload['input_ids']
        reserved = compiled_payload['output_reserved_tokens']
        if (exchange['path'] != '/v1/completions' or exchange['error'] is not None
                or exchange['timed_out'] is not False or exchange['status'] != 200
                or exchange['timeout_seconds'] != self.manifest['timeout_seconds']
                or exchange['request'] != self.request_payload(ids, reserved, logical_request['tools'])
                or json.loads(exchange['response_raw']) != exchange['response']):
            raise ValueError('Remote exchange differs from candidate/request')
        output, text = self._validate_response(exchange['response'], ids, reserved)
        if (list(output) != completed_payload['output_ids'] or text != completed_payload['text']
                or completed_payload['input_token_count'] != len(ids)
                or completed_payload['output_token_count'] != len(output)):
            raise ValueError('Remote exchange differs from completed event')

    def execute(self, artifact):
        with self._lock:
            self.last_exchange = None
            self._validate_runtime()
            if type(artifact) is not CompiledModelInput:
                raise ModelInputError('Authenticated compiled input required')
            artifact.validate()
            if (artifact.consumer_id != self._consumer_id or artifact.identity != self._identity
                    or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))):
                raise ModelInputError('Remote input authentication failed')
            request = self._bound_requests.get(artifact.artifact_digest)
            if request is None or request.request_digest != artifact.request_digest:
                raise ModelInputError('Remote input has no immutable request binding')
            try:
                response = self._post('/v1/completions', self.request_payload(
                    artifact.input_ids, artifact.output_reserved_tokens, request.tools))
                output, text = self._validate_response(response, artifact.input_ids, artifact.output_reserved_tokens)
                self._validate_runtime()
                return ModelInputResult(artifact.artifact_digest, len(artifact.input_ids), len(output), output, text)
            except Exception as exc:
                raise ModelInputExecutionError('Remote generation failed or returned invalid token evidence') from exc
