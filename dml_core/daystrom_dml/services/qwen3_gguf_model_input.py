"""Exact-token CPU companion for the separately admitted vendor Qwen3 GGUF.

GGUF quantized inference is an explicit new numerical runtime, not a claim of
BF16 tensor equivalence. No llama.cpp tokenizer, sampler or chat renderer runs
on the execution path. The authenticated HF token IDs are the only inputs.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
from pathlib import Path
import secrets
from threading import RLock

from ..contracts.model_input import CompiledModelInput, ModelInputBudgetError, ModelInputError, ModelInputRequest
from .model_input import LocalTransformersInputConsumer, _json_bytes


def backend_identity():
    """Bind the installed Python bridge and every shipped native library."""
    import llama_cpp

    if (importlib.metadata.version('llama-cpp-python') != '0.3.35'
            or llama_cpp.llama_supports_gpu_offload() or llama_cpp.llama_supports_rpc()):
        raise ModelInputError('Only the pinned CPU-only llama.cpp runtime is admitted')
    root = Path(llama_cpp.__file__).resolve().parent
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.so', '.dll', '.dylib'):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files or not any(name.endswith(('.so', '.dll', '.dylib')) for name in files):
        raise ModelInputError('The admitted llama.cpp native runtime is unavailable')
    return {'version': importlib.metadata.version('llama-cpp-python'), 'files': files,
            'backend': 'llama.cpp_cpu_vendor_Q4_K_M', 'gpu_layers': 0, 'threads': 4,
            'load_mode': 'MMAP', 'extra_buffer_repacking': False, 'check_tensors': True,
            'maximum_logits_outputs': 1,
            'threads_batch': 4, 'batch_tokens': 512, 'microbatch_tokens': 512,
            'context': 'fresh_per_request_input_plus_output_reservation',
            'kv_cache': 'F16_no_cross_request_state', 'flash_attention': False,
            'context_shift': False, 'implicit_bos': False,
            'logits': 'last_requested_batch_token_only', 'native_sampling': False,
            'system_info': llama_cpp.llama_print_system_info().decode('utf-8')}


class LocalQwen3GGUFInputConsumer(LocalTransformersInputConsumer):
    """Own a verified GGUF model; compilation remains authenticated and bounded."""

    def __init__(self, snapshot_directory):
        from .qwen3_gguf_model_snapshot import verify_qwen3_gguf_snapshot, runtime_versions
        from transformers import PreTrainedTokenizerFast
        import torch

        self._lock = RLock()
        self._closed = True
        self._backend = None
        self._snapshot_context = verify_qwen3_gguf_snapshot(snapshot_directory)
        try:
            self._snapshot = self._snapshot_context.__enter__()
            self._torch = torch
            self._identity = self._snapshot.identity
            self._template = self._snapshot.chat_template
            from .qwen3_model_snapshot import QWEN3_CHAT_TEMPLATE
            if self._template != QWEN3_CHAT_TEMPLATE:
                raise ModelInputError('GGUF action input requires the unchanged DML non-thinking template')
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self._snapshot.path / 'tokenizer.json'),
                **self._snapshot.special_tokens)
            self._tokenizer.chat_template = self._template
            self._tokenizer.model_max_length = self._identity.model_window_tokens
            self._tokenizer.backend_tokenizer.no_truncation()
            self._tokenizer.backend_tokenizer.no_padding()
            self._vocab_size = self._snapshot.config['vocab_size']
            vocabulary = self._tokenizer.get_vocab()
            self._allowed_ids = frozenset(vocabulary.values())
            if (len(self._allowed_ids) != len(vocabulary)
                    or any(type(i) is not int or not 0 <= i < self._vocab_size for i in self._allowed_ids)):
                raise ModelInputError('HF vocabulary does not match the admitted GGUF rows')
            for name in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>'):
                index = vocabulary.get(name)
                marker = self._tokenizer.added_tokens_decoder.get(index)
                if (type(index) is not int or marker is None
                        or marker.special is not (name not in ('<think>', '</think>'))
                        or self._tokenizer.encode(name, add_special_tokens=False) != [index]):
                    raise ModelInputError('HF framing marker differs from its admitted single token')
            if (self._tokenizer.eos_token_id != self._snapshot.config['eos_token_id']
                    or self._tokenizer.pad_token_id != self._snapshot.config['bos_token_id']):
                raise ModelInputError('HF configured special IDs differ')
            self._runtime_versions = runtime_versions()
            if self._runtime_versions != dict(self._snapshot.manifest.runtime_versions):
                raise ModelInputError('GGUF runtime versions differ from the verified snapshot')
            self._backend_identity = self._active_backend_identity()
            self._backend = self._create_backend(self._snapshot.path / 'model.gguf', self._vocab_size)
            self._backend.verify_tokenizer(self._tokenizer)
            self._snapshot.validate_integrity()
            self._runtime_digest = self._fingerprint()
            self._consumer_id = secrets.token_hex(24)
            self._auth_key = secrets.token_bytes(32)
            self._closed = False
        except BaseException:
            if self._backend is not None:
                self._backend.close()
            self._snapshot_context.__exit__(None, None, None)
            raise

    def _active_backend_identity(self):
        return backend_identity()

    def _create_backend(self, path, vocab_size):
        return _GGUFBackend(path, vocab_size)

    def _fingerprint(self):
        tokenizer = self._tokenizer
        return hashlib.sha256(_json_bytes({
            'identity': self._identity.to_payload(), 'template': self._template,
            'tokenizer_backend': tokenizer.backend_tokenizer.to_str(),
            'special_tokens': tokenizer.special_tokens_map,
            'tokenizer_template': tokenizer.chat_template,
            'tokenizer_window': tokenizer.model_max_length,
            'padding': tokenizer.padding_side, 'truncation': tokenizer.truncation_side,
            'backend': self._active_backend_identity(), 'vocab_size': self._vocab_size,
            'template_options': {'add_generation_prompt': False},
        })).hexdigest()

    def _validate_runtime(self):
        from .qwen3_gguf_model_snapshot import runtime_versions

        self._require_open()
        self._snapshot.validate_integrity()
        if self._runtime_versions != runtime_versions() or self._fingerprint() != self._runtime_digest:
            raise ModelInputError('GGUF tokenizer, backend, template or runtime identity changed')
        self._backend.validate()

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload({
            'messages': messages, 'tools': [] if tools is None else tools,
            'output_reserved_tokens': output_reserved_tokens})
        with self._lock:
            self._validate_runtime()
            if request.output_reserved_tokens >= self._identity.model_window_tokens:
                raise ModelInputBudgetError('Output reservation leaves no room for model input')
            try:
                encoded = self._tokenizer.apply_chat_template(
                    request.messages, tools=request.tools, chat_template=self._template,
                    tokenize=True, add_generation_prompt=False,
                    continue_final_message=False, truncation=False, padding=False,
                    return_dict=True, tokenizer_kwargs={'return_attention_mask': True})
                ids, mask = tuple(encoded['input_ids']), tuple(encoded['attention_mask'])
            except Exception as exc:
                raise ModelInputError('Exact GGUF input template or tokenization failed') from exc
            if any(type(i) is not int or i not in self._allowed_ids for i in ids):
                raise ModelInputError('Tokenizer returned an unadmitted GGUF vocabulary row')
            if mask != (1,) * len(ids):
                raise ModelInputError('GGUF exact input excludes masked input and padding')
            artifact = CompiledModelInput(
                identity=self._identity, request_digest=request.request_digest,
                input_ids=ids, attention_mask=mask, output_reserved_tokens=request.output_reserved_tokens,
                model_window_tokens=self._identity.model_window_tokens,
                consumer_id=self._consumer_id, nonce=secrets.token_hex(24), auth_tag='0' * 64)
            self._validate_runtime()
            return replace(artifact, auth_tag=self._auth_tag(artifact))

    def execute(self, artifact):
        raise ModelInputError('The GGUF model requires its explicit constrained action consumer')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._auth_key = b''
            self._backend.close()
            self._tokenizer = None
            self._snapshot_context.__exit__(None, None, None)


class _GGUFBackend:
    """Narrow low-level API: model weights shared, each decode context disposable."""

    def __init__(self, path, vocab_size):
        import ctypes
        from llama_cpp import llama_cpp as api
        from llama_cpp._internals import LlamaModel

        self._api = api
        api.llama_backend_init()
        params = api.llama_model_default_params()
        params.n_gpu_layers = 0
        params.load_mode = api.LLAMA_LOAD_MODE_MMAP
        params.use_extra_bufts = False
        params.no_host = False
        params.no_alloc = False
        params.load_mtp = False
        params.vocab_only = False
        params.check_tensors = True
        self._configure_model_params(params, api)
        self._model = LlamaModel(path_model=str(path), params=params, verbose=False)
        self._params_bytes = bytes(params)
        self._model_pointer = ctypes.cast(self._model.model, ctypes.c_void_p).value
        self._vocab_size = vocab_size
        if self._model.n_vocab() != vocab_size:
            self.close()
            raise ModelInputError('GGUF output row count differs from HF configuration')

    def _configure_model_params(self, params, api):
        """CPU defaults are immutable unless an explicit separate backend overrides."""

    def verify_tokenizer(self, tokenizer):
        vocabulary = tokenizer.get_vocab()
        # Check every used row, not merely a convenient sample of text strings.
        if any(self._model.token_get_text(index) != token for token, index in vocabulary.items()):
            raise ModelInputError('GGUF and HF token-to-row vocabularies differ')
        if (self._model.token_eos() != tokenizer.eos_token_id
                or self._model.token_bos() != tokenizer.pad_token_id
                or self._model.add_bos_token() or self._model.add_eos_token()):
            raise ModelInputError('GGUF special-token or implicit-token policy differs')

    def validate(self):
        import ctypes
        if (self._model.model is None
                or ctypes.cast(self._model.model, ctypes.c_void_p).value != self._model_pointer
                or bytes(self._model.params) != self._params_bytes
                or self._model.n_vocab() != self._vocab_size):
            raise ModelInputError('GGUF admitted model handle or CPU parameters changed')

    def request(self, input_ids, output_reserved_tokens):
        self.validate()
        return _GGUFRequest(self._model, self._api, input_ids, output_reserved_tokens, self._vocab_size)

    def close(self):
        self._model.close()


class _GGUFRequest:
    """Feed exact batches at absolute positions; read only the last token logits."""

    def __init__(self, model, api, input_ids, output_reserved_tokens, vocab_size):
        from llama_cpp._internals import LlamaBatch, LlamaContext

        self._context = None
        self._batch = None
        self._position = 0
        self._vocab_size = vocab_size
        self._limit = len(input_ids) + output_reserved_tokens
        if not input_ids or output_reserved_tokens < 1 or self._limit > model.n_ctx_train():
            raise ModelInputBudgetError('GGUF request exceeds its admitted model window')
        params = api.llama_context_default_params()
        params.n_ctx = self._limit
        params.n_batch = 512
        params.n_ubatch = 512
        params.n_seq_max = 1
        params.n_outputs_max = 1
        params.n_outputs_max_per_seq = 1
        params.n_threads = 4
        params.n_threads_batch = 4
        params.type_k = api.GGML_TYPE_F16
        params.type_v = api.GGML_TYPE_F16
        params.flash_attn_type = api.LLAMA_FLASH_ATTN_TYPE_DISABLED
        params.offload_kqv = False
        params.op_offload = False
        params.embeddings = False
        self._configure_context_params(params)
        try:
            self._context = LlamaContext(model=model, params=params, verbose=False)
            # llama.cpp may round allocation upward; the logical request bound is
            # still exactly _limit and is checked before every decode.
            if self._context.n_ctx() < self._limit:
                raise ModelInputError('GGUF backend reduced the required request context')
            self._batch = LlamaBatch(n_tokens=512, embd=0, n_seq_max=1, verbose=False)
            for start in range(0, len(input_ids), 512):
                self._decode(input_ids[start:start + 512])
        except BaseException:
            self.close()
            raise

    def _configure_context_params(self, params):
        """CPU context defaults; explicit CUDA subclass owns its separate policy."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _decode(self, tokens):
        if (not tokens or len(tokens) > 512 or self._position + len(tokens) > self._limit
                or any(type(i) is not int or not 0 <= i < self._vocab_size for i in tokens)):
            raise ModelInputError('GGUF exact-token decode exceeds its bound or vocabulary')
        self._batch.set_batch(tokens, n_past=self._position, logits_all=False)
        self._context.decode(self._batch)
        self._position += len(tokens)

    def logits(self):
        import numpy as np
        # -1 refers to the last requested logits row in the current batch. It is
        # NOT the absolute sequence position and does not address a scores cache.
        pointer = self._context.get_logits_ith(-1)
        if not pointer:
            raise ModelInputError('GGUF backend did not return last-position logits')
        return np.ctypeslib.as_array(pointer, shape=(self._vocab_size,)).copy()

    def advance(self, token):
        self._decode((token,))

    def close(self):
        if self._batch is not None:
            self._batch.close()
            self._batch = None
        if self._context is not None:
            self._context.close()
            self._context = None
