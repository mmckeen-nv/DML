"""Synthetic transport controls, never evidence of trained GGUF task quality."""
from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from daystrom_dml.contracts.agent_episode import QWEN3_GGUF_CONSUMER_PROFILE, initial_messages
from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services.model_input import ModelInputExecutionError
from daystrom_dml.services import qwen3_gguf_action_input as action
from daystrom_dml.services import qwen3_gguf_model_input as model
from test_agent_action_grammar import grammar_runtime as grammar_runtime
from test_qwen3_model_input import snapshot as snapshot
from test_qwen_action_input import _action, _tools


@pytest.fixture
def consumer(snapshot, grammar_runtime, monkeypatch):
    from daystrom_dml.services import qwen3_gguf_model_snapshot as foundation
    from daystrom_dml.services.qwen3_model_snapshot import verify_qwen3_snapshot
    from daystrom_dml.services.model_input import _versions

    @contextmanager
    def verified(_path):
        with verify_qwen3_snapshot(snapshot.path) as original:
            proxy = SimpleNamespace(path=original.path, config=original.config,
                chat_template=original.chat_template, special_tokens=original.special_tokens,
                manifest=original.manifest, validate_integrity=original.validate_integrity,
                identity=replace(original.identity, runtime_identity='dml-qwen3-gguf-model-input-runtime-v1:' + 'a' * 64))
            yield proxy

    class Backend:
        def __init__(self, path, vocab):
            self.closed = False
            self.requests = []
            self.vocab = vocab
            self.tokens = []
            self.contexts = []

        def verify_tokenizer(self, tokenizer):
            self.tokenizer = tokenizer

        def validate(self):
            assert not self.closed

        def close(self):
            self.closed = True

        @contextmanager
        def request(self, ids, reserved):
            import numpy as np
            self.requests.append((ids, reserved))
            state = SimpleNamespace(index=0, closed=False, advances=[])
            self.contexts.append(state)
            def logits():
                values = np.full(self.vocab, -1000.0, dtype=np.float32)
                values[self.tokens[state.index]] = 1000.0
                return values
            def advance(token):
                state.advances.append(token)
                state.index += 1
            try:
                yield SimpleNamespace(logits=logits, advance=advance)
            finally:
                state.closed = True

    monkeypatch.setattr(foundation, 'verify_qwen3_gguf_snapshot', verified)
    monkeypatch.setattr(foundation, 'runtime_versions', _versions)
    monkeypatch.setattr(model, '_GGUFBackend', Backend)
    monkeypatch.setattr(model, 'backend_identity', lambda: {'synthetic': 'transport-only'})
    monkeypatch.setattr(action, 'backend_identity', lambda: {'synthetic': 'transport-only'})
    with action.LocalQwen3GGUFActionInputConsumer(snapshot.path, consumer_profile=QWEN3_GGUF_CONSUMER_PROFILE) as value:
        yield value


def messages():
    return initial_messages('Read a memory.', consumer_profile=QWEN3_GGUF_CONSUMER_PROFILE)


def test_authenticated_exact_ids_fresh_context_and_single_eos(consumer, monkeypatch):
    first = consumer.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    second = consumer.compile(messages(), _tools('retire'), output_reserved_tokens=128)
    text = json.dumps(_action('retrieve'), separators=(',', ':'))
    consumer._backend.tokens = consumer._tokenizer.encode(text, add_special_tokens=False) + [consumer._tokenizer.eos_token_id]
    # Execution must not tokenize, render or insert a BOS token.
    monkeypatch.setattr(consumer._tokenizer, 'apply_chat_template', lambda *a, **k: pytest.fail('retokenized'))
    results = [consumer.execute(first), consumer.execute(first)]
    assert all(result.text == text for result in results)
    assert all(result.output_token_count == len(consumer._backend.tokens) for result in results)
    assert all(result.output_ids[-1] == consumer._tokenizer.eos_token_id for result in results)
    assert consumer._backend.requests == [(first.input_ids, 128)] * 2
    assert all(state.closed for state in consumer._backend.contexts)
    assert consumer._backend.contexts[0] is not consumer._backend.contexts[1]
    assert second.artifact_digest in consumer._bound_requests
    assert all(state.advances == consumer._backend.tokens[:-1] for state in consumer._backend.contexts)


@pytest.mark.parametrize('change', ['artifact', 'bound_request', 'template', 'backend', 'grammar', 'sampling'])
def test_authentication_and_runtime_drift_before_decode(consumer, monkeypatch, change):
    artifact = consumer.compile(messages(), _tools('retrieve'), output_reserved_tokens=128)
    if change == 'artifact':
        artifact = replace(artifact, input_ids=artifact.input_ids[:-1] + (0,))
    elif change == 'bound_request':
        request = json.loads(consumer._bound_requests[artifact.artifact_digest])
        request['messages'][-1]['content'] += 'changed'
        consumer._bound_requests[artifact.artifact_digest] = json.dumps(request).encode()
    elif change == 'template':
        consumer._tokenizer.chat_template += 'changed'
    elif change == 'backend':
        monkeypatch.setattr(model, 'backend_identity', lambda: {'changed': True})
    elif change == 'grammar':
        consumer._grammar_policy = {}
    else:
        consumer._sampling_policy['seed'] = 1
    with pytest.raises(ModelInputError):
        consumer.execute(artifact)
    assert not consumer._backend.requests


def test_decode_exception_closes_context_and_never_repairs(consumer):
    artifact = consumer.compile(messages(), [], output_reserved_tokens=2)
    consumer._backend.tokens = []
    with pytest.raises(ModelInputExecutionError):
        consumer.execute(artifact)
    assert consumer._backend.contexts[0].closed
    assert len(consumer._backend.requests) == 1


def test_reservation_returns_original_incomplete_output(consumer):
    artifact = consumer.compile(messages(), [], output_reserved_tokens=1)
    consumer._backend.tokens = consumer._tokenizer.encode('{', add_special_tokens=False)
    result = consumer.execute(artifact)
    assert result.output_ids == tuple(consumer._backend.tokens)
    assert result.text == '{'
    assert result.output_token_count == 1
    assert consumer._backend.contexts[0].advances == []


def test_sampling_matches_transformers_order_and_restores_rng():
    import torch
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    scores = torch.arange(40, dtype=torch.float32).reshape(1, -1) / 10
    inputs = torch.tensor([[1, 2]], dtype=torch.long)
    before = torch.random.get_rng_state().clone()
    with action._sampled_cpu_rng(torch, 0):
        actual = [action._sample_next(torch, inputs, scores.clone()) for _ in range(5)]
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        transformed = scores.clone()
        for warper in [TemperatureLogitsWarper(.7), TopKLogitsWarper(20), TopPLogitsWarper(.8)]:
            transformed = warper(inputs, transformed)
        expected = [int(torch.multinomial(transformed.softmax(-1), 1).item()) for _ in range(5)]
    assert actual == expected


def test_unknown_profile_rejected_without_backend():
    with pytest.raises(ModelInputError, match='profile'):
        action.LocalQwen3GGUFActionInputConsumer('/not/loaded', consumer_profile='auto')


def test_native_batches_use_absolute_positions_last_row_and_fresh_bounded_context(monkeypatch):
    import ctypes
    import sys
    from types import ModuleType
    import numpy as np

    calls, contexts, batches = [], [], []
    native = ModuleType('llama_cpp._internals')
    values = (ctypes.c_float * 8)(*range(8))
    class Context:
        def __init__(self, *, model, params, verbose):
            self.params = params
            self.closed = False
            contexts.append(self)
        def n_ctx(self):
            return self.params.n_ctx + 16  # backend may round allocation up
        def decode(self, batch):
            calls.append((batch.ids, batch.past, batch.all))
        def get_logits_ith(self, index):
            assert index == -1
            return ctypes.cast(values, ctypes.POINTER(ctypes.c_float))
        def close(self):
            self.closed = True
    class Batch:
        def __init__(self, **kwargs):
            assert kwargs == {'n_tokens': 512, 'embd': 0, 'n_seq_max': 1, 'verbose': False}
            self.closed = False
            batches.append(self)
        def set_batch(self, ids, n_past, logits_all):
            self.ids, self.past, self.all = tuple(ids), n_past, logits_all
        def close(self):
            self.closed = True
    native.LlamaContext, native.LlamaBatch = Context, Batch
    monkeypatch.setitem(sys.modules, 'llama_cpp._internals', native)
    api = SimpleNamespace(llama_context_default_params=lambda: SimpleNamespace(),
                          GGML_TYPE_F16=1, LLAMA_FLASH_ATTN_TYPE_DISABLED=0)
    handle = SimpleNamespace(n_ctx_train=lambda: 32768)
    ids = (1,) * 600
    with model._GGUFRequest(handle, api, ids, 256, 8) as request:
        assert np.array_equal(request.logits(), np.arange(8, dtype=np.float32))
        assert request.logits().dtype == np.float32
        request.advance(2)
        assert request._position == 601
    assert calls == [(ids[:512], 0, False), (ids[512:], 512, False), ((2,), 600, False)]
    assert contexts[0].params.n_ctx == 856
    assert contexts[0].params.n_threads == contexts[0].params.n_threads_batch == 4
    assert contexts[0].params.type_k == contexts[0].params.type_v == 1
    assert contexts[0].params.offload_kqv is contexts[0].params.op_offload is False
    assert contexts[0].closed and batches[0].closed
    with model._GGUFRequest(handle, api, (1,), 1, 8) as request:
        request.advance(2)
        with pytest.raises(ModelInputError, match='bound'):
            request.advance(3)
    assert contexts[1] is not contexts[0]
    assert calls[-2:] == [((1,), 0, False), ((2,), 1, False)]


def test_complete_vocab_and_special_flags_fail_closed():
    backend = model._GGUFBackend.__new__(model._GGUFBackend)
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 0, '<|im_end|>': 1}, eos_token_id=1, pad_token_id=2)
    backend._model = SimpleNamespace(token_get_text=lambda index: ['a', '<|im_end|>'][index],
        token_eos=lambda: 1, token_bos=lambda: 2, add_bos_token=lambda: False, add_eos_token=lambda: False)
    backend.verify_tokenizer(tokenizer)
    backend._model.token_get_text = lambda index: 'wrong'
    with pytest.raises(ModelInputError, match='vocabularies'):
        backend.verify_tokenizer(tokenizer)
    backend._model.token_get_text = lambda index: ['a', '<|im_end|>'][index]
    backend._model.add_bos_token = lambda: True
    with pytest.raises(ModelInputError, match='implicit'):
        backend.verify_tokenizer(tokenizer)


def test_pinned_native_abi_without_loading_weights():
    import importlib.util
    import os
    if importlib.util.find_spec('llama_cpp') is None:
        if os.environ.get('REQUIRE_MODEL_INPUT_TESTS') == '1':
            pytest.fail('Mandatory GGUF model-input lane requires llama-cpp-python')
        pytest.skip('Optional GGUF native dependency is unavailable')
    import llama_cpp
    from llama_cpp import llama_cpp as api
    from llama_cpp._internals import LlamaModel, LlamaContext, LlamaBatch

    identity = model.backend_identity()
    assert identity['version'] == '0.3.35'
    assert not llama_cpp.llama_supports_gpu_offload()
    assert not llama_cpp.llama_supports_rpc()
    assert any(name.endswith('.so') for name in identity['files'])
    params = api.llama_context_default_params()
    for name in ('n_ctx', 'n_batch', 'n_ubatch', 'n_seq_max', 'n_outputs_max', 'n_outputs_max_per_seq', 'n_threads', 'n_threads_batch',
                 'type_k', 'type_v', 'flash_attn_type', 'offload_kqv', 'op_offload', 'embeddings'):
        assert hasattr(params, name)
    for name in ('n_gpu_layers', 'load_mode', 'use_extra_bufts', 'no_host', 'no_alloc', 'load_mtp', 'vocab_only', 'check_tensors'):
        assert hasattr(api.llama_model_default_params(), name)
    assert type(api.LLAMA_LOAD_MODE_MMAP) is int
    assert type(api.GGML_TYPE_F16) is int
    assert type(api.LLAMA_FLASH_ATTN_TYPE_DISABLED) is int
    for cls, names in [(LlamaModel, ('n_vocab', 'n_ctx_train', 'token_get_text', 'token_bos',
                                    'token_eos', 'add_bos_token', 'add_eos_token', 'close')),
                       (LlamaContext, ('n_ctx', 'decode', 'get_logits_ith', 'close')),
                       (LlamaBatch, ('set_batch', 'close'))]:
        assert all(callable(getattr(cls, name, None)) for name in names)
