"""Fail-closed CUDA placement admission, without loading a model or requiring a GPU."""
from types import SimpleNamespace
import sys

import pytest

from daystrom_dml.contracts.agent_episode import (
    QWEN3_GGUF_CONSUMER_PROFILE, QWEN3_GGUF_ARM64_CONSUMER_PROFILE,
    QWEN3_GGUF_CUDA_CONSUMER_PROFILE,
)
from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services import qwen3_gguf_action_input as action
from daystrom_dml.services import qwen3_gguf_model_input as cpu
from daystrom_dml.services import qwen3_gguf_cuda_model_input as cuda


def test_full_offload_requires_actual_layer_and_buffer_evidence():
    valid = ['load_tensors: offloading output layer to GPU\n',
             'load_tensors: offloaded 33/33 layers to GPU\n',
             'load_tensors: CUDA0 model buffer size = 4500.00 MiB\n']
    proof = cuda._placement(valid, context=False)
    assert proof['offloaded_layers'] == [('33', '33')]
    for bad in [[], valid[:-1], valid[1:], [x.replace('33/33', '32/33') for x in valid],
                [x.replace('CUDA0', 'CPU') for x in valid],
                [x.replace('4500.00', '0.00') for x in valid]]:
        with pytest.raises(ModelInputError, match='placement'):
            cuda._placement(bad, context=False)


def test_context_requires_cuda_kv_and_compute_not_just_gpu_model():
    valid = ['init: CUDA0 KV buffer size = 64.00 MiB\n',
             'reserve: CUDA0 compute buffer size = 256.00 MiB\n']
    assert cuda._placement(valid, context=True)['buffer_mib']['KV'] == [64.0]
    for bad in [[], valid[:1], valid[1:], [x.replace('CUDA0', 'CPU') for x in valid]]:
        with pytest.raises(ModelInputError, match='placement'):
            cuda._placement(bad, context=True)


@pytest.mark.parametrize('key,value', [('CUDA_VISIBLE_DEVICES', ''), ('CUDA_VISIBLE_DEVICES', '1'),
    ('GGML_CUDA_ENABLE_UNIFIED_MEMORY', '0'), ('GGML_CUDA_FORCE_MMQ', '1')])
def test_undeclared_environment_rejected_before_device_query(monkeypatch, key, value):
    monkeypatch.setattr(cuda.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'aarch64')
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(cuda.subprocess, 'check_output', lambda *a, **k: pytest.fail('queried device'))
    with pytest.raises(ModelInputError):
        cuda._gpu_descriptor()


@pytest.mark.parametrize('profile', [QWEN3_GGUF_CONSUMER_PROFILE, QWEN3_GGUF_ARM64_CONSUMER_PROFILE])
def test_cpu_profiles_still_reject_cuda_library_before_backend_load(monkeypatch, profile):
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(llama_supports_gpu_offload=lambda: True,
                                                               llama_supports_rpc=lambda: False))
    monkeypatch.setattr(cpu.importlib.metadata, 'version', lambda name: '0.3.35')
    instance = object.__new__(action.LocalQwen3GGUFActionInputConsumer)
    instance._consumer_profile = profile
    with pytest.raises(ModelInputError, match='CPU-only'):
        instance._active_backend_identity()


def test_only_explicit_cuda_profile_selects_cuda_backend(monkeypatch):
    sentinels = {'cpu': object(), 'cuda': object()}
    monkeypatch.setattr(cpu, '_GGUFBackend', lambda *a: sentinels['cpu'])
    monkeypatch.setattr(cuda, '_CUDAGGUFBackend', lambda *a: sentinels['cuda'])
    for profile, expected in [(QWEN3_GGUF_CONSUMER_PROFILE, 'cpu'),
                              (QWEN3_GGUF_ARM64_CONSUMER_PROFILE, 'cpu'),
                              (QWEN3_GGUF_CUDA_CONSUMER_PROFILE, 'cuda')]:
        instance = object.__new__(action.LocalQwen3GGUFActionInputConsumer)
        instance._consumer_profile = profile
        assert instance._create_backend('unused', 1) is sentinels[expected]


def test_gpu_params_do_not_change_cpu_defaults():
    params = SimpleNamespace(n_gpu_layers=0, split_mode=99, main_gpu=99)
    cpu._GGUFBackend._configure_model_params(None, params, None)
    assert params.n_gpu_layers == 0
    cuda._CUDAGGUFBackend._configure_model_params(None, params, SimpleNamespace(LLAMA_SPLIT_MODE_NONE=0))
    assert vars(params) == {'n_gpu_layers': -1, 'split_mode': 0, 'main_gpu': 0}
    context = SimpleNamespace(offload_kqv=False, op_offload=False, flash_attn_type=0)
    cpu._GGUFRequest._configure_context_params(None, context)
    assert context.offload_kqv is False
    cuda._CUDAGGUFRequest._configure_context_params(None, context)
    assert vars(context) == {'offload_kqv': True, 'op_offload': True, 'flash_attn_type': 0}
