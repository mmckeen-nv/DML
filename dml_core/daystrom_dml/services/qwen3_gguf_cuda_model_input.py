"""Explicit CUDA GGUF inference; HF compilation and Torch sampling remain on CPU."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import subprocess
from threading import RLock

from ..contracts.model_input import ModelInputError
from .qwen3_gguf_model_input import _GGUFBackend, _GGUFRequest

_LOG_LOCK = RLock()


def _gpu_descriptor():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ModelInputError('CUDA GGUF profile requires Linux aarch64')
    if any(key.startswith('GGML_CUDA_') for key in os.environ):
        raise ModelInputError('CUDA GGUF excludes undeclared CUDA environment overrides')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
        raise ModelInputError('CUDA GGUF requires the explicitly admitted device zero')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,compute_cap,driver_version',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(', ') for line in raw.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][2] != '10.3':
        raise ModelInputError('CUDA GGUF requires one SM103 device')
    return dict(zip(('uuid', 'name', 'compute_capability', 'driver_version'), rows[0]))


@lru_cache(maxsize=256)
def _file_digest(path, signature):
    # ctime/inode/size/mtime bind the cached full-file hash to this file generation.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_digest(path):
    stat = path.stat()
    return _file_digest(str(path), (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def _linked_files(root):
    paths = set()
    for library in sorted(root.rglob('*.so')):
        output = subprocess.check_output(['ldd', str(library)], text=True)
        if 'not found' in output:
            raise ModelInputError('CUDA native runtime has unresolved linked libraries')
        for line in output.splitlines():
            for part in line.split():
                if part.startswith('/') and Path(part).is_file():
                    paths.add(Path(part).resolve())
    driver = subprocess.check_output(['/sbin/ldconfig', '-p'], text=True)
    candidates = [Path(line.split('=>', 1)[1].strip()).resolve() for line in driver.splitlines()
                  if 'libcuda.so.1 ' in line and '=>' in line]
    if len(set(candidates)) != 1:
        raise ModelInputError('CUDA driver library is missing or ambiguous')
    paths.update(candidates)
    if not any('libcudart' in p.name for p in paths) or not any('libcublas' in p.name for p in paths):
        raise ModelInputError('CUDA runtime and cuBLAS linkage are required')
    return {str(p): _native_digest(p) for p in sorted(paths)}


def cuda_backend_identity():
    import llama_cpp
    import torch
    device = _gpu_descriptor()
    if (metadata.version('llama-cpp-python') != '0.3.35'
            or not llama_cpp.llama_supports_gpu_offload() or llama_cpp.llama_supports_rpc()
            or torch.__version__ != '2.8.0+cpu' or torch.version.cuda is not None):
        raise ModelInputError('CUDA GGUF requires pinned CUDA llama.cpp and unchanged CPU Torch sampler')
    root = Path(llama_cpp.__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.so')}
    if not any('ggml-cuda' in name and name.endswith('.so') for name in files):
        raise ModelInputError('CUDA native backend library is missing')
    return {'version': '0.3.35', 'files': files, 'backend': 'llama.cpp_cuda_vendor_Q4_K_M',
            'device': device, 'gpu_layers': -1, 'split_mode': 'NONE', 'main_gpu': 0,
            'placement': 'all_repeating_and_output_layers_cuda_input_embedding_host',
            'load_mode': 'MMAP', 'extra_buffer_repacking': False, 'check_tensors': True,
            'threads': 4, 'threads_batch': 4, 'batch_tokens': 512, 'microbatch_tokens': 512,
            'maximum_logits_outputs': 1, 'context': 'fresh_per_request_input_plus_output_reservation',
            'kv_cache': 'CUDA_F16_no_cross_request_state', 'offload_kqv': True, 'op_offload': True,
            'flash_attention': False, 'context_shift': False, 'implicit_bos': False,
            'logits': 'last_requested_batch_token_only_host_copy', 'native_sampling': False,
            'sampler': 'unchanged_cpu_torch', 'placement_admission': 'native_layer_and_buffer_logs_v1',
            'linked_files': _linked_files(root),
            'system_info': llama_cpp.llama_print_system_info().decode('utf-8')}


@contextmanager
def _capture_logs(api):
    # Native logger is process-global. Restore the prior callback even on failure.
    with _LOG_LOCK:
        previous = api.llama_log_callback()
        data = ctypes.c_void_p()
        api.llama_log_get(ctypes.byref(previous), ctypes.byref(data))
        lines = []
        def collect(_level, text, _data):
            if sum(map(len, lines)) < 1024 * 1024:
                lines.append(text.decode('utf-8', errors='replace'))
        callback = api.llama_log_callback(collect)
        api.llama_log_set(callback, None)
        try:
            yield lines
        finally:
            api.llama_log_set(previous, data)


def _placement(logs, *, context):
    text = ''.join(logs)
    buffers = {name: [float(x) for x in re.findall(
        r'CUDA0\s+' + name + r' buffer size\s*=\s*([0-9.]+) MiB', text)]
        for name in ('model', 'KV', 'compute')}
    if context:
        valid = all(any(v > 0 for v in buffers[name]) for name in ('KV', 'compute'))
        layers = None
    else:
        layers = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
        valid = (len(layers) == 1 and int(layers[0][0]) == int(layers[0][1]) > 0
                 and 'offloading output layer to GPU' in text
                 and any(v > 0 for v in buffers['model']))
    if not valid:
        raise ModelInputError('CUDA GGUF actual layer or buffer placement was not proven')
    return {'logs': text, 'buffer_mib': buffers, 'offloaded_layers': layers,
            'logs_sha256': hashlib.sha256(text.encode()).hexdigest()}


class _CUDAGGUFBackend(_GGUFBackend):
    def __init__(self, path, vocab_size):
        from llama_cpp import llama_cpp as api
        self._cuda_identity = cuda_backend_identity()
        with _capture_logs(api) as logs:
            super().__init__(path, vocab_size)
        try:
            self.placement_evidence = _placement(logs, context=False)
        except BaseException:
            self.close()
            raise

    def _configure_model_params(self, params, api):
        params.n_gpu_layers = -1
        params.split_mode = api.LLAMA_SPLIT_MODE_NONE
        params.main_gpu = 0

    def validate(self):
        super().validate()
        if cuda_backend_identity() != self._cuda_identity:
            raise ModelInputError('CUDA device or native runtime changed after admission')

    def request(self, input_ids, output_reserved_tokens):
        self.validate()
        return _CUDAGGUFRequest(self._model, self._api, input_ids, output_reserved_tokens, self._vocab_size)


class _CUDAGGUFRequest(_GGUFRequest):
    def __init__(self, model, api, input_ids, output_reserved_tokens, vocab_size):
        with _capture_logs(api) as logs:
            super().__init__(model, api, input_ids, output_reserved_tokens, vocab_size)
        try:
            self.placement_evidence = _placement(logs, context=True)
        except BaseException:
            self.close()
            raise

    def _configure_context_params(self, params):
        params.offload_kqv = True
        params.op_offload = True
