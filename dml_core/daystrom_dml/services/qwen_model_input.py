"""One explicit Qwen2 CPU/F32 exact-input companion, separate from GPT-2."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import secrets
from threading import RLock

from ..contracts.model_input import ModelInputError
from .model_input import LocalTransformersInputConsumer, ModelInputExecutionError, _versions
from .qwen_model_snapshot import QWEN_CHAT_TEMPLATE, SPECIAL_TOKENS, expected_shapes, verify_qwen_snapshot


def decode_qwen_output(tokenizer, output_ids: tuple[int, ...]) -> str:
    """Decode exact IDs, omitting ONLY one verified final end-of-message token.

    Every generated ID remains in the result and its billable output count.
    Other control tokens, including non-terminal EOS, remain literal text.
    """
    vocabulary = tokenizer.get_vocab()
    allowed = frozenset(vocabulary.values())
    if any(type(token) is not int or token not in allowed for token in output_ids):
        raise ModelInputExecutionError("Qwen output contains an unused or invalid vocabulary ID")
    eos = vocabulary.get("<|im_end|>")
    if (type(eos) is not int or tokenizer.eos_token != "<|im_end|>"
            or tokenizer.eos_token_id != eos or eos not in tokenizer.all_special_ids):
        raise ModelInputExecutionError("Qwen terminal EOS identity differs")
    text_ids = output_ids[:-1] if output_ids and output_ids[-1] == eos else output_ids
    return tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class LocalQwenInputConsumer(LocalTransformersInputConsumer):
    """Reuse authenticated compile/dispatch while admitting only Qwen v1 data."""

    def _generation_config(self, output_tokens: int):
        config = super()._generation_config(output_tokens)
        # Transformers creates this DynamicCache inside each generate call.
        # No cache is accepted from callers, returned, or retained on the model.
        config.use_cache = True
        config.cache_implementation = "dynamic"
        return config

    def __init__(self, snapshot_directory: str | Path):
        self._lock = RLock()
        self._closed = True
        try:
            self._snapshot_context = verify_qwen_snapshot(snapshot_directory)
        except Exception as exc:
            raise ModelInputError("Local Qwen snapshot admission failed") from exc
        try:
            self._snapshot = self._snapshot_context.__enter__()
            import torch
            from safetensors.torch import load_file
            from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

            self._torch = torch
            self._identity = self._snapshot.identity
            self._template = self._snapshot.chat_template
            if self._template != QWEN_CHAT_TEMPLATE:
                raise ModelInputError("Unsupported Qwen exact-input template")
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self._snapshot.path / "tokenizer.json"), **SPECIAL_TOKENS)
            self._tokenizer.chat_template = self._template
            self._tokenizer.model_max_length = self._identity.model_window_tokens
            self._tokenizer.backend_tokenizer.no_truncation()
            self._tokenizer.backend_tokenizer.no_padding()
            config_data = self._snapshot.config
            vocabulary = self._tokenizer.get_vocab()
            self._allowed_ids = frozenset(vocabulary.values())
            if (len(self._allowed_ids) != len(vocabulary)
                    or any(type(token) is not int or not 0 <= token < config_data["vocab_size"]
                           for token in self._allowed_ids)):
                raise ModelInputError("Qwen tokenizer vocabulary does not match model rows")
            for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                token = vocabulary.get(name)
                marker = self._tokenizer.added_tokens_decoder.get(token)
                if (type(token) is not int or marker is None or marker.special is not True
                        or self._tokenizer.encode(name, add_special_tokens=False) != [token]):
                    raise ModelInputError("Qwen framing marker is not one admitted special token")
            if (config_data["eos_token_id"] != self._tokenizer.eos_token_id
                    or config_data["bos_token_id"] != self._tokenizer.pad_token_id):
                raise ModelInputError("Qwen configured special tokens differ from the tokenizer")
            config = Qwen2Config.from_dict(config_data)
            config._attn_implementation = "eager"
            self._snapshot.validate_integrity()
            state = load_file(str(self._snapshot.path / "model.safetensors"), device="cpu")
            shapes = expected_shapes(config_data)
            if state.keys() != shapes.keys():
                raise ModelInputError("Qwen learned tensor inventory differs")
            if any(tuple(state[name].shape) != shape or state[name].dtype != torch.float32
                   or not bool(torch.isfinite(state[name]).all()) for name, shape in shapes.items()):
                raise ModelInputError("Qwen tensors must have exact shapes and finite float32 values")
            # Build no real random weights. Assign verified tensors, then initialize
            # only the deterministic, non-learned rotary buffers on CPU.
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                self._model = Qwen2ForCausalLM(config)
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
            self._model.load_state_dict(state, strict=True, assign=True)
            self._model.model.rotary_emb = Qwen2RotaryEmbedding(config, device=torch.device("cpu"))
            self._model.tie_weights()
            loaded = self._model.state_dict()
            if (loaded.keys() != state.keys()
                    or any(not torch.equal(loaded[name], tensor) for name, tensor in state.items())
                    or self._model.lm_head.weight.data_ptr() != self._model.model.embed_tokens.weight.data_ptr()):
                raise ModelInputError("Qwen loaded weights or tied embedding alias differ")
            if any(value.device.type != "cpu" or value.dtype != torch.float32
                   for value in (*self._model.parameters(), *self._model.buffers())):
                raise ModelInputError("Qwen runtime is outside finite CPU float32")
            self._model.eval()
            self._model.requires_grad_(False)
            self._model.generation_config = self._generation_config(1)
            self._runtime_versions = _versions()
            if self._runtime_versions != dict(self._snapshot.manifest.runtime_versions):
                raise ModelInputError("Qwen runtime versions changed during initialization")
            self._runtime_digest = self._fingerprint()
            self._consumer_id = secrets.token_hex(24)
            self._auth_key = secrets.token_bytes(32)
            self._closed = False
        except Exception as exc:
            self._snapshot_context.__exit__(None, None, None)
            if isinstance(exc, ModelInputError):
                raise
            raise ModelInputError("Verified Qwen model could not be loaded") from exc

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        with self._lock:
            artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
            if any(token not in self._allowed_ids for token in artifact.input_ids):
                raise ModelInputError("Qwen input contains unused vocabulary rows")
            return artifact

    def execute(self, artifact):
        with self._lock:
            if getattr(self._model, "_cache", None) is not None:
                raise ModelInputError("Qwen cross-call cache state is excluded")
            result = super().execute(artifact)
            if getattr(self._model, "_cache", None) is not None:
                raise ModelInputExecutionError("Qwen generation retained forbidden cache state")
            text = decode_qwen_output(self._tokenizer, result.output_ids)
            self._validate_runtime()
            return replace(result, text=text)
