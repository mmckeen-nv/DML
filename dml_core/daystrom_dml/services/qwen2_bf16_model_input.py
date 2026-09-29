"""Explicit sharded Qwen2 CPU/BF16 companion with unchanged legacy consumers."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import secrets
from threading import RLock

from ..contracts.model_input import ModelInputError
from .model_input import LocalTransformersInputConsumer, ModelInputExecutionError, _json_bytes, _versions
from .qwen_model_input import decode_qwen_output
from .qwen2_bf16_model_snapshot import (
    QWEN2_BF16_CHAT_TEMPLATE, SHARD_FILES, SPECIAL_TOKENS, expected_shapes, verify_qwen2_bf16_snapshot,
)

_TENSOR_CHUNK_ELEMENTS = 1024 * 1024


def _raw_equal(left, right, torch):
    """Compare all supplied bits, including signed zero, with bounded views."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    a, b = left.detach().reshape(-1).view(torch.uint8), right.detach().reshape(-1).view(torch.uint8)
    return all(torch.equal(a[start:start + _TENSOR_CHUNK_ELEMENTS], b[start:start + _TENSOR_CHUNK_ELEMENTS])
               for start in range(0, a.numel(), _TENSOR_CHUNK_ELEMENTS))


class LocalQwen2BF16InputConsumer(LocalTransformersInputConsumer):
    """Strictly assign verified learned BF16 tensors and deterministic F32 RoPE."""

    def _generation_config(self, output_tokens):
        config = super()._generation_config(output_tokens)
        config.use_cache = True
        config.cache_implementation = "dynamic"
        return config

    def __init__(self, snapshot_directory: str | Path):
        self._lock = RLock()
        self._closed = True
        try:
            self._snapshot_context = verify_qwen2_bf16_snapshot(snapshot_directory)
        except Exception as exc:
            raise ModelInputError("Local Qwen2BF16 snapshot admission failed") from exc
        try:
            self._snapshot = self._snapshot_context.__enter__()
            import torch
            from safetensors.torch import load_file
            from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding
            from .model_input_snapshot import _json_object

            self._torch = torch
            self._identity = self._snapshot.identity
            self._template = self._snapshot.chat_template
            if self._template != QWEN2_BF16_CHAT_TEMPLATE:
                raise ModelInputError("Unsupported Qwen2BF16 non-thinking template")
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
                raise ModelInputError("Qwen2BF16 tokenizer vocabulary does not match model rows")
            for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
                token = vocabulary.get(name)
                marker = self._tokenizer.added_tokens_decoder.get(token)
                if (type(token) is not int or marker is None
                        or marker.special is not True
                        or self._tokenizer.encode(name, add_special_tokens=False) != [token]):
                    raise ModelInputError("Qwen2BF16 framing marker is not one admitted token")
            if (config_data["eos_token_id"] != self._tokenizer.eos_token_id
                    or config_data["bos_token_id"] != self._tokenizer.pad_token_id):
                raise ModelInputError("Qwen2BF16 configured special tokens differ")
            config = Qwen2Config.from_dict(config_data)
            config._attn_implementation = "eager"
            self._snapshot.validate_integrity()
            index = _json_object((self._snapshot.path / "model.safetensors.index.json").read_bytes(), "Qwen2BF16 index")
            state: dict[str, torch.Tensor] = {}
            for shard in SHARD_FILES:
                values = load_file(str(self._snapshot.path / shard), device="cpu")
                if state.keys() & values.keys() or any(index["weight_map"].get(name) != shard for name in values):
                    raise ModelInputError("Qwen2BF16 tensor shard ownership differs")
                state.update(values)
            shapes = expected_shapes(config_data)
            if state.keys() != shapes.keys() or state.keys() != index["weight_map"].keys():
                raise ModelInputError("Qwen2BF16 learned tensor inventory differs")
            for name, shape in shapes.items():
                tensor = state[name]
                if (tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                        or tensor.layout != torch.strided or not tensor.is_contiguous()):
                    raise ModelInputError("Qwen2BF16 learned tensors require exact contiguous CPU BF16 shapes")
                flat = tensor.reshape(-1)
                if any(not bool(torch.isfinite(flat[start:start + _TENSOR_CHUNK_ELEMENTS]).all())
                       for start in range(0, flat.numel(), _TENSOR_CHUNK_ELEMENTS)):
                    raise ModelInputError("Qwen2BF16 learned tensors must be finite")
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                self._model = Qwen2ForCausalLM(config)
            # The source has no head payload: strict loading adds only its
            # declared alias to the one verified embedding, never new values.
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
            self._model.load_state_dict(state, strict=True, assign=True)
            self._model.model.rotary_emb = Qwen2RotaryEmbedding(config, device=torch.device("cpu"))
            self._model.tie_weights()
            loaded = self._model.state_dict()
            if loaded.keys() != state.keys() or any(not _raw_equal(loaded[name], value, torch) for name, value in state.items()):
                raise ModelInputError("Qwen2BF16 assigned learned values differ from verified sources")
            self._expected_shapes = shapes
            self._admitted_config = config
            self._norm_class = Qwen2RMSNorm
            self._rotary_class = Qwen2RotaryEmbedding
            self._rotary_init = self._model.model.rotary_emb.rope_init_fn
            self._model.eval()
            self._model.requires_grad_(False)
            self._model.generation_config = self._generation_config(1)
            self._runtime_versions = _versions()
            if self._runtime_versions != dict(self._snapshot.manifest.runtime_versions):
                raise ModelInputError("Qwen2BF16 runtime versions changed during initialization")
            self._runtime_digest = self._fingerprint()
            self._consumer_id = secrets.token_hex(24)
            self._auth_key = secrets.token_bytes(32)
            self._closed = False
        except Exception as exc:
            self._snapshot_context.__exit__(None, None, None)
            if isinstance(exc, ModelInputError):
                raise
            raise ModelInputError("Verified Qwen2BF16 model could not be loaded") from exc

    def _forward_metadata(self):
        """Bind config-derived cached values used by the admitted eager forward path."""
        torch, model, config = self._torch, self._model, self._admitted_config
        decoder = model.model
        if (model.config is not config or decoder.config is not config or config._attn_implementation != "eager"
                or decoder.has_sliding_layers is not False
                or config.use_sliding_window is not False or config.sliding_window is not None
                or config.max_window_layers != self._snapshot.config["max_window_layers"]
                or config.layer_types != ["full_attention"] * config.num_hidden_layers
                or len(decoder.layers) != config.num_hidden_layers):
            raise ModelInputError("Qwen2BF16 forward metadata differs from admitted configuration")
        norms = [decoder.norm]
        layers = []
        for index, layer in enumerate(decoder.layers):
            attention, mlp = layer.self_attn, layer.mlp
            if (attention.config is not config or mlp.config is not config
                    or attention.layer_idx != index or layer.attention_type != "full_attention"
                    or attention.head_dim != config.hidden_size // config.num_attention_heads
                    or attention.num_key_value_groups != config.num_attention_heads // config.num_key_value_heads
                    or attention.scaling != (config.hidden_size // config.num_attention_heads)**-0.5
                    or attention.attention_dropout != config.attention_dropout
                    or attention.is_causal is not True or attention.sliding_window is not None
                    or attention.training is not False
                    or any(projection.bias is None for projection in (
                        attention.q_proj, attention.k_proj, attention.v_proj))
                    or attention.o_proj.bias is not None
                    or type(mlp.act_fn) is not torch.nn.SiLU or mlp.act_fn.inplace is not False):
                raise ModelInputError("Qwen2BF16 forward metadata differs from admitted configuration")
            norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
            layers.append({"index": attention.layer_idx, "attention_type": layer.attention_type,
                           "head_dim": attention.head_dim, "kv_groups": attention.num_key_value_groups,
                           "scaling": attention.scaling, "dropout": attention.attention_dropout,
                           "causal": attention.is_causal, "sliding_window": attention.sliding_window,
                           "training": attention.training, "activation": "torch.nn.SiLU", "inplace": mlp.act_fn.inplace})
        if any(type(norm) is not self._norm_class or norm.variance_epsilon != config.rms_norm_eps for norm in norms):
            raise ModelInputError("Qwen2BF16 forward metadata differs from admitted configuration")
        return {"config_aliases": "admitted_root_decoder_attention_mlp", "attention_implementation": "eager",
                "has_sliding_layers": False, "inactive_max_window_layers": config.max_window_layers,
                "layers": layers, "norm_epsilons": [norm.variance_epsilon for norm in norms]}

    def _fingerprint(self):
        """Full fresh BF16 bit scans only for this explicitly admitted architecture."""
        torch, model, tokenizer = self._torch, self._model, self._tokenizer
        rotary = model.model.rotary_emb
        if (model.lm_head.weight is not model.model.embed_tokens.weight
                or type(rotary) is not self._rotary_class or rotary.config is not model.config
                or rotary.original_inv_freq is not rotary.inv_freq or rotary.rope_init_fn is not self._rotary_init
                or rotary.rope_type != "default"
                or rotary.max_seq_len_cached != model.config.max_position_embeddings
                or rotary.original_max_seq_len != model.config.max_position_embeddings
                or rotary.attention_scaling != 1.0):
            raise ModelInputError("Qwen2BF16 tied weights or deterministic rotary aliases differ")
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        if parameters.keys() != self._expected_shapes.keys():
            raise ModelInputError("Qwen2BF16 runtime parameter inventory differs")
        if buffers.keys() != {"model.rotary_emb.inv_freq"}:
            raise ModelInputError("Qwen2BF16 runtime buffer inventory differs")
        digest = hashlib.sha256()
        digest.update(_json_bytes({
            "identity": self._identity.to_payload(), "template": self._template,
            "tokenizer_backend": tokenizer.backend_tokenizer.to_str(), "special_tokens": tokenizer.special_tokens_map,
            "tokenizer_template": tokenizer.chat_template, "tokenizer_window": tokenizer.model_max_length,
            "tokenizer_padding": tokenizer.padding_side, "tokenizer_truncation": tokenizer.truncation_side,
            "model_config": model.config.to_dict(), "generation_config": model.generation_config.to_dict(),
            "training": model.training, "versions": _versions(), "forward_metadata": self._forward_metadata(),
            "rotary": {"type": rotary.rope_type, "maximum": rotary.max_seq_len_cached,
                       "original_maximum": rotary.original_max_seq_len, "attention_scaling": rotary.attention_scaling},
        }))
        for group, values in (("parameter", parameters), ("buffer", buffers)):
            for name, value in values.items():
                expected_shape = (self._expected_shapes[name] if group == "parameter"
                                  else (model.config.hidden_size // model.config.num_attention_heads // 2,))
                expected_dtype = torch.bfloat16 if group == "parameter" else torch.float32
                if (value.device.type != "cpu" or value.dtype != expected_dtype or value.layout != torch.strided
                        or not value.is_contiguous() or tuple(value.shape) != expected_shape or value.requires_grad):
                    raise ModelInputError("Qwen2BF16 runtime tensor left its admitted shape, dtype or layout")
                digest.update(_json_bytes([group, name, str(value.dtype), list(value.shape), value.requires_grad]))
                digest.update(memoryview(value.detach().reshape(-1).view(torch.uint8).numpy()))
        return digest.hexdigest()

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        with self._lock:
            artifact = super().compile(messages, tools, output_reserved_tokens=output_reserved_tokens)
            if any(token not in self._allowed_ids for token in artifact.input_ids):
                raise ModelInputError("Qwen2BF16 input contains unused vocabulary rows")
            return artifact

    def execute(self, artifact):
        with self._lock:
            if getattr(self._model, "_cache", None) is not None:
                raise ModelInputError("Qwen2BF16 cross-call cache state is excluded")
            result = super().execute(artifact)
            if getattr(self._model, "_cache", None) is not None:
                raise ModelInputExecutionError("Qwen2BF16 generation retained forbidden cache state")
            self._validate_runtime()
            return replace(result, text=decode_qwen_output(self._tokenizer, result.output_ids))
