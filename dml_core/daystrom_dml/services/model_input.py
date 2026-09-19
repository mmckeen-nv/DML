"""One local GPT-2 consumer of authenticated, fully counted model inputs.

This companion does not change memory-provider APIs or qualify model quality.
Heavy optional dependencies are imported only when a consumer is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import importlib.metadata
import json
from pathlib import Path
import secrets
from threading import RLock
from typing import Any

from ..contracts.model_input import (
    CompiledModelInput, ModelInputBudgetError, ModelInputError, ModelInputRequest,
    SUPPORTED_CHAT_TEMPLATE,
)


class ModelInputExecutionError(RuntimeError):
    """Local inference failed or returned an invalid bounded continuation."""


@dataclass(frozen=True, slots=True)
class ModelInputResult:
    artifact_digest: str
    input_token_count: int
    output_token_count: int
    output_ids: tuple[int, ...]
    text: str


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "safetensors", "jinja2")}


class LocalTransformersInputConsumer:
    """Own a verified local model and compile/execute only its exact-input artifacts."""

    def __init__(self, snapshot_directory: str | Path):
        from .model_input_snapshot import SnapshotVerificationError, verify_local_snapshot

        if not isinstance(snapshot_directory, (str, Path)):
            raise ModelInputError("A local snapshot directory is required")
        self._lock = RLock()
        self._closed = True
        try:
            self._snapshot_context = verify_local_snapshot(snapshot_directory)
        except SnapshotVerificationError as exc:
            raise ModelInputError("Local model snapshot admission failed") from exc
        try:
            self._snapshot = self._snapshot_context.__enter__()
            import torch
            from safetensors.torch import load_file
            from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

            config_data = self._snapshot.config
            if (config_data.get("model_type") != "gpt2"
                    or config_data.get("is_encoder_decoder", False) is not False):
                raise ModelInputError("Only the built-in decoder-only GPT-2 architecture is supported")
            self._torch = torch
            self._identity = self._snapshot.identity
            self._template = self._snapshot.chat_template
            if self._template != SUPPORTED_CHAT_TEMPLATE:
                raise ModelInputError("Unsupported exact-input chat template")
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self._snapshot.path / "tokenizer.json"),
                **self._snapshot.special_tokens,
            )
            self._tokenizer.chat_template = self._template
            self._tokenizer.model_max_length = self._identity.model_window_tokens
            self._tokenizer.backend_tokenizer.no_truncation()
            self._tokenizer.backend_tokenizer.no_padding()
            config = GPT2Config.from_dict(config_data)
            if (type(config.n_positions) is not int
                    or self._identity.model_window_tokens > config.n_positions
                    or len(self._tokenizer) != config.vocab_size):
                raise ModelInputError("Tokenizer or model window does not match the verified model")
            special_ids = [self._tokenizer.convert_tokens_to_ids(value)
                           for value in self._snapshot.special_tokens.values() if value is not None]
            if any(type(value) is not int or not 0 <= value < config.vocab_size for value in special_ids):
                raise ModelInputError("Special token IDs do not match the verified vocabulary")
            neutral_generation = self._generation_config(1)
            for name, value in neutral_generation.to_dict().items():
                if name in config_data and not name.startswith("_") and name != "transformers_version":
                    setattr(config, name, value)
            self._snapshot.validate_integrity()
            # Force CPU construction even when the process has another default device.
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                self._model = GPT2LMHeadModel(config).to(device="cpu", dtype=torch.float32)
            state = load_file(str(self._snapshot.path / "model.safetensors"), device="cpu")
            if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                   for value in state.values()):
                raise ModelInputError("Model weights must be finite CPU float32 tensors")
            self._model.load_state_dict(state, strict=True)
            loaded = self._model.state_dict()
            if any(not torch.equal(loaded[name], value) for name, value in state.items()):
                raise ModelInputError("Model weight aliases do not match the verified snapshot")
            self._model.eval()
            self._model.requires_grad_(False)
            # Transformers otherwise inherits model-specific defaults even with an
            # explicit GenerationConfig. Replace that fallback with our owned policy.
            self._model.generation_config = self._generation_config(1)
            self._runtime_versions = _versions()
            if self._runtime_versions != dict(self._snapshot.manifest.runtime_versions):
                raise ModelInputError("Model runtime versions changed during initialization")
            self._runtime_digest = self._fingerprint()
            self._consumer_id = secrets.token_hex(24)
            self._auth_key = secrets.token_bytes(32)
            self._closed = False
        except Exception as exc:
            self._snapshot_context.__exit__(None, None, None)
            if isinstance(exc, ModelInputError):
                raise
            raise ModelInputError("Verified local model could not be loaded") from exc

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def close(self) -> None:
        """Release the private model and verified snapshot; closing twice is safe."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._auth_key = b""
            self._model = None
            self._tokenizer = None
            self._snapshot_context.__exit__(None, None, None)

    def _require_open(self) -> None:
        if self._closed:
            raise ModelInputError("Exact-input consumer is closed")

    def _generation_config(self, output_tokens: int):
        from transformers import GenerationConfig

        return GenerationConfig(
            max_new_tokens=output_tokens, do_sample=False, num_beams=1,
            num_beam_groups=1, num_return_sequences=1, use_cache=False,
            min_length=0, min_new_tokens=0, max_time=None,
            token_healing=False, forced_bos_token_id=None, forced_eos_token_id=None,
            bad_words_ids=None, force_words_ids=None, constraints=None,
            suppress_tokens=None, begin_suppress_tokens=None, sequence_bias=None,
            stop_strings=None, repetition_penalty=1.0, no_repeat_ngram_size=0,
            return_dict_in_generate=False, output_scores=False, output_logits=False,
            output_attentions=False, output_hidden_states=False,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
            decoder_start_token_id=None, cache_implementation=None, disable_compile=True,
        )

    def _fingerprint(self) -> str:
        """Check live model/tokenizer state, including mutations outside normal setters."""
        digest = hashlib.sha256()
        tokenizer = self._tokenizer
        model = self._model
        metadata = {
            "identity": self._identity.to_payload(), "template": self._template,
            "tokenizer_backend": tokenizer.backend_tokenizer.to_str(),
            "special_tokens": tokenizer.special_tokens_map,
            "tokenizer_template": tokenizer.chat_template,
            "tokenizer_window": tokenizer.model_max_length,
            "tokenizer_padding": tokenizer.padding_side,
            "tokenizer_truncation": tokenizer.truncation_side,
            "model_config": model.config.to_dict(),
            "generation_config": model.generation_config.to_dict(),
            "training": model.training,
            "versions": _versions(),
        }
        digest.update(_json_bytes(metadata))
        for group, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
            for name, value in values:
                if value.device.type != "cpu":
                    raise ModelInputError("Exact-input model moved outside the CPU runtime")
                digest.update(_json_bytes([group, name, str(value.dtype), list(value.shape), value.requires_grad]))
                digest.update(value.detach().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _validate_runtime(self) -> None:
        self._require_open()
        try:
            if self._runtime_versions != _versions() or self._fingerprint() != self._runtime_digest:
                raise ModelInputError("Exact-input model, tokenizer, template or runtime identity changed")
        except ModelInputError:
            raise
        except Exception as exc:
            raise ModelInputError("Exact-input runtime identity cannot be verified") from exc

    def _auth_tag(self, artifact: CompiledModelInput) -> str:
        return hmac.new(self._auth_key, _json_bytes(artifact.signing_payload()), hashlib.sha256).hexdigest()

    def compile(self, messages, tools=None, *, output_reserved_tokens: int) -> CompiledModelInput:
        """Render all supported fields once and freeze the real tokenizer's exact IDs."""
        request = ModelInputRequest.from_payload({
            "messages": messages, "tools": [] if tools is None else tools,
            "output_reserved_tokens": output_reserved_tokens,
        })
        with self._lock:
            self._validate_runtime()
            if output_reserved_tokens >= self._identity.model_window_tokens:
                raise ModelInputBudgetError("Output reservation leaves no room for model input")
            try:
                encoded = self._tokenizer.apply_chat_template(
                    request.messages, tools=request.tools, chat_template=self._template,
                    tokenize=True, add_generation_prompt=False, continue_final_message=False,
                    truncation=False, padding=False, return_dict=True,
                    tokenizer_kwargs={"return_attention_mask": True},
                )
                input_ids = tuple(encoded["input_ids"])
                attention_mask = tuple(encoded["attention_mask"])
            except Exception as exc:
                raise ModelInputError("Exact-input template or tokenization failed") from exc
            if any(type(token) is not int or token < 0 or token >= self._model.config.vocab_size for token in input_ids):
                raise ModelInputError("Tokenizer returned IDs outside the model vocabulary")
            if attention_mask != (1,) * len(input_ids):
                raise ModelInputError("Exact-input compilation does not admit padding or masked input")
            artifact = CompiledModelInput(
                identity=self._identity, request_digest=request.request_digest,
                input_ids=input_ids, attention_mask=attention_mask,
                output_reserved_tokens=request.output_reserved_tokens,
                model_window_tokens=self._identity.model_window_tokens,
                consumer_id=self._consumer_id, nonce=secrets.token_hex(24), auth_tag="0" * 64,
            )
            # Tokenization is permitted to touch no admitted runtime identity.
            self._validate_runtime()
            return replace(artifact, auth_tag=self._auth_tag(artifact))

    def execute(self, artifact: CompiledModelInput) -> ModelInputResult:
        """Dispatch authenticated IDs directly; this method never renders or tokenizes."""
        with self._lock:
            self._require_open()
            if type(artifact) is not CompiledModelInput:
                raise ModelInputError("An exact compiled model input is required")
            artifact.validate()
            if (artifact.consumer_id != self._consumer_id or artifact.identity != self._identity
                    or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))):
                raise ModelInputError("Compiled model input is not authenticated for this consumer")
            self._validate_runtime()
            torch = self._torch
            inputs = torch.tensor([artifact.input_ids], dtype=torch.long, device="cpu")
            mask = torch.tensor([artifact.attention_mask], dtype=torch.long, device="cpu")
            try:
                with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                    output = self._model.generate(
                        input_ids=inputs, attention_mask=mask,
                        generation_config=self._generation_config(artifact.output_reserved_tokens),
                        use_model_defaults=False,
                    )
                if (not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1
                        or output.dtype != torch.long or output.device.type != "cpu"):
                    raise ValueError("Invalid generated tensor")
                complete_ids = tuple(output[0].tolist())
                prefix_length = len(artifact.input_ids)
                if (complete_ids[:prefix_length] != artifact.input_ids
                        or not prefix_length <= len(complete_ids) <= prefix_length + artifact.output_reserved_tokens):
                    raise ValueError("Generated sequence changed its input prefix or exceeded its reservation")
                output_ids = complete_ids[prefix_length:]
                if any(token < 0 or token >= self._model.config.vocab_size for token in output_ids):
                    raise ValueError("Generated IDs are outside the vocabulary")
                text = self._tokenizer.decode(output_ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
            except Exception as exc:
                raise ModelInputExecutionError("Local model execution failed or returned an invalid continuation") from exc
            return ModelInputResult(artifact_digest=artifact.artifact_digest,
                                    input_token_count=len(artifact.input_ids),
                                    output_token_count=len(output_ids), output_ids=output_ids, text=text)
