"""Explicit sampled Qwen2 BF16 action profile; legacy profiles stay exact."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import hmac
import re
from threading import Lock

from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputRequest
from ..contracts.agent_episode import (
    QWEN2_BF16_SAMPLED_CONSUMER_PROFILE,
    recovery_guidance_identity, initial_messages,
)
from .agent_action_grammar import (
    ActionLogitsProcessor, MAX_BOUND_REQUESTS, action_schema, compile_action_grammar, policy_identity,
)
from .model_input import ModelInputExecutionError, ModelInputResult, _json_bytes
from .qwen_model_input import decode_qwen_output
from .qwen2_bf16_model_input import LocalQwen2BF16InputConsumer


CONSUMER_PROFILE = QWEN2_BF16_SAMPLED_CONSUMER_PROFILE
_SAMPLED_CPU_RNG_LOCK = Lock()


def sampling_policy_identity():
    """One declared candidate per request; nonce and task never choose the seed.

    The base runtime binds inherited generation settings. These exact overrides
    retain the previously reviewed comparison settings, not the Qwen2 vendor's
    repetition penalty or extra EOS choices. The full effective
    GenerationConfig is also covered by the consumer's live fingerprint.
    """
    return {"schema_version": "dml-qwen2-bf16-sampling-v1",
            "generation_overrides": {"do_sample": True, "temperature": 0.7,
                "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                "num_beams": 1, "num_beam_groups": 1, "num_return_sequences": 1,
                "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                "renormalize_logits": False, "repetition_penalty": 1.0},
            "eos_policy": "verified_tokenizer_im_end_only",
            "comparison": "attempt13_sampling_settings_with_explicit_neutral_repetition_penalty",
            "processor_order": ["authenticated_action_grammar", "temperature", "top_k", "top_p", "min_p"],
            "seed": 0, "seed_scope": "reset_cpu_default_generator_each_execute",
            "rng_restore": "torch.random.fork_rng(devices=[])",
            "rng_lock": "process_global_sampled_profile_generation_only",
            "selection": "one_multinomial_candidate_no_retry_or_reranking"}


@contextmanager
def _sampled_cpu_rng(torch, seed):
    """Serialize this profile's CPU sampling and restore on every exit.

    This lock coordinates these consumers, not unrelated external RNG users.
    Seeding only the CPU generator leaves accelerator RNGs untouched.
    """
    with _SAMPLED_CPU_RNG_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


def constrained_identity(base_identity, *, consumer_profile):
    """Explicit architecture, unchanged grammar and inherited policy composition."""
    if (consumer_profile != CONSUMER_PROFILE
            or re.fullmatch(r"dml-qwen2-bf16-model-input-runtime-v1:[0-9a-f]{64}", base_identity.runtime_identity) is None):
        raise ModelInputError("Qwen2 BF16 action profile requires its explicit profile and base runtime")
    policy = {"consumer_profile": consumer_profile, "base_runtime_identity": base_identity.runtime_identity,
              **policy_identity(), "inherited_guidance_policy": recovery_guidance_identity()}
    policy["sampling_policy"] = sampling_policy_identity()
    return replace(base_identity, runtime_identity="dml-qwen2-bf16-action-runtime-v1:" + hashlib.sha256(
        _json_bytes(policy)).hexdigest())


class LocalQwen2BF16ActionInputConsumer(LocalQwen2BF16InputConsumer):
    """Bound at most 64 immutable compiled requests for this consumer lifetime.

    Repeated artifacts and compile-A/compile-B/execute-A are supported. Capacity
    is explicit: no eviction or last-request fallback. Closing releases every
    binding. A new matcher/compiler is created within each execution deadline.
    """

    def __init__(self, snapshot_directory, *, consumer_profile):
        if consumer_profile != CONSUMER_PROFILE:
            raise ModelInputError("Unknown constrained action profile")
        self._consumer_profile = consumer_profile
        self._recovery_guidance = recovery_guidance_identity()
        self._grammar_policy = policy_identity()
        self._sampling_policy = sampling_policy_identity()
        self._bound_requests: dict[str, bytes] = {}
        super().__init__(snapshot_directory)
        try:
            self._base_action_identity = self._identity
            self._identity = constrained_identity(self._identity, consumer_profile=consumer_profile)
            self._runtime_digest = self._fingerprint()
        except BaseException:
            self.close()
            raise

    def _generation_config(self, output_tokens):
        config = super()._generation_config(output_tokens)
        for name, value in sampling_policy_identity()["generation_overrides"].items():
            setattr(config, name, value)
        return config

    def _validate_runtime(self):
        super()._validate_runtime()
        if self._grammar_policy != policy_identity():
            raise ModelInputError("Action grammar runtime or policy identity changed")
        guidance = recovery_guidance_identity()
        if self._recovery_guidance != guidance:
            raise ModelInputError("Action recovery guidance identity changed")
        if self._sampling_policy != sampling_policy_identity():
            raise ModelInputError("Qwen2 BF16 sampling policy identity changed")
        if self._identity != constrained_identity(self._base_action_identity, consumer_profile=self._consumer_profile):
            raise ModelInputError("Action profile and compiled runtime identity differ")

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload({
            "messages": messages, "tools": [] if tools is None else tools,
            "output_reserved_tokens": output_reserved_tokens,
        })
        expected = initial_messages("profile binding", consumer_profile=self._consumer_profile)[0]
        if not request.messages or request.messages[0] != expected:
            raise ModelInputError("Qwen2 BF16 recovery profile requires its exact first system message")
        action_schema(request.tools)
        with self._lock:
            self._require_open()
            if len(self._bound_requests) >= MAX_BOUND_REQUESTS:
                raise ModelInputError("Action consumer compiled-request capacity exhausted")
            artifact = super().compile(request.messages, request.tools,
                                       output_reserved_tokens=request.output_reserved_tokens)
            # These are the unchanged complete request bytes that the episode
            # runner records. No hidden prompt or schema text is inserted.
            self._bound_requests[artifact.artifact_digest] = _json_bytes(request.to_payload())
            return artifact

    def execute(self, artifact):
        with self._lock:
            self._require_open()
            if type(artifact) is not CompiledModelInput:
                raise ModelInputError("An exact compiled model input is required")
            artifact.validate()
            if (artifact.consumer_id != self._consumer_id or artifact.identity != self._identity
                    or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))):
                raise ModelInputError("Compiled action input is not authenticated for this consumer")
            self._validate_runtime()
            raw = self._bound_requests.get(artifact.artifact_digest)
            if type(raw) is not bytes:
                raise ModelInputError("Compiled action input has no bound request")
            request = ModelInputRequest.from_json(raw)
            if request.request_digest != artifact.request_digest:
                raise ModelInputError("Bound action request differs from its authenticated artifact")
            if getattr(self._model, "_cache", None) is not None:
                raise ModelInputError("Qwen cross-call cache state is excluded")
            torch = self._torch
            try:
                grammar = compile_action_grammar(self._tokenizer, self._model.config.vocab_size, request.tools)
                processor = ActionLogitsProcessor(grammar, self._tokenizer, artifact.input_ids,
                                                  artifact.output_reserved_tokens)
                inputs = torch.tensor([artifact.input_ids], dtype=torch.long, device="cpu")
                mask = torch.tensor([artifact.attention_mask], dtype=torch.long, device="cpu")
                rng = _sampled_cpu_rng(torch, self._sampling_policy["seed"])
                with rng, torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                    output = self._model.generate(
                        input_ids=inputs, attention_mask=mask,
                        generation_config=self._generation_config(artifact.output_reserved_tokens),
                        use_model_defaults=False, logits_processor=[processor],
                    )
                if (type(output) is not torch.Tensor or output.ndim != 2 or output.shape[0] != 1
                        or output.dtype != torch.long or output.device.type != "cpu"):
                    raise ValueError("Invalid generated tensor")
                complete_ids = tuple(output[0].tolist())
                prefix_length = len(artifact.input_ids)
                if (complete_ids[:prefix_length] != artifact.input_ids
                        or not prefix_length <= len(complete_ids) <= prefix_length + artifact.output_reserved_tokens):
                    raise ValueError("Generated sequence changed its prefix or exceeded its reservation")
                output_ids = complete_ids[prefix_length:]
                processor.finish(complete_ids)
                if getattr(self._model, "_cache", None) is not None:
                    raise ValueError("Qwen generation retained forbidden cache state")
                text = decode_qwen_output(self._tokenizer, output_ids)
                self._validate_runtime()
            except Exception as exc:
                raise ModelInputExecutionError("Constrained action execution failed or returned invalid token evidence") from exc
            return ModelInputResult(
                artifact_digest=artifact.artifact_digest, input_token_count=prefix_length,
                output_token_count=len(output_ids), output_ids=output_ids, text=text,
            )

    def close(self):
        with self._lock:
            self._bound_requests.clear()
            super().close()
