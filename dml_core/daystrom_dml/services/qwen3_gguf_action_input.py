"""Explicit sampled Qwen3 GGUF action profile; legacy profiles stay exact."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import hmac
import platform
import re
from threading import Lock

from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputRequest
from ..contracts.agent_episode import (
    QWEN3_GGUF_CONSUMER_PROFILE, QWEN3_GGUF_ARM64_CONSUMER_PROFILE, QWEN3_GGUF_CONSUMER_PROFILES,
    recovery_guidance_identity, initial_messages,
)
from .agent_action_grammar import (
    ActionLogitsProcessor, MAX_BOUND_REQUESTS, action_schema, compile_action_grammar, policy_identity,
)
from .model_input import ModelInputExecutionError, ModelInputResult, _json_bytes
from .qwen_model_input import decode_qwen_output
from .qwen3_gguf_model_input import LocalQwen3GGUFInputConsumer, backend_identity


CONSUMER_PROFILE = QWEN3_GGUF_CONSUMER_PROFILE
ARM64_CONSUMER_PROFILE = QWEN3_GGUF_ARM64_CONSUMER_PROFILE
_SAMPLED_CPU_RNG_LOCK = Lock()


def sampling_policy_identity():
    """One declared candidate per request; nonce and task never choose the seed.

    The explicit CPU loop retains the previously reviewed Transformers warper
    order and torch multinomial selection. Neutral settings are declared here;
    no GenerationConfig or native llama.cpp sampler is used. This policy and
    the exact native backend files are bound into the action runtime identity.
    """
    return {"schema_version": "dml-qwen3-gguf-sampling-v1",
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


def _arm64_platform():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ModelInputError("ARM64 Qwen3 GGUF profile requires Linux aarch64")
    return {"system": "Linux", "machine": "aarch64", "device": "cpu"}


def constrained_identity(base_identity, *, consumer_profile):
    """Explicit architecture, unchanged grammar and inherited policy composition."""
    if (consumer_profile not in QWEN3_GGUF_CONSUMER_PROFILES
            or re.fullmatch(r"dml-qwen3-gguf-model-input-runtime-v1:[0-9a-f]{64}", base_identity.runtime_identity) is None):
        raise ModelInputError("Qwen3 GGUF action profile requires its explicit profile and base runtime")
    policy = {"consumer_profile": consumer_profile, "base_runtime_identity": base_identity.runtime_identity,
              **policy_identity(), "inherited_guidance_policy": recovery_guidance_identity()}
    policy["sampling_policy"] = sampling_policy_identity()
    policy["backend_identity"] = backend_identity()
    prefix = "dml-qwen3-gguf-action-runtime-v1:"
    if consumer_profile == ARM64_CONSUMER_PROFILE:
        policy["host_architecture"] = _arm64_platform()
        prefix = "dml-qwen3-gguf-arm64-action-runtime-v1:"
    return replace(base_identity, runtime_identity=prefix + hashlib.sha256(
        _json_bytes(policy)).hexdigest())


class LocalQwen3GGUFActionInputConsumer(LocalQwen3GGUFInputConsumer):
    """Bound at most 64 immutable compiled requests for this consumer lifetime.

    Repeated artifacts and compile-A/compile-B/execute-A are supported. Capacity
    is explicit: no eviction or last-request fallback. Closing releases every
    binding. A new matcher/compiler is created within each execution deadline.
    """

    def __init__(self, snapshot_directory, *, consumer_profile):
        if consumer_profile not in QWEN3_GGUF_CONSUMER_PROFILES:
            raise ModelInputError("Unknown constrained action profile")
        if consumer_profile == ARM64_CONSUMER_PROFILE:
            _arm64_platform()
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

    def _validate_runtime(self):
        super()._validate_runtime()
        if self._grammar_policy != policy_identity():
            raise ModelInputError("Action grammar runtime or policy identity changed")
        guidance = recovery_guidance_identity()
        if self._recovery_guidance != guidance:
            raise ModelInputError("Action recovery guidance identity changed")
        if self._sampling_policy != sampling_policy_identity():
            raise ModelInputError("Qwen3 GGUF sampling policy identity changed")
        if self._identity != constrained_identity(self._base_action_identity, consumer_profile=self._consumer_profile):
            raise ModelInputError("Action profile and compiled runtime identity differ")

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload({
            "messages": messages, "tools": [] if tools is None else tools,
            "output_reserved_tokens": output_reserved_tokens,
        })
        expected = initial_messages("profile binding", consumer_profile=self._consumer_profile)[0]
        if not request.messages or request.messages[0] != expected:
            raise ModelInputError("Qwen3 GGUF recovery profile requires its exact first system message")
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
            torch = self._torch
            try:
                grammar = compile_action_grammar(self._tokenizer, self._vocab_size, request.tools)
                processor = ActionLogitsProcessor(grammar, self._tokenizer, artifact.input_ids,
                                                  artifact.output_reserved_tokens)
                complete_ids = artifact.input_ids
                prefix_length = len(complete_ids)
                with (_sampled_cpu_rng(torch, self._sampling_policy["seed"]), torch.inference_mode(),
                      torch.autocast(device_type="cpu", enabled=False),
                      self._backend.request(artifact.input_ids, artifact.output_reserved_tokens) as context):
                    for index in range(artifact.output_reserved_tokens):
                        scores = torch.from_numpy(context.logits()).unsqueeze(0)
                        inputs = torch.tensor([complete_ids], dtype=torch.long, device="cpu")
                        scores = processor(inputs, scores)
                        token = _sample_next(torch, inputs, scores)
                        complete_ids += (token,)
                        if token == self._tokenizer.eos_token_id:
                            break
                        if index + 1 < artifact.output_reserved_tokens:
                            context.advance(token)
                processor.finish(complete_ids)
                output_ids = complete_ids[prefix_length:]
                text = decode_qwen_output(self._tokenizer, output_ids)
                self._validate_runtime()
            except Exception as exc:
                raise ModelInputExecutionError("Constrained GGUF execution failed or returned invalid token evidence") from exc
            return ModelInputResult(
                artifact_digest=artifact.artifact_digest, input_token_count=prefix_length,
                output_token_count=len(output_ids), output_ids=output_ids, text=text,
            )

    def close(self):
        with self._lock:
            self._bound_requests.clear()
            super().close()


def _sample_next(torch, inputs, scores):
    """Use the same Transformers CPU warpers and torch multinomial as v1."""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    )

    # min_p=0.0 is disabled in GenerationMixin, not an extra processor.
    for warper in (TemperatureLogitsWarper(0.7), TopKLogitsWarper(20), TopPLogitsWarper(0.8)):
        scores = warper(inputs, scores)
    probabilities = torch.nn.functional.softmax(scores, dim=-1)
    if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities.sum(-1) > 0).all()):
        raise ModelInputExecutionError("GGUF grammar/sampling has no finite probability mass")
    return int(torch.multinomial(probabilities, num_samples=1).item())
