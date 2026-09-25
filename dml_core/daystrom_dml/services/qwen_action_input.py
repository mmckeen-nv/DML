"""Explicit Qwen action-JSON profile; generic model consumers stay unchanged."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac

from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputRequest
from ..contracts.agent_episode import (
    VALIDATION_CONSUMER_PROFILE, RECOVERY_CONSUMER_PROFILE, VALIDATION_PROFILES,
    execution_policy_identity, recovery_guidance_identity, initial_messages,
)
from .agent_action_grammar import (
    ActionLogitsProcessor, MAX_BOUND_REQUESTS, action_schema, compile_action_grammar, policy_identity,
)
from .model_input import ModelInputExecutionError, ModelInputResult, _json_bytes
from .qwen_model_input import LocalQwenInputConsumer, decode_qwen_output


CONSUMER_PROFILE = "qwen2-action-json-v1"


def constrained_identity(base_identity, *, consumer_profile=CONSUMER_PROFILE):
    """Reconstruct the full profile identity without loading model weights."""
    policy = {"base_runtime_identity": base_identity.runtime_identity, **policy_identity()}
    if consumer_profile not in (CONSUMER_PROFILE, *VALIDATION_PROFILES):
        raise ModelInputError("Unknown constrained action profile")
    version = "v1"
    if consumer_profile == VALIDATION_CONSUMER_PROFILE:
        policy["execution_policy"] = execution_policy_identity()
        version = "v2"
    elif consumer_profile == RECOVERY_CONSUMER_PROFILE:
        policy["recovery_guidance"] = recovery_guidance_identity()
        version = "v3"
    return replace(base_identity, runtime_identity="dml-qwen-action-runtime-" + version + ":" + hashlib.sha256(
        _json_bytes(policy)).hexdigest())


class LocalQwenActionInputConsumer(LocalQwenInputConsumer):
    """Bound at most 64 immutable compiled requests for this consumer lifetime.

    Repeated artifacts and compile-A/compile-B/execute-A are supported. Capacity
    is explicit: no eviction or last-request fallback. Closing releases every
    binding. A new matcher/compiler is created within each execution deadline.
    """

    def __init__(self, snapshot_directory, *, consumer_profile=CONSUMER_PROFILE):
        if consumer_profile not in (CONSUMER_PROFILE, *VALIDATION_PROFILES):
            raise ModelInputError("Unknown constrained action profile")
        self._consumer_profile = consumer_profile
        self._execution_policy = execution_policy_identity() if consumer_profile == VALIDATION_CONSUMER_PROFILE else None
        self._recovery_guidance = recovery_guidance_identity() if consumer_profile == RECOVERY_CONSUMER_PROFILE else None
        self._grammar_policy = policy_identity()
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
        expected = execution_policy_identity() if self._consumer_profile == VALIDATION_CONSUMER_PROFILE else None
        if self._execution_policy != expected:
            raise ModelInputError("Action execution protocol or rejection policy changed")
        guidance = recovery_guidance_identity() if self._consumer_profile == RECOVERY_CONSUMER_PROFILE else None
        if self._recovery_guidance != guidance:
            raise ModelInputError("Action recovery guidance identity changed")
        if self._identity != constrained_identity(self._base_action_identity, consumer_profile=self._consumer_profile):
            raise ModelInputError("Action profile and compiled runtime identity differ")

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload({
            "messages": messages, "tools": [] if tools is None else tools,
            "output_reserved_tokens": output_reserved_tokens,
        })
        if self._consumer_profile == RECOVERY_CONSUMER_PROFILE:
            expected = initial_messages("profile binding", consumer_profile=self._consumer_profile)[0]
            if not request.messages or request.messages[0] != expected:
                raise ModelInputError("Recovery profile requires its exact first system message")
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
                with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
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
