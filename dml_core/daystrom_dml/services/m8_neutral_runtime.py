"""Neutral M8 live-generation binding; development-only, no M7 relabeling.

Reuses the qualified generation mechanics (adapter loading, xgrammar
settings, sampler, stop/vocabulary controls) of the selected consumer
profile's runtime module (llama3_sft_runtime or qwen3_sft_runtime) without
its hardcoded DML system policy and tool schemas. The neutral M8 protocol
(m8_protocol) owns rendering, token evidence and action parsing; this module
owns grammar compilation and exact-token generation for one arm. The M7
runtime source, identity and evidence are unchanged.

consumer_profile selects the registered runtime/snapshot module pair; the
default is the qualified Llama3 SFTv3 profile and is byte-identical to the
original single-profile behavior. Every profile-specific pin (STOPS, vendor
template digest, tokenizer/control layout, snapshot manifest pins, qualified
hardware) comes from the selected profile's modules.

No model is loaded by importing this module; heavy dependencies stay lazy.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from threading import RLock
from types import ModuleType

from ..contracts.agent_episode import (
    LLAMA3_SFT_V3_CONSUMER_PROFILE,
    QWEN3_SFT_V1_CONSUMER_PROFILE,
    canonical_json,
)
from ..contracts.model_input import ModelInputError
from . import m8_protocol
from . import llama3_sft_runtime
from . import llama3_sft_v3_action_input as snapshot_v3
from . import qwen3_sft_runtime
from . import qwen3_sft_v1_action_input as snapshot_qwen3
from .model_input import ModelInputExecutionError


@contextmanager
def _seeded_cpu_rng(torch, seed):
    """Per-request forked CPU RNG with an explicit declared seed (default 0)."""
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        yield


RUNTIME_PREFIX = "dml-m8-neutral-action-runtime-v1:"
GRAMMAR_IDENTITY = "xgrammar0.1.33 strict any-whitespace; neutral retrieve/final branches"


@dataclass(frozen=True)
class NeutralConsumerProfile:
    """One registered neutral consumer profile: qualified runtime + snapshot module pair.

    snapshot is the manifest-pin module (verify_manifest, identity_for_manifest,
    sha, VERSIONS, MODEL, REVISION, CHECKPOINT, sampling_policy_identity);
    runtime is the generation-mechanics module (STOPS, VOCAB, WINDOW,
    EnabledAdapterRuntime, sample_next, special/control ids). template_kwargs
    are extra apply_chat_template keyword arguments the vendor template
    requires (empty for llama3; thinking disabled at the boundary for qwen3).
    hardware is the qualified (driver, device_name, capability) triple and
    platform_machine the qualified platform.machine() for non-offline admission.
    """

    consumer_profile: str
    snapshot: ModuleType
    runtime: ModuleType
    template_sha256: str | None  # None = m8_protocol.PINNED_TEMPLATE_SHA256, late-bound at call time
    template_kwargs: tuple
    hardware: tuple
    platform_machine: str


DEFAULT_CONSUMER_PROFILE = LLAMA3_SFT_V3_CONSUMER_PROFILE

PROFILES = {
    LLAMA3_SFT_V3_CONSUMER_PROFILE: NeutralConsumerProfile(
        consumer_profile=LLAMA3_SFT_V3_CONSUMER_PROFILE,
        snapshot=snapshot_v3,
        runtime=llama3_sft_runtime,
        template_sha256=None,
        template_kwargs=(),
        hardware=("590.48.01", "NVIDIA GB300", (10, 3)),
        platform_machine="aarch64",
    ),
    QWEN3_SFT_V1_CONSUMER_PROFILE: NeutralConsumerProfile(
        consumer_profile=QWEN3_SFT_V1_CONSUMER_PROFILE,
        snapshot=snapshot_qwen3,
        runtime=qwen3_sft_runtime,
        template_sha256=snapshot_qwen3.VENDOR_TEMPLATE_DIGEST,
        template_kwargs=(("enable_thinking", False),),
        hardware=("595.99.02", "NVIDIA RTX PRO 5000 72GB Blackwell", (12, 0)),
        platform_machine="x86_64",
    ),
}


def _profile(consumer_profile):
    if consumer_profile not in PROFILES:
        raise ModelInputError("Unknown M8 neutral consumer profile")
    return PROFILES[consumer_profile]


def _control_ids(runtime_module, tokenizer):
    """Output-control rows for the profile: qwen3 masks every added row
    (including non-special spellings like <think>); llama3 masks specials."""
    if hasattr(runtime_module, "control_ids"):
        return runtime_module.control_ids(tokenizer)
    return runtime_module.special_ids(tokenizer)


def _arm(arm):
    if type(arm) is not str or arm not in m8_protocol.ARMS:
        raise ValueError("Unknown M8 arm")
    return arm


def neutral_identity(arm, *, base_identity: dict, profile=None) -> dict:
    """Versioned neutral runtime identity; never an M7 runtime identity."""
    _arm(arm)
    selected = PROFILES[DEFAULT_CONSUMER_PROFILE] if profile is None else profile
    policy = {
        "protocol": m8_protocol.policy_identity(arm, template_sha256=selected.template_sha256),
        "base_identity": base_identity,
        "sampling": selected.snapshot.sampling_policy_identity(),
        "grammar": GRAMMAR_IDENTITY,
        "termination": "optional single terminal stop; complete-at-cap accepted by original parser; no repair",
    }
    return {
        "runtime_identity": RUNTIME_PREFIX + hashlib.sha256(canonical_json(policy)).hexdigest(),
        "policy": policy,
    }


def compile_grammar(tokenizer, arm, *, runtime_module=llama3_sft_runtime):
    """Compile the NEUTRAL action schema for one arm with M7's pinned settings."""
    _arm(arm)
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=runtime_module.VOCAB, stop_token_ids=runtime_module.STOPS
    )
    return xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False).compile_json_schema(
        m8_protocol.schema_bytes(arm).decode(),
        strict_mode=True,
        any_whitespace=True,
        max_whitespace_cnt=None,
    )


def project_output(output_ids, tokenizer, arm, *, runtime_module=llama3_sft_runtime):
    """Project exact output IDs through the neutral parser; never repair."""
    _arm(arm)
    raw = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ended = bool(output_ids and output_ids[-1] in runtime_module.STOPS)
    body = output_ids[:-1] if ended else output_ids
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result = dict(
        raw_text=raw,
        text=text,
        finish_reason="eos" if ended else "length",
        stop_token_id=output_ids[-1] if ended else None,
        derived_action=None,
        derived_action_text=None,
        action_error=None,
    )
    try:
        if any(type(t) is not int or not 0 <= t < runtime_module.VOCAB for t in output_ids):
            raise ValueError("Invalid token row")
        if any(t in _control_ids(runtime_module, tokenizer) for t in body):
            raise ValueError("Embedded output control")
        # Neutral parser: rejects unadvertised tools, malformed actions and
        # context_only tool use; no DML tool schema is consulted.
        action = m8_protocol.parse_action(text, arm)
        result.update(derived_action=action, derived_action_text=text)
    except Exception as exc:
        result["action_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def neutral_adapter_runtime_class(runtime_module):
    """Bind the neutral generation loop to one registered profile's runtime module."""
    if runtime_module not in {profile.runtime for profile in PROFILES.values()}:
        raise ValueError("Unknown neutral runtime module")
    stops = runtime_module.STOPS
    vocab = runtime_module.VOCAB

    class BoundNeutralM8AdapterRuntime(runtime_module.EnabledAdapterRuntime):
        """Enabled-adapter generation bound to one neutral arm's grammar/parser.

        The qualified sampler pins seed 0 per request. Development diagnostics
        (e.g. the competence battery) may declare alternative seeds; every result
        records the seed actually used. This never changes a qualified run's seed.
        """

        _runtime_module = runtime_module

        def __init__(self, model_path, checkpoint, tokenizer, identity, validate, *, arm, seed=0):
            _arm(arm)
            if type(seed) is not int or seed < 0:
                raise ValueError("Sampler seed must be a nonnegative integer")
            self._arm = arm
            self._seed = seed
            super().__init__(model_path, checkpoint, tokenizer, identity, validate)

        def execute(self, compiled):
            if not isinstance(compiled, m8_protocol.CompiledM8Request):
                raise ValueError("Neutral generation requires an exact CompiledM8Request")
            if compiled.arm != self._arm:
                raise ValueError("Compiled request arm differs from this runtime")

            import torch
            import xgrammar as xgr

            result: dict = dict(
                arm=self._arm,
                protocol_profile=m8_protocol.PROFILE,
                sampling_seed=self._seed,
                input_ids=list(compiled.input_ids),
                output_ids=[],
                model_identity=self.identity,
                input_token_count=None,
                output_token_count=None,
                usage_unknown=True,
                execution_error=None,
                action_error=None,
            )
            ids = []
            past = None
            try:
                self.validate()
                model = self.load_model()
                grammar = compile_grammar(self.tokenizer, self._arm, runtime_module=runtime_module)
                matcher = xgr.GrammarMatcher(
                    grammar,
                    override_stop_tokens=stops,
                    terminate_without_stop_token=False,
                    max_rollback_tokens=-1,
                )
                mask = xgr.allocate_token_bitmask(1, vocab)
                if hasattr(runtime_module, "allowed_output_ids"):
                    # qwen3: only real tokenizer rows are admitted; reserved
                    # embedding rows and every added control row stay masked.
                    allowed = torch.zeros(vocab, dtype=torch.bool, device="cpu")
                    allowed[runtime_module.allowed_output_ids(self.tokenizer)] = True
                else:
                    allowed = torch.ones(vocab, dtype=torch.bool, device="cpu")
                    allowed[list(runtime_module.special_ids(self.tokenizer) - set(stops))] = False
                all_ids = tuple(compiled.input_ids)
                input_tensor = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
                with (
                    torch.inference_mode(),
                    _seeded_cpu_rng(torch, self._seed),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    for step in range(compiled.output_reserved_tokens):
                        output = model(input_ids=input_tensor, past_key_values=past, use_cache=True, logits_to_keep=1)
                        past = output.past_key_values
                        scores = output.logits[:, -1, :].float().cpu()
                        if tuple(scores.shape) != (1, vocab) or not bool(torch.isfinite(scores).all()):
                            raise ValueError("Invalid logits")
                        matcher.fill_next_token_bitmask(mask)
                        xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=vocab, backend="cpu")
                        scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
                        if not bool(torch.isfinite(scores).any()):
                            raise ValueError("Empty grammar distribution")
                        token = runtime_module.sample_next(torch, torch.tensor([all_ids], dtype=torch.long), scores)
                        if not bool(allowed[token]) or not matcher.accept_token(token):
                            raise ValueError("Sample violates token mask")
                        ids.append(token)
                        all_ids += (token,)
                        if token in stops:
                            break
                        input_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
                self.validate()
                result.update(
                    output_ids=ids,
                    input_token_count=len(compiled.input_ids),
                    output_token_count=len(ids),
                    usage_unknown=False,
                )
                result.update(project_output(ids, self.tokenizer, self._arm, runtime_module=runtime_module))
            except Exception as exc:
                result.update(output_ids=ids, execution_error=type(exc).__name__ + ": " + str(exc))
                try:
                    result["raw_text"] = self.tokenizer.decode(
                        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                except Exception as decode_error:
                    result["raw_text"] = None
                    result["decode_error"] = type(decode_error).__name__ + ": " + str(decode_error)
            finally:
                past = None
            return result

    BoundNeutralM8AdapterRuntime.__name__ = (
        "NeutralM8AdapterRuntime" if runtime_module is llama3_sft_runtime else "Qwen3NeutralM8AdapterRuntime"
    )
    return BoundNeutralM8AdapterRuntime


NeutralM8AdapterRuntime = neutral_adapter_runtime_class(llama3_sft_runtime)
Qwen3NeutralM8AdapterRuntime = neutral_adapter_runtime_class(qwen3_sft_runtime)
_NEUTRAL_RUNTIME_CLASSES = {
    llama3_sft_runtime: NeutralM8AdapterRuntime,
    qwen3_sft_runtime: Qwen3NeutralM8AdapterRuntime,
}


class M8NeutralConsumer:
    """Authenticated-snapshot neutral consumer for ONE arm; development-only.

    Reuses the registered snapshot verification and attestation discipline
    from the selected profile's M7 consumer without its DML policy, tool
    schemas or renderer. consumer_profile selects the registered
    runtime/snapshot module pair (default: the qualified Llama3 SFTv3
    profile, byte-identical to the original single-profile consumer).
    Compilation goes through m8_protocol.compile_request (exact token IDs);
    execution goes through the profile-bound NeutralM8AdapterRuntime class.
    Compiled requests are single-use; failures are retained in
    last_execution, never repaired.
    """

    def __init__(self, snapshot_directory, *, arm, offline=False, seed=0, consumer_profile=DEFAULT_CONSUMER_PROFILE):
        _arm(arm)
        if type(seed) is not int or seed < 0:
            raise ModelInputError("Sampler seed must be a nonnegative integer")
        profile = _profile(consumer_profile)  # fail closed before any filesystem access
        self.consumer_profile = profile.consumer_profile
        self._profile = profile
        snapshot = profile.snapshot
        self._runtime_class = _NEUTRAL_RUNTIME_CLASSES[profile.runtime]
        self._arm = arm
        self._seed = seed
        self._lock = RLock()
        self._offline = offline
        self._closed = False
        self._root, self.manifest, self._files, self._model_path, self._adapter_path = snapshot.verify_manifest(
            snapshot_directory
        )
        self._manifest_digest = snapshot.sha(self._root / snapshot.MANIFEST_NAME)
        self._stats = {str(p): p.stat() for p in self._files.values()}
        base = snapshot.identity_for_manifest(self.manifest)
        self._identity = neutral_identity(
            arm,
            base_identity={
                "model_digest": base.model_digest,
                "tokenizer_digest": base.tokenizer_digest,
                "chat_template_digest": base.chat_template_digest,
                "model_window_tokens": base.model_window_tokens,
                "snapshot_model": snapshot.MODEL,
                "snapshot_revision": snapshot.REVISION,
                "adapter": snapshot.CHECKPOINT,
            },
            profile=profile,
        )
        self.identity = self._identity
        self._consumed: set[str] = set()
        self._runtime = None
        self.last_execution = None
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        )
        template = self._tokenizer.chat_template
        template_sha256 = (
            m8_protocol.PINNED_TEMPLATE_SHA256 if profile.template_sha256 is None else profile.template_sha256
        )
        if hashlib.sha256(template.encode()).hexdigest() != template_sha256:
            raise ModelInputError("Vendor chat template mismatch")
        raw = {
            x["id"]: x["content"]
            for x in json.loads((self._model_path / "tokenizer.json").read_text())["added_tokens"]
            if x["special"]
        }
        configured = {
            int(k): v["content"]
            for k, v in json.loads((self._model_path / "tokenizer_config.json").read_text())[
                "added_tokens_decoder"
            ].items()
            if v["special"]
        }
        loaded = {k: str(v) for k, v in self._tokenizer.added_tokens_decoder.items() if v.special}
        if profile.runtime is llama3_sft_runtime:
            if raw != configured or raw != loaded or set(raw) != set(range(128000, 128256)):
                raise ModelInputError("Control-token metadata mismatch")
            if len(self._tokenizer) != llama3_sft_runtime.VOCAB or set(self._tokenizer.get_vocab().values()) != set(
                range(llama3_sft_runtime.VOCAB)
            ):
                raise ModelInputError("Vocabulary rows differ")
            if any(self._tokenizer.convert_tokens_to_ids(value) != key for key, value in raw.items()):
                raise ModelInputError("Control rows differ")
        else:
            # Qwen3.5 layout: tokenizer.json carries the base special rows and
            # the non-special added spellings; tokenizer_config.json adds the
            # audio/TTS config-only specials. Pins live in the snapshot module.
            raw_tokens = json.loads((self._model_path / "tokenizer.json").read_text())["added_tokens"]
            raw_nonspecial = {x["id"]: x["content"] for x in raw_tokens if not x["special"]}
            configured_tokens = json.loads((self._model_path / "tokenizer_config.json").read_text())[
                "added_tokens_decoder"
            ]
            configured_nonspecial = {int(k): v["content"] for k, v in configured_tokens.items() if not v["special"]}
            loaded_nonspecial = {k: str(v) for k, v in self._tokenizer.added_tokens_decoder.items() if not v.special}
            if (
                raw != snapshot.SPECIAL_TOKENS
                or raw_nonspecial != snapshot.NONSPECIAL_ADDED_TOKENS
                or configured != {**snapshot.SPECIAL_TOKENS, **snapshot.CONFIG_ONLY_SPECIAL_TOKENS}
                or configured_nonspecial != snapshot.NONSPECIAL_ADDED_TOKENS
                or loaded != configured
                or loaded_nonspecial != configured_nonspecial
            ):
                raise ModelInputError("Control-token metadata mismatch")
            if len(self._tokenizer) != profile.runtime.TOKENIZER_ROWS or set(
                self._tokenizer.get_vocab().values()
            ) != set(range(profile.runtime.TOKENIZER_ROWS)):
                raise ModelInputError("Vocabulary rows differ")
            if any(self._tokenizer.convert_tokens_to_ids(value) != key for key, value in loaded.items()):
                raise ModelInputError("Control rows differ")
            if any(self._tokenizer.convert_tokens_to_ids(value) != key for key, value in loaded_nonspecial.items()):
                raise ModelInputError("Control rows differ")
        self._tokenizer.backend_tokenizer.no_truncation()
        self._tokenizer.backend_tokenizer.no_padding()
        self._tokenizer_digest = hashlib.sha256(self._tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        self._validate_runtime()
        if not self._offline:
            import torch

            observed_driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip()
            driver, device_name, capability = profile.hardware
            if (
                observed_driver != driver
                or torch.cuda.get_device_name(0) != device_name
                or torch.cuda.get_device_capability(0) != capability
            ):
                raise ModelInputError("Qualified CUDA hardware/driver differs")

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True
            self._consumed.clear()
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None

    def _require_open(self):
        if self._closed:
            raise ModelInputError("Consumer closed")

    def _validate_runtime(self):
        self._require_open()
        snapshot = self._profile.snapshot
        if snapshot.sha(self._root / snapshot.MANIFEST_NAME) != self._manifest_digest:
            raise ModelInputError("Manifest drift")
        for name, old in self._stats.items():
            now = Path(name).stat()
            if (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) != (
                now.st_dev,
                now.st_ino,
                now.st_size,
                now.st_mtime_ns,
            ):
                raise ModelInputError("Snapshot changed after admission")
        if hashlib.sha256(self._tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != self._tokenizer_digest:
            raise ModelInputError("Tokenizer drift")
        if not self._offline:
            actual = {name: importlib.metadata.version(name) for name in snapshot.VERSIONS}
            if (
                actual != snapshot.VERSIONS
                or platform.system() != "Linux"
                or platform.machine() != self._profile.platform_machine
                or os.getuid() != 1000
                or os.getgid() != 1000
            ):
                raise ModelInputError("Inference runtime differs")
            env = self.manifest["runtime"]["required_environment"]
            if env.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or any(os.environ.get(k) != v for k, v in env.items()):
                raise ModelInputError("Inference environment differs")

    def compile(
        self,
        messages,
        *,
        output_reserved_tokens=256,
        remaining_input_tokens=32768,
        remaining_output_tokens=1536,
        scope=None,
    ):
        """Compile exact neutral input IDs for this arm; stateless and repeatable."""
        with self._lock:
            self._validate_runtime()
            return m8_protocol.compile_request(
                self._tokenizer,
                self._arm,
                messages,
                output_reserved_tokens=output_reserved_tokens,
                remaining_input_tokens=remaining_input_tokens,
                remaining_output_tokens=remaining_output_tokens,
                window_tokens=self._profile.runtime.WINDOW,
                scope=scope,
                template_sha256=self._profile.template_sha256,
                template_kwargs=self._profile.template_kwargs,
            )

    def execute(self, compiled):
        """Generate once for one exact compiled request; retain every outcome."""
        with self._lock:
            self.last_execution = None
            self._require_open()
            if self._offline:
                raise ModelInputError("Offline consumer cannot execute")
            if not isinstance(compiled, m8_protocol.CompiledM8Request):
                raise ModelInputError("Neutral execution requires an exact CompiledM8Request")
            if compiled.arm != self._arm:
                raise ModelInputError("Compiled request arm differs from this consumer")
            if compiled.request_digest in self._consumed:
                raise ModelInputError("Compiled request already consumed")
            self._validate_runtime()
            if self._runtime is None:
                self._runtime = self._runtime_class(
                    self._model_path,
                    self._adapter_path,
                    self._tokenizer,
                    self._identity,
                    self._validate_runtime,
                    arm=self._arm,
                    seed=self._seed,
                )
            self._consumed.add(compiled.request_digest)
            result = self._runtime.execute(compiled)
            self.last_execution = deepcopy(result)
            if result.get("usage_unknown") or result.get("execution_error"):
                raise ModelInputExecutionError("Neutral generation failed; consumption may be unknown")
            return result

    def decode_output(self, output_ids):
        stops = self._profile.runtime.STOPS
        ids = list(output_ids)
        if ids and ids[-1] in stops:
            ids = ids[:-1]
        return self._tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def validate_output_tokens(self, output_ids):
        """Data-only token-mask replay against this arm's neutral grammar."""
        profile = self._profile
        runtime_module = profile.runtime
        import torch
        import xgrammar as xgr

        if importlib.metadata.version("xgrammar") != profile.snapshot.VERSIONS["xgrammar"]:
            raise ModelInputError("Replay grammar runtime differs")
        matcher = xgr.GrammarMatcher(
            compile_grammar(self._tokenizer, self._arm, runtime_module=runtime_module),
            override_stop_tokens=runtime_module.STOPS,
            terminate_without_stop_token=False,
            max_rollback_tokens=-1,
        )
        mask = xgr.allocate_token_bitmask(1, runtime_module.VOCAB)
        controls = _control_ids(runtime_module, self._tokenizer)
        for index, token in enumerate(output_ids):
            if type(token) is not int or not 0 <= token < runtime_module.VOCAB:
                raise ModelInputError("Invalid token ID")
            if token in controls and (token not in runtime_module.STOPS or index != len(output_ids) - 1):
                raise ModelInputError("Embedded output control")
            matcher.fill_next_token_bitmask(mask)
            scores = torch.zeros((1, runtime_module.VOCAB), dtype=torch.float32)
            xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=runtime_module.VOCAB, backend="cpu")
            if not bool(torch.isfinite(scores[0, token])) or not matcher.accept_token(token):
                raise ModelInputError("Output violates token grammar")
        return self.decode_output(output_ids)
