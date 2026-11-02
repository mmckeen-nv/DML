"""Neutral M8 live-generation binding; development-only, no M7 relabeling.

Reuses the qualified Llama3 generation mechanics (adapter loading, xgrammar
settings, sampler, stop/vocabulary controls) from llama3_sft_runtime without
its hardcoded DML system policy and tool schemas. The neutral M8 protocol
(m8_protocol) owns rendering, token evidence and action parsing; this module
owns grammar compilation and exact-token generation for one arm. The M7
runtime source, identity and evidence are unchanged.

No model is loaded by importing this module; heavy dependencies stay lazy.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from threading import RLock

from ..contracts.agent_episode import canonical_json
from ..contracts.model_input import ModelInputError
from . import m8_protocol
from . import llama3_sft_v3_action_input as snapshot_v3
from .llama3_sft_runtime import (
    STOPS,
    VOCAB,
    EnabledAdapterRuntime,
    sample_next,
    special_ids,
)
from .llama3_sft_v3_action_input import sampling_policy_identity
from .model_input import ModelInputExecutionError


@contextmanager
def _seeded_cpu_rng(torch, seed):
    """Per-request forked CPU RNG with an explicit declared seed (default 0)."""
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        yield


RUNTIME_PREFIX = "dml-m8-neutral-action-runtime-v1:"
GRAMMAR_IDENTITY = "xgrammar0.1.33 strict any-whitespace; neutral retrieve/final branches"


def _arm(arm):
    if type(arm) is not str or arm not in m8_protocol.ARMS:
        raise ValueError("Unknown M8 arm")
    return arm


def neutral_identity(arm, *, base_identity: dict) -> dict:
    """Versioned neutral runtime identity; never an M7 runtime identity."""
    _arm(arm)
    policy = {
        "protocol": m8_protocol.policy_identity(arm),
        "base_identity": base_identity,
        "sampling": sampling_policy_identity(),
        "grammar": GRAMMAR_IDENTITY,
        "termination": "optional single terminal stop; complete-at-cap accepted by original parser; no repair",
    }
    return {
        "runtime_identity": RUNTIME_PREFIX + hashlib.sha256(canonical_json(policy)).hexdigest(),
        "policy": policy,
    }


def compile_grammar(tokenizer, arm):
    """Compile the NEUTRAL action schema for one arm with M7's pinned settings."""
    _arm(arm)
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=VOCAB, stop_token_ids=STOPS)
    return xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False).compile_json_schema(
        m8_protocol.schema_bytes(arm).decode(),
        strict_mode=True,
        any_whitespace=True,
        max_whitespace_cnt=None,
    )


def project_output(output_ids, tokenizer, arm):
    """Project exact output IDs through the neutral parser; never repair."""
    _arm(arm)
    raw = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ended = bool(output_ids and output_ids[-1] in STOPS)
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
        if any(type(t) is not int or not 0 <= t < VOCAB for t in output_ids):
            raise ValueError("Invalid token row")
        if any(t in special_ids(tokenizer) for t in body):
            raise ValueError("Embedded output control")
        # Neutral parser: rejects unadvertised tools, malformed actions and
        # context_only tool use; no DML tool schema is consulted.
        action = m8_protocol.parse_action(text, arm)
        result.update(derived_action=action, derived_action_text=text)
    except Exception as exc:
        result["action_error"] = type(exc).__name__ + ": " + str(exc)
    return result


class NeutralM8AdapterRuntime(EnabledAdapterRuntime):
    """Enabled-adapter generation bound to one neutral arm's grammar/parser.

    The qualified sampler pins seed 0 per request. Development diagnostics
    (e.g. the competence battery) may declare alternative seeds; every result
    records the seed actually used. This never changes a qualified run's seed.
    """

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
            grammar = compile_grammar(self.tokenizer, self._arm)
            matcher = xgr.GrammarMatcher(
                grammar,
                override_stop_tokens=STOPS,
                terminate_without_stop_token=False,
                max_rollback_tokens=-1,
            )
            mask = xgr.allocate_token_bitmask(1, VOCAB)
            allowed = torch.ones(VOCAB, dtype=torch.bool, device="cpu")
            allowed[list(special_ids(self.tokenizer) - set(STOPS))] = False
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
                    if tuple(scores.shape) != (1, VOCAB) or not bool(torch.isfinite(scores).all()):
                        raise ValueError("Invalid logits")
                    matcher.fill_next_token_bitmask(mask)
                    xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=VOCAB, backend="cpu")
                    scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
                    if not bool(torch.isfinite(scores).any()):
                        raise ValueError("Empty grammar distribution")
                    token = sample_next(torch, torch.tensor([all_ids], dtype=torch.long), scores)
                    if not bool(allowed[token]) or not matcher.accept_token(token):
                        raise ValueError("Sample violates token mask")
                    ids.append(token)
                    all_ids += (token,)
                    if token in STOPS:
                        break
                    input_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
            self.validate()
            result.update(
                output_ids=ids,
                input_token_count=len(compiled.input_ids),
                output_token_count=len(ids),
                usage_unknown=False,
            )
            result.update(project_output(ids, self.tokenizer, self._arm))
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


class M8NeutralConsumer:
    """Authenticated-snapshot neutral consumer for ONE arm; development-only.

    Reuses the registered snapshot verification and attestation discipline
    from the M7 consumer without its DML policy, tool schemas or renderer.
    Compilation goes through m8_protocol.compile_request (exact token IDs);
    execution goes through NeutralM8AdapterRuntime. Compiled requests are
    single-use; failures are retained in last_execution, never repaired.
    """

    def __init__(self, snapshot_directory, *, arm, offline=False, seed=0):
        _arm(arm)
        if type(seed) is not int or seed < 0:
            raise ModelInputError("Sampler seed must be a nonnegative integer")
        self._arm = arm
        self._seed = seed
        self._lock = RLock()
        self._offline = offline
        self._closed = False
        self._root, self.manifest, self._files, self._model_path, self._adapter_path = snapshot_v3.verify_manifest(
            snapshot_directory
        )
        self._manifest_digest = snapshot_v3.sha(self._root / snapshot_v3.MANIFEST_NAME)
        self._stats = {str(p): p.stat() for p in self._files.values()}
        base = snapshot_v3.identity_for_manifest(self.manifest)
        self._identity = neutral_identity(
            arm,
            base_identity={
                "model_digest": base.model_digest,
                "tokenizer_digest": base.tokenizer_digest,
                "chat_template_digest": base.chat_template_digest,
                "model_window_tokens": base.model_window_tokens,
                "snapshot_model": snapshot_v3.MODEL,
                "snapshot_revision": snapshot_v3.REVISION,
                "adapter": snapshot_v3.CHECKPOINT,
            },
        )
        self.identity = self._identity
        self._consumed: set[str] = set()
        self._runtime: NeutralM8AdapterRuntime | None = None
        self.last_execution = None
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        )
        template = self._tokenizer.chat_template
        if hashlib.sha256(template.encode()).hexdigest() != m8_protocol.PINNED_TEMPLATE_SHA256:
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
        if raw != configured or raw != loaded or set(raw) != set(range(128000, 128256)):
            raise ModelInputError("Control-token metadata mismatch")
        if len(self._tokenizer) != VOCAB or set(self._tokenizer.get_vocab().values()) != set(range(VOCAB)):
            raise ModelInputError("Vocabulary rows differ")
        if any(self._tokenizer.convert_tokens_to_ids(value) != key for key, value in raw.items()):
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
            if (
                observed_driver != "590.48.01"
                or torch.cuda.get_device_name(0) != "NVIDIA GB300"
                or torch.cuda.get_device_capability(0) != (10, 3)
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
        if snapshot_v3.sha(self._root / snapshot_v3.MANIFEST_NAME) != self._manifest_digest:
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
            actual = {name: importlib.metadata.version(name) for name in snapshot_v3.VERSIONS}
            if (
                actual != snapshot_v3.VERSIONS
                or platform.system() != "Linux"
                or platform.machine() != "aarch64"
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
                scope=scope,
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
                self._runtime = NeutralM8AdapterRuntime(
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
        ids = list(output_ids)
        if ids and ids[-1] in STOPS:
            ids = ids[:-1]
        return self._tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def validate_output_tokens(self, output_ids):
        """Data-only token-mask replay against this arm's neutral grammar."""
        import torch
        import xgrammar as xgr

        if importlib.metadata.version("xgrammar") != snapshot_v3.VERSIONS["xgrammar"]:
            raise ModelInputError("Replay grammar runtime differs")
        matcher = xgr.GrammarMatcher(
            compile_grammar(self._tokenizer, self._arm),
            override_stop_tokens=STOPS,
            terminate_without_stop_token=False,
            max_rollback_tokens=-1,
        )
        mask = xgr.allocate_token_bitmask(1, VOCAB)
        for index, token in enumerate(output_ids):
            if type(token) is not int or not 0 <= token < VOCAB:
                raise ModelInputError("Invalid token ID")
            if token in special_ids(self._tokenizer) and (token not in STOPS or index != len(output_ids) - 1):
                raise ModelInputError("Embedded output control")
            matcher.fill_next_token_bitmask(mask)
            scores = torch.zeros((1, VOCAB), dtype=torch.float32)
            xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=VOCAB, backend="cpu")
            if not bool(torch.isfinite(scores[0, token])) or not matcher.accept_token(token):
                raise ModelInputError("Output violates token grammar")
        return self.decode_output(output_ids)
