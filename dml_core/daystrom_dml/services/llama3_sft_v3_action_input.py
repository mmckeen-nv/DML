"""Manifest-bound enabled-only Llama3 SFTv3 consumer and data-only replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import struct
import subprocess
from threading import RLock

from ..contracts.agent_episode import (
    LLAMA3_SFT_V3_CONSUMER_PROFILE,
    canonical_json,
    episode_tool_definitions,
)
from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputIdentity, ModelInputRequest
from .model_input import ModelInputExecutionError, ModelInputResult
from . import llama3_sft_runtime as runtime

MANIFEST_NAME = "llama3-sft-v3-manifest.json"
MANIFEST_VERSION = "dml-llama3-sft-snapshot-v2"
RUNTIME_PREFIX = "dml-llama3-sft-action-runtime-v2:"
REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CHECKPOINT = {
    "adapter_config.json": "3bbc80266e4c54239cbe77da587fdf6ac61df98bdc7757097dd064045f1aedf6",
    "adapter_model.safetensors": "e7bb69e35d5275be8aefdc9f2088db1eccb8503c4460cee27554125d2434cd1a",
}
TRAINING_PROVENANCE = {
    "recipe": "dml-llama3-lora-sft-v3",
    "selection": "final_scheduled_only",
    "optimizer_steps": 148,
    "train_examples": 592,
    "dev_examples": 148,
    "train_example_forwards": 1184,
    "dev_example_forwards": 296,
    "seed": 2026092903,
    "final_checkpoint_sha256": "4b8d15b2c29c3e70eaf039f3cad6fa4def117b43246212a28d838554b3c5e27e",
    "training_completed_sha256": "ff9228f6aff13a2f819b49f227e1e45271fb13fe60ef46a163d661dbbee94edf",
    "training_freeze_sha256": "ab7fa82043009cd85991a9b2f54a3f044e4de72f097a28647dd65a9b943bfad6",
    "independent_training_review_sha256": "3ad5d95323b1c8f2e0495db0b87a1e0304bf2957168177da2ca16f2bf264ad9c",
}
VERSIONS = {
    "torch": "2.11.0+cu130",
    "transformers": "5.6.2",
    "tokenizers": "0.22.2",
    "xgrammar": "0.1.33",
    "apache-tvm-ffi": "0.1.9",
    "safetensors": "0.7.0",
    "peft": "0.21.1",
}
IMAGE = "sha256:bf05c8a0d56f7ed2fe279e0c0c876dd2bb813dd34702d975fa6ab640860ce825"


OFFICIAL_INVENTORY_DIGEST = "c62190d63b6effa9dcacf4221d2fff34e3efa81e1613b239517093c40e2059ec"
VENDOR_TEMPLATE_DIGEST = "ba03a121d097859c7b5b9cd03af99aafe95275210d2876f642ad9929a150f122"
REQUIRED_ENVIRONMENT = {
    "USER": "nvidia",
    "LOGNAME": "nvidia",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/dml-llama3-inductor",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}


def sampling_policy_identity():
    return {
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "min_p": 0,
        "seed": 0,
        "repetition_penalty": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "sampler_device": "cpu",
        "processor_order": [
            "schema-grammar",
            "verified-vocabulary-and-controls",
            "temperature",
            "top_k",
            "top_p",
            "multinomial",
        ],
        "rng": "torch.random.fork_rng(devices=[]);torch.manual_seed(0) per request",
    }


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def source_order_request(request):
    """Restore the original producer's dictionary order, never alter string bytes.

    ModelInputRequest canonicalizes keys; the trained JSON transport does not.
    This view reconstructs the fixed DML producer ordering for live and replay.
    """
    definitions = episode_tool_definitions()
    by_name = {item["function"]["name"]: item for item in definitions}
    tools = []
    for tool in request.tools:
        name = tool["function"]["name"]
        if name not in by_name or tool != by_name[name]:
            raise ModelInputError("Tool schema differs from original DML definition")
        tools.append(deepcopy(by_name[name]))
    messages = []
    for message in request.messages:
        if message["role"] == "tool":
            order = ("role", "tool_call_id", "name", "content")
        else:
            order = ("role", "content", "tool_calls")
        if set(message) - set(order):
            raise ModelInputError("Unsupported logical message fields")
        value = {key: deepcopy(message[key]) for key in order if key in message}
        if "tool_calls" in value:
            value["tool_calls"] = [
                dict(
                    id=call["id"],
                    type=call["type"],
                    function=dict(name=call["function"]["name"], arguments=call["function"]["arguments"]),
                )
                for call in value["tool_calls"]
            ]
        messages.append(value)
    return messages, tools


def _relative(root, name):
    path = Path(name)
    if not isinstance(name, str) or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ModelInputError("Snapshot paths must be relative and contained")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ModelInputError("Snapshot path escapes root")
    return resolved


def verify_manifest(directory):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("model_id") != MODEL
        or manifest.get("revision") != REVISION
        or manifest.get("adapter_enabled") is not True
        or manifest.get("consumer_profile") != LLAMA3_SFT_V3_CONSUMER_PROFILE
        or manifest.get("training_provenance") != TRAINING_PROVENANCE
    ):
        raise ModelInputError("Not the enabled-only registered SFTv3 snapshot")
    if manifest.get("runtime", {}).get("versions") != VERSIONS or manifest["runtime"].get("image") != IMAGE:
        raise ModelInputError("Snapshot runtime does not match qualified runtime")
    paths = {}
    for name, expected in manifest["files"].items():
        path = _relative(root, name)
        if not path.is_file() or sha(path) != expected:
            raise ModelInputError("Snapshot file hash mismatch")
        paths[name] = path
    model = _relative(root, manifest["model_directory"])
    adapter = _relative(root, manifest["adapter_directory"])
    inventory = {
        str(p.relative_to(model)): manifest["files"][name] for name, p in paths.items() if p.is_relative_to(model)
    }
    if _digest(inventory) != OFFICIAL_INVENTORY_DIGEST:
        raise ModelInputError("Base inventory differs from authenticated official revision")
    if {str(p.relative_to(model)) for p in model.rglob("*") if p.is_file()} != set(inventory):
        raise ModelInputError("Unadvertised base files could influence loading")
    if {p.name for p in adapter.iterdir()} != set(CHECKPOINT):
        raise ModelInputError("Unadvertised adapter files")
    if manifest.get("chat_template_digest") != VENDOR_TEMPLATE_DIGEST:
        raise ModelInputError("Original vendor template digest required")
    if manifest["runtime"].get("required_environment") != REQUIRED_ENVIRONMENT:
        raise ModelInputError("Required runtime environment is fixed, not caller-selected")
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    ]
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if any(Path(name).name != name or name not in inventory for name in shards):
        raise ModelInputError("Shard index escapes authenticated inventory")
    required += sorted(shards)
    for name in required:
        if model / name not in paths.values():
            raise ModelInputError("Model inventory incomplete")
    for name, expected in CHECKPOINT.items():
        if adapter / name not in paths.values() or sha(adapter / name) != expected:
            raise ModelInputError("Only final scheduled SFTv3 checkpoint is admitted")
    config = json.loads((model / "config.json").read_text())
    generation = json.loads((model / "generation_config.json").read_text())
    if (
        config["max_position_embeddings"] != runtime.WINDOW
        or config["vocab_size"] != runtime.VOCAB
        or config["torch_dtype"] != "bfloat16"
        or config["eos_token_id"] != 128009
        or generation["eos_token_id"] != runtime.STOPS
    ):
        raise ModelInputError("Original Llama3 architecture/stops differ")
    with (adapter / "adapter_model.safetensors").open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 10000000:
            raise ModelInputError("Unbounded adapter header")
        tensors = {k: v for k, v in json.loads(handle.read(size)).items() if k != "__metadata__"}
    if len(tensors) != 448 or any("lora_" not in k or v["dtype"] != "F32" for k, v in tensors.items()):
        raise ModelInputError("Adapter tensor layout differs")
    return root, manifest, paths, model, adapter


def identity_for_manifest(manifest):
    policy = {
        "consumer_profile": LLAMA3_SFT_V3_CONSUMER_PROFILE,
        "hardware": {
            "device": "NVIDIA GB300",
            "capability": [10, 3],
            "driver": "590.48.01",
            "platform": "Linux aarch64",
            "uid": 1000,
            "gid": 1000,
        },
        "manifest": manifest,
        "transport": runtime.TRANSPORT,
        "system_policy": runtime.SYSTEM,
        "source_order": "original-dml-producer-dictionary-order-v1",
        "adapter_enabled": True,
        "merge": False,
        "inference_trainability": "requires_grad_false_after_arm_selection-v1",
        "precision": "BF16 base and compute; FP32 LoRA",
        "sampling": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "min_p": 0,
            "seed": 0,
            "order": [
                "schema-grammar",
                "verified-vocabulary-and-controls",
                "temperature",
                "top_k",
                "top_p",
                "multinomial",
            ],
            "rng": "torch.random.fork_rng(devices=[]);torch.manual_seed(0) per request",
        },
        "stops": runtime.STOPS,
        "grammar": "xgrammar0.1.33 strict any-whitespace; both original action branches",
        "termination": "optional single terminal stop; complete-at-cap accepted by original parser; no repair",
    }
    model_prefix = manifest["model_directory"].rstrip("/") + "/"
    tokenizer_name = model_prefix + "tokenizer.json"
    return ModelInputIdentity(
        model_digest=_digest({"files": manifest["files"], "checkpoint": CHECKPOINT}),
        tokenizer_digest=manifest["files"][tokenizer_name],
        chat_template_digest=manifest["chat_template_digest"],
        runtime_identity=RUNTIME_PREFIX + _digest(policy),
        model_window_tokens=runtime.WINDOW,
    )


class LocalLlama3SFTV3ActionInputConsumer:
    """Own bounded, authenticated, single-use exact-input artifacts."""

    def __init__(self, snapshot_directory, *, consumer_profile=LLAMA3_SFT_V3_CONSUMER_PROFILE, offline=False):
        if consumer_profile != LLAMA3_SFT_V3_CONSUMER_PROFILE:
            raise ModelInputError("Unknown Llama SFT consumer profile")
        self._lock = RLock()
        self._offline = offline
        self._closed = False
        self._root, self.manifest, self._files, self._model_path, self._adapter_path = verify_manifest(
            snapshot_directory
        )
        self._manifest_digest = sha(self._root / MANIFEST_NAME)
        self._stats = {str(p): p.stat() for p in self._files.values()}
        self._identity = identity_for_manifest(self.manifest)
        self.identity = self._identity
        self._consumer_id = secrets.token_hex(24)
        self._auth_key = secrets.token_bytes(32)
        self._bound_requests: dict[str, ModelInputRequest] = {}
        self._runtime: runtime.EnabledAdapterRuntime | None = None
        self.last_execution = None
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, local_files_only=True, trust_remote_code=False
        )
        self._template = json.loads((self._model_path / "tokenizer_config.json").read_text())["chat_template"]
        if hashlib.sha256(self._template.encode()).hexdigest() != self._identity.chat_template_digest:
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
        if len(self._tokenizer) != runtime.VOCAB or set(self._tokenizer.get_vocab().values()) != set(
            range(runtime.VOCAB)
        ):
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
            self._bound_requests.clear()
            self._auth_key = b""
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None

    def _require_open(self):
        if self._closed:
            raise ModelInputError("Consumer closed")

    def _validate_runtime(self):
        self._require_open()
        if sha(self._root / MANIFEST_NAME) != self._manifest_digest:
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
            actual = {name: importlib.metadata.version(name) for name in VERSIONS}
            if (
                actual != VERSIONS
                or platform.system() != "Linux"
                or platform.machine() != "aarch64"
                or os.getuid() != 1000
                or os.getgid() != 1000
            ):
                raise ModelInputError("Inference runtime differs")
            env = self.manifest["runtime"]["required_environment"]
            if env.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or any(os.environ.get(k) != v for k, v in env.items()):
                raise ModelInputError("Inference environment differs")

    def _auth_tag(self, artifact):
        return hmac.new(self._auth_key, canonical_json(artifact.signing_payload()), hashlib.sha256).hexdigest()

    def compile(self, messages, tools=None, *, output_reserved_tokens):
        request = ModelInputRequest.from_payload(
            dict(messages=messages, tools=[] if tools is None else tools, output_reserved_tokens=output_reserved_tokens)
        )
        if output_reserved_tokens > 256:
            raise ModelInputError("Output reservation exceeds qualified cap")
        with self._lock:
            self._validate_runtime()
            if len(self._bound_requests) >= 64:
                raise ModelInputError("Compiled-request capacity exhausted")
            ordered_messages, ordered_tools = source_order_request(request)
            view = runtime.render_messages(ordered_messages, ordered_tools)
            rendered = self._tokenizer.apply_chat_template(
                view, chat_template=self._template, tokenize=False, add_generation_prompt=True
            )
            ids = tuple(self._tokenizer.encode(rendered, add_special_tokens=False))
            artifact = CompiledModelInput(
                identity=self._identity,
                request_digest=request.request_digest,
                input_ids=ids,
                attention_mask=(1,) * len(ids),
                output_reserved_tokens=output_reserved_tokens,
                model_window_tokens=runtime.WINDOW,
                consumer_id=self._consumer_id,
                nonce=secrets.token_hex(24),
                auth_tag="0" * 64,
            )
            artifact = replace(artifact, auth_tag=self._auth_tag(artifact))
            self._bound_requests[artifact.artifact_digest] = request
            return artifact

    def release_compiled(self, artifact):
        """Discard replay-only bindings; cannot turn an online request reusable."""
        with self._lock:
            if (
                not self._offline
                or artifact.consumer_id != self._consumer_id
                or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))
            ):
                raise ModelInputError("Only owned offline artifacts may be released")
            if self._bound_requests.pop(artifact.artifact_digest, None) is None:
                raise ModelInputError("Unknown replay artifact")

    def decode_output(self, output_ids):
        ids = list(output_ids)
        if ids and ids[-1] in runtime.STOPS:
            ids = ids[:-1]
        return self._tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def validate_output_tokens(self, request, output_ids):
        """Data-only token-mask replay, including legitimately incomplete prefixes."""
        if isinstance(request, dict):
            request = ModelInputRequest.from_payload(request)
        _, tools = source_order_request(request)
        import torch
        import xgrammar as xgr

        if importlib.metadata.version("xgrammar") != VERSIONS["xgrammar"]:
            raise ModelInputError("Replay grammar runtime differs")
        matcher = runtime.new_matcher(runtime.compile_grammar(self._tokenizer, tools))
        mask = xgr.allocate_token_bitmask(1, runtime.VOCAB)
        for index, token in enumerate(output_ids):
            if type(token) is not int or not 0 <= token < runtime.VOCAB:
                raise ModelInputError("Invalid token ID")
            if token in runtime.special_ids(self._tokenizer) and (
                token not in runtime.STOPS or index != len(output_ids) - 1
            ):
                raise ModelInputError("Embedded output control")
            matcher.fill_next_token_bitmask(mask)
            scores = torch.zeros((1, runtime.VOCAB), dtype=torch.float32)
            xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=runtime.VOCAB, backend="cpu")
            if not bool(torch.isfinite(scores[0, token])) or not matcher.accept_token(token):
                raise ModelInputError("Output violates token grammar")
        return self.decode_output(output_ids)

    def validate_failed_execution(self, artifact, request, evidence, *, recorded_digest=None):
        """Replay outer failure and any retained runtime prefix without success credit."""
        if (
            evidence.get("artifact_digest")
            != (artifact.artifact_digest if recorded_digest is None else recorded_digest)
            or not evidence.get("exception_type")
            or not isinstance(evidence.get("exception_message"), str)
        ):
            raise ModelInputError("Failed execution envelope differs")
        result = evidence.get("runtime_result")
        if result is None:
            return
        if result.get("model_identity") != artifact.identity.to_payload() or result.get("input_ids") != list(
            artifact.input_ids
        ):
            raise ModelInputError("Failed execution input or identity differs")
        output = result.get("output_ids")
        if type(output) is not list or len(output) > artifact.output_reserved_tokens:
            raise ModelInputError("Failed execution output prefix exceeds reservation")
        self.validate_output_tokens(request, output)
        if result.get("raw_text") is None:
            if not result.get("decode_error"):
                raise ModelInputError("Absent raw text requires retained decode failure")
        elif result["raw_text"] != self._tokenizer.decode(
            output, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ):
            raise ModelInputError("Failed execution raw text differs from token IDs")
        if result.get("usage_unknown") is False:
            if result.get("input_token_count") != len(artifact.input_ids) or result.get("output_token_count") != len(
                output
            ):
                raise ModelInputError("Retained known counts differ from token rows")
        elif result.get("usage_unknown") is not True or not result.get("execution_error"):
            raise ModelInputError("Internal unknown failure requires actual error")

    def execute(self, artifact):
        with self._lock:
            self.last_execution = None
            self._require_open()
            if self._offline:
                raise ModelInputError("Offline consumer cannot execute")
            if (
                type(artifact) is not CompiledModelInput
                or artifact.identity != self._identity
                or artifact.consumer_id != self._consumer_id
                or not hmac.compare_digest(artifact.auth_tag, self._auth_tag(artifact))
            ):
                raise ModelInputError("Unauthenticated compiled input")
            request = self._bound_requests.pop(artifact.artifact_digest, None)
            if request is None or request.request_digest != artifact.request_digest:
                raise ModelInputError("Unknown or already-consumed compiled request")
            self.last_execution = None
            self._validate_runtime()
            if self._runtime is None:
                self._runtime = runtime.EnabledAdapterRuntime(
                    self._model_path,
                    self._adapter_path,
                    self._tokenizer,
                    self._identity.to_payload(),
                    self._validate_runtime,
                )
            _, tools = source_order_request(request)
            result = self._runtime.execute(
                dict(
                    input_ids=list(artifact.input_ids),
                    reserved=artifact.output_reserved_tokens,
                    request={"tools": tools},
                )
            )
            self.last_execution = deepcopy(result)
            if result.get("usage_unknown") or result.get("execution_error"):
                raise ModelInputExecutionError("SFT generation failed; consumption may be unknown")
            return ModelInputResult(
                artifact.artifact_digest,
                result["input_token_count"],
                result["output_token_count"],
                tuple(result["output_ids"]),
                result["text"],
            )
