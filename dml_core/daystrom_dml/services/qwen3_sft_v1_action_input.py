"""Manifest-bound enabled-only Qwen3.5 SFTv1 consumer and data-only replay.

Follows the Llama3 SFTv3 consumer structure for the Qwen/Qwen3.5-9B base. The
adapter checkpoint, training provenance, container image, official base
inventory and runtime version constants are fail-closed placeholders: no
manifest can be admitted until the Qwen3.5 SFT adapter is trained and the
qualified container pins replace the placeholder values in this module.

Tokenizer and template pins below were computed from the unauthenticated public
Hugging Face revision REVISION (config.json, tokenizer_config.json,
tokenizer.json and chat_template.jinja); verify_source_files re-checks them
against a local source directory.
"""

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

from ..contracts.agent_episode import QWEN3_SFT_V1_CONSUMER_PROFILE, canonical_json, episode_tool_definitions
from ..contracts.model_input import CompiledModelInput, ModelInputError, ModelInputIdentity, ModelInputRequest
from .model_input import ModelInputExecutionError, ModelInputResult
from . import qwen3_sft_runtime as runtime

MANIFEST_NAME = "qwen3-sft-v1-manifest.json"
MANIFEST_VERSION = "dml-qwen3-sft-snapshot-v1"
RUNTIME_PREFIX = "dml-qwen3-sft-action-runtime-v1:"
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# Placeholder pins: admit nothing until the trained adapter is qualified.
CHECKPOINT = {
    "adapter_config.json": "0" * 64,
    "adapter_model.safetensors": "0" * 64,
}
ADAPTER_TENSORS = 0  # no qualified adapter layout yet; zero admits nothing
TRAINING_PROVENANCE = {
    "recipe": "dml-qwen3_5-9b-lora-sft-v1",
    "status": "placeholder-unqualified",
}

# sha256 of each upstream file at REVISION, computed at profile authoring time.
SOURCE_SHA256 = {
    "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
}
TOKENIZER_JSON_SHA256 = SOURCE_SHA256["tokenizer.json"]
# sha256 of the chat_template string embedded in tokenizer_config.json (byte-
# identical to chat_template.jinja at REVISION).
VENDOR_TEMPLATE_DIGEST = SOURCE_SHA256["chat_template.jinja"]

# Inherits the main-line optional-runtime pins; update at container qualification.
VERSIONS = {
    "torch": "2.11.0+cu130",
    "transformers": "5.6.2",
    "tokenizers": "0.22.2",
    "xgrammar": "0.1.33",
    "apache-tvm-ffi": "0.1.9",
    "safetensors": "0.7.0",
    "peft": "0.21.1",
}
IMAGE = "sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d"  # pinned vllm amd64 (2026-11-07)
OFFICIAL_INVENTORY_DIGEST = "df77b4be88cb5401d49137ad1833107a10fb72436fbdaacf46d764fcf244289f"  # first qualification: Qwen3.5-9B @ REVISION, 16 files (2026-11-07)
REQUIRED_ENVIRONMENT = {
    "USER": "nvidia",
    "LOGNAME": "nvidia",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/dml-qwen3-inductor",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}

# Control-token layout at REVISION: tokenizer.json carries ids 0..248069
# (base BPE rows 0..248043 plus added rows 248044..248069); tokenizer_config.json
# adds audio/TTS rows 248070..248076. Model config vocab_size is 248320.
SPECIAL_TOKENS = {
    248044: "<|endoftext|>",
    248045: "<|im_start|>",
    248046: "<|im_end|>",
    248047: "<|object_ref_start|>",
    248048: "<|object_ref_end|>",
    248049: "<|box_start|>",
    248050: "<|box_end|>",
    248051: "<|quad_start|>",
    248052: "<|quad_end|>",
    248053: "<|vision_start|>",
    248054: "<|vision_end|>",
    248055: "<|vision_pad|>",
    248056: "<|image_pad|>",
    248057: "<|video_pad|>",
}
CONFIG_ONLY_SPECIAL_TOKENS = {
    248070: "<|audio_start|>",
    248071: "<|audio_end|>",
    248072: "<tts_pad>",
    248073: "<tts_text_bos>",
    248074: "<tts_text_eod>",
    248075: "<tts_text_bos_single>",
    248076: "<|audio_pad|>",
}
NONSPECIAL_ADDED_TOKENS = {
    248058: "<tool_call>",
    248059: "</tool_call>",
    248060: "<|fim_prefix|>",
    248061: "<|fim_middle|>",
    248062: "<|fim_suffix|>",
    248063: "<|fim_pad|>",
    248064: "<|repo_name|>",
    248065: "<|file_sep|>",
    248066: "<tool_response>",
    248067: "</tool_response>",
    248068: "<think>",
    248069: "</think>",
}

_DEVELOPMENT_RUNTIME_KEYS = frozenset({"label", "platform_machine", "driver", "device_name", "capability"})


def _validate_development_runtime(identity):
    """Shape-check a development runtime identity (never a production/qualified identity)."""
    if type(identity) is not dict or set(identity) != _DEVELOPMENT_RUNTIME_KEYS:
        raise ModelInputError(
            "development_runtime must declare exactly: label, platform_machine, driver, device_name, capability"
        )
    label = identity["label"]
    if type(label) is not str or not label.startswith("development-"):
        raise ModelInputError("development_runtime label must be a 'development-*' string")
    for key in ("platform_machine", "driver", "device_name"):
        if type(identity[key]) is not str or not identity[key]:
            raise ModelInputError(f"development_runtime {key} must be a non-empty string")
    capability = identity["capability"]
    if type(capability) is not tuple or len(capability) != 2 or any(type(v) is not int for v in capability):
        raise ModelInputError("development_runtime capability must be a (major, minor) int tuple")


_DEVELOPMENT_CHECKPOINT_KEYS = frozenset({"label", "manifest_sha256", "adapter_sha256"})


def _validate_development_checkpoint(identity):
    """Shape-check a development checkpoint identity (never a qualified/production checkpoint)."""
    if type(identity) is not dict or set(identity) != _DEVELOPMENT_CHECKPOINT_KEYS:
        raise ModelInputError("development_checkpoint must declare exactly: label, manifest_sha256, adapter_sha256")
    label = identity["label"]
    if type(label) is not str or not label.startswith("development-"):
        raise ModelInputError("development_checkpoint label must be a 'development-*' string")
    for key in ("manifest_sha256", "adapter_sha256"):
        value = identity[key]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ModelInputError(f"development_checkpoint {key} must be a 64-character lowercase hex string")


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


def verify_source_files(directory):
    """Re-check the pinned upstream source files against a local directory."""
    root = Path(directory).resolve(strict=True)
    paths = {}
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ModelInputError("Qwen3.5 source file digest differs from the pinned revision")
        paths[name] = path
    template = json.loads(paths["tokenizer_config.json"].read_text())["chat_template"]
    if hashlib.sha256(template.encode()).hexdigest() != VENDOR_TEMPLATE_DIGEST:
        raise ModelInputError("Embedded chat template differs from chat_template.jinja")
    if template != paths["chat_template.jinja"].read_text():
        raise ModelInputError("Embedded chat template is not byte-identical to chat_template.jinja")
    return paths


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


def verify_manifest(directory, *, development_checkpoint=None):
    if development_checkpoint is not None:
        _validate_development_checkpoint(development_checkpoint)
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if development_checkpoint is not None and sha(root / MANIFEST_NAME) != development_checkpoint["manifest_sha256"]:
        raise ModelInputError("Development checkpoint manifest digest differs")
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("model_id") != MODEL
        or manifest.get("revision") != REVISION
        or manifest.get("adapter_enabled") is not True
        or manifest.get("consumer_profile") != QWEN3_SFT_V1_CONSUMER_PROFILE
        or manifest.get("training_provenance") != TRAINING_PROVENANCE
    ):
        raise ModelInputError("Not the enabled-only registered SFTv1 snapshot")
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
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
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
    if development_checkpoint is not None:
        # Candidate admission covers the WEIGHTS only: adapter_config.json is
        # already verified file-by-file against the candidate manifest in the
        # paths loop above; the zero CHECKPOINT placeholder must not leak in.
        checkpoint = {"adapter_model.safetensors": development_checkpoint["adapter_sha256"]}
        checkpoint_error = "Development checkpoint adapter hash differs"
    else:
        checkpoint = CHECKPOINT
        checkpoint_error = "Only final scheduled SFTv1 checkpoint is admitted"
    for name, expected in checkpoint.items():
        if adapter / name not in paths.values() or sha(adapter / name) != expected:
            raise ModelInputError(checkpoint_error)
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", {})
    if (
        config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or text_config.get("max_position_embeddings") != runtime.WINDOW
        or text_config.get("vocab_size") != runtime.VOCAB
        or text_config.get("dtype") != "bfloat16"
        or text_config.get("eos_token_id") != 248044
    ):
        raise ModelInputError("Original Qwen3.5 architecture/stops differ")
    with (adapter / "adapter_model.safetensors").open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 10000000:
            raise ModelInputError("Unbounded adapter header")
        tensors = {k: v for k, v in json.loads(handle.read(size)).items() if k != "__metadata__"}
    if len(tensors) != ADAPTER_TENSORS or any("lora_" not in k or v["dtype"] != "F32" for k, v in tensors.items()):
        raise ModelInputError("Adapter tensor layout differs")
    return root, manifest, paths, model, adapter


def identity_for_manifest(manifest):
    policy = {
        "consumer_profile": QWEN3_SFT_V1_CONSUMER_PROFILE,
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
        "thinking": "disabled: enable_thinking=False renders a pre-closed empty think block in the prompt; "
        "all added control tokens (including <think>) are masked out of the output stream",
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


class LocalQwen3SFTV1ActionInputConsumer:
    """Own bounded, authenticated, single-use exact-input artifacts."""

    def __init__(
        self,
        snapshot_directory,
        *,
        consumer_profile=QWEN3_SFT_V1_CONSUMER_PROFILE,
        offline=False,
        development_runtime=None,
        development_checkpoint=None,
    ):
        if development_runtime is not None:
            _validate_development_runtime(development_runtime)
        self._development_runtime = None if development_runtime is None else dict(development_runtime)
        if development_checkpoint is not None:
            _validate_development_checkpoint(development_checkpoint)
        self._development_checkpoint = None if development_checkpoint is None else dict(development_checkpoint)
        if consumer_profile != QWEN3_SFT_V1_CONSUMER_PROFILE:
            raise ModelInputError("Unknown Qwen3 SFT consumer profile")
        self._lock = RLock()
        self._offline = offline
        self._closed = False
        self._root, self.manifest, self._files, self._model_path, self._adapter_path = verify_manifest(
            snapshot_directory, development_checkpoint=self._development_checkpoint
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
        raw_tokens = json.loads((self._model_path / "tokenizer.json").read_text())["added_tokens"]
        raw_special = {x["id"]: x["content"] for x in raw_tokens if x["special"]}
        raw_nonspecial = {x["id"]: x["content"] for x in raw_tokens if not x["special"]}
        configured_tokens = json.loads((self._model_path / "tokenizer_config.json").read_text())["added_tokens_decoder"]
        configured_special = {int(k): v["content"] for k, v in configured_tokens.items() if v["special"]}
        configured_nonspecial = {int(k): v["content"] for k, v in configured_tokens.items() if not v["special"]}
        loaded_special = {k: str(v) for k, v in self._tokenizer.added_tokens_decoder.items() if v.special}
        loaded_nonspecial = {k: str(v) for k, v in self._tokenizer.added_tokens_decoder.items() if not v.special}
        if (
            raw_special != SPECIAL_TOKENS
            or raw_nonspecial != NONSPECIAL_ADDED_TOKENS
            or configured_special != {**SPECIAL_TOKENS, **CONFIG_ONLY_SPECIAL_TOKENS}
            or configured_nonspecial != NONSPECIAL_ADDED_TOKENS
            or loaded_special != configured_special
            or loaded_nonspecial != configured_nonspecial
        ):
            raise ModelInputError("Control-token metadata mismatch")
        if len(self._tokenizer) != runtime.TOKENIZER_ROWS or set(self._tokenizer.get_vocab().values()) != set(
            range(runtime.TOKENIZER_ROWS)
        ):
            raise ModelInputError("Vocabulary rows differ")
        if any(self._tokenizer.convert_tokens_to_ids(value) != key for key, value in loaded_special.items()):
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
            if self._development_runtime is not None:
                expected = (
                    self._development_runtime["driver"],
                    self._development_runtime["device_name"],
                    self._development_runtime["capability"],
                )
                label = "Development"
            else:
                expected = ("590.48.01", "NVIDIA GB300", (10, 3))
                label = "Qualified"
            if (
                observed_driver,
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_capability(0),
            ) != expected:
                raise ModelInputError(f"{label} CUDA hardware/driver differs")

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
            machine = (
                self._development_runtime["platform_machine"] if self._development_runtime is not None else "aarch64"
            )
            if (
                actual != VERSIONS
                or platform.system() != "Linux"
                or platform.machine() != machine
                or os.getuid() != 1000
                or os.getgid() != 1000
            ):
                raise ModelInputError(
                    "Development inference runtime differs"
                    if self._development_runtime is not None
                    else "Inference runtime differs"
                )
            env = self.manifest["runtime"]["required_environment"]
            if env.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or any(os.environ.get(k) != v for k, v in env.items()):
                raise ModelInputError("Inference environment differs")

    @property
    def development_runtime_identity(self):
        """Declared development identity (None on the qualified runtime path)."""
        return None if self._development_runtime is None else dict(self._development_runtime)

    @property
    def development_checkpoint_identity(self):
        """Declared development checkpoint identity (None on the qualified snapshot path)."""
        return None if self._development_checkpoint is None else dict(self._development_checkpoint)

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
            rendered = runtime.apply_template(self._tokenizer, self._template, view)
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
        allowed = set(runtime.allowed_output_ids(self._tokenizer))
        for index, token in enumerate(output_ids):
            if type(token) is not int or not 0 <= token < runtime.VOCAB:
                raise ModelInputError("Invalid token ID")
            if token not in allowed and (token not in runtime.STOPS or index != len(output_ids) - 1):
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
