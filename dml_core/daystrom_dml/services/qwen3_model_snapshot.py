"""Separate CPU/BF16 Qwen3 profile retaining two original safetensors shards."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import tempfile
from typing import Any, Mapping

from ..contracts.model_input import ModelInputIdentity
from . import model_input_snapshot as base
from .qwen_model_snapshot import QWEN_CHAT_TEMPLATE, SPECIAL_TOKENS

SNAPSHOT_SCHEMA_VERSION = "dml-qwen3-sharded-model-snapshot-v1"
CONSUMER_PROFILE = "qwen3-instruct-nonthinking-bf16-v1"
SHARD_FILES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
REQUIRED_FILES = frozenset({"config.json", "tokenizer.json", "chat_template.jinja",
                            "model.safetensors.index.json", *SHARD_FILES})
QWEN3_CHAT_TEMPLATE = QWEN_CHAT_TEMPLATE + r"{{- '<think>\n\n</think>\n\n' -}}"
QWEN3_CHAT_TEMPLATE_DIGEST = hashlib.sha256(QWEN3_CHAT_TEMPLATE.encode()).hexdigest()
_CHUNK_BYTES = 1024 * 1024
_TENSOR_CHUNK_ELEMENTS = 512 * 1024
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_SHARD_BYTES = 4_200_000_000
_METADATA_LIMITS = {"config.json": 65536, "tokenizer.json": 16 * 1024 * 1024,
    "chat_template.jinja": 65536, "model.safetensors.index.json": 1024 * 1024,
    "snapshot.json": 65536, "LICENSE": 65536, "tokenizer_config.json": 65536}
_CONFIG_FIELDS = frozenset({"architectures", "attention_bias", "attention_dropout", "bos_token_id",
    "eos_token_id", "head_dim", "hidden_act", "hidden_size", "initializer_range", "intermediate_size",
    "max_position_embeddings", "max_window_layers", "model_type", "num_attention_heads",
    "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_scaling", "rope_theta",
    "sliding_window", "tie_word_embeddings", "torch_dtype", "transformers_version", "use_cache",
    "use_sliding_window", "vocab_size"})


def expected_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Include both supplied head and embedding; neither source payload is omitted."""
    hidden, intermediate, head = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    query, kv = config["num_attention_heads"] * head, config["num_key_value_heads"] * head
    shapes = {"model.embed_tokens.weight": (config["vocab_size"], hidden),
              "model.norm.weight": (hidden,), "lm_head.weight": (config["vocab_size"], hidden)}
    for index in range(config["num_hidden_layers"]):
        root = f"model.layers.{index}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            shapes[root + name] = (hidden,)
        for name in ("q_norm.weight", "k_norm.weight"):
            shapes[root + "self_attn." + name] = (head,)
        for name, width in (("q_proj", query), ("k_proj", kv), ("v_proj", kv)):
            shapes[root + "self_attn." + name + ".weight"] = (width, hidden)
        shapes[root + "self_attn.o_proj.weight"] = (hidden, query)
        for name in ("gate_proj", "up_proj"):
            shapes[root + "mlp." + name + ".weight"] = (intermediate, hidden)
        shapes[root + "mlp.down_proj.weight"] = (hidden, intermediate)
    return shapes


def check_config(config: Mapping[str, Any], window: int) -> None:
    if set(config) != _CONFIG_FIELDS:
        raise base.SnapshotVerificationError("Unknown or incomplete Qwen3 configuration")
    exact = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_act": "silu",
        "attention_bias": False, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "use_cache": True, "use_sliding_window": False, "sliding_window": None, "rope_scaling": None}
    if any(type(config[key]) is not type(value) or config[key] != value for key, value in exact.items()):
        raise base.SnapshotVerificationError("Qwen3 architecture is outside the admitted profile")
    for name in ("hidden_size", "intermediate_size", "head_dim", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "vocab_size", "max_position_embeddings"):
        if type(config[name]) is not int or not 1 <= config[name] <= 262144:
            raise base.SnapshotVerificationError("Invalid bounded Qwen3 dimension")
    if (config["head_dim"] % 2 or config["head_dim"] * config["num_attention_heads"] != config["hidden_size"]
            or config["num_attention_heads"] % config["num_key_value_heads"]
            or config["num_hidden_layers"] > 28 or type(window) is not int
            or not 1 <= window <= 32768 or not window <= config["max_position_embeddings"] <= 40960
            or type(config["max_window_layers"]) is not int
            or config["max_window_layers"] != config["num_hidden_layers"]):
        raise base.SnapshotVerificationError("Unsupported Qwen3 attention or context dimensions")
    for name in ("bos_token_id", "eos_token_id"):
        if type(config[name]) is not int or not 0 <= config[name] < config["vocab_size"]:
            raise base.SnapshotVerificationError("Invalid Qwen3 configured special token")
    for name in ("initializer_range", "rms_norm_eps", "rope_theta"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0:
            raise base.SnapshotVerificationError("Invalid Qwen3 scalar")
    if type(config["attention_dropout"]) not in (int, float) or config["attention_dropout"] != 0:
        raise base.SnapshotVerificationError("Qwen3 inference dropout must be disabled")
    if type(config["transformers_version"]) is not str:
        raise base.SnapshotVerificationError("Invalid Qwen3 producer version")
    if sum(math.prod(shape) for shape in expected_shapes(config).values()) > 2_100_000_000:
        raise base.SnapshotVerificationError("Qwen3 model exceeds the admitted stored-parameter bound")


def _stable(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def read_regular(source: Path, destination: Path | None = None) -> tuple[bytes, str]:
    """Bounded metadata capture; shard bytes are only streamed and hashed."""
    limit = _MAX_SHARD_BYTES if source.name in SHARD_FILES else _METADATA_LIMITS.get(source.name)
    if limit is None:
        raise base.SnapshotVerificationError("Unlisted Qwen3 artifact name")
    before = source.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit):
        raise base.SnapshotVerificationError("Invalid Qwen3 regular artifact or bounded size")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    output = None
    digest, captured = hashlib.sha256(), bytearray()
    with os.fdopen(descriptor, "rb") as handle:
        if _stable(os.fstat(handle.fileno())) != _stable(before):
            raise base.SnapshotVerificationError("Qwen3 artifact changed while opening")
        try:
            if destination is not None:
                output = destination.open("xb")
            total = 0
            while chunk := handle.read(_CHUNK_BYTES):
                total += len(chunk)
                if total > before.st_size:
                    raise base.SnapshotVerificationError("Qwen3 artifact grew while reading")
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
                if source.name not in SHARD_FILES:
                    captured.extend(chunk)
            if (total != before.st_size or _stable(source.lstat()) != _stable(before)
                    or _stable(os.fstat(handle.fileno())) != _stable(before)):
                raise base.SnapshotVerificationError("Qwen3 artifact changed while reading")
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()
    return bytes(captured), digest.hexdigest()


def validate_index(index: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    shapes = expected_shapes(config)
    if (type(index) is not dict or index.keys() != {"metadata", "weight_map"}
            or type(index["metadata"]) is not dict or index["metadata"].keys() != {"total_size"}
            or type(index["metadata"]["total_size"]) is not int
            or index["metadata"]["total_size"] != 2 * sum(math.prod(shape) for shape in shapes.values())
            or type(index["weight_map"]) is not dict or index["weight_map"].keys() != shapes.keys()
            or any(type(value) is not str or value not in SHARD_FILES for value in index["weight_map"].values())
            or set(index["weight_map"].values()) != set(SHARD_FILES)):
        raise base.SnapshotVerificationError("Qwen3 index coverage, size or shard mapping differs")
    return dict(index["weight_map"])


def _check_header(path: Path, config: Mapping[str, Any], index: Mapping[str, str]) -> None:
    """Reject duplicate JSON keys and enforce exact gap-free BF16 payload layout."""
    before = path.lstat()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if not stat.S_ISREG(before.st_mode) or _stable(os.fstat(stream.fileno())) != _stable(before):
            raise base.SnapshotVerificationError("Qwen3 shard changed while opening header")
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise base.SnapshotVerificationError("Truncated Qwen3 safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 1 <= length <= _MAX_HEADER_BYTES:
            raise base.SnapshotVerificationError("Qwen3 safetensors header exceeds bound")
        data = stream.read(length)
        if len(data) != length:
            raise base.SnapshotVerificationError("Truncated Qwen3 safetensors header")
        header = base._json_object(data, path.name)
        metadata = header.pop("__metadata__", None)
        if metadata is not None and (type(metadata) is not dict
                or any(type(k) is not str or type(v) is not str for k, v in metadata.items())):
            raise base.SnapshotVerificationError("Invalid Qwen3 safetensors metadata")
        names = {name for name, shard in index.items() if shard == path.name}
        if header.keys() != names:
            raise base.SnapshotVerificationError("Qwen3 actual shard inventory differs from index")
        shapes, intervals = expected_shapes(config), []
        for name, value in header.items():
            if (type(value) is not dict or value.keys() != {"dtype", "shape", "data_offsets"}
                    or value["dtype"] != "BF16" or type(value["shape"]) is not list
                    or any(type(x) is not int for x in value["shape"])
                    or value["shape"] != list(shapes[name]) or type(value["data_offsets"]) is not list
                    or len(value["data_offsets"]) != 2 or any(type(x) is not int for x in value["data_offsets"])):
                raise base.SnapshotVerificationError("Qwen3 safetensors shape, dtype or offsets differ")
            start, end = value["data_offsets"]
            if start < 0 or end - start != 2 * math.prod(shapes[name]):
                raise base.SnapshotVerificationError("Invalid Qwen3 safetensors payload length")
            intervals.append((start, end))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise base.SnapshotVerificationError("Qwen3 safetensors payload gaps or overlaps")
            cursor = end
        if cursor + 8 + length != before.st_size:
            raise base.SnapshotVerificationError("Qwen3 safetensors has trailing or incomplete payload")
        if _stable(path.lstat()) != _stable(before) or _stable(os.fstat(stream.fileno())) != _stable(before):
            raise base.SnapshotVerificationError("Qwen3 shard changed during header verification")


def tensor_digest(tensor: Any) -> str:
    import torch
    flat = tensor.detach().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), _TENSOR_CHUNK_ELEMENTS):
        digest.update(memoryview(flat[start:start + _TENSOR_CHUNK_ELEMENTS].view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_state(state: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch
    if sys.byteorder != "little":
        raise base.SnapshotVerificationError("Qwen3 BF16 source bytes require little-endian CPU")
    shapes = expected_shapes(config)
    if state.keys() != shapes.keys():
        raise base.SnapshotVerificationError("Qwen3 learned tensor coverage differs")
    records = []
    for name in sorted(shapes):
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tuple(tensor.shape) != shapes[name]):
            raise base.SnapshotVerificationError("Qwen3 tensor shape, dtype, device or layout differs")
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), _TENSOR_CHUNK_ELEMENTS):
            if not bool(torch.isfinite(flat[start:start + _TENSOR_CHUNK_ELEMENTS]).all()):
                raise base.SnapshotVerificationError("Qwen3 source contains non-finite tensor values")
        digest = tensor_digest(tensor)
        records.append({"name": name, "shape": list(shapes[name]), "source_dtype": "bfloat16",
                        "target_dtype": "bfloat16", "source_sha256": digest, "target_sha256": digest})
    embedding = state["model.embed_tokens.weight"].detach().reshape(-1)
    head = state["lm_head.weight"].detach().reshape(-1)
    for start in range(0, embedding.numel(), _TENSOR_CHUNK_ELEMENTS):
        if not torch.equal(embedding[start:start + _TENSOR_CHUNK_ELEMENTS].view(torch.uint8),
                           head[start:start + _TENSOR_CHUNK_ELEMENTS].view(torch.uint8)):
            raise base.SnapshotVerificationError("Qwen3 supplied head is not a byte-exact embedding duplicate")
    return records


def load_qwen3_state(directory: Path, config: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.torch import load_file
    mapping = validate_index(index, config)
    state: dict[str, Any] = {}
    for shard in SHARD_FILES:
        _check_header(directory / shard, config, mapping)
        values = load_file(str(directory / shard), device="cpu")
        if values.keys() != {name for name, value in mapping.items() if value == shard} or state.keys() & values.keys():
            raise base.SnapshotVerificationError("Qwen3 loaded shard inventory differs")
        state.update(values)
    validate_state(state, config)
    return state


def _inventory(directory: Path, *, include_manifest: bool = True) -> None:
    expected = REQUIRED_FILES | {"snapshot.json"} if include_manifest else REQUIRED_FILES
    if {entry.name for entry in directory.iterdir()} != expected:
        raise base.SnapshotVerificationError("Qwen3 snapshot contains missing or unlisted artifacts")
    for name in expected:
        info = (directory / name).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise base.SnapshotVerificationError("Qwen3 artifacts must be exclusive regular files")


def _directory(path: str | Path) -> tuple[Path, os.stat_result]:
    directory = Path(path).absolute()
    if ".." in directory.parts:
        raise base.SnapshotVerificationError("Qwen3 directory traversal is excluded")
    for ancestor in (directory, *directory.parents):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise base.SnapshotVerificationError("Qwen3 directory symlinks are excluded")
    return directory, directory.lstat()


def _manifest(payload: bytes) -> base.SnapshotManifest:
    raw = base._json_object(payload, "snapshot.json")
    if raw.keys() != base._MANIFEST_FIELDS or raw["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise base.SnapshotVerificationError("Unknown or incomplete Qwen3 manifest schema")
    for name in ("model_id", "model_revision"):
        if type(raw[name]) is not str or not raw[name].strip() or len(raw[name].encode()) > 1024:
            raise base.SnapshotVerificationError("Invalid Qwen3 source identity")
    if type(raw["context_window"]) is not int or not 1 <= raw["context_window"] <= 32768:
        raise base.SnapshotVerificationError("Invalid Qwen3 effective context window")
    if (type(raw["files"]) is not dict or raw["files"].keys() != REQUIRED_FILES
            or any(type(x) is not str or not base._HASH.fullmatch(x) for x in raw["files"].values())
            or type(raw["runtime_versions"]) is not dict or raw["runtime_versions"] != dict(base.RUNTIME_VERSION_PINS)
            or type(raw["special_tokens"]) is not dict or raw["special_tokens"] != SPECIAL_TOKENS):
        raise base.SnapshotVerificationError("Qwen3 manifest files, versions or special tokens differ")
    return base.SnapshotManifest(raw["schema_version"], raw["model_id"], raw["model_revision"],
        raw["context_window"], tuple(sorted(raw["runtime_versions"].items())),
        tuple(sorted(raw["special_tokens"].items())), tuple(sorted(raw["files"].items())))


def _identity(manifest: base.SnapshotManifest, hashes: Mapping[str, str], versions: Mapping[str, str]):
    return ModelInputIdentity(
        model_digest=base._canonical_digest({"model_id": manifest.model_id, "model_revision": manifest.model_revision,
            "context_window": manifest.context_window, "files": {name: hashes[name] for name in
                ("config.json", "model.safetensors.index.json", *SHARD_FILES)}}),
        tokenizer_digest=base._canonical_digest({"files": {"tokenizer.json": hashes["tokenizer.json"]},
            "special_tokens": dict(manifest.special_tokens)}), chat_template_digest=hashes["chat_template.jinja"],
        runtime_identity="dml-qwen3-model-input-runtime-v1:" + base._canonical_digest({
            "versions": versions, "python": platform.python_version(), "device": "cpu", "dtype": "bfloat16",
            "model_class": "Qwen3ForCausalLM", "tokenizer_class": "PreTrainedTokenizerFast", "attention": "eager",
            "cache": "request-local-dynamic-v1", "decoder": "one-terminal-eos-only-v1",
            "weights": "original-shards-strict-assigned-byte-equal-tied-head-v1",
            "buffers": "deterministic-qwen3-rope-float32-v1", "fingerprint": "raw-bf16-full-scan-v1",
            "thinking": "disabled-exact-input-prefix-v1"}), model_window_tokens=manifest.context_window)


class VerifiedQwen3Snapshot(base.VerifiedSnapshot):
    @property
    def index(self) -> dict[str, Any]:
        if self._closed:
            raise base.SnapshotVerificationError("Verified snapshot is closed")
        payload, _ = read_regular(self.path / "model.safetensors.index.json")
        return base._json_object(payload, "model.safetensors.index.json")

    def validate_integrity(self) -> None:
        try:
            _inventory(self.path, include_manifest=False)
            observed = {}
            for name, expected in self.manifest.files:
                payload, observed[name] = read_regular(self.path / name)
                if observed[name] != expected:
                    raise base.SnapshotVerificationError("Verified Qwen3 artifact changed")
                if name == "config.json" and payload != self._config_bytes:
                    raise base.SnapshotVerificationError("Verified Qwen3 configuration cache differs")
                if name == "chat_template.jinja" and payload != self._chat_template.encode():
                    raise base.SnapshotVerificationError("Verified Qwen3 template cache differs")
            versions = base._check_runtime(self.manifest)
            if self.identity != _identity(self.manifest, observed, versions):
                raise base.SnapshotVerificationError("Verified Qwen3 identity changed")
        except OSError as exc:
            raise base.SnapshotVerificationError("Cannot verify private Qwen3 artifacts") from exc


def verify_qwen3_snapshot(directory: str | Path) -> VerifiedQwen3Snapshot:
    temporary = None
    try:
        source, initial = _directory(directory)
        _inventory(source)
        manifest_bytes, _ = read_regular(source / "snapshot.json")
        manifest = _manifest(manifest_bytes)
        versions = base._check_runtime(manifest)
        temporary = tempfile.TemporaryDirectory(prefix="dml-qwen3-input-")
        target = Path(temporary.name)
        contents, observed = {}, {}
        for name, expected in manifest.files:
            contents[name], observed[name] = read_regular(source / name, target / name)
            if observed[name] != expected:
                raise base.SnapshotVerificationError("Qwen3 artifact hash mismatch")
        final, _ = read_regular(source / "snapshot.json")
        _inventory(source)
        current = source.lstat()
        if final != manifest_bytes or (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
            raise base.SnapshotVerificationError("Qwen3 source directory changed while copying")
        config = base._json_object(contents["config.json"], "config.json")
        check_config(config, manifest.context_window)
        mapping = validate_index(base._json_object(contents["model.safetensors.index.json"], "model.safetensors.index.json"), config)
        for shard in SHARD_FILES:
            _check_header(target / shard, config, mapping)
        base._json_object(contents["tokenizer.json"], "tokenizer.json")
        if contents["chat_template.jinja"] != QWEN3_CHAT_TEMPLATE.encode():
            raise base.SnapshotVerificationError("Unsupported Qwen3 non-thinking template")
        for name in REQUIRED_FILES:
            (target / name).chmod(0o400)
        target.chmod(0o500)
        return VerifiedQwen3Snapshot(temporary, manifest, _identity(manifest, observed, versions),
            contents["config.json"], QWEN3_CHAT_TEMPLATE, _verification_token=base._VERIFICATION_TOKEN)
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        raise
