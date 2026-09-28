"""Pinned official Qwen3-8B Q4_K_M artifact admission, never BF16 equivalence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import stat
import struct
import tempfile
from typing import Any

from ..contracts.model_input import ModelInputIdentity
from . import model_input_snapshot as base
from .qwen_model_snapshot import SPECIAL_TOKENS
from .qwen3_model_snapshot import QWEN3_CHAT_TEMPLATE, _directory, _stable

SNAPSHOT_SCHEMA_VERSION = "dml-qwen3-8b-gguf-snapshot-v1"
MODEL_ID = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
HF_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GGUF_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SOURCE_FILE_PINS = {
    "model.gguf": GGUF_SHA256,
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "chat_template.jinja": hashlib.sha256(QWEN3_CHAT_TEMPLATE.encode()).hexdigest(),
}
REQUIRED_FILES = frozenset(SOURCE_FILE_PINS)
RUNTIME_VERSION_PINS = {**dict(base.RUNTIME_VERSION_PINS), "llama-cpp-python": "0.3.35", "diskcache": "5.6.3"}
_LIMITS = {name: 16 * 1024 * 1024 for name in REQUIRED_FILES | {"snapshot.json"}}
_LIMITS["model.gguf"] = 6_000_000_000


def runtime_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_VERSION_PINS}
    if versions != RUNTIME_VERSION_PINS:
        raise base.SnapshotVerificationError("GGUF runtime package versions differ")
    return versions


def read_regular(path: Path, destination: Path | None = None) -> tuple[bytes, str]:
    import os
    limit = _LIMITS.get(path.name)
    before = path.lstat()
    if limit is None or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise base.SnapshotVerificationError("GGUF artifact is not a bounded exclusive regular file")
    digest, payload = hashlib.sha256(), bytearray()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as src:
        if _stable(os.fstat(src.fileno())) != _stable(before):
            raise base.SnapshotVerificationError("GGUF source changed at open")
        out = destination.open("xb") if destination is not None else None
        try:
            count = 0
            while chunk := src.read(1024 * 1024):
                count += len(chunk)
                if count > before.st_size:
                    raise base.SnapshotVerificationError("GGUF source grew")
                digest.update(chunk)
                if out is not None:
                    out.write(chunk)
                if path.name != "model.gguf":
                    payload.extend(chunk)
            if count != before.st_size or _stable(os.fstat(src.fileno())) != _stable(before) or _stable(path.lstat()) != _stable(before):
                raise base.SnapshotVerificationError("GGUF source changed during read")
            if out is not None:
                out.flush()
                os.fsync(out.fileno())
        finally:
            if out is not None:
                out.close()
    return bytes(payload), digest.hexdigest()


def expected_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {"token_embd.weight": (4096, 151936), "output.weight": (4096, 151936), "output_norm.weight": (4096,)}
    for i in range(36):
        prefix = f"blk.{i}."
        for name in ("attn_norm", "ffn_norm"):
            shapes[prefix + name + ".weight"] = (4096,)
        for name in ("attn_q", "attn_output"):
            shapes[prefix + name + ".weight"] = (4096, 4096)
        for name in ("attn_k", "attn_v"):
            shapes[prefix + name + ".weight"] = (4096, 1024)
        for name in ("attn_q_norm", "attn_k_norm"):
            shapes[prefix + name + ".weight"] = (128,)
        for name in ("ffn_gate", "ffn_up"):
            shapes[prefix + name + ".weight"] = (4096, 12288)
        shapes[prefix + "ffn_down.weight"] = (12288, 4096)
    return shapes


def parse_gguf(path: Path, *, verify_payload: bool = True) -> dict[str, Any]:
    """Bounded GGUF v3 metadata, exact tensor shapes, disjoint in-file spans."""
    size = path.stat().st_size
    with path.open("rb") as src:
        def take(n):
            if n < 0 or n > 32 * 1024 * 1024 or src.tell() + n > 64 * 1024 * 1024:
                raise base.SnapshotVerificationError("GGUF metadata exceeds bounds")
            data = src.read(n)
            if len(data) != n:
                raise base.SnapshotVerificationError("Truncated GGUF metadata")
            return data

        def number(fmt):
            return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

        def string():
            return take(number("Q")).decode("utf-8", errors="strict")

        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind, depth=0):
            if kind in formats:
                result = number(formats[kind])
                if isinstance(result, float) and not math.isfinite(result):
                    raise base.SnapshotVerificationError("Nonfinite GGUF metadata")
                return result
            if kind == 8:
                return string()
            if kind == 9 and depth == 0:
                subtype, count = number("I"), number("Q")
                if count > 2_000_000:
                    raise base.SnapshotVerificationError("GGUF array exceeds bounds")
                return [value(subtype, depth + 1) for _ in range(count)]
            raise base.SnapshotVerificationError("Unsupported GGUF metadata type")

        if take(4) != b"GGUF" or number("I") != 3:
            raise base.SnapshotVerificationError("Expected GGUF v3")
        tensors, entries = number("Q"), number("Q")
        if not 1 <= tensors <= 1024 or not 1 <= entries <= 1024:
            raise base.SnapshotVerificationError("GGUF inventory exceeds bounds")
        metadata = {}
        for _ in range(entries):
            key = string()
            if key in metadata:
                raise base.SnapshotVerificationError("Duplicate GGUF metadata")
            metadata[key] = value(number("I"))
        inventory = {}
        blocks = {0: (1, 4), 1: (1, 2), 12: (256, 144), 14: (256, 210)}
        for _ in range(tensors):
            name, dimensions = string(), number("I")
            if name in inventory or not 1 <= dimensions <= 4:
                raise base.SnapshotVerificationError("Invalid GGUF tensor identity")
            shape = tuple(number("Q") for _ in range(dimensions))
            kind, offset = number("I"), number("Q")
            if kind not in blocks or any(not 1 <= n <= 262144 for n in shape):
                raise base.SnapshotVerificationError("Unsupported GGUF tensor type or shape")
            block, width = blocks[kind]
            if shape[0] % block:
                raise base.SnapshotVerificationError("GGUF tensor violates block alignment")
            inventory[name] = {"shape": shape, "type": kind, "offset": offset, "nbytes": math.prod(shape) // block * width}
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise base.SnapshotVerificationError("Invalid GGUF alignment")
        start = (src.tell() + alignment - 1) // alignment * alignment
        if {name: row["shape"] for name, row in inventory.items()} != expected_tensor_shapes():
            raise base.SnapshotVerificationError("GGUF tensor inventory differs from Qwen3-8B")
        previous = 0
        for row in sorted(inventory.values(), key=lambda item: item["offset"]):
            if row["offset"] % alignment or row["offset"] < previous:
                raise base.SnapshotVerificationError("GGUF tensor spans overlap or are misaligned")
            previous = row["offset"] + row["nbytes"]
        if verify_payload and start + previous != size:
            raise base.SnapshotVerificationError("GGUF payload truncated or has trailing bytes")
        expected = {"general.architecture": "qwen3", "general.file_type": 15, "qwen3.block_count": 36,
                    "qwen3.embedding_length": 4096, "qwen3.feed_forward_length": 12288,
                    "qwen3.attention.head_count": 32, "qwen3.attention.head_count_kv": 8,
                    "tokenizer.ggml.eos_token_id": 151645}
        if any(metadata.get(key) != val for key, val in expected.items()):
            raise base.SnapshotVerificationError("GGUF Qwen3-8B architecture or quantization differs")
        return {"metadata": metadata, "tensors": inventory, "data_offset": start, "file_size": size}


def _inventory(directory: Path, manifest=True):
    expected = REQUIRED_FILES | {"snapshot.json"} if manifest else REQUIRED_FILES
    if {p.name for p in directory.iterdir()} != expected:
        raise base.SnapshotVerificationError("GGUF snapshot contains missing or unlisted files")


def _manifest(payload):
    raw = base._json_object(payload, "snapshot.json")
    if (raw.keys() != base._MANIFEST_FIELDS or raw["schema_version"] != SNAPSHOT_SCHEMA_VERSION
            or raw["model_id"] != MODEL_ID or raw["model_revision"] != MODEL_REVISION
            or type(raw["context_window"]) is not int or not 1 <= raw["context_window"] <= 32768
            or raw["files"] != SOURCE_FILE_PINS or raw["runtime_versions"] != RUNTIME_VERSION_PINS
            or raw["special_tokens"] != SPECIAL_TOKENS):
        raise base.SnapshotVerificationError("GGUF manifest differs from pinned profile")
    return base.SnapshotManifest(raw["schema_version"], raw["model_id"], raw["model_revision"],
        raw["context_window"], tuple(sorted(raw["runtime_versions"].items())),
        tuple(sorted(raw["special_tokens"].items())), tuple(sorted(raw["files"].items())))


def _identity(manifest, hashes, versions):
    return ModelInputIdentity(
        model_digest=base._canonical_digest({"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "metadata_revision": HF_MODEL_REVISION, "files": hashes, "context_window": manifest.context_window,
            "weights": "official-quantized-artifact-not-bf16-equivalence"}),
        tokenizer_digest=base._canonical_digest({"tokenizer": hashes["tokenizer.json"], "special_tokens": SPECIAL_TOKENS}),
        chat_template_digest=hashes["chat_template.jinja"],
        runtime_identity="dml-qwen3-gguf-model-input-runtime-v1:" + base._canonical_digest({
            "versions": versions, "python": platform.python_version(), "device": "cpu",
            "format": "GGUFv3-Q4_K_M", "weights": GGUF_SHA256}),
        model_window_tokens=manifest.context_window)


class VerifiedQwen3GGUFSnapshot(base.VerifiedSnapshot):
    @property
    def gguf_metadata(self):
        return parse_gguf(self.path / "model.gguf")["metadata"]

    @property
    def tensor_inventory(self):
        return parse_gguf(self.path / "model.gguf")["tensors"]

    def validate_integrity(self):
        _inventory(self.path, manifest=False)
        observed = {}
        for name, expected in self.manifest.files:
            content, observed[name] = read_regular(self.path / name)
            if observed[name] != expected:
                raise base.SnapshotVerificationError("Verified GGUF artifact changed")
            if name == "config.json" and content != self._config_bytes:
                raise base.SnapshotVerificationError("Verified GGUF config cache changed")
            if name == "chat_template.jinja" and content != self._chat_template.encode():
                raise base.SnapshotVerificationError("Verified GGUF template cache changed")
        if self.identity != _identity(self.manifest, observed, runtime_versions()):
            raise base.SnapshotVerificationError("Verified GGUF runtime identity changed")


def verify_qwen3_gguf_snapshot(directory: str | Path) -> VerifiedQwen3GGUFSnapshot:
    temporary = None
    try:
        source, before = _directory(directory)
        _inventory(source)
        manifest_bytes, _ = read_regular(source / "snapshot.json")
        manifest = _manifest(manifest_bytes)
        versions = runtime_versions()
        temporary = tempfile.TemporaryDirectory(prefix="dml-qwen3-gguf-input-")
        target = Path(temporary.name)
        contents, hashes = {}, {}
        for name, expected in manifest.files:
            contents[name], hashes[name] = read_regular(source / name, target / name)
            if hashes[name] != expected:
                raise base.SnapshotVerificationError("Pinned GGUF artifact hash differs")
        if read_regular(source / "snapshot.json")[0] != manifest_bytes or _stable(source.lstat()) != _stable(before):
            raise base.SnapshotVerificationError("GGUF source directory changed")
        _inventory(source)
        config = base._json_object(contents["config.json"], "config.json")
        check_config(config)
        parse_gguf(target / "model.gguf")
        tokenizer_config = base._json_object(contents["tokenizer_config.json"], "tokenizer_config.json")
        if not isinstance(tokenizer_config.get("chat_template"), str):
            raise base.SnapshotVerificationError("Original tokenizer template missing")
        template = contents["chat_template.jinja"].decode()
        for name in REQUIRED_FILES:
            (target / name).chmod(0o400)
        target.chmod(0o500)
        return VerifiedQwen3GGUFSnapshot(temporary, manifest, _identity(manifest, hashes, versions),
            contents["config.json"], template, _verification_token=base._VERIFICATION_TOKEN)
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        raise


def check_config(config):
    expected = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "num_hidden_layers": 36,
        "hidden_size": 4096, "intermediate_size": 12288, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": False,
        "max_position_embeddings": 40960, "rope_theta": 1000000, "eos_token_id": 151645}
    if any(type(config.get(k)) is not type(v) or config[k] != v for k, v in expected.items()):
        raise base.SnapshotVerificationError("Qwen3-8B metadata architecture differs")
