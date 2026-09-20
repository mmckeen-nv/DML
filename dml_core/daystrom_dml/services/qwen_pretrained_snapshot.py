"""Offline, exact-source Qwen2.5-1.5B-Instruct BF16-to-F32 preparation.

This single-model preparer accepts no downloaded code, pickle or caller-defined
trust pins. Conversion preserves every finite learned value exactly. Only the
config dtype declaration changes; the safe full-field template is separately
versioned. The tied output head remains an explicit loader alias.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .model_input_snapshot import RUNTIME_VERSION_PINS, _json_object, _read_regular
from .pretrained_snapshot import PretrainedSnapshotError, _directory, _hash_file, _json_bytes, _write
from .qwen_model_snapshot import (
    QWEN_CHAT_TEMPLATE, SNAPSHOT_SCHEMA_VERSION, SPECIAL_TOKENS,
    check_config, expected_shapes, verify_qwen_snapshot,
)

PREPARER_VERSION = "dml-pretrained-qwen2-instruct-v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SOURCE_FILE_PINS = {
    "config.json": (660, "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"),
    "tokenizer.json": (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "model.safetensors": (3087467144, "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"),
}


def tensor_digest(tensor: Any) -> str:
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def normalize_state(state: dict[str, Any], config: dict[str, Any]):
    """Validate exact learned coverage and byte-exact BF16 round trips."""
    import torch
    shapes = expected_shapes(config)
    if state.keys() != shapes.keys():
        raise PretrainedSnapshotError("Qwen learned tensor coverage differs")
    transformed, records = {}, []
    for name in sorted(shapes):
        source = state[name]
        if (not isinstance(source, torch.Tensor) or source.device.type != "cpu"
                or source.dtype != torch.bfloat16 or tuple(source.shape) != shapes[name]
                or not bool(torch.isfinite(source).all())):
            raise PretrainedSnapshotError("Qwen source tensor shape, dtype or finite values differ")
        raw_digest = tensor_digest(source)
        target = source.to(dtype=torch.float32).contiguous()
        if tensor_digest(target.to(dtype=torch.bfloat16)) != raw_digest:
            raise PretrainedSnapshotError("Qwen BF16-to-F32 conversion is not byte-exact on round trip")
        transformed[name] = target
        records.append({"name": name, "shape": list(shapes[name]), "source_dtype": "bfloat16",
            "target_dtype": "float32", "source_sha256": raw_digest,
            "target_sha256": tensor_digest(target), "round_trip_exact": True})
    return transformed, records


def prepare_qwen_snapshot(raw_directory: str | Path, destination: str | Path) -> dict[str, Any]:
    """Prepare a new offline bundle; a complete provenance record is mandatory."""
    from safetensors.torch import load_file, save_file
    raw = _directory(raw_directory)
    target = Path(destination).absolute()
    parent = _directory(target.parent)
    if target.name in {"", ".", ".."} or ".." in target.parts:
        raise PretrainedSnapshotError("Invalid Qwen destination")
    if {entry.name for entry in raw.iterdir()} != SOURCE_FILE_PINS.keys():
        raise PretrainedSnapshotError("Missing or unlisted Qwen raw files")
    if target.exists() or target.is_symlink():
        raise PretrainedSnapshotError("Qwen destination already exists")
    reserved = False
    try:
        with tempfile.TemporaryDirectory(prefix=".dml-qwen-pretrained-", dir=parent) as temporary:
            staging = Path(temporary)
            copied = staging / "raw"
            copied.mkdir()
            for name, (size, expected) in SOURCE_FILE_PINS.items():
                source = raw / name
                if source.lstat().st_size != size:
                    raise PretrainedSnapshotError("Pinned Qwen source size differs")
                _, observed = _read_regular(source, copied / name)
                if observed != expected:
                    raise PretrainedSnapshotError("Pinned Qwen source digest differs")
            config = _json_object((copied / "config.json").read_bytes(), "config.json")
            if config.get("torch_dtype") != "bfloat16":
                raise PretrainedSnapshotError("Pinned Qwen source dtype differs")
            config["torch_dtype"] = "float32"
            check_config(config, 32768)
            state = load_file(str(copied / "model.safetensors"), device="cpu")
            transformed, records = normalize_state(state, config)
            bundle = staging / "bundle"
            bundle.mkdir()
            _write(bundle / "config.json", _json_bytes(config))
            _write(bundle / "tokenizer.json", (copied / "tokenizer.json").read_bytes())
            _write(bundle / "chat_template.jinja", QWEN_CHAT_TEMPLATE.encode())
            save_file(transformed, str(bundle / "model.safetensors"), metadata={"format": "pt"})
            restored = load_file(str(bundle / "model.safetensors"), device="cpu")
            if restored.keys() != transformed.keys() or any(
                tensor_digest(restored[record["name"]]) != record["target_sha256"] for record in records
            ):
                raise PretrainedSnapshotError("Serialized Qwen learned tensors differ")
            manifest = {"schema_version": SNAPSHOT_SCHEMA_VERSION, "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "context_window": 32768,
                "runtime_versions": dict(RUNTIME_VERSION_PINS), "special_tokens": SPECIAL_TOKENS,
                "files": {path.name: _hash_file(path) for path in bundle.iterdir()}}
            _write(bundle / "snapshot.json", _json_bytes(manifest))
            with verify_qwen_snapshot(bundle) as verified:
                identity = verified.identity.to_payload()
            provenance = {"schema_version": PREPARER_VERSION, "complete": True,
                "model_id": MODEL_ID, "source_revision": MODEL_REVISION,
                "normalizer_sha256": _hash_file(Path(__file__)),
                "source_files": {name: {"bytes": size, "sha256": digest,
                    "source_url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"}
                    for name, (size, digest) in SOURCE_FILE_PINS.items()},
                "learned_tensors": records, "omitted_tensors": [],
                "dtype_conversion": "finite-bfloat16-to-float32-exact-round-trip-v1",
                "config_changes": {"torch_dtype": {"source": "bfloat16", "target": "float32"}},
                "tied_head": {"source": "model.embed_tokens.weight", "target": "lm_head.weight",
                              "materialization": "loader-alias-same-storage"},
                "template": "dml-qwen-full-json-chatml-v1",
                "bundle_files": {path.name: {"bytes": path.stat().st_size, "sha256": _hash_file(path)}
                                 for path in bundle.iterdir()},
                "model_identity": identity, "trained_model": True, "instruction_tuned": True,
                "live_qualified": False}
            target.mkdir(mode=0o700)
            reserved = True
            bundle.rename(target / "bundle")
            _write(target / "LICENSE", (copied / "LICENSE").read_bytes())
            _write(target / "provenance.partial", _json_bytes(provenance))
            (target / "provenance.partial").rename(target / "provenance.json")
            return provenance
    except BaseException:
        if reserved:
            shutil.rmtree(target)
        raise
