"""Offline, provenance-pinned normalization of one published GPT-2 snapshot.

This preparer does not download, execute repository code or accept caller-supplied
trust pins. It preserves every learned float32 byte, adds the required state-dict
prefix and materializes the tied output head. Only the twelve verified legacy
causal masks are omitted. The existing model consumer remains unchanged.

An output is complete only when ``provenance.json`` exists. A process crash may
leave an incomplete, reserved destination; it must never be reused automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from ..contracts.model_input import SUPPORTED_CHAT_TEMPLATE
from .model_input_snapshot import (
    RUNTIME_VERSION_PINS, SNAPSHOT_SCHEMA_VERSION, _json_object, _read_regular,
    verify_local_snapshot,
)


PREPARER_VERSION = "dml-pretrained-gpt2-v1"
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
LICENSE_REVISION = "0574c5708b094bfa0b0f6dfe3fd284d9a045acd9"
SOURCE_FILE_PINS = {
    "config.json": (665, "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4"),
    "tokenizer.json": (1355256, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"),
    "model.safetensors": (548105171, "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"),
    "LICENSE.gpt2": (1403, "0dbeda4bc78823b1d67ec0ee414d8690ee4cad826bc9c147b215adc3af202436"),
}
DIMENSIONS = {"vocab_size": 50257, "n_positions": 1024, "n_embd": 768, "n_layer": 12, "n_head": 12}


class PretrainedSnapshotError(ValueError):
    """Raw provenance, tensor coverage or output preparation was rejected."""


def _directory(path: str | Path) -> Path:
    result = Path(path).absolute()
    if ".." in result.parts:
        raise PretrainedSnapshotError("Directory traversal is excluded")
    for ancestor in (result, *result.parents):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise PretrainedSnapshotError("Directories must not contain symlinks")
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _shapes(dimensions: dict[str, int]) -> dict[str, tuple[int, ...]]:
    width = dimensions["n_embd"]
    shapes = {
        "wte.weight": (dimensions["vocab_size"], width),
        "wpe.weight": (dimensions["n_positions"], width),
        "ln_f.weight": (width,), "ln_f.bias": (width,),
    }
    block = {
        "ln_1.weight": (width,), "ln_1.bias": (width,),
        "ln_2.weight": (width,), "ln_2.bias": (width,),
        "attn.c_attn.weight": (width, 3 * width), "attn.c_attn.bias": (3 * width,),
        "attn.c_proj.weight": (width, width), "attn.c_proj.bias": (width,),
        "mlp.c_fc.weight": (width, 4 * width), "mlp.c_fc.bias": (4 * width,),
        "mlp.c_proj.weight": (4 * width, width), "mlp.c_proj.bias": (width,),
    }
    for layer in range(dimensions["n_layer"]):
        shapes.update({f"h.{layer}.{name}": shape for name, shape in block.items()})
    return shapes


def _tensor_digest(tensor: Any) -> str:
    # Compare bytes, including signed zero, without numeric equality or casting.
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _normalize_state(state: dict[str, Any], dimensions: dict[str, int]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    import torch

    shapes = _shapes(dimensions)
    window = dimensions["n_positions"]
    masks = {f"h.{layer}.attn.bias" for layer in range(dimensions["n_layer"])}
    if state.keys() != shapes.keys() | masks:
        raise PretrainedSnapshotError("Raw learned tensors or legacy buffers differ from exact coverage")
    transformed: dict[str, Any] = {}
    preserved, omitted = [], []
    for name in sorted(state):
        tensor = state[name]
        shape = (1, 1, window, window) if name in masks else shapes[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all())):
            raise PretrainedSnapshotError(f"Tensor shape, device, dtype or finite values differ: {name}")
        digest = _tensor_digest(tensor)
        if name in masks:
            expected = torch.ones((window, window), dtype=torch.float32, device="cpu").tril().reshape(shape)
            if digest != _tensor_digest(expected):
                raise PretrainedSnapshotError(f"Legacy causal mask bytes differ: {name}")
            omitted.append({"source_name": name, "sha256": digest, "shape": list(shape)})
            continue
        target = "transformer." + name
        transformed[target] = tensor.detach().clone().contiguous()
        if _tensor_digest(transformed[target]) != digest:
            raise PretrainedSnapshotError(f"Learned tensor bytes changed: {name}")
        preserved.append({"source_name": name, "target_name": target,
                          "sha256": digest, "shape": list(shape), "dtype": "float32"})
    transformed["lm_head.weight"] = transformed["transformer.wte.weight"].clone()
    return transformed, preserved, omitted


def prepare_gpt2_snapshot(raw_directory: str | Path, destination: str | Path) -> dict[str, Any]:
    """Prepare the pinned trained snapshot in a new directory, entirely offline.

    Raw input must contain exactly the four pinned source files. Successful
    output contains ``bundle/`` (the five admitted files), ``LICENSE.gpt2`` and
    the final completion record ``provenance.json``. Existing outputs always
    reject, including incomplete output from an interrupted earlier attempt.
    """
    from safetensors.torch import load_file, save_file

    raw = _directory(raw_directory)
    target = Path(destination).absolute()
    parent = _directory(target.parent)
    if target.name in {"", ".", ".."} or ".." in target.parts:
        raise PretrainedSnapshotError("Invalid destination")
    if {entry.name for entry in raw.iterdir()} != SOURCE_FILE_PINS.keys():
        raise PretrainedSnapshotError("Raw snapshot contains missing or unlisted files")
    if target.exists() or target.is_symlink():
        raise PretrainedSnapshotError("Destination already exists; incomplete output must be inspected")
    reserved = False
    try:
        with tempfile.TemporaryDirectory(prefix=".dml-pretrained-", dir=parent) as temporary:
            staging = Path(temporary)
            copied = staging / "raw"
            copied.mkdir()
            for name, (size, expected) in SOURCE_FILE_PINS.items():
                source = raw / name
                if source.lstat().st_size != size:
                    raise PretrainedSnapshotError(f"Pinned source size differs: {name}")
                _, digest = _read_regular(source, copied / name)
                if digest != expected:
                    raise PretrainedSnapshotError(f"Pinned source digest differs: {name}")
            config = _json_object((copied / "config.json").read_bytes(), "config.json")
            if any(type(config.get(name)) is not int or config[name] != value for name, value in DIMENSIONS.items()):
                raise PretrainedSnapshotError("Pinned GPT-2 dimensions differ")
            state = load_file(str(copied / "model.safetensors"), device="cpu")
            transformed, preserved, omitted = _normalize_state(state, DIMENSIONS)
            bundle = staging / "bundle"
            bundle.mkdir()
            for name in ("config.json", "tokenizer.json"):
                _write(bundle / name, (copied / name).read_bytes())
            _write(bundle / "chat_template.jinja", SUPPORTED_CHAT_TEMPLATE.encode("utf-8"))
            save_file(transformed, str(bundle / "model.safetensors"), metadata={"format": "pt"})
            restored = load_file(str(bundle / "model.safetensors"), device="cpu")
            if restored.keys() != transformed.keys() or any(
                _tensor_digest(restored[name]) != _tensor_digest(value) for name, value in transformed.items()
            ):
                raise PretrainedSnapshotError("Serialized learned tensor bytes changed")
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION, "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "context_window": DIMENSIONS["n_positions"],
                "runtime_versions": dict(RUNTIME_VERSION_PINS),
                "special_tokens": {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
                                   "unk_token": "<|endoftext|>", "pad_token": "<|endoftext|>"},
                "files": {path.name: _hash_file(path) for path in bundle.iterdir()},
            }
            _write(bundle / "snapshot.json", _json_bytes(manifest))
            with verify_local_snapshot(bundle) as verified:
                identity = verified.identity.to_payload()
            provenance = {
                "schema_version": PREPARER_VERSION, "complete": True,
                "model_id": MODEL_ID, "source_revision": MODEL_REVISION,
                "normalizer_sha256": _hash_file(Path(__file__)),
                "source_files": {name: {"bytes": size, "sha256": digest,
                    "source_url": (f"https://raw.githubusercontent.com/openai/gpt-2/{LICENSE_REVISION}/LICENSE"
                                   if name == "LICENSE.gpt2" else
                                   f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}")}
                    for name, (size, digest) in SOURCE_FILE_PINS.items()},
                "learned_tensors": preserved, "omitted_verified_causal_masks": omitted,
                "tied_head": {"source": "wte.weight", "target": "lm_head.weight",
                              "sha256": _tensor_digest(transformed["lm_head.weight"])},
                "bundle_files": {path.name: {"bytes": path.stat().st_size, "sha256": _hash_file(path)}
                                 for path in bundle.iterdir()},
                "model_identity": identity, "trained_model": True,
                "instruction_tuned": False, "live_qualified": False,
            }
            target.mkdir(mode=0o700)
            reserved = True
            bundle.rename(target / "bundle")
            _write(target / "LICENSE.gpt2", (copied / "LICENSE.gpt2").read_bytes())
            # The final record is the completion marker; interrupted writes do
            # not leave a partial JSON record claiming successful preparation.
            _write(target / "provenance.partial", _json_bytes(provenance))
            (target / "provenance.partial").rename(target / "provenance.json")
            return provenance
    except BaseException:
        if reserved:
            shutil.rmtree(target)
        raise
