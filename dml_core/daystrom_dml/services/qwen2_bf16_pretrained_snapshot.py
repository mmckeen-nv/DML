"""Finalize exact original Qwen2 BF16 shards without another raw payload copy.

The caller owns a fresh staged package: bundle/ contains the five original
runtime files; LICENSE and tokenizer_config.json are adjacent. Failure retains
every acquired source file, including the research license. Only a fully verified
package is atomically renamed to a fresh destination using Linux no-replace
publication.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Any

from . import qwen2_bf16_model_snapshot as snapshot
from .model_input_snapshot import RUNTIME_VERSION_PINS, _json_object
from .pretrained_snapshot import PretrainedSnapshotError, _json_bytes

PREPARER_VERSION = "dml-pretrained-qwen2-original-shards-bf16-v1"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SOURCE_FILE_PINS = MappingProxyType({
    "config.json": (661, "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b"),
    "tokenizer.json": (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "LICENSE": (7388, "ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9"),
    "model.safetensors.index.json": (35581, "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56"),
    "model-00001-of-00002.safetensors": (3968658944, "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2"),
    "model-00002-of-00002.safetensors": (2203268048, "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d"),
})
_RUNTIME_SOURCE_FILES = snapshot.REQUIRED_FILES - {"chat_template.jinja"}


@dataclass(frozen=True)
class _SourceProfile:
    model_id: str
    revision: str
    files: tuple[tuple[str, int, str], ...]


def _source_path(stage: Path, name: str) -> Path:
    return stage / "bundle" / name if name in _RUNTIME_SOURCE_FILES else stage / name


def _verify_sources(stage: Path, profile: _SourceProfile) -> dict[str, str]:
    observed = {}
    for name, size, expected in profile.files:
        path = _source_path(stage, name)
        if path.lstat().st_size != size:
            raise PretrainedSnapshotError("Pinned Qwen2 BF16 source size differs")
        _, observed[name] = snapshot.read_regular(path)
        if observed[name] != expected:
            raise PretrainedSnapshotError("Pinned Qwen2 BF16 source digest differs")
    return observed


def _publish_directory(source: Path, destination: Path, expected_identity: tuple[int, int]) -> None:
    """Atomic fail-closed no-clobber publication; never emulate with check+rename."""
    if sys.platform != "linux":
        raise PretrainedSnapshotError("Qwen2 BF16 atomic package publication requires Linux renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    operation = getattr(library, "renameat2", None)
    if operation is None:
        raise PretrainedSnapshotError("Qwen2 BF16 atomic no-replace publication is unavailable")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    source_parent, source_info = snapshot._directory(source.parent)
    target_parent, target_info = snapshot._directory(destination.parent)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    source_fd, target_fd = os.open(source_parent, flags), None
    try:
        target_fd = os.open(target_parent, flags)
        for descriptor, expected in ((source_fd, source_info), (target_fd, target_info)):
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
                raise PretrainedSnapshotError("Qwen2 BF16 publication parent changed")
        observed = os.stat(source.name, dir_fd=source_fd, follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != expected_identity:
            raise PretrainedSnapshotError("Qwen2 BF16 staged package identity changed")
        if operation(source_fd, os.fsencode(source.name), target_fd, os.fsencode(destination.name), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        os.fsync(target_fd)
        os.fsync(source_fd)
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def prepare_qwen2_bf16_snapshot(staged_directory: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify one private copy, retain exact source files, and publish by rename."""
    profile = _SourceProfile(MODEL_ID, MODEL_REVISION,
        tuple((name, size, digest) for name, (size, digest) in SOURCE_FILE_PINS.items()))
    stage, initial = snapshot._directory(staged_directory)
    bundle, bundle_initial = snapshot._directory(stage / "bundle")
    target = Path(destination).absolute()
    snapshot._directory(target.parent)
    if ".." in target.parts or target == stage or target.is_relative_to(stage):
        raise PretrainedSnapshotError("Invalid Qwen2 BF16 preparation destination")
    if target.exists() or target.is_symlink():
        raise PretrainedSnapshotError("Qwen2 BF16 preparation destination already exists")
    if ({entry.name for entry in stage.iterdir()} != {"bundle", "LICENSE", "tokenizer_config.json"}
            or {entry.name for entry in bundle.iterdir()} != _RUNTIME_SOURCE_FILES):
        raise PretrainedSnapshotError("Missing or unlisted Qwen2 BF16 staged source files")
    _verify_sources(stage, profile)
    config_bytes, _ = snapshot.read_regular(bundle / "config.json")
    config = _json_object(config_bytes, "config.json")
    snapshot.check_config(config, 32768)
    generated: list[tuple[Path, tuple[int, int]]] = []
    published = False

    def write_owned(path: Path, payload: bytes) -> None:
        # Register the created inode before writing so a short write/failure can
        # clean only our own generated metadata, never an acquired source file.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        info = os.fstat(fd)
        generated.append((path, (info.st_dev, info.st_ino)))
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    try:
        write_owned(bundle / "chat_template.jinja", snapshot.QWEN2_BF16_CHAT_TEMPLATE.encode())
        manifest: dict[str, Any] = {"schema_version": snapshot.SNAPSHOT_SCHEMA_VERSION, "model_id": profile.model_id,
            "model_revision": profile.revision, "context_window": 32768,
            "runtime_versions": dict(RUNTIME_VERSION_PINS), "special_tokens": snapshot.SPECIAL_TOKENS,
            "files": {name: snapshot.read_regular(bundle / name)[1] for name in snapshot.REQUIRED_FILES}}
        write_owned(bundle / "snapshot.json", _json_bytes(manifest))
        # This is the only private weight copy. It closes before publication.
        verified = snapshot.verify_qwen2_bf16_snapshot(bundle)
        with verified:
            state = snapshot.load_qwen2_bf16_state(verified.path, verified.config, verified.index)
            records = snapshot.validate_state(state, verified.config)
            del state
            verified.validate_integrity()
            identity = verified.identity.to_payload()
        _verify_sources(stage, profile)
        snapshot._inventory(bundle)
        if ({entry.name for entry in stage.iterdir()} != {"bundle", "LICENSE", "tokenizer_config.json"}
                or (stage.lstat().st_dev, stage.lstat().st_ino) != (initial.st_dev, initial.st_ino)
                or (bundle.lstat().st_dev, bundle.lstat().st_ino) != (bundle_initial.st_dev, bundle_initial.st_ino)):
            raise PretrainedSnapshotError("Qwen2 BF16 staging inventory or identity changed")
        bundle_files = {name: {"bytes": (bundle / name).lstat().st_size,
            "sha256": snapshot.read_regular(bundle / name)[1]} for name in snapshot.REQUIRED_FILES | {"snapshot.json"}}
        if (any(bundle_files[name]["sha256"] != digest for name, digest in manifest["files"].items())
                or bundle_files["snapshot.json"]["sha256"] != hashlib.sha256(_json_bytes(manifest)).hexdigest()
                or any(bundle_files[name] != {"bytes": size, "sha256": digest}
                       for name, size, digest in profile.files if name in _RUNTIME_SOURCE_FILES)):
            raise PretrainedSnapshotError("Final Qwen2 BF16 bundle differs from frozen source or manifest")
        provenance = {"schema_version": PREPARER_VERSION, "complete": True, "model_id": profile.model_id,
            "source_revision": profile.revision, "normalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_files": {name: {"bytes": size, "sha256": digest,
                "source_url": f"https://huggingface.co/{profile.model_id}/resolve/{profile.revision}/{name}"}
                for name, size, digest in profile.files}, "learned_tensors": records, "omitted_tensors": [],
            "dtype_conversion": "none-bfloat16-original-shards-v1", "config_changes": {},
            "original_shards_retained": True, "source_storage": "exact-original-shards-and-index-v1",
            "license": {"name": "Qwen Research License Agreement", "source_file": "LICENSE",
                "source_revision": profile.revision, "usage": "research-evaluation-only"},
            "tied_head": {"source": "model.embed_tokens.weight", "target": "lm_head.weight",
                "source_head_present": False, "source_embedding_retained": True,
                "materialization": "loader-alias-same-storage"},
            "template": "dml-qwen2-bf16-lossless-chatml-v1", "bundle_files": bundle_files,
            "model_identity": identity, "trained_model": True, "instruction_tuned": True, "live_qualified": False}
        write_owned(stage / "provenance.json", _json_bytes(provenance))
        for directory in (bundle, stage):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _publish_directory(stage, target, (initial.st_dev, initial.st_ino))
        published = True
        return provenance
    finally:
        if not published and stage.exists():
            for path, identity in reversed(generated):
                try:
                    observed = path.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == identity:
                    path.unlink()
