"""Verify a pinned local GPT-2 snapshot before any tokenizer or model loader runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from ..contracts.model_input import ModelInputIdentity, SUPPORTED_CHAT_TEMPLATE


SNAPSHOT_SCHEMA_VERSION = "dml-model-snapshot-v1"
REQUIRED_FILES = frozenset({"config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"})
RUNTIME_VERSION_PINS = MappingProxyType({
    "torch": "2.8.0+cpu", "transformers": "4.56.2", "tokenizers": "0.22.0",
    "safetensors": "0.6.2", "jinja2": "3.1.6",
})
_SPECIAL_TOKENS = frozenset({"bos_token", "eos_token", "unk_token", "pad_token"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = frozenset({
    "schema_version", "model_id", "model_revision", "context_window", "runtime_versions",
    "special_tokens", "files",
})
_VERIFICATION_TOKEN = object()


class SnapshotVerificationError(ValueError):
    """The local snapshot does not match the admitted immutable artifacts."""


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    schema_version: str
    model_id: str
    model_revision: str
    context_window: int
    runtime_versions: tuple[tuple[str, str], ...]
    special_tokens: tuple[tuple[str, str | None], ...]
    files: tuple[tuple[str, str], ...]


class VerifiedSnapshot:
    """Own a private verified copy for the entire model-loader lifetime."""

    def __init__(
        self, temporary: tempfile.TemporaryDirectory[str], manifest: SnapshotManifest,
        identity: ModelInputIdentity, config_bytes: bytes, chat_template: str,
        *, _verification_token: object = None,
    ) -> None:
        if _verification_token is not _VERIFICATION_TOKEN:
            raise SnapshotVerificationError("Use verify_local_snapshot to construct a verified snapshot")
        self._temporary = temporary
        self._path = Path(temporary.name)
        self._manifest = manifest
        self._identity = identity
        self._config_bytes = config_bytes
        self._chat_template = chat_template
        self._closed = False

    @property
    def path(self) -> Path:
        if self._closed:
            raise SnapshotVerificationError("Verified snapshot is closed")
        return self._path

    @property
    def manifest(self) -> SnapshotManifest:
        return self._manifest

    @property
    def identity(self) -> ModelInputIdentity:
        return self._identity

    @property
    def config(self) -> dict[str, Any]:
        return _json_object(self._config_bytes, "config.json")

    @property
    def special_tokens(self) -> dict[str, str | None]:
        return dict(self._manifest.special_tokens)

    @property
    def chat_template(self) -> str:
        return self._chat_template

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self._closed = True

    def validate_integrity(self) -> None:
        """Recheck the owned artifacts immediately before a concrete loader uses them."""
        try:
            _inventory(self.path, include_manifest=False)
            observed = {}
            for name, expected in self._manifest.files:
                payload, observed[name] = _read_regular(self.path / name)
                if observed[name] != expected:
                    raise SnapshotVerificationError(f"Verified artifact digest differs: {name}")
                if name == "config.json" and payload != self._config_bytes:
                    raise SnapshotVerificationError("Verified configuration bytes differ")
                if name == "chat_template.jinja" and payload != self._chat_template.encode("utf-8"):
                    raise SnapshotVerificationError("Verified chat template bytes differ")
            versions = _check_runtime(self._manifest)
            if self._identity != _build_identity(self._manifest, observed, versions):
                raise SnapshotVerificationError("Verified snapshot identity differs")
        except OSError as exc:
            raise SnapshotVerificationError("Unable to read verified snapshot") from exc

    def __enter__(self) -> VerifiedSnapshot:
        try:
            self.validate_integrity()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _json_object(payload: bytes, filename: str) -> dict[str, Any]:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotVerificationError(f"Duplicate JSON key in {filename}")
            result[key] = value
        return result

    def finite_number(_: str) -> None:
        raise SnapshotVerificationError(f"Non-finite JSON number in {filename}")

    def validate_json(value: Any, depth: int = 0) -> None:
        if depth > 64:
            raise SnapshotVerificationError(f"JSON nesting exceeds the snapshot limit in {filename}")
        if type(value) is str:
            value.encode("utf-8")
        elif type(value) is float and not math.isfinite(value):
            raise SnapshotVerificationError(f"Non-finite JSON number in {filename}")
        elif type(value) is dict:
            for key, child in value.items():
                validate_json(key, depth + 1)
                validate_json(child, depth + 1)
        elif type(value) is list:
            for child in value:
                validate_json(child, depth + 1)

    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_keys, parse_constant=finite_number)
        validate_json(parsed)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SnapshotVerificationError(f"Invalid JSON in {filename}") from exc
    if type(parsed) is not dict:
        raise SnapshotVerificationError(f"Expected JSON object in {filename}")
    return parsed


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_manifest(payload: bytes) -> SnapshotManifest:
    raw = _json_object(payload, "snapshot.json")
    if raw.keys() != _MANIFEST_FIELDS or raw["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotVerificationError("Unknown or incomplete snapshot manifest schema")
    for name in ("model_id", "model_revision"):
        value = raw[name]
        if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 1024:
            raise SnapshotVerificationError(f"Invalid snapshot {name}")
    window = raw["context_window"]
    if type(window) is not int or not 1 <= window < 2**31:
        raise SnapshotVerificationError("Invalid snapshot context_window")
    files = raw["files"]
    if type(files) is not dict or files.keys() != REQUIRED_FILES:
        raise SnapshotVerificationError("Snapshot files must name exactly the admitted artifacts")
    if any(type(value) is not str or not _HASH.fullmatch(value) for value in files.values()):
        raise SnapshotVerificationError("Snapshot artifact hashes must be lowercase SHA-256")
    versions = raw["runtime_versions"]
    if type(versions) is not dict or versions != dict(RUNTIME_VERSION_PINS):
        raise SnapshotVerificationError("Snapshot runtime versions are outside the pinned runtime")
    tokens = raw["special_tokens"]
    if type(tokens) is not dict or tokens.keys() != _SPECIAL_TOKENS:
        raise SnapshotVerificationError("Snapshot must declare the four supported special tokens")
    if any(value is not None and (type(value) is not str or not value) for value in tokens.values()):
        raise SnapshotVerificationError("Invalid snapshot special token")
    return SnapshotManifest(
        schema_version=raw["schema_version"], model_id=raw["model_id"], model_revision=raw["model_revision"],
        context_window=window, runtime_versions=tuple(sorted(versions.items())),
        special_tokens=tuple(sorted(tokens.items())), files=tuple(sorted(files.items())),
    )


def _check_runtime(manifest: SnapshotManifest) -> dict[str, str]:
    observed = {}
    for name, expected in manifest.runtime_versions:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SnapshotVerificationError(f"Missing pinned runtime package: {name}") from exc
        if observed[name] != expected:
            raise SnapshotVerificationError(f"Installed runtime version differs: {name}")
    return observed


def _inventory(directory: Path, *, include_manifest: bool = True) -> None:
    expected = REQUIRED_FILES | {"snapshot.json"} if include_manifest else REQUIRED_FILES
    if {entry.name for entry in directory.iterdir()} != expected:
        raise SnapshotVerificationError("Snapshot contains missing or unlisted files")
    for name in expected:
        if not stat.S_ISREG((directory / name).lstat().st_mode):
            raise SnapshotVerificationError("Snapshot artifacts must be regular files without symlinks")


def _directory(path: str | Path) -> tuple[Path, os.stat_result]:
    directory = Path(path).absolute()
    if ".." in directory.parts:
        raise SnapshotVerificationError("Snapshot directory must not contain traversal")
    for ancestor in (directory, *directory.parents):
        observed = ancestor.lstat()
        if not stat.S_ISDIR(observed.st_mode):
            raise SnapshotVerificationError("Snapshot directory must not contain symlinks")
    observed = directory.stat()
    _inventory(directory)
    return directory, observed


def _read_regular(source: Path, destination: Path | None = None) -> tuple[bytes, str]:
    """Copy exactly the opened regular file's bytes; never execute or deserialize it."""
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SnapshotVerificationError("Snapshot artifacts must be regular files without symlinks")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    captured = bytearray()
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SnapshotVerificationError("Snapshot artifact changed while opening")
        output = destination.open("xb") if destination is not None else None
        try:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
                if source.name != "model.safetensors":
                    captured.extend(chunk)
            after = source.lstat()

            def stable(result: os.stat_result) -> tuple[int, ...]:
                return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns, result.st_ctime_ns

            if stable(before) != stable(after) or stable(opened) != stable(os.fstat(handle.fileno())):
                raise SnapshotVerificationError("Snapshot artifact changed while copying")
        finally:
            if output is not None:
                output.close()
    return bytes(captured), digest.hexdigest()


def _check_config(config: Mapping[str, Any], manifest: SnapshotManifest) -> None:
    if config.get("model_type") != "gpt2" or config.get("architectures") != ["GPT2LMHeadModel"]:
        raise SnapshotVerificationError("Only the built-in GPT2LMHeadModel architecture is admitted")
    if any(name in config for name in ("auto_map", "custom_pipelines", "quantization_config")):
        raise SnapshotVerificationError("Custom code and quantized model configuration are excluded")
    if type(config.get("n_positions")) is not int or config["n_positions"] != manifest.context_window:
        raise SnapshotVerificationError("Snapshot context_window differs from model position capacity")
    if "n_ctx" in config and (type(config["n_ctx"]) is not int or config["n_ctx"] != manifest.context_window):
        raise SnapshotVerificationError("Snapshot context_window differs from model n_ctx")
    for name in ("vocab_size", "n_embd", "n_layer", "n_head"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise SnapshotVerificationError(f"Invalid GPT-2 configuration: {name}")
    if config["n_embd"] % config["n_head"]:
        raise SnapshotVerificationError("GPT-2 embedding width must divide into attention heads")


def _build_identity(manifest: SnapshotManifest, observed: Mapping[str, str], versions: Mapping[str, str]) -> ModelInputIdentity:
    return ModelInputIdentity(
        model_digest=_canonical_digest({
            "model_id": manifest.model_id, "model_revision": manifest.model_revision,
            "context_window": manifest.context_window,
            "files": {name: observed[name] for name in ("config.json", "model.safetensors")},
        }),
        tokenizer_digest=_canonical_digest({
            "files": {"tokenizer.json": observed["tokenizer.json"]},
            "special_tokens": dict(manifest.special_tokens),
        }),
        chat_template_digest=observed["chat_template.jinja"],
        runtime_identity="dml-model-input-runtime-v1:" + _canonical_digest({
            "versions": versions, "python": platform.python_version(), "device": "cpu",
            "dtype": "float32", "model_class": "GPT2LMHeadModel", "tokenizer_class": "PreTrainedTokenizerFast",
        }),
        model_window_tokens=manifest.context_window,
    )


def verify_local_snapshot(directory: str | Path) -> VerifiedSnapshot:
    """Return a private verified copy; the caller must close it or use ``with``.

    The source is data from a trusted provenance, never a repository to execute.
    Model loading only sees the private copy whose bytes established identity.
    """
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        source, initial_directory = _directory(directory)
        manifest_bytes, _ = _read_regular(source / "snapshot.json")
        manifest = _parse_manifest(manifest_bytes)
        versions = _check_runtime(manifest)
        temporary = tempfile.TemporaryDirectory(prefix="dml-model-input-")
        target = Path(temporary.name)
        contents: dict[str, bytes] = {}
        observed: dict[str, str] = {}
        for name, expected in manifest.files:
            contents[name], observed[name] = _read_regular(source / name, target / name)
            if observed[name] != expected:
                raise SnapshotVerificationError(f"Snapshot artifact digest differs: {name}")
        final_manifest, _ = _read_regular(source / "snapshot.json")
        if final_manifest != manifest_bytes:
            raise SnapshotVerificationError("Snapshot manifest changed while copying")
        _inventory(source)
        final_directory = source.lstat()
        if (final_directory.st_dev, final_directory.st_ino) != (initial_directory.st_dev, initial_directory.st_ino):
            raise SnapshotVerificationError("Snapshot directory changed while copying")
        config = _json_object(contents["config.json"], "config.json")
        _check_config(config, manifest)
        _json_object(contents["tokenizer.json"], "tokenizer.json")
        template = contents["chat_template.jinja"]
        if template != SUPPORTED_CHAT_TEMPLATE.encode("utf-8"):
            raise SnapshotVerificationError("Snapshot chat template is outside the admitted protocol")
        identity = _build_identity(manifest, observed, versions)
        for name in REQUIRED_FILES:
            (target / name).chmod(0o400)
        target.chmod(0o500)
        return VerifiedSnapshot(
            temporary, manifest, identity, contents["config.json"], template.decode("utf-8"),
            _verification_token=_VERIFICATION_TOKEN,
        )
    except BaseException as exc:
        if temporary is not None:
            temporary.cleanup()
        if isinstance(exc, (OSError, UnicodeError)):
            raise SnapshotVerificationError("Unable to verify local model snapshot") from exc
        raise
