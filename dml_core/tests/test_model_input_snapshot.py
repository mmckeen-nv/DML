"""Pure file-admission tests; native tokenizer/model execution has separate tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct

import pytest

from daystrom_dml.contracts.model_input import SUPPORTED_CHAT_TEMPLATE
from daystrom_dml.services import model_input_snapshot as snapshots
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError, verify_local_snapshot


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots.metadata, "version", lambda name: snapshots.RUNTIME_VERSION_PINS[name])
    directory = tmp_path.resolve() / "snapshot"
    directory.mkdir()
    config = {
        "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2", "n_positions": 64,
        "n_ctx": 64, "n_embd": 8, "n_head": 2, "n_layer": 1, "vocab_size": 4,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.5))
    (directory / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel"}}), encoding="utf-8")
    (directory / "chat_template.jinja").write_text(SUPPORTED_CHAT_TEMPLATE, encoding="utf-8")
    manifest = {
        "schema_version": snapshots.SNAPSHOT_SCHEMA_VERSION, "model_id": "local-fixture",
        "model_revision": "immutable-fixture-v1", "context_window": 64,
        "runtime_versions": dict(snapshots.RUNTIME_VERSION_PINS),
        "special_tokens": {"bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>"},
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in snapshots.REQUIRED_FILES},
    }
    (directory / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def update_manifest(source, update):
    path = source / "snapshot.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def replace_artifact(source, name, contents):
    (source / name).write_bytes(contents)
    update_manifest(source, lambda manifest: manifest["files"].update({name: hashlib.sha256(contents).hexdigest()}))


def test_verified_snapshot_owns_exact_bytes_and_cleans_up(source):
    with verify_local_snapshot(source) as verified:
        private = verified.path
        assert private != source
        assert {entry.name for entry in private.iterdir()} == snapshots.REQUIRED_FILES
        for name in snapshots.REQUIRED_FILES:
            assert (private / name).read_bytes() == (source / name).read_bytes()
        verified.identity.validate()
        assert verified.identity.model_window_tokens == 64
        assert verified.identity.chat_template_digest == hashlib.sha256(SUPPORTED_CHAT_TEMPLATE.encode()).hexdigest()
        assert verified.chat_template == SUPPORTED_CHAT_TEMPLATE
        config = verified.config
        config["n_positions"] = 1
        assert verified.config["n_positions"] == 64
        tokens = verified.special_tokens
        tokens["bos_token"] = "changed"
        assert verified.special_tokens["bos_token"] == "<bos>"
        with pytest.raises(FrozenInstanceError):
            verified.manifest.model_id = "changed"
    assert not private.exists()
    assert source.exists()
    with pytest.raises(SnapshotVerificationError, match="closed"):
        verified.path
    verified.close()


def test_source_mutation_after_verification_cannot_change_loaded_bytes(source):
    with verify_local_snapshot(source) as verified:
        expected = (verified.path / "model.safetensors").read_bytes()
        identity = verified.identity
        (source / "model.safetensors").write_bytes(b"changed source")
        (source / "tokenizer.json").unlink()
        verified.validate_integrity()
        assert (verified.path / "model.safetensors").read_bytes() == expected
        assert verified.identity == identity


def test_verified_copy_tampering_fails_integrity_check(source):
    with verify_local_snapshot(source) as verified:
        artifact = verified.path / "model.safetensors"
        artifact.chmod(0o600)
        artifact.write_bytes(b"different bytes")
        with pytest.raises(SnapshotVerificationError, match="digest"):
            verified.validate_integrity()


def test_verified_snapshot_cannot_be_constructed_with_caller_assertions():
    with pytest.raises(SnapshotVerificationError, match="verify_local_snapshot"):
        snapshots.VerifiedSnapshot(None, None, None, b"{}", "unverified")


def test_failed_context_entry_cleans_up_private_directory(source):
    verified = verify_local_snapshot(source)
    private = verified.path
    artifact = private / "model.safetensors"
    artifact.chmod(0o600)
    artifact.write_bytes(b"changed before context entry")
    with pytest.raises(SnapshotVerificationError, match="digest"), verified:
        pytest.fail("Changed private artifacts must not be admitted")
    assert not private.exists()


@pytest.mark.parametrize("filename", sorted(snapshots.REQUIRED_FILES))
def test_changed_artifact_bytes_fail_without_updating_pin(source, filename):
    (source / filename).write_bytes((source / filename).read_bytes() + b" ")
    with pytest.raises(SnapshotVerificationError, match="digest"):
        verify_local_snapshot(source)


@pytest.mark.parametrize("filename", sorted(snapshots.REQUIRED_FILES | {"snapshot.json"}))
def test_missing_required_file_rejects(source, filename):
    (source / filename).unlink()
    with pytest.raises(SnapshotVerificationError, match="missing or unlisted"):
        verify_local_snapshot(source)


@pytest.mark.parametrize("filename", ["generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin", "remote.py"])
def test_unlisted_loader_or_code_files_are_rejected(source, filename):
    (source / filename).write_bytes(b"unlisted")
    with pytest.raises(SnapshotVerificationError, match="missing or unlisted"):
        verify_local_snapshot(source)


@pytest.mark.parametrize("filename", sorted(snapshots.REQUIRED_FILES | {"snapshot.json"}))
def test_symlinked_artifacts_are_rejected(source, tmp_path, filename):
    target = tmp_path / "external"
    (source / filename).rename(target)
    (source / filename).symlink_to(target)
    with pytest.raises(SnapshotVerificationError, match="symlinks"):
        verify_local_snapshot(source)


def test_symlinked_snapshot_directory_is_rejected(source, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(SnapshotVerificationError, match="symlinks"):
        verify_local_snapshot(link)


def test_symlinked_ancestor_and_lexical_traversal_are_rejected(source, tmp_path):
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SnapshotVerificationError, match="symlinks"):
        verify_local_snapshot(ancestor / "snapshot")
    with pytest.raises(SnapshotVerificationError, match="traversal"):
        verify_local_snapshot(source / ".." / "snapshot")


@pytest.mark.parametrize("filename", ["../config.json", "/config.json", "nested/config.json", "CONFIG.JSON", "./config.json"])
def test_manifest_path_aliases_and_traversal_reject(source, filename):
    update_manifest(source, lambda manifest: manifest["files"].update({filename: manifest["files"].pop("config.json")}))
    with pytest.raises(SnapshotVerificationError, match="exactly"):
        verify_local_snapshot(source)


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", "future"), ("model_id", ""), ("model_revision", "  "),
    ("context_window", True), ("context_window", "64"), ("context_window", 65),
    ("context_window", 0), ("context_window", 2**31),
    ("runtime_versions", {"torch": "2.8.0+cpu"}),
    ("special_tokens", {"bos_token": "<bos>"}), ("files", []),
])
def test_malformed_manifest_fields_reject(source, field, value):
    update_manifest(source, lambda manifest: manifest.update({field: value}))
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


@pytest.mark.parametrize("unknown", ["trust_remote_code", "download_url", "generation_config"])
def test_unknown_manifest_fields_reject(source, unknown):
    update_manifest(source, lambda manifest: manifest.update({unknown: True}))
    with pytest.raises(SnapshotVerificationError, match="schema"):
        verify_local_snapshot(source)


@pytest.mark.parametrize(("field", "value"), [
    ("model_type", "custom"), ("architectures", ["CustomModel"]), ("auto_map", {}),
    ("quantization_config", {}), ("n_positions", True), ("n_ctx", 63), ("n_head", 3),
    ("n_layer", "1"), ("vocab_size", 0),
])
def test_unsupported_model_config_rejects_before_native_loading(source, field, value):
    config = json.loads((source / "config.json").read_bytes())
    config[field] = value
    replace_artifact(source, "config.json", json.dumps(config).encode())
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


@pytest.mark.parametrize("filename", ["snapshot.json", "config.json", "tokenizer.json"])
@pytest.mark.parametrize("contents", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":"\\ud800"}', b'{"x":' + b'[' * 70 + b'0' + b']' * 70 + b'}'])
def test_invalid_or_ambiguous_json_rejects_as_snapshot_error(source, filename, contents):
    if filename == "snapshot.json":
        (source / filename).write_bytes(contents)
    else:
        replace_artifact(source, filename, contents)
    with pytest.raises(SnapshotVerificationError):
        verify_local_snapshot(source)


def test_changed_chat_template_is_rejected_even_with_matching_file_hash(source):
    replace_artifact(source, "chat_template.jinja", b"{{ messages[0].content }}")
    with pytest.raises(SnapshotVerificationError, match="chat template"):
        verify_local_snapshot(source)


def test_observed_artifact_identity_changes_when_pinned_bytes_change(source):
    with verify_local_snapshot(source) as original:
        identity = original.identity
    original_weights = (source / "model.safetensors").read_bytes()
    replace_artifact(source, "model.safetensors", original_weights[:-4] + struct.pack("<f", 0.75))
    with verify_local_snapshot(source) as changed:
        assert changed.identity.model_digest != identity.model_digest
        assert changed.identity.tokenizer_digest == identity.tokenizer_digest
        assert changed.identity.chat_template_digest == identity.chat_template_digest
    replace_artifact(source, "tokenizer.json", b'{"version":"1.0","model":{"type":"WordLevel","changed":true}}')
    with verify_local_snapshot(source) as changed:
        assert changed.identity.tokenizer_digest != identity.tokenizer_digest


def test_special_token_declaration_is_bound_to_tokenizer_identity(source):
    with verify_local_snapshot(source) as original:
        digest = original.identity.tokenizer_digest
    update_manifest(source, lambda manifest: manifest["special_tokens"].update({"pad_token": None}))
    with verify_local_snapshot(source) as changed:
        assert changed.identity.tokenizer_digest != digest


def test_installed_runtime_version_drift_rejects(source, monkeypatch):
    monkeypatch.setattr(snapshots.metadata, "version", lambda _: "wrong-version")
    with pytest.raises(SnapshotVerificationError, match="Installed runtime"):
        verify_local_snapshot(source)


def test_copy_failure_and_manifest_race_clean_up_private_directory(source, monkeypatch):
    original_read = snapshots._read_regular
    copies = []

    def mutate_manifest(path, destination=None):
        if destination is not None:
            copies.append(destination.parent)
            if path.name == "tokenizer.json":
                update_manifest(source, lambda manifest: manifest.update({"model_revision": "changed-during-copy"}))
        return original_read(path, destination)

    monkeypatch.setattr(snapshots, "_read_regular", mutate_manifest)
    with pytest.raises(SnapshotVerificationError, match="manifest changed"):
        verify_local_snapshot(source)
    assert copies
    assert not any(path.exists() for path in copies)


def test_artifact_hash_failure_cleans_up_private_directory(source, monkeypatch):
    original_read = snapshots._read_regular
    copies: list[Path] = []

    def capture_copy(path, destination=None):
        if destination is not None:
            copies.append(destination.parent)
        return original_read(path, destination)

    monkeypatch.setattr(snapshots, "_read_regular", capture_copy)
    (source / "model.safetensors").write_bytes(b"modified after manifest")
    with pytest.raises(SnapshotVerificationError, match="digest"):
        verify_local_snapshot(source)
    assert copies
    assert not any(path.exists() for path in copies)
