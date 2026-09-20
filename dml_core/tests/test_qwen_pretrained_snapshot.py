"""Offline Qwen preparation: provenance pins, publication and failure controls."""
from __future__ import annotations

import hashlib
import json

import pytest

from daystrom_dml.services import qwen_pretrained_snapshot as prepare
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError
from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer
from qwen_model_input_fixture import create_qwen_snapshot


@pytest.fixture
def raw(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file

    fixture = create_qwen_snapshot(tmp_path / "fixture", context_window=32768)
    path = tmp_path / "raw"
    path.mkdir()
    config = json.loads((fixture.path / "config.json").read_bytes())
    config["torch_dtype"] = "bfloat16"
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_bytes((fixture.path / "tokenizer.json").read_bytes())
    (path / "tokenizer_config.json").write_bytes(b'{"test_fixture":true}')
    (path / "LICENSE").write_bytes(b"Synthetic test fixture, no trained weights.")
    state = {name: tensor.to(torch.bfloat16) for name, tensor in
             load_file(str(fixture.path / "model.safetensors")).items()}
    state["model.embed_tokens.weight"][0, 0] = -0.0
    save_file(state, str(path / "model.safetensors"))
    pins = {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.iterdir()}
    monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", pins)
    monkeypatch.setattr(prepare, "MODEL_ID", "synthetic-fixture-no-trained-model")
    return path, state


def test_complete_offline_preparation_is_admitted_and_preserves_provenance(raw, tmp_path):
    import torch
    from safetensors.torch import load_file

    source, original = raw
    target = tmp_path / "prepared"
    report = prepare.prepare_qwen_snapshot(source, target)
    assert {p.name for p in target.iterdir()} == {"bundle", "LICENSE", "provenance.json"}
    assert json.loads((target / "provenance.json").read_bytes()) == report
    assert report["complete"] is True and report["live_qualified"] is False
    assert report["omitted_tensors"] == []
    assert report["config_changes"] == {"torch_dtype": {"source": "bfloat16", "target": "float32"}}
    weights = load_file(str(target / "bundle" / "model.safetensors"))
    assert weights.keys() == original.keys()
    assert "lm_head.weight" not in weights
    for name, value in original.items():
        assert torch.equal(value.float(), weights[name])
        assert value.view(torch.uint8).numpy().tobytes() == weights[name].bfloat16().view(torch.uint8).numpy().tobytes()
    assert torch.signbit(weights["model.embed_tokens.weight"][0, 0])
    for name, record in report["bundle_files"].items():
        assert hashlib.sha256((target / "bundle" / name).read_bytes()).hexdigest() == record["sha256"]
    with LocalQwenInputConsumer(target / "bundle") as consumer:
        assert consumer._identity.to_payload() == report["model_identity"]
        assert consumer._model.lm_head.weight.data_ptr() == consumer._model.model.embed_tokens.weight.data_ptr()
    assert (target / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()


@pytest.mark.parametrize("name", sorted(prepare.SOURCE_FILE_PINS))
@pytest.mark.parametrize("mutation", ["bytes", "size", "missing", "symlink"])
def test_each_pinned_source_rejects_before_publication(raw, tmp_path, name, mutation):
    path = raw[0] / name
    data = path.read_bytes()
    if mutation == "bytes":
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif mutation == "size":
        path.write_bytes(data + b" ")
    elif mutation == "missing":
        path.unlink()
    else:
        other = tmp_path / name
        other.write_bytes(data)
        path.unlink()
        path.symlink_to(other)
    target = tmp_path / "prepared"
    with pytest.raises((prepare.PretrainedSnapshotError, SnapshotVerificationError)):
        prepare.prepare_qwen_snapshot(raw[0], target)
    assert not target.exists()


@pytest.mark.parametrize("kind", ["directory", "completed", "file", "symlink"])
def test_destination_never_overwritten(raw, tmp_path, kind):
    target = tmp_path / "prepared"
    if kind in {"directory", "completed"}:
        target.mkdir()
        if kind == "completed":
            (target / "provenance.json").write_bytes(b"original")
    elif kind == "file":
        target.write_bytes(b"original")
    else:
        target.symlink_to(raw[0], target_is_directory=True)
    with pytest.raises(prepare.PretrainedSnapshotError, match="already exists"):
        prepare.prepare_qwen_snapshot(raw[0], target)
    assert target.exists()
    if kind == "completed":
        assert (target / "provenance.json").read_bytes() == b"original"


def test_extra_raw_file_rejected(raw, tmp_path):
    (raw[0] / "unsafe.py").write_bytes(b"raise AssertionError('must not execute')")
    with pytest.raises(prepare.PretrainedSnapshotError, match="unlisted"):
        prepare.prepare_qwen_snapshot(raw[0], tmp_path / "prepared")


def test_serialization_corruption_cannot_publish(raw, tmp_path, monkeypatch):
    import safetensors.torch
    original = safetensors.torch.save_file

    def corrupt(state, name, **kwargs):
        state = {key: value.clone() for key, value in state.items()}
        state["model.norm.weight"][0] += 1
        original(state, name, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", corrupt)
    with pytest.raises(prepare.PretrainedSnapshotError, match="Serialized"):
        prepare.prepare_qwen_snapshot(raw[0], tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_completion_failure_cleans_only_owned_output(raw, tmp_path, monkeypatch):
    original = prepare._write

    def fail(path, data):
        if path.name == "provenance.partial":
            raise OSError("simulated disk full")
        original(path, data)

    monkeypatch.setattr(prepare, "_write", fail)
    with pytest.raises(OSError, match="disk full"):
        prepare.prepare_qwen_snapshot(raw[0], tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".dml-qwen-pretrained-*"))


@pytest.mark.parametrize("which", ["source", "parent", "traversal"])
def test_directory_alias_rejected(raw, tmp_path, which):
    alias = tmp_path / "alias"
    alias.symlink_to(raw[0] if which == "source" else tmp_path, target_is_directory=True)
    source = alias if which == "source" else raw[0]
    target = alias / "prepared" if which == "parent" else tmp_path / "prepared"
    if which == "traversal":
        target = raw[0] / ".." / "prepared"
    with pytest.raises(prepare.PretrainedSnapshotError):
        prepare.prepare_qwen_snapshot(source, target)
