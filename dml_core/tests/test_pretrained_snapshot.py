"""Pinned pretrained normalization, independent real-model shape and byte oracles."""
from __future__ import annotations

import hashlib
import json

import pytest

from daystrom_dml.services import pretrained_snapshot as prepare
from daystrom_dml.services.model_input import LocalTransformersInputConsumer
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError
from model_input_fixture import require_model_input_dependencies


@pytest.fixture
def raw(tmp_path, monkeypatch):
    require_model_input_dependencies()
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers import GPT2Config, GPT2LMHeadModel

    directory = tmp_path / "raw"
    directory.mkdir()
    dimensions = {"vocab_size": 5, "n_positions": 8, "n_embd": 4, "n_layer": 1, "n_head": 1}
    config = GPT2Config(**dimensions, n_ctx=8, architectures=["GPT2LMHeadModel"],
                        bos_token_id=0, eos_token_id=0, pad_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = GPT2LMHeadModel(config)
    state = {name.removeprefix("transformer."): value.detach().clone()
             for name, value in model.state_dict().items() if name != "lm_head.weight"}
    state["wte.weight"][0, 0] = -0.0
    state["h.0.attn.bias"] = torch.ones((8, 8), dtype=torch.float32).tril().reshape(1, 1, 8, 8)
    save_file(state, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    tokenizer = Tokenizer(models.WordLevel({"<|endoftext|>": 0, "hello": 1, "world": 2,
                                          "system": 3, "assistant": 4}, unk_token="<|endoftext|>"))
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "LICENSE.gpt2").write_bytes(b"Test fixture license; no trained weights.")
    pins = {path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in directory.iterdir()}
    monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", pins)
    monkeypatch.setattr(prepare, "DIMENSIONS", dimensions)
    monkeypatch.setattr(prepare, "MODEL_ID", "test-fixture-not-trained")
    return directory, state, dimensions


def test_prep_preserves_independent_model_bytes_and_strict_consumer_admits(raw, tmp_path):
    import torch
    from safetensors.torch import load_file

    directory, state, _ = raw
    target = tmp_path / "prepared"
    report = prepare.prepare_gpt2_snapshot(directory, target)
    assert {p.name for p in target.iterdir()} == {"bundle", "LICENSE.gpt2", "provenance.json"}
    assert {p.name for p in (target / "bundle").iterdir()} == {
        "snapshot.json", "model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja",
    }
    assert json.loads((target / "provenance.json").read_bytes()) == report
    weights = load_file(str(target / "bundle" / "model.safetensors"))
    for name, original in state.items():
        if name != "h.0.attn.bias":
            restored = weights["transformer." + name]
            assert restored.numpy().tobytes() == original.numpy().tobytes()
    assert torch.signbit(weights["lm_head.weight"][0, 0])
    assert weights["lm_head.weight"].numpy().tobytes() == state["wte.weight"].numpy().tobytes()
    assert len(report["learned_tensors"]) == 16
    assert len(report["omitted_verified_causal_masks"]) == 1
    assert report["live_qualified"] is False
    for name, record in report["bundle_files"].items():
        assert hashlib.sha256((target / "bundle" / name).read_bytes()).hexdigest() == record["sha256"]
    with LocalTransformersInputConsumer(target / "bundle") as consumer:
        assert consumer._identity.to_payload() == report["model_identity"]
    for name in ("config.json", "tokenizer.json"):
        assert (target / "bundle" / name).read_bytes() == (directory / name).read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "extra", "already_prefixed", "extra_head"])
def test_exact_raw_tensor_coverage_rejects(raw, mutation):
    _, state, dimensions = raw
    state = dict(state)
    if mutation == "missing":
        del state["ln_f.bias"]
    elif mutation == "extra":
        state["h.0.attn.masked_bias"] = state["ln_f.bias"]
    elif mutation == "already_prefixed":
        state["transformer.wte.weight"] = state.pop("wte.weight")
    else:
        state["lm_head.weight"] = state["wte.weight"]
    with pytest.raises(prepare.PretrainedSnapshotError, match="coverage"):
        prepare._normalize_state(state, dimensions)


@pytest.mark.parametrize("mutation", ["shape", "float16", "float64", "integer", "nan", "positive_inf", "negative_inf", "not_tensor"])
def test_learned_tensor_admission_rejects(raw, mutation):
    import torch

    _, state, dimensions = raw
    value = state["ln_f.bias"].clone()
    if mutation == "shape":
        value = value.reshape(2, 2)
    elif mutation in {"float16", "float64"}:
        value = value.to(getattr(torch, mutation))
    elif mutation == "integer":
        value = value.to(torch.int32)
    elif mutation == "not_tensor":
        value = [0.0] * 4
    else:
        value[0] = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[mutation]
    state["ln_f.bias"] = value
    with pytest.raises(prepare.PretrainedSnapshotError, match="Tensor shape, device, dtype or finite"):
        prepare._normalize_state(state, dimensions)


@pytest.mark.parametrize("mutation", ["unmasked", "negative_zero", "diagonal", "shape", "bool"])
def test_legacy_buffers_must_have_exact_causal_bytes(raw, mutation):
    _, state, dimensions = raw
    mask = state["h.0.attn.bias"].clone()
    if mutation == "unmasked":
        mask[0, 0, 0, 7] = 1
    elif mutation == "negative_zero":
        mask[0, 0, 0, 7] = -0.0
    elif mutation == "diagonal":
        mask[0, 0, 0, 0] = 0
    elif mutation == "shape":
        mask = mask.reshape(8, 8)
    else:
        mask = mask.bool()
    state["h.0.attn.bias"] = mask
    with pytest.raises(prepare.PretrainedSnapshotError):
        prepare._normalize_state(state, dimensions)


@pytest.mark.parametrize("name", ["config.json", "model.safetensors", "tokenizer.json", "LICENSE.gpt2"])
@pytest.mark.parametrize("mutation", ["bytes", "size", "missing", "symlink"])
def test_all_source_pins_fail_closed_before_output(raw, tmp_path, name, mutation):
    directory, _, _ = raw
    path = directory / name
    contents = path.read_bytes()
    if mutation == "bytes":
        path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    elif mutation == "size":
        path.write_bytes(contents + b" ")
    elif mutation == "missing":
        path.unlink()
    else:
        other = tmp_path / name
        other.write_bytes(contents)
        path.unlink()
        path.symlink_to(other)
    with pytest.raises((prepare.PretrainedSnapshotError, SnapshotVerificationError)):
        prepare.prepare_gpt2_snapshot(directory, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unlisted_raw_file_rejects(raw, tmp_path):
    directory, _, _ = raw
    (directory / "custom.py").write_text("raise AssertionError('must never execute')", encoding="utf-8")
    with pytest.raises(prepare.PretrainedSnapshotError, match="unlisted"):
        prepare.prepare_gpt2_snapshot(directory, tmp_path / "out")


@pytest.mark.parametrize("kind", ["empty_directory", "completed_directory", "file", "symlink"])
def test_existing_destination_is_never_reused(raw, tmp_path, kind):
    directory, _, _ = raw
    target = tmp_path / "out"
    if kind.endswith("directory"):
        target.mkdir()
        if kind == "completed_directory":
            (target / "provenance.json").write_bytes(b"original")
    elif kind == "file":
        target.write_bytes(b"original")
    else:
        target.symlink_to(directory, target_is_directory=True)
    with pytest.raises(prepare.PretrainedSnapshotError, match="already exists"):
        prepare.prepare_gpt2_snapshot(directory, target)
    assert target.exists()
    if kind == "completed_directory":
        assert (target / "provenance.json").read_bytes() == b"original"


def test_serialization_corruption_cannot_publish(raw, tmp_path, monkeypatch):
    import safetensors.torch

    original = safetensors.torch.save_file

    def corrupt(tensors, filename, **kwargs):
        tensors = {name: value.clone() for name, value in tensors.items()}
        tensors["transformer.ln_f.bias"][0] += 1
        original(tensors, filename, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", corrupt)
    with pytest.raises(prepare.PretrainedSnapshotError, match="Serialized"):
        prepare.prepare_gpt2_snapshot(raw[0], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_final_write_failure_removes_owned_output_and_staging(raw, tmp_path, monkeypatch):
    original = prepare._write

    def fail_record(path, payload):
        if path.name == "provenance.partial":
            raise OSError("simulated disk full")
        original(path, payload)

    monkeypatch.setattr(prepare, "_write", fail_record)
    with pytest.raises(OSError, match="disk full"):
        prepare.prepare_gpt2_snapshot(raw[0], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".dml-pretrained-*"))


@pytest.mark.parametrize("which", ["source", "parent", "traversal"])
def test_directory_aliases_reject(raw, tmp_path, which):
    directory, _, _ = raw
    alias = tmp_path / "alias"
    alias.symlink_to(directory if which == "source" else tmp_path, target_is_directory=True)
    source = alias if which == "source" else directory
    target = alias / "out" if which == "parent" else tmp_path / "out"
    if which == "traversal":
        target = directory / ".." / "out"
    with pytest.raises(prepare.PretrainedSnapshotError):
        prepare.prepare_gpt2_snapshot(source, target)


def test_normalization_has_independent_head_storage(raw):
    _, state, dimensions = raw
    converted, _, _ = prepare._normalize_state(state, dimensions)
    head = converted["lm_head.weight"]
    embedding = converted["transformer.wte.weight"]
    assert head.data_ptr() != embedding.data_ptr()
    before = embedding.clone()
    head[0, 0] = 123.0
    assert embedding.numpy().tobytes() == before.numpy().tobytes()
    assert state["wte.weight"].numpy().tobytes() == before.numpy().tobytes()
