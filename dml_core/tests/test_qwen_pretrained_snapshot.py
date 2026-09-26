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
    fixture = create_qwen_snapshot(tmp_path / "fixture", context_window=32768)

    import torch
    from safetensors.torch import load_file, save_file

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


@pytest.mark.parametrize("mutation", ["payload", "dtype", "shape"])
def test_serialization_corruption_cannot_publish(raw, tmp_path, monkeypatch, mutation):
    import struct

    original = prepare._write_normalized_state

    def corrupt(state, config, destination):
        records = original(state, config, destination)
        with destination.open("r+b") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            if mutation == "payload":
                offset = 8 + header_size + header["model.norm.weight"]["data_offsets"][0]
                stream.seek(offset)
                original_byte = stream.read(1)
                stream.seek(offset)
                stream.write(bytes([original_byte[0] ^ 1]))
            else:
                payload = stream.read()
                if mutation == "dtype":
                    header["model.norm.weight"]["dtype"] = "I32"
                else:
                    header["model.norm.weight"]["shape"] = [4, 4]
                altered = json.dumps(header, separators=(",", ":")).encode()
                altered += b" " * (-len(altered) % 8)
                stream.seek(0)
                stream.write(struct.pack("<Q", len(altered)) + altered + payload)
                stream.truncate()
        return records

    monkeypatch.setattr(prepare, "_write_normalized_state", corrupt)
    with pytest.raises(prepare.PretrainedSnapshotError, match="Serialized"):
        prepare.prepare_qwen_snapshot(raw[0], tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".dml-qwen-pretrained-*"))


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


@pytest.mark.parametrize("chunk_elements", [7, 65536])
def test_streamed_weights_match_independent_serializer_and_eager_oracle(
    raw, tmp_path, monkeypatch, chunk_elements,
):
    import torch
    from safetensors.torch import load_file, save_file

    source, state = raw
    state["model.embed_tokens.weight"].reshape(-1)[:6] = torch.tensor(
        [0.0, -0.0, torch.finfo(torch.bfloat16).max, -torch.finfo(torch.bfloat16).max,
         torch.finfo(torch.bfloat16).tiny, torch.finfo(torch.bfloat16).tiny / 128],
        dtype=torch.bfloat16,
    )
    config = json.loads((source / "config.json").read_bytes())
    expected, expected_records = prepare.normalize_state(state, config)
    oracle = tmp_path / "eager.safetensors"
    save_file(expected, str(oracle), metadata={"format": "pt"})
    monkeypatch.setattr(prepare, "_NORMALIZE_CHUNK_ELEMENTS", chunk_elements)
    destination = tmp_path / "streamed.safetensors"
    records = prepare._write_normalized_state(state, config, destination)
    assert destination.read_bytes() == oracle.read_bytes()
    assert records == expected_records
    restored = load_file(str(destination), device="cpu")
    assert restored.keys() == state.keys()
    for name, original in state.items():
        assert torch.equal(restored[name], expected[name])
        assert torch.equal(restored[name].bfloat16().view(torch.uint16), original.view(torch.uint16))


def test_streamed_conversion_bounds_actual_live_float32_storage(raw, tmp_path, monkeypatch):
    import weakref

    import torch
    from safetensors.torch import load_file
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Allocations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.live = []
            self.peak_bytes = self.largest_bytes = 0

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    storage = value.untyped_storage()
                    size = storage.nbytes()
                    self.live.append((weakref.ref(value), storage.data_ptr(), size))
                    self.largest_bytes = max(self.largest_bytes, size)
            self.live = [item for item in self.live if item[0]() is not None]
            live_bytes = sum({pointer: size for _, pointer, size in self.live}.values())
            self.peak_bytes = max(self.peak_bytes, live_bytes)
            return result

    source, state = raw
    chunk_elements = 31
    monkeypatch.setattr(prepare, "_NORMALIZE_CHUNK_ELEMENTS", chunk_elements)
    config = json.loads((source / "config.json").read_bytes())
    destination = tmp_path / "bounded.safetensors"
    allocations = Allocations()
    with allocations:
        prepare._write_normalized_state(state, config, destination)
    assert 0 < allocations.largest_bytes <= chunk_elements * 4
    assert allocations.peak_bytes <= chunk_elements * 8
    assert sum(tensor.numel() * 4 for tensor in state.values()) > allocations.peak_bytes * 100
    restored = load_file(str(destination), device="cpu")
    for name, original in state.items():
        assert torch.equal(restored[name], original.float())


@pytest.mark.parametrize("mutation", [
    "missing", "unknown", "shape", "dtype", "meta_device", "not_tensor", "noncontiguous",
    "nan", "positive_infinity", "negative_infinity",
])
def test_streamed_writer_rejects_unqualified_tensors(raw, tmp_path, monkeypatch, mutation):
    source, state = raw
    state = dict(state)
    name = "model.layers.0.self_attn.q_proj.weight"
    if mutation == "missing":
        del state[name]
    elif mutation == "unknown":
        state["unknown.weight"] = state[name].clone()
    elif mutation == "shape":
        state[name] = state[name][:1].clone()
    elif mutation == "dtype":
        state[name] = state[name].float()
    elif mutation == "meta_device":
        state[name] = state[name].to(device="meta")
    elif mutation == "not_tensor":
        state[name] = object()
    elif mutation == "noncontiguous":
        state[name] = state[name].transpose(0, 1)
        assert not state[name].is_contiguous()
    else:
        state[name].reshape(-1)[-1] = {
            "nan": float("nan"), "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
        }[mutation]
    monkeypatch.setattr(prepare, "_NORMALIZE_CHUNK_ELEMENTS", 7)
    config = json.loads((source / "config.json").read_bytes())
    with pytest.raises(prepare.PretrainedSnapshotError):
        prepare._write_normalized_state(state, config, tmp_path / "rejected.safetensors")


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_streamed_writer_does_not_overwrite_existing_destination(raw, tmp_path, kind):
    source, state = raw
    original = tmp_path / "original"
    original.write_bytes(b"preserve unrelated output")
    destination = tmp_path / "streamed.safetensors"
    if kind == "file":
        destination.write_bytes(original.read_bytes())
    else:
        destination.symlink_to(original)
    config = json.loads((source / "config.json").read_bytes())
    with pytest.raises(FileExistsError):
        prepare._write_normalized_state(state, config, destination)
    assert original.read_bytes() == destination.read_bytes() == b"preserve unrelated output"
    assert destination.is_symlink() is (kind == "symlink")


@pytest.mark.parametrize("failure", ["late_nonfinite", "disk_full"])
def test_streaming_failure_removes_only_owned_staging(raw, tmp_path, monkeypatch, failure):
    from pathlib import Path

    from safetensors.torch import save_file

    source, state = raw
    sentinel = tmp_path / "unrelated"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(prepare, "_NORMALIZE_CHUNK_ELEMENTS", 7)
    if failure == "late_nonfinite":
        state["model.norm.weight"][-1] = float("nan")
        save_file(state, str(source / "model.safetensors"))
        pins = {path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in source.iterdir()}
        monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", pins)
        exception, message = prepare.PretrainedSnapshotError, "finite"
    else:
        original_open = Path.open

        class DiskFull:
            def __init__(self, stream):
                self.stream = stream
                self.written = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, data):
                self.written += len(data)
                if self.written > 8192:
                    raise OSError("simulated streaming disk full")
                return self.stream.write(data)

        def open_with_full_disk(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if path.name == "model.safetensors" and path.parent.name == "bundle" and mode == "xb":
                return DiskFull(stream)
            return stream

        monkeypatch.setattr(Path, "open", open_with_full_disk)
        exception, message = OSError, "streaming disk full"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    with pytest.raises(exception, match=message):
        prepare.prepare_qwen_snapshot(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".dml-qwen-pretrained-*"))
    assert sentinel.read_bytes() == b"preserve"
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()} == before
