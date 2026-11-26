"""Adversarial admission for original two-shard CPU/BF16 Qwen2BF16 artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tracemalloc

import pytest

from daystrom_dml.services import qwen2_bf16_model_snapshot as snapshot
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError
from qwen2_bf16_model_input_fixture import create_qwen2_bf16_snapshot


@pytest.fixture
def tiny(tmp_path):
    return create_qwen2_bf16_snapshot(tmp_path / "snapshot")


def rehash(path):
    manifest = json.loads((path / "snapshot.json").read_bytes())
    manifest["files"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                         for name in snapshot.REQUIRED_FILES}
    (path / "snapshot.json").write_text(json.dumps(manifest))


def test_private_original_shards_preserve_bytes_and_isolate_source(tiny):
    originals = {name: (tiny.path / name).read_bytes() for name in snapshot.REQUIRED_FILES}
    with snapshot.verify_qwen2_bf16_snapshot(tiny.path) as verified:
        private = verified.path
        assert private != tiny.path
        assert verified.identity.model_window_tokens == 4096
        assert verified.identity.runtime_identity.startswith("dml-qwen2-bf16-model-input-runtime-v1:")
        state = snapshot.load_qwen2_bf16_state(private, verified.config, verified.index)
        assert len(state) == 26 and "lm_head.weight" not in state
        records = snapshot.validate_state(state, verified.config)
        assert len(records) == 26 and all(row["source_sha256"] == row["target_sha256"] for row in records)
        assert all((private / name).read_bytes() == data for name, data in originals.items())
        for name in snapshot.SHARD_FILES:
            assert (private / name).stat().st_ino != (tiny.path / name).stat().st_ino
        (tiny.path / snapshot.SHARD_FILES[0]).write_bytes(b"source changed after capture")
        verified.validate_integrity()
        assert (private / snapshot.SHARD_FILES[0]).read_bytes() == originals[snapshot.SHARD_FILES[0]]
        del state
    assert not private.exists()


@pytest.mark.parametrize("name", sorted(snapshot.REQUIRED_FILES))
@pytest.mark.parametrize("fault", ["corrupt", "missing", "symlink", "hardlink"])
def test_each_artifact_fails_closed(tiny, tmp_path, name, fault):
    path = tiny.path / name
    data = path.read_bytes()
    if fault == "corrupt":
        path.write_bytes(data + b"x")
    elif fault == "missing":
        path.unlink()
    else:
        other = tmp_path / name
        if fault == "hardlink":
            os.link(path, other)
        else:
            other.write_bytes(data)
            path.unlink()
            path.symlink_to(other)
    with pytest.raises((SnapshotVerificationError, OSError)):
        snapshot.verify_qwen2_bf16_snapshot(tiny.path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "dml-qwen-model-snapshot-v1"), ("context_window", 40960),
    ("context_window", True), ("model_id", ""), ("special_tokens", {}), ("runtime_versions", {}),
    ("files", {"model.safetensors": "0" * 64}),
])
def test_manifest_profile_cannot_be_inferred(tiny, field, value):
    manifest = dict(tiny.manifest)
    manifest[field] = value
    (tiny.path / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotVerificationError):
        snapshot.verify_qwen2_bf16_snapshot(tiny.path)


@pytest.mark.parametrize("field,value", [
    ("architectures", ["Qwen3ForCausalLM"]), ("torch_dtype", "float32"), ("attention_bias", True),
    ("rope_scaling", {}), ("sliding_window", 4095), ("use_sliding_window", True),
    ("num_hidden_layers", 37), ("hidden_size", 15), ("num_key_value_heads", 3),
    ("max_window_layers", 71), ("max_position_embeddings", 32769), ("tie_word_embeddings", False),
    ("attention_dropout", 0.1), ("rms_norm_eps", float("nan")), ("vocab_size", True),
])
def test_config_profile_does_not_widen_legacy_or_model_bounds(tiny, field, value):
    config = json.loads((tiny.path / "config.json").read_bytes())
    config[field] = value
    with pytest.raises(SnapshotVerificationError):
        snapshot.check_config(config, 4096)


@pytest.mark.parametrize("fault", ["missing", "extra", "foreign_shard", "wrong_total", "wrong_shard", "duplicate_json"])
def test_index_requires_exact_coverage_and_containing_shard(tiny, fault):
    path = tiny.path / "model.safetensors.index.json"
    index = json.loads(path.read_bytes())
    name = next(iter(index["weight_map"]))
    if fault == "missing":
        del index["weight_map"][name]
    elif fault == "extra":
        index["weight_map"]["unexpected.weight"] = snapshot.SHARD_FILES[0]
    elif fault == "foreign_shard":
        index["weight_map"][name] = "../outside.safetensors"
    elif fault == "wrong_total":
        index["metadata"]["total_size"] += 2
    elif fault == "wrong_shard":
        index["weight_map"][name] = snapshot.SHARD_FILES[0] if index["weight_map"][name] == snapshot.SHARD_FILES[1] else snapshot.SHARD_FILES[1]
    if fault == "duplicate_json":
        path.write_text('{"metadata":{},"metadata":{},"weight_map":{}}')
    else:
        path.write_text(json.dumps(index))
    rehash(tiny.path)
    with pytest.raises(SnapshotVerificationError):
        snapshot.verify_qwen2_bf16_snapshot(tiny.path)


@pytest.mark.parametrize("fault", ["dtype", "shape", "gap", "overlap", "trailing", "duplicate_json", "header_bound"])
def test_header_validation_runs_on_snapshot_admission_before_loader(tiny, fault):
    path = tiny.path / snapshot.SHARD_FILES[0]
    original = path.read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8:8 + length])
    payload = original[8 + length:]
    name = next(name for name in header if name != "__metadata__")
    if fault == "dtype":
        header[name]["dtype"] = "F16"
    elif fault == "shape":
        header[name]["shape"] = [1, 1]
    elif fault in {"gap", "overlap"}:
        delta = 2 if fault == "gap" else -2
        header[name]["data_offsets"] = [value + delta for value in header[name]["data_offsets"]]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if fault == "duplicate_json":
        encoded = b'{"duplicate":{},"duplicate":{}}'
    encoded += b" " * (-len(encoded) % 8)
    body = struct.pack("<Q", len(encoded)) + encoded + payload
    if fault == "trailing":
        body += b"x"
    if fault == "header_bound":
        body = struct.pack("<Q", snapshot._MAX_HEADER_BYTES + 1) + body[8:]
    path.write_bytes(body)
    rehash(tiny.path)
    with pytest.raises(SnapshotVerificationError):
        snapshot.verify_qwen2_bf16_snapshot(tiny.path)


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "shape", "nan", "positive_inf", "negative_inf", "extra_head", "noncontiguous", "meta"])
def test_full_tensor_proof_rejects_malformed_original_payloads_or_extra_head(tiny, fault, monkeypatch):
    import torch
    config = json.loads((tiny.path / "config.json").read_bytes())
    index = json.loads((tiny.path / "model.safetensors.index.json").read_bytes())
    state = snapshot.load_qwen2_bf16_state(tiny.path, config, index)
    name = "model.norm.weight"
    if fault == "missing":
        del state[name]
    elif fault == "extra":
        state["unexpected"] = state[name]
    elif fault == "dtype":
        state[name] = state[name].float()
    elif fault == "shape":
        state[name] = state[name].reshape(4, 4)
    elif fault in {"nan", "positive_inf", "negative_inf"}:
        state[name] = state[name].clone()
        state[name][-1] = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[fault]
        monkeypatch.setattr(snapshot, "_TENSOR_CHUNK_ELEMENTS", 3)
    elif fault == "extra_head":
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    elif fault == "noncontiguous":
        state[name] = torch.ones((16, 2), dtype=torch.bfloat16)[:, 0]
    else:
        state[name] = torch.empty((16,), device="meta", dtype=torch.bfloat16)
    with pytest.raises(SnapshotVerificationError):
        snapshot.validate_state(state, config)


def test_shard_reader_has_no_payload_sized_python_capture(tmp_path):
    source = tmp_path / snapshot.SHARD_FILES[0]
    block = bytes(range(256)) * 4096
    with source.open("wb") as output:
        for _ in range(20):
            output.write(block)
    expected = hashlib.sha256(block * 20).hexdigest()
    tracemalloc.start()
    try:
        captured, digest = snapshot.read_regular(source)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert captured == b"" and digest == expected
    assert peak < 5 * 1024 * 1024


def test_metadata_bound_rejects_before_capture(tmp_path):
    source = tmp_path / "config.json"
    source.write_bytes(b"x" * (snapshot._METADATA_LIMITS[source.name] + 1))
    with pytest.raises(SnapshotVerificationError, match="bounded size"):
        snapshot.read_regular(source)


@pytest.mark.parametrize("mutation", ["grow", "shrink", "replace"])
def test_stream_reader_rejects_midread_size_or_inode_changes(tmp_path, monkeypatch, mutation):
    source = tmp_path / "config.json"
    source.write_bytes(b"12345678")
    monkeypatch.setattr(snapshot, "_CHUNK_BYTES", 4)
    fdopen = snapshot.os.fdopen

    class MutatingReader:
        def __init__(self, handle):
            self.handle = handle
            self.changed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            data = self.handle.read(size)
            if not self.changed:
                self.changed = True
                if mutation == "grow":
                    with source.open("ab") as handle:
                        handle.write(b"x" * 100000)
                elif mutation == "shrink":
                    source.write_bytes(b"1")
                else:
                    source.rename(tmp_path / "original")
                    source.write_bytes(b"12345678")
            return data

    monkeypatch.setattr(snapshot.os, "fdopen", lambda fd, mode: MutatingReader(fdopen(fd, mode)))
    with pytest.raises(SnapshotVerificationError, match="grew|changed"):
        snapshot.read_regular(source)


def test_reader_never_overwrites_existing_destination(tmp_path):
    source = tmp_path / "config.json"
    destination = tmp_path / "owned-by-another-operation"
    source.write_bytes(b"{}")
    destination.write_bytes(b"retain")
    with pytest.raises(FileExistsError):
        snapshot.read_regular(source, destination)
    assert destination.read_bytes() == b"retain"


def test_header_admission_failure_removes_its_private_copy(tiny, tmp_path, monkeypatch):
    private_root = tmp_path / "private"
    private_root.mkdir()
    monkeypatch.setattr(snapshot.tempfile, "tempdir", str(private_root))
    path = tiny.path / snapshot.SHARD_FILES[0]
    path.write_bytes(b"not a safetensors file")
    rehash(tiny.path)
    with pytest.raises(SnapshotVerificationError):
        snapshot.verify_qwen2_bf16_snapshot(tiny.path)
    assert list(private_root.iterdir()) == []


def test_private_payload_mutation_is_detected(tiny):
    with snapshot.verify_qwen2_bf16_snapshot(tiny.path) as verified:
        path = verified.path / snapshot.SHARD_FILES[0]
        path.chmod(0o600)
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        with pytest.raises(SnapshotVerificationError, match="changed"):
            verified.validate_integrity()


def test_exact_3b_architecture_fits_declared_parameter_bound_without_tensor_allocation(tiny):
    config = json.loads((tiny.path / 'config.json').read_bytes())
    config.update(hidden_size=2048, intermediate_size=11008, num_hidden_layers=36,
                  num_attention_heads=16, num_key_value_heads=2, vocab_size=151936,
                  max_position_embeddings=32768, sliding_window=32768, max_window_layers=70)
    before = dict(config)
    snapshot.check_config(config, 32768)
    shapes = snapshot.expected_shapes(config)
    assert config == before and config['max_window_layers'] == 70
    assert len(shapes) == 434 and 'lm_head.weight' not in shapes
    assert sum(math.prod(shape) for shape in shapes.values()) == 3_085_938_688
    assert shapes['model.layers.35.self_attn.q_proj.bias'] == (2048,)
    assert shapes['model.layers.35.self_attn.k_proj.bias'] == (256,)
    assert not any('q_norm' in name or 'k_norm' in name for name in shapes)
    config['intermediate_size'] = 12000
    with pytest.raises(SnapshotVerificationError, match='parameter bound'):
        snapshot.check_config(config, 32768)


def test_tensor_validation_and_digests_use_bounded_raw_bf16_views(tiny, monkeypatch):
    import torch

    config = json.loads((tiny.path / 'config.json').read_bytes())
    index = json.loads((tiny.path / 'model.safetensors.index.json').read_bytes())
    state = snapshot.load_qwen2_bf16_state(tiny.path, config, index)
    monkeypatch.setattr(snapshot, '_TENSOR_CHUNK_ELEMENTS', 128)
    original_isfinite, original_numpy = torch.isfinite, torch.Tensor.numpy
    sizes, numpy_sizes = [], []

    def bounded_isfinite(value):
        sizes.append(value.numel())
        assert value.dtype == torch.bfloat16 and value.numel() <= 128
        return original_isfinite(value)

    def raw_numpy(value, *args, **kwargs):
        numpy_sizes.append(value.numel())
        assert value.dtype == torch.uint8 and value.numel() <= 256
        return original_numpy(value, *args, **kwargs)

    monkeypatch.setattr(torch, 'isfinite', bounded_isfinite)
    monkeypatch.setattr(torch.Tensor, 'numpy', raw_numpy)
    records = snapshot.validate_state(state, config)
    assert len(records) == 26 and sizes and numpy_sizes
    assert sum(sizes) == sum(value.numel() for value in state.values())
