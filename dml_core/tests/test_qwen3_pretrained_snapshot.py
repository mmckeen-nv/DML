"""Owned staging, exact original-byte retention and atomic publication controls."""
from __future__ import annotations

import hashlib
import json

import pytest

from daystrom_dml.services import qwen3_model_snapshot as snapshot
from daystrom_dml.services import qwen3_pretrained_snapshot as prepare
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError
from qwen3_model_input_fixture import create_qwen3_snapshot


@pytest.fixture
def staged(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    fixture = create_qwen3_snapshot(stage / "bundle", context_window=32768)
    (fixture.path / "snapshot.json").unlink()
    (fixture.path / "chat_template.jinja").unlink()
    (stage / "LICENSE").write_bytes(b"Synthetic fixture; no trained weights.")
    (stage / "tokenizer_config.json").write_bytes(b'{"synthetic":true}')
    paths = [*fixture.path.iterdir(), stage / "LICENSE", stage / "tokenizer_config.json"]
    pins = {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", pins)
    monkeypatch.setattr(prepare, "MODEL_ID", "synthetic-fixture-no-trained-model")
    return stage, pins


def test_preparation_retains_every_original_inode_and_uses_one_private_copy(staged, tmp_path, monkeypatch):
    stage, pins = staged
    before = {name: (prepare._source_path(stage, name).stat().st_ino,
                     prepare._source_path(stage, name).read_bytes()) for name in pins}
    observed = []
    reader = snapshot.read_regular

    def count(source, destination=None):
        if source.name in snapshot.SHARD_FILES and destination is not None:
            observed.append((source.name, destination.parent))
        return reader(source, destination)

    monkeypatch.setattr(snapshot, "read_regular", count)
    target = tmp_path / "prepared"
    report = prepare.prepare_qwen3_snapshot(stage, target)
    assert not stage.exists()
    assert {p.name for p in target.iterdir()} == {"bundle", "LICENSE", "tokenizer_config.json", "provenance.json"}
    assert report == json.loads((target / "provenance.json").read_bytes())
    assert report["original_shards_retained"] and report["config_changes"] == {}
    assert report["omitted_tensors"] == [] and report["tied_head"]["raw_bytes_equal"]
    assert len(report["learned_tensors"]) == 25
    assert sorted(name for name, _ in observed) == sorted(snapshot.SHARD_FILES)
    assert len({parent for _, parent in observed}) == 1
    assert all(not parent.exists() for _, parent in observed)
    for name, (inode, data) in before.items():
        path = prepare._source_path(target, name)
        assert path.stat().st_ino == inode and path.read_bytes() == data
    with snapshot.verify_qwen3_snapshot(target / "bundle") as verified:
        assert verified.identity.to_payload() == report["model_identity"]


@pytest.mark.parametrize("name", sorted(prepare.SOURCE_FILE_PINS))
@pytest.mark.parametrize("fault", ["bytes", "size", "missing", "symlink"])
def test_every_official_pin_is_checked_before_generated_files(staged, tmp_path, name, fault):
    stage, _ = staged
    path = prepare._source_path(stage, name)
    data = path.read_bytes()
    if fault == "bytes":
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif fault == "size":
        path.write_bytes(data + b"x")
    elif fault == "missing":
        path.unlink()
    else:
        other = tmp_path / name
        other.write_bytes(data)
        path.unlink()
        path.symlink_to(other)
    with pytest.raises((prepare.PretrainedSnapshotError, SnapshotVerificationError, OSError)):
        prepare.prepare_qwen3_snapshot(stage, tmp_path / "target")
    assert not (stage / "bundle" / "snapshot.json").exists()
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_preexisting_destination_is_never_overwritten(staged, tmp_path, kind):
    stage, _ = staged
    target = tmp_path / "target"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_bytes(b"original")
    else:
        target.symlink_to(stage, target_is_directory=True)
    before = target.lstat()
    with pytest.raises(prepare.PretrainedSnapshotError, match="already exists"):
        prepare.prepare_qwen3_snapshot(stage, target)
    assert target.lstat().st_ino == before.st_ino and stage.exists()


@pytest.mark.parametrize("failure", ["tensor", "publish"])
def test_failed_preparation_preserves_all_sources_and_cleans_only_generated_metadata(staged, tmp_path, monkeypatch, failure):
    stage, pins = staged
    before = {name: prepare._source_path(stage, name).read_bytes() for name in pins}

    def fail(*args, **kwargs):
        raise OSError("synthetic failure")

    monkeypatch.setattr(snapshot if failure == "tensor" else prepare,
                        "load_qwen3_state" if failure == "tensor" else "_publish_directory", fail)
    with pytest.raises(OSError, match="synthetic failure"):
        prepare.prepare_qwen3_snapshot(stage, tmp_path / "target")
    assert {p.name for p in stage.iterdir()} == {"bundle", "LICENSE", "tokenizer_config.json"}
    assert {p.name for p in (stage / "bundle").iterdir()} == prepare._RUNTIME_SOURCE_FILES
    assert all(prepare._source_path(stage, name).read_bytes() == data for name, data in before.items())
    assert not (tmp_path / "target").exists()


def test_atomic_publication_refuses_a_destination_created_after_validation(staged, tmp_path, monkeypatch):
    stage, pins = staged
    target = tmp_path / "target"
    original = prepare._publish_directory

    def race(source, destination, identity):
        destination.mkdir()
        (destination / "owner.txt").write_bytes(b"other owner")
        original(source, destination, identity)

    monkeypatch.setattr(prepare, "_publish_directory", race)
    with pytest.raises(FileExistsError):
        prepare.prepare_qwen3_snapshot(stage, target)
    assert (target / "owner.txt").read_bytes() == b"other owner"
    assert all(prepare._source_path(stage, name).exists() for name in pins)


def test_alias_mismatch_never_publishes_even_with_self_consistent_source_pins(staged, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    stage, pins = staged
    path = stage / "bundle" / snapshot.SHARD_FILES[1]
    values = load_file(str(path))
    values["lm_head.weight"] = values["lm_head.weight"].clone()
    values["lm_head.weight"][0, 0] = torch.tensor(1, dtype=torch.bfloat16)
    save_file(values, str(path))
    pins[path.name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", pins)
    with pytest.raises(SnapshotVerificationError, match="byte-exact"):
        prepare.prepare_qwen3_snapshot(stage, tmp_path / "target")
    assert path.exists() and not (tmp_path / "target").exists()


def test_source_profile_is_captured_before_io(staged, tmp_path, monkeypatch):
    stage, pins = staged
    original = prepare._verify_sources
    calls = []

    def mutate_globals(path, profile):
        calls.append(profile)
        monkeypatch.setattr(prepare, "MODEL_ID", "mutated-global")
        monkeypatch.setattr(prepare, "SOURCE_FILE_PINS", {})
        return original(path, profile)

    monkeypatch.setattr(prepare, "_verify_sources", mutate_globals)
    result = prepare.prepare_qwen3_snapshot(stage, tmp_path / "target")
    assert len(calls) == 2 and calls[0] is calls[1]
    assert result["model_id"] == "synthetic-fixture-no-trained-model"
    assert set(result["source_files"]) == set(pins)


def test_late_source_mutation_cannot_publish_inconsistent_provenance(staged, tmp_path, monkeypatch):
    stage, _ = staged
    original = prepare._verify_sources
    calls = []

    def mutate_after_last_verification(path, profile):
        result = original(path, profile)
        calls.append(profile)
        if len(calls) == 2:
            target = path / "bundle" / "tokenizer.json"
            data = target.read_bytes()
            target.write_bytes(data + b" ")
        return result

    monkeypatch.setattr(prepare, "_verify_sources", mutate_after_last_verification)
    with pytest.raises(prepare.PretrainedSnapshotError, match="frozen source or manifest"):
        prepare.prepare_qwen3_snapshot(stage, tmp_path / "target")
    assert not (tmp_path / "target").exists()
    assert not (stage / "provenance.json").exists()


def test_atomic_publication_preserves_cross_directory_inode(tmp_path):
    source = tmp_path / "source"
    parent = tmp_path / "parent"
    source.mkdir()
    parent.mkdir()
    (source / "file").write_bytes(b"invented")
    info = source.stat()
    prepare._publish_directory(source, parent / "target", (info.st_dev, info.st_ino))
    assert not source.exists()
    assert (parent / "target").stat().st_ino == info.st_ino
    assert (parent / "target" / "file").read_bytes() == b"invented"
