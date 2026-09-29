"""Separate exact-source Coder admission; synthetic weights prove mechanics only."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from daystrom_dml.services import qwen_pretrained_snapshot as prepare
from daystrom_dml.services.model_input_snapshot import SnapshotVerificationError
from daystrom_dml.services.qwen_model_input import LocalQwenInputConsumer
from model_input_fixture import require_model_input_dependencies
from qwen_model_input_fixture import create_qwen_snapshot


def _pins(path):
    return {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in path.iterdir()}


@pytest.fixture
def sources(tmp_path, monkeypatch):
    require_model_input_dependencies()
    import torch
    from safetensors.torch import load_file, save_file

    result = []
    for index, prefix in enumerate(("", "CODER_")):
        fixture = create_qwen_snapshot(tmp_path / f"fixture-{index}",
                                       context_window=32768, seed=20260920 + index)
        raw = tmp_path / f"raw-{index}"
        raw.mkdir()
        config = json.loads((fixture.path / "config.json").read_bytes())
        config["torch_dtype"] = "bfloat16"
        (raw / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (raw / "tokenizer.json").write_bytes((fixture.path / "tokenizer.json").read_bytes())
        (raw / "tokenizer_config.json").write_bytes(b'{"synthetic_fixture":true}')
        (raw / "LICENSE").write_bytes(b"Synthetic fixture; no trained model weights.")
        state = {name: tensor.to(torch.bfloat16) for name, tensor in
                 load_file(str(fixture.path / "model.safetensors")).items()}
        state["model.embed_tokens.weight"][0, 0] = -0.0
        save_file(state, str(raw / "model.safetensors"))
        monkeypatch.setattr(prepare, prefix + "SOURCE_FILE_PINS", _pins(raw))
        monkeypatch.setattr(prepare, prefix + "MODEL_ID", f"synthetic-no-trained-model-{index}")
        monkeypatch.setattr(prepare, prefix + "MODEL_REVISION", f"synthetic-revision-{index}")
        result.append((raw, state, prefix))
    return result


def _preparer(prefix):
    return prepare.prepare_qwen_coder_snapshot if prefix else prepare.prepare_qwen_snapshot


def test_distinct_provenance_and_exact_learned_bytes(sources, tmp_path):
    import torch
    from safetensors.torch import load_file

    identities = []
    for index, (raw, original, prefix) in enumerate(sources):
        target = tmp_path / f"prepared-{index}"
        report = _preparer(prefix)(raw, target)
        assert report["complete"] is True and report["live_qualified"] is False
        assert report["model_id"] == f"synthetic-no-trained-model-{index}"
        assert report["source_revision"] == f"synthetic-revision-{index}"
        assert report["schema_version"] == getattr(prepare, prefix + "PREPARER_VERSION")
        assert report["normalizer_sha256"] == hashlib.sha256(
            Path(prepare.__file__).read_bytes()).hexdigest()
        assert report["omitted_tensors"] == []
        assert report["config_changes"] == {"torch_dtype": {"source": "bfloat16", "target": "float32"}}
        assert json.loads((target / "provenance.json").read_bytes()) == report
        manifest = json.loads((target / "bundle" / "snapshot.json").read_bytes())
        assert manifest["model_id"] == report["model_id"]
        assert manifest["model_revision"] == report["source_revision"]
        for name, (size, digest) in _pins(raw).items():
            assert report["source_files"][name] == {"bytes": size, "sha256": digest,
                "source_url": f"https://huggingface.co/{report['model_id']}/resolve/{report['source_revision']}/{name}"}
        weights = load_file(str(target / "bundle" / "model.safetensors"))
        assert weights.keys() == original.keys()
        for name, value in original.items():
            assert weights[name].dtype == torch.float32
            assert prepare.tensor_digest(value) == prepare.tensor_digest(weights[name].bfloat16())
            assert torch.equal(value.float(), weights[name])
        assert torch.signbit(weights["model.embed_tokens.weight"][0, 0])
        with LocalQwenInputConsumer(target / "bundle") as consumer:
            assert consumer._identity.to_payload() == report["model_identity"]
            assert consumer._model.lm_head.weight.data_ptr() == consumer._model.model.embed_tokens.weight.data_ptr()
        identities.append(report["model_identity"])
    assert identities[0] != identities[1]


def test_builtin_pins_are_immutable_and_public_apis_cannot_override(tmp_path):
    assert prepare.MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"
    assert prepare.MODEL_REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert prepare.PREPARER_VERSION == "dml-pretrained-qwen2-instruct-v1"
    assert prepare.CODER_MODEL_ID == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert prepare.CODER_MODEL_REVISION == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    assert prepare.CODER_PREPARER_VERSION == "dml-pretrained-qwen2-coder-instruct-v1"
    assert prepare.SOURCE_FILE_PINS["model.safetensors"] == (
        3087467144, "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee")
    assert dict(prepare.CODER_SOURCE_FILE_PINS) == {
        "config.json": (660, "88f9a17863c05fb313515d2ff74b1098e0c35579f99068e32beda00618508ae0"),
        "tokenizer.json": (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
        "tokenizer_config.json": (7305, "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4"),
        "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
        "model.safetensors": (3087467144, "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"),
    }
    for prefix in ("", "CODER_"):
        pins = getattr(prepare, prefix + "SOURCE_FILE_PINS")
        with pytest.raises(TypeError):
            pins["config.json"] = (0, "untrusted")
        function = _preparer(prefix)
        assert tuple(inspect.signature(function).parameters) == ("raw_directory", "destination")
        for override in ("source_pins", "model_id", "revision", "profile"):
            with pytest.raises(TypeError):
                function(tmp_path, tmp_path / "prepared", **{override: "untrusted"})


def test_bidirectional_source_substitution_rejects_before_tensor_load(sources, tmp_path, monkeypatch):
    import safetensors.torch

    def must_not_load(*args, **kwargs):
        raise AssertionError("Unpinned source reached the tensor loader")

    monkeypatch.setattr(safetensors.torch, "load_file", must_not_load)
    for index, (raw, _, _) in enumerate(sources):
        other_prefix = sources[1 - index][2]
        target = tmp_path / f"prepared-{index}"
        with pytest.raises(prepare.PretrainedSnapshotError, match="Pinned Qwen source"):
            _preparer(other_prefix)(raw, target)
        assert not target.exists()


def test_each_call_binds_profile_before_copy_despite_global_rebinding(sources, tmp_path, monkeypatch):
    original_read = prepare._read_regular
    for index, (raw, _, prefix) in enumerate(sources):
        expected_id = getattr(prepare, prefix + "MODEL_ID")
        expected_revision = getattr(prepare, prefix + "MODEL_REVISION")
        expected_version = getattr(prepare, prefix + "PREPARER_VERSION")
        original_pins = dict(getattr(prepare, prefix + "SOURCE_FILE_PINS"))
        changed = False
        with monkeypatch.context() as patch:
            def mutate_then_copy(*args, **kwargs):
                nonlocal changed
                if not changed:
                    changed = True
                    # Mutate the test-owned map as well as rebinding every global.
                    getattr(prepare, prefix + "SOURCE_FILE_PINS")["model.safetensors"] = (0, "changed")
                    patch.setattr(prepare, prefix + "SOURCE_FILE_PINS", {"intruder": (0, "changed")})
                    for suffix in ("MODEL_ID", "MODEL_REVISION", "PREPARER_VERSION"):
                        patch.setattr(prepare, prefix + suffix, "changed-during-copy")
                return original_read(*args, **kwargs)

            patch.setattr(prepare, "_read_regular", mutate_then_copy)
            target = tmp_path / f"prepared-{index}"
            report = _preparer(prefix)(raw, target)
        assert changed
        assert report["model_id"] == expected_id
        assert report["source_revision"] == expected_revision
        assert report["schema_version"] == expected_version
        assert {name: (row["bytes"], row["sha256"]) for name, row in report["source_files"].items()} == original_pins
        assert all(f"/{expected_id}/resolve/{expected_revision}/" in row["source_url"]
                   for row in report["source_files"].values())


def test_coder_cannot_widen_architecture_ceiling(sources, tmp_path, monkeypatch):
    import safetensors.torch

    raw = sources[1][0]
    config = json.loads((raw / "config.json").read_bytes())
    config["num_hidden_layers"] = 33
    (raw / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(prepare, "CODER_SOURCE_FILE_PINS", _pins(raw))

    def must_not_load(*args, **kwargs):
        raise AssertionError("Out-of-bounds config reached the tensor loader")

    monkeypatch.setattr(safetensors.torch, "load_file", must_not_load)
    with pytest.raises(SnapshotVerificationError, match="Unsupported Qwen attention"):
        prepare.prepare_qwen_coder_snapshot(raw, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_coder_failed_publication_preserves_input_and_removes_owned_output(sources, tmp_path, monkeypatch):
    raw = sources[1][0]
    before = _pins(raw)
    original_write = prepare._write

    def fail_completion(path, data):
        if path.name == "provenance.partial":
            raise OSError("simulated completion write failure")
        original_write(path, data)

    monkeypatch.setattr(prepare, "_write", fail_completion)
    with pytest.raises(OSError, match="simulated completion"):
        prepare.prepare_qwen_coder_snapshot(raw, tmp_path / "prepared")
    assert _pins(raw) == before
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".dml-qwen-pretrained-*"))
