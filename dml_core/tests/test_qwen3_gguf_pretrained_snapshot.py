"""Preparation must preserve failed stages and never replace prior packages."""
import pytest

from daystrom_dml.services import qwen3_gguf_pretrained_snapshot as preparation


def test_existing_destination_is_never_replaced(tmp_path):
    stage, destination = tmp_path / 'stage', tmp_path / 'prepared'
    stage.mkdir()
    destination.mkdir()
    sentinel = destination / 'retained'
    sentinel.write_bytes(b'previous candidate')
    with pytest.raises(preparation.snapshot.base.SnapshotVerificationError):
        preparation.prepare_qwen3_gguf_snapshot(stage, destination)
    assert sentinel.read_bytes() == b'previous candidate'
    assert stage.exists()


def test_wrong_original_metadata_remains_for_audit(tmp_path):
    stage = tmp_path / 'stage'
    stage.mkdir()
    for name in preparation.snapshot.REQUIRED_FILES - {'chat_template.jinja'}:
        (stage / name).write_bytes(b'wrong original')
    with pytest.raises(preparation.snapshot.base.SnapshotVerificationError, match='immutable pin'):
        preparation.prepare_qwen3_gguf_snapshot(stage, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()
    assert (stage / 'config.json').read_bytes() == b'wrong original'
    assert not (stage / 'snapshot.json').exists()


def test_unlisted_source_is_rejected_without_changes(tmp_path):
    stage = tmp_path / 'stage'
    stage.mkdir()
    (stage / 'unlisted').write_bytes(b'preserve')
    with pytest.raises(preparation.snapshot.base.SnapshotVerificationError, match='inventory'):
        preparation.prepare_qwen3_gguf_snapshot(stage, tmp_path / 'prepared')
    assert {p.name for p in stage.iterdir()} == {'unlisted'}
