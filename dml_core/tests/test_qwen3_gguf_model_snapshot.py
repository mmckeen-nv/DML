"""Malformed GGUF admission must fail before any runtime/model load."""
import json
import struct

import pytest

from daystrom_dml.services import qwen3_gguf_model_snapshot as snapshot


def encoded_string(text):
    raw = text.encode()
    return struct.pack('<Q', len(raw)) + raw


def header(*, overlap=False, architecture='qwen3', duplicate=False, shape_error=False, tensor_type=12):
    metadata = {'general.architecture': architecture, 'general.file_type': 15,
        'qwen3.block_count': 36, 'qwen3.embedding_length': 4096,
        'qwen3.feed_forward_length': 12288, 'qwen3.attention.head_count': 32,
        'qwen3.attention.head_count_kv': 8, 'tokenizer.ggml.eos_token_id': 151645}
    rows = list(metadata.items())
    if duplicate:
        rows.append(rows[0])
    shapes = snapshot.expected_tensor_shapes()
    result = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(shapes), len(rows)))
    for key, value in rows:
        result += encoded_string(key)
        if isinstance(value, str):
            result += struct.pack('<I', 8) + encoded_string(value)
        else:
            result += struct.pack('<II', 4, value)
    offset = 0
    for index, (name, shape) in enumerate(shapes.items()):
        if shape_error and index == 0:
            shape = (4096, 151935)
        kind = tensor_type if len(shape) == 2 else 0
        result += encoded_string(name) + struct.pack('<I', len(shape))
        result += struct.pack('<' + 'Q' * len(shape), *shape)
        result += struct.pack('<IQ', kind, 0 if overlap else offset)
        elements = 1
        for dimension in shape:
            elements *= dimension
        offset += elements * 4 if kind == 0 else elements // 256 * 144
        offset = (offset + 31) // 32 * 32
    return bytes(result)


def test_exact_architecture_header_can_be_inspected_without_payload(tmp_path):
    path = tmp_path / 'model.gguf'
    path.write_bytes(header())
    parsed = snapshot.parse_gguf(path, verify_payload=False)
    assert len(parsed['tensors']) == 399
    assert parsed['metadata']['qwen3.block_count'] == 36
    with pytest.raises(snapshot.base.SnapshotVerificationError, match='truncated'):
        snapshot.parse_gguf(path)


@pytest.mark.parametrize('options,message', [
    ({'overlap': True}, 'overlap'), ({'architecture': 'llama'}, 'architecture'),
    ({'duplicate': True}, 'Duplicate'), ({'shape_error': True}, 'inventory'),
    ({'tensor_type': 2}, 'type'),
])
def test_rejects_malformed_architecture_before_model_load(tmp_path, options, message):
    path = tmp_path / 'model.gguf'
    path.write_bytes(header(**options))
    with pytest.raises(snapshot.base.SnapshotVerificationError, match=message):
        snapshot.parse_gguf(path, verify_payload=False)


@pytest.mark.parametrize('data', [b'', b'GGUF' + struct.pack('<IQQ', 2, 399, 1),
    b'GGUF' + struct.pack('<IQQQ', 3, 399, 1, 2**40)])
def test_rejects_truncation_version_and_unbounded_metadata(tmp_path, data):
    path = tmp_path / 'model.gguf'
    path.write_bytes(data)
    with pytest.raises(snapshot.base.SnapshotVerificationError):
        snapshot.parse_gguf(path)


def test_source_reader_rejects_links(tmp_path):
    original = tmp_path / 'config.json'
    original.write_text('{}')
    linked = tmp_path / 'tokenizer.json'
    linked.symlink_to(original)
    with pytest.raises(snapshot.base.SnapshotVerificationError):
        snapshot.read_regular(linked)
    linked.unlink()
    linked.hardlink_to(original)
    with pytest.raises(snapshot.base.SnapshotVerificationError):
        snapshot.read_regular(original)


def test_manifest_cannot_select_different_weights_or_runtime():
    raw = {'schema_version': snapshot.SNAPSHOT_SCHEMA_VERSION, 'model_id': snapshot.MODEL_ID,
        'model_revision': snapshot.MODEL_REVISION, 'context_window': 32768,
        'runtime_versions': snapshot.RUNTIME_VERSION_PINS, 'special_tokens': snapshot.SPECIAL_TOKENS,
        'files': snapshot.SOURCE_FILE_PINS}
    assert snapshot._manifest(json.dumps(raw).encode()).model_revision == snapshot.MODEL_REVISION
    for field, changed in [('files', {**snapshot.SOURCE_FILE_PINS, 'model.gguf': '0' * 64}),
                           ('runtime_versions', {}), ('context_window', True), ('model_revision', 'main')]:
        with pytest.raises(snapshot.base.SnapshotVerificationError):
            snapshot._manifest(json.dumps({**raw, field: changed}).encode())
