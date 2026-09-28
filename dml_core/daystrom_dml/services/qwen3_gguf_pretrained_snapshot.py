"""Prepare the pinned official quantized artifact without modifying model bytes."""
from __future__ import annotations

from pathlib import Path

from . import qwen3_gguf_model_snapshot as snapshot
from .pretrained_snapshot import _json_bytes
from .qwen3_pretrained_snapshot import _publish_directory

PREPARER_VERSION = "dml-qwen3-8b-official-gguf-v1"
MODEL_ID = snapshot.MODEL_ID
MODEL_REVISION = snapshot.MODEL_REVISION
SOURCE_FILE_PINS = snapshot.SOURCE_FILE_PINS
GGUF_SOURCE_FILENAME = "Qwen3-8B-Q4_K_M.gguf"
GGUF_SOURCE_SIZE = 5027783488


def prepare_qwen3_gguf_snapshot(stage_directory: str | Path, destination: str | Path,
                                *, context_window: int = 32768) -> dict:
    """Own a fresh staged directory; publish once using Linux no-replace rename.

    Stage contains exactly the five source files under their runtime names.
    Failure retains stage and any generated control files for inspection.
    """
    stage, info = snapshot._directory(stage_directory)
    destination = Path(destination).absolute()
    snapshot._directory(destination.parent)
    if destination.exists() or {p.name for p in stage.iterdir()} != snapshot.REQUIRED_FILES - {"chat_template.jinja"}:
        raise snapshot.base.SnapshotVerificationError("Preparation requires fresh exact source inventory")
    if type(context_window) is not int or not 1 <= context_window <= 32768:
        raise snapshot.base.SnapshotVerificationError("Invalid GGUF context window")
    hashes = {}
    for name in sorted(snapshot.REQUIRED_FILES - {"chat_template.jinja"}):
        _, hashes[name] = snapshot.read_regular(stage / name)
        if hashes[name] != SOURCE_FILE_PINS[name]:
            raise snapshot.base.SnapshotVerificationError("Original GGUF source differs from immutable pin")
    if (stage / "model.gguf").stat().st_size != GGUF_SOURCE_SIZE:
        raise snapshot.base.SnapshotVerificationError("GGUF source size differs")
    snapshot.parse_gguf(stage / "model.gguf")
    with (stage / "chat_template.jinja").open("xb") as handle:
        handle.write(snapshot.QWEN3_CHAT_TEMPLATE.encode())
    manifest = {"schema_version": snapshot.SNAPSHOT_SCHEMA_VERSION, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "context_window": context_window,
        "runtime_versions": snapshot.runtime_versions(), "special_tokens": snapshot.SPECIAL_TOKENS,
        "files": dict(SOURCE_FILE_PINS)}
    with (stage / "snapshot.json").open("xb") as handle:
        handle.write(_json_bytes(manifest))
    # The actual private-copy verifier admits exactly the artifact that will be published.
    with snapshot.verify_qwen3_gguf_snapshot(stage) as verified:
        identity = verified.identity
    _publish_directory(stage, destination, (info.st_dev, info.st_ino))
    return {"preparer_version": PREPARER_VERSION, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "metadata_model_id": "Qwen/Qwen3-8B", "metadata_revision": snapshot.HF_MODEL_REVISION,
        "quantized_artifact_sha256": snapshot.GGUF_SHA256, "source_files": hashes,
        "snapshot_directory": str(destination), "context_window": context_window,
        "model_digest": identity.model_digest, "bf16_tensor_equivalence_claimed": False,
        "conversion_performed": False, "generation_calls": 0}
