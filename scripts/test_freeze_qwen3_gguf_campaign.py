"""Readiness must bind the exact local model, source and reviewed evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_qwen3_gguf_campaign import readiness_attestation  # noqa: E402


def declaration(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("immutable evidence")
    receipt = {"schema_version": "dml-qwen3-gguf-readiness-v1", "approved": True,
        "source_commit": "a" * 40, "consumer_profile": "test-profile",
        "model_identity": {"runtime": "test-runtime"}, "snapshot_sha256": {"model.gguf": "b" * 64},
        "four_case_qualified": True, "source_ci_qualified": True, "lifecycle_qualified": True,
        "evidence_files": {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()}}
    return receipt, evidence


def check(tmp_path, receipt):
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(receipt))
    return readiness_attestation(path, hashlib.sha256(path.read_bytes()).hexdigest(),
        commit="a" * 40, profile="test-profile", identity={"runtime": "test-runtime"},
        snapshots={"model.gguf": "b" * 64})


def test_exact_readiness_binds_evidence_files(tmp_path):
    receipt, evidence = declaration(tmp_path)
    approved, files = check(tmp_path, receipt)
    assert approved["path"] == str(tmp_path / "readiness.json")
    assert str(evidence) in files


@pytest.mark.parametrize("key,value", [
    ("approved", False), ("source_commit", "c" * 40), ("consumer_profile", "other"),
    ("model_identity", {"runtime": "other"}), ("snapshot_sha256", {}),
    ("four_case_qualified", False), ("source_ci_qualified", False),
    ("lifecycle_qualified", False), ("evidence_files", {}),
])
def test_readiness_rejects_unqualified_or_different_candidate(tmp_path, key, value):
    receipt, _ = declaration(tmp_path)
    receipt[key] = value
    with pytest.raises(ValueError, match="exact candidate"):
        check(tmp_path, receipt)


def test_readiness_rejects_modified_retained_evidence(tmp_path):
    receipt, evidence = declaration(tmp_path)
    evidence.write_text("changed evidence")
    with pytest.raises(ValueError, match="evidence changed"):
        check(tmp_path, receipt)


def test_readiness_requires_the_reviewed_receipt_hash(tmp_path):
    receipt, _ = declaration(tmp_path)
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash differs"):
        readiness_attestation(path, "0" * 64, commit="a" * 40, profile="test-profile",
            identity={"runtime": "test-runtime"}, snapshots={"model.gguf": "b" * 64})
