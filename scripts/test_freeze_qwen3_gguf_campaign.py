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


@pytest.mark.parametrize("fault", [None, "missing", "unbound", "not_gpu", "wrong_source", "wrong_identity", "wrong_profile"])
@pytest.mark.parametrize("profile", ["qwen3-8b-gguf-cuda-action-json-completion-v1",
                                     "qwen3-8b-gguf-cuda-action-json-retrieval-v2"])
def test_cuda_readiness_requires_actual_bound_gpu_admission(tmp_path, fault, profile):
    receipt, _ = declaration(tmp_path)
    receipt['consumer_profile'] = profile
    gpu = {'passed': True, 'gpu_inference_verified': True, 'source_commit': receipt['source_commit'],
        'consumer_profile': profile, 'model_identity': receipt['model_identity']}
    if fault == 'wrong_profile':
        gpu['consumer_profile'] = 'different-gpu-policy'
    elif fault == 'not_gpu':
        gpu['gpu_inference_verified'] = False
    elif fault == 'wrong_source':
        gpu['source_commit'] = 'c' * 40
    elif fault == 'wrong_identity':
        gpu['model_identity'] = {'runtime': 'cpu-runtime'}
    path = tmp_path / 'gpu-admission.json'
    path.write_text(json.dumps(gpu))
    gpu_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt['gpu_admission_qualified'] = True
    receipt['gpu_admission_evidence'] = {'path': str(path), 'sha256': gpu_hash}
    receipt['evidence_files'][str(path)] = gpu_hash
    if profile.endswith('retrieval-v2'):
        add_planning(tmp_path, receipt)
    if fault == 'missing':
        receipt.pop('gpu_admission_qualified')
    elif fault == 'unbound':
        receipt['evidence_files'].pop(str(path))
    ready = tmp_path / 'readiness.json'
    ready.write_text(json.dumps(receipt))
    kwargs = dict(commit=receipt['source_commit'], profile=profile,
        identity=receipt['model_identity'], snapshots=receipt['snapshot_sha256'])
    if fault is None:
        _, files = readiness_attestation(ready, hashlib.sha256(ready.read_bytes()).hexdigest(), **kwargs)
        assert files[str(path)] == gpu_hash
    else:
        with pytest.raises(ValueError, match='GPU'):
            readiness_attestation(ready, hashlib.sha256(ready.read_bytes()).hexdigest(), **kwargs)


def test_freezer_cuda_profiles_match_authoritative_registry():
    from daystrom_dml.contracts.agent_episode import QWEN3_GGUF_CUDA_CONSUMER_PROFILES
    from freeze_qwen3_gguf_campaign import CUDA_PROFILES
    assert CUDA_PROFILES == QWEN3_GGUF_CUDA_CONSUMER_PROFILES


def add_planning(tmp_path, receipt):
    evidence, roles = {}, {}
    for role in ('suite', 'result', 'primary_replay', 'independent_replay'):
        path = tmp_path / (role + '.json')
        path.write_text(json.dumps({'role': role}))
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        roles[role] = str(path)
    planning = {'passed': True, 'source_commit': receipt['source_commit'],
        'consumer_profile': receipt['consumer_profile'], 'model_identity': receipt['model_identity'],
        'cases_passed': 2, 'cases_declared': 2, 'evidence_files': evidence, 'evidence_roles': roles}
    receipt['evidence_files'].update(evidence)
    path = tmp_path / 'planning.json'
    path.write_text(json.dumps(planning))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt['planning_qualified'] = True
    receipt['planning_evidence'] = {'path': str(path), 'sha256': sha}
    receipt['evidence_files'][str(path)] = sha
    return planning, path


@pytest.mark.parametrize('fault', [None, 'missing', 'unbound_receipt', 'failed', 'partial',
    'wrong_source', 'wrong_profile', 'wrong_identity', 'missing_replay', 'unbound_file', 'same_replays', 'tampered_file'])
def test_retrieval_readiness_requires_two_case_planning_and_both_replays(tmp_path, fault):
    receipt, _ = declaration(tmp_path)
    profile = 'qwen3-8b-gguf-cuda-action-json-retrieval-v2'
    receipt['consumer_profile'] = profile
    gpu = tmp_path / 'gpu.json'
    gpu.write_text(json.dumps({'passed': True, 'gpu_inference_verified': True,
        'source_commit': receipt['source_commit'], 'consumer_profile': profile,
        'model_identity': receipt['model_identity']}))
    sha = hashlib.sha256(gpu.read_bytes()).hexdigest()
    receipt.update(gpu_admission_qualified=True, gpu_admission_evidence={'path': str(gpu), 'sha256': sha})
    receipt['evidence_files'][str(gpu)] = sha
    planning, path = add_planning(tmp_path, receipt)
    if fault == 'missing':
        receipt.pop('planning_qualified')
    elif fault == 'failed':
        planning['passed'] = False
    elif fault == 'partial':
        planning['cases_passed'] = 1
    elif fault == 'wrong_source':
        planning['source_commit'] = 'b' * 40
    elif fault == 'wrong_profile':
        planning['consumer_profile'] = 'qwen3-8b-gguf-cuda-action-json-completion-v1'
    elif fault == 'wrong_identity':
        planning['model_identity'] = {'runtime': 'previous-runtime'}
    elif fault == 'missing_replay':
        planning['evidence_roles'].pop('independent_replay')
    elif fault == 'same_replays':
        planning['evidence_roles']['independent_replay'] = planning['evidence_roles']['primary_replay']
    elif fault == 'unbound_file':
        receipt['evidence_files'].pop(planning['evidence_roles']['result'])
    elif fault == 'tampered_file':
        Path(planning['evidence_roles']['result']).write_text('altered')
    path.write_text(json.dumps(planning))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt['planning_evidence']['sha256'] = sha
    receipt['evidence_files'][str(path)] = sha
    if fault == 'unbound_receipt':
        receipt['evidence_files'].pop(str(path))
    ready = tmp_path / 'readiness.json'
    ready.write_text(json.dumps(receipt))
    kwargs = dict(commit=receipt['source_commit'], profile=profile,
        identity=receipt['model_identity'], snapshots=receipt['snapshot_sha256'])
    if fault is None:
        _, files = readiness_attestation(ready, hashlib.sha256(ready.read_bytes()).hexdigest(), **kwargs)
        assert all(name in files for name in planning['evidence_files'])
    else:
        with pytest.raises(ValueError, match='[Pp]lanning|evidence changed'):
            readiness_attestation(ready, hashlib.sha256(ready.read_bytes()).hexdigest(), **kwargs)
