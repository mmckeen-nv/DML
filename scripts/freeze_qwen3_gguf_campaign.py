#!/usr/bin/env python3
"""Freeze an independently qualified local GGUF candidate; never generate."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from durable_campaign import digest, publish
from freeze_remote_campaign import declared_limits, producer_command


def readiness_attestation(path, expected_sha256, *, commit, profile, identity, snapshots):
    """Require the independently reviewed host/source/synthetic evidence bundle."""
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError("Readiness attestation hash differs")
    receipt = json.loads(path.read_bytes())
    if (receipt.get("schema_version") != "dml-qwen3-gguf-readiness-v1"
            or receipt.get("approved") is not True
            or receipt.get("source_commit") != commit
            or receipt.get("consumer_profile") != profile
            or receipt.get("model_identity") != identity
            or receipt.get("snapshot_sha256") != snapshots
            or any(receipt.get(key) is not True for key in (
                "four_case_qualified", "source_ci_qualified", "lifecycle_qualified"))
            or type(receipt.get("evidence_files")) is not dict or not receipt["evidence_files"]):
        raise ValueError("Readiness does not approve this exact candidate")
    files = {str(path): expected_sha256}
    for name, expected in receipt["evidence_files"].items():
        evidence = Path(name)
        if not evidence.is_absolute() or digest(evidence) != expected:
            raise ValueError("Readiness evidence changed")
        files[str(evidence)] = expected
    return {"path": str(path), "sha256": expected_sha256}, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sqlite-library", type=Path, required=True)
    parser.add_argument("--consumer-profile", required=True)
    parser.add_argument("--readiness-attestation", type=Path, required=True)
    parser.add_argument("--readiness-attestation-sha256", required=True)
    args = parser.parse_args(argv)
    root, source = args.candidate_root.resolve(), args.source_root.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("Freeze requires a clean exact-source checkout")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    sys.path.insert(0, str(source / "dml_core"))
    from daystrom_dml.contracts.agent_episode import QWEN3_GGUF_CONSUMER_PROFILES, execution_protocol_for_profile
    from daystrom_dml.services.agent_episode import EpisodeLimits
    from daystrom_dml.services.episode_verifiers import load_episode_corpus
    from daystrom_dml.services.qwen3_gguf_action_input import LocalQwen3GGUFActionInputConsumer
    from scripts.agent_campaign_evidence import GATES, SPEC_VERSION_V2, _digest
    from scripts.agent_episodes import _source_digests
    profile = args.consumer_profile
    if profile not in QWEN3_GGUF_CONSUMER_PROFILES:
        raise ValueError("Freeze requires an explicit GGUF profile")
    bundle, run = root / "snapshot", root / "campaign-once"
    temporary = root / "tmp"
    if not temporary.is_dir():
        raise ValueError("Persistent private-copy temporary directory is required")
    import os
    if Path(os.environ.get("TMPDIR", "")).resolve() != temporary:
        raise ValueError("Set TMPDIR to the candidate's persistent tmp directory")
    with LocalQwen3GGUFActionInputConsumer(bundle, consumer_profile=profile) as consumer:
        identity = consumer._identity.to_payload()
    snapshots = {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}
    reviewed, attestation_files = readiness_attestation(args.readiness_attestation,
        args.readiness_attestation_sha256, commit=commit, profile=profile,
        identity=identity, snapshots=snapshots)
    corpus = load_episode_corpus()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    sources = {name: digest(source / name) for name in tracked if name}
    required = {"scripts/freeze_qwen3_gguf_campaign.py", "scripts/durable_campaign.py",
        "dml_core/scripts/qwen3_gguf_synthetic.py",
        "dml_core/daystrom_dml/services/qwen3_gguf_action_input.py",
        "dml_core/daystrom_dml/services/qwen3_gguf_model_input.py",
        "dml_core/daystrom_dml/services/qwen3_gguf_model_snapshot.py"}
    if not required <= sources.keys():
        raise ValueError("Required GGUF qualification sources are not tracked")
    runtime_files = {str(Path(sys.executable).resolve()): digest(Path(sys.executable).resolve()),
                     str(args.sqlite_library.resolve()): digest(args.sqlite_library.resolve())}
    versions = {}
    for dist in importlib.metadata.distributions():
        versions[dist.metadata["Name"]] = dist.version
        for relative in dist.files or []:
            path = Path(dist.locate_file(relative)).resolve()
            if path.is_file() and path.suffix != ".pyc":
                runtime_files[str(path)] = digest(path)
    limits = EpisodeLimits(**declared_limits())
    run.mkdir(mode=0o700)
    publish(root / "runtime-inventory.json", {"versions": versions, "files": runtime_files})
    spec = {"schema_version": SPEC_VERSION_V2, "consumer_profile": profile,
        "execution_protocol": execution_protocol_for_profile(profile), "acceptance": GATES,
        "selection": [{"scenario_id": scenario["id"], "task_id": task["id"]}
            for scenario in corpus["scenarios"] for task in scenario["tasks"]],
        "corpus_digest": _digest(corpus), "limits": asdict(limits), "source_sha256": sources,
        "producer_source_sha256": _source_digests(consumer_profile=profile),
        "snapshot_sha256": snapshots, "model_identity": identity,
        "candidate_id": root.name, "frozen_at": time.time(), "source_commit": commit,
        "source_ci_qualified": True, "production_ready": False,
        "runtime_inventory_sha256": digest(root / "runtime-inventory.json"),
        "lifecycle_qualification_sha256": digest(root / "lifecycle/qualification.json"),
        "readiness_attestation": reviewed,
        "limitations": ["Vendor Q4_K_M is not equivalent to original BF16 tensors",
            "Hardware-specific floating-point output is not cross-host deterministic",
            "No physical reboot or power-loss qualification is inferred"],
        "prior_attempt": "Fresh complete campaign; previous Qwen and Nemotron attempts remain immutable and are not resumed"}
    publish(run / "spec.json", spec)
    files = {str(source / name): value for name, value in sources.items()}
    files.update({str(bundle / name): value for name, value in snapshots.items()})
    files.update(runtime_files)
    files.update(attestation_files)
    for path in (run / "spec.json", root / "runtime-inventory.json", root / "lifecycle/qualification.json"):
        files[str(path)] = digest(path)
    freeze = {"files": files,
        "command": producer_command(sys.executable, profile, bundle, run, asdict(limits)),
        "cwd": str(source), "max_seconds": 3300,
        "environment": {"PYTHONPATH": str(source / "dml_core"), "DML_SKIP_VENV_REEXEC": "1",
            "TMPDIR": str(temporary), "LD_LIBRARY_PATH": str(args.sqlite_library.resolve().parent),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}}
    publish(run / "freeze.json", freeze)
    print(json.dumps({"run": str(run), "spec_sha256": digest(run / "spec.json"),
        "freeze_sha256": digest(run / "freeze.json"), "runtime_files": len(runtime_files)}))


if __name__ == "__main__":
    main()
