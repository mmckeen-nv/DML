#!/usr/bin/env python3
"""Freeze one remote candidate; this never launches model generation."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from durable_campaign import digest, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sqlite-library", type=Path, required=True)
    args = parser.parse_args()
    root, source = args.candidate_root.resolve(), args.source_root.resolve()
    run = root / "campaign-once"
    run.mkdir(mode=0o700)
    sys.path.insert(0, str(source / "dml_core"))
    from daystrom_dml.contracts.agent_episode import REMOTE_VLLM_CONSUMER_PROFILE, execution_protocol_for_profile
    from daystrom_dml.services.agent_episode import EpisodeLimits
    from daystrom_dml.services.episode_verifiers import load_episode_corpus
    from daystrom_dml.services.remote_vllm_action_input import verify_remote_manifest, remote_identity, EXACT_TOKEN_LIMITATIONS
    from scripts.agent_campaign_evidence import GATES, SPEC_VERSION_V2, _digest
    from scripts.agent_episodes import _source_digests
    bundle = root / "snapshot"
    manifest = verify_remote_manifest(bundle)
    corpus = load_episode_corpus()
    source_files = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    sources = {name: digest(source / name) for name in source_files if name}
    snapshots = {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}
    runtime_files = {str(Path(sys.executable).resolve()): digest(Path(sys.executable).resolve()),
                     str(args.sqlite_library.resolve()): digest(args.sqlite_library.resolve())}
    versions = {}
    for dist in importlib.metadata.distributions():
        versions[dist.metadata["Name"]] = dist.version
        for relative in dist.files or []:
            path = Path(dist.locate_file(relative)).resolve()
            if path.is_file() and path.suffix != ".pyc":
                runtime_files[str(path)] = digest(path)
    publish(root / "runtime-inventory.json", {"versions": versions, "files": runtime_files})
    limits = EpisodeLimits(max_steps=6, output_tokens=256, max_input_tokens=32768,
        max_output_tokens=1536, max_transcript_bytes=262144, max_event_bytes=4194304,
        max_episode_bytes=16777216, wall_time_seconds=300)
    profile = REMOTE_VLLM_CONSUMER_PROFILE
    spec = {"schema_version": SPEC_VERSION_V2, "consumer_profile": profile,
        "execution_protocol": execution_protocol_for_profile(profile), "acceptance": GATES,
        "selection": [{"scenario_id": s["id"], "task_id": t["id"]}
            for s in corpus["scenarios"] for t in s["tasks"]],
        "corpus_digest": _digest(corpus), "limits": asdict(limits), "source_sha256": sources,
        "producer_source_sha256": _source_digests(consumer_profile=profile), "snapshot_sha256": snapshots,
        "model_identity": remote_identity(manifest).to_payload(),
        "candidate_id": root.name, "frozen_at": time.time(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_ci_qualified": False, "production_ready": False,
        "runtime_inventory_sha256": digest(root / "runtime-inventory.json"),
        "lifecycle_qualification_sha256": digest(root / "lifecycle/qualification.json"),
        "exact_token_limitations": EXACT_TOKEN_LIMITATIONS,
        "loaded_model_revision_attested": False,
        "prior_attempt": "Historical attempt15 remains interrupted and is not resumed; no historical raw evidence reconstructed"}
    publish(run / "spec.json", spec)
    command = [sys.executable, "-m", "scripts.agent_episodes", "--consumer-profile", profile,
        "--snapshot-directory", str(bundle), "--work-directory", str(run / "episodes"),
        "--output", str(run / "campaign.json")]
    for key, value in asdict(limits).items():
        command += ["--" + key.replace("_", "-"), str(value)]
    files = {str(source / name): value for name, value in sources.items()}
    files.update({str(bundle / name): value for name, value in snapshots.items()})
    files.update(runtime_files)
    for path in [run / "spec.json", root / "runtime-inventory.json", root / "lifecycle/qualification.json"]:
        files[str(path)] = digest(path)
    freeze = {"files": files, "command": command, "cwd": str(source), "max_seconds": 3300,
        "environment": {"PYTHONPATH": str(source / "dml_core"), "DML_SKIP_VENV_REEXEC": "1",
            "LD_LIBRARY_PATH": str(args.sqlite_library.resolve().parent),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}}
    publish(run / "freeze.json", freeze)
    print(json.dumps({"run": str(run), "spec_sha256": digest(run / "spec.json"),
                      "freeze_sha256": digest(run / "freeze.json"), "runtime_files": len(runtime_files)}))


if __name__ == "__main__":
    main()
