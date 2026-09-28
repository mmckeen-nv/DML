#!/usr/bin/env python3
"""Freeze one remote candidate; this never launches model generation."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from durable_campaign import digest, publish


def declared_limits():
    """Use the same JSON number types as the producer's argument parser."""
    return dict(max_steps=6, output_tokens=256, max_input_tokens=32768,
        max_output_tokens=1536, max_transcript_bytes=262144, max_event_bytes=4194304,
        max_episode_bytes=16777216, wall_time_seconds=300.0)


def selected_profile(manifest, requested=None):
    from daystrom_dml.contracts.agent_episode import REMOTE_VLLM_CONSUMER_PROFILE, REMOTE_VLLM_CONSUMER_PROFILES
    declared = manifest.get("consumer_profile", REMOTE_VLLM_CONSUMER_PROFILE)
    if declared not in REMOTE_VLLM_CONSUMER_PROFILES or requested not in (None, declared):
        raise ValueError("Requested profile differs from remote manifest")
    return declared


def runtime_attestation(path, expected_sha256, manifest, bundle):
    """Bind reviewed host evidence; never infer loaded revision from a cache."""
    if path is None and expected_sha256 is None:
        return None, {}
    if path is None or expected_sha256 is None:
        raise ValueError("Runtime attestation requires both path and reviewed SHA256")
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError("Reviewed runtime attestation digest differs")
    receipt = json.loads(path.read_bytes())
    if (receipt.get("schema_version") != "dml-remote-runtime-attestation-v1"
            or receipt.get("loaded_model_revision_attested") is not True
            or any(receipt.get(key) != manifest[key] for key in ("endpoint", "model", "model_revision"))
            or receipt.get("manifest_sha256") != digest(Path(bundle) / "remote-vllm-manifest.json")
            or not isinstance(receipt.get("evidence_files"), dict) or not receipt["evidence_files"]):
        raise ValueError("Runtime attestation does not bind the selected model/runtime")
    files = {str(path): expected_sha256}
    for name, expected in receipt["evidence_files"].items():
        evidence = Path(name)
        if not evidence.is_absolute() or digest(evidence) != expected:
            raise ValueError("Runtime attestation evidence differs")
        files[str(evidence)] = expected
    return {"path": str(path), "sha256": expected_sha256,
            "scope": "reviewed host launch and pinned model files; HTTP does not attest engine tensors"}, files


def producer_command(executable, profile, bundle, run, limits):
    command = [str(executable), "-m", "scripts.agent_episodes", "--consumer-profile", profile,
        "--snapshot-directory", str(bundle), "--work-directory", str(run / "episodes"),
        "--output", str(run / "campaign.json")]
    for key, value in limits.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sqlite-library", type=Path, required=True)
    parser.add_argument("--consumer-profile", help="Must match manifest; omitted selects manifest or legacy v1")
    parser.add_argument("--runtime-attestation", type=Path)
    parser.add_argument("--runtime-attestation-sha256", help="SHA256 independently reviewed before freezing")
    args = parser.parse_args(argv)
    root, source = args.candidate_root.resolve(), args.source_root.resolve()
    run = root / "campaign-once"
    sys.path.insert(0, str(source / "dml_core"))
    from daystrom_dml.contracts.agent_episode import REMOTE_VLLM_JSON_CONSUMER_PROFILE, NATIVE_REMOTE_VLLM_CONSUMER_PROFILES, NATIVE_RECOVERY_CONSUMER_PROFILES, execution_protocol_for_profile
    from daystrom_dml.services.agent_episode import EpisodeLimits
    from daystrom_dml.services.episode_verifiers import load_episode_corpus
    from daystrom_dml.services.remote_vllm_action_input import verify_remote_manifest, RemoteVLLMActionInputConsumer, EXACT_TOKEN_LIMITATIONS
    from scripts.agent_campaign_evidence import GATES, SPEC_VERSION_V2, _digest
    from scripts.agent_episodes import _source_digests
    bundle = root / "snapshot"
    manifest = verify_remote_manifest(bundle)
    profile = selected_profile(manifest, args.consumer_profile)
    consumer_type = RemoteVLLMActionInputConsumer
    limitations = EXACT_TOKEN_LIMITATIONS
    if profile in NATIVE_REMOTE_VLLM_CONSUMER_PROFILES:
        from daystrom_dml.services.native_remote_vllm_action_input import NativeRemoteVLLMActionInputConsumer, NATIVE_EXACT_TOKEN_LIMITATIONS
        consumer_type = NativeRemoteVLLMActionInputConsumer
        limitations = NATIVE_EXACT_TOKEN_LIMITATIONS
    with consumer_type(bundle, consumer_profile=profile, offline=True) as consumer:
        model_identity = consumer.identity.to_payload()
    attestation, attestation_files = runtime_attestation(args.runtime_attestation,
        args.runtime_attestation_sha256, manifest, bundle)
    run.mkdir(mode=0o700)
    corpus = load_episode_corpus()
    source_files = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    sources = {name: digest(source / name) for name in source_files if name}
    if profile in (REMOTE_VLLM_JSON_CONSUMER_PROFILE, *NATIVE_REMOTE_VLLM_CONSUMER_PROFILES):
        required = {"dml_core/daystrom_dml/services/remote_vllm_action_input.py",
                    "dml_core/daystrom_dml/services/qwen_model_snapshot.py",
                    "dml_core/scripts/remote_vllm_synthetic.py", "scripts/freeze_remote_campaign.py"}
        if profile in NATIVE_REMOTE_VLLM_CONSUMER_PROFILES:
            required |= {"dml_core/daystrom_dml/services/native_remote_vllm_action_input.py",
                         "dml_core/scripts/native_dml_synthetic.py"}
        if profile in NATIVE_RECOVERY_CONSUMER_PROFILES:
            required |= {"dml_core/daystrom_dml/services/receipt_conflict_boundary.py",
                         "dml_core/daystrom_dml/services/receipt_lifecycle.py",
                         "dml_core/daystrom_dml/services/receipt_supersession.py",
                         "dml_core/daystrom_dml/journal.py",
                         "dml_core/scripts/native_recovery_suite.py",
                         "dml_core/scripts/native_recovery_synthetic.py"}
        if not required <= sources.keys():
            raise ValueError("V2 freeze requires tracked renderer, synthetic qualification and freeze sources")
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
    limits = EpisodeLimits(**declared_limits())
    spec = {"schema_version": SPEC_VERSION_V2, "consumer_profile": profile,
        "execution_protocol": execution_protocol_for_profile(profile), "acceptance": GATES,
        "selection": [{"scenario_id": s["id"], "task_id": t["id"]}
            for s in corpus["scenarios"] for t in s["tasks"]],
        "corpus_digest": _digest(corpus), "limits": asdict(limits), "source_sha256": sources,
        "producer_source_sha256": _source_digests(consumer_profile=profile), "snapshot_sha256": snapshots,
        "model_identity": model_identity,
        "candidate_id": root.name, "frozen_at": time.time(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_ci_qualified": False, "production_ready": False,
        "runtime_inventory_sha256": digest(root / "runtime-inventory.json"),
        "lifecycle_qualification_sha256": digest(root / "lifecycle/qualification.json"),
        "exact_token_limitations": limitations,
        "loaded_model_revision_attested": attestation is not None, "runtime_attestation": attestation,
        "prior_attempt": "Historical attempt15 remains interrupted and is not resumed; no historical raw evidence reconstructed"}
    publish(run / "spec.json", spec)
    command = producer_command(sys.executable, profile, bundle, run, asdict(limits))
    files = {str(source / name): value for name, value in sources.items()}
    files.update({str(bundle / name): value for name, value in snapshots.items()})
    files.update(runtime_files)
    files.update(attestation_files)
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
