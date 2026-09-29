#!/usr/bin/env python3
"""Copy cached model files, verifying actual bytes against an immutable HF revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("An exact commit SHA is required")
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(args.repository, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Official immutable revision differs")
    siblings = {item.rfilename: item for item in info.siblings}
    index = json.loads((args.cache_snapshot / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values()) | {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json",
        "configuration_nemotron_h.py",
    }
    names.update(name for name in siblings if name.endswith(".py"))
    if any(Path(name).name != name or name not in siblings for name in names):
        raise ValueError("Unexpected load-related model file")
    args.destination.mkdir(mode=0o700)
    started = time.time()
    records = {}
    for name in sorted(names):
        metadata = siblings[name]
        source = args.cache_snapshot / name
        if not source.exists():
            source = Path(hf_hub_download(args.repository, name, revision=args.revision,
                                         cache_dir=str(args.destination.parent / "source-cache")))
        size = source.stat().st_size
        if metadata.size != size:
            raise ValueError("Official size mismatch: " + name)
        sha256 = hashlib.sha256()
        gitblob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        target = args.destination / name
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                sha256.update(block)
                gitblob.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if metadata.lfs:
            expected = metadata.lfs.sha256
            if sha256.hexdigest() != expected:
                raise ValueError("Official LFS SHA256 mismatch: " + name)
            method = "official_huggingface_lfs_sha256"
        else:
            expected = metadata.blob_id
            if gitblob.hexdigest() != expected:
                raise ValueError("Official Git blob SHA1 mismatch: " + name)
            method = "official_huggingface_git_blob_sha1"
        target.chmod(0o400)
        records[name] = {"size": size, "sha256": sha256.hexdigest(),
                         "official_digest": expected, "verification": method}
        print(json.dumps({"verified": name, "size": size}), flush=True)
    args.destination.chmod(0o500)
    receipt = {"schema_version": "dml-pinned-model-bytes-v1", "repository": args.repository,
        "revision": args.revision, "destination": str(args.destination.resolve()),
        "official_revision_observed": info.sha, "started_at": started, "completed_at": time.time(),
        "files": records, "total_bytes": sum(v["size"] for v in records.values()),
        "attestation_scope": "host-verified artifact bytes; not yet loaded-engine or GPU-tensor attestation"}
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for path in {args.destination, args.destination.parent, args.receipt.parent}:
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


if __name__ == "__main__":
    main()
