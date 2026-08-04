"""Real llama.cpp KV-cache reuse/checkpoint probe with digest-only output."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

from daystrom_dml.context.adapters.llama_cpp import LlamaCppExecutionAdapter
from daystrom_dml.context.probe import atomic_write_json


def run_probe(
    *,
    endpoint_url: str,
    runtime_id: str,
    runtime_version: str,
    slot_id: int,
    checkpoint_name: str,
    records: int,
) -> dict[str, Any]:
    adapter = LlamaCppExecutionAdapter(
        endpoint_url,
        runtime_id=runtime_id,
        runtime_version=runtime_version,
    )
    prefix = "".join(
        f"Record {index:04d}: the immutable project marker is ORBIT; preserve this line exactly.\n"
        for index in range(records)
    )
    prompt = prefix + "Question: reply with the project marker. Answer:"

    adapter.erase_slot(slot_id)
    cold = adapter.complete(prompt, slot_id=slot_id, n_predict=1, seed=7)
    hot = adapter.complete(prompt, slot_id=slot_id, n_predict=1, seed=7)
    saved = adapter.save_slot(slot_id, checkpoint_name)
    adapter.erase_slot(slot_id)
    restored = adapter.restore_slot(slot_id, checkpoint_name)
    restored_run = adapter.complete(prompt, slot_id=slot_id, n_predict=1, seed=7)

    equivalent = cold.output_token_ids == hot.output_token_ids == restored_run.output_token_ids
    passed = (
        cold.prompt_tokens_processed > hot.prompt_tokens_processed
        and cold.prompt_tokens_processed > restored_run.prompt_tokens_processed
        and hot.prompt_tokens_reused > 0
        and restored_run.prompt_tokens_reused > 0
        and equivalent
    )
    return {
        "artifact_version": "dcm-kv-execution-probe-v1",
        "runtime": adapter.capabilities().to_dict(),
        "prefix_digest": hashlib.sha256(prefix.encode()).hexdigest(),
        "cold": cold.to_telemetry(),
        "hot": hot.to_telemetry(),
        "checkpoint": saved.to_dict(),
        "restore": restored.to_dict(),
        "restored_run": restored_run.to_telemetry(),
        "output_token_equivalent": equivalent,
        "pass": passed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:18080")
    parser.add_argument("--runtime-id", default="llama-cpp-local")
    parser.add_argument("--runtime-version", default="unknown")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="dcm-prefix-cache.bin")
    parser.add_argument("--records", type=int, default=160)
    parser.add_argument("--artifact", type=Path, default=Path("/tmp/dcm-kv-execution-probe.json"))
    args = parser.parse_args(argv)

    result = run_probe(
        endpoint_url=args.endpoint_url,
        runtime_id=args.runtime_id,
        runtime_version=args.runtime_version,
        slot_id=args.slot_id,
        checkpoint_name=args.checkpoint_name,
        records=args.records,
    )
    atomic_write_json(args.artifact, result)
    print(args.artifact)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
