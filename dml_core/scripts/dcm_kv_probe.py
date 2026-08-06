"""Managed real-runtime KV save/erase/restore/purge probe with digest-only output."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

from daystrom_dml.api_contracts import DaystromScope
from daystrom_dml.context.adapters.llama_cpp import LlamaCppExecutionAdapter
from daystrom_dml.context.admission import admit_context_segments
from daystrom_dml.context.checkpoints import (
    ExecutionCheckpointIdentity,
    FileExecutionCheckpointRegistry,
)
from daystrom_dml.context.execution import RuntimeExecutionError
from daystrom_dml.context.probe import atomic_write_json
from daystrom_dml.context.runtime_probe import RUNTIME_EXECUTION_PROBE_V2, run_runtime_execution_probe
from daystrom_dml.context.schema import ContextAuthority, ContextPriority, ContextSegment


def run_probe(
    *,
    endpoint_url: str,
    runtime_id: str,
    runtime_version: str,
    checkpoint_directory: Path,
    registry_directory: Path,
    model_id: str,
    model_digest: str,
    tokenizer_digest: str,
    positional_config: dict[str, Any],
    tenant_id: str,
    session_id: str,
    slot_id: int,
    checkpoint_id: str,
    records: int,
    model_limit_tokens: int,
    ttl_seconds: float,
) -> dict[str, Any]:
    if records <= 0 or records > 10_000:
        raise RuntimeExecutionError("records must be between 1 and 10000")
    if model_limit_tokens <= 0:
        raise RuntimeExecutionError("model_limit_tokens must be positive")
    checkpoint_directory = checkpoint_directory.expanduser().resolve()
    registry_directory = registry_directory.expanduser().resolve()
    if not checkpoint_directory.is_dir():
        raise RuntimeExecutionError("checkpoint_directory must be an existing directory")

    prefix = "".join(
        f"Record {index:04d}: the immutable project marker is ORBIT; preserve this line exactly.\n"
        for index in range(records)
    )
    prompt = prefix + "Question: reply with the project marker. Answer:"
    estimated_tokens = max(1, (len(prompt) + 3) // 4)
    if estimated_tokens > model_limit_tokens:
        raise RuntimeExecutionError("probe prompt exceeds model_limit_tokens")

    scope = DaystromScope(tenant_id=tenant_id, session_id=session_id)
    packet = admit_context_segments(
        scope=scope,
        segments=[
            ContextSegment(
                segment_id="runtime-probe-prefix",
                kind="runtime_probe",
                content=prompt,
                authority=ContextAuthority.IMMUTABLE,
                priority=ContextPriority.CRITICAL,
                scope=scope,
                estimated_tokens=estimated_tokens,
            )
        ],
        model_id=model_id,
        runtime_id=runtime_id,
        endpoint_url=endpoint_url,
        model_limit_tokens=model_limit_tokens,
    )
    adapter = LlamaCppExecutionAdapter(
        endpoint_url,
        runtime_id=runtime_id,
        runtime_version=runtime_version,
        checkpoint_directory=checkpoint_directory,
    )
    identity = ExecutionCheckpointIdentity.from_packet(
        packet,
        adapter.capabilities(),
        model_digest=model_digest,
        tokenizer_digest=tokenizer_digest,
        positional_config=positional_config,
        immutable_prefix=prompt,
        runtime_endpoint_url=endpoint_url,
    )
    return run_runtime_execution_probe(
        adapter=adapter,
        registry=FileExecutionCheckpointRegistry(registry_directory),
        identity=identity,
        prompt=prompt,
        slot_id=slot_id,
        checkpoint_id=checkpoint_id,
        ttl_seconds=ttl_seconds,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:18080")
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--registry-directory", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-digest", required=True, help="Exact sha256:<64 lowercase hex> model digest")
    parser.add_argument("--tokenizer-digest", required=True, help="Exact sha256:<64 lowercase hex> tokenizer digest")
    parser.add_argument("--positional-config-json", required=True, help="Non-empty JSON object binding context/RoPE settings")
    parser.add_argument("--tenant-id", default="local-probe")
    parser.add_argument("--session-id")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--records", type=int, default=160)
    parser.add_argument("--model-limit-tokens", type=int, required=True)
    parser.add_argument("--ttl-seconds", type=float, default=300.0)
    parser.add_argument("--artifact", type=Path, default=Path.cwd() / "dcm-kv-execution-probe.json")
    args = parser.parse_args(argv)

    try:
        positional_config = json.loads(args.positional_config_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--positional-config-json is invalid JSON: {exc.msg}")
    if not isinstance(positional_config, dict) or not positional_config:
        parser.error("--positional-config-json must be a non-empty JSON object")

    session_id = args.session_id or f"probe-{uuid.uuid4().hex}"
    checkpoint_id = args.checkpoint_id or f"probe-{uuid.uuid4().hex}"
    registry_context: Any
    if args.registry_directory is None:
        registry_context = tempfile.TemporaryDirectory(prefix="dcm-kv-registry-")
    else:
        args.registry_directory.mkdir(parents=True, exist_ok=True)
        registry_context = nullcontext(str(args.registry_directory))

    try:
        with registry_context as registry_value:
            result = run_probe(
                endpoint_url=args.endpoint_url,
                runtime_id=args.runtime_id,
                runtime_version=args.runtime_version,
                checkpoint_directory=args.checkpoint_directory,
                registry_directory=Path(registry_value),
                model_id=args.model_id,
                model_digest=args.model_digest,
                tokenizer_digest=args.tokenizer_digest,
                positional_config=positional_config,
                tenant_id=args.tenant_id,
                session_id=session_id,
                slot_id=args.slot_id,
                checkpoint_id=checkpoint_id,
                records=args.records,
                model_limit_tokens=args.model_limit_tokens,
                ttl_seconds=args.ttl_seconds,
            )
    except Exception as exc:
        reason_digest = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()
        result = {
            "artifact_version": RUNTIME_EXECUTION_PROBE_V2,
            "pass": False,
            "error_type": type(exc).__name__,
            "reason_digest": reason_digest,
        }

    atomic_write_json(args.artifact, result)
    print(args.artifact)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
