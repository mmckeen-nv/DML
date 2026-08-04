"""Command-line entry point for the DCM real-model A/B probe harness."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from daystrom_dml.context.probe import (
    EvaluationSpec,
    OpenAICompatibleModelClient,
    ProbeSettings,
    atomic_write_json,
    attach_runtime_messages,
    build_probe_request,
    run_ab_probe,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    endpoint_url = (
        args.endpoint_url
        or os.environ.get("DCM_MODEL_PROBE_ENDPOINT_URL")
        or os.environ.get("OPENAI_COMPATIBLE_ENDPOINT_URL")
        or ""
    )
    model_id = args.model or os.environ.get("DCM_MODEL_PROBE_MODEL") or os.environ.get("OPENAI_COMPATIBLE_MODEL") or ""
    api_key = (
        args.api_key
        or os.environ.get("DCM_MODEL_PROBE_API_KEY")
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not endpoint_url:
        parser.error("--endpoint-url or DCM_MODEL_PROBE_ENDPOINT_URL is required")
    if not model_id:
        parser.error("--model or DCM_MODEL_PROBE_MODEL is required")

    baseline_messages = _load_json_arg(args.baseline_messages_json, args.baseline_messages_file, "baseline messages")
    managed_messages = _load_json_arg(args.managed_messages_json, args.managed_messages_file, "managed messages")
    authority_manifest = _load_json_arg(
        args.managed_authority_manifest_json,
        args.managed_authority_manifest_file,
        "managed authority manifest",
    )
    settings = ProbeSettings(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    request_contract = build_probe_request(
        run_id=args.run_id,
        endpoint_url=endpoint_url,
        model_id=model_id,
        baseline_messages=baseline_messages,
        managed_messages=managed_messages,
        managed_authority_manifest=authority_manifest,
        settings=settings,
    )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "request": request_contract.to_dict()}, sort_keys=True, indent=2))
        return 0

    if not args.allow_network:
        print("live probe mode requires explicit --allow-network", file=sys.stderr)
        return 2

    attach_runtime_messages(request_contract, baseline_messages, managed_messages)
    client = OpenAICompatibleModelClient(
        api_key=api_key,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
    )
    result = run_ab_probe(
        request_contract,
        client,
        EvaluationSpec(
            required_facts=args.required_fact,
            forbidden_markers=args.forbidden_marker,
            instruction_survival_marker=args.instruction_survival_marker,
        ),
        endpoint_url=endpoint_url,
    )
    payload = result.to_dict()
    print(json.dumps(payload, sort_keys=True, indent=2))
    if args.output_json:
        atomic_write_json(Path(args.output_json), payload)
    return 0 if result.status == "completed" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a provider-neutral DCM baseline/managed model probe.")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print sanitized manifests only")
    parser.add_argument("--allow-network", action="store_true", help="required for live endpoint calls")
    parser.add_argument("--endpoint-url", help="OpenAI-compatible /v1/chat/completions endpoint URL")
    parser.add_argument("--model", help="model identifier passed to the endpoint")
    parser.add_argument("--api-key", help="optional API key; never printed or stored")
    parser.add_argument("--run-id", help="optional stable run id")
    parser.add_argument("--baseline-messages-json", help="baseline messages as a JSON array")
    parser.add_argument("--managed-messages-json", help="managed messages as a JSON array")
    parser.add_argument("--managed-authority-manifest-json", help="managed authority manifest JSON")
    parser.add_argument("--baseline-messages-file", help="path to baseline messages JSON")
    parser.add_argument("--managed-messages-file", help="path to managed messages JSON")
    parser.add_argument("--managed-authority-manifest-file", help="path to managed authority manifest JSON")
    parser.add_argument("--temperature", type=float, default=0, help="sampling temperature, default deterministic 0")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="maximum output tokens")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="request timeout")
    parser.add_argument("--max-request-bytes", type=int, default=256 * 1024, help="request body byte limit")
    parser.add_argument("--max-response-bytes", type=int, default=1024 * 1024, help="response body byte limit")
    parser.add_argument("--required-fact", action="append", default=[], help="fact string expected in both outputs")
    parser.add_argument("--forbidden-marker", action="append", default=[], help="marker that must not appear in outputs")
    parser.add_argument("--instruction-survival-marker", help="marker expected to survive in outputs")
    parser.add_argument("--output-json", help="optional result path, written atomically only when supplied")
    return parser


def _load_json_arg(inline: Optional[str], file_path: Optional[str], label: str) -> Any:
    if inline and file_path:
        raise SystemExit(f"{label}: provide either inline JSON or file path, not both")
    if inline:
        return json.loads(inline)
    if file_path:
        return json.loads(Path(file_path).read_text(encoding="utf-8"))
    raise SystemExit(f"{label}: JSON input is required")


if __name__ == "__main__":
    raise SystemExit(main())
