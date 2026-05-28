#!/usr/bin/env python3
"""Prepare a DML-assisted frontier-model prompt through the local provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get("DML_PROVIDER_URL", "http://127.0.0.1:8765")


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --prompt, --prompt-file, or stdin")


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach DML provider at {url}: {exc.reason}") from exc


def _compact_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    telemetry = dict(result.get("telemetry") or {})
    return {
        "dml_context_tokens": telemetry.get("dml_context_tokens"),
        "frontier_input_tokens": telemetry.get("frontier_input_tokens"),
        "direct_input_tokens_estimate": telemetry.get("direct_input_tokens_estimate"),
        "input_tokens_saved_estimate": telemetry.get("input_tokens_saved_estimate"),
        "input_savings_pct_estimate": telemetry.get("input_savings_pct_estimate"),
        "retrieval_latency_ms": telemetry.get("retrieval_latency_ms"),
        "retrieved_items": telemetry.get("retrieved_items"),
        "survival_ledger_included": telemetry.get("survival_ledger_included"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a DML-scoped frontier prompt for long-horizon agent inference.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--tenant-id", default=os.environ.get("DML_TENANT_ID", "openclaw"))
    parser.add_argument("--client-id")
    parser.add_argument("--session-id", default=os.environ.get("DML_SESSION_ID"))
    parser.add_argument("--instance-id")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--local-max-tokens", type=int, default=256)
    parser.add_argument("--frontier-max-tokens", type=int, default=1200)
    parser.add_argument("--direct-input-tokens-estimate", type=int)
    parser.add_argument("--no-local-draft", action="store_true")
    parser.add_argument("--frontier-prompt-only", action="store_true")
    parser.add_argument("--telemetry-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = {
        "prompt": _read_prompt(args),
        "tenant_id": args.tenant_id,
        "client_id": args.client_id,
        "session_id": args.session_id,
        "instance_id": args.instance_id,
        "top_k": args.top_k,
        "local_max_tokens": args.local_max_tokens,
        "frontier_max_tokens": args.frontier_max_tokens,
        "include_local_draft": not args.no_local_draft,
        "direct_input_tokens_estimate": args.direct_input_tokens_estimate,
    }
    result = _post_json(
        f"{args.base_url.rstrip('/')}/api/frontier/prepare",
        payload,
        args.timeout_s,
    )
    if args.frontier_prompt_only:
        print(result.get("frontier_prompt") or "")
        return 0
    if args.telemetry_only:
        print(json.dumps(_compact_telemetry(result), indent=2, sort_keys=True))
        return 0
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
