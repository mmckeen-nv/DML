#!/usr/bin/env python3
"""Call OpenClaw/NVIDIA Responses-compatible frontier endpoints for swarm workers.

Reads secrets from environment only. Do not put API keys in prompts, logs, or repo files.

Example:
    OPENCLAW_INFERENCE_API_KEY=... \
    python scripts/frontier_swarm_call.py \
      --model openai/openai/gpt-5.2-codex \
      --prompt "Review docs/daystrom-cognition-network-implementation-plan.md"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = os.getenv(
    "OPENCLAW_INFERENCE_RESPONSES_URL",
    "https://inference-api.nvidia.com/v1/responses",
)

DEFAULT_MODELS = [
    "azure/anthropic/claude-opus-4-6",
    "openai/openai/gpt-5.2-codex",
    "aws/anthropic/bedrock-claude-sonnet-4-6",
    "openai/openai/gpt-5.1-codex",
]


def _extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def call_model(
    *,
    model: str,
    prompt: str,
    reasoning_effort: str,
    max_output_tokens: int,
    url: str = DEFAULT_URL,
) -> dict[str, Any]:
    key = os.getenv("OPENCLAW_INFERENCE_API_KEY")
    if not key:
        raise SystemExit("OPENCLAW_INFERENCE_API_KEY is not set")

    payload = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read().decode("utf-8", "replace")
            data = json.loads(raw)
            return {
                "ok": True,
                "status": response.status,
                "model": model,
                "text": _extract_text(data),
                "raw": data,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return {"ok": False, "status": exc.code, "model": model, "error": body}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Model to call. Repeat for swarm fanout.")
    parser.add_argument("--prompt", help="Prompt text. If omitted, stdin is used.")
    parser.add_argument("--prompt-file", help="Read prompt from a file instead of --prompt/stdin.")
    parser.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--json", action="store_true", help="Emit full JSON result(s).")
    args = parser.parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file, encoding="utf-8").read()
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()

    models = args.model or DEFAULT_MODELS
    results = [
        call_model(
            model=model,
            prompt=prompt,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
        for model in models
    ]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            print(f"=== {result['model']} ===")
            if result.get("ok"):
                print(result.get("text", "").strip())
            else:
                print(f"ERROR {result.get('status')}: {result.get('error', '')[:1000]}")
            print()
    return 0 if all(result.get("ok") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
