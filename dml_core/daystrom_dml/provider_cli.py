"""Ollama-style client CLI for the Daystrom DML provider."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from . import provider_server


DEFAULT_BASE_URL = os.environ.get("DML_PROVIDER_URL", "http://127.0.0.1:8765")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _client(args: argparse.Namespace) -> httpx.Client:
    return httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout_s)


def _meta_from_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--meta must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--meta must be a JSON object")
    return payload


def cmd_serve(args: argparse.Namespace) -> int:
    provider_server.main(
        [
            "--host",
            args.host,
            "--port",
            str(args.port),
            *([] if args.config_path is None else ["--config-path", args.config_path]),
            *([] if args.storage_dir is None else ["--storage-dir", args.storage_dir]),
        ]
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get("/health")
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    payload = {
        "text": args.text,
        "tenant_id": args.tenant_id,
        "client_id": args.client_id,
        "session_id": args.session_id,
        "instance_id": args.instance_id,
        "kind": args.kind,
        "meta": _meta_from_args(args.meta),
    }
    with _client(args) as client:
        response = client.post("/api/remember", json=payload)
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    payload = {
        "query": args.query,
        "tenant_id": args.tenant_id,
        "client_id": args.client_id,
        "session_id": args.session_id,
        "instance_id": args.instance_id,
        "top_k": args.top_k,
    }
    with _client(args) as client:
        response = client.post("/api/recall", json=payload)
        response.raise_for_status()
        result = response.json()
    if args.context_only:
        print(result.get("raw_context") or "")
    else:
        _print_json(result)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    payload = {
        "query": args.query,
        "tenant_id": args.tenant_id,
        "client_id": args.client_id,
        "session_id": args.session_id,
        "instance_id": args.instance_id,
        "top_k": args.top_k,
    }
    with _client(args) as client:
        response = client.post("/api/resume", json=payload)
        response.raise_for_status()
        result = response.json()
    if args.context_only:
        print(result.get("raw_context") or "")
    else:
        _print_json(result)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    params = {"q": args.query, "tenant_id": args.tenant_id, "top_k": args.top_k}
    if args.session_id:
        params["session_id"] = args.session_id
    with _client(args) as client:
        response = client.get("/api/search", params=params)
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get(f"/api/fetch/{args.memory_id}")
        response.raise_for_status()
        _print_json(response.json())
    return 0


def _app_profile(app: str, *, base_url: str, tenant_id: str, storage_dir: str | None) -> dict[str, Any]:
    profile = {
        "app": app,
        "provider": "daystrom-dml",
        "base_url": base_url.rstrip("/"),
        "tenant_id": tenant_id,
        "storage_dir": storage_dir,
        "commands": {
            "serve": f"dml serve --storage-dir {storage_dir or '$DML_STORE'}",
            "remember": "dml remember --text '...' --meta '{\"source\":\"agent\"}'",
            "recall": "dml recall --query 'current task' --context-only",
            "resume": "dml resume --context-only",
            "frontier_prepare": "python skills/daystrom-dml/scripts/dml_frontier_prepare.py --prompt-file task.md --telemetry-only",
        },
        "mcp": {
            "command": "dml-mcp-server",
            "args": ["--transport", "stdio", "--storage", storage_dir or "$DML_STORE"],
        },
        "endpoints": {
            "frontier_prepare": f"{base_url.rstrip('/')}/api/frontier/prepare",
        },
    }
    if app == "openclaw":
        profile["environment"] = {
            "DML_PROVIDER_URL": base_url.rstrip("/"),
            "DML_TENANT_ID": tenant_id,
            "DML_STORE": storage_dir or "$OPENCLAW_HOME/dml-store",
        }
        profile["wrapper_hint"] = "Use skills/daystrom-dml/scripts/dml_memory.py for local file-locking commands."
    elif app == "hermes":
        profile["environment"] = {
            "DML_PROVIDER_URL": base_url.rstrip("/"),
            "DML_TENANT_ID": tenant_id,
            "HERMES_MEMORY_PROVIDER": "daystrom-dml",
        }
        profile["usage_hint"] = "Call /api/recall before a turn and /api/remember after durable state changes."
    else:
        profile["environment"] = {"DML_PROVIDER_URL": base_url.rstrip("/"), "DML_TENANT_ID": tenant_id}
    return profile


def cmd_install_app(args: argparse.Namespace) -> int:
    profile = _app_profile(args.app, base_url=args.base_url, tenant_id=args.tenant_id, storage_dir=args.storage_dir)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        profile["written_to"] = str(output)
    _print_json(profile)
    return 0


def _add_provider_args(parser: argparse.ArgumentParser, *, defaults: bool = False) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL if defaults else argparse.SUPPRESS)
    parser.add_argument("--timeout-s", type=float, default=30.0 if defaults else argparse.SUPPRESS)


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tenant-id", default=os.environ.get("DML_TENANT_ID", "openclaw"))
    parser.add_argument("--client-id")
    parser.add_argument("--session-id", default=os.environ.get("DML_SESSION_ID"))
    parser.add_argument("--instance-id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dml", description="Daystrom DML provider client")
    _add_provider_args(parser, defaults=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the local DML provider daemon")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--storage-dir")
    serve.add_argument("--config-path")
    serve.set_defaults(func=cmd_serve)

    status = sub.add_parser("status", help="Check provider health")
    _add_provider_args(status)
    status.set_defaults(func=cmd_status)

    remember = sub.add_parser("remember", help="Store a memory through the provider")
    _add_provider_args(remember)
    _add_scope_args(remember)
    remember.add_argument("--text", required=True)
    remember.add_argument("--kind", default="note")
    remember.add_argument("--meta")
    remember.set_defaults(func=cmd_remember)

    recall = sub.add_parser("recall", help="Recall memory context")
    _add_provider_args(recall)
    _add_scope_args(recall)
    recall.add_argument("--query", required=True)
    recall.add_argument("--top-k", type=int, default=6)
    recall.add_argument("--context-only", action="store_true")
    recall.set_defaults(func=cmd_recall)

    resume = sub.add_parser("resume", help="Recall continuity context")
    _add_provider_args(resume)
    _add_scope_args(resume)
    resume.add_argument("--query", default="active continuity checkpoint compaction handoff resume next action")
    resume.add_argument("--top-k", type=int, default=12)
    resume.add_argument("--context-only", action="store_true")
    resume.set_defaults(func=cmd_resume)

    search = sub.add_parser("search", help="Search memory and return handles")
    _add_provider_args(search)
    search.add_argument("--query", required=True)
    search.add_argument("--tenant-id", default=os.environ.get("DML_TENANT_ID", "openclaw"))
    search.add_argument("--session-id", default=os.environ.get("DML_SESSION_ID"))
    search.add_argument("--top-k", type=int, default=6)
    search.set_defaults(func=cmd_search)

    fetch = sub.add_parser("fetch", help="Fetch one memory by id")
    _add_provider_args(fetch)
    fetch.add_argument("memory_id")
    fetch.set_defaults(func=cmd_fetch)

    install_app = sub.add_parser("install-app", help="Emit an agent app install profile")
    _add_provider_args(install_app)
    install_app.add_argument("--app", choices=["openclaw", "hermes", "generic"], default="generic")
    install_app.add_argument("--tenant-id", default=os.environ.get("DML_TENANT_ID", "openclaw"))
    install_app.add_argument("--storage-dir")
    install_app.add_argument("--output")
    install_app.set_defaults(func=cmd_install_app)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except httpx.HTTPError as exc:
        print(f"dml: provider request failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
