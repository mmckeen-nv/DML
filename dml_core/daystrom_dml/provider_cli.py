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
from .cognition.seed_proposer import DEFAULT_OLLAMA_BASE_URL, propose_seed_updates, run_seed_loop
from .cognition.seed_trial import run_seed_trial


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


def _json_object(raw: str | None, *, label: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload


def _read_json_file(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} must contain a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _write_json_file(path: str, payload: Any) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _dcn_payload(args: argparse.Namespace) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if getattr(args, "max_total_context_tokens", None) is not None:
        constraints["max_total_context_tokens"] = args.max_total_context_tokens
    if getattr(args, "max_memory_tokens", None) is not None:
        constraints["max_memory_tokens"] = args.max_memory_tokens
    if getattr(args, "max_personality_tokens", None) is not None:
        constraints["max_personality_tokens"] = args.max_personality_tokens
    if getattr(args, "no_tools", False):
        constraints["allow_tools"] = False
    if getattr(args, "allow_learning", False):
        constraints["allow_learning"] = True
    return {
        "content": args.text,
        "type": args.event_type,
        "metadata": _json_object(args.metadata, label="--metadata"),
        "scope": {
            "tenant_id": args.tenant_id,
            "client_id": args.client_id,
            "session_id": args.session_id,
            "instance_id": args.instance_id,
        },
        "constraints": constraints,
    }


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


def _dcn_eval_smoke_passed(payload: dict[str, Any]) -> bool:
    report = payload.get("report")
    artifact = payload.get("artifact")
    readiness = artifact.get("readiness") if isinstance(artifact, dict) else {}
    return bool(
        payload.get("status") == "ok"
        and isinstance(report, dict)
        and report.get("passed") is True
        and isinstance(readiness, dict)
        and readiness.get("ready") is True
    )


def cmd_dcn_eval_smoke(args: argparse.Namespace) -> int:
    """Run the provider-hosted offline DCN eval smoke readiness probe."""
    with _client(args) as client:
        response = client.get("/api/dcn/eval/smoke")
        response.raise_for_status()
        payload = response.json()
    if args.output:
        _write_json_file(args.output, payload.get("artifact") if args.artifact_only else payload)
    _print_json(payload)
    return 0 if isinstance(payload, dict) and _dcn_eval_smoke_passed(payload) else 1


def cmd_dcn_observe(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.post("/api/dcn/observe", json=_dcn_payload(args))
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_packet(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.post("/api/dcn/cognitive-packet", json=_dcn_payload(args))
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_feedback(args: argparse.Namespace) -> int:
    payload = {
        "decision_id": args.decision_id,
        "outcome": args.outcome,
        "signals": _json_object(args.signals, label="--signals"),
        "notes": args.notes or "",
    }
    with _client(args) as client:
        response = client.post("/api/dcn/feedback", json=payload)
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_audit_tail(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get("/api/dcn/audit", params={"limit": args.limit})
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_policy_show(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get("/api/dcn/policy")
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_policy_export(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.post("/api/dcn/policy/export")
        response.raise_for_status()
        payload = response.json()
    if args.output:
        _write_json_file(args.output, payload.get("snapshot") if args.snapshot_only else payload)
    _print_json(payload)
    return 0


def cmd_dcn_policy_import(args: argparse.Namespace) -> int:
    snapshot = _read_json_file(args.input)
    with _client(args) as client:
        response = client.post("/api/dcn/policy/import", json={"snapshot": snapshot})
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_policy_checkpoints(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get("/api/dcn/policy/checkpoints")
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_policy_checkpoint(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.post("/api/dcn/policy/checkpoint", json={"label": args.label})
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_policy_rollback(args: argparse.Namespace) -> int:
    payload = {"checkpoint_id": args.checkpoint_id} if args.checkpoint_id else {}
    with _client(args) as client:
        response = client.post("/api/dcn/policy/rollback", json=payload)
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_promote(args: argparse.Namespace) -> int:
    payload = {
        "target_mode": args.mode,
        "checkpoint_id": args.checkpoint_id,
        "hygiene_evidence": _json_object(args.hygiene_evidence, label="--hygiene-evidence"),
        "operator": args.operator,
        "reason": args.reason or "",
    }
    with _client(args) as client:
        response = client.post("/api/dcn/mode/promote", json=payload)
        response.raise_for_status()
        result = response.json()
        _print_json(result)
    return 0 if isinstance(result, dict) and result.get("promoted") is True else 1


def cmd_dcn_promotions(args: argparse.Namespace) -> int:
    with _client(args) as client:
        response = client.get("/api/dcn/mode/promotions", params={"limit": args.limit})
        response.raise_for_status()
        _print_json(response.json())
    return 0


def cmd_dcn_seed_trial(args: argparse.Namespace) -> int:
    payload = _read_json_file(args.input)
    artifact = run_seed_trial(payload)
    if args.output:
        _write_json_file(args.output, artifact)
    _print_json(artifact)
    return 0


def cmd_dcn_seed_propose(args: argparse.Namespace) -> int:
    payload = _read_json_file(args.input)
    proposal = propose_seed_updates(
        payload,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        timeout=args.timeout,
    )
    if args.output:
        _write_json_file(args.output, proposal)
    _print_json(proposal)
    return 0


def cmd_dcn_seed_loop(args: argparse.Namespace) -> int:
    payload = _read_json_file(args.input)
    artifact = run_seed_loop(
        payload,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        timeout=args.timeout,
    )
    if args.output:
        _write_json_file(args.output, artifact)
    if args.proposal_output:
        _write_json_file(args.proposal_output, artifact["proposal"])
    if args.trial_output:
        _write_json_file(args.trial_output, artifact["trial"])
    _print_json(artifact)
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
            "dcn_observe": "dml dcn observe --text 'continue the DML work'",
            "dcn_packet": "dml dcn packet --text 'continue the DML work' --session-id abc",
            "dcn_feedback": "dml dcn feedback --decision-id ... --outcome verified --signals '{\"tests_passed\":true}'",
            "dcn_policy_show": "dml dcn policy show",
            "dcn_policy_export": "dml dcn policy export --output dcn-policy.json --snapshot-only",
            "dcn_policy_import": "dml dcn policy import --input dcn-policy.json",
            "dcn_policy_checkpoint": "dml dcn policy checkpoint --label before-active-learn",
            "dcn_policy_checkpoints": "dml dcn policy checkpoints",
            "dcn_policy_rollback": "dml dcn policy rollback --checkpoint-id ...",
            "dcn_promote_active_learn": "dml dcn promote --mode active_learn --checkpoint-id ... --hygiene-evidence '{\"passed\":true}'",
            "dcn_promotions": "dml dcn promotions --limit 20",
            "dcn_audit_tail": "dml dcn audit-tail --limit 20",
            "dcn_eval_smoke": "dml dcn eval-smoke --output dcn-eval-artifact.json --artifact-only",
            "dcn_seed_trial": "dml dcn seed-trial --input sanitized-feedback.json --output dcn-seed-trial-artifact.json",
            "dcn_seed_propose": "dml dcn seed-propose --input sanitized-feedback.json --output dcn-seed-proposal.json",
            "dcn_seed_loop": "dml dcn seed-loop --input sanitized-feedback.json --output dcn-seed-loop-artifact.json",
            "frontier_prepare": "python skills/daystrom-dml/scripts/dml_frontier_prepare.py --prompt-file task.md --telemetry-only",
        },
        "mcp": {
            "command": "dml-mcp-server",
            "args": ["--transport", "stdio", "--storage", storage_dir or "$DML_STORE"],
        },
        "endpoints": {
            "dcn_policy": f"{base_url.rstrip('/')}/api/dcn/policy",
            "dcn_policy_export": f"{base_url.rstrip('/')}/api/dcn/policy/export",
            "dcn_policy_import": f"{base_url.rstrip('/')}/api/dcn/policy/import",
            "dcn_policy_checkpoints": f"{base_url.rstrip('/')}/api/dcn/policy/checkpoints",
            "dcn_policy_checkpoint": f"{base_url.rstrip('/')}/api/dcn/policy/checkpoint",
            "dcn_policy_rollback": f"{base_url.rstrip('/')}/api/dcn/policy/rollback",
            "dcn_mode_promote": f"{base_url.rstrip('/')}/api/dcn/mode/promote",
            "dcn_mode_promotions": f"{base_url.rstrip('/')}/api/dcn/mode/promotions",
            "dcn_audit": f"{base_url.rstrip('/')}/api/dcn/audit",
            "dcn_eval_smoke": f"{base_url.rstrip('/')}/api/dcn/eval/smoke",
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


def _add_dcn_request_args(parser: argparse.ArgumentParser) -> None:
    _add_provider_args(parser)
    _add_scope_args(parser)
    parser.add_argument("--text", required=True, help="User/event text to evaluate")
    parser.add_argument("--event-type", default="user_message")
    parser.add_argument("--metadata", help="JSON object with event metadata")
    parser.add_argument("--max-total-context-tokens", type=int)
    parser.add_argument("--max-memory-tokens", type=int)
    parser.add_argument("--max-personality-tokens", type=int)
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--allow-learning", action="store_true")


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

    dcn = sub.add_parser("dcn", help="Daystrom Cognition Network operator probes")
    dcn_sub = dcn.add_subparsers(dest="dcn_cmd", required=True)

    dcn_eval_smoke = dcn_sub.add_parser(
        "eval-smoke",
        aliases=["readiness"],
        help="Run the offline fixture-only DCN eval smoke readiness probe",
    )
    _add_provider_args(dcn_eval_smoke)
    dcn_eval_smoke.add_argument("--output", help="Write the eval response or artifact JSON to this file")
    dcn_eval_smoke.add_argument("--artifact-only", action="store_true", help="With --output, write only the sanitized artifact object")
    dcn_eval_smoke.set_defaults(func=cmd_dcn_eval_smoke)

    dcn_promote = dcn_sub.add_parser("promote", help="Promote DCN runtime mode behind checkpoint/eval/hygiene gates")
    _add_provider_args(dcn_promote)
    dcn_promote.add_argument("--mode", default="active_learn", choices=["active_learn"])
    dcn_promote.add_argument("--checkpoint-id", required=True)
    dcn_promote.add_argument("--hygiene-evidence", required=True, help="JSON object proving hygiene smoke passed, e.g. '{\"passed\":true}'")
    dcn_promote.add_argument("--operator", default="operator")
    dcn_promote.add_argument("--reason")
    dcn_promote.set_defaults(func=cmd_dcn_promote)

    dcn_promotions = dcn_sub.add_parser("promotions", help="List recent DCN mode promotion audit records")
    _add_provider_args(dcn_promotions)
    dcn_promotions.add_argument("--limit", type=int, default=20)
    dcn_promotions.set_defaults(func=cmd_dcn_promotions)

    dcn_seed_trial = dcn_sub.add_parser(
        "seed-trial",
        help="Run an offline non-promoting seed-model learning candidate trial from sanitized feedback JSON",
    )
    dcn_seed_trial.add_argument("--input", required=True, help="Sanitized seed-trial feedback/proposal JSON")
    dcn_seed_trial.add_argument("--output", help="Write the sanitized seed-trial artifact JSON")
    dcn_seed_trial.set_defaults(func=cmd_dcn_seed_trial)

    dcn_seed_propose = dcn_sub.add_parser(
        "seed-propose",
        help="Ask the local seed model for sanitized DCN procedural candidates without promotion",
    )
    dcn_seed_propose.add_argument("--input", required=True, help="Sanitized feedback batch JSON")
    dcn_seed_propose.add_argument("--output", help="Write the sanitized model proposal JSON")
    dcn_seed_propose.add_argument("--model", default="llama3:8b")
    dcn_seed_propose.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    dcn_seed_propose.add_argument("--timeout", type=float, default=60.0)
    dcn_seed_propose.set_defaults(func=cmd_dcn_seed_propose)

    dcn_seed_loop = dcn_sub.add_parser(
        "seed-loop",
        help="Run seed-propose then seed-trial validation without import or promotion",
    )
    dcn_seed_loop.add_argument("--input", required=True, help="Sanitized feedback batch JSON")
    dcn_seed_loop.add_argument("--output", help="Write combined sanitized loop artifact JSON")
    dcn_seed_loop.add_argument("--proposal-output", help="Write proposal JSON separately")
    dcn_seed_loop.add_argument("--trial-output", help="Write seed-trial artifact JSON separately")
    dcn_seed_loop.add_argument("--model", default="llama3:8b")
    dcn_seed_loop.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    dcn_seed_loop.add_argument("--timeout", type=float, default=60.0)
    dcn_seed_loop.set_defaults(func=cmd_dcn_seed_loop)

    dcn_observe = dcn_sub.add_parser("observe", help="Inspect the deterministic DCN plan for text")
    _add_dcn_request_args(dcn_observe)
    dcn_observe.set_defaults(func=cmd_dcn_observe)

    dcn_packet = dcn_sub.add_parser("packet", help="Build a DCN cognitive packet for text")
    _add_dcn_request_args(dcn_packet)
    dcn_packet.set_defaults(func=cmd_dcn_packet)

    dcn_feedback = dcn_sub.add_parser("feedback", help="Record DCN outcome feedback")
    _add_provider_args(dcn_feedback)
    dcn_feedback.add_argument("--decision-id", required=True)
    dcn_feedback.add_argument("--outcome", required=True)
    dcn_feedback.add_argument("--signals", help="JSON object with feedback signals")
    dcn_feedback.add_argument("--notes")
    dcn_feedback.set_defaults(func=cmd_dcn_feedback)

    dcn_audit_tail = dcn_sub.add_parser("audit-tail", help="Read recent DCN audit/feedback entries")
    _add_provider_args(dcn_audit_tail)
    dcn_audit_tail.add_argument("--limit", type=int, default=20)
    dcn_audit_tail.set_defaults(func=cmd_dcn_audit_tail)

    dcn_policy = dcn_sub.add_parser("policy", help="Inspect or move explicit DCN procedural policy overlays")
    dcn_policy_sub = dcn_policy.add_subparsers(dest="policy_cmd", required=True)

    dcn_policy_show = dcn_policy_sub.add_parser("show", help="Show active DCN policy metadata")
    _add_provider_args(dcn_policy_show)
    dcn_policy_show.set_defaults(func=cmd_dcn_policy_show)

    dcn_policy_export = dcn_policy_sub.add_parser("export", help="Export explicit procedural-learning overlay snapshot")
    _add_provider_args(dcn_policy_export)
    dcn_policy_export.add_argument("--output")
    dcn_policy_export.add_argument("--snapshot-only", action="store_true", help="Write only the snapshot object when --output is set")
    dcn_policy_export.set_defaults(func=cmd_dcn_policy_export)

    dcn_policy_import = dcn_policy_sub.add_parser("import", help="Import explicit procedural-learning overlay snapshot")
    _add_provider_args(dcn_policy_import)
    dcn_policy_import.add_argument("--input", required=True)
    dcn_policy_import.set_defaults(func=cmd_dcn_policy_import)

    dcn_policy_checkpoints = dcn_policy_sub.add_parser("checkpoints", help="List redacted procedural policy checkpoints")
    _add_provider_args(dcn_policy_checkpoints)
    dcn_policy_checkpoints.set_defaults(func=cmd_dcn_policy_checkpoints)

    dcn_policy_checkpoint = dcn_policy_sub.add_parser("checkpoint", help="Create a procedural policy rollback checkpoint")
    _add_provider_args(dcn_policy_checkpoint)
    dcn_policy_checkpoint.add_argument("--label", default="operator")
    dcn_policy_checkpoint.set_defaults(func=cmd_dcn_policy_checkpoint)

    dcn_policy_rollback = dcn_policy_sub.add_parser("rollback", help="Rollback procedural policy overlay to a checkpoint or baseline")
    _add_provider_args(dcn_policy_rollback)
    dcn_policy_rollback.add_argument("--checkpoint-id")
    dcn_policy_rollback.set_defaults(func=cmd_dcn_policy_rollback)

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
