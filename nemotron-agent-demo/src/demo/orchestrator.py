from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.memory import dml_http_client
from src.playground import cluster_manager
from src.playground import manager as playground_manager

from .agents import AgentResult, call_agent
from .metrics import StageMetrics, compute_throughput, estimate_tokens

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
HUMAN_INPUT_PATH = BASE_DIR / "prompt_library" / "human_input.json"
RUN_STATE_NAME = "run_state.json"
AGENT_LOGS_DIRNAME = "agent_logs"
PROJECT_CODE_DIRNAME = "project_code"
_agent_logs_migrated = False
DEFAULT_FALLBACK_PORT = 6969
RESERVED_PORTS = {8000}


@dataclass
class StageState:
    name: str
    status: str = "queued"
    ms: float = 0.0
    ttft_ms: float = 0.0
    tok_s: float = 0.0
    tokens: int = 0
    output: str = ""
    error: Optional[str] = None


DEFAULT_STAGES = ["supervisor", "planner", "coder", "reviewer", "ops", "aggregator"]
LONG_RUN_MARKER = "LONG_AGENT_RUN_MODE: true"
GIVE_UP_PHRASE = "we cannot complete the task"
HANDOFF_RE = re.compile(r"^\s*(?:NEXT_ROLE|HANDOFF_TO)\s*:\s*(\w+)\s*$", re.IGNORECASE | re.MULTILINE)
HUMAN_INPUT_REQUIRED_RE = re.compile(
    r"^\s*HUMAN_INPUT_REQUIRED\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
PORT_RE = re.compile(r"(?:port\\s+|:)(\\d{2,5})", re.IGNORECASE)
ALLOWED_TOOLS = [
    "playground.exec",
    "playground.exec_detached",
    "playground.write_file",
    "playground.docker",
    "playground.expose_port",
    "bare.exec",
    "bare.write_file",
    "cluster.exec",
    "cluster.logs",
    "cluster.validate",
]
TOOL_REQUEST_SCHEMA_TEXT = (
    "TOOL_REQUEST_SCHEMA v1:\n"
    "Each tool request must be valid JSON. Allowed shapes:\n"
    "1) Single object: {\"tool\": \"playground.exec\", \"cmd\": [\"bash\",\"-lc\",\"ls\"], \"timeout_s\": 60}\n"
    "2) Array of objects: [{...}, {...}]\n"
    "3) Wrapper: {\"tool_calls\": [{...}, {...}]}\n"
    "Formatting rules:\n"
    "- Output ONLY JSON for tool requests (no extra prose inside the JSON block).\n"
    "- Do NOT include raw newlines inside JSON strings.\n"
    "- Do NOT use shell line continuations (\"\\\") inside JSON strings.\n"
    "- For multi-step shell commands, join with '&&' or ';' inside a single string.\n"
    "Rules:\n"
    "- tool: string, one of: playground.exec, playground.exec_detached, playground.write_file, playground.docker, "
    "playground.expose_port, bare.exec, bare.write_file, cluster.exec, cluster.logs, cluster.validate\n"
    "- cmd: list[str] for *.exec\n"
    "- args: list[str] for playground.docker\n"
    "- path/content: strings for playground.write_file\n"
    "- host_port/container_port: integers for playground.expose_port\n"
    "- container: string for cluster.exec/cluster.logs\n"
    "- tail: integer for cluster.logs\n"
    "Examples (valid JSON):\n"
    "{\"tool\":\"bare.exec\",\"cmd\":[\"sshpass\",\"-p\",\"nvidia\",\"ssh\",\"-o\",\"StrictHostKeyChecking=no\",\"nvidia@192.168.50.81\",\"uname -a\"],\"timeout_s\":30}\n"
    "{\"tool\":\"bare.exec\",\"cmd\":[\"bash\",\"-lc\",\"cd /opt/ai-stack/openwebui-ollama && docker compose up -d\"],\"timeout_s\":600}\n"
    "{\"tool_calls\":[{\"tool\":\"playground.exec\",\"cmd\":[\"bash\",\"-lc\",\"ls -la /workspace\"],\"timeout_s\":60}]}\n"
    "You may return raw JSON or a fenced ```json``` block, but the JSON must be valid."
)
TOOL_REQUEST_SCHEMA_JSON = {
    "schema_version": "1",
    "allowed_tools": ALLOWED_TOOLS,
    "tool_request": {
        "tool": "string (required)",
        "cmd": "list[str] for *.exec",
        "args": "list[str] for playground.docker",
        "path": "string for playground.write_file",
        "content": "string for playground.write_file",
        "timeout_s": "int (optional)",
        "host_port": "int for playground.expose_port",
        "container_port": "int for playground.expose_port",
        "container": "string for cluster.exec/cluster.logs",
        "tail": "int for cluster.logs",
    },
    "examples": [
        {"tool": "playground.exec", "cmd": ["bash", "-lc", "ls -la"], "timeout_s": 60},
        {
            "tool": "bare.exec",
            "cmd": [
                "sshpass",
                "-p",
                "nvidia",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "nvidia@192.168.50.81",
                "uname -a",
            ],
            "timeout_s": 30,
        },
        {"tool": "playground.write_file", "path": "/workspace/agent_projects/<run_id>/app.py", "content": "print('hello')"},
    ],
}
FIXED_PLAYGROUND_COMMANDS = {
    "coder": ["bash", "-lc", "ls -la /workspace && python --version"],
    "aggregator": ["bash", "-lc", "find /workspace -maxdepth 3 -type f | head -n 50"],
}
BARE_ALLOWLIST = {
    "bash",
    "sh",
    "sudo",
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "uvicorn",
    "node",
    "npm",
    "make",
    "docker",
    "docker-compose",
    "sshpass",
    "ssh",
    "scp",
    "rsync",
    "ls",
    "pwd",
    "whoami",
    "env",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "cat",
    "curl",
    "jq",
    "sed",
    "grep",
    "find",
}
BARE_DENY_TOKENS = {"apt", "apt-get", "dnf", "yum", "pacman", "apk", "brew"}
BARE_READONLY_ALLOW_ABS = {
    "ls",
    "cat",
    "head",
    "tail",
    "grep",
    "sed",
    "find",
    "stat",
    "pwd",
    "whoami",
    "env",
    "uname",
    "nvidia-smi",
    "df",
    "du",
    "id",
    "getent",
    "hostname",
    "uptime",
    "ss",
    "netstat",
}


def _extract_password_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"password\s*[:=]\s*([^\s/]+)",
        r"password\s+([^\s/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip("\"'.,;")
            if value:
                return value
    return None


def _extract_ssh_target(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    user_match = re.search(r"\b([A-Za-z0-9._-]+)@(\d{1,3}(?:\.\d{1,3}){3})\b", text)
    if user_match:
        return user_match.group(1), user_match.group(2)
    host_match = re.search(r"target\s+host\s*[:=]\s*(\d{1,3}(?:\.\d{1,3}){3})", text, re.IGNORECASE)
    host = host_match.group(1) if host_match else None
    user_match = re.search(r"username\s*(?:[:=]\s*|\s+)([A-Za-z0-9._-]+)", text, re.IGNORECASE)
    if not user_match:
        user_match = re.search(r"user\s*(?:[:=]\s*|\s+)([A-Za-z0-9._-]+)", text, re.IGNORECASE)
    user = user_match.group(1) if user_match else None
    return user, host


def _ssh_key_available() -> bool:
    if os.environ.get("SSH_AUTH_SOCK"):
        return True
    for home in {Path.home(), Path("/root"), Path("/home/nvidia")}:
        ssh_dir = home / ".ssh"
        try:
            if not ssh_dir.exists():
                continue
        except PermissionError:
            continue
        for key_name in ("id_ed25519", "id_rsa", "id_ecdsa", "id_dsa"):
            try:
                if (ssh_dir / key_name).exists():
                    return True
            except PermissionError:
                continue
    return False


def _prompt_prefers_ssh_key(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "ssh key",
            "ssh-key",
            "key pair",
            "keypair",
            "private key",
            "public key",
            "key-based",
            "key based",
        )
    )


def _parse_max_tokens(raw: Optional[str], default: Optional[int]) -> Optional[int]:
    if raw is None:
        return default
    cleaned = str(raw).strip().lower()
    if cleaned in {"", "none", "null", "no", "false", "0"}:
        return None
    try:
        value = int(cleaned)
    except ValueError:
        return default
    return value if value > 0 else None


def _max_tokens_for_stage(stage_name: str, fast: bool) -> Optional[int]:
    role_key = str(stage_name or "").strip().upper()
    raw = os.getenv(f"ROLE_MAX_TOKENS_{role_key}") if role_key else None
    if raw is None:
        env_key = "AGENT_MAX_TOKENS_FAST" if fast else "AGENT_MAX_TOKENS"
        raw = os.getenv(env_key)
    return _parse_max_tokens(raw, 4096)


def _limit_tokens(max_tokens: Optional[int], limit: int) -> int:
    if max_tokens is None:
        return limit
    return min(limit, max_tokens)


def _load_human_messages() -> List[str]:
    try:
        raw = HUMAN_INPUT_PATH.read_text()
    except FileNotFoundError:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    cleaned: List[str] = []
    for item in payload:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _agent_logs_dir(run_id: str) -> Path:
    return BASE_DIR / "agent_projects" / run_id / AGENT_LOGS_DIRNAME


def _project_code_dir(run_id: str) -> Path:
    return BASE_DIR / "agent_projects" / run_id / PROJECT_CODE_DIRNAME


def _migrate_agent_logs(run_root: Path) -> None:
    outputs_dir = run_root / "outputs"
    agent_logs_dir = run_root / AGENT_LOGS_DIRNAME
    if outputs_dir.exists() and not agent_logs_dir.exists():
        try:
            outputs_dir.rename(agent_logs_dir)
        except OSError:
            return


def _migrate_all_agent_logs() -> None:
    global _agent_logs_migrated
    if _agent_logs_migrated:
        return
    root = BASE_DIR / "agent_projects"
    if not root.exists():
        _agent_logs_migrated = True
        return
    for run_dir in root.iterdir():
        if run_dir.is_dir():
            _migrate_agent_logs(run_dir)
    _agent_logs_migrated = True


def _ensure_agent_logs_dir(run_root: Path) -> Path:
    _migrate_agent_logs(run_root)
    agent_logs_dir = run_root / AGENT_LOGS_DIRNAME
    agent_logs_dir.mkdir(parents=True, exist_ok=True)
    return agent_logs_dir


def _ensure_project_code_dir(run_root: Path) -> Path:
    project_code_dir = run_root / PROJECT_CODE_DIRNAME
    project_code_dir.mkdir(parents=True, exist_ok=True)
    try:
        project_code_dir.chmod(0o777)
    except PermissionError:
        pass
    return project_code_dir


def _run_state_path(run_id: str) -> Path:
    return BASE_DIR / "agent_projects" / run_id / RUN_STATE_NAME


def _load_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    path = _run_state_path(run_id)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_run_state(run_id: str, payload: Dict[str, Any]) -> None:
    path = _run_state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _clear_run_state(run_id: str) -> None:
    path = _run_state_path(run_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _load_stage_artifacts(run_id: str) -> Dict[str, str]:
    outputs_dir = _agent_logs_dir(run_id)
    if not outputs_dir.exists():
        return {}
    artifacts: Dict[str, str] = {}
    for path in outputs_dir.glob("*.md"):
        try:
            content = path.read_text()
        except OSError:
            continue
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
            if lines and not lines[0].strip():
                lines = lines[1:]
        artifacts[path.stem.lower()] = "\n".join(lines).strip("\n")
    return artifacts


def _safe_stage_name(stage_name: str) -> str:
    safe_name = re.sub(r"[^a-z0-9_-]+", "-", stage_name.lower()).strip("-")
    return safe_name or "agent"


def _should_autobuild_webserver(goal: str) -> bool:
    goal_text = (goal or "").lower()
    if "hello world" in goal_text:
        return True
    if "hello" in goal_text and "world" in goal_text:
        return any(token in goal_text for token in ("web", "server", "http", "website", "site"))
    return False


def _extract_requested_port(goal: str, scenario: Optional[str]) -> Optional[int]:
    text = f"{goal} {scenario or ''}"
    patterns = [
        re.compile(r"0\\.0\\.0\\.0:(\\d{2,5})"),
        re.compile(r"port\\s+(\\d{2,5})", re.IGNORECASE),
        re.compile(r":(\\d{2,5})"),
    ]
    for pattern in patterns:
        for match in pattern.findall(text):
            try:
                port = int(match)
            except ValueError:
                continue
            if 1 <= port <= 65535:
                return port
    return None


def _resolve_service_port(goal: str, scenario: Optional[str]) -> int:
    requested = _extract_requested_port(goal, scenario)
    if requested in RESERVED_PORTS:
        return DEFAULT_FALLBACK_PORT
    return requested or DEFAULT_FALLBACK_PORT


def _requires_project_scaffold(goal: str, scenario: Optional[str]) -> bool:
    text = f"{goal} {scenario or ''}".lower()
    if LONG_RUN_MARKER.lower() in text:
        return True
    return bool(
        re.search(
            r"\\b(app|service|api|server|website|web|frontend|backend|docker|compose|container|microservice|project|ui)\\b",
            text,
        )
    )


def _is_code_request(goal: str, scenario: Optional[str]) -> bool:
    if _requires_project_scaffold(goal, scenario):
        return True
    return _should_autobuild_webserver(goal)


def _extract_human_input_requests(output: str) -> List[str]:
    if not output:
        return []
    return [match.strip() for match in HUMAN_INPUT_REQUIRED_RE.findall(output) if match.strip()]


def _format_access_summary(
    run_id: str,
    playground_info: Dict[str, Any],
    cluster_info: Dict[str, Any],
    bare_info: Optional[Dict[str, Any]] = None,
    code_requested: bool = False,
) -> str:
    lines: List[str] = []
    run_root = BASE_DIR / "agent_projects" / run_id
    lines.append("Access summary:")
    lines.append(f"- Run ID: {run_id}")
    lines.append(f"- Project path (host): {run_root}")
    lines.append(f"- Agent logs: {run_root / AGENT_LOGS_DIRNAME}")
    if code_requested or (run_root / PROJECT_CODE_DIRNAME).exists():
        lines.append(f"- Project code: {run_root / PROJECT_CODE_DIRNAME}")
    if bare_info and bare_info.get("enabled"):
        workspace = bare_info.get("workspace_host") or str(run_root / PROJECT_CODE_DIRNAME)
        lines.append(f"- Bare metal workspace: {workspace}")

    if playground_info.get("enabled"):
        lines.append(f"- Playground container: {playground_info.get('name') or '—'}")
        if playground_info.get("workspace_host") or playground_info.get("workspace_container"):
            lines.append(f"- Playground workspace (host): {playground_info.get('workspace_host') or '—'}")
            lines.append(f"- Playground workspace (container): {playground_info.get('workspace_container') or '—'}")
        exposed_ports = playground_info.get("exposed_ports") or []
        if exposed_ports:
            lines.append(f"- Playground exposed ports: {', '.join(str(p) for p in exposed_ports)}")
        web_port = playground_info.get("web_port")
        if web_port:
            lines.append(f"- Playground web URL: http://localhost:{web_port}")
        else:
            lines.append("- Playground web URL: not exposed (use playground.expose_port)")

    if cluster_info.get("enabled"):
        lines.append(f"- Cluster run ID: {cluster_info.get('run_id') or run_id}")
        lines.append(f"- Cluster network: {cluster_info.get('network') or '—'}")
        containers = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])]) or "—"
        lines.append(f"- Cluster containers: {containers}")
        api_port = cluster_info.get("api_port")
        web_port = cluster_info.get("web_port")
        lines.append(f"- Cluster API URL: {f'http://localhost:{api_port}' if api_port else '—'}")
        lines.append(f"- Cluster Web URL: {f'http://localhost:{web_port}' if web_port else '—'}")
        if cluster_info.get("workspace_host") or cluster_info.get("workspace_container"):
            lines.append(f"- Cluster workspace (host): {cluster_info.get('workspace_host') or '—'}")
            lines.append(f"- Cluster workspace (container): {cluster_info.get('workspace_container') or '—'}")

    return "\n".join(lines)


def _format_human_input_block(requests: List[str]) -> str:
    if not requests:
        return ""
    lines = ["HUMAN INPUT REQUIRED:"]
    lines.extend([f"- {item}" for item in requests if item])
    return "\n".join(lines)


def _should_retry_output(output: str) -> bool:
    if not output:
        return True
    text = output.strip().lower()
    return text.startswith("no content returned")


def _compact_system_messages(messages: List[str]) -> List[str]:
    compact: List[str] = []
    for msg in messages:
        if msg.startswith("TOOL_REQUEST_SCHEMA_JSON:"):
            continue
        if msg.startswith("DML_COOKBOOK_GUIDANCE:"):
            continue
        compact.append(msg)
    return compact


def _compact_extra_context(extra_context: str) -> str:
    if not extra_context:
        return ""
    marker = "Tool Command Log:"
    if marker in extra_context:
        extra_context = extra_context.split(marker, 1)[0]
    extra_context = extra_context.strip()
    if len(extra_context) > 2000:
        extra_context = extra_context[-2000:]
    return extra_context


def _write_stage_artifact(run_id: str, stage_name: str, content: str) -> Optional[str]:
    try:
        run_root = _ensure_agent_logs_dir(BASE_DIR / "agent_projects" / run_id)
        try:
            run_root.chmod(0o777)
        except PermissionError:
            pass
        safe_name = _safe_stage_name(stage_name)
        path = run_root / f"{safe_name}.md"
        path.write_text(content)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def _append_tool_repair_log(run_id: str, label: str, output: str, errors: List[str]) -> None:
    try:
        run_root = _ensure_agent_logs_dir(BASE_DIR / "agent_projects" / run_id)
        path = run_root / "tool_repairs.md"
        snippet = (output or "").strip()
        if len(snippet) > 4000:
            snippet = snippet[:4000] + "\n...truncated..."
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {label}\n")
            if errors:
                handle.write("Errors:\n")
                for err in errors:
                    handle.write(f"- {err}\n")
            handle.write("Output:\n```\n")
            handle.write(snippet)
            handle.write("\n```\n\n")
    except Exception:  # noqa: BLE001
        return


def _initial_state(goal: str, scenario: Optional[str], fast: bool) -> Dict:
    stage_order = ["supervisor", "planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stages = [StageState(name=s.title()) for s in stage_order]
    return {
        "goal": goal,
        "scenario": scenario,
        "stages": [stage.__dict__ for stage in stages],
        "metrics": {"total_ms": 0, "approx_tok_s": 0, "approx_ttft_ms": 0},
        "final": "",
    }


def _serialize(
    stages: List[StageState],
    goal: str,
    scenario: Optional[str],
    final: str,
    total_ms: float,
    playground: Optional[Dict] = None,
    bare: Optional[Dict] = None,
    cluster: Optional[Dict] = None,
    dml: Optional[Dict] = None,
    events: Optional[List[str]] = None,
) -> Dict:
    completed = [s for s in stages if s.status == "done"]
    total_tokens = sum(s.tokens for s in completed)
    total_ttft = sum(s.ttft_ms for s in completed)
    summed_tok_s = sum(s.tok_s for s in stages if s.status in {"done", "running"} and s.tok_s > 0)
    approx_tok_s = summed_tok_s if summed_tok_s > 0 else (compute_throughput(total_tokens, total_ms) if total_ms else 0.0)
    approx_ttft = total_ttft / len(completed) if completed else 0.0
    return {
        "goal": goal,
        "scenario": scenario,
        "stages": [stage.__dict__ for stage in stages],
        "metrics": {
            "total_ms": total_ms,
            "total_tokens": total_tokens,
            "approx_tok_s": approx_tok_s,
            "approx_ttft_ms": approx_ttft,
        },
        "final": final,
        "playground": playground or {},
        "bare": bare or {},
        "cluster": cluster or {},
        "dml": dml or {},
        "events": events or [],
    }


def _is_long_run(goal: str, scenario: Optional[str]) -> bool:
    if LONG_RUN_MARKER.lower() in goal.lower():
        return True
    if scenario and LONG_RUN_MARKER.lower() in scenario.lower():
        return True
    return False


def _extract_fenced_json_blocks(text: str) -> List[str]:
    if not text:
        return []
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in fences if block and block.strip()]


def _log_json_block_error(block: str, exc: json.JSONDecodeError, context: str) -> None:
    snippet = block.strip()
    if len(snippet) > 400:
        snippet = snippet[:400] + "\n...truncated..."
    logger.warning("%s: fenced JSON parse error (%s). Block snippet:\n%s", context, exc.msg, snippet)


def _scan_json_objects(blob: str) -> List[Any]:
    found: List[Any] = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(blob)
    while idx < length:
        ch = blob[idx]
        if ch not in "{[":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(blob[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        found.append(obj)
        idx += max(end, 1)
    return found


def _collect_tool_requests(obj: Any, sink: List[Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if obj.get("tool"):
            sink.append(obj)
            return
        tool_calls = obj.get("tool_calls")
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if isinstance(item, dict) and item.get("tool"):
                    sink.append(item)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_tool_requests(item, sink)


def _extract_tool_requests(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    requests: List[Dict[str, Any]] = []
    for block in _extract_fenced_json_blocks(text):
        if '"tool"' in block or "tool_calls" in block:
            try:
                json.loads(block)
            except json.JSONDecodeError as exc:
                _log_json_block_error(block, exc, "_extract_tool_requests")
        for obj in _scan_json_objects(block):
            _collect_tool_requests(obj, requests)
    # Fallback: scan the full text for inline JSON objects.
    for obj in _scan_json_objects(text):
        _collect_tool_requests(obj, requests)
    return requests


def _extract_tool_requests_with_errors(text: str) -> tuple[List[Dict[str, Any]], List[str], bool]:
    if not text:
        return [], [], False
    errors: List[str] = []
    tool_mentions = re.findall(r'"tool"\s*:', text)
    tool_hints = bool(tool_mentions) or "playground." in text or "cluster." in text
    for block in _extract_fenced_json_blocks(text):
        if '"tool"' in block or "tool_calls" in block:
            try:
                json.loads(block)
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON in fenced block: {exc.msg}")
                _log_json_block_error(block, exc, "_extract_tool_requests_with_errors")
    requests = _extract_tool_requests(text)
    if tool_mentions and len(requests) < len(tool_mentions):
        errors.append(
            f"Found {len(tool_mentions)} tool markers but parsed {len(requests)} tool objects."
        )
    if tool_hints and not requests:
        errors.append("Tool hints present but no valid tool JSON parsed.")
    return requests, errors, tool_hints


def _validate_tool_request(req: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(req, dict):
        return ["Tool request must be an object."]
    tool = req.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return ["Missing required string field 'tool'."]
    if tool in {"playground.exec", "playground.exec_detached", "cluster.exec", "bare.exec"}:
        cmd = req.get("cmd")
        if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
            errors.append("Field 'cmd' must be list[str].")
        timeout_s = req.get("timeout_s")
        if timeout_s is not None and not isinstance(timeout_s, int):
            errors.append("Field 'timeout_s' must be int when provided.")
    elif tool == "playground.docker":
        args = req.get("args") or req.get("cmd")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            errors.append("Field 'args' must be list[str].")
        timeout_s = req.get("timeout_s")
        if timeout_s is not None and not isinstance(timeout_s, int):
            errors.append("Field 'timeout_s' must be int when provided.")
    elif tool in {"playground.write_file", "bare.write_file"}:
        path = req.get("path")
        content = req.get("content")
        if not isinstance(path, str) or not path:
            errors.append("Field 'path' must be non-empty string.")
        if not isinstance(content, str):
            errors.append("Field 'content' must be string.")
    elif tool == "playground.expose_port":
        host_port = req.get("host_port")
        container_port = req.get("container_port") or req.get("target_port") or host_port
        if not isinstance(host_port, int):
            errors.append("Field 'host_port' must be int.")
        if not isinstance(container_port, int):
            errors.append("Field 'container_port' must be int.")
    elif tool == "cluster.logs":
        container = req.get("container")
        tail = req.get("tail")
        if not isinstance(container, str) or not container:
            errors.append("Field 'container' must be non-empty string.")
        if tail is not None and not isinstance(tail, int):
            errors.append("Field 'tail' must be int when provided.")
    elif tool == "cluster.validate":
        pass
    else:
        errors.append(f"Unsupported tool '{tool}'.")
    return errors


def _validate_tool_schema(payload: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["Schema must be a JSON object."]
    if payload.get("schema_version") != "1":
        errors.append("schema_version must be '1'.")
    allowed = payload.get("allowed_tools")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        errors.append("allowed_tools must be a list of strings.")
    else:
        expected = set(ALLOWED_TOOLS)
        got = set(allowed)
        missing = expected - got
        extra = got - expected
        if missing:
            errors.append(f"allowed_tools missing: {', '.join(sorted(missing))}.")
        if extra:
            errors.append(f"allowed_tools has unexpected entries: {', '.join(sorted(extra))}.")
    return errors


def _run_tool_preflight(
    stage_name: str,
    goal: str,
    scenario: Optional[str],
    extra_context: str,
    system_messages: List[str],
    max_tokens: Optional[int],
) -> tuple[bool, List[str], str]:
    def _parse_schema_json(text: str) -> tuple[Any | None, str | None]:
        if not text or not text.strip():
            return None, "Schema JSON parse error: empty output."
        objects = _scan_json_objects(text)
        if not objects:
            return None, "Schema JSON parse error: no JSON object found."
        for obj in objects:
            if isinstance(obj, dict) and "schema_version" in obj:
                return obj, None
        return None, "Schema JSON parse error: schema object not found."

    preflight_schema = {
        "schema_version": "1",
        "allowed_tools": list(ALLOWED_TOOLS),
    }
    schema_text = json.dumps(preflight_schema, indent=2)
    preflight_messages = [
        "TOOL_REQUEST_SCHEMA_JSON:\n" + schema_text,
        "TOOL_REQUEST_PREFLIGHT: Return ONLY the JSON schema object above. No Markdown, no extra text.",
        "The user goal is irrelevant for this step; only return the schema JSON.",
    ]
    preflight_result = call_agent(
        "schema_preflight",
        "Return the schema JSON only.",
        None,
        max_tokens=_limit_tokens(max_tokens, 256),
        extra_context="",
        system_messages=preflight_messages,
    )
    output = preflight_result.output or ""
    errors: List[str] = []
    parsed, parse_error = _parse_schema_json(output)
    if parse_error:
        errors.append(parse_error)
    if parsed is not None:
        errors.extend(_validate_tool_schema(parsed))
    if not errors:
        return True, [], output

    max_repairs = int(os.getenv("TOOL_PREFLIGHT_REPAIR_MAX", "3"))
    last_output = output
    last_errors = errors
    for attempt in range(1, max_repairs + 1):
        repair_messages = [
            "TOOL_REQUEST_SCHEMA_JSON:\n" + schema_text,
        ]
        repair_notice = (
            "TOOL_REQUEST_PREFLIGHT_REPAIR: Fix the schema output. Return ONLY valid JSON that matches "
            "TOOL_REQUEST_SCHEMA_JSON. No extra text."
        )
        if attempt == max_repairs:
            repair_notice += " FINAL ATTEMPT: If this fails, the run will terminate."
        repair_messages.append(repair_notice)
        repair_context = "\n\n".join(
            filter(
                None,
                [
                    "",
                    f"Attempt {attempt} of {max_repairs}.",
                    "Previous schema output (invalid):\n" + (last_output or "").strip(),
                    "Issues:\n- " + "\n- ".join(last_errors),
                ],
            )
        )
        repair_result = call_agent(
            "schema_preflight",
            "Return the schema JSON only.",
            None,
            max_tokens=_limit_tokens(max_tokens, 256),
            extra_context=repair_context,
            system_messages=repair_messages,
        )
        repair_output = repair_result.output or ""
        repair_errors: List[str] = []
        repaired, repair_parse_error = _parse_schema_json(repair_output)
        if repair_parse_error:
            repair_errors.append(repair_parse_error)
        if repaired is not None:
            repair_errors.extend(_validate_tool_schema(repaired))
        if not repair_errors:
            return True, [], repair_output
        last_output = repair_output
        last_errors = repair_errors
    return False, last_errors, last_output


def _format_tool_context(entry: Dict[str, Any]) -> str:
    return (
        "Playground command result:\n"
        f"$ {entry.get('cmd')}\n"
        f"Exit code: {entry.get('exit_code')}\n"
        f"Stdout:\n{entry.get('stdout')}\n"
        f"Stderr:\n{entry.get('stderr')}\n"
    )


def _format_cluster_context(entry: Dict[str, Any]) -> str:
    return (
        "Cluster command result:\n"
        f"$ {entry.get('cmd')}\n"
        f"Exit code: {entry.get('exit_code')}\n"
        f"Stdout:\n{entry.get('stdout')}\n"
        f"Stderr:\n{entry.get('stderr')}\n"
    )


def _format_validation_context(validation: Dict[str, Any], cluster_info: Dict[str, Any]) -> str:
    containers = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])]) or "none"
    status_lines = [
        f"run_id: {cluster_info.get('run_id')}",
        f"network: {cluster_info.get('network')}",
        f"containers: {containers}",
        f"api_port: {cluster_info.get('api_port')}",
        f"web_port: {cluster_info.get('web_port')}",
        f"workspace_host: {cluster_info.get('workspace_host')}",
        f"workspace_container: {cluster_info.get('workspace_container')}",
    ]
    return (
        "Cluster validation report:\n"
        f"{json.dumps(validation, indent=2)}\n\n"
        "Cluster status:\n"
        + "\n".join(status_lines)
    )


def _parse_ops_failure(output: str) -> Optional[Dict[str, str]]:
    sections: Dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in {"OPS_STATUS", "OPS_ERROR", "OPS_TO_PLANNER"}:
            sections[key] = value.strip()
    if sections.get("OPS_STATUS", "").upper() != "FAIL":
        return None
    return {
        "error": sections.get("OPS_ERROR", "").strip(),
        "instruction": sections.get("OPS_TO_PLANNER", "").strip(),
        "raw": output.strip(),
    }


def _agent_gave_up(output: str) -> bool:
    return GIVE_UP_PHRASE in output.lower()


def _extract_handoff(output: str) -> Optional[str]:
    match = HANDOFF_RE.search(output or "")
    if not match:
        return None
    role = match.group(1).strip().lower()
    if role in {"supervisor", "planner", "coder", "reviewer", "ops", "aggregator"}:
        return role
    return None


def _run_playground_command(
    playground_name: str,
    cmd: List[str],
    playground_log: List[Dict[str, Any]],
    tool_context_chunks: List[str],
    timeout_s: int = 60,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
    entry["cmd"] = " ".join(cmd)
    if agent:
        entry["agent"] = agent
    playground_log.append(entry)
    tool_context_chunks.append(_format_tool_context(entry))
    return entry


def _clamp_bare_output(text: str, limit: int = 200_000) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return (data[:limit] + b"\n...output truncated...").decode("utf-8", errors="replace")


def _extract_bare_permissions(messages: List[str]) -> set[str]:
    permissions: set[str] = set()
    for message in messages:
        for line in message.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("ALLOW_BARE_ALL"):
                permissions.add("*")
            elif line.upper().startswith("ALLOW_BARE:"):
                permissions.add(line.split(":", 1)[1].strip())
    return permissions


def _validate_bare_command(cmd: List[str], project_root: Path, permissions: set[str]) -> tuple[bool, str, bool]:
    if not cmd:
        return False, "Empty command list.", False
    command_name = cmd[0].split("/")[-1]
    if command_name not in BARE_ALLOWLIST:
        return False, f"Command '{command_name}' is not in allowlist.", True
    is_remote_ssh = command_name in {"ssh", "sshpass"}
    command_text = " ".join(cmd)
    lowered = command_text.lower()
    if "sudo" in lowered:
        if "*" in permissions or any("sudo" in perm.lower() for perm in permissions):
            pass
        else:
            return False, "sudo requires explicit permission.", True
    if not is_remote_ssh:
        for token in BARE_DENY_TOKENS:
            if token in lowered:
                if "*" in permissions or any(token in perm.lower() for perm in permissions):
                    break
                return False, f"Command contains forbidden token '{token}'.", True
    if command_name == "docker" and "*" not in permissions:
        safe_docker = {"--version", "version", "ps", "images", "info"}
        if len(cmd) >= 2 and cmd[1] in safe_docker:
            pass
        elif len(cmd) >= 3 and cmd[1] == "compose" and cmd[2] in {"version", "ls"}:
            pass
        else:
            return False, "docker commands require explicit permission unless checking status/version.", True
    if "pip" in command_name or " pip " in f" {lowered} ":
        if ".venv" not in lowered and "python -m venv" not in lowered:
            return False, "pip usage requires a .venv (python -m venv .venv, then .venv/bin/pip).", False
    if is_remote_ssh:
        return True, "", False
    for arg in cmd[1:]:
        if arg.startswith("/"):
            try:
                resolved = Path(arg).resolve()
            except OSError:
                return False, "Invalid absolute path.", False
            if resolved != project_root and project_root not in resolved.parents:
                if command_name in BARE_READONLY_ALLOW_ABS:
                    continue
                if "*" in permissions:
                    continue
                return False, "Absolute paths must remain under PROJECT_CODE_DIR.", True
    if "*" in permissions:
        return True, "", False
    for token in permissions:
        if token and token in command_text:
            return True, "", False
    return True, "", False


def _prepend_bare_safe_dir(cmd: List[str], project_root: Path) -> List[str]:
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc":
        return ["bash", "-lc", f"cd {project_root} && {cmd[2]}"]
    return cmd


def _normalize_exec_cmd(cmd: List[str]) -> List[str]:
    if not cmd:
        return cmd
    cleaned: List[str] = []
    changed = False
    for part in cmd:
        if isinstance(part, str) and len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", "\""}:
            cleaned.append(part[1:-1])
            changed = True
        else:
            if isinstance(part, str) and part.endswith(")") and len(part) > 1 and not part.startswith("("):
                cleaned.append(part[:-1])
                changed = True
            else:
                cleaned.append(part)
    if changed:
        cmd = cleaned
    if cmd and cmd[0] == "sshpass":
        cmd = _strip_batchmode_args(cmd)
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc" and isinstance(cmd[2], str):
        shell_cmd = cmd[2]
        shell_cmd = re.sub(
            r"(sshpass\s+-p\s+)([^\s\"')]+)\)",
            r"\\1\\2",
            shell_cmd,
        )
        shell_cmd = re.sub(
            r"(sshpass\s+-p\s+['\"])([^\"')]+)\\)",
            r"\\1\\2",
            shell_cmd,
        )
        shell_cmd = shell_cmd.replace(" -o BatchMode=yes", "")
        cmd = [cmd[0], cmd[1], shell_cmd]
    if cmd[0] in {"bash", "sh"}:
        return cmd
    if len(cmd) <= 1:
        return cmd
    parts_with_spaces = [part for part in cmd if " " in part.strip()]
    if not parts_with_spaces:
        return cmd
    if " " in cmd[0] or len(parts_with_spaces) > 1:
        shell_cmd = " && ".join(part.strip() for part in cmd if part.strip())
        return ["bash", "-lc", shell_cmd]
    return cmd


def _strip_batchmode_args(cmd: List[str]) -> List[str]:
    if not cmd:
        return cmd
    sanitized: List[str] = []
    skip_next = False
    i = 0
    while i < len(cmd):
        if skip_next:
            skip_next = False
            i += 1
            continue
        if cmd[i] == "-o" and i + 1 < len(cmd) and str(cmd[i + 1]).startswith("BatchMode="):
            skip_next = True
            i += 1
            continue
        if isinstance(cmd[i], str) and cmd[i].startswith("BatchMode="):
            i += 1
            continue
        sanitized.append(cmd[i])
        i += 1
    return sanitized


def _ssh_auth_failed(entry: Dict[str, Any]) -> bool:
    stderr = (entry.get("stderr") or "")
    stdout = (entry.get("stdout") or "")
    combined = f"{stderr}\n{stdout}"
    indicators = (
        "permission denied",
        "publickey",
        "authentication failed",
        "no supported authentication",
    )
    return any(token in combined.lower() for token in indicators)


def _resolve_bare_executable(cmd: List[str]) -> List[str]:
    if not cmd:
        return cmd
    executable = cmd[0]
    if "/" in executable:
        return cmd
    resolved = shutil.which(executable)
    if not resolved:
        for candidate in (f"/usr/local/bin/{executable}", f"/usr/bin/{executable}", f"/bin/{executable}"):
            if Path(candidate).exists():
                resolved = candidate
                break
    if resolved:
        updated = list(cmd)
        updated[0] = resolved
        return updated
    return cmd


def _bare_exec_env() -> Dict[str, str]:
    env = os.environ.copy()
    path_entries = [p for p in env.get("PATH", "").split(":") if p]
    for candidate in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        if candidate not in path_entries:
            path_entries.append(candidate)
    env["PATH"] = ":".join(path_entries)
    return env


def _run_bare_exec(cmd: List[str], project_root: Path, timeout_s: int) -> Dict[str, Any]:
    resolved_cmd = _resolve_bare_executable(cmd)
    try:
        result = subprocess.run(
            resolved_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_bare_exec_env(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": _clamp_bare_output(exc.stdout or ""),
            "stderr": _clamp_bare_output(exc.stderr or "Command timed out."),
        }
    except OSError as exc:
        if cmd:
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": f"{exc} (executable '{cmd[0]}' not found on PATH)",
            }
        return {"exit_code": 127, "stdout": "", "stderr": str(exc)}
    return {
        "exit_code": result.returncode,
        "stdout": _clamp_bare_output(result.stdout or ""),
        "stderr": _clamp_bare_output(result.stderr or ""),
    }


def _run_bare_write(path: Path, content: str) -> Dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"exit_code": 0, "stdout": f"Wrote {path}", "stderr": ""}
    except OSError as exc:
        return {"exit_code": 125, "stdout": "", "stderr": str(exc)}


def _update_stage(stages: List[StageState], name: str, **kwargs) -> None:
    for stage in stages:
        if stage.name.lower() == name.lower():
            for key, value in kwargs.items():
                setattr(stage, key, value)
            break


def run_demo_stream(
    goal: str,
    fast: bool = False,
    scenario: Optional[str] = None,
    use_dml: bool = False,
    dml_top_k: int = 6,
    use_playground: bool = False,
    playground_image: str = "nemotron-playground:latest",
    auto_remove_playground: bool = False,
    use_bare_metal: bool = False,
    use_cluster: bool = False,
    cluster_image: str = "nemotron-playground:latest",
    cluster_size: int = 3,
    cluster_run_id: Optional[str] = None,
    parallel_agents: Optional[bool] = None,
) -> Generator[Dict, None, None]:
    stages: List[StageState] = []
    stage_order = ["supervisor", "planner", "coder", "reviewer"]
    if not fast:
        stage_order.append("ops")
    stage_order.append("aggregator")
    stage_queue = list(stage_order)
    stages = [StageState(name=s.title()) for s in stage_order]

    _migrate_all_agent_logs()
    resume_state: Optional[Dict[str, Any]] = None
    resume_run_id = None
    if isinstance(cluster_run_id, str) and cluster_run_id.strip() and "/" not in cluster_run_id and " " not in cluster_run_id:
        resume_run_id = cluster_run_id.strip()
        resume_state = _load_run_state(resume_run_id)
    resuming = bool(resume_state and resume_state.get("awaiting_human_input"))
    if resuming and resume_run_id:
        run_id = resume_run_id
    elif use_cluster and resume_run_id:
        run_id = resume_run_id
    else:
        run_id = str(uuid.uuid4())
    run_root = BASE_DIR / "agent_projects" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root.chmod(0o777)
    except PermissionError:
        pass
    _ensure_agent_logs_dir(run_root)
    readme_path = run_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text("Initialized by Nemotron Station run.\n")
    if use_bare_metal:
        use_playground = False
        use_cluster = False
    code_requested = _is_code_request(goal, scenario) or (run_root / PROJECT_CODE_DIRNAME).exists()
    project_root = _ensure_project_code_dir(run_root) if code_requested else run_root
    project_root_container = (
        f"/workspace/agent_projects/{run_id}/{PROJECT_CODE_DIRNAME}" if code_requested else f"/workspace/agent_projects/{run_id}"
    )
    ssh_password = _extract_password_from_text(goal) or _extract_password_from_text(scenario or "")
    ssh_user = None
    ssh_host = None
    ssh_key_preferred = _prompt_prefers_ssh_key(goal) or _prompt_prefers_ssh_key(scenario or "")
    for candidate in (goal, scenario or ""):
        if not candidate:
            continue
        found_user, found_host = _extract_ssh_target(candidate)
        if found_host and not ssh_host:
            ssh_host = found_host
        if found_user and not ssh_user:
            ssh_user = found_user
    if ssh_host and not ssh_user:
        ssh_user = "nvidia"
    if not ssh_password:
        for message in _load_human_messages():
            ssh_password = _extract_password_from_text(message)
            if ssh_password:
                break
    if not ssh_key_preferred:
        for message in _load_human_messages():
            if _prompt_prefers_ssh_key(message):
                ssh_key_preferred = True
                break
    ssh_key_available = _ssh_key_available()
    # If a password is provided in the prompt, prefer it unless the user explicitly requests key-based auth.
    ssh_use_password = bool(ssh_password and not ssh_key_preferred)
    if resuming and resume_state:
        requesting_stage = str(resume_state.get("requesting_stage") or "").strip().lower()
        saved_queue = resume_state.get("stage_queue") or []
        if isinstance(saved_queue, list):
            allowed = set(stage_order)
            filtered_queue = [str(item).strip().lower() for item in saved_queue]
            filtered_queue = [item for item in filtered_queue if item in allowed]
        else:
            filtered_queue = list(stage_queue)
        if requesting_stage and requesting_stage in stage_order:
            stage_queue = [requesting_stage] + [item for item in filtered_queue if item != requesting_stage]
        else:
            stage_queue = filtered_queue or list(stage_queue)
        _clear_run_state(run_id)
    long_run_mode = _is_long_run(goal, scenario)
    attempt = 1
    max_attempts = int(os.getenv("AGENT_MAX_ATTEMPTS", "0"))
    playground_name = f"nemotron-playground-{run_id.split('-')[0]}"
    playground_log: List[Dict[str, Any]] = []
    playground_info: Dict[str, Any] = {
        "enabled": bool(use_playground),
        "name": playground_name if use_playground else "",
        "image": playground_image,
        "requested_image": playground_image,
        "status": "disabled" if not use_playground else "pending",
        "error": None,
        "warning": None,
        "log": playground_log,
        "auto_remove": bool(auto_remove_playground),
        "ready_for_removal": False,
        "workspace_host": "",
        "workspace_container": "",
    }
    bare_log: List[Dict[str, Any]] = []
    bare_info: Dict[str, Any] = {
        "enabled": bool(use_bare_metal),
        "status": "enabled" if use_bare_metal else "disabled",
        "error": None,
        "log": bare_log,
        "workspace_host": str(project_root) if use_bare_metal else "",
    }
    cluster_log: List[Dict[str, Any]] = []
    tool_context_chunks: List[str] = []
    events: List[str] = []
    events.append(f"Run ID: {run_id}")
    if resuming:
        events.append("Resuming run after human input.")
    if max_attempts == 0:
        events.append(f"Attempt {attempt} (unlimited retries)")
    else:
        events.append(f"Attempt {attempt} of {max_attempts}")
    if not use_playground and not use_cluster and not use_bare_metal:
        events.append("Playground/Cluster tools are disabled; agents can only return text output.")
    if use_bare_metal:
        events.append("Bare metal tools enabled; commands execute on host within project_code.")
    else:
        events.append("Tool commands run via docker exec; container logs may not show them.")
        if use_playground and playground_info.get("web_port"):
            events.append(f"Playground web URL: http://localhost:{playground_info.get('web_port')}")
        elif use_playground:
            events.append("No host port exposed. Use playground.expose_port to publish a port.")
    cluster_info: Dict[str, Any] = {
        "enabled": bool(use_cluster),
        "run_id": run_id,
        "size": cluster_size,
        "image": cluster_image,
        "status": "disabled" if not use_cluster else "pending",
        "network": "",
        "containers": [],
        "api_port": None,
        "web_port": None,
        "workspace_host": "",
        "workspace_container": "",
        "error": None,
        "validation": {},
        "validation_history": [],
        "iteration": 0,
        "max_iters": None,
        "fix_actions": [],
        "log": cluster_log,
        "ready_for_removal": False,
    }
    if use_playground:
        playground_status = playground_manager.ensure_playground(playground_image, playground_name, run_id, repo_mount=None)
        playground_info.update(
            {
                "status": playground_status.get("status", "unknown"),
                "error": playground_status.get("error"),
                "workspace_host": playground_status.get("workspace_host"),
                "workspace_container": playground_status.get("workspace_container"),
                "warning": playground_status.get("warning"),
                "image": playground_status.get("image", playground_image),
                "requested_image": playground_status.get("requested_image", playground_image),
                "web_port": playground_status.get("web_port"),
            }
        )
    if use_cluster:
        cluster_status = cluster_manager.create_cluster(run_id, cluster_image, cluster_size, workspace_host=None)
        cluster_info.update(
            {
                "status": cluster_status.get("status", "unknown"),
                "error": cluster_status.get("error"),
                "network": cluster_status.get("network", ""),
                "containers": cluster_status.get("containers", []),
                "api_port": cluster_status.get("api_port"),
                "web_port": cluster_status.get("web_port"),
                "workspace_host": cluster_status.get("workspace_host", ""),
                "workspace_container": cluster_status.get("workspace_container", ""),
            }
        )
        if cluster_status.get("log"):
            cluster_log.extend(cluster_status.get("log", []))
            for entry in cluster_status.get("log", []):
                tool_context_chunks.append(_format_cluster_context(entry))
        if cluster_status.get("reused"):
            validation = cluster_manager.validate_cluster(run_id)
            cluster_info["validation"] = validation
            cluster_info["validation_history"].append({"iteration": 0, "validation": validation})
            entry = {
                "cmd": "cluster.validate (reuse)",
                "exit_code": 0 if validation.get("ok") else 1,
                "stdout": json.dumps(validation, indent=2),
                "stderr": "" if validation.get("ok") else validation.get("error", ""),
            }
            tool_context_chunks.append(_format_cluster_context(entry))
            cluster_log.append(entry)

    scenario_key = scenario or "general"
    if use_cluster:
        scenario_key = f"{scenario_key}-cluster-{cluster_size}"
    dml_error: Optional[str] = None
    dml_enabled = False
    dml_get_calls = 0
    dml_ingest_calls = 0
    cookbook_info = {
        "found": False,
        "cookbook_text": "",
        "sources": [],
        "latency_ms": 0,
    }
    ingest_info = {
        "ok": False,
        "ingested_id": "",
        "summary_id": "",
        "summary_latency_ms": 0,
        "error": None,
    }
    if use_dml:
        try:
            dml_get_calls += 1
            if dml_get_calls > 1:
                logger.error("dml_get_calls_per_run exceeded: %d", dml_get_calls)
            cookbook = dml_http_client.get_cookbook(scenario_key, goal, dml_top_k)
            dml_enabled = True
            cookbook_info.update(
                {
                    "found": cookbook.found,
                    "cookbook_text": cookbook.cookbook_text,
                    "sources": cookbook.sources,
                    "latency_ms": cookbook.latency_ms,
                }
            )
        except dml_http_client.DMLServiceError as exc:
            dml_error = str(exc)
            dml_enabled = False
    dml_info = {
        "requested": use_dml,
        "enabled": bool(use_dml and dml_enabled),
        "top_k": dml_top_k,
        "error": dml_error,
        "cookbook": cookbook_info,
        "ingest": ingest_info,
        "counters": {
            "dml_get_calls_per_run": dml_get_calls,
            "dml_ingest_calls_per_run": dml_ingest_calls,
        },
    }

    start_time = time.perf_counter()
    parallel_enabled = (
        parallel_agents
        if parallel_agents is not None
        else os.getenv("AGENT_PARALLEL", "1").strip().lower() in {"1", "true", "yes"}
    )
    yield _serialize(
        stages,
        goal,
        scenario,
        final="",
        total_ms=0,
        playground=playground_info,
        bare=bare_info,
        cluster=cluster_info,
        dml=dml_info,
        events=events,
    )

    outputs: Dict[str, AgentResult] = {}
    if resuming:
        for stage_name, output in _load_stage_artifacts(run_id).items():
            outputs[stage_name] = AgentResult(stage_name, output)
    tool_repair_attempted: set[str] = set()
    ops_repair_attempts: Dict[str, int] = {}
    tool_preflight_status: Dict[str, str] = {}
    file_write_repair_attempted: set[str] = set()
    failed = False
    ops_escalation: Optional[str] = None
    ops_fix_count = 0
    awaiting_human_input = False
    human_input_requests: List[str] = []
    final_override: Optional[str] = None
    handoff_count = 0
    handoff_max = int(os.getenv("AGENT_HANDOFF_MAX", "6"))
    trace: Dict[str, Dict[str, Any]] = {
        "stages": {},
        "timings": {},
        "errors": [],
        "ops_escalations": [],
        "dml": {
            "cookbook_found": cookbook_info["found"],
            "cookbook_sources": cookbook_info["sources"],
        },
        "playground": {
            "name": playground_info.get("name"),
            "image": playground_info.get("image"),
            "workspace_host": playground_info.get("workspace_host"),
            "workspace_container": playground_info.get("workspace_container"),
            "enabled": playground_info.get("enabled"),
        },
        "bare": {
            "enabled": bare_info.get("enabled"),
            "workspace_host": bare_info.get("workspace_host"),
        },
        "cluster": {
            "run_id": run_id,
            "size": cluster_size,
            "image": cluster_image,
            "network": cluster_info.get("network"),
            "containers": cluster_info.get("containers"),
            "api_port": cluster_info.get("api_port"),
            "web_port": cluster_info.get("web_port"),
            "api_url": f"http://localhost:{cluster_info.get('api_port')}" if cluster_info.get("api_port") else None,
            "web_url": f"http://localhost:{cluster_info.get('web_port')}" if cluster_info.get("web_port") else None,
            "workspace_host": cluster_info.get("workspace_host"),
            "workspace_container": cluster_info.get("workspace_container"),
            "enabled": cluster_info.get("enabled"),
        },
        "human_input_requests": [],
        "handoffs": [],
    }
    base_system_messages: List[str] = []
    if ssh_host and ssh_user:
        base_system_messages.append(f"SSH_TARGET: {ssh_user}@{ssh_host}.")
    if ssh_password:
        base_system_messages.append(
            "SSH_AUTH_POLICY: Credentials are provided in the user request. "
            "You are authorized to use password-based SSH non-interactively (sshpass) if needed. "
            "Do not request HUMAN_INPUT_REQUIRED for SSH credentials."
        )
        if ssh_key_available:
            base_system_messages.append(
                "SSH_FALLBACK: If key-based SSH fails, fall back to sshpass with the provided password."
            )
    if dml_enabled and cookbook_info["found"] and cookbook_info["cookbook_text"]:
        base_system_messages.append(f"DML_COOKBOOK_GUIDANCE:\n{cookbook_info['cookbook_text']}")
    if use_playground:
        base_system_messages.append(
            "PLAYGROUND_TOOLS_AVAILABLE: Use JSON tool requests to run container commands, e.g.\n"
            "```json\n"
            "{\"tool\":\"playground.exec\",\"cmd\":[\"bash\",\"-lc\",\"ls -la /workspace\"],\"timeout_s\":60}\n"
            "```\n"
            "Expose ports to the host with:\n"
            "```json\n"
            "{\"tool\":\"playground.expose_port\",\"host_port\":18000,\"container_port\":8000}\n"
            "```\n"
            "Infrastructure tooling (if mounted): use ssh/scp/rsync for remote hosts and kubectl for clusters. "
            "If PLAYGROUND_SSH_DIR or PLAYGROUND_KUBECONFIG is mounted, you can access /root/.ssh and /root/.kube/config.\n"
            "For host Docker/Compose, use playground.docker with args like:\n"
            "```json\n"
            "{\"tool\":\"playground.docker\",\"args\":[\"compose\",\"up\",\"-d\"],\"timeout_s\":600}\n"
            "```\n"
            "Manage Docker containers (create/restart/remove) with:\n"
            "```json\n"
            "{\"tool\":\"playground.docker\",\"args\":[\"run\",\"-d\",\"--name\",\"nemotron-playground-mybox\",\"ubuntu\",\"sleep\",\"infinity\"],\"timeout_s\":60}\n"
            "```\n"
            "Or write files with:\n"
            "```json\n"
            "{\"tool\":\"playground.write_file\",\"path\":\"/workspace/README.md\",\"content\":\"...\"}\n"
            "```"
        )
    if use_bare_metal:
        base_system_messages.append(
            "BARE_METAL_TOOLS_AVAILABLE: Use JSON tool requests to run commands on the host under PROJECT_CODE_DIR.\n"
            "Examples:\n"
            "```json\n"
            "{\"tool\":\"bare.exec\",\"cmd\":[\"bash\",\"-lc\",\"ls -la\"],\"timeout_s\":60}\n"
            "```\n"
            "```json\n"
            "{\"tool\":\"bare.write_file\",\"path\":\"<PROJECT_CODE_DIR>/app.py\",\"content\":\"...\"}\n"
            "```\n"
            "Bare metal safety:\n"
            "- Avoid sudo unless it is required and non-destructive; request human permission first.\n"
            "- Do not use apt/yum/brew or touch system files.\n"
            "- If you need permissions beyond PROJECT_CODE_DIR, request HUMAN_INPUT_REQUIRED with details.\n"
            "- If you need pip, first create a venv: python -m venv .venv and then use .venv/bin/pip.\n"
            "- SSH must be non-interactive. If only a password is available, use sshpass with StrictHostKeyChecking disabled.\n"
            "- Read-only checks (ls/cat) may target absolute paths; anything destructive still needs permission.\n"
        )
    if use_cluster:
        base_system_messages.append(
            "CLUSTER_TOOLS_AVAILABLE: Use JSON tool requests, e.g.\n"
            "```json\n"
            "{\"tool\":\"cluster.exec\",\"container\":\"<container>\",\"cmd\":[\"bash\",\"-lc\",\"ls -la /workspace\"],\"timeout_s\":60}\n"
            "```\n"
            "Or validate with:\n"
            "```json\n"
            "{\"tool\":\"cluster.validate\"}\n"
            "```"
        )
    if use_playground or use_cluster or use_bare_metal:
        base_system_messages.append(
            f"AGENT_PROJECTS_DIR: /workspace/agent_projects/{run_id} (write generated project files here; this maps to repo ./agent_projects/{run_id})."
        )
        if code_requested:
            base_system_messages.append(
                f"PROJECT_CODE_DIR: /workspace/agent_projects/{run_id}/{PROJECT_CODE_DIRNAME} (put code here)."
            )
        base_system_messages.append(
            "SAFE_WORKSPACE_RULE: All tool commands and file writes must stay under AGENT_PROJECTS_DIR."
        )
        base_system_messages.append(
            "DELIVERABLES_RULE: Put user-facing files in AGENT_PROJECTS_DIR (not under agent_logs/). "
            "agent_logs/ is reserved for logs and stage artifacts."
        )
        base_system_messages.append(TOOL_REQUEST_SCHEMA_TEXT)
        base_system_messages.append("TOOL_REQUEST_SCHEMA_JSON:\n" + json.dumps(TOOL_REQUEST_SCHEMA_JSON, indent=2))
        requested_port = _resolve_service_port(goal, scenario)
        if requested_port:
            if use_bare_metal:
                base_system_messages.append(
                    f"SERVER_PORT_REQUIREMENT: If you start a server, bind to 0.0.0.0:{requested_port} "
                    "on the host (bare metal)."
                )
            else:
                base_system_messages.append(
                    f"SERVER_PORT_REQUIREMENT: If you start a server, bind to 0.0.0.0:{requested_port} "
                    f"and expose it with playground.expose_port host_port={requested_port} container_port={requested_port}."
                )
        base_system_messages.append(
            "REQUIRED: If tools are available, run at least one tool command and write at least one file under AGENT_PROJECTS_DIR."
        )
    base_system_messages.append(
        "HANDOFF_PROTOCOL:\n"
        "- To route work to another role, include a line: NEXT_ROLE: <supervisor|planner|coder|reviewer|ops|aggregator>\n"
        "- Use this only when more work is required; otherwise omit it.\n"
    )
    if use_cluster:
        container_list = ", ".join([c.get("name", "") for c in cluster_info.get("containers", [])])
        base_system_messages.append(
            "CLUSTER_TOPOLOGY:\n"
            f"- run_id: {run_id}\n"
            f"- network: {cluster_info.get('network')}\n"
            f"- containers: {container_list or 'none'}\n"
            f"- api_host_port: {cluster_info.get('api_port')}\n"
            f"- web_host_port: {cluster_info.get('web_port')}\n"
            f"- api_url: http://localhost:{cluster_info.get('api_port')}\n"
            f"- web_url: http://localhost:{cluster_info.get('web_port')}\n"
            f"- workspace_host: {cluster_info.get('workspace_host')}\n"
            f"- workspace_container: {cluster_info.get('workspace_container')}\n"
        )
        if cluster_info.get("error"):
            base_system_messages.append(f"CLUSTER_BOOTSTRAP_ERROR:\n{cluster_info.get('error')}\n")

    def _build_extra_context(stage_name: str, tool_context_snapshot: Optional[List[str]] = None) -> str:
        extra_context = ""
        if stage_name in {"supervisor", "planner"} and ops_escalation:
            extra_context = f"Ops escalation:\n{ops_escalation}"
        if outputs:
            context_parts = [f"{k.title()} Output:\n{v.output}" for k, v in outputs.items()]
            extra_context = "\n\n".join(filter(None, [extra_context, "\n\n".join(context_parts)]))
        human_messages = _load_human_messages()
        if human_messages and stage_name == "supervisor":
            human_block = "Human input (most recent last):\n" + "\n".join(f"- {msg}" for msg in human_messages)
            extra_context = "\n\n".join(filter(None, [extra_context, human_block]))
        context_chunks = tool_context_snapshot if tool_context_snapshot is not None else tool_context_chunks
        if context_chunks:
            tool_context = "\n\n".join(context_chunks)
            extra_context = "\n\n".join(filter(None, [extra_context, f"Tool Command Log:\n{tool_context}"]))
        return extra_context

    def _project_files_exist() -> bool:
        run_root = BASE_DIR / "agent_projects" / run_id
        if not run_root.exists():
            return False
        project_code = run_root / PROJECT_CODE_DIRNAME
        if project_code.exists():
            for path in project_code.rglob("*"):
                if path.is_file():
                    return True
            return False
        for path in run_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(run_root)
            if rel.parts and rel.parts[0] in {AGENT_LOGS_DIRNAME, PROJECT_CODE_DIRNAME}:
                continue
            if rel.name in {"README.md", RUN_STATE_NAME}:
                continue
            return True
        return False

    def _promote_outputs_to_deliverable() -> bool:
        code_root = project_root
        deliverable_path = code_root / "DELIVERABLES.md"
        if deliverable_path.exists():
            return True
        content = ""
        if outputs:
            preferred = outputs.get("aggregator") or outputs.get("supervisor")
            if preferred and preferred.output:
                content = preferred.output.strip()
            if not content:
                parts: List[str] = []
                for stage in ("supervisor", "planner", "coder", "reviewer", "ops", "aggregator"):
                    result = outputs.get(stage)
                    if result and result.output:
                        parts.append(f"## {stage.title()} Output\n\n{result.output.strip()}")
                if parts:
                    content = "# Deliverables\n\n" + "\n\n".join(parts)
        if not content:
            return False
        try:
            deliverable_path.write_text(content + "\n")
        except OSError:
            return False
        return True

    def _completion_check() -> tuple[bool, str]:
        if not _project_files_exist():
            if not _promote_outputs_to_deliverable():
                return False, "No project files created under agent_projects."
        code_root = project_root
        if _requires_project_scaffold(goal, scenario):
            missing: List[str] = []
            if not (code_root / "Dockerfile").exists():
                missing.append("Dockerfile")
            if not ((code_root / "docker-compose.yml").exists() or (code_root / "docker-compose.yaml").exists()):
                missing.append("docker-compose.yml")
            if missing:
                return False, f"Missing required deliverables: {', '.join(missing)}"
        if _should_autobuild_webserver(goal):
            app_path = code_root / "app.py"
            if not app_path.exists():
                return False, "Missing app.py for hello world server."
            if use_playground:
                check_cmd = _prepend_safe_dir(["bash", "-lc", "curl -sf http://localhost:8000"])
                entry = playground_manager.exec_cmd(playground_name, check_cmd, timeout_s=30)
                entry["cmd"] = " ".join(check_cmd)
                entry["tool"] = "playground.exec"
                entry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(entry))
                playground_log.append(entry)
                if entry.get("exit_code") == 0:
                    return True, ""
                start_cmd = _prepend_safe_dir(["bash", "-lc", "python3 app.py"])
                start_entry = playground_manager.exec_cmd_detached(playground_name, start_cmd)
                start_entry["cmd"] = " ".join(start_cmd)
                start_entry["tool"] = "playground.exec_detached"
                start_entry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(start_entry))
                playground_log.append(start_entry)
                if start_entry.get("exit_code") == 0:
                    events.append("Orchestrator started hello world server (retry).")
                check_retry = playground_manager.exec_cmd(playground_name, check_cmd, timeout_s=30)
                check_retry["cmd"] = " ".join(check_cmd)
                check_retry["tool"] = "playground.exec"
                check_retry["agent"] = "orchestrator"
                tool_context_chunks.append(_format_tool_context(check_retry))
                playground_log.append(check_retry)
                if check_retry.get("exit_code") == 0:
                    return True, ""
                err = check_retry.get("stderr") or check_retry.get("stdout") or "server not responding"
                return False, f"Server check failed: {err}"
            return False, "Playground disabled; cannot verify server response."
        return True, ""

    def _finalize_project_layout() -> None:
        if not code_requested:
            return
        _ensure_project_code_dir(run_root)
        for item in run_root.iterdir():
            if item.name in {AGENT_LOGS_DIRNAME, PROJECT_CODE_DIRNAME, RUN_STATE_NAME, "README.md"}:
                continue
            target = project_root / item.name
            if target.exists():
                continue
            try:
                shutil.move(str(item), str(target))
            except OSError:
                continue

    def _build_system_messages(stage_name: str) -> List[str]:
        system_messages = list(base_system_messages)
        if long_run_mode and use_playground and stage_name == "coder":
            system_messages.append(
                "All generated files must be written under /workspace inside the playground container. "
                "Use tool steps to create files and run commands."
            )
        if long_run_mode and use_bare_metal and stage_name == "coder":
            system_messages.append(
                "All generated files must be written under PROJECT_CODE_DIR on the host. "
                "Use bare.write_file and bare.exec for commands."
            )
        if long_run_mode and use_cluster and stage_name == "coder":
            system_messages.append(
                "Cluster tools are available: use cluster.exec for container commands, cluster.logs for log collection, "
                "and cluster.validate for validation."
            )
        return system_messages

    def _prepend_safe_dir(cmd: List[str]) -> List[str]:
        if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-lc":
            return ["bash", "-lc", f"cd {project_root_container} && {cmd[2]}"]
        return cmd

    def _rewrite_tool_request_for_mode(req: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
        tool = req.get("tool")
        note: Optional[str] = None
        if not isinstance(tool, str):
            return req, None

        container_root = f"/workspace/agent_projects/{run_id}"
        host_root = str(run_root)
        host_code_root = str(project_root)
        cluster_containers = cluster_info.get("containers", []) if isinstance(cluster_info, dict) else []

        def _cluster_default_container() -> Optional[str]:
            for container in cluster_containers:
                if container.get("role") == "api" and container.get("name"):
                    return str(container.get("name"))
            for container in cluster_containers:
                if container.get("name"):
                    return str(container.get("name"))
            return None

        def _cluster_write_cmd(path_value: str, content_value: str) -> List[str]:
            payload = base64.b64encode(content_value.encode("utf-8")).decode("ascii")
            script = (
                "if command -v python3 >/dev/null 2>&1; then\n"
                "  python3 - <<'PY'\n"
                "import base64, pathlib\n"
                f"path = pathlib.Path({path_value!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_bytes(base64.b64decode('{payload}'))\n"
                "PY\n"
                "elif command -v python >/dev/null 2>&1; then\n"
                "  python - <<'PY'\n"
                "import base64, pathlib\n"
                f"path = pathlib.Path({path_value!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_bytes(base64.b64decode('{payload}'))\n"
                "PY\n"
                "else\n"
                f"  mkdir -p \"$(dirname {path_value!r})\" && echo '{payload}' | base64 -d > {path_value!r}\n"
                "fi"
            )
            return ["bash", "-lc", script]

        def _maybe_wrap_ssh(cmd_value: List[str]) -> List[str]:
            if not ssh_use_password:
                return cmd_value
            if not cmd_value:
                return cmd_value
            if cmd_value[0] == "sshpass":
                return cmd_value
            if cmd_value[0] == "ssh":
                return [
                    "sshpass",
                    "-p",
                    ssh_password,
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "PubkeyAuthentication=no",
                    "-o",
                    "PreferredAuthentications=password",
                    "-o",
                    "HostKeyAlgorithms=+ssh-ed25519",
                    "-o",
                    "PubkeyAcceptedAlgorithms=+ssh-ed25519",
                ] + cmd_value[1:]
            return cmd_value

        def _remote_target_available() -> bool:
            return bool(ssh_host and ssh_user)

        def _should_remote_exec(cmd_value: List[str]) -> bool:
            if not _remote_target_available():
                return False
            if not cmd_value:
                return False
            if cmd_value[0] in {"ssh", "sshpass"}:
                return False
            cmd_text = " ".join(cmd_value)
            if str(project_root) in cmd_text or container_root in cmd_text:
                return False
            remote_tokens = [
                "/opt/ai-stack",
                "docker ",
                "docker-compose",
                "ollama",
                "openwebui",
                "nvidia-smi",
                "/etc/os-release",
                "uname",
                "ufw",
                "systemctl",
                "ss -lntp",
                "curl ",
            ]
            return any(token in cmd_text for token in remote_tokens)

        def _wrap_remote_exec(cmd_value: List[str]) -> List[str]:
            base_cmd = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "HostKeyAlgorithms=+ssh-ed25519",
                "-o",
                "PubkeyAcceptedAlgorithms=+ssh-ed25519",
            ]
            if not ssh_use_password:
                base_cmd += ["-o", "BatchMode=yes"]
            base_cmd += [f"{ssh_user}@{ssh_host}"] + cmd_value
            return _maybe_wrap_ssh(base_cmd)

        def _should_remote_write(path_value: Optional[str]) -> bool:
            if not _remote_target_available():
                return False
            if not isinstance(path_value, str) or not path_value.startswith("/"):
                return False
            if path_value.startswith(str(project_root)):
                return False
            if path_value.startswith("/workspace/agent_projects"):
                return False
            return True

        def _remote_write_cmd(path_value: str, content_value: str) -> List[str]:
            payload = base64.b64encode(content_value.encode("utf-8")).decode("ascii")
            script = (
                "if command -v python3 >/dev/null 2>&1; then\n"
                "  python3 - <<'PY'\n"
                "import base64, pathlib\n"
                f"path = pathlib.Path({path_value!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_bytes(base64.b64decode('{payload}'))\n"
                "PY\n"
                "elif command -v python >/dev/null 2>&1; then\n"
                "  python - <<'PY'\n"
                "import base64, pathlib\n"
                f"path = pathlib.Path({path_value!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                f"path.write_bytes(base64.b64decode('{payload}'))\n"
                "PY\n"
                "else\n"
                f"  mkdir -p \"$(dirname {path_value!r})\" && echo '{payload}' | base64 -d > {path_value!r}\n"
                "fi"
            )
            return _wrap_remote_exec(["bash", "-lc", script])

        def _map_exec_paths(cmd_value: List[str]) -> List[str]:
            if not cmd_value:
                return cmd_value
            if cmd_value[0] in {"ssh", "sshpass"}:
                return cmd_value
            mapped: List[str] = []
            for arg in cmd_value:
                if isinstance(arg, str) and container_root in arg:
                    mapped.append(arg.replace(container_root, host_root))
                else:
                    mapped.append(arg)
            return mapped

        def _rewrite_shell_ssh(cmd_value: List[str]) -> tuple[List[str], Optional[str]]:
            if (
                len(cmd_value) < 3
                or cmd_value[0] != "bash"
                or cmd_value[1] != "-lc"
                or not isinstance(cmd_value[2], str)
            ):
                return cmd_value, None
            shell_cmd = cmd_value[2].strip()
            if not shell_cmd.startswith("ssh "):
                return cmd_value, None
            if "sshpass" in shell_cmd:
                return cmd_value, None
            rest = shell_cmd[len("ssh ") :].strip()
            base_opts = (
                "-o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null "
                "-o HostKeyAlgorithms=+ssh-ed25519 "
                "-o PubkeyAcceptedAlgorithms=+ssh-ed25519"
            )
            if ssh_password:
                pw = shlex.quote(ssh_password)
                if ssh_key_available:
                    ssh_try = f"ssh -o BatchMode=yes {base_opts} {rest}"
                    ssh_fallback = (
                        f"sshpass -p {pw} ssh {base_opts} "
                        "-o PubkeyAuthentication=no -o PreferredAuthentications=password "
                        f"{rest}"
                    )
                    return [cmd_value[0], cmd_value[1], f"{ssh_try} || {ssh_fallback}"], "Injected ssh key+password fallback"
                ssh_cmd = (
                    f"sshpass -p {pw} ssh {base_opts} "
                    "-o PubkeyAuthentication=no -o PreferredAuthentications=password "
                    f"{rest}"
                )
                return [cmd_value[0], cmd_value[1], ssh_cmd], "Injected sshpass for shell ssh"
            ssh_cmd = f"ssh -o BatchMode=yes {base_opts} {rest}"
            return [cmd_value[0], cmd_value[1], ssh_cmd], "Injected non-interactive ssh options"

        def _maybe_rewrite_ssh_keyscan(cmd_value: List[str]) -> tuple[List[str], Optional[str]]:
            if not cmd_value or cmd_value[0] != "ssh-keyscan":
                return cmd_value, None
            host = None
            for arg in reversed(cmd_value[1:]):
                if arg in {">", ">>", "|"} or arg.startswith("-"):
                    continue
                host = arg
                break
            if not host:
                return cmd_value, None
            if ssh_use_password and ssh_password:
                rewritten = [
                    "sshpass",
                    "-p",
                    ssh_password,
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "PubkeyAuthentication=no",
                    "-o",
                    "PreferredAuthentications=password",
                    "-o",
                    "HostKeyAlgorithms=+ssh-ed25519",
                    "-o",
                    "PubkeyAcceptedAlgorithms=+ssh-ed25519",
                    host,
                    "true",
                ]
                return rewritten, "Rewrote ssh-keyscan -> sshpass ssh (non-interactive)"
            rewritten = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                host,
                "true",
            ]
            return rewritten, "Rewrote ssh-keyscan -> ssh (non-interactive)"

        def _map_path(path_value: str, to_container: bool) -> str:
            if not path_value:
                return path_value
            if to_container:
                if path_value.startswith(host_code_root):
                    rel = path_value[len(host_code_root) :].lstrip("/")
                    return f"{container_root}/{PROJECT_CODE_DIRNAME}/{rel}" if rel else f"{container_root}/{PROJECT_CODE_DIRNAME}"
                if path_value.startswith(host_root):
                    rel = path_value[len(host_root) :].lstrip("/")
                    return f"{container_root}/{rel}" if rel else container_root
            else:
                if path_value.startswith(container_root):
                    rel = path_value[len(container_root) :].lstrip("/")
                    return f"{host_root}/{rel}" if rel else host_root
            return path_value

        if use_bare_metal and not use_playground and tool.startswith("playground."):
            rewritten = dict(req)
            if tool in {"playground.exec", "playground.exec_detached"}:
                rewritten["tool"] = "bare.exec"
                cmd_value = rewritten.get("cmd")
                if isinstance(cmd_value, list):
                    mapped_cmd = _map_exec_paths(cmd_value)
                    if mapped_cmd != cmd_value:
                        rewritten["cmd"] = mapped_cmd
                        cmd_value = mapped_cmd
                    shell_rewritten, shell_note = _rewrite_shell_ssh(cmd_value)
                    if shell_note:
                        rewritten["cmd"] = shell_rewritten
                        note = shell_note
                        return rewritten, note
                    if _should_remote_exec(cmd_value):
                        rewritten["cmd"] = _wrap_remote_exec(cmd_value)
                        note = "Rewrote playground.exec -> bare.exec (remote ssh)"
                        return rewritten, note
                    rewritten_cmd, rewrite_note = _maybe_rewrite_ssh_keyscan(cmd_value)
                    if rewrite_note:
                        rewritten["cmd"] = rewritten_cmd
                        note = rewrite_note
                    wrapped = _maybe_wrap_ssh(rewritten.get("cmd", cmd_value))
                    if wrapped != rewritten.get("cmd", cmd_value):
                        rewritten["cmd"] = wrapped
                        note = "Rewrote playground.exec -> bare.exec (sshpass injected)"
                    elif note is None:
                        note = f"Rewrote {tool} -> bare.exec"
                else:
                    note = f"Rewrote {tool} -> bare.exec"
            elif tool == "playground.write_file":
                rewritten["tool"] = "bare.write_file"
                path_value = rewritten.get("path")
                if isinstance(path_value, str):
                    rewritten["path"] = _map_path(path_value, to_container=False)
                if _should_remote_write(rewritten.get("path")):
                    content_value = rewritten.get("content", "")
                    rewritten = {
                        "tool": "bare.exec",
                        "cmd": _remote_write_cmd(str(rewritten.get("path")), str(content_value)),
                        "timeout_s": int(rewritten.get("timeout_s", 60)),
                    }
                    note = "Rewrote playground.write_file -> bare.exec (remote write)"
                    return rewritten, note
                note = f"Rewrote {tool} -> bare.write_file"
            return rewritten, note

        if use_playground and not use_bare_metal and tool.startswith("bare."):
            rewritten = dict(req)
            if tool == "bare.exec":
                rewritten["tool"] = "playground.exec"
                note = "Rewrote bare.exec -> playground.exec"
            elif tool == "bare.write_file":
                rewritten["tool"] = "playground.write_file"
                path_value = rewritten.get("path")
                if isinstance(path_value, str):
                    rewritten["path"] = _map_path(path_value, to_container=True)
                note = "Rewrote bare.write_file -> playground.write_file"
            return rewritten, note

        if use_cluster and not use_playground and not use_bare_metal and tool.startswith(("playground.", "bare.")):
            rewritten = dict(req)
            container_name = _cluster_default_container()
            if tool in {"playground.exec", "playground.exec_detached", "bare.exec"}:
                rewritten["tool"] = "cluster.exec"
                if container_name:
                    rewritten["container"] = container_name
                note = f"Rewrote {tool} -> cluster.exec"
            elif tool in {"playground.write_file", "bare.write_file"}:
                rewritten["tool"] = "cluster.exec"
                path_value = rewritten.get("path")
                content_value = rewritten.get("content", "")
                if isinstance(path_value, str):
                    mapped_path = _map_path(path_value, to_container=True)
                else:
                    mapped_path = f"{container_root}/fallback.txt"
                rewritten["cmd"] = _cluster_write_cmd(mapped_path, str(content_value))
                rewritten.pop("path", None)
                rewritten.pop("content", None)
                if container_name:
                    rewritten["container"] = container_name
                note = f"Rewrote {tool} -> cluster.exec(write_file)"
            return rewritten, note

        if tool == "bare.exec":
            rewritten = dict(req)
            cmd_value = rewritten.get("cmd")
            if isinstance(cmd_value, list):
                mapped_cmd = _map_exec_paths(cmd_value)
                if mapped_cmd != cmd_value:
                    rewritten["cmd"] = mapped_cmd
                    cmd_value = mapped_cmd
                shell_rewritten, shell_note = _rewrite_shell_ssh(cmd_value)
                if shell_note:
                    rewritten["cmd"] = shell_rewritten
                    note = shell_note
                    return rewritten, note
                if _should_remote_exec(cmd_value):
                    rewritten["cmd"] = _wrap_remote_exec(cmd_value)
                    note = "Rewrote bare.exec -> bare.exec (remote ssh)"
                    return rewritten, note
                rewritten_cmd, rewrite_note = _maybe_rewrite_ssh_keyscan(cmd_value)
                if rewrite_note:
                    rewritten["cmd"] = rewritten_cmd
                    note = rewrite_note
                    cmd_value = rewritten_cmd
                wrapped = _maybe_wrap_ssh(cmd_value)
                if wrapped != cmd_value:
                    rewritten["cmd"] = wrapped
                    note = "Injected sshpass for non-interactive SSH"
                    return rewritten, note
                if rewrite_note:
                    return rewritten, note
            return rewritten, None

        if tool == "bare.write_file":
            path_value = req.get("path")
            content_value = req.get("content", "")
            if _should_remote_write(path_value):
                rewritten = {
                    "tool": "bare.exec",
                    "cmd": _remote_write_cmd(str(path_value), str(content_value)),
                    "timeout_s": int(req.get("timeout_s", 60)),
                }
                note = "Rewrote bare.write_file -> bare.exec (remote write)"
                return rewritten, note

        return req, None

    def _queue_permission_request(stage_name: str, reason: str) -> None:
        nonlocal awaiting_human_input, human_input_requests, stage_queue
        if reason and reason not in human_input_requests:
            human_input_requests.append(reason)
            trace["human_input_requests"].append(reason)
        awaiting_human_input = True
        _save_run_state(
            run_id,
            {
                "awaiting_human_input": True,
                "requesting_stage": stage_name,
                "human_input_requests": list(human_input_requests),
                "stage_queue": list(stage_queue),
            },
        )
        events.append(f"{stage_name.title()} requires permission: {reason}")
        if len(events) > 500:
            del events[:-500]

    def _handle_stage_result(
        stage_name: str,
        result: AgentResult,
        elapsed_ms: float,
        extra_context: str,
        system_messages: List[str],
        max_tokens: Optional[int],
    ) -> None:
        nonlocal failed, ops_escalation, ops_fix_count, handoff_count, awaiting_human_input, human_input_requests, stage_queue, tool_repair_attempted, ops_repair_attempts, tool_preflight_status, file_write_repair_attempted
        retry_info: Dict[str, Any] = {}
        if _should_retry_output(result.output):
            retry_context = _compact_extra_context(extra_context)
            retry_messages = _compact_system_messages(system_messages)
            retry_messages.append(
                "RETRY_EMPTY_OUTPUT: Your last response was empty. Respond concisely with actionable output. "
                "If tools are required, emit valid tool JSON."
            )
            retry_start = time.perf_counter()
            retry_result = call_agent(
                stage_name,
                goal,
                scenario,
                max_tokens=max_tokens,
                extra_context=retry_context,
                system_messages=retry_messages,
            )
            retry_elapsed = (time.perf_counter() - retry_start) * 1000
            retry_info = {
                "attempted": True,
                "elapsed_ms": retry_elapsed,
                "used_compact_context": True,
                "used_compact_system_messages": True,
            }
            if not _should_retry_output(retry_result.output):
                result = retry_result
                elapsed_ms += retry_elapsed
                events.append(f"{stage_name.title()} retried after empty output.")
                if len(events) > 500:
                    del events[:-500]
            else:
                retry_info["failed"] = True
        tokens = result.tokens or estimate_tokens(result.output)
        tok_s = compute_throughput(tokens, elapsed_ms)
        metrics = StageMetrics(ms=elapsed_ms, ttft_ms=elapsed_ms, tokens=tokens, tok_s=tok_s)
        outputs[stage_name] = result
        stage_trace: Dict[str, Any] = {
            "output": result.output,
            "error": None,
            "ms": metrics.ms,
            "ttft_ms": metrics.ttft_ms,
            "tok_s": metrics.tok_s,
            "tokens": metrics.tokens,
            "extra_context": extra_context,
            "system_messages": system_messages,
            "max_tokens": max_tokens,
        }
        if retry_info:
            stage_trace["retry"] = retry_info
        artifact_output_override: Optional[str] = None
        if use_playground or use_cluster or use_bare_metal:
            output_text = result.output or ""
            preflight_errors: List[str] = []
            preflight_output = ""
            if stage_name not in tool_preflight_status:
                ok, preflight_errors, preflight_output = _run_tool_preflight(
                    stage_name,
                    goal,
                    scenario,
                    extra_context,
                    system_messages,
                    max_tokens,
                )
                tool_preflight_status[stage_name] = "ok" if ok else "failed"
                if ok:
                    events.append(f"{stage_name.title()} tool schema preflight passed.")
                else:
                    events.append(
                        f"{stage_name.title()} tool schema preflight failed: "
                        + " | ".join(preflight_errors)
                    )
                if len(events) > 500:
                    del events[:-500]
                stage_trace["tool_preflight"] = {
                    "status": tool_preflight_status[stage_name],
                    "errors": preflight_errors,
                    "output": preflight_output,
                }
                if not ok:
                    raise RuntimeError(
                        f"Tool schema preflight failed after {int(os.getenv('TOOL_PREFLIGHT_REPAIR_MAX', '3'))} repair attempts."
                    )

            tool_requests: List[Dict[str, Any]] = []
            parse_errors: List[str] = []
            tool_hints = False
            tool_requests, parse_errors, tool_hints = _extract_tool_requests_with_errors(output_text)
            if tool_preflight_status.get(stage_name) != "ok":
                parse_errors.append("Tool preflight failed; continuing with tool parsing.")
            rewritten_requests: List[Dict[str, Any]] = []
            rewrite_notes: List[str] = []
            for req in tool_requests:
                rewritten, note = _rewrite_tool_request_for_mode(req)
                rewritten_requests.append(rewritten)
                if note:
                    rewrite_notes.append(note)
            if rewrite_notes:
                stage_trace["tool_rewrites"] = rewrite_notes
                events.append(f"{stage_name.title()} tool requests rewritten: " + "; ".join(rewrite_notes))
                if len(events) > 500:
                    del events[:-500]
            tool_requests = rewritten_requests
            normalize_notes: List[str] = []
            for req in tool_requests:
                tool_name = req.get("tool")
                if tool_name in {"playground.exec", "playground.exec_detached", "cluster.exec", "bare.exec"}:
                    cmd = req.get("cmd")
                    if isinstance(cmd, list):
                        normalized = _normalize_exec_cmd(cmd)
                        if normalized != cmd:
                            req["cmd"] = normalized
                            normalize_notes.append(f"Normalized {tool_name} cmd to bash -lc")
            if normalize_notes:
                stage_trace.setdefault("tool_rewrites", []).extend(normalize_notes)
                events.append(f"{stage_name.title()} tool requests normalized: " + "; ".join(normalize_notes))
                if len(events) > 500:
                    del events[:-500]
            validation_errors: List[str] = []
            for req in tool_requests:
                validation_errors.extend(_validate_tool_request(req))
                tool_name = req.get("tool")
                if tool_name in {"playground.exec", "playground.exec_detached", "playground.write_file", "playground.docker", "playground.expose_port"} and not use_playground:
                    validation_errors.append(f"Tool {tool_name} requested but playground is disabled.")
                if tool_name in {"bare.exec", "bare.write_file"} and not use_bare_metal:
                    validation_errors.append(f"Tool {tool_name} requested but bare metal is disabled.")
                if tool_name in {"cluster.exec", "cluster.logs", "cluster.validate"} and not use_cluster:
                    validation_errors.append(f"Tool {tool_name} requested but cluster is disabled.")
            if (
                tool_requests
                and not validation_errors
                and parse_errors == ["Tool preflight failed; continuing with tool parsing."]
            ):
                parse_errors = []
            repair_output = ""
            if (parse_errors or validation_errors or (tool_hints and not tool_requests)) and stage_name not in tool_repair_attempted:
                tool_repair_attempted.add(stage_name)
                issues = parse_errors + validation_errors
                snippet = output_text.strip()
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + "\n...truncated..."
                repair_messages = list(system_messages)
                repair_messages.append(
                    "TOOL_REQUEST_REPAIR: Your last output contained invalid or unparseable tool JSON. "
                    "Return ONLY valid JSON tool requests that follow TOOL_REQUEST_SCHEMA v1. No extra text."
                )
                repair_context = "\n\n".join(
                    filter(
                        None,
                        [
                            extra_context,
                            "Previous output (invalid tool JSON):\n" + snippet,
                            "Issues:\n- " + "\n- ".join(issues) if issues else "",
                        ],
                    )
                )
                repair_result = call_agent(
                    stage_name,
                    goal,
                    scenario,
                    max_tokens=_limit_tokens(max_tokens, 256),
                    extra_context=repair_context,
                    system_messages=repair_messages,
                )
                repair_output = repair_result.output or ""
                _append_tool_repair_log(run_id, f"{stage_name.title()} tool repair", repair_output, issues)
                repaired_requests, repaired_parse_errors, _ = _extract_tool_requests_with_errors(repair_output)
                rewritten_repaired: List[Dict[str, Any]] = []
                rewrite_notes = []
                for req in repaired_requests:
                    rewritten, note = _rewrite_tool_request_for_mode(req)
                    rewritten_repaired.append(rewritten)
                    if note:
                        rewrite_notes.append(note)
                repaired_requests = rewritten_repaired
                if rewrite_notes:
                    stage_trace.setdefault("tool_rewrites", []).extend(rewrite_notes)
                repaired_validation_errors: List[str] = []
                for req in repaired_requests:
                    repaired_validation_errors.extend(_validate_tool_request(req))
                    tool_name = req.get("tool")
                    if tool_name in {"playground.exec", "playground.exec_detached", "playground.write_file", "playground.docker", "playground.expose_port"} and not use_playground:
                        repaired_validation_errors.append(f"Tool {tool_name} requested but playground is disabled.")
                    if tool_name in {"bare.exec", "bare.write_file"} and not use_bare_metal:
                        repaired_validation_errors.append(f"Tool {tool_name} requested but bare metal is disabled.")
                    if tool_name in {"cluster.exec", "cluster.logs", "cluster.validate"} and not use_cluster:
                        repaired_validation_errors.append(f"Tool {tool_name} requested but cluster is disabled.")
                if repaired_requests and not repaired_parse_errors and not repaired_validation_errors:
                    tool_requests = repaired_requests
                    parse_errors = []
                    validation_errors = []
                    if repair_output:
                        artifact_output_override = repair_output
                    events.append(f"{stage_name.title()} tool JSON repaired after validation errors.")
                else:
                    if repaired_parse_errors or repaired_validation_errors:
                        events.append(
                            f"{stage_name.title()} tool JSON repair failed: "
                            + " | ".join(repaired_parse_errors + repaired_validation_errors)
                        )
            if (parse_errors or validation_errors or (tool_hints and not tool_requests)) and stage_name != "ops":
                max_ops_repairs = int(os.getenv("OPS_TOOL_REPAIR_MAX", "3"))
                while parse_errors or validation_errors or (tool_hints and not tool_requests):
                    ops_count = ops_repair_attempts.get(stage_name, 0)
                    if ops_count >= max_ops_repairs:
                        break
                    ops_repair_attempts[stage_name] = ops_count + 1
                    ops_messages = list(base_system_messages)
                    ops_messages.append(
                        "OPS_TOOL_JSON_REPAIR: The previous agent output contained invalid tool JSON. "
                        "Return ONLY valid JSON tool requests that follow TOOL_REQUEST_SCHEMA v1. "
                        "Allowed shapes: single object, array, or {\"tool_calls\":[...]} wrapper. "
                        "No extra text, no Markdown."
                    )
                    ops_messages.append(TOOL_REQUEST_SCHEMA_TEXT)
                    ops_messages.append("TOOL_REQUEST_SCHEMA_JSON:\n" + json.dumps(TOOL_REQUEST_SCHEMA_JSON, indent=2))
                    ops_context = "\n\n".join(
                        filter(
                            None,
                            [
                                extra_context,
                                "Previous output (invalid tool JSON):\n" + (output_text.strip() or "<empty>"),
                                "Issues:\n- " + "\n- ".join(parse_errors + validation_errors),
                            ],
                        )
                    )
                    ops_result = call_agent(
                        "ops",
                        goal,
                        scenario,
                        max_tokens=_limit_tokens(max_tokens, 512),
                        extra_context=ops_context,
                        system_messages=ops_messages,
                    )
                    ops_repair_output = ops_result.output or ""
                    _append_tool_repair_log(
                        run_id,
                        f"Ops tool repair for {stage_name.title()} (attempt {ops_repair_attempts[stage_name]})",
                        ops_repair_output,
                        parse_errors + validation_errors,
                    )
                    ops_requests, ops_parse_errors, _ = _extract_tool_requests_with_errors(ops_repair_output)
                    rewritten_ops: List[Dict[str, Any]] = []
                    rewrite_notes = []
                    for req in ops_requests:
                        rewritten, note = _rewrite_tool_request_for_mode(req)
                        rewritten_ops.append(rewritten)
                        if note:
                            rewrite_notes.append(note)
                    ops_requests = rewritten_ops
                    if rewrite_notes:
                        stage_trace.setdefault("tool_rewrites", []).extend(rewrite_notes)
                    ops_validation_errors: List[str] = []
                    for req in ops_requests:
                        ops_validation_errors.extend(_validate_tool_request(req))
                        tool_name = req.get("tool")
                        if tool_name in {"playground.exec", "playground.exec_detached", "playground.write_file", "playground.docker", "playground.expose_port"} and not use_playground:
                            ops_validation_errors.append(f"Tool {tool_name} requested but playground is disabled.")
                        if tool_name in {"bare.exec", "bare.write_file"} and not use_bare_metal:
                            ops_validation_errors.append(f"Tool {tool_name} requested but bare metal is disabled.")
                        if tool_name in {"cluster.exec", "cluster.logs", "cluster.validate"} and not use_cluster:
                            ops_validation_errors.append(f"Tool {tool_name} requested but cluster is disabled.")
                    if ops_requests and not ops_parse_errors and not ops_validation_errors:
                        tool_requests = ops_requests
                        parse_errors = []
                        validation_errors = []
                        if ops_repair_output:
                            artifact_output_override = ops_repair_output
                        events.append(f"Ops repaired tool JSON for {stage_name.title()}.")
                        break
                    parse_errors = ops_parse_errors or parse_errors
                    validation_errors = ops_validation_errors or validation_errors
                    if ops_parse_errors or ops_validation_errors:
                        events.append(
                            f"Ops tool JSON repair failed for {stage_name.title()}: "
                            + " | ".join(ops_parse_errors + ops_validation_errors)
                        )
                        if len(events) > 500:
                            del events[:-500]
            if artifact_output_override is None and (parse_errors or validation_errors):
                artifact_output_override = json.dumps(
                    {
                        "error": "tool_json_invalid",
                        "issues": parse_errors + validation_errors,
                    },
                    indent=2,
                )
            tool_entries: List[Dict[str, Any]] = []
            wrote_file = False
            bare_permissions = _extract_bare_permissions(_load_human_messages()) if use_bare_metal else set()
            for request in tool_requests:
                req_errors = _validate_tool_request(request)
                if req_errors:
                    events.append(
                        f"{stage_name.title()} tool request skipped: " + " | ".join(req_errors)
                    )
                    if len(events) > 500:
                        del events[:-500]
                    continue
                tool_name = request.get("tool")
                entry: Dict[str, Any]
                if tool_name in {"playground.exec", "playground.exec_detached"} and use_playground:
                    cmd = request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    detach = bool(request.get("detach")) or tool_name == "playground.exec_detached"
                    if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                        entry = {
                            "cmd": str(cmd),
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid command format. Expected list[str].",
                        }
                    else:
                        safe_cmd = _prepend_safe_dir(cmd)
                        if detach:
                            entry = playground_manager.exec_cmd_detached(playground_name, safe_cmd)
                        else:
                            entry = playground_manager.exec_cmd(playground_name, safe_cmd, timeout_s=timeout_s)
                        entry["cmd"] = " ".join(safe_cmd)
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                    if detach and entry.get("exit_code") == 0:
                        events.append(f"{stage_name.title()} agent started detached command: {' '.join(safe_cmd)}")
                        if len(events) > 500:
                            del events[:-500]
                elif tool_name == "playground.write_file" and use_playground:
                    path = request.get("path")
                    content = request.get("content")
                    if not isinstance(path, str) or not isinstance(content, str):
                        entry = {
                            "cmd": f"write_file {path}",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid write_file payload. Expected path/content strings.",
                        }
                    else:
                        entry = playground_manager.write_file(playground_name, path, content)
                        entry["cmd"] = f"write_file {path}"
                    entry["agent"] = stage_name
                    wrote_file = True
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                elif tool_name == "playground.docker" and use_playground:
                    args = request.get("args") or request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                        entry = {
                            "cmd": str(args),
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid docker args. Expected list[str].",
                        }
                    else:
                        entry = playground_manager.docker_cmd(args, timeout_s=timeout_s)
                        entry["cmd"] = f"docker {' '.join(args)}"
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                elif tool_name == "playground.expose_port" and use_playground:
                    host_port = request.get("host_port") or request.get("port")
                    container_port = request.get("container_port") or request.get("target_port") or host_port
                    try:
                        host_port = int(host_port)
                        container_port = int(container_port)
                    except (TypeError, ValueError):
                        entry = {
                            "cmd": f"expose_port {host_port}:{container_port}",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid port values. Expected integers.",
                        }
                    else:
                        result = playground_manager.expose_port(playground_name, host_port, container_port)
                        if result.get("ok"):
                            playground_info["web_port"] = host_port
                            playground_info.setdefault("exposed_ports", [])
                            if host_port not in playground_info["exposed_ports"]:
                                playground_info["exposed_ports"].append(host_port)
                            events.append(f"Port {host_port} exposed to playground {playground_name}.")
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 0,
                                "stdout": json.dumps(result),
                                "stderr": "",
                            }
                        else:
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 1,
                                "stdout": "",
                                "stderr": result.get("error", "Failed to expose port."),
                            }
                    entry["tool"] = tool_name
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                elif tool_name == "cluster.exec" and use_cluster:
                    container = request.get("container")
                    cmd = request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    if not isinstance(container, str) or not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                        entry = {
                            "cmd": f"{container} {cmd}",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid cluster.exec payload. Expected container + cmd list.",
                        }
                    else:
                        safe_cmd = _prepend_safe_dir(cmd)
                        entry = cluster_manager.exec_in(container, safe_cmd, timeout_s=timeout_s)
                        entry["cmd"] = f"{container} :: {' '.join(safe_cmd)}"
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_cluster_context(entry))
                    cluster_log.append(entry)
                elif tool_name == "cluster.validate" and use_cluster:
                    validation = cluster_manager.validate_cluster(run_id)
                    cluster_info["validation"] = validation
                    entry = {
                        "cmd": "cluster.validate",
                        "exit_code": 0 if validation.get("ok") else 1,
                        "stdout": json.dumps(validation, indent=2),
                        "stderr": "" if validation.get("ok") else validation.get("error", ""),
                    }
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_cluster_context(entry))
                    cluster_log.append(entry)
                elif tool_name == "cluster.logs" and use_cluster:
                    container = request.get("container")
                    if not isinstance(container, str):
                        entry = {
                            "cmd": f"{container} logs",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid cluster.logs payload. Expected container string.",
                        }
                    else:
                        tail_value = request.get("tail", 200)
                        try:
                            tail = int(tail_value)
                        except (TypeError, ValueError):
                            tail = 200
                        entry = cluster_manager.container_logs(container, tail=tail)
                        entry["cmd"] = f"{container} :: logs (tail={tail})"
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_cluster_context(entry))
                    cluster_log.append(entry)
                elif tool_name == "cluster.validate" and use_cluster:
                    validation = cluster_manager.validate_cluster(run_id)
                    entry = {
                        "cmd": "cluster.validate (fixer)",
                        "exit_code": 0 if validation.get("ok") else 1,
                        "stdout": json.dumps(validation, indent=2),
                        "stderr": "" if validation.get("ok") else validation.get("error", ""),
                    }
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_cluster_context(entry))
                    cluster_log.append(entry)
                elif tool_name == "bare.exec" and use_bare_metal:
                    cmd = request.get("cmd")
                    timeout_s = int(request.get("timeout_s", 60))
                    if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                        entry = {
                            "cmd": str(cmd),
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid command format. Expected list[str].",
                        }
                    else:
                        allowed, reason, requires_permission = _validate_bare_command(cmd, project_root, bare_permissions)
                        if not allowed:
                            if requires_permission:
                                _queue_permission_request(
                                    stage_name,
                                    f"Permission required for bare.exec: {reason} (cmd={' '.join(cmd)})",
                                )
                            entry = {
                                "cmd": " ".join(cmd),
                                "exit_code": 126,
                                "stdout": "",
                                "stderr": reason,
                            }
                        else:
                            safe_cmd = _prepend_bare_safe_dir(cmd, project_root)
                            entry = _run_bare_exec(safe_cmd, project_root, timeout_s=timeout_s)
                            entry["cmd"] = " ".join(safe_cmd)
                            if (
                                ssh_password
                                and not ssh_use_password
                                and isinstance(safe_cmd, list)
                                and safe_cmd
                                and safe_cmd[0] == "ssh"
                                and _ssh_auth_failed(entry)
                            ):
                                retry_cmd = _strip_batchmode_args(safe_cmd)
                                retry_cmd = [
                                    "sshpass",
                                    "-p",
                                    ssh_password,
                                    "ssh",
                                    "-o",
                                    "StrictHostKeyChecking=no",
                                    "-o",
                                    "UserKnownHostsFile=/dev/null",
                                    "-o",
                                    "PubkeyAuthentication=no",
                                    "-o",
                                    "PreferredAuthentications=password",
                                    "-o",
                                    "HostKeyAlgorithms=+ssh-ed25519",
                                    "-o",
                                    "PubkeyAcceptedAlgorithms=+ssh-ed25519",
                                ] + retry_cmd[1:]
                                retry_entry = _run_bare_exec(retry_cmd, project_root, timeout_s=timeout_s)
                                retry_entry["cmd"] = " ".join(retry_cmd)
                                retry_entry["tool"] = "bare.exec"
                                retry_entry["agent"] = stage_name
                                tool_context_chunks.append(_format_tool_context(retry_entry))
                                bare_log.append(retry_entry)
                                entry = retry_entry
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    bare_log.append(entry)
                elif tool_name == "bare.write_file" and use_bare_metal:
                    path = request.get("path")
                    content = request.get("content")
                    if not isinstance(path, str) or not isinstance(content, str):
                        entry = {
                            "cmd": f"write_file {path}",
                            "exit_code": 125,
                            "stdout": "",
                            "stderr": "Invalid write_file payload. Expected path/content strings.",
                        }
                    else:
                        target = Path(path)
                        if not target.is_absolute():
                            target = project_root / target
                        try:
                            target = target.resolve()
                        except OSError:
                            entry = {
                                "cmd": f"write_file {path}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid path.",
                            }
                        else:
                            if project_root not in target.parents and target != project_root:
                                _queue_permission_request(
                                    stage_name,
                                    f"Permission required for bare.write_file outside PROJECT_CODE_DIR: {target}",
                                )
                                entry = {
                                    "cmd": f"write_file {target}",
                                    "exit_code": 126,
                                    "stdout": "",
                                    "stderr": "Path must be under PROJECT_CODE_DIR.",
                                }
                            else:
                                entry = _run_bare_write(target, content)
                                entry["cmd"] = f"write_file {target}"
                                wrote_file = True
                    entry["agent"] = stage_name
                    tool_context_chunks.append(_format_tool_context(entry))
                    bare_log.append(entry)
                else:
                    continue
                entry["tool"] = tool_name
                tool_entries.append(entry)
            if tool_entries:
                stage_trace["tool_requests"] = tool_entries
            if parse_errors or validation_errors or repair_output:
                stage_trace["tool_validation"] = {
                    "parse_errors": parse_errors,
                    "validation_errors": validation_errors,
                    "repair_output": repair_output,
                }
            if (
                stage_name == "coder"
                and (use_playground or use_bare_metal)
                and not wrote_file
                and stage_name not in file_write_repair_attempted
            ):
                file_write_repair_attempted.add(stage_name)
                file_tool = "bare.write_file" if use_bare_metal else "playground.write_file"
                repair_messages = list(system_messages)
                repair_messages.append(
                    "FILE_WRITE_REQUIRED: You did not write any files. Return ONLY valid JSON tool requests that "
                    f"create the required files under AGENT_PROJECTS_DIR using {file_tool}. No prose."
                )
                repair_messages.append(TOOL_REQUEST_SCHEMA_TEXT)
                repair_context = _compact_extra_context(extra_context)
                repair_result = call_agent(
                    stage_name,
                    goal,
                    scenario,
                    max_tokens=_limit_tokens(max_tokens, 512),
                    extra_context=repair_context,
                    system_messages=repair_messages,
                )
                new_requests, new_parse_errors, _ = _extract_tool_requests_with_errors(repair_result.output or "")
                new_validation_errors: List[str] = []
                for req in new_requests:
                    new_validation_errors.extend(_validate_tool_request(req))
                if new_parse_errors or new_validation_errors:
                    events.append(
                        f"{stage_name.title()} file-write repair failed: "
                        + " | ".join(new_parse_errors + new_validation_errors)
                    )
                    if len(events) > 500:
                        del events[:-500]
                else:
                    for req in new_requests:
                        if req.get("tool") == "playground.write_file" and use_playground:
                            path = req.get("path")
                            content = req.get("content")
                            entry = playground_manager.write_file(playground_name, path, content)
                            entry["cmd"] = f"write_file {path}"
                            entry["tool"] = "playground.write_file"
                            entry["agent"] = stage_name
                            tool_context_chunks.append(_format_tool_context(entry))
                            playground_log.append(entry)
                            if entry.get("exit_code") == 0:
                                wrote_file = True
                        elif req.get("tool") == "bare.write_file" and use_bare_metal:
                            path = req.get("path")
                            content = req.get("content")
                            if isinstance(path, str) and isinstance(content, str):
                                target = Path(path)
                                if not target.is_absolute():
                                    target = project_root / target
                                try:
                                    target = target.resolve()
                                except OSError:
                                    entry = {
                                        "cmd": f"write_file {path}",
                                        "exit_code": 125,
                                        "stdout": "",
                                        "stderr": "Invalid path.",
                                    }
                                else:
                                    entry = _run_bare_write(target, content)
                                    entry["cmd"] = f"write_file {target}"
                                    if entry.get("exit_code") == 0:
                                        wrote_file = True
                            else:
                                entry = {
                                    "cmd": f"write_file {path}",
                                    "exit_code": 125,
                                    "stdout": "",
                                    "stderr": "Invalid write_file payload.",
                                }
                            entry["tool"] = "bare.write_file"
                            entry["agent"] = stage_name
                            tool_context_chunks.append(_format_tool_context(entry))
                            bare_log.append(entry)
                        elif req.get("tool") in {"playground.exec", "playground.exec_detached"} and use_playground:
                            cmd = req.get("cmd")
                            timeout_s = int(req.get("timeout_s", 60))
                            detach = bool(req.get("detach")) or req.get("tool") == "playground.exec_detached"
                            if isinstance(cmd, list) and all(isinstance(arg, str) for arg in cmd):
                                safe_cmd = _prepend_safe_dir(cmd)
                                if detach:
                                    entry = playground_manager.exec_cmd_detached(playground_name, safe_cmd)
                                else:
                                    entry = playground_manager.exec_cmd(playground_name, safe_cmd, timeout_s=timeout_s)
                                entry["cmd"] = " ".join(safe_cmd)
                                entry["tool"] = req.get("tool")
                                entry["agent"] = stage_name
                                tool_context_chunks.append(_format_tool_context(entry))
                                playground_log.append(entry)
                        elif req.get("tool") == "playground.expose_port" and use_playground:
                            host_port = req.get("host_port")
                            container_port = req.get("container_port") or req.get("target_port") or host_port
                            if isinstance(host_port, int) and isinstance(container_port, int):
                                result = playground_manager.expose_port(playground_name, host_port, container_port)
                                entry = {
                                    "cmd": f"expose_port {host_port}:{container_port}",
                                    "exit_code": 0 if result.get("ok") else 1,
                                    "stdout": json.dumps(result),
                                    "stderr": "" if result.get("ok") else result.get("error", "Failed to expose port."),
                                    "tool": "playground.expose_port",
                                    "agent": stage_name,
                                }
                                tool_context_chunks.append(_format_tool_context(entry))
                                playground_log.append(entry)
                        elif req.get("tool") == "bare.exec" and use_bare_metal:
                            cmd = req.get("cmd")
                            timeout_s = int(req.get("timeout_s", 60))
                            if isinstance(cmd, list) and all(isinstance(arg, str) for arg in cmd):
                                safe_cmd = _prepend_bare_safe_dir(cmd, project_root)
                                entry = _run_bare_exec(safe_cmd, project_root, timeout_s=timeout_s)
                                entry["cmd"] = " ".join(safe_cmd)
                            else:
                                entry = {
                                    "cmd": str(cmd),
                                    "exit_code": 125,
                                    "stdout": "",
                                    "stderr": "Invalid command format. Expected list[str].",
                                }
                            entry["tool"] = "bare.exec"
                            entry["agent"] = stage_name
                            tool_context_chunks.append(_format_tool_context(entry))
                            bare_log.append(entry)
                    if wrote_file:
                        events.append(f"{stage_name.title()} file-write repair succeeded.")
                        if len(events) > 500:
                            del events[:-500]
                        if tool_preflight_status.get(stage_name) == "failed":
                            tool_preflight_status[stage_name] = "ok_recovered"
            if use_playground and not wrote_file:
                safe_name = _safe_stage_name(stage_name)
                container_path = f"/workspace/agent_projects/{run_id}/{AGENT_LOGS_DIRNAME}/{safe_name}.md"
                artifact_output = artifact_output_override if artifact_output_override is not None else output_text
                artifact_text = f"# {stage_name.title()} Output\n\n{artifact_output}\n"
                entry = playground_manager.write_file(playground_name, container_path, artifact_text)
                entry["cmd"] = f"write_file {container_path}"
                entry["tool"] = "playground.write_file"
                entry["agent"] = stage_name
                tool_context_chunks.append(_format_tool_context(entry))
                playground_log.append(entry)
                if entry.get("exit_code") == 0:
                    events.append(f"{stage_name.title()} agent wrote {container_path}.")
                    if len(events) > 500:
                        del events[:-500]
                else:
                    err = entry.get("stderr") or entry.get("stdout") or "unknown error"
                    events.append(f"{stage_name.title()} agent failed to write {container_path}: {err}")
                    if len(events) > 500:
                        del events[:-500]
            verify_entry: Optional[Dict[str, Any]] = None
            if stage_name == "coder" and use_playground and not wrote_file and _should_autobuild_webserver(goal):
                fallback_port = _resolve_service_port(goal, scenario)
                app_path = f"{project_root_container}/app.py"
                app_content = (
                    "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "import os\n\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        self.send_response(200)\n"
                    "        self.send_header('Content-Type', 'text/plain; charset=utf-8')\n"
                    "        self.end_headers()\n"
                    "        self.wfile.write(b'Hello world!')\n\n"
                    "def main():\n"
                    f"    port = int(os.getenv('PORT', '{fallback_port}'))\n"
                    "    server = HTTPServer(('', port), Handler)\n"
                    "    print(f'Serving on :{port}')\n"
                    "    server.serve_forever()\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
                requirements_content = "# No external dependencies\n"
                dockerfile_content = (
                    "FROM python:3.11-slim\n"
                    "WORKDIR /app\n"
                    "COPY requirements.txt .\n"
                    "RUN pip install --no-cache-dir -r requirements.txt\n"
                    "COPY . .\n"
                    f"EXPOSE {fallback_port}\n"
                    "CMD [\"python\", \"app.py\"]\n"
                )
                compose_content = (
                    "version: '3.8'\n\n"
                    "services:\n"
                    "  hello-world:\n"
                    "    build: .\n"
                    "    ports:\n"
                    f"      - \"{fallback_port}:{fallback_port}\"\n"
                    "    restart: unless-stopped\n"
                )
                start_sh_content = (
                    "#!/usr/bin/env bash\n"
                    "set -e\n"
                    "python3 -m venv .venv\n"
                    ". .venv/bin/activate\n"
                    "if grep -v '^#' requirements.txt | grep -q '\\\\S'; then\n"
                    "  pip install -r requirements.txt\n"
                    "fi\n"
                    "exec python app.py\n"
                )
                fallback_files = {
                    "app.py": app_content,
                    "requirements.txt": requirements_content,
                    "Dockerfile": dockerfile_content,
                    "docker-compose.yml": compose_content,
                    "start.sh": start_sh_content,
                }
                write_ok = False
                for filename, content in fallback_files.items():
                    file_path = f"{project_root_container}/{filename}"
                    entry = playground_manager.write_file(playground_name, file_path, content)
                    entry["cmd"] = f"write_file {file_path}"
                    entry["tool"] = "playground.write_file"
                    entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(entry))
                    playground_log.append(entry)
                    if entry.get("exit_code") == 0:
                        write_ok = True
                wrote_file = wrote_file or write_ok
                if write_ok:
                    events.append(f"Orchestrator wrote {app_path}.")
                    if len(events) > 500:
                        del events[:-500]
                    start_cmd = _prepend_safe_dir(["bash", "-lc", "python3 app.py"])
                    start_entry = playground_manager.exec_cmd_detached(playground_name, start_cmd)
                    start_entry["cmd"] = " ".join(start_cmd)
                    start_entry["tool"] = "playground.exec_detached"
                    start_entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(start_entry))
                    playground_log.append(start_entry)
                    if start_entry.get("exit_code") == 0:
                        events.append("Orchestrator started hello world server (detached).")
                        if len(events) > 500:
                            del events[:-500]
                    verify_cmd = _prepend_safe_dir(["bash", "-lc", f"sleep 1; curl -sf http://localhost:{fallback_port}"])
                    verify_entry = playground_manager.exec_cmd(playground_name, verify_cmd, timeout_s=30)
                    verify_entry["cmd"] = " ".join(verify_cmd)
                    verify_entry["tool"] = "playground.exec"
                    verify_entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(verify_entry))
                    playground_log.append(verify_entry)
            if stage_name == "coder" and use_bare_metal and not wrote_file and _should_autobuild_webserver(goal):
                fallback_port = _resolve_service_port(goal, scenario)
                app_path = project_root / "app.py"
                app_content = (
                    "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "import os\n\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        self.send_response(200)\n"
                    "        self.send_header('Content-Type', 'text/plain; charset=utf-8')\n"
                    "        self.end_headers()\n"
                    "        self.wfile.write(b'Hello world!')\n\n"
                    "def main():\n"
                    f"    port = int(os.getenv('PORT', '{fallback_port}'))\n"
                    "    server = HTTPServer(('', port), Handler)\n"
                    "    print(f'Serving on :{port}')\n"
                    "    server.serve_forever()\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
                requirements_content = "# No external dependencies\n"
                dockerfile_content = (
                    "FROM python:3.11-slim\n"
                    "WORKDIR /app\n"
                    "COPY requirements.txt .\n"
                    "RUN pip install --no-cache-dir -r requirements.txt\n"
                    "COPY . .\n"
                    f"EXPOSE {fallback_port}\n"
                    "CMD [\"python\", \"app.py\"]\n"
                )
                compose_content = (
                    "version: '3.8'\n\n"
                    "services:\n"
                    "  hello-world:\n"
                    "    build: .\n"
                    "    ports:\n"
                    f"      - \"{fallback_port}:{fallback_port}\"\n"
                    "    restart: unless-stopped\n"
                )
                start_sh_content = (
                    "#!/usr/bin/env bash\n"
                    "set -e\n"
                    "python3 -m venv .venv\n"
                    ". .venv/bin/activate\n"
                    "if grep -v '^#' requirements.txt | grep -q '\\\\S'; then\n"
                    "  pip install -r requirements.txt\n"
                    "fi\n"
                    "exec python app.py\n"
                )
                fallback_files = {
                    "app.py": app_content,
                    "requirements.txt": requirements_content,
                    "Dockerfile": dockerfile_content,
                    "docker-compose.yml": compose_content,
                    "start.sh": start_sh_content,
                }
                write_ok = False
                for filename, content in fallback_files.items():
                    file_path = project_root / filename
                    entry = _run_bare_write(file_path, content)
                    entry["cmd"] = f"write_file {file_path}"
                    entry["tool"] = "bare.write_file"
                    entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(entry))
                    bare_log.append(entry)
                    if entry.get("exit_code") == 0:
                        write_ok = True
                wrote_file = wrote_file or write_ok
                if write_ok:
                    events.append(f"Orchestrator wrote {app_path}.")
                    if len(events) > 500:
                        del events[:-500]
                    verify_cmd = ["bash", "-lc", f"PORT={fallback_port} python3 app.py >/tmp/agent_fallback.log 2>&1 & pid=$!; sleep 1; curl -sf http://127.0.0.1:{fallback_port}/; code=$?; kill $pid; wait $pid 2>/dev/null; exit $code"]
                    verify_cmd = _prepend_bare_safe_dir(verify_cmd, project_root)
                    verify_entry = _run_bare_exec(verify_cmd, project_root, timeout_s=30)
                    verify_entry["cmd"] = " ".join(verify_cmd)
                    verify_entry["tool"] = "bare.exec"
                    verify_entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(verify_entry))
                    bare_log.append(verify_entry)
                    if verify_entry.get("exit_code") == 0:
                        events.append("Orchestrator verified hello world response (bare metal).")
                        if len(events) > 500:
                            del events[:-500]
                else:
                    events.append(f"Orchestrator failed to write fallback files under {project_root}.")
                    if len(events) > 500:
                        del events[:-500]
            if stage_name == "coder" and use_cluster and not wrote_file and _should_autobuild_webserver(goal):
                fallback_port = _resolve_service_port(goal, scenario)
                app_path = project_root / "app.py"
                app_content = (
                    "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "import os\n\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        self.send_response(200)\n"
                    "        self.send_header('Content-Type', 'text/plain; charset=utf-8')\n"
                    "        self.end_headers()\n"
                    "        self.wfile.write(b'Hello world!')\n\n"
                    "def main():\n"
                    f"    port = int(os.getenv('PORT', '{fallback_port}'))\n"
                    "    server = HTTPServer(('', port), Handler)\n"
                    "    print(f'Serving on :{port}')\n"
                    "    server.serve_forever()\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
                requirements_content = "# No external dependencies\n"
                dockerfile_content = (
                    "FROM python:3.11-slim\n"
                    "WORKDIR /app\n"
                    "COPY requirements.txt .\n"
                    "RUN pip install --no-cache-dir -r requirements.txt\n"
                    "COPY . .\n"
                    f"EXPOSE {fallback_port}\n"
                    "CMD [\"python\", \"app.py\"]\n"
                )
                compose_content = (
                    "version: '3.8'\n\n"
                    "services:\n"
                    "  hello-world:\n"
                    "    build: .\n"
                    "    ports:\n"
                    f"      - \"{fallback_port}:{fallback_port}\"\n"
                    "    restart: unless-stopped\n"
                )
                start_sh_content = (
                    "#!/usr/bin/env bash\n"
                    "set -e\n"
                    "python3 -m venv .venv\n"
                    ". .venv/bin/activate\n"
                    "if grep -v '^#' requirements.txt | grep -q '\\\\S'; then\n"
                    "  pip install -r requirements.txt\n"
                    "fi\n"
                    "exec python app.py\n"
                )
                fallback_files = {
                    "app.py": app_content,
                    "requirements.txt": requirements_content,
                    "Dockerfile": dockerfile_content,
                    "docker-compose.yml": compose_content,
                    "start.sh": start_sh_content,
                }
                write_ok = False
                for filename, content in fallback_files.items():
                    file_path = project_root / filename
                    entry = _run_bare_write(file_path, content)
                    entry["cmd"] = f"write_file {file_path}"
                    entry["tool"] = "bare.write_file"
                    entry["agent"] = "orchestrator"
                    tool_context_chunks.append(_format_tool_context(entry))
                    bare_log.append(entry)
                    if entry.get("exit_code") == 0:
                        write_ok = True
                wrote_file = wrote_file or write_ok
                if write_ok:
                    events.append(f"Orchestrator wrote {app_path}.")
                    if len(events) > 500:
                        del events[:-500]
                else:
                    events.append(f"Orchestrator failed to write fallback files under {project_root}.")
                    if len(events) > 500:
                        del events[:-500]
            if long_run_mode and use_playground:
                fixed_command = FIXED_PLAYGROUND_COMMANDS.get(stage_name)
                if fixed_command:
                    fixed_entry = _run_playground_command(
                        playground_name,
                        fixed_command,
                        playground_log,
                        tool_context_chunks,
                        agent=stage_name,
                    )
                    stage_trace["playground_commands"] = [fixed_entry]
                if stage_name == "planner":
                    skeleton_entries = []
                    skeleton_entries.append(
                        _run_playground_command(
                            playground_name,
                            ["bash", "-lc", "mkdir -p /workspace/app /workspace/tests"],
                            playground_log,
                            tool_context_chunks,
                            agent=stage_name,
                        )
                    )
                    skeleton_entries.append(
                        _run_playground_command(
                            playground_name,
                            [
                                "bash",
                                "-lc",
                                "cat <<'EOF' > /workspace/README.md\n# Workspace\n\nInitialized by long-run mode.\nEOF",
                            ],
                            playground_log,
                            tool_context_chunks,
                            agent=stage_name,
                        )
                    )
                    stage_trace.setdefault("playground_commands", []).extend(skeleton_entries)
        artifact_output = artifact_output_override if artifact_output_override is not None else (result.output or "")
        if artifact_output_override is not None and artifact_output_override != (result.output or ""):
            stage_trace["output_raw"] = result.output
            stage_trace["output"] = artifact_output_override
        artifact_text = f"# {stage_name.title()} Output\n\n{artifact_output}\n"
        artifact_path = _write_stage_artifact(run_id, stage_name, artifact_text)
        if artifact_path:
            events.append(f"{stage_name.title()} agent saved output to {artifact_path}.")
            if len(events) > 500:
                del events[:-500]

        trace["stages"][stage_name] = stage_trace
        if _agent_gave_up(result.output):
            trace["errors"].append(
                {"stage": stage_name, "error": f"Agent requested stop: {GIVE_UP_PHRASE}"}
            )
            failed = True
        _update_stage(
            stages,
            stage_name,
            status="done",
            ms=metrics.ms,
            ttft_ms=metrics.ttft_ms,
            tok_s=metrics.tok_s,
            tokens=metrics.tokens,
            output=result.output,
        )
        requested = _extract_human_input_requests(result.output)
        if requested:
            awaiting_human_input = True
            human_input_requests.extend(requested)
            trace["human_input_requests"].extend(requested)
            _save_run_state(
                run_id,
                {
                    "awaiting_human_input": True,
                    "requesting_stage": stage_name,
                    "human_input_requests": list(human_input_requests),
                    "stage_queue": list(stage_queue),
                },
            )
            events.append(
                f"{stage_name.title()} requested human input: " + " | ".join(requested)
            )
            if len(events) > 500:
                del events[:-500]
        if stage_name in {"supervisor", "planner"} and ops_escalation:
            ops_escalation = None
        if handoff_count < handoff_max and not awaiting_human_input:
            next_role = _extract_handoff(result.output)
            if next_role:
                stage_queue.insert(0, next_role)
                handoff_count += 1
                trace["handoffs"].append({"from": stage_name, "to": next_role, "count": handoff_count})
        return

    def _finalize_stage(stage_name: str) -> Optional[str]:
        nonlocal failed, ops_escalation, ops_fix_count, attempt, outputs, awaiting_human_input, human_input_requests, final_override
        total_ms = (time.perf_counter() - start_time) * 1000
        if awaiting_human_input:
            base_text = ""
            if outputs:
                base_text = outputs.get("aggregator", outputs.get(stage_name, AgentResult(stage_name, ""))).output
            human_block = _format_human_input_block(human_input_requests)
            summary = _format_access_summary(run_id, playground_info, cluster_info, bare_info, code_requested)
            final_text = "\n\n".join(filter(None, [base_text, human_block, summary])).strip()
            final_override = final_text
            trace["timings"]["total_ms"] = total_ms
            yield _serialize(
                stages,
                goal,
                scenario,
                final=final_text,
                total_ms=total_ms,
                playground=playground_info,
                bare=bare_info,
                cluster=cluster_info,
                dml=dml_info,
                events=events,
            )
            return final_text
        final_text = outputs.get("aggregator", AgentResult(stage_name, "")).output if outputs else ""
        if stage_name == "aggregator":
            summary = _format_access_summary(run_id, playground_info, cluster_info, bare_info, code_requested)
            final_text = f"{final_text}\n\n{summary}".strip()
        trace["timings"]["total_ms"] = total_ms
        yield _serialize(
            stages,
            goal,
            scenario,
            final=final_text,
            total_ms=total_ms,
            playground=playground_info,
            bare=bare_info,
            cluster=cluster_info,
            dml=dml_info,
            events=events,
        )

        if not failed and stage_name == "ops" and stage_name in outputs:
            ops_failure = _parse_ops_failure(outputs[stage_name].output)
            if ops_failure:
                trace["ops_escalations"].append(ops_failure)
                ops_fix_count += 1
                ops_escalation = "\n".join(
                    filter(
                        None,
                        [
                            f"Error: {ops_failure.get('error')}",
                            f"Instruction: {ops_failure.get('instruction')}",
                            f"Ops output:\n{ops_failure.get('raw')}",
                        ],
                    )
                )
                retry_stages = ["supervisor", "planner", "coder", "reviewer", "ops"]
                if "aggregator" in stage_queue:
                    insert_at = stage_queue.index("aggregator")
                    stage_queue[insert_at:insert_at] = retry_stages
                else:
                    stage_queue.extend(retry_stages)

        if not failed and stage_name == "aggregator":
            ok, reason = _completion_check()
            if not ok:
                events.append(f"Completion check failed: {reason}")
                if max_attempts == 0 or attempt < max_attempts:
                    attempt += 1
                    if max_attempts == 0:
                        events.append(f"Retrying attempt {attempt} (unlimited retries)")
                    else:
                        events.append(f"Retrying attempt {attempt} of {max_attempts}")
                    ops_escalation = f"Task incomplete: {reason}"
                    outputs.clear()
                    for stage in stages:
                        stage.status = "queued"
                        stage.ms = 0.0
                        stage.ttft_ms = 0.0
                        stage.tok_s = 0.0
                        stage.tokens = 0
                        stage.output = ""
                        stage.error = None
                    retry_stages = list(stage_order)
                    stage_queue.extend(retry_stages)
                    yield _serialize(
                        stages,
                        goal,
                        scenario,
                        final=final_text,
                        total_ms=total_ms,
                        playground=playground_info,
                        bare=bare_info,
                        cluster=cluster_info,
                        dml=dml_info,
                        events=events,
                    )
                    return None
                if max_attempts != 0:
                    failed = True
                    trace["errors"].append({"stage": "completion", "error": reason})

        if failed:
            if use_playground:
                playground_info["ready_for_removal"] = True
                if auto_remove_playground:
                    removal = playground_manager.remove_playground(playground_name)
                    playground_info["remove_result"] = removal
                    if removal.get("ok"):
                        playground_info["status"] = "removed"
                    else:
                        playground_info["status"] = "error"
                        playground_info["error"] = removal.get("error")
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    bare=bare_info,
                    cluster=cluster_info,
                    dml=dml_info,
                    events=events,
                )
            if use_cluster:
                cluster_info["ready_for_removal"] = True
            if outputs:
                return final_text
            return ""
        return None

    def _call_agent_timed(
        stage_name: str,
        extra_context: str,
        system_messages: List[str],
        max_tokens: Optional[int],
    ) -> tuple[str, AgentResult, float]:
        start = time.perf_counter()
        result = call_agent(
            stage_name,
            goal,
            scenario,
            max_tokens=max_tokens,
            extra_context=extra_context,
            system_messages=system_messages or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return stage_name, result, elapsed_ms

    while stage_queue:
        stage_name = stage_queue.pop(0)

        if parallel_enabled and stage_name == "planner":
            _update_stage(stages, stage_name, status="running")
            yield _serialize(
                stages,
                goal,
                scenario,
                final="",
                total_ms=(time.perf_counter() - start_time) * 1000,
                playground=playground_info,
                bare=bare_info,
                cluster=cluster_info,
                dml=dml_info,
                events=events,
            )
            extra_context = _build_extra_context(stage_name)
            max_tokens = _max_tokens_for_stage(stage_name, fast)
            system_messages = _build_system_messages(stage_name)
            try:
                _, result, elapsed_ms = _call_agent_timed(stage_name, extra_context, system_messages, max_tokens)
                _handle_stage_result(stage_name, result, elapsed_ms, extra_context, system_messages, max_tokens)
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                trace["stages"][stage_name] = {
                    "output": "",
                    "error": str(exc),
                    "ms": elapsed_ms,
                    "ttft_ms": elapsed_ms,
                    "tok_s": 0.0,
                    "tokens": 0,
                    "extra_context": extra_context,
                    "system_messages": system_messages,
                    "max_tokens": max_tokens,
                }
                trace["errors"].append({"stage": stage_name, "error": str(exc)})
                _update_stage(
                    stages,
                    stage_name,
                    status="failed",
                    ms=elapsed_ms,
                    ttft_ms=elapsed_ms,
                    error=str(exc),
                )
                failed = True

            finalize_result = yield from _finalize_stage(stage_name)
            if finalize_result is not None:
                break
            continue

        if parallel_enabled and stage_name in {"coder", "reviewer", "ops"}:
            batch = [stage_name]
            while stage_queue and stage_queue[0] in {"coder", "reviewer", "ops"}:
                batch.append(stage_queue.pop(0))
            for name in batch:
                _update_stage(stages, name, status="running")
            yield _serialize(
                stages,
                goal,
                scenario,
                final="",
                total_ms=(time.perf_counter() - start_time) * 1000,
                playground=playground_info,
                bare=bare_info,
                cluster=cluster_info,
                dml=dml_info,
                events=events,
            )
            tool_context_snapshot = list(tool_context_chunks)
            shared_context = _build_extra_context(batch[0], tool_context_snapshot=tool_context_snapshot)
            futures = {}
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                for name in batch:
                    max_tokens = _max_tokens_for_stage(name, fast)
                    system_messages = _build_system_messages(name)
                    futures[name] = executor.submit(_call_agent_timed, name, shared_context, system_messages, max_tokens)
            results: Dict[str, Dict[str, Any]] = {}
            for name, future in futures.items():
                try:
                    stage_key, result, elapsed_ms = future.result()
                    stage_system_messages = _build_system_messages(stage_key)
                    results[stage_key] = {
                        "result": result,
                        "elapsed_ms": elapsed_ms,
                        "system_messages": stage_system_messages,
                        "max_tokens": _max_tokens_for_stage(stage_key, fast),
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    results[name] = {
                        "result": AgentResult(name, ""),
                        "elapsed_ms": 0.0,
                        "system_messages": _build_system_messages(name),
                        "max_tokens": _max_tokens_for_stage(name, fast),
                        "error": str(exc),
                    }

            for name in batch:
                try:
                    entry = results[name]
                    if entry.get("error"):
                        raise RuntimeError(entry["error"])
                    result = entry["result"]
                    elapsed_ms = float(entry["elapsed_ms"])
                    stage_system_messages = entry["system_messages"]
                    stage_max_tokens = entry.get("max_tokens")
                    _handle_stage_result(
                        name,
                        result,
                        elapsed_ms,
                        shared_context,
                        stage_system_messages,
                        stage_max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    entry = results.get(name, {})
                    stage_system_messages = entry.get("system_messages", [])
                    stage_max_tokens = entry.get("max_tokens", _max_tokens_for_stage(name, fast))
                    trace["stages"][name] = {
                        "output": "",
                        "error": str(exc),
                        "ms": elapsed_ms,
                        "ttft_ms": elapsed_ms,
                        "tok_s": 0.0,
                        "tokens": 0,
                        "extra_context": shared_context,
                        "system_messages": stage_system_messages,
                        "max_tokens": stage_max_tokens,
                    }
                    trace["errors"].append({"stage": name, "error": str(exc)})
                    _update_stage(
                        stages,
                        name,
                        status="failed",
                        ms=elapsed_ms,
                        ttft_ms=elapsed_ms,
                        error=str(exc),
                    )
                    failed = True
                finalize_result = yield from _finalize_stage(name)
                if finalize_result is not None:
                    break
            if failed:
                break
            continue

        _update_stage(stages, stage_name, status="running")
        yield _serialize(
            stages,
            goal,
            scenario,
            final="",
            total_ms=(time.perf_counter() - start_time) * 1000,
            playground=playground_info,
            bare=bare_info,
            cluster=cluster_info,
            dml=dml_info,
            events=events,
        )

        stage_start = time.perf_counter()
        try:
            extra_context = _build_extra_context(stage_name)
            max_tokens = _max_tokens_for_stage(stage_name, fast)
            system_messages = _build_system_messages(stage_name)
            result = call_agent(
                stage_name,
                goal,
                scenario,
                max_tokens=max_tokens,
                extra_context=extra_context,
                system_messages=system_messages or None,
            )
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            _handle_stage_result(stage_name, result, elapsed_ms, extra_context, system_messages, max_tokens)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            trace["stages"][stage_name] = {
                "output": "",
                "error": str(exc),
                "ms": elapsed_ms,
                "ttft_ms": elapsed_ms,
                "tok_s": 0.0,
                "tokens": 0,
                "extra_context": extra_context,
                "system_messages": system_messages,
                "max_tokens": max_tokens,
            }
            trace["errors"].append({"stage": stage_name, "error": str(exc)})
            _update_stage(
                stages,
                stage_name,
                status="failed",
                ms=elapsed_ms,
                ttft_ms=elapsed_ms,
                error=str(exc),
            )
            failed = True

        finalize_result = yield from _finalize_stage(stage_name)
        if finalize_result is not None:
            break

    total_ms = (time.perf_counter() - start_time) * 1000
    if not awaiting_human_input:
        _finalize_project_layout()
    if final_override:
        final_text = final_override
    else:
        final_text = outputs.get("aggregator", AgentResult("aggregator", "")).output
        final_text = f"{final_text}\n\n{_format_access_summary(run_id, playground_info, cluster_info, bare_info, code_requested)}".strip()
    if playground_log:
        trace["playground"]["log"] = playground_log
    if bare_log:
        trace.setdefault("bare", {})["log"] = bare_log
    if use_cluster:
        if long_run_mode and not awaiting_human_input:
            fix_iterations: List[Dict[str, Any]] = []
            last_validation: Dict[str, Any] = {}
            iteration = 0
            while True:
                iteration += 1
                cluster_info["iteration"] = iteration
                validation = cluster_manager.validate_cluster(run_id)
                last_validation = validation
                cluster_info["validation"] = validation
                cluster_info["validation_history"].append({"iteration": iteration, "validation": validation})
                entry = {
                    "cmd": f"cluster.validate (iter {iteration})",
                    "exit_code": 0 if validation.get("ok") else 1,
                    "stdout": json.dumps(validation, indent=2),
                    "stderr": "" if validation.get("ok") else validation.get("error", ""),
                }
                cluster_log.append(entry)
                tool_context_chunks.append(_format_cluster_context(entry))
                trace["cluster"]["validation"] = validation
                trace["cluster"]["validation_history"] = cluster_info["validation_history"]
                if cluster_log:
                    trace["cluster"]["log"] = cluster_log
                total_ms = (time.perf_counter() - start_time) * 1000
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    bare=bare_info,
                    cluster=cluster_info,
                    dml=dml_info,
                    events=events,
                )
                if validation.get("ok"):
                    break
                fixer_context = _format_validation_context(validation, cluster_info)
                fixer_system_messages = list(base_system_messages)
                fixer_system_messages.append(
                    "Cluster tools are available: use cluster.exec for container commands and cluster.logs for logs."
                )
                fixer_result = call_agent(
                    "fixer",
                    goal,
                    scenario,
                    max_tokens=_max_tokens_for_stage("fixer", fast),
                    extra_context=fixer_context,
                    system_messages=fixer_system_messages,
                )
                if _agent_gave_up(fixer_result.output):
                    failed = True
                    cluster_info["error"] = f"Agent requested stop: {GIVE_UP_PHRASE}"
                    break
                tool_requests = _extract_tool_requests(fixer_result.output)
                tool_entries: List[Dict[str, Any]] = []
                fix_actions: List[Dict[str, Any]] = []
                for request in tool_requests:
                    tool_name = request.get("tool")
                    entry = {}
                    if tool_name in {"playground.exec", "playground.exec_detached"} and use_playground:
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        detach = bool(request.get("detach")) or tool_name == "playground.exec_detached"
                        if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": str(cmd),
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid command format. Expected list[str].",
                            }
                        else:
                            if detach:
                                entry = playground_manager.exec_cmd_detached(playground_name, cmd)
                            else:
                                entry = playground_manager.exec_cmd(playground_name, cmd, timeout_s=timeout_s)
                            entry["cmd"] = " ".join(cmd)
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "playground.expose_port" and use_playground:
                        host_port = request.get("host_port") or request.get("port")
                        container_port = request.get("container_port") or request.get("target_port") or host_port
                        try:
                            host_port = int(host_port)
                            container_port = int(container_port)
                        except (TypeError, ValueError):
                            entry = {
                                "cmd": f"expose_port {host_port}:{container_port}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid port values. Expected integers.",
                            }
                        else:
                            result = playground_manager.expose_port(playground_name, host_port, container_port)
                            if result.get("ok"):
                                playground_info["web_port"] = host_port
                                playground_info.setdefault("exposed_ports", [])
                                if host_port not in playground_info["exposed_ports"]:
                                    playground_info["exposed_ports"].append(host_port)
                                events.append(f"Port {host_port} exposed to playground {playground_name}.")
                                entry = {
                                    "cmd": f"expose_port {host_port}:{container_port}",
                                    "exit_code": 0,
                                    "stdout": json.dumps(result),
                                    "stderr": "",
                                }
                            else:
                                entry = {
                                    "cmd": f"expose_port {host_port}:{container_port}",
                                    "exit_code": 1,
                                    "stdout": "",
                                    "stderr": result.get("error", "Failed to expose port."),
                                }
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "playground.write_file" and use_playground:
                        path = request.get("path")
                        content = request.get("content")
                        if not isinstance(path, str) or not isinstance(content, str):
                            entry = {
                                "cmd": f"write_file {path}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid write_file payload. Expected path/content strings.",
                            }
                        else:
                            entry = playground_manager.write_file(playground_name, path, content)
                            entry["cmd"] = f"write_file {path}"
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_tool_context(entry))
                        playground_log.append(entry)
                    elif tool_name == "cluster.exec" and use_cluster:
                        container = request.get("container")
                        cmd = request.get("cmd")
                        timeout_s = int(request.get("timeout_s", 60))
                        if not isinstance(container, str) or not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
                            entry = {
                                "cmd": f"{container} {cmd}",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.exec payload. Expected container + cmd list.",
                            }
                        else:
                            entry = cluster_manager.exec_in(container, cmd, timeout_s=timeout_s)
                            entry["cmd"] = f"{container} :: {' '.join(cmd)}"
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    elif tool_name == "cluster.logs" and use_cluster:
                        container = request.get("container")
                        if not isinstance(container, str):
                            entry = {
                                "cmd": f"{container} logs",
                                "exit_code": 125,
                                "stdout": "",
                                "stderr": "Invalid cluster.logs payload. Expected container string.",
                            }
                        else:
                            tail_value = request.get("tail", 200)
                            try:
                                tail = int(tail_value)
                            except (TypeError, ValueError):
                                tail = 200
                            entry = cluster_manager.container_logs(container, tail=tail)
                            entry["cmd"] = f"{container} :: logs (tail={tail})"
                        entry["agent"] = "fixer"
                        tool_context_chunks.append(_format_cluster_context(entry))
                        cluster_log.append(entry)
                    else:
                        continue
                    tool_entries.append(entry)
                    fix_actions.append(
                        {
                            "iteration": iteration,
                            "action": entry.get("cmd", ""),
                            "exit_code": entry.get("exit_code"),
                        }
                    )
                if tool_entries:
                    fix_iterations.append(
                        {
                            "iteration": iteration,
                            "validation": validation,
                            "fixer_output": fixer_result.output,
                            "tool_requests": tool_entries,
                        }
                    )
                if fix_actions:
                    cluster_info["fix_actions"].extend(fix_actions)
                total_ms = (time.perf_counter() - start_time) * 1000
                yield _serialize(
                    stages,
                    goal,
                    scenario,
                    final=final_text,
                    total_ms=total_ms,
                    playground=playground_info,
                    bare=bare_info,
                    cluster=cluster_info,
                    dml=dml_info,
                    events=events,
                )
            trace["cluster"]["fix_iterations"] = fix_iterations
            if last_validation and not last_validation.get("ok") and not failed:
                failed = True
                cluster_info["error"] = f"Validation failed after {iteration} iterations."
        else:
            validation = cluster_manager.validate_cluster(run_id)
            cluster_info["validation"] = validation
            cluster_info["validation_history"].append({"iteration": 1, "validation": validation})
            cluster_log.append(
                {
                    "cmd": "cluster.validate (auto)",
                    "exit_code": 0 if validation.get("ok") else 1,
                    "stdout": json.dumps(validation, indent=2),
                    "stderr": "" if validation.get("ok") else validation.get("error", ""),
                }
            )
            trace["cluster"]["validation"] = validation
            trace["cluster"]["validation_history"] = cluster_info["validation_history"]
            if cluster_log:
                trace["cluster"]["log"] = cluster_log
    if use_dml and dml_enabled:
        run_report = {
            "scenario_key": scenario_key,
            "goal": goal,
            "run_id": run_id,
            "trace": trace,
            "final": final_text,
            "success": not failed,
            "artifacts": [{"type": "playground_command", **entry} for entry in playground_log]
            + [{"type": "bare_command", **entry} for entry in bare_log]
            + [{"type": "cluster_command", **entry} for entry in cluster_log],
            "meta": {
                "scenario": scenario,
                "fast": fast,
                "cookbook_sources": cookbook_info["sources"],
                "cluster_topology": trace.get("cluster"),
            },
        }
        try:
            dml_ingest_calls += 1
            if dml_ingest_calls > 1:
                logger.error("dml_ingest_calls_per_run exceeded: %d", dml_ingest_calls)
            result = dml_http_client.ingest_run_report(run_report)
            ingest_info.update(
                {
                    "ok": result.ok,
                    "ingested_id": result.ingested_id,
                    "summary_id": result.summary_id,
                    "summary_latency_ms": result.summary_latency_ms,
                    "error": None,
                }
            )
        except dml_http_client.DMLServiceError as exc:
            ingest_info.update({"error": str(exc)})
        dml_info["counters"]["dml_get_calls_per_run"] = dml_get_calls
        dml_info["counters"]["dml_ingest_calls_per_run"] = dml_ingest_calls
    if use_playground:
        playground_info["ready_for_removal"] = True
        if auto_remove_playground:
            removal = playground_manager.remove_playground(playground_name)
            playground_info["remove_result"] = removal
            if removal.get("ok"):
                playground_info["status"] = "removed"
            else:
                playground_info["status"] = "error"
                playground_info["error"] = removal.get("error")
    if use_cluster:
        cluster_info["ready_for_removal"] = True
    yield _serialize(
        stages,
        goal,
        scenario,
        final=final_text,
        total_ms=total_ms,
        playground=playground_info,
        bare=bare_info,
        cluster=cluster_info,
        dml=dml_info,
        events=events,
    )


def run_demo(
    goal: str,
    fast: bool = False,
    scenario: Optional[str] = None,
    use_dml: bool = False,
    dml_top_k: int = 6,
    use_playground: bool = False,
    playground_image: str = "nemotron-playground:latest",
    auto_remove_playground: bool = False,
    use_bare_metal: bool = False,
    use_cluster: bool = False,
    cluster_image: str = "nemotron-playground:latest",
    cluster_size: int = 3,
    cluster_run_id: Optional[str] = None,
) -> Dict:
    last_state = {}
    for state in run_demo_stream(
        goal,
        fast=fast,
        scenario=scenario,
        use_dml=use_dml,
        dml_top_k=dml_top_k,
        use_playground=use_playground,
        playground_image=playground_image,
        auto_remove_playground=auto_remove_playground,
        use_bare_metal=use_bare_metal,
        use_cluster=use_cluster,
        cluster_image=cluster_image,
        cluster_size=cluster_size,
        cluster_run_id=cluster_run_id,
    ):
        last_state = deepcopy(state)
    return last_state
