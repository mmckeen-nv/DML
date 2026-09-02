#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import platform
import re
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class DeployError(RuntimeError):
    pass


ALLOWED_ENV_KEYS = {
    "DML_REPO_ROOT",
    "DML_SOURCE_COMMIT",
    "DAYSTROM_KV_SECRET_FILE",
    "DAYSTROM_CPU_KV_BYTES",
    "DAYSTROM_MAX_TTL_SECONDS",
    "DAYSTROM_MAX_RECORDS",
    "DAYSTROM_DEPLOY_STATE_DIR",
    "DAYSTROM_ALLOW_PUBLIC_BIND",
    "HF_CACHE_DIR",
    "VLLM_IMAGE",
    "VLLM_CONTAINER_NAME",
    "VLLM_BIND_ADDRESS",
    "VLLM_PORT",
    "VLLM_MODEL_PATH",
    "VLLM_SERVED_MODEL",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_MOE_BACKEND",
    "VLLM_QUANTIZATION",
    "VLLM_DTYPE",
    "VLLM_KV_CACHE_DTYPE",
    "VLLM_MAMBA_SSM_CACHE_DTYPE",
    "VLLM_TENSOR_PARALLEL_SIZE",
    "VLLM_PIPELINE_PARALLEL_SIZE",
    "VLLM_DATA_PARALLEL_SIZE",
    "VLLM_STARTUP_TIMEOUT_SECONDS",
    "VLLM_VERIFY_TIMEOUT_SECONDS",
}

DERIVED_ENV_KEYS = {
    "DAYSTROM_KV_TRANSFER_CONFIG",
}

REQUIRED_ENV_KEYS = {
    "DML_REPO_ROOT",
    "DML_SOURCE_COMMIT",
    "DAYSTROM_KV_SECRET_FILE",
    "HF_CACHE_DIR",
    "VLLM_SERVED_MODEL",
}

STATE_SCHEMA = 1
MAX_STATE_BACKUPS = 64
MAX_HTTP_BODY_BYTES = 1 << 20

NUMERIC_BOUNDS: dict[str, tuple[type[Any], float, float]] = {
    "DAYSTROM_CPU_KV_BYTES": (int, 1024.0 * 1024.0, float(2**63 - 1)),
    "DAYSTROM_MAX_TTL_SECONDS": (int, 1.0, 7.0 * 24.0 * 3600.0),
    "DAYSTROM_MAX_RECORDS": (int, 1.0, 10_000_000.0),
    "VLLM_PORT": (int, 1.0, 65535.0),
    "VLLM_GPU_MEMORY_UTILIZATION": (float, 0.01, 0.99),
    "VLLM_MAX_NUM_SEQS": (int, 1.0, 4096.0),
    "VLLM_MAX_MODEL_LEN": (int, 1.0, 10_000_000.0),
    "VLLM_TENSOR_PARALLEL_SIZE": (int, 1.0, 1024.0),
    "VLLM_PIPELINE_PARALLEL_SIZE": (int, 1.0, 1024.0),
    "VLLM_DATA_PARALLEL_SIZE": (int, 1.0, 1024.0),
    "VLLM_STARTUP_TIMEOUT_SECONDS": (int, 30.0, 3600.0),
    "VLLM_VERIFY_TIMEOUT_SECONDS": (int, 3.0, 180.0),
}

DEFAULTS = {
    "VLLM_IMAGE": "vllm/vllm-openai:v0.20.0",
    "VLLM_CONTAINER_NAME": "nemotron-warroom-vllm",
    "VLLM_BIND_ADDRESS": "127.0.0.1",
    "VLLM_PORT": "8000",
    "VLLM_MODEL_PATH": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "VLLM_SERVED_MODEL": "nvidia/nemotron-3-super",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.70",
    "VLLM_MAX_NUM_SEQS": "2",
    "VLLM_MAX_MODEL_LEN": "65536",
    "VLLM_MOE_BACKEND": "marlin",
    "VLLM_QUANTIZATION": "fp4",
    "VLLM_DTYPE": "auto",
    "VLLM_KV_CACHE_DTYPE": "fp8",
    "VLLM_MAMBA_SSM_CACHE_DTYPE": "float16",
    "VLLM_TENSOR_PARALLEL_SIZE": "1",
    "VLLM_PIPELINE_PARALLEL_SIZE": "1",
    "VLLM_DATA_PARALLEL_SIZE": "1",
    "DAYSTROM_CPU_KV_BYTES": "8589934592",
    "DAYSTROM_MAX_TTL_SECONDS": "900",
    "DAYSTROM_MAX_RECORDS": "128",
    "VLLM_STARTUP_TIMEOUT_SECONDS": "900",
    "VLLM_VERIFY_TIMEOUT_SECONDS": "20",
    "DAYSTROM_ALLOW_PUBLIC_BIND": "",
}

ENV_LINE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_CONTAINER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

NONEMPTY_NO_CONTROL_SCALAR_KEYS = {
    "VLLM_IMAGE",
    "VLLM_MODEL_PATH",
    "VLLM_SERVED_MODEL",
    "VLLM_DTYPE",
    "VLLM_KV_CACHE_DTYPE",
    "VLLM_MAMBA_SSM_CACHE_DTYPE",
    "VLLM_MOE_BACKEND",
    "VLLM_QUANTIZATION",
}


def _contains_control_characters(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def parse_env_file(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DeployError(f"missing environment file: {path}") from exc
    if b"\x00" in raw:
        raise DeployError(f"{path}: contains NUL bytes")
    if b"\r" in raw:
        raise DeployError(f"{path}: carriage returns are not allowed")
    if b"\t" in raw:
        raise DeployError(f"{path}: tabs are not allowed")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeployError(f"{path}: must be UTF-8 text") from exc

    out: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_LINE_RE.match(line)
        if not match:
            raise DeployError(f"{path}:{line_no}: expected strict KEY=VALUE format")
        key = match.group(1)
        value = match.group(2)
        if key in out:
            raise DeployError(f"{path}:{line_no}: duplicate key {key}")
        if "$" in value or "`" in value:
            raise DeployError(f"{path}:{line_no}: unsupported shell interpolation or command substitution in {key}")
        if value.endswith("\\"):
            raise DeployError(f"{path}:{line_no}: line-continuation escapes are not allowed in {key}")
        out[key] = value

    for key in out:
        if key.startswith(("DML_", "DAYSTROM_", "VLLM_", "HF_")) and key not in ALLOWED_ENV_KEYS:
            raise DeployError(f"{path}: unknown deployment key {key}")
    return out


def merged_env(parsed: dict[str, str]) -> dict[str, str]:
    combined = dict(DEFAULTS)
    combined.update(parsed)
    combined["DAYSTROM_KV_TRANSFER_CONFIG"] = build_kv_transfer_config(combined)
    return combined


def _validate_numeric(name: str, value: str) -> None:
    kind, lower, upper = NUMERIC_BOUNDS[name]
    if value == "":
        raise DeployError(f"{name} must not be empty")
    try:
        parsed: float
        if kind is int:
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError("expected integer")
            parsed = float(int(value))
        else:
            parsed = float(value)
    except ValueError as exc:
        raise DeployError(f"{name} must be numeric ({kind.__name__})") from exc
    if not math.isfinite(parsed) or parsed < lower or parsed > upper:
        raise DeployError(f"{name} out of range [{lower:g}, {upper:g}]")


def _run_git(args: list[str], repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DeployError("git is required for doctor/preflight validation") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "git command failed"
        raise DeployError(stderr)
    return result.stdout.strip()


def _is_within(parent: Path, candidate: Path) -> bool:
    parent_resolved = parent.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(parent_resolved)
        return True
    except ValueError:
        return False


def _validate_filesystem_ready(config: dict[str, str], current_repo_root: Path) -> None:
    repo_root = Path(config["DML_REPO_ROOT"])
    if not repo_root.exists():
        raise DeployError("DML_REPO_ROOT does not exist")
    if not (repo_root / "dml_core" / "daystrom_dml").is_dir():
        raise DeployError("DML dml_core source is missing")
    _run_git(["rev-parse", "--git-dir"], repo_root)
    head = _run_git(["rev-parse", "HEAD"], repo_root)
    if head != config["DML_SOURCE_COMMIT"]:
        raise DeployError(f"source commit mismatch: expected {config['DML_SOURCE_COMMIT']}, got {head}")
    dirty = _run_git(["status", "--porcelain", "--untracked-files=all", "--", "dml_core"], repo_root)
    if dirty:
        raise DeployError("dml_core contains tracked or untracked modifications")

    secret_file = Path(config["DAYSTROM_KV_SECRET_FILE"])
    if not secret_file.is_file():
        raise DeployError("control-key file is missing")
    secret_mode = stat.S_IMODE(secret_file.stat().st_mode)
    if secret_mode & 0o077:
        raise DeployError("control-key permissions must deny group/world access")
    if secret_file.stat().st_size < 32:
        raise DeployError("control-key file is too short; require at least 32 bytes")
    if _is_within(current_repo_root, secret_file):
        raise DeployError("control-key file must live outside this repository")

    hf_cache = Path(config["HF_CACHE_DIR"])
    if not hf_cache.is_dir():
        raise DeployError("HF_CACHE_DIR is not a directory")


def validate_env(config: dict[str, str], mode: str, current_repo_root: Path) -> None:
    for key in REQUIRED_ENV_KEYS:
        if not config.get(key):
            raise DeployError(f"set {key} in .env")

    for key in NUMERIC_BOUNDS:
        _validate_numeric(key, config[key])

    for key in NONEMPTY_NO_CONTROL_SCALAR_KEYS:
        value = config.get(key, "")
        if not value:
            raise DeployError(f"set {key} in .env")
        if _contains_control_characters(value):
            raise DeployError(f"{key} contains control characters")

    source_commit = config["DML_SOURCE_COMMIT"]
    if source_commit == "REPLACE_WITH_40_HEX_REVIEWED_COMMIT":
        raise DeployError("DML_SOURCE_COMMIT still uses placeholder value")
    if not FULL_COMMIT_RE.fullmatch(source_commit):
        raise DeployError("DML_SOURCE_COMMIT must be an exact 40-hex commit")

    container_name = config.get("VLLM_CONTAINER_NAME", "")
    if not container_name:
        raise DeployError("set VLLM_CONTAINER_NAME in .env")
    if _contains_control_characters(container_name):
        raise DeployError("VLLM_CONTAINER_NAME contains control characters")
    if not SAFE_CONTAINER_NAME_RE.fullmatch(container_name):
        raise DeployError("VLLM_CONTAINER_NAME is invalid; use [a-z0-9][a-z0-9_.-]*")

    bind = config["VLLM_BIND_ADDRESS"].strip().lower()
    loopback = {"127.0.0.1", "localhost", "::1"}
    if bind not in loopback and config.get("DAYSTROM_ALLOW_PUBLIC_BIND", "") != "yes":
        raise DeployError(
            "public bind requested; set DAYSTROM_ALLOW_PUBLIC_BIND=yes only after confirming network controls"
        )

    for key in ("DML_REPO_ROOT", "DAYSTROM_KV_SECRET_FILE", "HF_CACHE_DIR", "DAYSTROM_DEPLOY_STATE_DIR"):
        value = config.get(key, "")
        if not value:
            raise DeployError(f"set {key} in .env")
        if not Path(value).is_absolute():
            raise DeployError(f"{key} must be an absolute path")

    if mode == "basic":
        return

    _validate_filesystem_ready(config, current_repo_root)
    if mode == "doctor":
        return
    if platform.system() != "Linux":
        raise DeployError("host OS must be Linux for this deployment")
    if sys.version_info < (3, 10):
        raise DeployError("python3 >= 3.10 is required")


def build_kv_transfer_config(config: dict[str, str]) -> str:
    _validate_numeric("DAYSTROM_CPU_KV_BYTES", config["DAYSTROM_CPU_KV_BYTES"])
    _validate_numeric("DAYSTROM_MAX_TTL_SECONDS", config["DAYSTROM_MAX_TTL_SECONDS"])
    _validate_numeric("DAYSTROM_MAX_RECORDS", config["DAYSTROM_MAX_RECORDS"])
    payload = {
        "kv_connector": "DaystromCooperativeKVConnector",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(config["DAYSTROM_CPU_KV_BYTES"]),
            "daystrom_max_records": int(config["DAYSTROM_MAX_RECORDS"]),
            "daystrom_max_ttl_seconds": int(config["DAYSTROM_MAX_TTL_SECONDS"]),
            "daystrom_secret_path": "/run/secrets/daystrom-kv-control.key",
        },
        "kv_connector_module_path": "daystrom_dml.context.vllm_bridge.connector",
        "kv_role": "kv_both",
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _decode_utf8(data: bytes, where: str) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DeployError(f"{where} response is not valid UTF-8") from exc


def _read_http_body(response: Any, where: str) -> bytes:
    body = response.read(MAX_HTTP_BODY_BYTES + 1)
    if len(body) > MAX_HTTP_BODY_BYTES:
        raise DeployError(f"{where} response body exceeds 1 MiB")
    return body


def _json_request(url: str, timeout_seconds: int, payload: bytes | None = None) -> Any:
    request = urllib.request.Request(url=url, data=payload)
    request.add_header("Accept", "application/json")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status < 200 or response.status >= 300:
                raise DeployError(f"request failed at {url}: HTTP {response.status}")
            body = _read_http_body(response, url)
    except urllib.error.HTTPError as exc:
        raise DeployError(f"request failed at {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DeployError(f"request failed at {url}: {exc.reason}") from exc
    except OSError as exc:
        raise DeployError(f"request failed at {url}: {exc}") from exc

    text = _decode_utf8(body, url)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeployError(f"{url} returned malformed JSON") from exc


def parse_metrics_idle(text: str) -> dict[str, float]:
    wanted = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    totals: dict[str, float] = {name: 0.0 for name in wanted}
    seen: dict[str, bool] = {name: False for name in wanted}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        metric_name = parts[0].split("{", 1)[0]
        if metric_name not in totals:
            continue
        if len(parts) not in {2, 3}:
            raise DeployError(f"invalid metric sample for {metric_name}: {line}")
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise DeployError(f"invalid metric value for {metric_name}: {parts[1]}") from exc
        if not math.isfinite(value):
            raise DeployError(f"invalid metric value for {metric_name}: {parts[1]}")
        if len(parts) == 3:
            try:
                ts_value = float(parts[2])
            except ValueError as exc:
                raise DeployError(f"invalid metric timestamp for {metric_name}: {parts[2]}") from exc
            if not math.isfinite(ts_value):
                raise DeployError(f"invalid metric timestamp for {metric_name}: {parts[2]}")
        totals[metric_name] += value
        seen[metric_name] = True

    missing = [name for name in wanted if not seen[name]]
    if missing:
        raise DeployError("metrics missing required idle counters: " + ", ".join(missing))
    return totals


def assert_idle(port: int, timeout_seconds: int) -> dict[str, float]:
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            if response.status < 200 or response.status >= 300:
                raise DeployError(f"metrics query failed at {url}: HTTP {response.status}")
            payload = _decode_utf8(_read_http_body(response, url), url)
    except urllib.error.HTTPError as exc:
        raise DeployError(f"metrics query failed at {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DeployError(f"metrics query failed at {url}: {exc.reason}") from exc
    except OSError as exc:
        raise DeployError(f"metrics query failed at {url}: {exc}") from exc

    totals = parse_metrics_idle(payload)
    for metric_name, total in totals.items():
        if total != 0.0:
            raise DeployError(f"refusing replacement: {metric_name}={total}")
    return totals


def verify_runtime(port: int, served_model: str, timeout_seconds: int) -> None:
    health_url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise DeployError(f"health check failed with status {response.status}")
            _read_http_body(response, health_url)
    except urllib.error.HTTPError as exc:
        raise DeployError(f"health check failed at {health_url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DeployError(f"health check failed at {health_url}: {exc.reason}") from exc
    except OSError as exc:
        raise DeployError(f"health check failed at {health_url}: {exc}") from exc

    models = _json_request(f"http://127.0.0.1:{port}/v1/models", timeout_seconds)
    if not isinstance(models, dict) or "data" not in models or not isinstance(models["data"], list):
        raise DeployError("/v1/models payload is malformed")
    model_ids = sorted({str(item.get("id", "")) for item in models["data"] if isinstance(item, dict)})
    if model_ids != [served_model]:
        raise DeployError(f"served model mismatch: expected only {served_model}, got {model_ids}")

    payload = json.dumps(
        {
            "model": served_model,
            "prompt": "daystrom smoke test",
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode("utf-8")
    completion = _json_request(f"http://127.0.0.1:{port}/v1/completions", timeout_seconds, payload=payload)
    if not isinstance(completion, dict) or not isinstance(completion.get("choices"), list) or not completion["choices"]:
        raise DeployError("completion smoke check failed: no choices returned")


def _empty_state_document() -> dict[str, Any]:
    return {"schema": STATE_SCHEMA, "backups": []}


def _write_atomic_json(path: Path, payload: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _read_state_document(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return _empty_state_document()
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployError(f"state file is unreadable: {state_file}") from exc
    if not isinstance(data, dict):
        raise DeployError("state file payload must be an object")
    schema = data.get("schema", STATE_SCHEMA)
    if schema != STATE_SCHEMA:
        raise DeployError(f"unsupported state schema: {schema}")
    backups = data.get("backups")
    if backups is None:
        legacy_name = data.get("backup_name")
        if legacy_name is None:
            backups = []
        elif isinstance(legacy_name, str) and legacy_name:
            backups = [{"name": legacy_name, "recorded_at": int(data.get("recorded_at", int(time.time())))}]
        else:
            raise DeployError("state file backup_name is invalid")
    if not isinstance(backups, list):
        raise DeployError("state file backups must be a list")
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in backups:
        if not isinstance(item, dict):
            raise DeployError("state file backup entries must be objects")
        name = item.get("name")
        recorded_at = item.get("recorded_at")
        if not isinstance(name, str) or not name:
            raise DeployError("state file backup name is invalid")
        if not SAFE_CONTAINER_NAME_RE.fullmatch(name):
            raise DeployError("state file backup name is invalid")
        if name in seen_names:
            raise DeployError("state file contains duplicate backup names")
        seen_names.add(name)
        if not isinstance(recorded_at, int) or recorded_at <= 0:
            raise DeployError("state file backup timestamp is invalid")
        normalized.append({"name": name, "recorded_at": recorded_at})
    if len(normalized) > MAX_STATE_BACKUPS:
        raise DeployError(f"state file contains too many backup entries (max {MAX_STATE_BACKUPS})")
    return {"schema": STATE_SCHEMA, "backups": normalized}


def state_peek_backup(state_file: Path) -> str | None:
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    return backups[0]["name"] if backups else None


def state_push_backup(state_file: Path, backup_name: str) -> None:
    if not SAFE_CONTAINER_NAME_RE.fullmatch(backup_name):
        raise DeployError("backup name is invalid")
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    backups = [item for item in backups if item["name"] != backup_name]
    if len(backups) >= MAX_STATE_BACKUPS:
        raise DeployError(
            f"backup state is full (max {MAX_STATE_BACKUPS}); restore or remove obsolete backups before deploying"
        )
    backups.insert(0, {"name": backup_name, "recorded_at": int(time.time())})
    _write_atomic_json(state_file, {"schema": STATE_SCHEMA, "backups": backups})


def state_require_push_capacity(state_file: Path) -> None:
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    if len(backups) >= MAX_STATE_BACKUPS:
        raise DeployError(
            f"backup state is full (max {MAX_STATE_BACKUPS}); restore or remove obsolete backups before deploying"
        )


def state_pop_backup(state_file: Path, backup_name: str) -> bool:
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    next_backups = [item for item in backups if item["name"] != backup_name]
    if len(next_backups) == len(backups):
        return False
    _write_atomic_json(state_file, {"schema": STATE_SCHEMA, "backups": next_backups})
    return True


def state_prune_backups(state_file: Path, keep: int) -> list[str]:
    if keep < 0:
        raise DeployError("keep must be >= 0")
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    removed = [item["name"] for item in backups[keep:]]
    _write_atomic_json(state_file, {"schema": STATE_SCHEMA, "backups": backups[:keep]})
    return removed


def state_list_old_backups(state_file: Path, keep: int) -> list[str]:
    if keep < 0:
        raise DeployError("keep must be >= 0")
    state = _read_state_document(state_file)
    backups: list[dict[str, Any]] = state["backups"]
    return [item["name"] for item in backups[keep:]]


def clear_state(state_file: Path) -> None:
    state_file.unlink(missing_ok=True)


def generate_key(path: Path, byte_count: int, force: bool, repo_root: Path) -> None:
    if byte_count < 32:
        raise DeployError("key byte count must be >= 32")
    if not path.is_absolute():
        raise DeployError("key output path must be absolute")
    if _is_within(repo_root, path):
        raise DeployError("key output path must be outside this repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise DeployError(f"refusing to overwrite existing key: {path}")

    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        token = secrets.token_hex(byte_count) + "\n"
        with os.fdopen(fd, "wb") as handle:
            handle.write(token.encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        if path.exists() and not force:
            raise DeployError(f"refusing to overwrite existing key: {path}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def check_port_available(bind_address: str, port: int) -> bool:
    families = [socket.AF_INET6] if ":" in bind_address else [socket.AF_INET]
    if bind_address in {"localhost", ""}:
        families = [socket.AF_INET, socket.AF_INET6]
    for family in families:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_address, port))
            except OSError as exc:
                if exc.errno in {errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT}:
                    continue
                return False
    return True


def container_owns_port(container_name: str, bind_address: str, port: int) -> bool:
    inspect = subprocess.run(
        ["docker", "inspect", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        return False
    try:
        payload = json.loads(inspect.stdout)
    except json.JSONDecodeError as exc:
        raise DeployError("docker inspect returned malformed JSON") from exc
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise DeployError("docker inspect payload is malformed")
    state = payload[0].get("State")
    if not isinstance(state, dict) or state.get("Running") is not True:
        return False
    ports = payload[0].get("NetworkSettings", {}).get("Ports")
    if not isinstance(ports, dict):
        return False
    mappings = ports.get("8000/tcp")
    if not isinstance(mappings, list):
        return False
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        host_ip = str(mapping.get("HostIp", ""))
        host_port = str(mapping.get("HostPort", ""))
        if host_port != str(port):
            continue
        if host_ip == bind_address:
            return True
        if bind_address == "localhost" and host_ip in {"127.0.0.1", "::1"}:
            return True
    return False


def _cmd_export_env(args: argparse.Namespace) -> int:
    env = merged_env(parse_env_file(Path(args.env_file)))
    for key in sorted(ALLOWED_ENV_KEYS | DERIVED_ENV_KEYS):
        value = env.get(key)
        if value is None:
            continue
        if "\t" in key or "\n" in key or "\r" in key:
            raise DeployError(f"invalid key for tab-delimited export: {key}")
        if "\t" in value or "\n" in value or "\r" in value:
            raise DeployError(f"{key} cannot contain tabs or newlines")
        print(f"{key}\t{value}")
    return 0


def _cmd_validate_env(args: argparse.Namespace) -> int:
    config = merged_env(parse_env_file(Path(args.env_file)))
    validate_env(config, args.mode, Path(args.current_repo_root))
    return 0


def _cmd_assert_idle(args: argparse.Namespace) -> int:
    assert_idle(args.port, args.timeout_seconds)
    return 0


def _cmd_verify_runtime(args: argparse.Namespace) -> int:
    verify_runtime(args.port, args.served_model, args.timeout_seconds)
    return 0


def _cmd_state_push(args: argparse.Namespace) -> int:
    state_push_backup(Path(args.state_file), args.backup_name)
    return 0


def _cmd_state_require_capacity(args: argparse.Namespace) -> int:
    state_require_push_capacity(Path(args.state_file))
    return 0


def _cmd_state_peek(args: argparse.Namespace) -> int:
    backup = state_peek_backup(Path(args.state_file))
    if backup:
        print(backup)
    return 0


def _cmd_state_pop(args: argparse.Namespace) -> int:
    removed = state_pop_backup(Path(args.state_file), args.backup_name)
    return 0 if removed else 2


def _cmd_state_prune(args: argparse.Namespace) -> int:
    removed = state_prune_backups(Path(args.state_file), args.keep)
    for item in removed:
        print(item)
    return 0


def _cmd_state_list_old(args: argparse.Namespace) -> int:
    removed = state_list_old_backups(Path(args.state_file), args.keep)
    for item in removed:
        print(item)
    return 0


def _cmd_clear_state(args: argparse.Namespace) -> int:
    clear_state(Path(args.state_file))
    return 0


def _cmd_generate_key(args: argparse.Namespace) -> int:
    generate_key(Path(args.output), args.bytes, args.force, Path(args.repo_root))
    return 0


def _cmd_check_port(args: argparse.Namespace) -> int:
    if check_port_available(args.bind_address, args.port):
        return 0
    raise DeployError(f"port already in use on {args.bind_address}:{args.port}")


def _cmd_check_port_owner(args: argparse.Namespace) -> int:
    if container_owns_port(args.container_name, args.bind_address, args.port):
        return 0
    return 2


def _cmd_kv_transfer_config(args: argparse.Namespace) -> int:
    config = merged_env(parse_env_file(Path(args.env_file)))
    print(build_kv_transfer_config(config))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daystrom deployment helper")
    sub = parser.add_subparsers(dest="command", required=True)

    export_env = sub.add_parser("export-env")
    export_env.add_argument("--env-file", required=True)
    export_env.set_defaults(func=_cmd_export_env)

    validate_env_cmd = sub.add_parser("validate-env")
    validate_env_cmd.add_argument("--env-file", required=True)
    validate_env_cmd.add_argument("--mode", choices=["basic", "doctor", "preflight"], required=True)
    validate_env_cmd.add_argument("--current-repo-root", required=True)
    validate_env_cmd.set_defaults(func=_cmd_validate_env)

    assert_idle_cmd = sub.add_parser("assert-idle")
    assert_idle_cmd.add_argument("--port", type=int, required=True)
    assert_idle_cmd.add_argument("--timeout-seconds", type=int, default=5)
    assert_idle_cmd.set_defaults(func=_cmd_assert_idle)

    verify_runtime_cmd = sub.add_parser("verify-runtime")
    verify_runtime_cmd.add_argument("--port", type=int, required=True)
    verify_runtime_cmd.add_argument("--served-model", required=True)
    verify_runtime_cmd.add_argument("--timeout-seconds", type=int, default=20)
    verify_runtime_cmd.set_defaults(func=_cmd_verify_runtime)

    state_push_cmd = sub.add_parser("state-push")
    state_push_cmd.add_argument("--state-file", required=True)
    state_push_cmd.add_argument("--backup-name", required=True)
    state_push_cmd.set_defaults(func=_cmd_state_push)

    state_capacity_cmd = sub.add_parser("state-require-capacity")
    state_capacity_cmd.add_argument("--state-file", required=True)
    state_capacity_cmd.set_defaults(func=_cmd_state_require_capacity)

    state_peek_cmd = sub.add_parser("state-peek")
    state_peek_cmd.add_argument("--state-file", required=True)
    state_peek_cmd.set_defaults(func=_cmd_state_peek)

    state_pop_cmd = sub.add_parser("state-pop")
    state_pop_cmd.add_argument("--state-file", required=True)
    state_pop_cmd.add_argument("--backup-name", required=True)
    state_pop_cmd.set_defaults(func=_cmd_state_pop)

    state_prune_cmd = sub.add_parser("state-prune")
    state_prune_cmd.add_argument("--state-file", required=True)
    state_prune_cmd.add_argument("--keep", type=int, required=True)
    state_prune_cmd.set_defaults(func=_cmd_state_prune)

    state_list_old_cmd = sub.add_parser("state-list-old")
    state_list_old_cmd.add_argument("--state-file", required=True)
    state_list_old_cmd.add_argument("--keep", type=int, required=True)
    state_list_old_cmd.set_defaults(func=_cmd_state_list_old)

    clear_state_cmd = sub.add_parser("clear-state")
    clear_state_cmd.add_argument("--state-file", required=True)
    clear_state_cmd.set_defaults(func=_cmd_clear_state)

    generate_key_cmd = sub.add_parser("generate-key")
    generate_key_cmd.add_argument("--output", required=True)
    generate_key_cmd.add_argument("--bytes", type=int, default=32)
    generate_key_cmd.add_argument("--force", action="store_true")
    generate_key_cmd.add_argument("--repo-root", required=True)
    generate_key_cmd.set_defaults(func=_cmd_generate_key)

    check_port_cmd = sub.add_parser("check-port")
    check_port_cmd.add_argument("--bind-address", required=True)
    check_port_cmd.add_argument("--port", type=int, required=True)
    check_port_cmd.set_defaults(func=_cmd_check_port)

    check_port_owner_cmd = sub.add_parser("check-port-owner")
    check_port_owner_cmd.add_argument("--container-name", required=True)
    check_port_owner_cmd.add_argument("--bind-address", required=True)
    check_port_owner_cmd.add_argument("--port", type=int, required=True)
    check_port_owner_cmd.set_defaults(func=_cmd_check_port_owner)

    kv_transfer_config_cmd = sub.add_parser("kv-transfer-config")
    kv_transfer_config_cmd.add_argument("--env-file", required=True)
    kv_transfer_config_cmd.set_defaults(func=_cmd_kv_transfer_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except DeployError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
