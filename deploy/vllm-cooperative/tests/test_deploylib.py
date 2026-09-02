from __future__ import annotations

import importlib.util
import json
import subprocess
import urllib.error
from pathlib import Path
from typing import Any, Literal

import pytest


def _load_deploylib():
    helper_path = Path(__file__).resolve().parents[1] / "deploylib.py"
    spec = importlib.util.spec_from_file_location("deploylib", helper_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


deploylib = _load_deploylib()


def _write_env(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _init_repo(tmp_path: Path) -> tuple[Path, str, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "dml_core" / "daystrom_dml").mkdir(parents=True)
    tracked = repo / "dml_core" / "daystrom_dml" / "tracked.txt"
    tracked.write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    secret = tmp_path / "outside.key"
    secret.write_text("a" * 64, encoding="utf-8")
    secret.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    return repo, commit, secret, hf_cache


def _base_config(repo: Path, commit: str, secret: Path, hf_cache: Path, state_dir: Path) -> dict[str, str]:
    return deploylib.merged_env(
        {
            "DML_REPO_ROOT": str(repo),
            "DML_SOURCE_COMMIT": commit,
            "DAYSTROM_KV_SECRET_FILE": str(secret),
            "HF_CACHE_DIR": str(hf_cache),
            "DAYSTROM_DEPLOY_STATE_DIR": str(state_dir),
            "VLLM_SERVED_MODEL": "model",
        }
    )


def test_parse_env_rejects_duplicate_key(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", ["DML_REPO_ROOT=/x", "DML_REPO_ROOT=/y"])
    with pytest.raises(deploylib.DeployError, match="duplicate key"):
        deploylib.parse_env_file(env_path)


def test_parse_env_rejects_interpolation(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", ["DML_REPO_ROOT=${HOME}/repo"])
    with pytest.raises(deploylib.DeployError, match="unsupported shell interpolation"):
        deploylib.parse_env_file(env_path)


def test_parse_env_rejects_carriage_returns_in_raw_bytes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"DML_REPO_ROOT=/x\r\n")
    with pytest.raises(deploylib.DeployError, match="carriage returns"):
        deploylib.parse_env_file(env_path)


def test_parse_env_rejects_tabs_in_raw_bytes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"DML_REPO_ROOT=\t/x\n")
    with pytest.raises(deploylib.DeployError, match="tabs are not allowed"):
        deploylib.parse_env_file(env_path)


def test_parse_env_rejects_unknown_prefixed_key(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", ["VLLM_NOT_A_REAL_KEY=1"])
    with pytest.raises(deploylib.DeployError, match="unknown deployment key"):
        deploylib.parse_env_file(env_path)


def test_parse_env_preserves_spaces_in_values(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", ["HF_CACHE_DIR=/path/with spaces/cache"])
    parsed = deploylib.parse_env_file(env_path)
    assert parsed["HF_CACHE_DIR"] == "/path/with spaces/cache"


def test_export_env_is_tab_delimited_and_rejects_tabs_in_values(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", ["HF_CACHE_DIR=/path/with\tbad"])
    args = deploylib.build_parser().parse_args(["export-env", "--env-file", str(env_path)])
    with pytest.raises(deploylib.DeployError, match="tabs are not allowed"):
        args.func(args)


def test_validate_env_doctor_rejects_placeholder_commit(tmp_path: Path) -> None:
    config = deploylib.merged_env(
        {
            "DML_REPO_ROOT": "/abs/repo",
            "DML_SOURCE_COMMIT": "REPLACE_WITH_40_HEX_REVIEWED_COMMIT",
            "DAYSTROM_KV_SECRET_FILE": "/abs/key",
            "HF_CACHE_DIR": "/abs/hf",
            "VLLM_SERVED_MODEL": "model",
        }
    )
    with pytest.raises(deploylib.DeployError, match="placeholder"):
        deploylib.validate_env(config, mode="doctor", current_repo_root=tmp_path)


def test_validate_env_doctor_requires_public_bind_ack(tmp_path: Path) -> None:
    config = deploylib.merged_env(
        {
            "DML_REPO_ROOT": "/abs/repo",
            "DML_SOURCE_COMMIT": "a" * 40,
            "DAYSTROM_KV_SECRET_FILE": "/abs/key",
            "HF_CACHE_DIR": "/abs/hf",
            "VLLM_SERVED_MODEL": "model",
            "VLLM_BIND_ADDRESS": "0.0.0.0",
            "DAYSTROM_ALLOW_PUBLIC_BIND": "",
        }
    )
    with pytest.raises(deploylib.DeployError, match="public bind"):
        deploylib.validate_env(config, mode="doctor", current_repo_root=tmp_path)


def test_validate_env_doctor_checks_filesystem_readiness(tmp_path: Path) -> None:
    repo, commit, secret, hf_cache = _init_repo(tmp_path)
    config = _base_config(repo, commit, secret, hf_cache, tmp_path / "state")
    deploylib.validate_env(config, mode="doctor", current_repo_root=Path(__file__).resolve().parents[3])


def test_validate_env_basic_skips_filesystem_and_git_checks(tmp_path: Path) -> None:
    config = deploylib.merged_env(
        {
            "DML_REPO_ROOT": str(tmp_path / "missing-repo"),
            "DML_SOURCE_COMMIT": "a" * 40,
            "DAYSTROM_KV_SECRET_FILE": str(tmp_path / "missing.key"),
            "HF_CACHE_DIR": str(tmp_path / "missing-hf"),
            "DAYSTROM_DEPLOY_STATE_DIR": str(tmp_path / "state"),
            "VLLM_SERVED_MODEL": "model",
        }
    )
    deploylib.validate_env(config, mode="basic", current_repo_root=tmp_path)


def test_validate_env_rejects_repo_internal_key_outside_deploy(tmp_path: Path) -> None:
    repo, commit, _, hf_cache = _init_repo(tmp_path)
    secret_inside_repo = repo / "secrets" / "root-level.key"
    secret_inside_repo.parent.mkdir(parents=True)
    secret_inside_repo.write_text("a" * 64, encoding="utf-8")
    secret_inside_repo.chmod(0o600)
    config = _base_config(repo, commit, secret_inside_repo, hf_cache, tmp_path / "state")
    with pytest.raises(deploylib.DeployError, match="outside this repository"):
        deploylib.validate_env(config, mode="doctor", current_repo_root=repo)


def test_validate_env_rejects_invalid_container_name(tmp_path: Path) -> None:
    config = deploylib.merged_env(
        {
            "DML_REPO_ROOT": "/abs/repo",
            "DML_SOURCE_COMMIT": "a" * 40,
            "DAYSTROM_KV_SECRET_FILE": "/abs/key",
            "HF_CACHE_DIR": "/abs/hf",
            "DAYSTROM_DEPLOY_STATE_DIR": "/abs/state",
            "VLLM_SERVED_MODEL": "model",
            "VLLM_CONTAINER_NAME": "BadName",
        }
    )
    with pytest.raises(deploylib.DeployError, match="VLLM_CONTAINER_NAME is invalid"):
        deploylib.validate_env(config, mode="basic", current_repo_root=tmp_path)


def test_validate_env_doctor_rejects_nonabsolute_state_dir(tmp_path: Path) -> None:
    repo, commit, secret, hf_cache = _init_repo(tmp_path)
    config = _base_config(repo, commit, secret, hf_cache, tmp_path / "state")
    config["DAYSTROM_DEPLOY_STATE_DIR"] = "relative/path"
    with pytest.raises(deploylib.DeployError, match="absolute path"):
        deploylib.validate_env(config, mode="doctor", current_repo_root=Path(__file__).resolve().parents[3])


def test_validate_env_preflight_requires_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, commit, secret, hf_cache = _init_repo(tmp_path)
    config = _base_config(repo, commit, secret, hf_cache, tmp_path / "state")
    monkeypatch.setattr(deploylib.platform, "system", lambda: "Darwin")
    with pytest.raises(deploylib.DeployError, match="host OS must be Linux"):
        deploylib.validate_env(config, mode="preflight", current_repo_root=Path(__file__).resolve().parents[3])


def test_metrics_parser_handles_labeled_and_unlabeled_lines() -> None:
    metrics = "\n".join(
        [
            '# HELP vllm:num_requests_running running requests',
            'vllm:num_requests_running{engine="0"} 0',
            'vllm:num_requests_running 0',
            'vllm:num_requests_waiting{engine="0"} 0',
            'vllm:num_requests_waiting 0',
        ]
    )
    totals = deploylib.parse_metrics_idle(metrics)
    assert totals["vllm:num_requests_running"] == 0.0
    assert totals["vllm:num_requests_waiting"] == 0.0


def test_metrics_parser_accepts_timestamp_token() -> None:
    metrics = "\n".join(
        [
            "vllm:num_requests_running 0 1720000000",
            "vllm:num_requests_waiting{engine=\"0\"} 0 1720000001",
        ]
    )
    totals = deploylib.parse_metrics_idle(metrics)
    assert totals["vllm:num_requests_running"] == 0.0


def test_metrics_parser_rejects_bad_timestamp() -> None:
    metrics = "\n".join(
        [
            "vllm:num_requests_running 0 not_a_ts",
            "vllm:num_requests_waiting 0",
        ]
    )
    with pytest.raises(deploylib.DeployError, match="timestamp"):
        deploylib.parse_metrics_idle(metrics)


def test_metrics_parser_fails_closed_on_missing_counter() -> None:
    metrics = "vllm:num_requests_running 0\n"
    with pytest.raises(deploylib.DeployError, match="missing required idle counters"):
        deploylib.parse_metrics_idle(metrics)


def test_state_push_pop_and_prune(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    deploylib.state_push_backup(state_path, "container-backup-1")
    deploylib.state_push_backup(state_path, "container-backup-2")
    assert deploylib.state_peek_backup(state_path) == "container-backup-2"
    removed = deploylib.state_prune_backups(state_path, keep=1)
    assert removed == ["container-backup-1"]
    popped = deploylib.state_pop_backup(state_path, "container-backup-2")
    assert popped is True
    assert deploylib.state_peek_backup(state_path) is None


def test_state_rejects_duplicate_backup_entries(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "backups": [
                    {"name": "b1", "recorded_at": 1},
                    {"name": "b1", "recorded_at": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(deploylib.DeployError, match="duplicate backup names"):
        deploylib.state_peek_backup(state_path)


def test_state_rejects_excessive_backup_entries(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    backups = [{"name": f"b{i}", "recorded_at": i + 1} for i in range(deploylib.MAX_STATE_BACKUPS + 1)]
    state_path.write_text(json.dumps({"schema": 1, "backups": backups}), encoding="utf-8")
    with pytest.raises(deploylib.DeployError, match="too many backup entries"):
        deploylib.state_peek_backup(state_path)


def test_state_push_refuses_overflow_without_poisoning_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    for index in range(deploylib.MAX_STATE_BACKUPS):
        deploylib.state_push_backup(state_path, f"container-backup-{index}")

    with pytest.raises(deploylib.DeployError, match="backup state is full"):
        deploylib.state_push_backup(state_path, "container-overflow")

    state = deploylib._read_state_document(state_path)
    assert len(state["backups"]) == deploylib.MAX_STATE_BACKUPS
    assert all(item["name"] != "container-overflow" for item in state["backups"])


def test_generate_key_refuses_repo_internal_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    key_path = repo_root / "inside.key"
    with pytest.raises(deploylib.DeployError, match="outside this repository"):
        deploylib.generate_key(key_path, byte_count=32, force=False, repo_root=repo_root)


def test_generate_key_atomic_failure_preserves_existing_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = tmp_path / "key.txt"
    key_path.write_text("KEEP\n", encoding="utf-8")
    key_path.chmod(0o600)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def _boom(_: Path, __: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(deploylib.os, "replace", _boom)
    with pytest.raises(OSError):
        deploylib.generate_key(key_path, byte_count=32, force=True, repo_root=repo_root)
    assert key_path.read_text(encoding="utf-8") == "KEEP\n"


class _Response:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self, _limit: int | None = None) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> Literal[False]:
        return False


def test_verify_runtime_wraps_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_: Any, **__: Any) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(deploylib.urllib.request, "urlopen", _fail)
    with pytest.raises(deploylib.DeployError, match="health check failed"):
        deploylib.verify_runtime(port=8000, served_model="model", timeout_seconds=1)


def test_verify_runtime_rejects_malformed_models_json(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _Response(200, b"ok"),
        _Response(200, b"{not-json"),
    ]

    def _next(*_: Any, **__: Any) -> _Response:
        return responses.pop(0)

    monkeypatch.setattr(deploylib.urllib.request, "urlopen", _next)
    with pytest.raises(deploylib.DeployError, match="returned malformed JSON"):
        deploylib.verify_runtime(port=8000, served_model="model", timeout_seconds=1)


def test_verify_runtime_bounds_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = b"x" * (deploylib.MAX_HTTP_BODY_BYTES + 4)
    responses = [_Response(200, b"ok"), _Response(200, huge)]

    def _next(*_: Any, **__: Any) -> _Response:
        return responses.pop(0)

    monkeypatch.setattr(deploylib.urllib.request, "urlopen", _next)
    with pytest.raises(deploylib.DeployError, match="exceeds 1 MiB"):
        deploylib.verify_runtime(port=8000, served_model="model", timeout_seconds=1)


def test_kv_transfer_config_is_deterministic_and_parseable(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        [
            "DML_REPO_ROOT=/abs/repo",
            "DML_SOURCE_COMMIT=" + "a" * 40,
            "DAYSTROM_KV_SECRET_FILE=/abs/key",
            "HF_CACHE_DIR=/abs/hf",
            "DAYSTROM_DEPLOY_STATE_DIR=/abs/state",
            "VLLM_SERVED_MODEL=model",
            "DAYSTROM_CPU_KV_BYTES=8589934592",
            "DAYSTROM_MAX_TTL_SECONDS=900",
            "DAYSTROM_MAX_RECORDS=128",
        ],
    )
    cfg_1 = subprocess.run(
        ["python3", str(Path(__file__).resolve().parents[1] / "deploylib.py"), "kv-transfer-config", "--env-file", str(env_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cfg_2 = subprocess.run(
        ["python3", str(Path(__file__).resolve().parents[1] / "deploylib.py"), "kv-transfer-config", "--env-file", str(env_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert cfg_1 == cfg_2
    payload = json.loads(cfg_1)
    assert payload["kv_connector"] == "DaystromCooperativeKVConnector"


def test_check_port_owner_requires_exact_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    inspect_payload = [
        {
            "State": {"Running": True},
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]}}}
    ]

    def _run(*_: Any, **__: Any) -> Any:
        class _Proc:
            returncode = 0
            stdout = json.dumps(inspect_payload)
            stderr = ""

        return _Proc()

    monkeypatch.setattr(deploylib.subprocess, "run", _run)
    assert deploylib.container_owns_port("x", "127.0.0.1", 8000) is True
    assert deploylib.container_owns_port("x", "0.0.0.0", 8000) is False


def test_deploy_script_is_bash_parseable() -> None:
    script_path = Path(__file__).resolve().parents[1] / "deploy.sh"
    result = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
