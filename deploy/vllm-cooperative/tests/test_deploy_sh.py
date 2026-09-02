from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "deploy.sh"


def _helper_path() -> Path:
    return Path(__file__).resolve().parents[1] / "deploylib.py"


def _write_fake_git(bin_dir: Path, repo_root: Path, commit: str) -> None:
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$*\" == *\"rev-parse --show-toplevel\"* ]]; then\n"
        f"  printf '%s\\n' '{repo_root}'\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$*\" == *\"rev-parse HEAD\"* ]]; then\n"
        f"  printf '%s\\n' '{commit}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)


def _write_fake_docker(bin_dir: Path) -> None:
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import re\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "args = sys.argv[1:]\n"
        "state_path = Path(os.environ['FAKE_DOCKER_STATE_FILE'])\n"
        "log_path = Path(os.environ['FAKE_DOCKER_LOG'])\n"
        "\n"
        "def load_state():\n"
        "    if not state_path.exists():\n"
        "        return {'containers': {}, 'health_plan': [], 'fail_rm': []}\n"
        "    return json.loads(state_path.read_text(encoding='utf-8'))\n"
        "\n"
        "def save_state(state):\n"
        "    state_path.write_text(json.dumps(state, sort_keys=True), encoding='utf-8')\n"
        "\n"
        "def log(msg):\n"
        "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with log_path.open('a', encoding='utf-8') as handle:\n"
        "        handle.write(msg + '\\n')\n"
        "\n"
        "def parse_name_filter(argv):\n"
        "    for i, token in enumerate(argv):\n"
        "        if token == '--filter' and i + 1 < len(argv):\n"
        "            val = argv[i + 1]\n"
        "            if val.startswith('name=^/') and val.endswith('$'):\n"
        "                return val[len('name=^/'):-1]\n"
        "    return ''\n"
        "\n"
        "def inspect_template(state, template, name):\n"
        "    container = state['containers'].get(name)\n"
        "    if container is None:\n"
        "        return None\n"
        "    if template == '{{.State.Running}}':\n"
        "        return 'true' if container.get('running') else 'false'\n"
        "    if template == '{{.Config.Image}}':\n"
        "        return container.get('image', '')\n"
        "    if '.State.Health.Status' in template:\n"
        "        plan = state.get('health_plan', [])\n"
        "        if plan:\n"
        "            status = plan.pop(0)\n"
        "            container['health'] = status\n"
        "            state['containers'][name] = container\n"
        "            state['health_plan'] = plan\n"
        "            save_state(state)\n"
        "        status = container.get('health', 'healthy')\n"
        "        if status == 'exited' or not container.get('running'):\n"
        "            return 'exited'\n"
        "        if status in {'healthy', 'unhealthy', 'starting'}:\n"
        "            return status\n"
        "        return 'running'\n"
        "    return ''\n"
        "\n"
        "state = load_state()\n"
        "log(' '.join(args))\n"
        "\n"
        "if not args:\n"
        "    raise SystemExit(1)\n"
        "\n"
        "if args[0] == 'compose':\n"
        "    if len(args) >= 2 and args[1] == 'version':\n"
        "        print('Docker Compose version v2.fake')\n"
        "        raise SystemExit(0)\n"
        "    if 'config' in args:\n"
        "        kv = os.environ.get('DAYSTROM_KV_TRANSFER_CONFIG', '')\n"
        "        log('KVCFG=' + kv)\n"
        "        if not kv:\n"
        "            raise SystemExit(1)\n"
        "        raise SystemExit(0)\n"
        "    if 'up' in args:\n"
        "        if os.environ.get('FAKE_COMPOSE_UP_FAIL') == '1':\n"
        "            raise SystemExit(1)\n"
        "        name = os.environ.get('VLLM_CONTAINER_NAME', 'nemotron-warroom-vllm')\n"
        "        image = os.environ.get('VLLM_IMAGE', 'vllm/vllm-openai:v0.20.0')\n"
        "        state['containers'][name] = {'running': True, 'health': 'running', 'image': image}\n"
        "        save_state(state)\n"
        "        raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'info':\n"
        "    if '--format' in args:\n"
        "        print('{\"nvidia\":{}}')\n"
        "    else:\n"
        "        print('fake-docker-info')\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'run':\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'pull':\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'image' and len(args) >= 3 and args[1] == 'inspect':\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'ps':\n"
        "    wanted = parse_name_filter(args)\n"
        "    exists = wanted in state['containers']\n"
        "    if '--format' in args:\n"
        "        if exists:\n"
        "            container = state['containers'][wanted]\n"
        "            status = 'Up 1 second' if container.get('running') else 'Exited (0) 1 second ago'\n"
        "            print(f\"{wanted}|{container.get('image', '')}|{status}|127.0.0.1:8000->8000/tcp\")\n"
        "        raise SystemExit(0)\n"
        "    if exists:\n"
        "        print('fake-container-id')\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'inspect':\n"
        "    if '-f' in args:\n"
        "        idx = args.index('-f')\n"
        "        template = args[idx + 1]\n"
        "        name = args[-1]\n"
        "        out = inspect_template(state, template, name)\n"
        "        if out is None:\n"
        "            raise SystemExit(1)\n"
        "        print(out)\n"
        "        raise SystemExit(0)\n"
        "    name = args[-1]\n"
        "    if name not in state['containers']:\n"
        "        raise SystemExit(1)\n"
        "    container = state['containers'][name]\n"
        "    print(json.dumps([{'State': {'Running': bool(container.get('running'))}, 'NetworkSettings': {'Ports': {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': str(os.environ.get('VLLM_PORT', '8000'))}]}}}]))\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'rename' and len(args) == 3:\n"
        "    old, new = args[1], args[2]\n"
        "    if old not in state['containers'] or new in state['containers']:\n"
        "        raise SystemExit(1)\n"
        "    state['containers'][new] = state['containers'].pop(old)\n"
        "    save_state(state)\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'start' and len(args) >= 2:\n"
        "    name = args[-1]\n"
        "    if name not in state['containers']:\n"
        "        raise SystemExit(1)\n"
        "    container = state['containers'][name]\n"
        "    container['running'] = True\n"
        "    if container.get('health') == 'exited':\n"
        "        container['health'] = 'running'\n"
        "    state['containers'][name] = container\n"
        "    save_state(state)\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'stop' and len(args) >= 2:\n"
        "    name = args[-1]\n"
        "    if name not in state['containers']:\n"
        "        raise SystemExit(1)\n"
        "    container = state['containers'][name]\n"
        "    container['running'] = False\n"
        "    state['containers'][name] = container\n"
        "    save_state(state)\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'rm':\n"
        "    name = args[-1]\n"
        "    if name in state.get('fail_rm', []):\n"
        "        raise SystemExit(1)\n"
        "    state['containers'].pop(name, None)\n"
        "    save_state(state)\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if args[0] == 'logs':\n"
        "    print('fake logs')\n"
        "    raise SystemExit(0)\n"
        "\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)


def _write_python_wrapper(bin_dir: Path) -> None:
    wrapper = bin_dir / "python3"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"REAL_PYTHON=\"{sys.executable}\"\n"
        "if [[ $# -ge 2 && \"$1\" == *\"deploylib.py\" && \"$2\" == \"verify-runtime\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ $# -ge 2 && \"$1\" == *\"deploylib.py\" && \"$2\" == \"assert-idle\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [[ $# -ge 3 && \"$1\" == *\"deploylib.py\" && \"$2\" == \"validate-env\" ]]; then\n"
        "  argv=()\n"
        "  swap=0\n"
        "  for token in \"$@\"; do\n"
        "    if [[ $swap -eq 1 ]]; then\n"
        "      if [[ \"$token\" == \"preflight\" ]]; then\n"
        "        token=doctor\n"
        "      fi\n"
        "      swap=0\n"
        "    fi\n"
        "    argv+=(\"$token\")\n"
        "    if [[ \"$token\" == \"--mode\" ]]; then\n"
        "      swap=1\n"
        "    fi\n"
        "  done\n"
        "  exec \"$REAL_PYTHON\" \"${argv[@]}\"\n"
        "fi\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    return repo, commit


def _write_env_file(
    path: Path,
    *,
    repo_root: Path,
    commit: str,
    key_path: Path,
    hf_cache: Path,
    state_dir: Path,
    served_model: str,
    port: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"DML_REPO_ROOT={repo_root}",
                f"DML_SOURCE_COMMIT={commit}",
                f"DAYSTROM_KV_SECRET_FILE={key_path}",
                f"HF_CACHE_DIR={hf_cache}",
                f"DAYSTROM_DEPLOY_STATE_DIR={state_dir}",
                f"VLLM_PORT={port}",
                f"VLLM_SERVED_MODEL={served_model}",
                "VLLM_CONTAINER_NAME=nemotron-warroom-vllm",
                "VLLM_IMAGE=vllm/vllm-openai:v0.20.0",
                "VLLM_STARTUP_TIMEOUT_SECONDS=30",
                "VLLM_VERIFY_TIMEOUT_SECONDS=20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_script(
    tmp_path: Path,
    action_args: list[str],
    env_file: Path,
    docker_state: Mapping[str, object],
    *,
    use_python_wrapper: bool = False,
    env_overrides: Mapping[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    script = _script_path()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_docker(bin_dir)
    if use_python_wrapper:
        _write_python_wrapper(bin_dir)

    state_file = tmp_path / "fake-docker-state.json"
    state_file.write_text(json.dumps(docker_state), encoding="utf-8")
    log_file = tmp_path / "fake-docker.log"
    log_file.write_text("", encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["ENV_FILE"] = str(env_file)
    env["FAKE_DOCKER_STATE_FILE"] = str(state_file)
    env["FAKE_DOCKER_LOG"] = str(log_file)
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["bash", str(script), *action_args],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, state_file, log_file


def _run_init(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    script = _script_path()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commit = "a" * 40
    _write_fake_git(bin_dir, repo_root, commit)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path / "home")
    Path(env["HOME"]).mkdir(parents=True)

    return subprocess.run(
        ["bash", str(script), "init", *args],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_init_force_env_preserves_existing_key(tmp_path: Path) -> None:
    env_path = tmp_path / "deploy.env"
    key_path = tmp_path / "secrets" / "control.key"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("ORIGINAL-KEY\n", encoding="utf-8")
    key_path.chmod(0o600)

    env_path.write_text(
        "\n".join(
            [
                f"DML_REPO_ROOT={tmp_path / 'repo'}",
                "DML_SOURCE_COMMIT=" + "b" * 40,
                f"DAYSTROM_KV_SECRET_FILE={key_path}",
                f"HF_CACHE_DIR={tmp_path / 'hf-cache'}",
                "VLLM_SERVED_MODEL=model",
                f"DAYSTROM_DEPLOY_STATE_DIR={tmp_path / 'state dir'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "hf-cache").mkdir()

    result = _run_init(tmp_path, ["--force-env", "--env-file", str(env_path)])
    assert result.returncode == 0, result.stderr
    assert key_path.read_text(encoding="utf-8") == "ORIGINAL-KEY\n"


def test_init_rotate_key_replaces_existing_key(tmp_path: Path) -> None:
    env_path = tmp_path / "deploy.env"
    key_path = tmp_path / "secrets" / "control.key"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("ORIGINAL-KEY\n", encoding="utf-8")
    key_path.chmod(0o600)
    (tmp_path / "hf-cache").mkdir()

    env_path.write_text(
        "\n".join(
            [
                f"DML_REPO_ROOT={tmp_path / 'repo'}",
                "DML_SOURCE_COMMIT=" + "b" * 40,
                f"DAYSTROM_KV_SECRET_FILE={key_path}",
                f"HF_CACHE_DIR={tmp_path / 'hf-cache'}",
                "VLLM_SERVED_MODEL=model",
                f"DAYSTROM_DEPLOY_STATE_DIR={tmp_path / 'state'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_init(tmp_path, ["--force-env", "--rotate-key", "--env-file", str(env_path)])
    assert result.returncode == 0, result.stderr
    assert "invalidates live authorizations" in result.stdout
    assert key_path.read_text(encoding="utf-8") != "ORIGINAL-KEY\n"


def test_init_refresh_commit_preserves_existing_key(tmp_path: Path) -> None:
    env_path = tmp_path / "deploy.env"
    key_path = tmp_path / "secrets" / "control.key"
    key_path.parent.mkdir(parents=True)
    key_path.write_text("ORIGINAL-KEY\n", encoding="utf-8")
    key_path.chmod(0o600)
    (tmp_path / "hf-cache").mkdir()

    env_path.write_text(
        "\n".join(
            [
                f"DML_REPO_ROOT={tmp_path / 'repo'}",
                "DML_SOURCE_COMMIT=" + "b" * 40,
                f"DAYSTROM_KV_SECRET_FILE={key_path}",
                f"HF_CACHE_DIR={tmp_path / 'hf-cache'}",
                "VLLM_SERVED_MODEL=model",
                f"DAYSTROM_DEPLOY_STATE_DIR={tmp_path / 'state'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_init(tmp_path, ["--refresh-commit", "--env-file", str(env_path)])
    assert result.returncode == 0, result.stderr
    assert key_path.read_text(encoding="utf-8") == "ORIGINAL-KEY\n"


def test_script_uses_gpus_all_and_env_numeric_policy_args() -> None:
    text = _script_path().read_text(encoding="utf-8")
    assert "--gpus all" in text
    assert "int(os.environ['DAYSTROM_MAX_TTL_SECONDS'])" in text
    assert "int(os.environ['DAYSTROM_MAX_RECORDS'])" in text


def test_doctor_invalid_env_stops_before_state_dir_creation(tmp_path: Path) -> None:
    script = _script_path()
    env_file = tmp_path / "bad.env"
    state_dir = tmp_path / "state"
    env_file.write_text(
        "\n".join(
            [
                "DML_REPO_ROOT=/abs/repo",
                "DML_SOURCE_COMMIT=not-a-commit",
                "DAYSTROM_KV_SECRET_FILE=/abs/key",
                "HF_CACHE_DIR=/abs/hf",
                "VLLM_SERVED_MODEL=model",
                f"DAYSTROM_DEPLOY_STATE_DIR={state_dir}",
                "BROKEN LINE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script), "doctor"],
        cwd=script.parent,
        env={**os.environ, "ENV_FILE": str(env_file)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected strict KEY=VALUE format" in result.stderr
    assert not state_dir.exists()


def test_status_works_when_repo_is_dirty(tmp_path: Path) -> None:
    repo, _commit = _init_repo(tmp_path)
    dirty_file = repo / "dml_core" / "daystrom_dml" / "tracked.txt"
    dirty_file.write_text("dirty\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit="f" * 40,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    result, _, _ = _run_script(
        tmp_path,
        ["status"],
        env_file,
        {"containers": {}, "health_plan": [], "fail_rm": []},
    )
    assert result.returncode == 0, result.stderr
    assert "RECORDED_BACKUP none" in result.stdout


def test_rollback_works_after_head_moves(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    (repo / "dml_core" / "daystrom_dml" / "new.txt").write_text("change\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "move head"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "deploy-state.json").write_text(
        json.dumps({"schema": 1, "backups": [{"name": "backup-a", "recorded_at": 1}]}),
        encoding="utf-8",
    )
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    docker_state = {
        "containers": {
            "backup-a": {"running": False, "health": "healthy", "image": "vllm/vllm-openai:v0.20.0"},
        },
        "health_plan": ["healthy"],
        "fail_rm": [],
    }
    result, state_file, _ = _run_script(tmp_path, ["rollback"], env_file, docker_state, use_python_wrapper=True)
    assert result.returncode == 0, result.stderr
    assert "RESTORE_OK" in result.stdout
    final_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert "nemotron-warroom-vllm" in final_state["containers"]


def test_deploy_waits_for_health_not_running(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    result, _, log_file = _run_script(
        tmp_path,
        ["deploy"],
        env_file,
        {"containers": {}, "health_plan": ["running", "running", "healthy"], "fail_rm": []},
        use_python_wrapper=True,
    )
    assert result.returncode == 0, result.stderr
    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    health_checks = [line for line in log_lines if ".State.Health.Status" in line]
    assert len(health_checks) >= 3


def test_compose_up_failure_restores_previous_container(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    docker_state = {
        "containers": {
            "nemotron-warroom-vllm": {
                "running": True,
                "health": "healthy",
                "image": "vllm/vllm-openai:v0.20.0",
            }
        },
        "health_plan": ["healthy"],
        "fail_rm": [],
    }

    result, state_file, _ = _run_script(
        tmp_path,
        ["deploy"],
        env_file,
        docker_state,
        use_python_wrapper=True,
        env_overrides={"FAKE_COMPOSE_UP_FAIL": "1"},
    )

    assert result.returncode != 0
    assert "compose startup failed and backup was restored" in result.stderr
    final_docker_state = json.loads(state_file.read_text(encoding="utf-8"))
    restored = final_docker_state["containers"]["nemotron-warroom-vllm"]
    assert restored["running"] is True
    deployment_state = json.loads((state_dir / "deploy-state.json").read_text(encoding="utf-8"))
    assert deployment_state["backups"] == []


def test_failed_backup_activation_preserves_backup_and_recovery(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "deploy-state.json"
    state_path.write_text(
        json.dumps({"schema": 1, "backups": [{"name": "backup-a", "recorded_at": 1}]}),
        encoding="utf-8",
    )
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    docker_state = {
        "containers": {
            "nemotron-warroom-vllm": {"running": True, "health": "healthy", "image": "vllm/vllm-openai:v0.20.0"},
            "backup-a": {"running": False, "health": "healthy", "image": "vllm/vllm-openai:v0.20.0"},
        },
        "health_plan": ["unhealthy"],
        "fail_rm": [],
    }
    result, state_file, _ = _run_script(tmp_path, ["rollback"], env_file, docker_state, use_python_wrapper=True)
    assert result.returncode == 0, result.stderr
    assert "ROLLBACK_NOOP" in result.stdout
    docker_state_after = json.loads(state_file.read_text(encoding="utf-8"))
    assert "backup-a" in docker_state_after["containers"]
    assert "nemotron-warroom-vllm" in docker_state_after["containers"]
    backups_after = json.loads(state_path.read_text(encoding="utf-8"))["backups"]
    assert backups_after[0]["name"] == "backup-a"


def test_prune_failure_preserves_state_entry(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "deploy-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "backups": [
                    {"name": "backup-new", "recorded_at": 3},
                    {"name": "backup-mid", "recorded_at": 2},
                    {"name": "backup-old", "recorded_at": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model="model",
        port=_unused_tcp_port(),
    )
    docker_state = {
        "containers": {
            "backup-old": {"running": False, "health": "healthy", "image": "vllm/vllm-openai:v0.20.0"},
            "nemotron-warroom-vllm": {"running": True, "health": "healthy", "image": "vllm/vllm-openai:v0.20.0"},
        },
        "health_plan": ["healthy"],
        "fail_rm": ["backup-old"],
    }
    result, _, _ = _run_script(tmp_path, ["deploy"], env_file, docker_state, use_python_wrapper=True)
    assert result.returncode != 0
    assert "unable to remove old backup container" in result.stderr
    backups_after = [entry["name"] for entry in json.loads(state_path.read_text(encoding="utf-8"))["backups"]]
    assert "backup-old" in backups_after


def test_preflight_exports_validated_kv_transfer_config_for_compose(tmp_path: Path) -> None:
    repo, commit = _init_repo(tmp_path)
    served_model = "model"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    key_path = tmp_path / "secret.key"
    key_path.write_text("a" * 64, encoding="utf-8")
    key_path.chmod(0o600)
    hf_cache = tmp_path / "hf"
    hf_cache.mkdir()
    env_file = tmp_path / ".env"
    _write_env_file(
        env_file,
        repo_root=repo,
        commit=commit,
        key_path=key_path,
        hf_cache=hf_cache,
        state_dir=state_dir,
        served_model=served_model,
        port=_unused_tcp_port(),
    )
    result, _, log_file = _run_script(
        tmp_path,
        ["preflight"],
        env_file,
        {"containers": {}, "health_plan": [], "fail_rm": []},
        use_python_wrapper=True,
    )
    assert result.returncode == 0, result.stderr
    expected_cfg = subprocess.run(
        ["python3", str(_helper_path()), "kv-transfer-config", "--env-file", str(env_file)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert f"KVCFG={expected_cfg}" in log_file.read_text(encoding="utf-8")
