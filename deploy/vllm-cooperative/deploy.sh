#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-"$HERE/.env"}
COMPOSE=(docker compose --project-directory "$HERE" --env-file "$ENV_FILE" -f "$HERE/compose.yaml")
ACTION=${1:-deploy}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

load_env() {
  [[ -f "$ENV_FILE" ]] || fail "missing environment file: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  : "${DML_REPO_ROOT:?set DML_REPO_ROOT in $ENV_FILE}"
  : "${DML_SOURCE_COMMIT:?set DML_SOURCE_COMMIT in $ENV_FILE}"
  : "${DAYSTROM_KV_SECRET_FILE:?set DAYSTROM_KV_SECRET_FILE in $ENV_FILE}"
  : "${HF_CACHE_DIR:?set HF_CACHE_DIR in $ENV_FILE}"
  VLLM_CONTAINER_NAME=${VLLM_CONTAINER_NAME:-nemotron-warroom-vllm}
  VLLM_PORT=${VLLM_PORT:-8000}
  VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0}
  DAYSTROM_DEPLOY_STATE_DIR=${DAYSTROM_DEPLOY_STATE_DIR:-"$HOME/.local/state/daystrom-vllm"}
  mkdir -p "$DAYSTROM_DEPLOY_STATE_DIR"
  BACKUP_FILE="$DAYSTROM_DEPLOY_STATE_DIR/rollback-container"
}

assert_idle() {
  local running
  running=$(docker ps -q --filter "name=^/${VLLM_CONTAINER_NAME}$")
  [[ -n "$running" ]] || return 0
  python3 - "$VLLM_PORT" <<'PY'
import sys, urllib.request
port = int(sys.argv[1])
text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
    total = sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(metric + "{")
    )
    if total:
        raise SystemExit(f"refusing deployment: {metric}={total}")
PY
}

preflight() {
  command -v docker >/dev/null || fail "docker is required"
  command -v git >/dev/null || fail "git is required"
  command -v python3 >/dev/null || fail "python3 is required"
  docker compose version >/dev/null
  [[ -d "$DML_REPO_ROOT/.git" ]] || fail "DML_REPO_ROOT is not a git checkout"
  [[ -d "$DML_REPO_ROOT/dml_core/daystrom_dml" ]] || fail "DML dml_core source is missing"
  [[ -d "$HF_CACHE_DIR" ]] || fail "HF_CACHE_DIR is not a directory"
  [[ -f "$DAYSTROM_KV_SECRET_FILE" && -s "$DAYSTROM_KV_SECRET_FILE" ]] || fail "control-key file is missing or empty"

  local actual mode dirty
  actual=$(git -C "$DML_REPO_ROOT" rev-parse HEAD)
  [[ "$actual" == "$DML_SOURCE_COMMIT" ]] || fail "source commit mismatch: expected $DML_SOURCE_COMMIT, got $actual"
  dirty=$(git -C "$DML_REPO_ROOT" status --porcelain --untracked-files=all -- dml_core)
  [[ -z "$dirty" ]] || fail "dml_core contains tracked or untracked modifications"
  mode=$(stat -c '%a' "$DAYSTROM_KV_SECRET_FILE")
  (( (8#$mode & 077) == 0 )) || fail "control-key permissions must deny group/world access"

  "${COMPOSE[@]}" config --quiet
  docker run --rm --entrypoint python3 \
    -e PYTHONPATH=/opt/daystrom/dml_core \
    -v "$DML_REPO_ROOT/dml_core:/opt/daystrom/dml_core:ro" \
    -v "$DAYSTROM_KV_SECRET_FILE:/run/secrets/daystrom-kv-control.key:ro" \
    "$VLLM_IMAGE" \
    -c 'import vllm; from daystrom_dml.context.vllm_bridge.connector import DaystromCooperativeKVConnector; from daystrom_dml.context.vllm_bridge.policy import DaystromKVPolicy; DaystromKVPolicy("/run/secrets/daystrom-kv-control.key", max_ttl_seconds=900, max_records=128); print(vllm.__version__, DaystromCooperativeKVConnector.__name__)' \
    >/dev/null
  assert_idle
  printf 'PREFLIGHT_OK commit=%s container=%s\n' "$actual" "$VLLM_CONTAINER_NAME"
}

restore_backup() {
  [[ -f "$BACKUP_FILE" ]] || return 0
  local backup
  backup=$(<"$BACKUP_FILE")
  [[ -n "$backup" ]] || return 0
  docker rm -f "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
  if docker inspect "$backup" >/dev/null 2>&1; then
    docker rename "$backup" "$VLLM_CONTAINER_NAME"
    docker start "$VLLM_CONTAINER_NAME" >/dev/null
    printf 'ROLLBACK_OK restored=%s\n' "$backup"
  fi
}

deploy() {
  preflight
  local existing managed backup stamp
  existing=$(docker ps -aq --filter "name=^/${VLLM_CONTAINER_NAME}$")
  managed=""
  if [[ -n "$existing" ]]; then
    managed=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$existing" 2>/dev/null || true)
  fi
  if [[ -n "$existing" && "$managed" != "daystrom-vllm" ]]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    backup="${VLLM_CONTAINER_NAME}-pre-compose-${stamp}"
    docker stop -t 30 "$VLLM_CONTAINER_NAME" >/dev/null
    docker rename "$VLLM_CONTAINER_NAME" "$backup"
    printf '%s\n' "$backup" >"$BACKUP_FILE"
  fi

  trap '"${COMPOSE[@]}" down >/dev/null 2>&1 || true; restore_backup' ERR
  "${COMPOSE[@]}" up -d
  local i
  for i in $(seq 1 180); do
    if ! docker inspect -f '{{.State.Running}}' "$VLLM_CONTAINER_NAME" 2>/dev/null | grep -qx true; then
      docker logs --tail 160 "$VLLM_CONTAINER_NAME" >&2 || true
      return 1
    fi
    if python3 - "$VLLM_PORT" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/health", timeout=3).read()
PY
    then
      trap - ERR
      printf 'DEPLOY_OK commit=%s container=%s wait_seconds=%s\n' "$DML_SOURCE_COMMIT" "$VLLM_CONTAINER_NAME" "$((i * 5))"
      return 0
    fi
    sleep 5
  done
  return 1
}

rollback() {
  assert_idle
  "${COMPOSE[@]}" down
  restore_backup
}

status() {
  docker ps -a --filter "name=^/${VLLM_CONTAINER_NAME}$" --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}'
  python3 - "$VLLM_PORT" <<'PY'
import sys, urllib.request
port = int(sys.argv[1])
for path in ("/health", "/v1/models"):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            print(path, response.status)
    except Exception as exc:
        print(path, type(exc).__name__)
PY
}

load_env
case "$ACTION" in
  preflight) preflight ;;
  deploy) deploy ;;
  rollback) rollback ;;
  status) status ;;
  *) fail "usage: $0 [preflight|deploy|rollback|status]" ;;
esac
