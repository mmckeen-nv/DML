#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-"$HERE/.env"}
COMPOSE=()
HELPER="$HERE/deploylib.py"
ACTION=${1:-help}
shift || true
CURRENT_REPO_ROOT=""

VLLM_IMAGE=""
VLLM_CONTAINER_NAME=""
VLLM_BIND_ADDRESS=""
VLLM_PORT=""
VLLM_SERVED_MODEL=""
VLLM_MODEL_PATH=""
VLLM_GPU_MEMORY_UTILIZATION=""
VLLM_MAX_NUM_SEQS=""
VLLM_MAX_MODEL_LEN=""
VLLM_MOE_BACKEND=""
VLLM_QUANTIZATION=""
VLLM_DTYPE=""
VLLM_KV_CACHE_DTYPE=""
VLLM_MAMBA_SSM_CACHE_DTYPE=""
VLLM_TENSOR_PARALLEL_SIZE=""
VLLM_PIPELINE_PARALLEL_SIZE=""
VLLM_DATA_PARALLEL_SIZE=""
DML_REPO_ROOT=""
DML_SOURCE_COMMIT=""
DAYSTROM_KV_SECRET_FILE=""
HF_CACHE_DIR=""
DAYSTROM_DEPLOY_STATE_DIR=""
DAYSTROM_ALLOW_PUBLIC_BIND=""
DAYSTROM_CPU_KV_BYTES=""
DAYSTROM_MAX_TTL_SECONDS=""
DAYSTROM_MAX_RECORDS=""
VLLM_STARTUP_TIMEOUT_SECONDS=""
VLLM_VERIFY_TIMEOUT_SECONDS=""
DAYSTROM_KV_TRANSFER_CONFIG=""

STATE_FILE=""
LOCK_DIR=""
LOCK_META_FILE=""
LOCK_HELD=0
RECOVER_STALE_LOCK=0
MAX_BACKUPS=2

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage: ./deploy.sh <action> [options]

Actions:
  init [--force-env] [--refresh-commit] [--rotate-key] [--env-file PATH] [--key-file PATH]
  doctor
  pull
  preflight [--pull]
  deploy [--recover-stale-lock]
  verify
  status
  rollback [--recover-stale-lock]
  logs [--tail N]
EOF
}

require_helper() {
  [[ -f "$HELPER" ]] || fail "missing helper: $HELPER"
}

resolve_current_repo_root() {
  if [[ -n "$CURRENT_REPO_ROOT" ]]; then
    return 0
  fi
  CURRENT_REPO_ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)
  [[ -n "$CURRENT_REPO_ROOT" ]] || CURRENT_REPO_ROOT="$HERE"
}

load_exported_values() {
  local env_tsv=$1
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" == *$'\t'* ]] || fail "malformed helper export row"
    local key value
    key=${line%%$'\t'*}
    value=${line#*$'\t'}
    case "$key" in
      DML_REPO_ROOT) DML_REPO_ROOT="$value" ;;
      DML_SOURCE_COMMIT) DML_SOURCE_COMMIT="$value" ;;
      DAYSTROM_KV_SECRET_FILE) DAYSTROM_KV_SECRET_FILE="$value" ;;
      HF_CACHE_DIR) HF_CACHE_DIR="$value" ;;
      DAYSTROM_DEPLOY_STATE_DIR) DAYSTROM_DEPLOY_STATE_DIR="$value" ;;
      DAYSTROM_CPU_KV_BYTES) DAYSTROM_CPU_KV_BYTES="$value" ;;
      DAYSTROM_MAX_TTL_SECONDS) DAYSTROM_MAX_TTL_SECONDS="$value" ;;
      DAYSTROM_MAX_RECORDS) DAYSTROM_MAX_RECORDS="$value" ;;
      VLLM_IMAGE) VLLM_IMAGE="$value" ;;
      VLLM_CONTAINER_NAME) VLLM_CONTAINER_NAME="$value" ;;
      VLLM_BIND_ADDRESS) VLLM_BIND_ADDRESS="$value" ;;
      VLLM_PORT) VLLM_PORT="$value" ;;
      VLLM_SERVED_MODEL) VLLM_SERVED_MODEL="$value" ;;
      VLLM_MODEL_PATH) VLLM_MODEL_PATH="$value" ;;
      VLLM_GPU_MEMORY_UTILIZATION) VLLM_GPU_MEMORY_UTILIZATION="$value" ;;
      VLLM_MAX_NUM_SEQS) VLLM_MAX_NUM_SEQS="$value" ;;
      VLLM_MAX_MODEL_LEN) VLLM_MAX_MODEL_LEN="$value" ;;
      VLLM_MOE_BACKEND) VLLM_MOE_BACKEND="$value" ;;
      VLLM_QUANTIZATION) VLLM_QUANTIZATION="$value" ;;
      VLLM_DTYPE) VLLM_DTYPE="$value" ;;
      VLLM_KV_CACHE_DTYPE) VLLM_KV_CACHE_DTYPE="$value" ;;
      VLLM_MAMBA_SSM_CACHE_DTYPE) VLLM_MAMBA_SSM_CACHE_DTYPE="$value" ;;
      VLLM_TENSOR_PARALLEL_SIZE) VLLM_TENSOR_PARALLEL_SIZE="$value" ;;
      VLLM_PIPELINE_PARALLEL_SIZE) VLLM_PIPELINE_PARALLEL_SIZE="$value" ;;
      VLLM_DATA_PARALLEL_SIZE) VLLM_DATA_PARALLEL_SIZE="$value" ;;
      VLLM_STARTUP_TIMEOUT_SECONDS) VLLM_STARTUP_TIMEOUT_SECONDS="$value" ;;
      VLLM_VERIFY_TIMEOUT_SECONDS) VLLM_VERIFY_TIMEOUT_SECONDS="$value" ;;
      DAYSTROM_ALLOW_PUBLIC_BIND) DAYSTROM_ALLOW_PUBLIC_BIND="$value" ;;
      DAYSTROM_KV_TRANSFER_CONFIG) DAYSTROM_KV_TRANSFER_CONFIG="$value" ;;
    esac
  done <<<"$env_tsv"
}

load_env() {
  require_helper
  resolve_current_repo_root
  [[ -f "$ENV_FILE" ]] || fail "missing environment file: $ENV_FILE"
  local env_tsv
  env_tsv=$(python3 "$HELPER" export-env --env-file "$ENV_FILE") || fail "failed to load $ENV_FILE"
  load_exported_values "$env_tsv"

  python3 "$HELPER" validate-env --env-file "$ENV_FILE" --mode basic --current-repo-root "$CURRENT_REPO_ROOT"

  [[ "$DAYSTROM_DEPLOY_STATE_DIR" = /* ]] || fail "DAYSTROM_DEPLOY_STATE_DIR must be an absolute path"
  STATE_FILE="$DAYSTROM_DEPLOY_STATE_DIR/deploy-state.json"
  LOCK_DIR="$DAYSTROM_DEPLOY_STATE_DIR/deploy.lock"
  LOCK_META_FILE="$LOCK_DIR/owner.tsv"
  export DAYSTROM_KV_TRANSFER_CONFIG
}

validate_doctor_env() {
  resolve_current_repo_root
  python3 "$HELPER" validate-env --env-file "$ENV_FILE" --mode doctor --current-repo-root "$CURRENT_REPO_ROOT"
}

validate_preflight_env() {
  resolve_current_repo_root
  python3 "$HELPER" validate-env --env-file "$ENV_FILE" --mode preflight --current-repo-root "$CURRENT_REPO_ROOT"
}

compose_config_quiet() {
  "${COMPOSE[@]}" config --quiet
}

compose_up() {
  "${COMPOSE[@]}" up -d
}

render_env_content() {
  cat <<EOF
# Daystrom cooperative vLLM deployment configuration
DML_REPO_ROOT=$DML_REPO_ROOT
DML_SOURCE_COMMIT=$DML_SOURCE_COMMIT
DAYSTROM_KV_SECRET_FILE=$DAYSTROM_KV_SECRET_FILE
HF_CACHE_DIR=$HF_CACHE_DIR

VLLM_IMAGE=$VLLM_IMAGE
VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME
VLLM_BIND_ADDRESS=$VLLM_BIND_ADDRESS
DAYSTROM_ALLOW_PUBLIC_BIND=$DAYSTROM_ALLOW_PUBLIC_BIND
VLLM_PORT=$VLLM_PORT
VLLM_MODEL_PATH=$VLLM_MODEL_PATH
VLLM_SERVED_MODEL=$VLLM_SERVED_MODEL
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
VLLM_MOE_BACKEND=$VLLM_MOE_BACKEND
VLLM_QUANTIZATION=$VLLM_QUANTIZATION
VLLM_DTYPE=$VLLM_DTYPE
VLLM_KV_CACHE_DTYPE=$VLLM_KV_CACHE_DTYPE
VLLM_MAMBA_SSM_CACHE_DTYPE=$VLLM_MAMBA_SSM_CACHE_DTYPE
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE
VLLM_PIPELINE_PARALLEL_SIZE=$VLLM_PIPELINE_PARALLEL_SIZE
VLLM_DATA_PARALLEL_SIZE=$VLLM_DATA_PARALLEL_SIZE

DAYSTROM_CPU_KV_BYTES=$DAYSTROM_CPU_KV_BYTES
DAYSTROM_MAX_TTL_SECONDS=$DAYSTROM_MAX_TTL_SECONDS
DAYSTROM_MAX_RECORDS=$DAYSTROM_MAX_RECORDS

VLLM_STARTUP_TIMEOUT_SECONDS=$VLLM_STARTUP_TIMEOUT_SECONDS
VLLM_VERIFY_TIMEOUT_SECONDS=$VLLM_VERIFY_TIMEOUT_SECONDS
DAYSTROM_DEPLOY_STATE_DIR=$DAYSTROM_DEPLOY_STATE_DIR
EOF
}

write_env_atomic() {
  local target_env=$1
  local content=$2
  python3 - "$target_env" "$content" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
payload = sys.argv[2]
target.parent.mkdir(parents=True, exist_ok=True)
fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", dir=str(target.parent))
tmp_path = Path(tmp_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, target)
finally:
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
PY
}

have_named_container() {
  docker ps -aq --filter "name=^/${VLLM_CONTAINER_NAME}$" | grep -q .
}

is_running_container() {
  docker inspect -f '{{.State.Running}}' "$VLLM_CONTAINER_NAME" 2>/dev/null | grep -qx true
}

assert_idle() {
  if have_named_container && is_running_container; then
    python3 "$HELPER" assert-idle --port "$VLLM_PORT" --timeout-seconds 5
  fi
}

assert_port_available_or_owned() {
  if python3 "$HELPER" check-port --bind-address "$VLLM_BIND_ADDRESS" --port "$VLLM_PORT" >/dev/null; then
    return 0
  fi
  if python3 "$HELPER" check-port-owner --container-name "$VLLM_CONTAINER_NAME" --bind-address "$VLLM_BIND_ADDRESS" --port "$VLLM_PORT" >/dev/null; then
    return 0
  fi
  fail "port collision on ${VLLM_BIND_ADDRESS}:${VLLM_PORT} is not owned by running ${VLLM_CONTAINER_NAME}"
}

doctor() {
  load_env
  validate_doctor_env
  note "DOCTOR_OK env=$ENV_FILE"
  note "Doctor is filesystem-only and does not require Linux, Docker, daemon, images, or GPU runtime."
}

runtime_prereqs() {
  command -v docker >/dev/null || fail "docker is required"
  command -v python3 >/dev/null || fail "python3 is required"
  resolve_compose
  docker info >/dev/null 2>&1 || fail "docker daemon is unavailable"
  local runtimes
  runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  [[ "$runtimes" == *"nvidia"* ]] || fail "nvidia docker runtime is unavailable"
}

resolve_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose --project-directory "$HERE" --env-file "$ENV_FILE" -f "$HERE/compose.yaml")
  elif command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
    COMPOSE=(docker-compose --project-directory "$HERE" --env-file "$ENV_FILE" -f "$HERE/compose.yaml")
  else
    fail "Docker Compose is required (docker compose plugin or docker-compose executable)"
  fi
}

compatibility_check() {
  docker run --rm --gpus all --entrypoint python3 \
    -e PYTHONPATH=/opt/daystrom/dml_core \
    -e DAYSTROM_MAX_TTL_SECONDS="$DAYSTROM_MAX_TTL_SECONDS" \
    -e DAYSTROM_MAX_RECORDS="$DAYSTROM_MAX_RECORDS" \
    -v "$DML_REPO_ROOT/dml_core:/opt/daystrom/dml_core:ro" \
    -v "$DAYSTROM_KV_SECRET_FILE:/run/secrets/daystrom-kv-control.key:ro" \
    "$VLLM_IMAGE" \
    -c "import os; import vllm; from daystrom_dml.context.vllm_bridge.connector import DaystromCooperativeKVConnector; from daystrom_dml.context.vllm_bridge.policy import DaystromKVPolicy; assert vllm.__version__.startswith('0.20.'), vllm.__version__; DaystromKVPolicy('/run/secrets/daystrom-kv-control.key', max_ttl_seconds=int(os.environ['DAYSTROM_MAX_TTL_SECONDS']), max_records=int(os.environ['DAYSTROM_MAX_RECORDS'])); assert DaystromCooperativeKVConnector.__name__ == 'DaystromCooperativeKVConnector'" \
    >/dev/null || fail "vLLM image/source compatibility check failed"
}

pull_action() {
  load_env
  validate_preflight_env
  runtime_prereqs
  docker pull "$VLLM_IMAGE" >/dev/null || fail "unable to pull image: $VLLM_IMAGE"
  note "PULL_OK image=$VLLM_IMAGE"
}

preflight() {
  local pull_missing=0
  while (($# > 0)); do
    case "$1" in
      --pull) pull_missing=1 ;;
      *) fail "unknown preflight option: $1" ;;
    esac
    shift
  done

  load_env
  validate_preflight_env
  runtime_prereqs
  compose_config_quiet || fail "compose file is malformed"

  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    if [[ "$pull_missing" -eq 1 ]]; then
      note "Image missing locally, pulling $VLLM_IMAGE"
      docker pull "$VLLM_IMAGE" >/dev/null || fail "unable to pull image: $VLLM_IMAGE"
    else
      fail "required image not present locally: $VLLM_IMAGE (run ./deploy.sh pull or ./deploy.sh preflight --pull)"
    fi
  fi

  assert_port_available_or_owned
  compatibility_check
  assert_idle
  note "PREFLIGHT_OK commit=$DML_SOURCE_COMMIT container=$VLLM_CONTAINER_NAME"
}

pid_absent_provably() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  if ps -p "$pid" -o pid= >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

clear_lock_dir() {
  [[ -d "$LOCK_DIR" ]] || return 0
  [[ -f "$LOCK_META_FILE" ]] && rm -f "$LOCK_META_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null || return 1
}

cleanup_lock() {
  if [[ "$LOCK_HELD" -eq 1 ]]; then
    clear_lock_dir || true
    LOCK_HELD=0
  fi
}

with_lock() {
  mkdir -p "$DAYSTROM_DEPLOY_STATE_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf 'pid\t%s\ncreated_at\t%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOCK_META_FILE"
    LOCK_HELD=1
    trap cleanup_lock EXIT
    return 0
  fi

  local owner_pid="" owner_created=""
  if [[ -f "$LOCK_META_FILE" ]]; then
    while IFS=$'\t' read -r field value; do
      case "$field" in
        pid) owner_pid="$value" ;;
        created_at) owner_created="$value" ;;
      esac
    done <"$LOCK_META_FILE"
  fi

  if [[ "$RECOVER_STALE_LOCK" -eq 1 && -n "$owner_pid" ]] && pid_absent_provably "$owner_pid"; then
    note "Recovering stale lock owned by absent pid=$owner_pid created_at=${owner_created:-unknown}"
    clear_lock_dir || fail "unable to recover stale lock at $LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || fail "deployment lock held: $LOCK_DIR"
    printf 'pid\t%s\ncreated_at\t%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOCK_META_FILE"
    LOCK_HELD=1
    trap cleanup_lock EXIT
    return 0
  fi

  fail "deployment lock held: $LOCK_DIR owner_pid=${owner_pid:-unknown} created_at=${owner_created:-unknown}; use --recover-stale-lock only when owner pid is gone"
}

backup_current_service() {
  local stamp backup suffix was_running=0
  have_named_container || return 0
  python3 "$HELPER" state-require-capacity --state-file "$STATE_FILE"
  if is_running_container; then
    was_running=1
    assert_idle
    docker stop -t 30 "$VLLM_CONTAINER_NAME" >/dev/null
  fi
  stamp=$(date -u +%Y%m%d%H%M%S)
  suffix=$(python3 -c 'import secrets; print(secrets.token_hex(4))')
  backup="${VLLM_CONTAINER_NAME}-backup-${stamp}-${suffix}"
  docker rename "$VLLM_CONTAINER_NAME" "$backup"
  docker inspect "$backup" >/dev/null 2>&1 || fail "backup rename failed"
  if ! python3 "$HELPER" state-push --state-file "$STATE_FILE" --backup-name "$backup"; then
    docker rename "$backup" "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || fail "backup state write failed and original container rename could not be reversed"
    if [[ "$was_running" -eq 1 ]]; then
      docker start "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || fail "backup state write failed and original container could not be restarted"
    fi
    fail "backup state write failed; original container restored"
  fi
  note "BACKUP_RECORDED name=$backup"
}

print_failure_diagnostics() {
  note "Collecting bounded diagnostics..."
  docker ps -a --filter "name=^/${VLLM_CONTAINER_NAME}$" --format 'container={{.Names}} image={{.Image}} status={{.Status}} ports={{.Ports}}' || true
  docker logs --tail 200 "$VLLM_CONTAINER_NAME" 2>/dev/null || true
}

container_health_status() {
  docker inspect -f '{{if .State.Running}}{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}{{else}}exited{{end}}' "$VLLM_CONTAINER_NAME" 2>/dev/null || true
}

wait_for_container_healthy() {
  local timeout_seconds elapsed status progress_tick
  timeout_seconds=$VLLM_STARTUP_TIMEOUT_SECONDS
  elapsed=0
  progress_tick=0
  while (( elapsed < timeout_seconds )); do
    status=$(container_health_status)
    case "$status" in
      healthy) return 0 ;;
      unhealthy|exited)
        note "Startup status became terminal: $status"
        return 1
        ;;
      running|starting|"") ;;
      *) ;;
    esac
    if (( progress_tick == 0 )); then
      note "Waiting for container health... status=${status:-unknown} elapsed=${elapsed}s/${timeout_seconds}s"
      progress_tick=15
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    progress_tick=$((progress_tick - 5))
  done
  return 1
}

restore_backup_transactional() {
  local backup
  backup=$(python3 "$HELPER" state-peek --state-file "$STATE_FILE" 2>/dev/null || true)
  if [[ -z "$backup" ]]; then
    note "RESTORE_SKIPPED reason=no recorded backup state"
    return 1
  fi
  if ! docker inspect "$backup" >/dev/null 2>&1; then
    note "RESTORE_SKIPPED reason=recorded backup not found name=$backup"
    return 1
  fi

  local backup_image
  backup_image=$(docker inspect -f '{{.Config.Image}}' "$backup" 2>/dev/null || true)
  [[ -n "$backup_image" ]] || fail "recorded backup has no image metadata: $backup"
  docker image inspect "$backup_image" >/dev/null 2>&1 || fail "recorded backup image missing locally: $backup_image"

  local had_current=0 recovery_name="" stamp suffix
  if have_named_container; then
    had_current=1
    if is_running_container; then
      docker stop -t 30 "$VLLM_CONTAINER_NAME" >/dev/null || true
    fi
    stamp=$(date -u +%Y%m%d%H%M%S)
    suffix=$(python3 -c 'import secrets; print(secrets.token_hex(4))')
    recovery_name="${VLLM_CONTAINER_NAME}-recovery-${stamp}-${suffix}"
    docker rename "$VLLM_CONTAINER_NAME" "$recovery_name" || fail "unable to stage current container for recovery"
  fi

  if ! docker rename "$backup" "$VLLM_CONTAINER_NAME"; then
    if [[ "$had_current" -eq 1 ]]; then
      docker rename "$recovery_name" "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    fail "unable to activate backup container: $backup"
  fi

  if ! docker start "$VLLM_CONTAINER_NAME" >/dev/null || ! wait_for_container_healthy || ! python3 "$HELPER" verify-runtime --port "$VLLM_PORT" --served-model "$VLLM_SERVED_MODEL" --timeout-seconds "$VLLM_VERIFY_TIMEOUT_SECONDS"; then
    note "Backup activation failed; restoring previous container"
    docker stop "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rename "$VLLM_CONTAINER_NAME" "$backup" >/dev/null 2>&1 || true
    if [[ "$had_current" -eq 1 ]]; then
      docker rename "$recovery_name" "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
      docker start "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    return 1
  fi

  if [[ -n "$recovery_name" ]]; then
    docker rm -f "$recovery_name" >/dev/null 2>&1 || true
  fi
  python3 "$HELPER" state-pop --state-file "$STATE_FILE" --backup-name "$backup" >/dev/null || fail "unable to clear restored backup state: $backup"
  note "RESTORE_OK container=$VLLM_CONTAINER_NAME source_backup=$backup"
  return 0
}

prune_backups_after_success() {
  local old_backups
  old_backups=$(python3 "$HELPER" state-list-old --state-file "$STATE_FILE" --keep "$MAX_BACKUPS") || fail "unable to list old backups"
  while IFS= read -r backup_name; do
    [[ -n "$backup_name" ]] || continue
    if docker inspect "$backup_name" >/dev/null 2>&1; then
      docker rm -f "$backup_name" >/dev/null || fail "unable to remove old backup container: $backup_name"
    else
      if docker ps -aq --filter "name=^/${backup_name}$" | grep -q .; then
        fail "unable to prove backup container is absent: $backup_name"
      fi
    fi
    python3 "$HELPER" state-pop --state-file "$STATE_FILE" --backup-name "$backup_name" >/dev/null || fail "unable to prune backup state entry: $backup_name"
    note "BACKUP_PRUNED name=$backup_name"
  done <<<"$old_backups"
}

verify() {
  load_env
  python3 "$HELPER" verify-runtime --port "$VLLM_PORT" --served-model "$VLLM_SERVED_MODEL" --timeout-seconds "$VLLM_VERIFY_TIMEOUT_SECONDS"
  note "VERIFY_OK model=$VLLM_SERVED_MODEL"
  note "Hardware KV lifecycle canary remains a separate operator runbook."
}

deploy() {
  while (($# > 0)); do
    case "$1" in
      --recover-stale-lock) RECOVER_STALE_LOCK=1 ;;
      *) fail "unknown deploy option: $1" ;;
    esac
    shift
  done

  preflight
  with_lock
  backup_current_service

  if ! compose_up; then
    print_failure_diagnostics
    if restore_backup_transactional; then
      fail "compose startup failed and backup was restored"
    fi
    fail "compose startup failed and backup restore failed"
  fi

  if ! wait_for_container_healthy; then
    print_failure_diagnostics
    if restore_backup_transactional; then
      fail "container did not become healthy before timeout; backup restored"
    fi
    fail "container did not become healthy before timeout and backup restore failed"
  fi

  if ! python3 "$HELPER" verify-runtime --port "$VLLM_PORT" --served-model "$VLLM_SERVED_MODEL" --timeout-seconds "$VLLM_VERIFY_TIMEOUT_SECONDS"; then
    print_failure_diagnostics
    if restore_backup_transactional; then
      fail "post-start verify failed and backup was restored"
    fi
    fail "post-start verify failed and backup restore failed"
  fi

  prune_backups_after_success
  note "DEPLOY_OK commit=$DML_SOURCE_COMMIT container=$VLLM_CONTAINER_NAME"
  note "Hardware KV lifecycle canary remains a separate operator runbook."
}

rollback() {
  while (($# > 0)); do
    case "$1" in
      --recover-stale-lock) RECOVER_STALE_LOCK=1 ;;
      *) fail "unknown rollback option: $1" ;;
    esac
    shift
  done

  load_env
  runtime_prereqs
  with_lock
  assert_idle
  if ! restore_backup_transactional; then
    note "ROLLBACK_NOOP no valid recorded backup exists"
  fi
}

status() {
  load_env
  docker ps -a --filter "name=^/${VLLM_CONTAINER_NAME}$" --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}'
  local backup
  backup=$(python3 "$HELPER" state-peek --state-file "$STATE_FILE" 2>/dev/null || true)
  if [[ -n "$backup" ]]; then
    note "RECORDED_BACKUP $backup"
  else
    note "RECORDED_BACKUP none"
  fi
  python3 - "$VLLM_PORT" <<'PY'
import sys
import urllib.request

port = int(sys.argv[1])
for path in ("/health", "/v1/models"):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            print(path, response.status)
    except Exception as exc:
        print(path, type(exc).__name__)
PY
}

logs_action() {
  load_env
  local tail_count=200
  while (($# > 0)); do
    case "$1" in
      --tail)
        shift
        [[ $# -gt 0 ]] || fail "--tail requires a value"
        tail_count="$1"
        ;;
      *) fail "unknown logs option: $1" ;;
    esac
    shift
  done
  docker logs --tail "$tail_count" "$VLLM_CONTAINER_NAME"
}

set_init_defaults() {
  DML_REPO_ROOT=$1
  DAYSTROM_KV_SECRET_FILE=$2
  HF_CACHE_DIR=$3
  DAYSTROM_DEPLOY_STATE_DIR=$4
  VLLM_IMAGE="vllm/vllm-openai:v0.20.0"
  VLLM_CONTAINER_NAME="nemotron-warroom-vllm"
  VLLM_BIND_ADDRESS="127.0.0.1"
  DAYSTROM_ALLOW_PUBLIC_BIND=""
  VLLM_PORT="8000"
  VLLM_MODEL_PATH="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  VLLM_SERVED_MODEL="nvidia/nemotron-3-super"
  VLLM_GPU_MEMORY_UTILIZATION="0.70"
  VLLM_MAX_NUM_SEQS="2"
  VLLM_MAX_MODEL_LEN="65536"
  VLLM_MOE_BACKEND="marlin"
  VLLM_QUANTIZATION="fp4"
  VLLM_DTYPE="auto"
  VLLM_KV_CACHE_DTYPE="fp8"
  VLLM_MAMBA_SSM_CACHE_DTYPE="float16"
  VLLM_TENSOR_PARALLEL_SIZE="1"
  VLLM_PIPELINE_PARALLEL_SIZE="1"
  VLLM_DATA_PARALLEL_SIZE="1"
  DAYSTROM_CPU_KV_BYTES="8589934592"
  DAYSTROM_MAX_TTL_SECONDS="900"
  DAYSTROM_MAX_RECORDS="128"
  VLLM_STARTUP_TIMEOUT_SECONDS="900"
  VLLM_VERIFY_TIMEOUT_SECONDS="20"
}

init() {
  local force_env=0 refresh_commit=0 rotate_key=0
  local target_env="$ENV_FILE"
  local key_file="$HOME/.local/state/daystrom-vllm/daystrom-kv-control.key"
  local repo_root source_commit hf_cache state_dir existing_env=0 existing_tsv="" existing_key=""

  repo_root=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)
  [[ -n "$repo_root" ]] || fail "unable to locate repository root from $HERE"
  source_commit=$(git -C "$repo_root" rev-parse HEAD)
  hf_cache="$HOME/.cache/huggingface"
  state_dir="$HOME/.local/state/daystrom-vllm"

  while (($# > 0)); do
    case "$1" in
      --force-env) force_env=1 ;;
      --refresh-commit) refresh_commit=1 ;;
      --rotate-key) rotate_key=1 ;;
      --env-file)
        shift
        [[ $# -gt 0 ]] || fail "--env-file requires a value"
        target_env="$1"
        ;;
      --key-file)
        shift
        [[ $# -gt 0 ]] || fail "--key-file requires a value"
        key_file="$1"
        ;;
      *) fail "unknown init option: $1" ;;
    esac
    shift
  done

  [[ -e "$target_env" ]] && existing_env=1
  [[ "$refresh_commit" -eq 1 && "$existing_env" -ne 1 ]] && fail "--refresh-commit requires an existing env file: $target_env"
  if [[ "$refresh_commit" -eq 0 && "$force_env" -eq 0 && "$existing_env" -eq 1 ]]; then
    fail "refusing to overwrite existing env file: $target_env (use --force-env or --refresh-commit)"
  fi

  set_init_defaults "$repo_root" "$key_file" "$hf_cache" "$state_dir"
  if [[ "$existing_env" -eq 1 ]]; then
    existing_tsv=$(python3 "$HELPER" export-env --env-file "$target_env") || fail "existing env file is invalid: $target_env"
    load_exported_values "$existing_tsv"
    while IFS= read -r line; do
      [[ "$line" == DAYSTROM_KV_SECRET_FILE$'\t'* ]] || continue
      existing_key=${line#*$'\t'}
    done <<<"$existing_tsv"
    if [[ -n "$existing_key" && "$rotate_key" -ne 1 ]]; then
      if [[ "$key_file" != "$HOME/.local/state/daystrom-vllm/daystrom-kv-control.key" && "$key_file" != "$existing_key" ]]; then
        fail "refusing to change key path without --rotate-key; existing key path is $existing_key"
      fi
      DAYSTROM_KV_SECRET_FILE="$existing_key"
    fi
  fi

  DML_REPO_ROOT="$repo_root"
  DML_SOURCE_COMMIT="$source_commit"

  if [[ "$rotate_key" -eq 1 ]]; then
    note "WARNING: rotating control key invalidates live authorizations until clients refresh"
    python3 "$HELPER" generate-key --output "$DAYSTROM_KV_SECRET_FILE" --bytes 32 --force --repo-root "$repo_root"
  elif [[ ! -f "$DAYSTROM_KV_SECRET_FILE" ]]; then
    python3 "$HELPER" generate-key --output "$DAYSTROM_KV_SECRET_FILE" --bytes 32 --repo-root "$repo_root"
  fi

  local env_content
  env_content=$(render_env_content)
  write_env_atomic "$target_env" "$env_content"
  note "INIT_OK env=$target_env key_file=$DAYSTROM_KV_SECRET_FILE"
  note "Control key preserved unless --rotate-key is used; value was not printed."
}

case "$ACTION" in
  init) init "$@" ;;
  doctor) doctor ;;
  pull) pull_action ;;
  preflight) preflight "$@" ;;
  deploy) deploy "$@" ;;
  verify) verify ;;
  status) status ;;
  rollback) rollback "$@" ;;
  logs) logs_action "$@" ;;
  help|-h|--help) usage ;;
  *)
    usage
    fail "unknown action: $ACTION"
    ;;
esac
