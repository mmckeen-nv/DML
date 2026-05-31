#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-$OPENCLAW_HOME/workspace}"
DAYSTROM_DML_HOME="${DAYSTROM_DML_HOME:-$OPENCLAW_HOME/daystrom-dml-v2}"
VENV="${DAYSTROM_DML_VENV:-$DAYSTROM_DML_HOME/.venv-dml}"
SKILL_TARGET="${DML_SKILL_TARGET:-$OPENCLAW_WORKSPACE/skills/daystrom-dml}"
STORE="${DML_STORE:-$OPENCLAW_HOME/dml-store}"
EXTRAS="${DML_INSTALL_EXTRAS:-server,mcp}"
DRY_RUN=0
RUN_SMOKE=1
PROFILE="openclaw"
PROFILE_OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-smoke) RUN_SMOKE=0 ;;
    --profile)
      PROFILE="${2:-}"
      shift
      ;;
    --profile-output)
      PROFILE_OUTPUT="${2:-}"
      shift
      ;;
    --help)
      cat <<EOF
Usage: scripts/install_daystrom_dml.sh [--dry-run] [--skip-smoke] [--profile openclaw|hermes|generic] [--profile-output path]

Environment:
  OPENCLAW_HOME         default: $HOME/.openclaw
  DAYSTROM_DML_HOME    default: \$OPENCLAW_HOME/daystrom-dml-v2
  DAYSTROM_DML_VENV    default: \$DAYSTROM_DML_HOME/.venv-dml
  DML_STORE            default: \$OPENCLAW_HOME/dml-store
  DML_INSTALL_EXTRAS   default: server,mcp
EOF
      exit 0
      ;;
  esac
  shift
done

case "$PROFILE" in
  openclaw|hermes|generic) ;;
  *) echo "unsupported --profile: $PROFILE" >&2; exit 1 ;;
esac

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = "0" ]; then
    "$@"
  fi
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

run mkdir -p "$DAYSTROM_DML_HOME" "$OPENCLAW_WORKSPACE/skills" "$STORE"
if [ ! -d "$VENV" ]; then
  run python3 -m venv "$VENV"
fi

run "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
run "$VENV/bin/python" -m pip install -e "$REPO_ROOT[$EXTRAS]"
run rsync -a --exclude='__pycache__/' --exclude='.pytest_cache/' "$REPO_ROOT/openclaw-wrapper/" "$SKILL_TARGET/"

if [ -z "$PROFILE_OUTPUT" ]; then
  PROFILE_OUTPUT="$DAYSTROM_DML_HOME/${PROFILE}-dml-profile.json"
fi
run "$VENV/bin/python" -m daystrom_dml.provider_cli install-app \
  --app "$PROFILE" \
  --storage-dir "$STORE" \
  --base-url "http://127.0.0.1:8765" \
  --output "$PROFILE_OUTPUT"

if [ "$RUN_SMOKE" = "1" ]; then
  run "$VENV/bin/python" "$SKILL_TARGET/scripts/dml_memory.py" \
    --storage-dir "$STORE" \
    --config-path "$SKILL_TARGET/config/dml_portable_linux.yaml" \
    --no-require-gpu \
    health
fi

cat <<EOF

DML install complete.

Provider UI:
  $VENV/bin/dml-provider --storage-dir "$STORE" --host 127.0.0.1 --port 8765

MCP:
  $VENV/bin/dml-mcp-server --transport stdio --storage "$STORE"

OpenClaw wrapper:
  $VENV/bin/python "$SKILL_TARGET/scripts/dml_memory.py" --storage-dir "$STORE" --no-require-gpu resume

Frontier prompt preparation:
  $VENV/bin/python "$SKILL_TARGET/scripts/dml_frontier_prepare.py" --base-url "http://127.0.0.1:8765" --prompt "current task" --telemetry-only

Agent profile:
  $PROFILE_OUTPUT
EOF
