#!/usr/bin/env bash
set -euo pipefail

THREAD_KEY="${1:?thread key required}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="${2:-${DML_CHECKPOINT_DIR:-$WORKSPACE/out/dml-checkpoints}}"
THREAD_SAFE_KEY="$(python3 "$(dirname "$0")/thread_safe_key.py" "$THREAD_KEY")"
CANONICAL_PATH="$CHECKPOINT_DIR/${THREAD_SAFE_KEY}.md"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [ -f "$CANONICAL_PATH" ]; then
  printf '%s\n' "$CANONICAL_PATH"
  exit 0
fi

find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*_${THREAD_SAFE_KEY}.md" | sort | tail -1
