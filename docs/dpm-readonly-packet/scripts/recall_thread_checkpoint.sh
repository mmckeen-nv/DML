#!/usr/bin/env bash
set -euo pipefail

THREAD_KEY="${1:?thread key required}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_PATH="$($SCRIPT_DIR/find_thread_checkpoint.sh "$THREAD_KEY")"

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "no checkpoint found for thread: $THREAD_KEY" >&2
  exit 1
fi

cat "$CHECKPOINT_PATH"
