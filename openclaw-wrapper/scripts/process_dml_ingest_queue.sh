#!/usr/bin/env bash
set -euo pipefail

QUEUE_PATH="${1:-/home/nvidia/.openclaw/workspace/out/dml_ingest_queue.jsonl}"
DAYSTROM_DML_HOME="${DAYSTROM_DML_HOME:-/home/nvidia/.openclaw/daystrom-dml-v2}"
STORAGE_DIR="${2:-$DAYSTROM_DML_HOME/data}"
DML_SCRIPT="/home/nvidia/.openclaw/workspace/skills/daystrom-dml/scripts/dml_memory.py"
DML_CONFIG_PATH="${DML_CONFIG_PATH:-$DAYSTROM_DML_HOME/openclaw-wrapper/config/dml_gpu_only.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INGEST_TIMEOUT_SECONDS="${INGEST_TIMEOUT_SECONDS:-180}"

summarize_checkpoint() {
  local checkpoint_path="$1"
  local thread updated_at state task note intent selected_path next_action captured_at capture_mode capture_contract

  thread="$(grep -m1 '^- thread:' "$checkpoint_path" | sed 's/^- thread: //')"
  updated_at="$(grep -m1 '^- updated_at:' "$checkpoint_path" | sed 's/^- updated_at: //')"
  state="$(grep -m1 '^state:' "$checkpoint_path" | sed 's/^state: //')"
  task="$(grep -m1 '^task:' "$checkpoint_path" | sed 's/^task: //')"
  note="$(grep -m1 '^note:' "$checkpoint_path" | sed 's/^note: //')"
  intent="$(grep -m1 '^intent:' "$checkpoint_path" | sed 's/^intent: //')"
  selected_path="$(grep -m1 '^selected_path:' "$checkpoint_path" | sed 's/^selected_path: //')"
  next_action="$(grep -m1 '^next_action:' "$checkpoint_path" | sed 's/^next_action: //')"
  captured_at="$(grep -m1 '^captured_at:' "$checkpoint_path" | sed 's/^captured_at: //')"
  capture_mode="$(grep -m1 '^capture_mode:' "$checkpoint_path" | sed 's/^capture_mode: //')"
  capture_contract="$(grep -m1 '^capture_contract:' "$checkpoint_path" | sed 's/^capture_contract: //')"

  cat <<EOF
[source:rolling_thread_checkpoint]
thread: ${thread:-unknown}
updated_at: ${updated_at:-unknown}
state: ${state:-unknown}
task: ${task:-unknown}
note: ${note:-none}
intent: ${intent:-none}
selected_path: ${selected_path:-none}
next_action: ${next_action:-none}
captured_at: ${captured_at:-unknown}
capture_mode: ${capture_mode:-rolling_thread_stash}
capture_contract: ${capture_contract:-default_everything_should_be_stashed}
EOF
}

summarize_missing_checkpoint_tombstone() {
  local checkpoint_path="$1"
  local queued_at="$2"
  local detected_at="$3"
  local checkpoint_basename checkpoint_date checkpoint_thread

  checkpoint_basename="$(basename "$checkpoint_path")"
  checkpoint_date="${checkpoint_basename%%_*}"
  checkpoint_thread="${checkpoint_basename#*_}"
  checkpoint_thread="${checkpoint_thread%.md}"

  cat <<EOF
[source:rolling_thread_checkpoint_tombstone]
reference_state: stale_missing_checkpoint
checkpoint_path: $checkpoint_path
checkpoint_file: $checkpoint_basename
checkpoint_date: ${checkpoint_date:-unknown}
thread: ${checkpoint_thread:-unknown}
queued_at: ${queued_at:-unknown}
detected_missing_at: ${detected_at:-unknown}
action: tombstoned_before_prune
cleanup_policy: non_destructive
continuity_signal: preserved_reference_without_checkpoint_body
operator_followup: inspect queue_and_registry_surfaces_for_stale_reference_repair
EOF
}

if [ ! -f "$QUEUE_PATH" ]; then
  echo "queue not found: $QUEUE_PATH" >&2
  exit 1
fi

TMP_OUT="${QUEUE_PATH}.tmp"
: > "$TMP_OUT"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  checkpointPath="$(printf '%s' "$line" | jq -r '.checkpointPath')"
  status="$(printf '%s' "$line" | jq -r '.status // "queued"')"

  if [ "$status" != "queued" ]; then
    printf '%s\n' "$line" >> "$TMP_OUT"
    continue
  fi

  if [ ! -f "$checkpointPath" ]; then
    queuedAt="$(printf '%s' "$line" | jq -r '.queuedAt // "unknown"')"
    detectedAt="$(date -u +%FT%TZ)"
    tombstoneContent="$(summarize_missing_checkpoint_tombstone "$checkpointPath" "$queuedAt" "$detectedAt")"
    printf '%s' "$line" | jq -c \
      --arg status "missing" \
      --arg message "missing_checkpoint_tombstone" \
      --arg detectedAt "$detectedAt" \
      --arg tombstoneContent "$tombstoneContent" \
      '.status = $status
       | .processedAt = now
       | .message = $message
       | .missingCheckpoint = {
           state: "stale_missing_checkpoint",
           detectedAt: $detectedAt,
           cleanupPolicy: "non_destructive",
           continuitySignal: "preserved_reference_without_checkpoint_body",
           tombstoneSummary: $tombstoneContent
         }' >> "$TMP_OUT"
    printf '\n' >> "$TMP_OUT"
    continue
  fi

  textContent="$(summarize_checkpoint "$checkpointPath")"
  itemResult="ok"
  itemMessage="processed"
  if timeout "${INGEST_TIMEOUT_SECONDS}s" "$PYTHON_BIN" "$DML_SCRIPT" --config-path "$DML_CONFIG_PATH" --require-gpu --storage-dir "$STORAGE_DIR" ingest --text "$textContent"; then
    itemResult="done"
    itemMessage="processed_summary"
  else
    exitCode="$?"
    itemResult="error"
    if [ "$exitCode" = "124" ]; then
      itemMessage="timeout"
    else
      itemMessage="ingest_failed"
    fi
  fi

  printf '%s' "$line" | jq -c --arg status "$itemResult" --arg message "$itemMessage" '.status = $status | .processedAt = now | .message = $message' >> "$TMP_OUT"
  printf '\n' >> "$TMP_OUT"
done < "$QUEUE_PATH"

mv "$TMP_OUT" "$QUEUE_PATH"
