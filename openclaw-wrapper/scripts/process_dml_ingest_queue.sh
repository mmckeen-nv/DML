#!/usr/bin/env bash
set -euo pipefail

OPENCLAW_HOME="${OPENCLAW_HOME:-/Users/markmckeen/.openclaw}"
OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-$OPENCLAW_HOME/workspace}"
QUEUE_PATH="${1:-$OPENCLAW_WORKSPACE/out/dml_ingest_queue.jsonl}"
DAYSTROM_DML_HOME="${DAYSTROM_DML_HOME:-$OPENCLAW_HOME/daystrom-dml-v2}"
STORAGE_DIR="${2:-${DML_STORE:-$OPENCLAW_HOME/dml-store}}"
DML_SCRIPT="${DML_SCRIPT:-$OPENCLAW_WORKSPACE/skills/daystrom-dml/scripts/dml_memory.py}"
DML_CONFIG_PATH="${DML_CONFIG_PATH:-$OPENCLAW_WORKSPACE/skills/daystrom-dml/config/dml_portable_linux.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$DAYSTROM_DML_HOME/.venv-dml/bin/python}"
INGEST_TIMEOUT_SECONDS="${INGEST_TIMEOUT_SECONDS:-180}"
DML_REQUIRE_GPU="${DML_REQUIRE_GPU:-0}"
DML_TENANT_ID="${DML_TENANT_ID:-openclaw}"
DML_CLIENT_ID="${DML_CLIENT_ID:-}"
DML_SESSION_ID="${DML_SESSION_ID:-}"
DML_INSTANCE_ID="${DML_INSTANCE_ID:-}"

require_gpu_arg() {
  local normalized
  normalized="$(printf '%s' "$DML_REQUIRE_GPU" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes|on) printf '%s\n' "--require-gpu" ;;
    *) printf '%s\n' "--no-require-gpu" ;;
  esac
}

run_with_optional_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout "${INGEST_TIMEOUT_SECONDS}s" "$@"
  else
    "$@"
  fi
}

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

checkpoint_field() {
  local checkpoint_path="$1"
  local field="$2"
  if [ "$field" = "thread" ]; then
    grep -m1 '^- thread:' "$checkpoint_path" | sed 's/^- thread: //' || true
  elif [ "$field" = "updated_at" ]; then
    grep -m1 '^- updated_at:' "$checkpoint_path" | sed 's/^- updated_at: //' || true
  else
    grep -m1 "^$field:" "$checkpoint_path" | sed "s/^$field: //" || true
  fi
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
  thread="$(checkpoint_field "$checkpointPath" thread)"
  updatedAt="$(checkpoint_field "$checkpointPath" updated_at)"
  state="$(checkpoint_field "$checkpointPath" state)"
  task="$(checkpoint_field "$checkpointPath" task)"
  nextAction="$(checkpoint_field "$checkpointPath" next_action)"
  capturedAt="$(checkpoint_field "$checkpointPath" captured_at)"
  summaryText="thread: ${thread:-unknown} | state: ${state:-unknown} | task: ${task:-unknown} | next: ${nextAction:-none}"
  metaJson="$(jq -nc \
    --arg source "rolling_thread_checkpoint" \
    --arg namespace "active_continuity" \
    --arg memory_state "active" \
    --arg tenant_id "$DML_TENANT_ID" \
    --arg client_id "$DML_CLIENT_ID" \
    --arg session_id "$DML_SESSION_ID" \
    --arg instance_id "$DML_INSTANCE_ID" \
    --arg scope "thread" \
    --arg checkpoint_path "$checkpointPath" \
    --arg continuity_signal "resume_checkpoint" \
    --arg thread "${thread:-unknown}" \
    --arg updated_at "${updatedAt:-unknown}" \
    --arg state "${state:-unknown}" \
    --arg task "${task:-unknown}" \
    --arg next_action "${nextAction:-none}" \
    --arg captured_at "${capturedAt:-unknown}" \
    --arg summary "$summaryText" \
    '{
      source: $source,
      namespace: $namespace,
      memory_state: $memory_state,
      merge_policy: "never",
      no_merge: true,
      tenant_id: $tenant_id,
      client_id: (if $client_id == "" then null else $client_id end),
      session_id: (if $session_id == "" then null else $session_id end),
      instance_id: (if $instance_id == "" then null else $instance_id end),
      scope: $scope,
      checkpoint_path: $checkpoint_path,
      continuity_signal: $continuity_signal,
      thread: $thread,
      updated_at: $updated_at,
      state: $state,
      task: $task,
      next_action: $next_action,
      captured_at: $captured_at,
      summary: $summary,
      summary_source: "deterministic"
    }')"
  itemResult="ok"
  itemMessage="processed"
  cmd=(
    "$PYTHON_BIN" "$DML_SCRIPT"
    --config-path "$DML_CONFIG_PATH"
    "$(require_gpu_arg)"
    --storage-dir "$STORAGE_DIR"
    ingest
    --tenant-id "$DML_TENANT_ID"
    --kind plan
    --meta "$metaJson"
    --text "$textContent"
  )
  [ -z "$DML_CLIENT_ID" ] || cmd+=(--client-id "$DML_CLIENT_ID")
  [ -z "$DML_SESSION_ID" ] || cmd+=(--session-id "$DML_SESSION_ID")
  [ -z "$DML_INSTANCE_ID" ] || cmd+=(--instance-id "$DML_INSTANCE_ID")
  if run_with_optional_timeout "${cmd[@]}"; then
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
