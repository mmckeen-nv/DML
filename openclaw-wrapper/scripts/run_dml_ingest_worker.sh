#!/usr/bin/env bash
set -euo pipefail

OPENCLAW_HOME="${OPENCLAW_HOME:-/Users/markmckeen/.openclaw}"
WORKSPACE="${OPENCLAW_WORKSPACE:-$OPENCLAW_HOME/workspace}"
QUEUE_PATH="${QUEUE_PATH:-$WORKSPACE/out/dml_ingest_queue.jsonl}"
LOG_PATH="${LOG_PATH:-$WORKSPACE/out/dml_ingest_worker.log}"
LOCK_PATH="${LOCK_PATH:-$WORKSPACE/out/dml_ingest_worker.lock}"
STATUS_PATH="${STATUS_PATH:-$WORKSPACE/out/continuity-ingest-status.json}"
PROCESS_SCRIPT="${PROCESS_SCRIPT:-$WORKSPACE/skills/daystrom-dml/scripts/process_dml_ingest_queue.sh}"
KEEP_WARM_SCRIPT="${KEEP_WARM_SCRIPT:-$WORKSPACE/skills/daystrom-dml/scripts/keep_dml_models_warm.sh}"
STORAGE_DIR="${STORAGE_DIR:-${DML_STORE:-$OPENCLAW_HOME/dml-store}}"

count_status() {
  local wanted="$1"
  local path="$2"
  [ -f "$path" ] || { echo 0; return; }
  jq -Rs --arg wanted "$wanted" 'split("\n") | map(select(length > 0) | fromjson? | select(.status == $wanted)) | length' "$path"
}

count_total() {
  local path="$1"
  [ -f "$path" ] || { echo 0; return; }
  jq -Rs 'split("\n") | map(select(length > 0) | fromjson?) | length' "$path"
}

queue_oldest_epoch() {
  local path="$1"
  [ -f "$path" ] || { echo null; return; }
  jq -Rs '[split("\n") | map(select(length > 0) | fromjson? | select(.status == "queued") | (.queuedAt // empty) | fromdateiso8601?)] | min // null' "$path"
}

mkdir -p "$WORKSPACE/out"
RUN_OUTCOME_CLASS="unknown"
ERROR_STAGE="none"
BEFORE_TOTAL="$(count_total "$QUEUE_PATH")"
BEFORE_QUEUED="$(count_status queued "$QUEUE_PATH")"
BEFORE_DONE="$(count_status done "$QUEUE_PATH")"
BEFORE_ERROR="$(count_status error "$QUEUE_PATH")"
BEFORE_MISSING="$(count_status missing "$QUEUE_PATH")"
BEFORE_OLDEST_QUEUED_EPOCH="$(queue_oldest_epoch "$QUEUE_PATH")"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  HOLDER_PIDS="$( (command -v fuser >/dev/null && fuser "$LOCK_PATH" 2>/dev/null) || true )"
  HOLDER_PIDS="$(printf '%s' "$HOLDER_PIDS" | xargs 2>/dev/null || true)"
  HOLDER_CMD="none"
  HOLDER_ELAPSED=""
  HOLDER_STAT=""
  HOLDER_CHILD_CMD=""
  HOLDER_CHILD_ELAPSED=""
  HOLDER_CHILD_STAT=""
  LOCK_BLOCKER_KIND="active-holder"
  if [ -n "$HOLDER_PIDS" ]; then
    HOLDER_PID="$(printf '%s' "$HOLDER_PIDS" | awk '{print $1}')"
    HOLDER_CMD="$(ps -o args= -p "$HOLDER_PID" 2>/dev/null | head -n 1)"
    HOLDER_CMD="${HOLDER_CMD:-none}"
    HOLDER_ELAPSED="$(ps -o etimes= -p "$HOLDER_PID" 2>/dev/null | awk '{$1=$1; print}' | head -n 1)"
    HOLDER_STAT="$(ps -o stat= -p "$HOLDER_PID" 2>/dev/null | awk '{$1=$1; print}' | head -n 1)"
    HOLDER_CHILD_PID="$(pgrep -P "$HOLDER_PID" | head -n 1 || true)"
    if [ -n "$HOLDER_CHILD_PID" ]; then
      HOLDER_CHILD_CMD="$(ps -o args= -p "$HOLDER_CHILD_PID" 2>/dev/null | head -n 1)"
      HOLDER_CHILD_ELAPSED="$(ps -o etimes= -p "$HOLDER_CHILD_PID" 2>/dev/null | awk '{$1=$1; print}' | head -n 1)"
      HOLDER_CHILD_STAT="$(ps -o stat= -p "$HOLDER_CHILD_PID" 2>/dev/null | awk '{$1=$1; print}' | head -n 1)"
      case "$HOLDER_CHILD_CMD" in
        *sync-dml-general.py*) LOCK_BLOCKER_KIND="legacy-general-sync" ;;
      esac
    fi
  fi
  RUN_OUTCOME_CLASS="blocked"
  ERROR_STAGE="lock-acquire"
  jq -n \
    --arg ts "$(date -u +%FT%TZ)" \
    --arg queue_path "$QUEUE_PATH" \
    --arg log_path "$LOG_PATH" \
    --arg lock_path "$LOCK_PATH" \
    --arg process_script "$PROCESS_SCRIPT" \
    --arg run_outcome_class "$RUN_OUTCOME_CLASS" \
    --arg error_stage "$ERROR_STAGE" \
    --arg holder_pids "$HOLDER_PIDS" \
    --arg holder_cmd "$HOLDER_CMD" \
    --arg holder_elapsed "$HOLDER_ELAPSED" \
    --arg holder_stat "$HOLDER_STAT" \
    --arg holder_child_cmd "$HOLDER_CHILD_CMD" \
    --arg holder_child_elapsed "$HOLDER_CHILD_ELAPSED" \
    --arg holder_child_stat "$HOLDER_CHILD_STAT" \
    --arg blocker_kind "$LOCK_BLOCKER_KIND" \
    --argjson before_total "$BEFORE_TOTAL" \
    --argjson before_queued "$BEFORE_QUEUED" \
    --argjson before_done "$BEFORE_DONE" \
    --argjson before_error "$BEFORE_ERROR" \
    --argjson before_missing "$BEFORE_MISSING" \
    --argjson oldest_queued_epoch "$BEFORE_OLDEST_QUEUED_EPOCH" \
    '{
      ts: $ts,
      queue_path: $queue_path,
      log_path: $log_path,
      lock_path: $lock_path,
      process_script: $process_script,
      state: "lock-held",
      run_outcome_class: $run_outcome_class,
      error_stage: $error_stage,
      blocker_kind: $blocker_kind,
      message: "worker already running",
      holder: {
        pids: ($holder_pids | if length == 0 then [] else split(" ") end),
        command: $holder_cmd,
        elapsed_seconds: ($holder_elapsed | if length == 0 then null else tonumber end),
        stat: ($holder_stat | if length == 0 then null else . end),
        child: {
          command: ($holder_child_cmd | if length == 0 then null else . end),
          elapsed_seconds: ($holder_child_elapsed | if length == 0 then null else tonumber end),
          stat: ($holder_child_stat | if length == 0 then null else . end)
        }
      },
      queue: {
        oldest_queued_epoch: $oldest_queued_epoch,
        oldest_queued_iso: ($oldest_queued_epoch | if (type != "number") then null else todateiso8601 end)
      },
      counts: {
        before: {
          total: $before_total,
          queued: $before_queued,
          done: $before_done,
          error: $before_error,
          missing: $before_missing
        },
        after: {
          total: $before_total,
          queued: $before_queued,
          done: $before_done,
          error: $before_error,
          missing: $before_missing
        }
      }
    }' > "$STATUS_PATH"
  echo "[$(date -u +%FT%TZ)] lock-held: worker already running blocker_kind=$LOCK_BLOCKER_KIND pids=[$HOLDER_PIDS] cmd=$HOLDER_CMD child=${HOLDER_CHILD_CMD:-none}" >> "$LOG_PATH"
  echo "$STATUS_PATH"
  exit 0
fi

RUN_STATE="ok"
RUN_MESSAGE="processed queue"
RUN_OUTCOME_CLASS="success"
ERROR_STAGE="none"
if [ -x "$KEEP_WARM_SCRIPT" ]; then
  if ! "$KEEP_WARM_SCRIPT" >> "$LOG_PATH" 2>&1; then
    echo "[$(date -u +%FT%TZ)] keep-warm warning: helper failed before queue pass" >> "$LOG_PATH"
  fi
fi
if [ ! -f "$QUEUE_PATH" ]; then
  RUN_STATE="no-queue"
  RUN_MESSAGE="queue file missing"
  RUN_OUTCOME_CLASS="degraded"
  ERROR_STAGE="queue-check"
else
  if ! "$PROCESS_SCRIPT" "$QUEUE_PATH" "$STORAGE_DIR" >> "$LOG_PATH" 2>&1; then
    RUN_STATE="error"
    RUN_MESSAGE="queue processor failed"
    RUN_OUTCOME_CLASS="error"
    ERROR_STAGE="queue-process"
  fi
fi

AFTER_TOTAL="$(count_total "$QUEUE_PATH")"
AFTER_QUEUED="$(count_status queued "$QUEUE_PATH")"
AFTER_DONE="$(count_status done "$QUEUE_PATH")"
AFTER_ERROR="$(count_status error "$QUEUE_PATH")"
AFTER_MISSING="$(count_status missing "$QUEUE_PATH")"

AFTER_OLDEST_QUEUED_EPOCH="$(queue_oldest_epoch "$QUEUE_PATH")"

jq -n \
  --arg ts "$(date -u +%FT%TZ)" \
  --arg queue_path "$QUEUE_PATH" \
  --arg log_path "$LOG_PATH" \
  --arg lock_path "$LOCK_PATH" \
  --arg process_script "$PROCESS_SCRIPT" \
  --arg state "$RUN_STATE" \
  --arg message "$RUN_MESSAGE" \
  --arg run_outcome_class "$RUN_OUTCOME_CLASS" \
  --arg error_stage "$ERROR_STAGE" \
  --argjson before_total "$BEFORE_TOTAL" \
  --argjson before_queued "$BEFORE_QUEUED" \
  --argjson before_done "$BEFORE_DONE" \
  --argjson before_error "$BEFORE_ERROR" \
  --argjson before_missing "$BEFORE_MISSING" \
  --argjson after_total "$AFTER_TOTAL" \
  --argjson after_queued "$AFTER_QUEUED" \
  --argjson after_done "$AFTER_DONE" \
  --argjson after_error "$AFTER_ERROR" \
  --argjson after_missing "$AFTER_MISSING" \
  --argjson before_oldest_queued_epoch "$BEFORE_OLDEST_QUEUED_EPOCH" \
  --argjson after_oldest_queued_epoch "$AFTER_OLDEST_QUEUED_EPOCH" \
  '{
    ts: $ts,
    queue_path: $queue_path,
    log_path: $log_path,
    lock_path: $lock_path,
    process_script: $process_script,
    state: $state,
    message: $message,
    run_outcome_class: $run_outcome_class,
    error_stage: $error_stage,
    queue: {
      oldest_queued_before_epoch: $before_oldest_queued_epoch,
      oldest_queued_before_iso: ($before_oldest_queued_epoch | if (type != "number") then null else todateiso8601 end),
      oldest_queued_after_epoch: $after_oldest_queued_epoch,
      oldest_queued_after_iso: ($after_oldest_queued_epoch | if (type != "number") then null else todateiso8601 end)
    },
    counts: {
      before: {
        total: $before_total,
        queued: $before_queued,
        done: $before_done,
        error: $before_error,
        missing: $before_missing
      },
      after: {
        total: $after_total,
        queued: $after_queued,
        done: $after_done,
        error: $after_error,
        missing: $after_missing
      }
    }
  }' > "$STATUS_PATH"

echo "$STATUS_PATH"
