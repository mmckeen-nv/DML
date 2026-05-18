#!/usr/bin/env bash
set -euo pipefail

THREAD_KEY="${1:?thread key required}"
SUMMARY_TEXT="${2:?summary text required}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${DML_CHECKPOINT_DIR:-$WORKSPACE/out/dml-checkpoints}"
THREAD_SAFE_KEY="$(python3 "$(dirname "$0")/thread_safe_key.py" "$THREAD_KEY")"
OUT_PATH="$OUT_DIR/${THREAD_SAFE_KEY}.md"
UPDATED_AT_UTC="$(date -u +%FT%TZ)"

mkdir -p "$OUT_DIR"

cat > "$OUT_PATH" <<EOF
# DML Thread Checkpoint

- thread: $THREAD_KEY
- thread_key: $THREAD_KEY
- thread_safe_key: $THREAD_SAFE_KEY
- checkpoint_path: $OUT_PATH
- updated_at: $UPDATED_AT_UTC
- checkpoint_policy: rolling_thread_checkpoint
- provider: ${PROVIDER:-}
- channel: ${CHANNEL:-}
- chat_id: ${CHAT_ID:-}
- topic_id: ${TOPIC_ID:-}
- thread_label: ${THREAD_LABEL:-$THREAD_TOPIC}
- session_scope: ${SESSION_SCOPE:-thread}

## Summary
$SUMMARY_TEXT
EOF

THREAD_ID="${THREAD_ID:-}"
THREAD_TOPIC="${THREAD_TOPIC:-$THREAD_KEY}"
PROVIDER="${PROVIDER:-}"
CHANNEL="${CHANNEL:-}"
CHAT_ID="${CHAT_ID:-}"
TOPIC_ID="${TOPIC_ID:-}"
THREAD_LABEL="${THREAD_LABEL:-$THREAD_TOPIC}"
SESSION_SCOPE="${SESSION_SCOPE:-thread}"
if [ -n "$THREAD_ID" ]; then
  REGISTRY_PATH="${DML_THREAD_REGISTRY:-$WORKSPACE/runtime/continuity/thread_registry.json}"
  mkdir -p "$(dirname "$REGISTRY_PATH")"
  TMP_REGISTRY="$(mktemp)"
  NOW_UTC="$(date -u +%FT%TZ)"

  if [ -f "$REGISTRY_PATH" ]; then
    jq \
      --arg thread_id "$THREAD_ID" \
      --arg thread_key "$THREAD_KEY" \
      --arg thread_safe_key "$THREAD_SAFE_KEY" \
      --arg thread_topic "$THREAD_TOPIC" \
      --arg checkpoint_path "$OUT_PATH" \
      --arg provider "$PROVIDER" \
      --arg channel "$CHANNEL" \
      --arg chat_id "$CHAT_ID" \
      --arg topic_id "$TOPIC_ID" \
      --arg thread_label "$THREAD_LABEL" \
      --arg session_scope "$SESSION_SCOPE" \
      --arg now "$NOW_UTC" \
      '.schema_version = (.schema_version // "v1")
       | .threads = (.threads // {})
       | .threads[$thread_id] = ((.threads[$thread_id] // {}) + {
           key: $thread_key,
           thread_key: $thread_key,
           thread_safe_key: $thread_safe_key,
           topic: $thread_topic,
           provider: $provider,
           channel: $channel,
           chat_id: $chat_id,
           topic_id: $topic_id,
           thread_label: $thread_label,
           session_scope: $session_scope,
           status: "active",
           latest_checkpoint: $checkpoint_path,
           last_summary_at: $now,
           updated_at: $now
         })' "$REGISTRY_PATH" > "$TMP_REGISTRY"
  else
    jq -n \
      --arg thread_id "$THREAD_ID" \
      --arg thread_key "$THREAD_KEY" \
      --arg thread_safe_key "$THREAD_SAFE_KEY" \
      --arg thread_topic "$THREAD_TOPIC" \
      --arg checkpoint_path "$OUT_PATH" \
      --arg provider "$PROVIDER" \
      --arg channel "$CHANNEL" \
      --arg chat_id "$CHAT_ID" \
      --arg topic_id "$TOPIC_ID" \
      --arg thread_label "$THREAD_LABEL" \
      --arg session_scope "$SESSION_SCOPE" \
      --arg now "$NOW_UTC" \
      '{schema_version: "v1", threads: {($thread_id): {key: $thread_key, thread_key: $thread_key, thread_safe_key: $thread_safe_key, topic: $thread_topic, provider: $provider, channel: $channel, chat_id: $chat_id, topic_id: $topic_id, thread_label: $thread_label, session_scope: $session_scope, status: "active", latest_checkpoint: $checkpoint_path, last_summary_at: $now, updated_at: $now}}}' > "$TMP_REGISTRY"
  fi

  mv "$TMP_REGISTRY" "$REGISTRY_PATH"
fi

echo "$OUT_PATH"
