#!/usr/bin/env bash
set -euo pipefail

WS="/Users/markmckeen/.openclaw/workspace"
DAYSTROM_DML_HOME="${DAYSTROM_DML_HOME:-/Users/markmckeen/.openclaw/daystrom-dml-v2}"
DML_SCRIPT="$WS/skills/daystrom-dml/scripts/dml_memory.py"
CONFIG_PATH="${DML_CONFIG_PATH:-$DAYSTROM_DML_HOME/openclaw-wrapper/config/dml_gpu_only.yaml}"
FRESH_STORE_DEFAULT="${DML_FRESH_STORE_DEFAULT:-/tmp/dml-ollama-fresh-store}"
LIVE_STORE_DEFAULT="${DML_LIVE_STORE_DEFAULT:-$DAYSTROM_DML_HOME/data}"
OLLAMA_URL_DEFAULT="http://localhost:11434/api/embeddings"
OLLAMA_MODEL_DEFAULT="qwen3-embedding:0.6b"
OLLAMA_PROMPT_DEFAULT="dml ollama endpoint smoke"
MODE="${1:-all}"

backend_proof_json() {
  local store="$1"
  python3 "$DML_SCRIPT" \
    --config-path "$CONFIG_PATH" \
    --storage-dir "$store" \
    backend-proof
}

fresh_store_smoke() {
  local store="${DML_FRESH_STORE:-$FRESH_STORE_DEFAULT}"
  backend_proof_json "$store"
}

live_store_smoke() {
  local store="${DML_LIVE_STORE:-$LIVE_STORE_DEFAULT}"
  backend_proof_json "$store"
}

endpoint_smoke() {
  local url="${OLLAMA_EMBED_URL:-$OLLAMA_URL_DEFAULT}"
  local model="${OLLAMA_EMBED_MODEL:-$OLLAMA_MODEL_DEFAULT}"
  local prompt="${OLLAMA_EMBED_PROMPT:-$OLLAMA_PROMPT_DEFAULT}"
  local response
  response="$(curl -fsS "$url" -H 'Content-Type: application/json' -d "$(python3 - <<PY
import json
print(json.dumps({"model": "$model", "prompt": "$prompt"}))
PY
)")"
  RESPONSE_JSON="$response" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["RESPONSE_JSON"])
embedding = payload.get("embedding")
if not isinstance(embedding, list) or not embedding:
    raise SystemExit("endpoint smoke failed: embeddings payload missing non-empty embedding")
print(json.dumps({
    "status": "ok",
    "action": "endpoint-smoke",
    "embedder_backend": "ollama",
    "embedding_length": len(embedding),
    "ollama_model_name": payload.get("model"),
}, indent=2))
PY
}

case "$MODE" in
  fresh-store)
    fresh_store_smoke
    ;;
  live-store)
    live_store_smoke
    ;;
  endpoint)
    endpoint_smoke
    ;;
  backend-proof)
    fresh="$(fresh_store_smoke)"
    live="$(live_store_smoke)"
    FRESH_JSON="$fresh" LIVE_JSON="$live" python3 - <<'PY'
import json
import os

print(json.dumps({
    "status": "ok",
    "action": "backend-proof",
    "fresh_store": json.loads(os.environ["FRESH_JSON"]),
    "live_store": json.loads(os.environ["LIVE_JSON"]),
}, indent=2))
PY
    ;;
  contract-matrix)
    fresh="$(fresh_store_smoke)"
    live="$(live_store_smoke)"
    endpoint="$(endpoint_smoke)"
    FRESH_JSON="$fresh" LIVE_JSON="$live" ENDPOINT_JSON="$endpoint" python3 - <<'PY'
import json
import os

fresh = json.loads(os.environ["FRESH_JSON"])
live = json.loads(os.environ["LIVE_JSON"])
endpoint = json.loads(os.environ["ENDPOINT_JSON"])

print(json.dumps({
    "status": "ok",
    "action": "contract-matrix",
    "matrix_version": "ollama-gpu-only-v1",
    "fresh_store": {
        "expected": {
            "embedder_backend": "ollama",
            "embedder_ready": True,
            "embedder_target_device": "ollama-managed",
            "embedding_device_cfg": "cuda",
        },
        "observed": fresh,
    },
    "live_store": {
        "expected": {
            "embedder_backend": "ollama",
            "embedder_ready": True,
            "embedder_target_device": "ollama-managed",
            "embedding_device_cfg": "cuda",
        },
        "observed": live,
    },
    "endpoint": {
        "expected": {
            "embedder_backend": "ollama",
            "embedding_length_min": 1,
        },
        "observed": endpoint,
    },
}, indent=2))
PY
    ;;
  all)
    fresh="$(fresh_store_smoke)"
    live="$(live_store_smoke)"
    endpoint="$(endpoint_smoke)"
    FRESH_JSON="$fresh" LIVE_JSON="$live" ENDPOINT_JSON="$endpoint" python3 - <<'PY'
import json
import os

print(json.dumps({
    "status": "ok",
    "action": "dml-ollama-smoke",
    "fresh_store": json.loads(os.environ["FRESH_JSON"]),
    "live_store": json.loads(os.environ["LIVE_JSON"]),
    "endpoint": json.loads(os.environ["ENDPOINT_JSON"]),
}, indent=2))
PY
    ;;
  *)
    echo "usage: $0 [fresh-store|live-store|endpoint|backend-proof|contract-matrix|all]" >&2
    exit 2
    ;;
esac
