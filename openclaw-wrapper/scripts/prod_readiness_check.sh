#!/usr/bin/env bash
set -euo pipefail

WS="/Users/markmckeen/.openclaw/workspace"
DAYSTROM_DML_HOME="${DAYSTROM_DML_HOME:-/Users/markmckeen/.openclaw/daystrom-dml-v2}"
VENV="${DAYSTROM_DML_VENV:-$DAYSTROM_DML_HOME/.venv-dml}"
DML_PY="$WS/skills/daystrom-dml/scripts/dml_memory.py"
STORE="${DML_STORE:-$DAYSTROM_DML_HOME/data-prod-smoke}"
TENANT_ID="${TENANT_ID:-openclaw}"

TMP_DIR="$(mktemp -d /tmp/dml-prod-readiness.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "FAIL: missing venv at $VENV" >&2
  exit 1
fi

source "$VENV/bin/activate"

echo "[1/5] CUDA + torch"
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0))
PY

echo "[2/5] Ollama embedding smoke"
python3 - <<'PY'
import json
import urllib.request
payload = json.dumps({
    'model': 'qwen3-embedding:0.6b',
    'prompt': 'prod readiness embedding check'
}).encode('utf-8')
req = urllib.request.Request(
    'http://127.0.0.1:11434/api/embeddings',
    data=payload,
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    body = json.loads(resp.read().decode('utf-8'))
vec = body.get('embedding') or []
assert vec, 'expected non-empty embedding vector from Ollama'
print('model', body.get('model'))
print('dim', len(vec))
PY

echo "[3/5] DML ingest smoke"
python3 "$DML_PY" --storage-dir "$STORE" ingest \
  --text "prod-readiness: usd fallback path validated with ground-truth sidecar" \
  --kind note \
  --meta '{"phase":"execute","tool":"prod-readiness-check"}' >"$TMP_DIR/ingest.json"
cat "$TMP_DIR/ingest.json"

echo "[4/5] DML retrieve smoke"
python3 "$DML_PY" --storage-dir "$STORE" retrieve \
  --query "usd fallback path" \
  --top-k 4 \
  --tenant-id "$TENANT_ID" >"$TMP_DIR/retrieve.json"
python3 - "$TMP_DIR/retrieve.json" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1]))
assert isinstance(obj.get('items'), list), 'expected items list in retrieval payload'
assert obj.get('context_tokens', 0) > 0, 'expected non-zero context_tokens'
print('tenant_id', obj.get('tenant_id'))
print('confidence', obj.get('memory_confidence'))
print('gt_triggered', obj.get('ground_truth_triggered'))
print('has_gt', 'ground_truth' in obj)
print('context_tokens', obj.get('context_tokens'))
PY

echo "[5/5] Confidence-gated fallback smoke"
python3 "$DML_PY" --storage-dir "$STORE" retrieve \
  --query "qzv plum narf unknown context zero" \
  --top-k 3 \
  --tenant-id "$TENANT_ID" \
  --ground-truth-policy always >"$TMP_DIR/gate.json"
python3 - "$TMP_DIR/gate.json" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1]))
assert obj.get('ground_truth_triggered') is True, 'expected ground-truth trigger'
assert 'ground_truth' in obj, 'expected ground_truth payload'
assert obj.get('memory_confidence', 1.0) <= 0.6, 'expected low-confidence retrieval in fallback test'
print('low_confidence', obj.get('memory_confidence'))
print('reformed_chunks', obj.get('memory_reformed_chunks'))
PY

echo "PASS: production readiness checks completed"
