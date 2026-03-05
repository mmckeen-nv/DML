#!/usr/bin/env bash
set -euo pipefail

WS="/home/nvidia/.openclaw/workspace"
VENV="$WS/.venv-dmlgpu"
DML_PY="$WS/skills/daystrom-dml/scripts/dml_memory.py"
STORE="$WS/data/dml-gpu-prod-smoke"

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

echo "[2/5] Embedding on CUDA"
python3 - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
v = m.encode('prod readiness embedding check', normalize_embeddings=True)
print('device', m.device)
print('dim', len(v))
PY

echo "[3/5] DML ingest smoke"
python3 "$DML_PY" --storage-dir "$STORE" ingest \
  --text "prod-readiness: usd fallback path validated with ground-truth sidecar" \
  --kind note \
  --meta '{"phase":"execute","tool":"prod-readiness-check"}' >/tmp/dml_prod_ingest.json
cat /tmp/dml_prod_ingest.json

echo "[4/5] DML retrieve smoke"
python3 "$DML_PY" --storage-dir "$STORE" retrieve \
  --query "usd fallback path" \
  --top-k 4 \
  --tenant-id openclaw >/tmp/dml_prod_retrieve.json
python3 - <<'PY'
import json
obj=json.load(open('/tmp/dml_prod_retrieve.json'))
print('confidence', obj.get('memory_confidence'))
print('gt_triggered', obj.get('ground_truth_triggered'))
print('has_gt', 'ground_truth' in obj)
print('context_tokens', obj.get('context_tokens'))
PY

echo "[5/5] Confidence-gated fallback smoke"
python3 "$DML_PY" --storage-dir "$STORE" retrieve \
  --query "qzv plum narf unknown context zero" \
  --top-k 3 \
  --tenant-id openclaw \
  --ground-truth-policy always >/tmp/dml_prod_gate.json
python3 - <<'PY'
import json
obj=json.load(open('/tmp/dml_prod_gate.json'))
assert obj.get('ground_truth_triggered') is True, 'expected ground-truth trigger'
assert 'ground_truth' in obj, 'expected ground_truth payload'
print('low_confidence', obj.get('memory_confidence'))
print('reformed_chunks', obj.get('memory_reformed_chunks'))
PY

echo "PASS: production readiness checks completed"
