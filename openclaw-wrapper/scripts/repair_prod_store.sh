#!/usr/bin/env bash
set -euo pipefail

WS="/Users/markmckeen/.openclaw/workspace"
VENV="$WS/.venv-dmlgpu"
STORE="$WS/data/dml-gpu-prod"
CLI="$WS/skills/daystrom-dml/scripts/dml_memory.py"

source "$VENV/bin/activate"

echo "[repair] resetting store: $STORE"
rm -rf "$STORE"
mkdir -p "$STORE"

FILES=(
  "$WS/skills/vlm-battlebot/SKILL.md"
  "$WS/skills/vlm-battlebot/references/spec.schema.json"
  "$WS/skills/vlm-battlebot/references/vanguard_reference_example.json"
  "$WS/skills/vlm-battlebot/scripts/generate_battlebot.py"
  "$WS/skills/daystrom-dml/SKILL.md"
  "$WS/skills/daystrom-dml/DEPLOY_PROD.md"
  "$WS/agentic-framework/ORCHESTRATION.md"
  "$WS/out/four_bot_battle_run.log"
  "$WS/lobsterbot_v2_training_run_20260303_231806.log"
)

count=0
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  text=$(python3 - <<'PY' "$f"
from pathlib import Path
import sys
p=Path(sys.argv[1])
print(p.read_text(errors='ignore')[:250000])
PY
)
  python3 "$CLI" --storage-dir "$STORE" ingest \
    --text "$text" \
    --kind note \
    --meta "{\"source_file\":\"$f\",\"phase\":\"execute\",\"tool\":\"repair_prod_store\"}" >/dev/null
  count=$((count+1))
done

echo "[repair] ingested files: $count"

echo "[repair] sanity retrieve"
python3 "$CLI" --storage-dir "$STORE" retrieve \
  --query "What is the USD fallback path and anti-blob constraints?" \
  --top-k 6 \
  --tenant-id openclaw >/tmp/dml_prod_repair_verify.json
python3 - <<'PY'
import json
j=json.load(open('/tmp/dml_prod_repair_verify.json'))
print('confidence', j.get('memory_confidence'))
print('tokens', j.get('context_tokens'))
print('gt_triggered', j.get('ground_truth_triggered'))
print('status', 'ok')
PY

echo "[repair] done"
