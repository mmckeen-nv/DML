#!/usr/bin/env bash
set -euo pipefail
WS="/Users/markmckeen/.openclaw/workspace"
OUT="/opt/homebrew/lib/node_modules/openclaw/dist/control-ui/assets/dml-savings.json"
source "$WS/.venv-dmlgpu/bin/activate"
TMP=$(mktemp)
rm -rf "$WS/data/dml-gpu-benchmark-dashboard"
python3 "$WS/skills/daystrom-dml/scripts/benchmark_openclaw_memory.py" \
  --config-path "$WS/skills/daystrom-dml/config/dml_gpu_dashboard.yaml" \
  --storage-dir "$WS/data/dml-gpu-benchmark-dashboard" \
  --top-k 6 \
  --input-glob 'skills/vlm-battlebot/**/*.md' \
  --input-glob 'skills/vlm-battlebot/**/*.json' \
  --input-glob 'skills/vlm-battlebot/**/*.py' \
  --input-glob 'agentic-framework/runs/*.json' \
  --query 'How do I export USD and what is fallback path?' \
  --query 'Summarize anti-blob chassis constraints and primitive stack rules.' \
  --query 'What wheel layout and weapon mount choices were made for the battlebot?' > "$TMP"
python3 - <<'PY' "$TMP" "$OUT"
import json,sys,time,os
raw=open(sys.argv[1]).read()
start=raw.find('{')
obj=json.loads(raw[start:])
baseline=float(obj.get('baseline_tokens') or 0)
avg_ctx=float(obj.get('avg_dml_tokens') or 0)
avoided=max(0.0, baseline-avg_ctx)
price_per_1m=float(os.getenv('OPENCLAW_DML_INPUT_PRICE_PER_1M','2.5'))
est_usd=(avoided/1_000_000.0)*price_per_1m

prev={}
out_path=sys.argv[2]
try:
  prev=json.loads(open(out_path).read())
except Exception:
  prev={}
prev_lifetime=float(prev.get('lifetimeTokensSavedEstimate') or 0.0)
prev_samples=int(prev.get('lifetimeSamples') or 0)

lifetime_tokens=prev_lifetime+avoided
lifetime_samples=prev_samples+1
lifetime_usd=(lifetime_tokens/1_000_000.0)*price_per_1m

out={
  'updatedAt': time.strftime('%Y-%m-%d %H:%M:%S'),
  'avgTokenSavingsPct': obj.get('avg_token_savings_pct'),
  'avgLatencyMs': obj.get('avg_latency_ms'),
  'avgMemoryConfidence': round((obj.get('avg_precision_at_k',0)+obj.get('avg_ndcg_at_k',0))/2,3),
  'baselineTokensEstimate': obj.get('baseline_tokens'),
  'avgContextTokens': obj.get('avg_dml_tokens'),
  'avgTokensAvoided': round(avoided,2),
  'lifetimeTokensSavedEstimate': round(lifetime_tokens,2),
  'lifetimeSamples': lifetime_samples,
  'lifetimeUsdSavedEstimate': round(lifetime_usd,6),
  'pricing': {
    'inputPricePer1M': price_per_1m,
    'estimatedUsdSavedPerQuery': round(est_usd,6),
  }
}
open(out_path,'w').write(json.dumps(out,indent=2))
print('wrote',out_path)
PY
rm -f "$TMP"
