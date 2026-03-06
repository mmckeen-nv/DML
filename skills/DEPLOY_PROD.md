# Daystrom DML Production Deploy Checklist

## 1) Environment
- [ ] GPU venv exists: `/home/nvidia/.openclaw/workspace/.venv-dmlgpu`
- [ ] CUDA torch active in venv (`torch.cuda.is_available() == True`)
- [ ] DML config is GPU-only: `skills/daystrom-dml/config/dml_gpu_only.yaml`

## 2) Architecture defaults
- [ ] Primary retrieval: DML memory
- [ ] Policy: `ground-truth-policy=low-confidence`
- [ ] Sidecar RAG mode: `hybrid`
- [ ] Ground-truth timeout set (default `--ground-truth-timeout-ms 2500`) to cap tail latency
- [ ] Memory reform enabled (`--reform-memory`)
- [ ] Ground-truth strict mode disabled for availability (`--no-strict-ground-truth`) unless required

## 3) Run one-command readiness
```bash
source /home/nvidia/.openclaw/workspace/.venv-dmlgpu/bin/activate
/home/nvidia/.openclaw/workspace/skills/daystrom-dml/scripts/prod_readiness_check.sh
```

Expected: `PASS: production readiness checks completed`

## 4) Runtime usage
```bash
source /home/nvidia/.openclaw/workspace/.venv-dmlgpu/bin/activate
python3 /home/nvidia/.openclaw/workspace/skills/daystrom-dml/scripts/dml_memory.py retrieve \
  --query "How do I export USD and what is fallback path?" \
  --top-k 6 \
  --tenant-id openclaw
```

## 5) Monitoring
- [ ] Watch retrieval confidence trend (`memory_confidence`)
- [ ] Track ground-truth trigger rate (`ground_truth_triggered`)
- [ ] Track reformation rate (`memory_reformed_chunks`)
- [ ] Periodically benchmark token savings + precision/nDCG with `benchmark_openclaw_memory.py`
