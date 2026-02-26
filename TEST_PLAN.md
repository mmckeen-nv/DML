# DML Test Plan - Validation Framework

## 0) Success Criteria

**When DML is production-ready, you must show:**

✅ Higher task success rate at same token budget (15-25% improvement)
✅ Fewer repeated mistakes / retries after first failure (30% reduction)
✅ Lower wall-clock time or fewer tool calls for same task (-10 to -20%)
✅ Memory quality: relevant recalls, low hallucinated "memories"
✅ Stable behavior over long loops (no memory blow-up / drift)

## 1) Test Matrix (10-30 seeds per workload)

### Modes:
1. **Baseline:** OpenClaw with memory off
2. **RAG-only:** persistent store only (no agentic mode/router)
3. **DML Agentic (router off):** agentic schema + promotion, fixed settings
4. **DML Agentic (router on):** full router profiles + phase modifiers

### Variations:
- **Token budget:** 300 / 500 / 800
- **Top K:** 4 / 8 / 12
- **Similarity threshold:** 0.25 / 0.4 / 0.6 (phase dependent)

## 2) Workloads

### A) DevOps / Docker cluster task
**Goal:** Deploy something multi-step, verify it's running, handle one induced fault
**Signals:**
- ✅ Correct container up
- ✅ Correct port open
- ✅ Logs show healthy
- ✅ Recovers after a forced restart

### B) Coding task
**Goal:** Implement a feature, run tests, fix failing test, commit
**Signals:**
- ✅ Tests go from failing → passing
- ✅ Diff matches requested change
- ✅ No regressions

### C) Debug task (best for proving memory value)
**Goal:** Reproduce a bug, narrow root cause, apply fix, confirm
**Induce:** Change env var, break config, or version mismatch
**Signals:**
- ✅ Identifies correct failing component
- ✅ Doesn't repeat same dead-end after first failure

### D) Long-horizon "episode" task
**Goal:** 45-90 minutes of mixed plan/build/execute/debug, with interruptions
**Signals:**
- ✅ Maintains plan continuity
- ✅ Uses past observations
- ✅ Doesn't thrash token budget

## 3) Memory Instrumentation (MUST capture)

For every agent step, log:

```python
{
  "phase": "plan|build|execute|debug|reflect",
  "kinds_written": ["action", "observation", ...],
  "router_decision": {
    "task_profile": "...",
    "chosen_threshold": ...,
    "top_k": ...,
    "token_budget": ...
  },
  "retrieval_report": {
    "context_tokens": ...,
    "top_k": ...,
    "items": [{"text": "...", "score": ...}, ...]
  },
  "promotion_pipeline": {
    "accepted": [...],
    "rejected": [...],
    "reasons": [...]
  }
}
```

## 4) Promotion Tests (quality gate)

**Test "fail closed" behavior:**

1. Feed in intentionally bad memories:
   - Missing provenance
   - Wrong schema
   - Malformed JSON

2. Ensure they're rejected under strict mode (>95% rejection rate)

3. Confirm accepted memories correlate with later success (not noise)

## 5) Retrieval Correctness Tests

For each workload, create 20-50 "gold" queries:

```python
[
  "What port did we expose for X?",
  "What was the last observed error?",
  "Which env var fixed the issue?",
  ...
]
```

**Score metrics:**
- **Recall@K:** Did it retrieve the right item?
- **Precision@K:** How much junk came along?
- **Latency:** Retrieval time

## 6) GPU Acceleration Validation

**Prove GPU acceleration works:**

1. **Embeddings device is CUDA:**
   ```python
   model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
   print(model.device)  # Should be cuda:0
   ```

2. **FAISS index queries are on GPU:**
   ```python
   gpu_resources = faiss.StandardGpuResources()
   index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
   ```

3. **LLM backend is on GPU:**
   ```python
   model = AutoModelForCausalLM.from_pretrained('gpt2', device_map="auto")
   print(next(model.parameters()).device)  # Should be cuda:0
   ```

**Log metrics:**
- Embedding latency
- Retrieval latency
- End-to-end step latency

## 7) Minimal Harness

DML includes an eval harness (simulation mode), but it's not tool-integrated.

**Real test harness should:**
1. Wrap OpenClaw runs
2. Collect instrumentation above
3. Run with different seeds (10-30)
4. Compare modes (baseline vs DML)

## 8) Pass/Fail Thresholds (Suggested)

### For each workload:

| Metric | Target |
|--------|--------|
| **Success rate** | +15-25% over baseline or RAG-only |
| **Tool calls** | -10-20% (or same, but higher success) |
| **Repeat mistake rate** | -30% |
| **Context tokens** | Does not grow unbounded over long runs |
| **Promotion rejection** | >95% in strict mode (malformed entries) |

## Implementation Priority

1. ✅ **GPU Validation** - Verify embeddings/LLM on GPU
2. ⚠️ **Promotion Tests** - Test strict mode quality gate
3. ⚠️ **Instrumentation** - Log all memory operations
4. ⚠️ **Debug Task** - Best for proving memory value
5. ⚠️ **Long-Horizon Task** - Killer test for memory
6. ⚠️ **Workload Tests** - DevOps, Coding, etc.
7. ⚠️ **Test Matrix** - Run all variations

## Current Status

- ✅ GPU acceleration working (embedding device auto-detects CUDA)
- ⚠️ FAISS on CPU (would need conda for GPU)
- ⚠️ LLM on CPU (would benefit from GPU)
- ⚠️ Instrumentation - needs implementation
- ⚠️ Test harness - needs implementation
- ✅ All 57 tests passing
- ✅ DML accessible in OpenClaw environment

## Next Steps

1. Implement memory instrumentation
2. Create test harness for OpenClaw integration
3. Run promotion quality tests
4. Execute debug workload (highest value proof)
5. Run long-horizon episode task
6. Analyze results against pass/fail thresholds