# Live validation for chained native vLLM transitions

## Purpose

Prove the compound `restore parent → execute suffix → publish child` path against an isolated vLLM 0.20 canary. This extends the single-checkpoint procedure in [vllm-status-live-validation.md](vllm-status-live-validation.md); it does not authorize changes to production Ollama or vLLM services.

Offline tests prove orchestration and fail-closed behavior only. A live claim requires native managed-CPU reuse counters, complete child readiness, deterministic cold equivalence, and acknowledged physical cleanup from the real runtime.

## Preconditions

Complete Gates A–C of the status runbook first. Additionally require:

- the exact reviewed DML source SHA and vLLM image digest;
- an isolated canary port/container with concurrency one;
- `VLLM_SERVER_DEV_MODE=1` for local GPU APC reset;
- a plan compiled for the exact model/runtime IDs;
- UTF-8 parent/current prompt files whose runtime tokenizer counts equal the plan's `stable_prefix_tokens` and `current_tokens`;
- a fresh owner-only HMAC key mounted into the canary;
- enough measured GPU reserve without unloading Ollama or disturbing other workloads.

Do not use the 58,864-token documentation example unless its exact prompt payload and tokenizer-derived boundaries are available. The example artifact is a compiler demonstration, not a runnable prompt fixture.

## Bounded execution

Run from the reviewed checkout:

```bash
python dml_core/scripts/dcm_native_context_transition_canary.py \
  --plan '<payload-free-plan.json>' \
  --parent-prompt-file '<stable-prefix.txt>' \
  --current-prompt-file '<current-context.txt>' \
  --endpoint-url "http://$DGX_HOST:$CANARY_PORT" \
  --model-id "$MODEL_ID" \
  --runtime-id '<exact-plan-runtime-id>' \
  --runtime-version '0.20.0' \
  --secret-path '<owner-only-mounted-key-file>' \
  --source-ref "$SOURCE_REF" \
  --artifact '<payload-free-result.json>' \
  --ttl-seconds 300
```

The harness performs exactly one concurrency-one lifecycle:

1. Save the plan's stable parent prefix and require signed complete readiness.
2. Clear only local GPU APC with `reset_external=false`.
3. Execute the compound transition and require:
   - runtime prompt tokens equal `plan.current_tokens`;
   - `gpu_apc_matched_tokens == 0`;
   - `cpu_offload_matched_tokens > 0`;
   - `cache_route == "cpu_fallback"`;
   - child `stored_blocks == expected_blocks > 0`.
4. Clear only local GPU APC again.
5. Run a full-prefill cold control with identical prompt, temperature, seed, and output budget.
6. Require the transition and cold-control UTF-8 output digests to match.
7. Purge the cold, child, and parent checkpoint identities in reverse creation order. Every identity that may have reached the runtime is included in cleanup even if a later invariant fails.

Stop after the first primary invariant failure. Cleanup attempts are bounded and do not call all-cache reset.

## Pass criteria

A result is passing only when:

- parent and current runtime token counts match the compiled plan;
- the transition demonstrates isolated managed CPU reuse rather than GPU APC;
- the child checkpoint is physically complete;
- deterministic output matches the cold full-prefill control;
- every created checkpoint receives `purge_complete`;
- the artifact contains digests, counters, reason codes, sampling settings, and latencies only;
- postflight confirms the canary is removed and unrelated services are unchanged.

The harness intentionally does not claim production readiness, restart persistence, generic delete capability, or autonomous recovery.

## Postflight

Complete Gate E of the status runbook. Run the repository hygiene scanner and added-line secret audit before committing a live artifact. Never commit prompt files, the HMAC key, nonces, signatures, raw completions, block hashes, or container environment dumps.
