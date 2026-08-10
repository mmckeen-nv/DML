# Native context exhaustion A/B validation

This campaign answers two different questions and keeps their evidence separate.

1. **Execution-state A/B:** does a controller-authorized CPU KV checkpoint reduce repeated prefill latency versus a genuinely cold full-history recomputation?
2. **Logical exhaustion:** how close can a request get to the runtime's served context limit, and what happens when prompt plus requested output exceeds it?

A checkpoint can reduce repeated prefill work. It does **not** extend the runtime's served logical context limit. Context compaction, exact page selection, and future scheduler-level active paging are separate mechanisms.

## Paired A/B contract

`dml_core/scripts/dcm_context_exhaustion_ab.py` runs a growing sequence of tokenizer-calibrated plans. For every rung it randomizes the order of two identical-prompt lanes:

- **Enabled:** reset local GPU APC, restore the exact parent from the managed CPU checkpoint, prefill the suffix, and publish a ready child checkpoint. Acceptance requires `cache_route=cpu_fallback`, positive `cpu_offload_matched_tokens`, and zero `gpu_apc_matched_tokens`.
- **Disabled:** reset local GPU APC and recompute the complete prompt under a fresh save identity. Acceptance requires zero CPU and GPU reuse and a `miss` or `not_applicable` route.

Both lanes use `temperature=0`, seed `17`, eight requested output tokens, concurrency one, and identical prompt bytes. Every pair requires equal output digests. Checkpoints are selectively purged in reverse order and the final artifact requires complete physical zeroization accounting.

The harness rejects any A/B rung above `served_limit`. Over-limit behavior belongs to the boundary lane and must not be represented as managed continuation.

Primary per-rung metrics are:

- wall-clock completion latency;
- actual prompt and completion tokens;
- managed CPU and GPU APC matched tokens;
- prefill tokens charged (`prompt_tokens - cpu_offload_matched_tokens`);
- recomputation-avoidance fraction;
- paired latency delta and enabled-faster percentage;
- output equivalence;
- signed child readiness and purge accounting.

Checkpoint publication/readiness is setup cost and remains visible separately. A production economics report must amortize that cost over the number of future reuses rather than hiding it inside or outside the preferred lane.

## 2026-08-10 cold boundary pilot

The live `nvidia/nemotron-3-super` vLLM 0.20 endpoint reported `max_model_len=65536`. A bounded, synthetic-only, concurrency-one pilot cleared local GPU APC before every request and observed zero prefix-cache hits.

| Prompt tokens | Prompt + requested output | Wall latency | Native TTFT | Result |
|---:|---:|---:|---:|---|
| 32,768 | 32,776 | 1.859 s | 1.854 s | HTTP 200 |
| 49,152 | 49,160 | 2.834 s | 2.783 s | HTTP 200 |
| 60,000 | 60,008 | 3.491 s | 3.484 s | HTTP 200 |
| 65,000 | 65,008 | 3.794 s | 3.787 s | HTTP 200 |
| 65,528 | 65,536 | 3.793 s | 3.786 s | HTTP 200 |
| 65,530 | 65,538 | 0.135 s | n/a | HTTP 400 `BadRequestError` |

The current deployment therefore accepts a request whose prompt plus requested output equals 65,536 and rejects a request two tokens above it. This is the deployment's confirmed logical boundary, not the model's larger native architectural target.

Digest-only evidence: `docs/artifacts/vllm-cold-context-boundary-pilot-2026-08-10.json`.

Rendered evidence chart:

- PNG: `docs/artifacts/vllm-context-exhaustion-chart-2026-08-10.png`
- editable SVG: `docs/artifacts/vllm-context-exhaustion-chart-2026-08-10.svg`
- deterministic renderer: `docs/artifacts/render_vllm_context_exhaustion_chart.py`

## Existing paired managed evidence

The preceding native transition-chain proof contains one directly paired final-shape comparison at 18,500 prompt tokens:

- managed CPU transition: `436.044 ms`, with 12,672 native CPU-matched tokens;
- cold full recomputation: `1,169.102 ms`, with zero reuse;
- paired saving: `733.059 ms`;
- managed lane faster by `62.70%` (`2.68x` cold/managed latency ratio);
- deterministic output digests equal.

This is a valid measured point, not yet a complete scaling curve. The new harness exists to repeat that route-proven pair toward the 65,536-token boundary.

## Exhaustion interpretation

At the current endpoint limit, both enabled and disabled lanes must keep prompt plus requested output at or below 65,536. Managed checkpointing can make the accepted request cheaper, but cannot make a 65,538-token request legal.

To continue a conversation beyond that boundary today, the host must compact, omit, summarize, or exactly retrieve selected older material into a new request that fits. A future claim that Daystrom preserves more *native positions* without compaction requires a higher served limit plus scheduler/cache-manager control of live sequence state, physical page residency, restoration, and exact continuation. CPU checkpoint restore alone is not that proof.
