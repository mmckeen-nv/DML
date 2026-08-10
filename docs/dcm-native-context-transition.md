# Native-context generation transitions

`NativeContextTransitionCompiler` joins Daystrom's semantic working-set lineage
to exact runtime checkpoint work. It converts two validated `ContextPacket`
generations plus an optional parent checkpoint binding into a deterministic,
payload-free execution plan.

This is the bridge between selecting context and managing materialized native
state. It does not treat semantic retention as proof of reusable KV/SSM state.
Reuse ends at the first positional or digest divergence.

## Integrity chain

Compilation fails closed unless:

- both packets pass their complete packet, manifest, rendered-message, budget,
  scope, model, runtime, and per-segment digest checks;
- the current manifest names the exact parent manifest;
- the working-set transition's stable prefix and suffix match a fresh
  recalculation;
- the checkpoint record binds the exact parent packet and manifest;
- the checkpoint is unexpired and covers the complete reusable prefix; and
- the model-native and served limits are valid.

`NativeContextCheckpointBinding` deliberately keeps three identities separate:

1. the runtime checkpoint digest used by a connector request;
2. the durable checkpoint record digest; and
3. the full model/runtime/packet identity binding digest.

A registry digest is never substituted for the runtime checkpoint digest.

## Execution plan

A reusable parent produces these adapter-neutral steps:

1. `restore_parent_prefix` for the verified unchanged prefix;
2. `prefill_suffix` from the first digest or positional divergence; and
3. `checkpoint_current_generation` after completion.

Without a compatible parent checkpoint, the compiler emits `prefill_full` and
then `checkpoint_current_generation`. Exact replaced or evicted pages are
reported as digest-bound `page_out_exact` work; added or replaced pages are
reported as `page_in_exact` work.

The plan contains IDs, positions, token counts, digests, reason codes, and
checkpoint bindings. It contains no segment payloads or rendered messages.

## CLI

```bash
python3 dml_core/scripts/dcm_native_context_transition.py \
  --input transition-input.json \
  --artifact transition-plan.json
```

The input contains serialized parent/current packets, limits, and an optional
`NativeContextCheckpointBinding`. Unknown top-level fields are rejected.
Failure artifacts disclose only an error class and reason digest.

The representative digest-only plan is
[`docs/examples/dcm/native-context-transition-nemotron-262k-served-65k.json`](examples/dcm/native-context-transition-nemotron-262k-served-65k.json).
It moves a 176,864-token logical generation to a 58,864-token resident
generation while restoring a 36,864-token prefix and prefilling only the
22,000-token changed suffix.

## vLLM compound actuator

`VLLMCooperativeExecutionAdapter.execute_native_transition()` consumes the
integrity-checked plan and performs a version-pinned chained generation that:

- restores the exact parent checkpoint;
- verifies positive native CPU/offload or GPU-prefix reuse counters;
- requires the runtime-reported prompt length to equal the plan's current-token
  boundary;
- prefills the planned suffix;
- binds the completed child packet to its new runtime checkpoint digest; and
- publishes that child checkpoint only after a separately signed readiness
  request proves `stored_blocks == expected_blocks > 0`.

The generation itself is one dual-digest `transition` request. The HMAC covers
both the parent restore digest and distinct child checkpoint digest. The vLLM
0.20 eager offload manager receives one `update_state_after_alloc` call, which
loads the external prefix and registers the same request for child stores.
Tampered child identities, reused parent/child identities, prefix mismatch,
existing-child conflicts, runtime/plan drift, and readiness failure all fail
closed. Result telemetry remains payload-free.

`saved_tokens` is request-finish scheduler telemetry, not physical coverage
proof. Complete child coverage comes only from the separately signed readiness
result.

This offline-tested actuator does not by itself claim shared-host live proof.
The bounded procedure and executable harness are documented in [Live validation
for chained native vLLM transitions](vllm-transition-live-validation.md).
