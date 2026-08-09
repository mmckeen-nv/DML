# Experimental controller-gated vLLM KV connector

This vertical slice targets **vLLM 0.20.x** and uses vLLM's V1 `KVConnectorBase_V1` plugin path without forking vLLM. It operates as a **GPU-first hybrid**: vLLM checks its local GPU Automatic Prefix Cache first, then `DaystromCooperativeKVConnector` subclasses `SimpleCPUOffloadConnector` to restore any authorized prefix remainder from CPU RAM.

GPU APC is opportunistic and hash-driven; it is not controller-scoped storage. Daystrom authorization, checkpoint identity, and selective physical purge continue to apply only to the managed CPU fallback tier.

## What is implemented

- Request-level save, restore, selective purge, and payload-free status authorization through OpenAI `kv_transfer_params`.
- HMAC-SHA256 envelopes with bounded TTL and cardinality.
- Controller checkpoint identity bound inside the runtime to the exact ordered vLLM block hashes observed during save.
- Restore allowed only when the saved block-hash sequence is an exact prefix of the new request.
- Unapproved requests neither register CPU stores nor query the external CPU cache.
- Native matched-token counts come from vLLM's scheduler contracts, not wall-clock inference. Response evidence separates `gpu_apc_matched_tokens` from managed `cpu_offload_matched_tokens` and labels the route as `gpu_apc`, `cpu_fallback`, `gpu_apc_and_cpu`, or `miss`.
- GPU prefix caching is mandatory: connector startup fails rather than silently degrading to CPU-only or cold behavior.
- Selective purge partitions unique from still-shared prefix blocks, protects target CPU rows from allocator reuse, removes their lookup keys, flushes in-flight DMA on every worker, zeroes the pinned CPU tensors in place, aggregates all-worker acknowledgements, and only then frees the rows and removes the checkpoint record.
- Shared rows remain available to their other authorized checkpoint owners and are reported separately. Their exact logical hashes are revalidated against live owners again at worker acknowledgement; changed ownership blocks purge commit with `purge_shared_ownership_changed` and keeps protected state nonterminal.
- Payload-free connector evidence is returned in response `kv_transfer_params.daystrom`.
- A freshly signed `status` request reports logical lifecycle plus live CPU-row readiness as `checkpoint_pending`, `checkpoint_ready`, `checkpoint_partial`, `checkpoint_evicted`, or `checkpoint_below_granularity`. Ready requires confirmed store completion and `stored_blocks == expected_blocks > 0`; invalid signatures receive no status evidence.
- A dependency-light client, `VLLMCooperativeExecutionAdapter`, validates save/restore evidence, exposes typed signed status and bounded pending→ready polling, and requires `purge_complete` plus physical counters for request-bound purge.

## Deliberate fail-closed boundary

This is **not yet** a complete DML checkpoint lifecycle adapter:

- Selective purge is request-bound to this connector; it is not exposed as DML's generic `delete_checkpoint()` or KV-erase capability.
- Purge succeeds only in eager-offload mode after confirmed store inventory has been captured, and only while every inventoried unshared target row is still resident and idle. Lazy mode, missing/evicted rows, active references, pending stores, worker errors, or incomplete multi-worker acknowledgements fail closed without a `purge_complete` result.
- Bytes shared with another live checkpoint are intentionally retained and counted as `shared_blocks`; deleting them would corrupt the other owner.
- `reset_cache()` clears ordinary authorization state and raises `NotImplementedError`; it never reports all-cache physical deletion. Pending selective purges remain tracked until worker acknowledgement.
- Stable slot affinity is not exposed by vLLM.
- Records and CPU-offloaded KV are process-local and do not survive restart.

Accordingly the client still reports `supports_kv_erase=False`, `supports_kv_checkpoint_delete=False`, and `supports_slot_affinity=False`. It advertises only payload-free metadata `request_bound_selective_purge=True` with the resident/unshared constraint. DML's generic full `erase → save → restore → purge` runtime probe must continue to reject this adapter until those generic lifecycle contracts are real.

## vLLM configuration

Create a random control key in a runtime-mounted file with owner-only permissions. Do not put the key in CLI arguments, environment variables, logs, or artifacts.

Mount `dml_core` read-only into the vLLM container and add it to `PYTHONPATH`. Enable vLLM prefix caching because `SimpleCPUOffloadConnector` requires it. Configure the external module path:

```json
{
  "kv_connector": "DaystromCooperativeKVConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "daystrom_dml.context.vllm_bridge.connector",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 8589934592,
    "daystrom_secret_path": "/run/secrets/daystrom-kv-control.key",
    "daystrom_max_ttl_seconds": 900,
    "daystrom_max_records": 128
  }
}
```

Pass the JSON with vLLM's `--kv-transfer-config`, add `--enable-prefix-caching`, and explicitly add `--no-disable-hybrid-kv-cache-manager`. vLLM 0.20 otherwise auto-disables the hybrid manager whenever any KV connector is configured; Nemotron's hybrid attention/Mamba cache specs cannot be unified under that fallback and startup fails closed.

For a proof that reuse came from the connector rather than GPU APC, start the API server with `VLLM_SERVER_DEV_MODE=1`, perform an approved save, then call:

```text
POST /reset_prefix_cache?reset_external=false
```

This clears local GPU prefix blocks but preserves external connector state. A following approved restore must report a positive native `matched_tokens` count. Do **not** use `reset_external=true`: the connector intentionally rejects all-cache reset because it cannot prove physical deletion of every external row. Selective deletion uses a signed `operation: "purge"` request for one checkpoint instead.

## Request envelope

The OpenAI completion request includes:

```json
{
  "kv_transfer_params": {
    "daystrom": {
      "schema_version": "daystrom-vllm-kv-v1",
      "operation": "save",
      "checkpoint_digest": "sha256:<controller identity digest>",
      "expires_at": 1700000000.0,
      "nonce": "bounded-opaque-nonce",
      "authorization": "<HMAC-SHA256>"
    }
  }
}
```

The response evidence contains only schema, operation, checkpoint digest, reason code, native token counters, cache route, selective-purge counters (`purged_blocks`, `purged_bytes`, `shared_blocks`), and—only for `status`—`checkpoint_ready`, `stored_blocks`, and `expected_blocks`. `matched_tokens` remains the backward-compatible managed CPU restore count; `gpu_apc_matched_tokens` reports the local GPU prefix count supplied by vLLM, and `cpu_offload_matched_tokens` reports the external fallback count. It excludes prompts, outputs, token IDs, nonces, signatures, block hashes, and the control key. A purge client accepts only `reason_code: "purge_complete"`; `purge_pending`, busy/missing-block failures, and worker-count mismatches are failures.

A `gpu_apc` route proves a native local-cache hit, not controller-authorized checkpoint restoration. Only positive `cpu_offload_matched_tokens` is evidence that the managed Daystrom fallback restored KV.

## Verification

Local unit tests stub the exact vLLM 0.20 connector methods and verify:

- unauthorized store/load paths remain no-ops;
- approved save requests register with the eager CPU-offload manager;
- restore requires an exact saved prefix and uses the parent's native token count;
- restore completion delegates cleanup to the parent manager;
- response telemetry is payload-free;
- shared prefixes are retained rather than corrupting another checkpoint;
- target rows are held out of the allocator, zeroed in place only after DMA flush, acknowledged across workers, and freed only after commit;
- missing, busy, or incomplete purge paths fail closed;
- signed status distinguishes pending, ready, partial, evicted, below-granularity, missing, and purged checkpoints without returning evidence to an invalid HMAC;
- generic reset/delete capabilities remain unsupported.

The load-aware shared-host canary procedure is documented in [Load-aware live validation for signed checkpoint status](vllm-status-live-validation.md).

## Next-step recovery design

The default-off, capability-gated rollout plan is documented in [Capability-gated autonomous recovery plan for cooperative vLLM KV](vllm-autonomous-recovery-plan.md). The plan preserves the unsupported generic capability flags and does not enable autonomy.