# Experimental controller-gated vLLM KV connector

This vertical slice targets **vLLM 0.20.x** and uses vLLM's V1 `KVConnectorBase_V1` plugin path without forking vLLM. `DaystromCooperativeKVConnector` subclasses `SimpleCPUOffloadConnector`, so approved requests use the runtime's real GPU↔CPU KV transfer machinery.

## What is implemented

- Request-level save/restore authorization through OpenAI `kv_transfer_params`.
- HMAC-SHA256 envelopes with bounded TTL and cardinality.
- Controller checkpoint identity bound inside the runtime to the exact ordered vLLM block hashes observed during save.
- Restore allowed only when the saved block-hash sequence is an exact prefix of the new request.
- Unapproved requests neither register CPU stores nor query the external CPU cache.
- Native matched-token counts come from `SimpleCPUOffloadScheduler`, not wall-clock inference.
- Payload-free connector evidence is returned in response `kv_transfer_params.daystrom`.
- A dependency-light client, `VLLMCooperativeExecutionAdapter`, validates that evidence.

## Deliberate fail-closed boundary

This is **not yet** a complete DML checkpoint lifecycle adapter:

- CPU KV physical zeroization/purge is not implemented.
- `reset_cache()` clears authorization state and raises `NotImplementedError`; it never reports fake deletion.
- Selective external block deletion is not implemented.
- Stable slot affinity is not exposed by vLLM.
- Records and CPU-offloaded KV are process-local and do not survive restart.

Accordingly the client reports `supports_kv_erase=False`, `supports_kv_checkpoint_delete=False`, and `supports_slot_affinity=False`. DML's full `erase → save → restore → purge` runtime probe must reject this connector until those capabilities are real.

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

This clears local GPU prefix blocks but preserves external connector state. A following approved restore must report a positive native `matched_tokens` count. Do **not** use `reset_external=true`: this connector intentionally raises because physical purge is not implemented.

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

The response evidence contains only schema, operation, checkpoint digest, reason code, and native token counters. It excludes prompts, outputs, token IDs, nonces, signatures, block hashes, and the control key.

## Verification

Local unit tests stub the exact vLLM 0.20 connector methods and verify:

- unauthorized store/load paths remain no-ops;
- approved save requests register with the eager CPU-offload manager;
- restore requires an exact saved prefix and uses the parent's native token count;
- restore completion delegates cleanup to the parent manager;
- response telemetry is payload-free;
- unsupported purge fails closed.
