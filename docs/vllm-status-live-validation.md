# Load-aware live validation for signed checkpoint status

## Purpose

Validate the vLLM 0.20 cooperative connector's signed `status` operation on the shared DGX without stopping, unloading, restarting, or reconfiguring an active Ollama workload.

This runbook is deliberately non-enabling: it does not authorize autonomous recovery, generic checkpoint deletion, all-cache reset, or restart recovery. It validates one short-lived connector process epoch and leaves the current production services unchanged unless an operator separately approves a canary deployment.

## Hard safety rules

- Inspect before acting. Do not start, stop, restart, update, or remove Ollama, OpenShell, or an existing vLLM container during preflight.
- Do not bind a port already in use. Use a separately approved canary port and container name.
- Do not reclaim GPU memory by unloading Ollama models. If headroom is insufficient, abort and defer.
- Concurrency is exactly one; use one checkpoint and short deterministic completions.
- Never use `reset_external=true`.
- Never print, copy into shell history, or persist the HMAC control key. Mount an owner-only secret file.
- Preserve only payload-free evidence: source/runtime identity, checkpoint digest, reason codes, readiness counters, native cache counters, latencies, and output digests.
- Stop after the first invariant failure. Do not retry mutation requests automatically.

## Operator-supplied values

```bash
export DGX_HOST='<approved-host>'
export DGX_USER='<approved-user>'
export CANARY_PORT='<approved-unused-port>'
export CANARY_CONTAINER='daystrom-vllm-status-canary'
export MODEL_ID='<exact-served-model-id>'
export SOURCE_REF='<exact-reviewed-commit-or-tree-id>'
```

Do not put credentials or the HMAC key in these variables.

## Gate A — read-only host preflight

Run read-only inspection over SSH:

```bash
ssh "$DGX_USER@$DGX_HOST" '
  set -eu
  echo "== uptime/load =="
  uptime
  echo "== GPU =="
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  echo "== GPU processes =="
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  echo "== Ollama service/processes =="
  systemctl is-active ollama 2>/dev/null || true
  ollama ps 2>/dev/null || true
  echo "== listeners =="
  ss -ltn
  echo "== containers =="
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
'
```

Abort before deployment if any of the following is true:

- the selected canary port is already listening;
- GPU free memory is below the separately measured canary startup requirement plus an operator safety reserve;
- GPU utilization or request queues indicate sustained production work;
- an Ollama model would need to be unloaded to make room;
- the existing vLLM/OpenShell state is unhealthy or ambiguous;
- the exact source, vLLM version, model, tokenizer, or connector configuration cannot be pinned.

The threshold is intentionally not hard-coded: use the measured peak from the prior known-good connector deployment and add a reserve. Do not estimate from model parameter count.

## Gate B — immutable canary preparation

Only after operator approval:

1. Pin the exact reviewed source tree and vLLM `0.20.x` image digest.
2. Mount `dml_core` read-only.
3. Create a new random control key directly on the DGX in a temporary owner-only file (`0600`); never return its contents to the client or artifact.
4. Start a separate canary container on the approved unused port with:
   - `--enable-prefix-caching`;
   - `--no-disable-hybrid-kv-cache-manager`;
   - `DaystromCooperativeKVConnector` with `kv_role=kv_both`;
   - eager CPU offload;
   - bounded CPU capacity, TTL, and record cardinality;
   - `VLLM_SERVER_DEV_MODE=1` only if the local GPU prefix reset proof will be run;
   - restart policy `no`.
5. Do not modify the existing production container or Ollama service.

Capture only image/source digests, runtime version, model ID, container ID digest, canary port, and start timestamp. Do not capture mounts containing the key or request payload values.

## Gate C — health and idle proof

Require all of the following before mutation:

```bash
curl --fail --silent --show-error "http://$DGX_HOST:$CANARY_PORT/health" >/dev/null
curl --fail --silent --show-error "http://$DGX_HOST:$CANARY_PORT/v1/models"
curl --fail --silent --show-error "http://$DGX_HOST:$CANARY_PORT/metrics"
```

Acceptance:

- `/health` is HTTP 200;
- the exact expected model is served;
- running and waiting queues are zero immediately before the canary;
- no startup traceback, OOM, or connector initialization error exists;
- Ollama service/model state is unchanged from Gate A;
- GPU reserve remains above the operator threshold.

## Gate D — bounded status lifecycle

Use the dependency-light `build_kv_transfer_params()` helper to create every signed envelope from the mounted key file. Every request gets a fresh nonce and a TTL no longer than the connector maximum. Do not log the envelope.

Run one concurrency-1 lifecycle:

1. **Unknown checkpoint status**
   - Send a valid signed `operation: "status"` for a fresh checkpoint digest.
   - Require `record_not_found`, `checkpoint_ready=false`, `stored_blocks=0`, and `expected_blocks=0`.

2. **Invalid-signature control**
   - Send the same status shape with an invalid HMAC.
   - Require no `kv_transfer_params.daystrom` response evidence. Any lifecycle detail is a security failure.

3. **Authorized save**
   - Send one deterministic, bounded prompt with `operation: "save"`.
   - Require `save_authorized`.

4. **Pending/ready polling**
   - Send fresh signed `status` requests at a bounded cadence (maximum three checks; no mutation retries).
   - Permit `checkpoint_pending` transiently.
   - Require terminal `checkpoint_ready=true` with `stored_blocks == expected_blocks > 0` before attempting restore.
   - Treat `checkpoint_partial`, `checkpoint_evicted`, `checkpoint_below_granularity`, `purge_shared_ownership_changed`, timeout, transport loss, or identity drift as an abort.

5. **Managed restore proof**
   - Clear local GPU APC only with `POST /reset_prefix_cache?reset_external=false`.
   - Send the exact authorized restore prefix.
   - Require positive connector-native `cpu_offload_matched_tokens`; GPU APC counters alone are not proof.

6. **Optional eviction observation**
   - Do not create pressure merely to force eviction on the shared host.
   - If ordinary canary activity naturally evicts rows, a signed status may report `checkpoint_partial` or `checkpoint_evicted`; record it as correct fail-closed evidence, not a test failure.

7. **Selective purge**
   - Issue one signed request-bound purge only after queues are idle and the checkpoint is terminal.
   - Require `purge_complete` and the expected physical counters. A pending purge permits only the connector's bounded fresh-nonce status behavior; no replacement save.

8. **Post-purge status**
   - Send one fresh signed `status`.
   - Require `purge_complete` and `checkpoint_ready=false`.

9. **Restore-denied audit probe**
   - After another local-GPU-only reset, require restore denial with purge lifecycle evidence and zero managed restore increase.

## Evidence schema

Persist one redacted JSON artifact containing only:

```json
{
  "schema_version": "daystrom-vllm-status-canary-v1",
  "source_ref": "<digest>",
  "runtime_version": "<exact-version>",
  "model_id": "<id>",
  "endpoint_origin_digest": "sha256:<digest>",
  "checkpoint_digest": "sha256:<digest>",
  "ollama_state_unchanged": true,
  "preflight": {"queues_idle": true, "gpu_reserve_ok": true},
  "events": [
    {
      "operation": "status",
      "reason_code": "checkpoint_ready",
      "checkpoint_ready": true,
      "stored_blocks": 1,
      "expected_blocks": 1,
      "latency_ms": 0
    }
  ],
  "output_digests": ["sha256:<digest>"],
  "result": "pass"
}
```

Do not persist prompts, completions, request IDs, nonces, signatures, the control key, block hashes, token IDs, headers, credentials, container environment, or raw logs. Run the repository hygiene scanner and compute the artifact SHA-256 after validation.

## Gate E — postflight and rollback

1. Confirm canary request queues are empty.
2. Stop and remove only the separately named canary container.
3. Delete only the ephemeral canary key file.
4. Verify the original Ollama service/model state, existing vLLM/OpenShell containers, GPU process list, and listening ports match preflight expectations.
5. Verify no canary process or listener remains.
6. If the canary failed, preserve sanitized evidence and classify the checkpoint/process epoch as invalidated. Do not call all-cache reset or claim physical cleanup that was not acknowledged.

## Pass criteria

The live status contract passes only if:

- invalid HMAC receives no lifecycle response;
- valid unknown status reports `record_not_found`;
- save transitions through pending (if observed) to ready;
- ready requires `stored_blocks == expected_blocks > 0`;
- local GPU APC reset does not destroy CPU readiness;
- managed restore has positive native CPU-offload evidence;
- partial/evicted state never reports ready;
- purge completion is worker-acknowledged and post-purge status is not ready;
- no secret/prompt material enters artifacts;
- Ollama and unrelated workloads are unchanged;
- the canary is removed and the host returns to its preflight service posture.
