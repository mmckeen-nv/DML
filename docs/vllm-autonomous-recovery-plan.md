# Capability-gated autonomous recovery plan for cooperative vLLM KV

## Status

This document is a **non-enabling implementation plan**. It does not turn on autonomous save, restore, purge, generic checkpoint deletion, slot erase, or recovery after restart.

The verified vLLM 0.20 vertical slice currently provides:

- request-authorized CPU KV save;
- exact-prefix restore with connector-native matched-token evidence;
- request-bound selective purge of confirmed, resident, idle, unshared CPU rows;
- all-worker physical zeroization acknowledgements; and
- payload-free lifecycle evidence.

It deliberately does **not** provide stable slot affinity, generic slot erase, generic checkpoint deletion, all-cache physical deletion, or checkpoint survival across process restart. Those unsupported capabilities remain hard boundaries, not backlog labels that a controller may reinterpret.

## Goal

Add a small recovery controller that can decide whether a previously verified low-level operation is safe to attempt, while keeping authority narrower than the runtime adapter's advertised capabilities. Recovery must fail closed on runtime drift, identity drift, missing evidence, concurrency, ambiguous ownership, or unsupported cleanup.

The controller is successful when it can improve same-process continuation latency without changing answer semantics, corrupting another checkpoint's shared prefix, retaining secret or prompt material in evidence, or claiming recovery after an unverified operation.

## Non-goals

- Recovering CPU KV after a vLLM process or host restart.
- Treating authorization records as proof that physical KV still exists.
- Inferring restore success from wall-clock latency.
- Turning request-bound selective purge into DML's generic checkpoint-delete capability.
- Using `reset_cache()` or `reset_external=true` as physical deletion evidence.
- Persisting prompts, token IDs, block hashes, signatures, nonces, control keys, model output, or raw memory context in recovery journals.
- Allowing an LLM or learned policy to expand capabilities, TTLs, tenant scope, retry limits, or cleanup authority.

## Current capability matrix

The current adapter explicitly reports `supports_kv_erase=False`, `supports_kv_checkpoint_delete=False`, and `supports_slot_affinity=False`; request-bound selective purge is metadata, not a substitute for any of those flags.

| Capability | Current state | Evidence required | Autonomous authority |
|---|---|---|---|
| Save CPU KV | Implemented | `save_authorized`, exact runtime identity | Shadow or bounded same-session policy only |
| Restore CPU KV | Implemented | `restore_authorized` and positive native `matched_tokens` | Bounded same-process policy only |
| Request-bound selective purge | Implemented with constraints | `purge_complete`, positive physical counters when exclusive rows exist, explicit shared-row count | Terminal cleanup only |
| Shared-prefix containment | Implemented | Other checkpoint restores after selective purge; shared rows reported | Required before canary |
| Stable slot affinity | Unsupported | None | Forbidden |
| Generic slot erase | Unsupported | None | Forbidden |
| Generic checkpoint delete | Unsupported | None | Forbidden |
| All-cache external reset | Unsupported | None | Forbidden |
| Restart recovery | Unsupported; state is process-local | None | Forbidden |

The controller must intersect its policy with this matrix. A policy may remove authority; it may never add authority.

## Proposed components

### 1. `KVRecoveryCapabilitySnapshot`

An immutable snapshot captured from the execution adapter before any mutation:

- adapter type and schema version;
- exact runtime ID, runtime version, endpoint identity, model ID, and process epoch;
- supported operation flags;
- request-bound selective-purge constraints;
- maximum authorization TTL and record cardinality;
- timestamp and sanitized capability digest.

A process epoch must change whenever vLLM restarts. No trustworthy process-epoch surface exists in the current adapter. Until one is implemented, the controller establishes an in-memory epoch after the first successful connector-evidenced response and invalidates it on a failed health probe, transport failure, HTTP 5xx, endpoint reconnect, or controller restart. It may not reconstruct an epoch from a stable URL or runtime version.

### 2. `KVRecoveryIntent`

A controller-created, payload-free record containing:

- tenant, client, session, and request scope digests;
- rendered-context identity digest;
- model/tokenizer/position/runtime identity digest;
- checkpoint digest;
- requested operation and reason code;
- monotonic deadline, attempt number, and policy version.

The raw prompt is supplied only to the completion call. It is never serialized into the intent or journal.

### 3. `KVRecoveryPolicy`

A deterministic allowlist evaluated before every operation. Initial limits:

- mode defaults to `disabled`;
- same tenant, session, model, endpoint, process epoch, and rendered-context prefix only;
- maximum one save authorization, one restore authorization, and one terminal purge authorization per checkpoint; the adapter's two bounded, freshly signed purge-status requests are permitted only while rows remain protected and no replacement save is issued;
- no automatic retry after authorization failure, missing inventory, missing blocks, busy blocks, pending stores, worker mismatch, worker error, timeout, or runtime identity drift;
- short signed-envelope TTL bounded by the adapter's maximum;
- bounded live checkpoint count below the connector's configured maximum;
- no save when host load exceeds an operator-configured threshold;
- no mutation when another controller lease owns the session;
- no learned-policy override of these limits.

### 4. `KVRecoveryController`

A state machine with explicit terminal states:

```text
DISABLED → OBSERVED
OBSERVED → SAVE_PENDING → SAVED
OBSERVED → RESTORE_PENDING → RESTORED
OBSERVED → PURGE_PENDING → PURGED

Any pending state → ABORTED
Any nonterminal state → INVALIDATED on runtime/process identity change
PURGE_PENDING → PURGE_PENDING only for a bounded signed status request
```

Save, restore, and purge are independent operation flows selected from `OBSERVED`; a checkpoint may be saved then purged without restore, or restored/purged after the controller observes an existing same-epoch record. State transitions require response evidence, not request completion alone. `PURGED` requires `purge_complete`; a pending or failed purge remains nonterminal and blocks replacement save for that checkpoint identity.

### 5. Sanitized recovery journal

The journal may record only:

- intent/capability digests;
- operation and reason code;
- native token counters;
- physical purge block/byte/shared counters;
- runtime identity and process epoch;
- bounded latency and attempt counts;
- state transition and policy version.

It must pass a dedicated payload/secret-hygiene test equivalent to the assertions in `integrations/hermes/plugins/daystrom_dml/smoke_hygiene.py`, extended with recovery-journal fixtures and high-entropy field checks. The existing smoke is not, by itself, a general recovery-journal scanner. The journal must not be sufficient to reconstruct a prompt, response, token sequence, signature, nonce, or block hash.

## Decision flow

### Save

1. Capture and validate the capability snapshot.
2. Acquire a bounded controller lease for the scoped session.
3. Confirm same-process health, runtime identity, host-load limit, checkpoint cardinality, and policy mode.
4. Build the checkpoint digest from controller identity inputs; never from secret material.
5. Issue one short-lived signed save.
6. Accept `SAVED` only on `save_authorized` evidence.
7. Record no claim that every prompt token was offloaded; later restore evidence remains authoritative.

### Restore

1. Reject if the process epoch, runtime, model, endpoint, tenant, session, or checkpoint identity changed.
2. Issue one signed restore for an exact context prefix.
3. Accept `RESTORED` only on `restore_authorized` with positive connector-native `matched_tokens`.
4. Enforce the positive-match rule in `KVRecoveryController`. The current adapter deliberately returns connector evidence for an authorized zero-match no-op; it does not classify recovery success and must not be broadened to do so.
5. Treat zero native matches as a miss, even if latency is low or output is deterministic.
6. Continue normal generation from the runtime response. Recovery failure must fall back to an ordinary cold request only when policy explicitly permits it and no mutation remains pending.

### Terminal selective purge

1. Purge only after the session/checkpoint is terminal or an operator explicitly requests cleanup.
2. Refuse purge when save transfers are pending, lazy mode is active, inventory is missing/empty, target rows are missing or busy, or worker acknowledgement is incomplete.
3. Accept completion only on `purge_complete`.
4. Before any autonomous rollout, extend scheduler completion to revalidate that every retained shared hash still has another unexpired, non-purged checkpoint owner. The current connector partitions ownership when purge is scheduled but does not close this owner-expiry race at worker acknowledgement.
5. If shared ownership disappears before completion, keep cleanup nonterminal and report a distinct failure instead of claiming `PURGED`; do not zero or free rows without a newly proven ownership partition.
6. Require positive `purged_blocks` and `purged_bytes` when the checkpoint has exclusive resident rows.
7. Permit zero exclusive rows only when evidence reports retained shared rows and the completion-time ownership check confirms another live checkpoint.
8. After completion, require subsequent restore denial for audit probes; never expose the denied request's prompt in evidence.

## Fault classification and response

| Failure | Controller response | Retry |
|---|---|---|
| Invalid/expired signature or replay | Abort and security-audit sanitized reason | Never automatically |
| Runtime/model/endpoint/process drift | Invalidate checkpoint | Never |
| Save/restore zero native match | Record miss; optional ordinary cold request | No checkpoint retry |
| Pending store or busy rows | Keep checkpoint blocked; operator visibility | Status check only |
| Missing/evicted rows or inventory | Abort purge and preserve failure state | Never automatically |
| Incomplete worker acknowledgement | Keep rows protected and purge pending | Signed status/retry command only |
| Health probe loss | Invalidate all controller checkpoints for that epoch | Never |
| Host overload | Do not mutate; defer decision | Re-evaluate after bounded backoff |
| Journal/hygiene failure | Disable controller | Operator repair required |

Retries use a new nonce and authorization envelope. They do not create replacement checkpoint identities while a purge is pending.

## Rollout gates

### Gate 0 — verified primitives (complete)

Required evidence:

- unit coverage for authorization, replay/TTL limits, save registration, native restore counts, shared ownership, busy/missing rows, pending stores, multi-worker acknowledgement, and physical zeroization;
- live save → GPU-cache reset → CPU restore → selective purge → restore-denied proof;
- live shared-prefix proof in which checkpoint A's exclusive rows are zeroized, shared rows are retained, checkpoint B still restores, and B's final cleanup zeroizes the retained rows;
- reproducible deployment with exact source pin, read-only code mount, secret-file mount, health check, restart policy, and rollback.

### Gate 1 — observe-only controller

Implement capability snapshots, intents, policy decisions, process epoch, and sanitized journal. The controller emits what it would do but performs no save, restore, or purge. Compare decisions with operator expectations across at least seven days of representative traffic.

Acceptance:

- zero prompt/secret leakage findings;
- zero decisions outside the capability matrix;
- deterministic decisions for identical sanitized inputs;
- host-load and cardinality limits exercised;
- restart invalidation demonstrated.

### Gate 2 — operator-assisted lifecycle

Allow an operator to approve each save, restore, and terminal purge. No automatic mutation.

Acceptance:

- every state transition has runtime-native evidence;
- failed/pending purge blocks replacement save;
- cold fallback is explicit and never presented as restored execution;
- canary rollback disables the controller without requiring all-cache reset.

### Gate 3 — bounded same-session autonomy

Allow automatic save/restore/terminal purge only for one canary tenant on one pinned runtime process. Keep generic DML erase/delete/slot-affinity capabilities false.

Initial limits:

- one active runtime endpoint;
- one process epoch;
- one controller lease per session;
- no cross-restart recovery;
- no concurrent mutation of one checkpoint;
- maximum three operation authorizations across the full lifecycle; the adapter's bounded purge-status requests do not count as new mutation authorizations but remain capped at two;
- operator kill switch and automatic circuit break after the first invariant failure.

Promotion requires an A/B artifact showing useful latency improvement, deterministic output checks, native restore counters, physical purge evidence, and no increase in runtime errors.

### Gate 4 — generic DML checkpoint lifecycle

Blocked until the runtime actually exposes stable slot affinity, generic slot erase, physical checkpoint deletion, and restart-safe identity/registry semantics. At that point, implement those capabilities in the adapter and pass the full `ExecutionCheckpointController` probe. Do not map request-bound selective purge onto these generic flags.

## Test plan

### Unit and property tests

- capability intersection never widens adapter authority;
- state transitions reject skipped or repeated operations;
- process/runtime/model/tenant/session drift invalidates restore;
- zero native match cannot become `RESTORED`;
- pending purge blocks replacement save;
- status retry is idempotent and uses a fresh authorization nonce;
- journals reject payload-bearing keys and high-entropy secret material;
- bounded TTL/cardinality/attempt/load policies hold under generated inputs.

### Runtime fault injection

- delayed store completion;
- busy CPU rows;
- evicted/missing rows;
- one worker missing its purge acknowledgement;
- worker zeroization error;
- endpoint reconnect and process-epoch change;
- controller crash after scheduling purge but before receiving acknowledgement;
- shared prefix with one owner purged and the other restored concurrently.

Every injected ambiguity must leave the controller in `ABORTED`, `INVALIDATED`, or `PURGE_PENDING`, never a successful terminal state.

### Live canary evidence

For each canary cycle preserve only sanitized evidence:

- exact source/runtime/process identity;
- prompt-token count, not prompt text;
- cold, save, restore, and warm latency distributions;
- connector-native matched tokens;
- output digest equivalence for deterministic probes;
- purged block/byte/shared counters;
- post-purge restore denial;
- runtime error, waiting-request, and running-request counters.

Shared-host observations are not capacity SLAs. Re-run under isolated load before setting production thresholds.

## Implementation sequence

1. Add dependency-light KV recovery types and deterministic policy under a new `daystrom_dml.context.kv_recovery` module. Do not reuse or replace the existing `daystrom_dml.context.recovery` module, which implements a different fault-retry flow.
2. Add adapter capability snapshots and a process-epoch surface without changing existing generic capability flags.
3. Implement the observe-only state machine and payload-free journal.
4. Add unit/property tests and hygiene fixtures.
5. Add an operator-assisted CLI/API that requires explicit approval per mutation.
6. Run fault injection against a pinned vLLM 0.20 test runtime.
7. Enable a single-tenant canary behind a default-off feature flag and kill switch.
8. Review evidence before any same-session autonomous mutation.
9. Keep Gate 4 blocked until the missing generic runtime contracts are implemented and independently probed.

## Kill switch and rollback

The controller feature flag must default to off. Disabling it stops new save/restore/purge decisions immediately and preserves pending state for operator inspection. Rollback must not call external all-cache reset or claim physical cleanup. Operators may complete a known pending selective purge with signed status evidence or let process-local state disappear on a controlled runtime restart, recording the latter as invalidation rather than verified purge.

That boundary is intentional: safe autonomy is a smaller policy over proven capabilities, not a reason to pretend unsupported capabilities exist.
