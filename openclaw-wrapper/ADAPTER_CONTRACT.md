# DML Agent Memory Adapter Contract

Contract version: `dml-agent-memory-v1`

This contract is the stable beta surface for wiring Daystrom DML into an
agentic harness. A harness does not need to understand the internal lattice. It
only needs to write structured memories and read compact context packets.

## Commands

All commands emit JSON and should be safe for another process to parse.

Mutating commands acquire a shared store write lock at
`$DML_STORE/.dml_store.lock`. The default is fail-fast. Set
`--lock-timeout-ms <ms>` before the subcommand when a harness should wait for
another writer. A blocked writer returns JSON with `status: "blocked"`,
`error: "store_write_lock_held"`, and lock holder metadata.

Mutating commands also append compact events to `$DML_STORE/dml_audit.jsonl`.
Use global `--audit-actor <label>` to identify the harness or user-facing
bridge. Audit entries store operation metadata, scope, counts, hashes, and
status; they must not store raw memory text.

### Health

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path config/dml_portable_linux.yaml \
  --no-require-gpu \
  health
```

Use this before trusting the store. The command checks the durable JSONL state
header, checksum, record count, embedding dimensions, summary coverage, and
continuity memory count. Add `--probe-backend` when the caller also wants to
instantiate the adapter and verify embedding/LLM backend surfaces.

Expected top-level fields:

- `status`: `ok`, `degraded`, or `fail`
- `contract_version`: `dml-agent-memory-v1`
- `state.exists`, `state.checksum_ok`, `state.count_ok`
- `state.record_count`
- `state.embedding_dimensions`
- `state.active_continuity_count`
- `state.unscoped_count`
- `state.records_by_tenant`
- `state.active_continuity_by_tenant`
- `audit.event_count`, `audit.latest_ts`
- `store_lock.path`, `store_lock.metadata`
- `errors`

### Backup

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  backup \
  --label before-maintenance \
  --keep 20
```

Creates a timestamped backup directory under `$DML_STORE/backups` by default.
The backup includes `dml_state.jsonl` and any present sidecar files such as the
dedup index, embedding migration report, and DPM preference graph. Each backup
includes `backup_manifest.json` with file sizes and SHA-256 checksums.

### Verify

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  verify
```

Runs the health checks and then loads the state through the real persistence
loader. Use this before migrations, after restores, and when health reports a
degraded store.

### Restore

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  restore \
  --backup "$DML_STORE/backups/20260521T000000Z-before-maintenance"
```

Restore validates the backup manifest checksum, makes a pre-restore backup of
the current store by default, and replaces state atomically. Add
`--no-pre-restore-backup` only when the current store is intentionally
discardable.

### Audit Tail

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  audit-tail \
  --limit 20
```

Returns recent append-only audit events. Use this to debug multi-agent write
activity without exposing raw memory text.

### Recall Eval

```bash
python scripts/recall_eval.py \
  --output-dir /tmp/dml-recall-eval
```

Runs a low-cost recall regression suite through the wrapper CLI. By default it
uses a temporary isolated store, writes deterministic fixture memories, and
scores:

- expected marker recall
- tenant isolation
- session isolation
- active continuity resume
- ingest/retrieve/resume latency

The command exits `0` when all cases pass and emits a JSON report to stdout.
When `--output-dir` is set, it also writes `recall_eval_report.json` and
`recall_eval_report.md`. Use `--storage-dir` only with a disposable test store;
the eval intentionally writes fixture memories.

### Beta Readiness

```bash
python scripts/beta_readiness.py \
  --storage-dir "$DML_STORE" \
  --tenant-id openclaw \
  --output-dir /tmp/dml-beta-readiness
```

Runs the portable beta gate:

- `health`
- `verify`
- `conflicts`
- `audit-tail`
- isolated `recall_eval.py`

The command exits `0` only when required checks pass and unresolved conflicts
are within budget. Add `--skip-recall-eval` for a fast store-only preflight.

### Ingest

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path config/dml_portable_linux.yaml \
  --no-require-gpu \
  ingest \
  --tenant-id openclaw \
  --kind note \
  --meta '{"source":"harness","namespace":"active_continuity"}' \
  --text "..."
```

Required metadata for beta integrations:

- `source`: harness or component name
- `kind`: set by `--kind`; one of `action`, `observation`, `note`, `plan`,
  `error`, `artifact`
- `namespace`: logical memory lane, such as `active_continuity`

Optional claim-conflict metadata:

- `conflict_key` or `claim_key`: stable key for a fact/decision that should be
  unique within the scoped lane, such as `deploy_mode` or `active_branch`
- `claim_value` or `conflict_value`: current value for that key

When a new write has the same tenant/client/session/instance/namespace and
claim key as an existing active memory but a different claim value, ingest keeps
the memory retrievable and annotates it with `conflict_state: "conflicted"`,
`conflict_scope`, and compact `conflicts_with` references. The conflict record
uses metadata, IDs, values, and text hashes; it does not duplicate raw prior
memory text.

Scope metadata:

- New writes default to `tenant_id=openclaw`.
- Harnesses should pass `--tenant-id` explicitly for multi-user deployments.
- Use `--client-id`, `--session-id`, and `--instance-id` when a memory should
  be isolated below the tenant level.
- Legacy unscoped memories can still be used through the compatibility fallback,
  but new multi-user harnesses should not create unscoped records.

Recommended continuity metadata:

- `thread`
- `state`
- `task`
- `next_action`
- `captured_at` or `updated_at`
- `memory_state`: `active` unless intentionally quarantined/suppressed

Summary policy:

- `--summary-policy auto` is the default and should be used by most harnesses.
- Use `--summary-policy cheap` when the harness has already extracted compact
  state and wants to avoid LLM summarization cost.
- Use `--summary-policy llm` for large, ambiguous natural-language chunks.
- Use `--summary-policy skip` for raw audit records that should not be cached
  into prompt-facing summaries.

### Retrieve

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path config/dml_portable_linux.yaml \
  --no-require-gpu \
  retrieve \
  --query "current task context" \
  --top-k 6
```

Use `raw_context` as the prompt-facing block. Use `items` for audit and UI.
If retrieved items carry `conflict_state: "conflicted"`, the response includes
`conflict_count`, `conflicts`, and a leading `=== Memory Conflicts ===` block so
agents can ask for confirmation instead of silently blending contradictory
claims.

### Conflicts

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  conflicts \
  --tenant-id openclaw \
  --namespace ops \
  --conflict-key active_branch
```

Lists unresolved scoped claim groups from persisted memory. Values include
record IDs, sources, memory states, conflict states, and text hashes.

### Resolve Conflict

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --audit-actor openclaw \
  resolve-conflict \
  --tenant-id openclaw \
  --namespace ops \
  --conflict-key active_branch \
  --accept-value dml-continuity-cleanup-2026-04-12
```

Accepts the chosen claim value and suppresses competing values in the same
scope. The command acquires the shared write lock, rewrites `dml_state.jsonl`
through the persistence layer, and appends an audit event.

### Resume

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path config/dml_portable_linux.yaml \
  --no-require-gpu \
  resume \
  --query "active continuity checkpoint compaction handoff resume next action"
```

Use this at agent boot, after context compaction, or after a harness restart.
The command prioritizes `active_continuity` memories and returns:

- `raw_context`: compact continuity handoff block
- `latest_checkpoint`: structured `thread`, `state`, `task`, `next_action`
- `continuity_items`
- `fallback_used`

## Harness Loop

1. Call `health`.
2. On startup or compaction recovery, call `resume`.
3. Before normal turns, call `retrieve` with the current task/query.
4. After meaningful state changes, call `ingest`.
5. Before risky maintenance or migration, call `backup` and `verify`.
6. Before shutdown/compaction, write a structured continuity checkpoint.

## Compatibility Rules

- Harnesses should treat DML as an external memory sidecar.
- Harnesses should not mutate `dml_state.jsonl` directly.
- The JSONL store is human-auditable but checksum protected; write through the
  wrapper or adapter only.
- Keep metadata JSON object shaped and explicit. Avoid hiding structured state
  only inside prose.
