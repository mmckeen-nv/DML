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

### Ingest

```bash
python scripts/dml_memory.py \
  --storage-dir "$DML_STORE" \
  --config-path config/dml_portable_linux.yaml \
  --no-require-gpu \
  ingest \
  --kind note \
  --meta '{"source":"harness","namespace":"active_continuity"}' \
  --text "..."
```

Required metadata for beta integrations:

- `source`: harness or component name
- `kind`: set by `--kind`; one of `action`, `observation`, `note`, `plan`,
  `error`, `artifact`
- `namespace`: logical memory lane, such as `active_continuity`

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
