# Continuity Checkpoint Contract

Status: draft
Scope: Packet 1 foundation only

## Canonical checkpoint path contract
- checkpoint directory: `<workspace>/out/dml-checkpoints`
- checkpoint filename: `<thread_safe_key>.md`
- `thread_safe_key` is derived from `thread_key` by:
  1. replacing each `/` and `:` with `_`
  2. removing characters outside `[A-Za-z0-9_.-]`
- writer emits the absolute checkpoint path on stdout
- registry stores the same absolute checkpoint path in `latest_checkpoint`
- finder resolves the canonical path first, then may fall back to legacy glob matches for backward compatibility

## Canonical continuity metadata schema

### checkpoint body
Required fields in the markdown header bullets:
- `thread`: original thread key
- `thread_key`: original thread key
- `thread_safe_key`: sanitized canonical key used in filename
- `checkpoint_path`: absolute emitted checkpoint path
- `updated_at`: RFC3339 UTC timestamp
- `checkpoint_policy`: `rolling_thread_checkpoint`

### registry entry
Required fields per thread entry:
- `key`: original thread key
- `thread_key`: original thread key
- `thread_safe_key`: sanitized canonical key
- `topic`: thread topic label
- `status`: current registry status
- `latest_checkpoint`: absolute canonical checkpoint path
- `last_summary_at`: RFC3339 UTC timestamp
- `updated_at`: RFC3339 UTC timestamp

## Recall contract
- recall resolves registry match by thread id first, then canonical key/topic matching
- retrieval status must be:
  - `hit` when registry match exists and `latest_checkpoint` is readable
  - `registry_only` when registry match exists but checkpoint file is missing/unreadable
  - `miss` only when no registry match exists
- no false `miss` is allowed when a registry match exists but checkpoint path contracts drift

## Compatibility rule
- finder may support older dated filenames during transition
- registry and new writer output must use the canonical undated path contract above
