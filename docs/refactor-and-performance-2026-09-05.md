# DML bug fixes and performance work queue

Date: 2026-09-05. Base: `8f42a04` on `mmckeen-nv/DML/main`.

The earlier review commit `ba5f965` and its document were unavailable in the accessible repository. This work verified the supplied findings against the current checkout and reproduced the remaining defects with regression tests.

## Fixes

- Memory merging now filters candidates by tenant, client, session, instance, thread, project, relationship, kind, and phase before comparing similarity. Both incoming and existing `no_merge`/`merge_policy=never` records are protected. Selection and mutation now share one lock; duplicated merge-search code was removed.
- Explicit scope and kind arguments take precedence over arbitrary metadata in the provider and adapter write APIs.
- Scoped recall now matches all four primary scope fields exactly in both semantic retrieval and recent-memory fallback. An omitted session/client/instance means the unassigned scope. Scoped recall no longer falls back to legacy unscoped data. The lower-level `retrieve_filtered` API retains its existing broad-search default; the service adapter explicitly requests strict scope.
- The provider now uses the existing optional bearer-auth middleware. The CLI reads `DML_API_TOKEN` (or `DML_ADMIN_TOKEN`); the browser has a password field held only in the current page. Public health, documentation, and static assets remain accessible. Non-ASCII invalid credentials no longer cause a comparison exception.
- Query embeddings are cached by the exact input, preserving case and whitespace. LRU reads, writes, eviction, and clearing are synchronized without holding the cache lock during embedding calls.
- Windows credential persistence establishes a current-user-only ACL on the empty temporary file before writing the secret. ACL failures abort before writing bytes. POSIX permissions remain 0600.
- Installed distributions now include the default configuration and vLLM bridge package. CPU-only builds produce a portable pure-Python wheel. Both metadata definitions include the workload/KV command entry points and development build dependency.
- The background worker can execute Python workers with the active interpreter and explicitly launches Bash scripts on Windows when Bash is installed.
- Tests use an isolated dummy adapter for metrics, shared Hermes host stubs independent of collection order, and serialized writes in the stress-test fake store. Symlink tests skip only when Windows lacks the required privilege; the Bash queue integration is marked POSIX-only.
- CI now runs the complete core, provider, context, Hermes, and OpenClaw test directories, including the new scope/cache and wheel-content regressions.

## Compatibility and limits

Scoped callers must supply the same tenant/client/session/instance tuple used when storing memories. Existing callers that intentionally searched all sessions through `retrieve_context` must select the intended session explicitly. Legacy unscoped memories remain available through the local unscoped adapter API; migrate their ownership explicitly if they should be exposed through a scoped provider.

Bearer tokens authenticate access to the service; they do not map individual credentials to individual tenants. A valid service token remains a trusted credential capable of selecting scope. Authentication remains disabled when neither token is configured, preserving local operation.

These changes prevent new cross-scope merges; they cannot reliably untangle records that were already merged. No existing memory data or repository branches were deleted. The experimental vLLM connector still requires its documented runtime and GPU dependencies.

## Prioritized performance enhancements

The following are recommendations based on the current implementation, not measured speedup claims. Preserve scope boundaries, rollback behavior, and acknowledged-write durability while implementing them.

| Priority | Enhancement | Evidence in the code | Expected benefit | Verification |
|---|---|---|---|---|
| 1 | Batch persistent-index inserts into one durable commit | `persistent_index.py: PersistentVectorIndex.extend` repeatedly calls `add`, which rewrites the index each time | Reduce repeated serialization and disk writes during imports | Compare 1/100/1,000-document batches; count commits, bytes written, and elapsed time; inject commit failures |
| 1 | Incremental journal plus periodic snapshots | `dml_adapter.py: _capture_mutation_snapshot` copies complete store state; `_persist_dml_state` writes complete state per mutation | Reduce write latency, allocation pressure, and lock hold time as memory grows | Ingest at 1k/10k/100k records; measure p50/p95, peak memory, and recovery time; test crash/replay and rollback |
| 1 | Index exact scopes and reuse embedding matrices | `memory_store.py: retrieve_filtered`, `_best_match`, and `_score_candidates` scan/filter records and rebuild matrices; `persistent_index.py: search` stacks vectors per query | Reduce work per scoped recall and repeated matrix allocations | Vary tenant count and records per tenant; compare exact result IDs, p95 latency, allocations, and cross-scope canaries |
| 2 | Batch embedding calls for ingestion | Adapter write paths call `embedder.embed` for individual records | Reduce provider round trips and improve accelerator utilization | Measure documents/second, provider request count, memory, and batch failure/retry behavior with the actual embedding backend |
| 2 | Keep long-lived provider clients/adapters for repeated work | Provider CLI constructs a new HTTP client per invocation; wrapper workflows can launch separate processes | Reduce connection setup and repeated store/model initialization | Compare CLI-per-operation with persistent-service usage under identical durability and recall settings |
| 2 | Coalesce identical concurrent embedding misses | The bounded query cache is now synchronized, but concurrent misses can still compute the same embedding | Avoid duplicate provider work during bursts | Burst identical prompts; count backend calls, verify cancellation/error propagation, and preserve exact input keys |
| 2 | Use bounded top-k selection in remaining full-sort paths | `memory_store.py: _semantic_neighbors` sorts all candidates; `dml_adapter.py: _recent_context_items` sorts all recent candidates | Reduce CPU and temporary allocations when only a handful of results are needed | Compare ordering/ties and top-k recall at k=4/8/32 against current results |
| 3 | Scope-aware approximate vector search for large stores | Current lattice similarity scoring uses exact matrix scans | Reduce search time at large record counts | Establish exact-search baselines first; compare recall@k, p95 latency, index size, update cost, and mandatory pre-ranking scope filters |

Recommended sequence: establish a repeatable benchmark, implement batched persistence, then scope indexes/matrix reuse, then embedding batching. Add approximate search only after exact-search costs are measured. Existing query caching and vectorized scoring already exist; the opportunities above extend those mechanisms.

## Benchmark acceptance criteria

Use fixed corpora at 1k, 10k, and 100k records; k=4/8/32; one and many tenants; cold and warm queries; and 1/4/16 concurrent clients. Record p50/p95/p99 latency, throughput, peak RSS, embedding request count, bytes written, lock wait, and recall@k. Separate deterministic CPU measurements from real embedding-provider/GPU measurements. Every optimization must retain exact scope isolation and pass durability/failure-injection checks.

## Validation

- Full suite: **854 passed, 17 skipped**, one Starlette/AnyIO deprecation warning (Windows, Python 3.12).
- Added 25 regression cases covering scope/merge boundaries, metadata precedence, provider authentication, cache behavior, Windows ACL failure, and installable package contents. The original 18 scope/auth cases all failed before the fixes.
- Ruff passed for changed Python files and the existing maintained CI surfaces.
- Mypy passed for all 35 maintained files using `--python-version 3.12`. The local NumPy release uses Python 3.12 type syntax, so the repository's Python 3.10 target cannot parse that installed dependency's stubs; CI's Python 3.10/3.11 dependency resolution remains separate.
- Provider JavaScript syntax check, Hermes hygiene smoke, and `git diff --check` passed.
- CPU wheel build produced `cma-0.1.0-py3-none-any.whl`; package-content regression passed, and clean installed-package imports/default configuration were verified.
- Skips include Windows symlink privilege requirements, the POSIX Bash integration, and optional runtime/platform checks. GPU execution and the live vLLM/embedding-provider paths were not validated locally. Hosted CI results are separate from these local results.
