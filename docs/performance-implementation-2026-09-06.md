# DML performance implementation and measurements

Date: 2026-09-06. This completes the eight implementation suggestions in the [original work queue](refactor-and-performance-2026-09-05.md). Measurements below are synthetic CPU benchmarks, not production service guarantees.

## Implemented changes

| Enhancement | Result | Activation |
|---|---|---|
| Batched index persistence | `PersistentVectorIndex.extend` validates a whole batch, performs one durable write, and rolls back on failure. Searches reuse a cached matrix. | Use `extend` for imports. |
| Incremental persistence and snapshots | Optional SQLite WAL storage updates changed/deleted rows, preserves record order and lineage, and creates periodic logical snapshots. Rollback copies retain native arrays instead of expanding them into Python float lists. | Native rollback copies are automatic. Journal storage requires explicit configuration/migration. |
| Scope indexes and matrix reuse | Exact tenant/client/session/instance buckets and bounded matrix caches avoid repeated whole-store filtering and stacking. Writes invalidate caches; returned candidates have their scope checked again. | Automatic for strict scoped retrieval and compatible merge paths. |
| Batched embeddings and ingestion | Native batches for Sentence Transformers and Ollama, with validated per-text fallback for other embedders. Adapter and provider batches commit lattice state once and roll back together. | Use `ingest_memory_batch`, `ProviderClient.remember_many`, or the batch CLI. |
| Reusable provider connections | `ProviderClient` keeps one HTTP connection pool across repeated requests. | Keep one client open for a workload. |
| Concurrent embedding miss sharing | Simultaneous identical query misses share one computation and its result or exception. Failed requests can be retried. Exact input strings remain distinct cache keys. | Automatic. |
| Bounded top-k selection | Recent-memory fallback and semantic-neighbor selection use bounded heaps when only a few results are required. | Automatic. |
| Approximate search for large scopes | Optional per-scope/per-kind Faiss HNSW indexes shortlist candidates before exact scoring and scope revalidation. Exact search remains the default and fallback. | Set `ann_min_items` above zero; install the Faiss extra. |

Batch requests accept at most 256 records and 1 MiB of text; embedding sub-batches default to 64. Explicit scope arguments take precedence over record metadata. Optional RAG mirroring reuses supplied embeddings when text is unchanged, but individual mirror backends retain their own persistence behavior.

## Measured results

Run on Windows 11, Python 3.12.14, NumPy 2.5.2, with one CPU math thread and no external embedding/model calls. Recall measurements use 20 queries, k=8, and fixed synthetic vectors. The reference performs broad candidate scanning over the same scope-equivalent corpus; every cached exact result matched the reference. See [raw results](performance-results-2026-09-06.json) for p50/p95/p99 and cold-build times.

| Exact recall corpus | Reference p95 | Cached p95 | Speedup |
|---|---:|---:|---:|
| 1,000 records, 1 tenant | 1.418 ms | 0.998 ms | 1.42x |
| 1,000 records, 64 tenants | 0.298 ms | 0.123 ms | 2.42x |
| 10,000 records, 1 tenant | 10.139 ms | 7.043 ms | 1.44x |
| 10,000 records, 64 tenants | 1.634 ms | 0.172 ms | 9.48x |
| 100,000 records, 1 tenant | 111.184 ms | 76.036 ms | 1.46x |
| 100,000 records, 64 tenants | 12.210 ms | 1.960 ms | 6.23x |

- Inserting 1,000 persistent-index records took **4,886 ms with individual inserts versus 11.168 ms in one batch**. Durable commits dropped from 1,000 to 1; serialized bytes dropped from 134,072,995 to 267,934.
- Updating one record in a 10,000-record store took **88.912 ms p95 for full JSON versus 48.900 ms for the journal** over ten updates. Each journal commit changed one record row.
- The 10,000-record rollback snapshot comparison took **309.028 ms with JSON expansion versus 105.878 ms with native arrays**.
- Optional HNSW at 100,000 records in one tenant took **0.480 ms warm p95**, with recall@8 of 1.0 on these 20 queries. Initial index construction took **13.4 seconds**. At 10,000 records, warm p95 was 0.343 ms and construction took 692 ms.

These measurements do not cover real provider throughput, GPU execution, concurrent service load, peak RSS, or process-kill recovery. Small sample percentiles and synthetic ANN recall should not be treated as production acceptance thresholds. The benchmark can be repeated with:

```text
python -m scripts.performance_benchmark --output performance-results.json
```

For comparable CPU results, set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python.

## Using batched ingestion and a persistent client

```python
from daystrom_dml.provider_client import ProviderClient

with ProviderClient("http://127.0.0.1:8765") as client:
    client.remember_many([
        {"text": "First note", "tenant_id": "team-a", "session_id": "project-1"},
        {"text": "Second note", "tenant_id": "team-a", "session_id": "project-1"},
    ])
    result = client.recall(
        "Find the notes", tenant_id="team-a", session_id="project-1"
    )
```

The client reads DML_API_TOKEN or DML_ADMIN_TOKEN, or accepts an explicit `token=`. Tokens remain trusted service credentials; they do not establish per-token tenant ownership. Supply the same primary scope tuple for writes and reads.

The equivalent import command reads a JSON array of records:

```text
dml remember-batch --input memories.json --batch-size 64
```

Native Ollama batching uses the documented input-array form of [POST /api/embed](https://docs.ollama.com/capabilities/embeddings). Oversized workloads should be split into deliberate checkpoints of at most 256 records.

## Optional journal storage

JSON remains the default. For an existing store, stop all writers and import its authoritative snapshot first. Substitute the actual storage directory below; if checksummed JSONL persistence is active, import that configured file instead.

```text
dml-journal import store/dml_store.json store/dml_state.sqlite3
```

Then enable the following settings in the DML configuration and restart writers:

```yaml
persistence:
  journal: true
  snapshot_interval: 128
```

The database must live in the adapter's storage directory as `dml_state.sqlite3`. Import refuses an initialized destination. Startup refuses implicit migration from an existing JSON/JSONL store, preventing an empty journal from silently replacing its data. No user data was migrated by this code change.

Use a local filesystem: [SQLite WAL requires processes on the same host](https://www.sqlite.org/wal.html). Transactions use FULL synchronous durability. The first commit and every configured interval create a logical full snapshot inside the database. Records and lineage are restored from transactionally consistent live tables.

Export an up-to-date standalone snapshot for backups or existing JSON tooling:

```text
dml-journal export store/dml_state.sqlite3 backup/dml_store.json
```

Old JSON files are not updated while journaling is enabled. To return to JSON storage, stop writers, export current journal state, and restore it through the appropriate JSON/JSONL recovery path before disabling journaling. Do not simply switch to an old snapshot.

The journal reduces disk rewriting; it still encodes the full in-memory payload to detect changes. Mutation rollback still copies store state. Further reductions in large-store allocation and lock duration require tracking dirty records at the mutation layer.

## Optional approximate search

```yaml
ann_min_items: 10000
ann_candidate_multiplier: 8
scope_cache_bytes: 67108864
```

`ann_min_items: 0` preserves exact search. HNSW operates on normalized vectors with inner-product similarity, following [Faiss's cosine-search guidance](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances). It preserves exact scope and kind boundaries, but approximate candidate selection can change ranking or omit neighbors. Validate recall against real data before enabling it.

Writes invalidate the ANN index, so frequent writes can make rebuilding expensive. HNSW indexes consume memory beyond the configurable matrix-cache byte limit. Exact mode is preferable for small scopes and workloads where construction cost or exact recall matters more than warm-query latency.

## Validation

- Full local suite: **875 passed, 17 skipped**, one existing Starlette/AnyIO deprecation warning.
- Regression coverage includes one-commit batching, failed-write rollback, embedding shape/count validation, concurrent miss success/failure sharing, exact scope equivalence, cache invalidation and size bounds, ANN isolation, journal replay and cross-adapter refresh, explicit migration/export protections, provider batching, and installed wheel contents.
- Local Ruff checks and maintained Mypy surfaces passed. Hosted CI now includes performance regressions with Faiss on Linux, macOS, and Windows.
- Hosted CI results and final merge references are recorded in the pull request; local benchmarks do not substitute for those checks.
