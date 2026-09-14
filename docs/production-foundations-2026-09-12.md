# Production foundations implementation

This is the first implementation tranche of the [ten-area plan](productionization-plan-2026-09-12.md).
The repository remains alpha. The [production contract](contracts/production-v1.md)
defines implemented behavior, migration and remaining release gates.

| Area | Implemented in this tranche | Still required |
| --- | --- | --- |
| 1. Production contract | Small explicit target; machine-readable maturity and honest health | Receipt-based stable profile and release graduation |
| 2. Adapter decomposition | Extracted lattice persistence, cache, compaction, lifecycle predicates and retrieval evidence; journal/checkpoint services strengthened | Persistence transaction coordinator, complete retrieval/lifecycle extraction |
| 3. Crash consistency | Strict startup; checksummed journal states/decisions; seven abrupt process-kill hooks; disk quota, corruption and serialization cases | All component/mutation hooks, projection outbox, hardware/power-loss matrix |
| 4. Adversarial memory | Eight-case versioned corpus, scoped canaries, restart checks and provenance/trust/expiry guards | Natural-language conflict adjudication and real-agent outcomes |
| 5. Baseline | Independent durable SQLite + embeddings + recency + top-k + compaction implementation and runner | Held-out live episodes, equalized compaction and statistically supported value gate |
| 6. Agent outcomes | Validated terminal-event reducer with failure-inclusive token costs, quality rates, TTFT/latency/overhead/recovery distributions and growth observations | Wire all agent harnesses; actual 1k/10k/100k-turn agent campaigns |
| 7. Stability boundaries | Provider contract inventory; native KV remains experimental; exact rendered budget boundary | Full runtime/model/tokenizer binding audit and hardware compatibility canaries |
| 8. Version migration | Journal schema 1, record versions, unknown-version rejection, explicit schema-0 side-by-side upgrade and commit-pinned fixture | More actual release fixtures and all persistent artifact families |
| 9. Concurrency | Journal CAS/read snapshots, owned scoped reads/background aging, cache invalidation epochs and 256-client journal stress | Multi-process/provider stress campaigns, lock-latency SLOs, moving remaining model work outside locks |
| 10. Observability | Atomic journal decision history; retrieval/suppression/context digests; checkpoint and provider degradation | Durable full request replay, complete authority/promotion reasons, bounded retention/export |

## Validation commands

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DML_SKIP_VENV_REEXEC=1 \
  python -m pytest dml_core/daystrom_dml/tests dml_core/tests integrations/hermes/tests openclaw-wrapper/tests
DML_STRESS_CLIENTS=256 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest dml_core/tests/test_production_foundations.py::test_concurrent_clients_use_compare_and_swap_without_lost_updates
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.production_baseline --turns 1000 --output production-artifacts/baseline.json
python -m scripts.production_baseline --task-events /path/terminal-events.jsonl --output production-artifacts/agent-outcomes.json
```

The baseline runner supports 10,000 and 100,000 turns, but does not invoke a model.
Only metrics from actual terminal task events may be labeled agent outcomes.
Each event uses `schema_version: dml-task-outcome-v1`, unique `episode_id` and
`task_id`, verified boolean `success`, input/output/maintenance token totals,
`latency_ms` and `retrieval_ms`. Optional measurements include `ttft_ms`,
`repeat_errors`/`repeat_opportunities`, `contradictions`/`factual_outputs`,
`false_memory_claims`/`recalled_claims`, `store_bytes`, `memory_records`, and
`recovery_ms`. Missing quality denominators yield null, never an invented zero.

## Recorded local evidence (2026-09-12)

On Python 3.12.14, the complete core, Hermes and wrapper suite passed: **931 passed,
9 skipped** (optional integrations), with one upstream Starlette deprecation warning.
The separate **256-client** journal CAS stress test passed without lost updates.
Ruff and the maintained mypy selection, expanded to include services (42 files),
passed; local mypy used `--python-version 3.12` to match installed NumPy stubs.
CI retains Python 3.10/3.11 type checks and Linux/macOS/Windows portability coverage.
The Hermes hygiene smoke also passed. Remote CI results remain separate evidence.

The [recorded 1,000-turn offline comparison](artifacts/production-baseline-2026-09-12.json)
found the expected evidence in 1,000/1,000 turns for both systems. Baseline retrieval
p95 was **1.46 ms**, versus **3.21 ms** for DML in this local run. DML has not earned
a performance advantage in this smoke. These single-run measurements over 100
fixed records are not statistical estimates of production performance or agent
quality. No 10k/100k real-agent campaign has been run.

No user memory stores were migrated and no runtime/GPU services were contacted.

## Serial receipt follow-up

The [append-only receipt contract](receipt-hardening-2026-09-12.md) adds an opt-in
schema-2 journal and a narrow ingestion service. It advances areas 1, 2, 3, 7, 8,
9 and 10: scoped historical receipts, atomic receipt/decision persistence, explicit
side-by-side migration, embedding compatibility, process-death/concurrency evidence
and retryable API outcomes. Stable APIs remain empty in the maturity inventory.
Receipts for other lifecycle mutations and transactional external projections
remain separate release gates. The original ten-area plan remains the task scope.


## Serial projection follow-up

The [disposable SQLite projection](projection-hardening-2026-09-14.md) advances
P03 with atomic full-snapshot publication and pinned-source queries. It reuses
the accepted journal format, keeps source writes independent of backend outages,
and adds crash/concurrency/query regression evidence. Incremental outboxes, existing
FAISS qualification and automatic provider routing remain separate gates.


## Serial delta delivery follow-up

[Coalesced projection deltas](projection-delta-hardening-2026-09-14.md) add checked
base/target publication, a narrow backend protocol and explicit adapter/CLI integration.
Receipt availability remains independent of backend I/O. Full-state verification
and complete ID manifests remain necessary; durable operation outboxes, workers
and existing FAISS backend qualification are still separate gates.


## Serial worker follow-up

[Bounded projection retry scheduling](projection-worker-hardening-2026-09-14.md)
adds explicit adapter-owned delivery with coalesced notifications, capped failure
backoff, terminal shutdown and historical pinned status. Process-death recovery
derives pending work from durable source/target state. Receipt acknowledgements
remain independent of backend availability. Durable operation outboxes, growing
store costs and qualification of the existing FAISS backend remain separate gates.


## Serial transactional outbox follow-up

[Transactional operation events](outbox-hardening-2026-09-14.md) add opt-in journal
schema 3 with full-state events committed beside memory/receipts/decisions, verified
historical prefixes, and ordered idempotent delivery to a dedicated SQLite consumer.
Existing schema-1/2 authorities are preserved. Explicit migration, remaining public
lifecycle receipts, bounded history retention and ordered background scheduling
remain separate serial gates.

## Serial outbox migration follow-up

[Explicit schema-2 to schema-4 migration](outbox-migration-hardening-2026-09-14.md)
preserves receipt and decision bytes, records an honest baseline for previously
unavailable full-state history, and adds versioned consumer adoption. Seven actual
process-kill boundaries, commit-pinned legacy reader/state fixtures and simultaneous
consumer delivery qualify the migration path. Offline manual cutover remains required.
The next serial gate is receipt coverage for remaining public lifecycle mutations;
history retention, ordered background delivery and real-agent value remain open.
