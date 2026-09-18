# Production foundations implementation

This is the first implementation tranche of the [ten-area plan](productionization-plan-2026-09-12.md).
The repository remains alpha. The [production contract](contracts/production-v1.md)
defines implemented behavior, migration and remaining release gates.
The current [finite remaining-work ledger](production-remaining-work-2026-09-18.md)
contains eleven first-release milestones and two deferred broader milestones;
three milestones have reviewed-source acceptance, leaving eight first-release milestones
and two deferred milestones.

The table summarizes the foundations and subsequent serial gates. Historical
sections below retain the evidence and scope recorded when each gate was built.

| Area | Implemented through the serial gates | Still required |
| --- | --- | --- |
| 1. Production contract | Explicit target, maturity/health, durable receipt APIs and revision-pinned scoped retrieval/context integration; reviewed supported-profile freeze and exact-input companion | Complete broader release qualification |
| 2. Adapter decomposition | Narrow persistence, cache, scoped selection, context/report, lifecycle, receipt, projection, delivery and retention-inspection services; reviewed transaction coordination extraction | Remaining legacy retrieval/lifecycle extraction is deferred beyond the first release |
| 3. Crash consistency | Verified journals/checkpoints, atomic receipts/outbox, explicit migrations, real process-kill, quota, corruption and serialization cases | Remaining mutation/component inventory, supported filesystem/power-loss qualification and recovery runbook |
| 4. Adversarial memory | Eight-case corpus plus lifecycle-specific conflict, scope, authority, provenance and historical-replay regressions | Real-agent semantic outcomes and long-running incorrect-retrieval feedback cases |
| 5. Baseline | Independent durable SQLite + embeddings + recency + top-k + compaction implementation and runner | Held-out live episodes, equalized compaction and statistically supported value gate |
| 6. Agent outcomes | Outcome reducer, failure-inclusive costs, quality/latency/recovery distributions, CI offline baseline and journal-history cost evidence | Wire real agent harnesses; continuous 1k/10k and 100k-turn release campaigns |
| 7. Stability boundaries | Provider contract inventory, reviewed supported-profile boundary and exact model/tokenizer input companion; native KV remains experimental | Release qualification; native-KV restore identity audit and hardware canaries are deferred |
| 8. Version migration | Explicit side-by-side schema 0/1/2/4 paths, schema-3 fresh authorities, preserved historical receipts and commit-pinned compatibility fixtures | More released-version fixtures and coverage of all persistent artifact families |
| 9. Concurrency | Journal CAS/pinned reads, 256-client receipt/lifecycle/delivery tests, competing processes and deterministic race regressions | Mixed-operation HTTP/provider campaigns, cancellation/starvation checks and lock-latency SLOs |
| 10. Observability | Durable mutation decisions/source proofs, response retrieval traces, degradation status and scoped retention inspection | Durable full request/context replay, complete decision coverage and bounded audit retention/export |

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
Later serial gates added retirement, supersession, content correction and
first-level promotion/merge receipts, plus reference projection delivery. External
backend qualification remains outside those bounded results. The original ten
areas remain workstreams; the finite ledger defines the current release scope.


## Serial projection follow-up

The [disposable SQLite projection](projection-hardening-2026-09-14.md) advances
P03 with atomic full-snapshot publication and pinned-source queries. It reuses
the accepted journal format, keeps source writes independent of backend outages,
and adds crash/concurrency/query regression evidence. Later gates added incremental
delivery and durable outboxes. Existing FAISS qualification and automatic provider
routing were not established by this gate.


## Serial delta delivery follow-up

[Coalesced projection deltas](projection-delta-hardening-2026-09-14.md) add checked
base/target publication, a narrow backend protocol and explicit adapter/CLI integration.
Receipt availability remains independent of backend I/O. Full-state verification
and complete ID manifests remain necessary. Later gates added workers and durable
operation outboxes; existing FAISS backend qualification remains unproven.


## Serial worker follow-up

[Bounded projection retry scheduling](projection-worker-hardening-2026-09-14.md)
adds explicit adapter-owned delivery with coalesced notifications, capped failure
backoff, terminal shutdown and historical pinned status. Process-death recovery
derives pending work from durable source/target state. Receipt acknowledgements
remain independent of backend availability. The following gate added durable
operation outboxes. Growing-store costs and qualification of the existing FAISS
backend were not established by this worker gate.


## Serial transactional outbox follow-up

[Transactional operation events](outbox-hardening-2026-09-14.md) add opt-in journal
schema 3 with full-state events committed beside memory/receipts/decisions, verified
historical prefixes, and ordered idempotent delivery to a dedicated SQLite consumer.
Existing schema-1/2 authorities are preserved. Later gates added explicit migration
and the supported lifecycle receipts. Bounded audit retention and ordered
background scheduling were not established by this gate.

## Serial outbox migration follow-up

[Explicit schema-2 to schema-4 migration](outbox-migration-hardening-2026-09-14.md)
preserves receipt and decision bytes, records an honest baseline for previously
unavailable full-state history, and adds versioned consumer adoption. Seven actual
process-kill boundaries, commit-pinned legacy reader/state fixtures and simultaneous
consumer delivery qualify the migration path. Offline manual cutover remains required.
The subsequent retirement through promotion gates added the supported lifecycle
receipt coverage. Audit retention and real-agent value remain in the finite
release ledger; this migration did not qualify ordered background delivery.

## Serial retirement follow-up

[Receipted memory retirement](retirement-hardening-2026-09-14.md) adds the first
qualified lifecycle mutation to the receipt profile: scoped tombstones guarded by
the exact current record digest, atomic historical receipts and outbox decisions,
and normal retrieval suppression. It retains historical content and capacity.
Subsequent gates added supersession, content updates and first-level promotion/merge.
Physical erasure remains unsupported and is excluded from the finite release plan.

## Serial supersession follow-up

[Receipted supersession](supersession-hardening-2026-09-14.md) extends lifecycle
receipt coverage to explicit same-scope replacement links. Both exact record
digests guard the decision, only the old memory is changed, and normal retrieval
suppresses it without modifying replacement trust. This is the tenth serial gate,
not completion of the ten original areas. Later gates added content updates,
first-level promotion/merge and retention inspection. Full mutation qualification,
decision replay and real-agent value at growing-store scale remain in the finite
ledger; physical erasure is excluded and native-KV qualification is deferred.

## Serial content-update follow-up

[Receipted content correction](content-update-hardening-2026-09-16.md) is the
eleventh serial hardening gate. Exact record and embedding-space checks guard
atomic text/vector updates while preserving scope, trust, lifecycle and creation
time. Preparation runs outside ownership and historical retries bypass the model.
Later gates added first-level promotion/merge and retention inspection. Broader
qualification and replay remain in the finite ledger; physical erasure is excluded
and native compatibility is deferred.

## Serial first-level promotion/merge follow-up

[Receipted promotion/merge](promotion-hardening-2026-09-17.md) is the twelfth
serial hardening gate, extending P03's durable mutations and P07's explicit
lifecycle boundaries. The caller selects base memories and supplies derived text;
one atomic append preserves every source and records complete source provenance.
Multi-source merges respect `no_merge`; output scope/trust remain unchanged and
ranking attributes cannot increase. First-level derivation is an independent
snapshot with its own lifecycle. The next gate added retention inspection.
Recursive promotion, cascading invalidation and physical erasure are excluded
from the finite first-release scope; broader production qualification remains open.

## Serial retention-inspection follow-up

[Verified retention inspection](retention-inspection-hardening-2026-09-17.md)
is the thirteenth serial gate. One pinned verified transaction counts known
same-scope memory and first-level source-proof occurrences across current records,
lineage, the archived journal snapshot, receipts and outbox states. Historical-only
records remain inspectable without exposing payloads or receipt keys. The static
contract and report explicitly state that retirement does not erase content and
that physical erasure is unsupported. No purge, history compaction or capacity
reclamation is implemented; those require a separately versioned history/retry
contract and external-copy policy. This is a bounded operator-evidence gate within
the original workstreams, not completion of their production qualification.

## Serial scoped retrieval/context follow-up

[Scoped retrieval and context services](retrieval-context-hardening-2026-09-17.md)
are the fourteenth serial gate, advancing P06 and areas 1, 2 and 9. Frozen resolved
inputs drive scoped selection and response evidence; narrow services handle recent
fallback, suppression, ledger lookup and detached context reports. The existing
store retains ranking, and the adapter retains model preparation, ownership,
routing, DPM and metrics. Baseline characterization preserves exact rendered output
and evidence across legacy, schema-1 and receipt-schema-2/3/4 paths. Receipt reads
retain ownership through report construction, with explicit embedding-identity
checks after preparation. Exact model-input budget binding and durable replay
remain first-release gates; remaining legacy retrieval/lifecycle extraction is
deferred beyond that release. Maturity
remains candidate; this does not qualify the whole repository for production.

## Serial transaction-coordinator follow-up

The reviewed [transaction and persistence coordinator gate](transaction-coordinator-hardening-2026-09-18.md)
is the fifteenth serial gate and milestone 2 of the finite release ledger. It
separates ownership/nesting/rollback orchestration from component
snapshot/restore/refresh/commit behavior, retaining adapter compatibility shims.
It also addresses caught publish-then-error outcomes in legacy persistence.
Receipt-schema-2/3/4 authority and retry semantics remain unchanged. Compensating
legacy file writes do not provide crash atomicity across components. Final
independent review accepted 9.6/10; 164 focused tests, 256-client stress and
2,948 full-suite tests passed, with 9 full-suite skips. Evidence and qualification
limits are recorded in the gate's hardening record;
this work alone does not close the remaining release qualification milestones.

## Serial supported-profile follow-up

The [supported-profile freeze](production-profile-hardening-2026-09-18.md) is the
sixteenth serial gate and closes milestone 1 at the reviewed-source gate. The explicit
[`dml-receipted-local-v1` candidate](supported-production-profile-v1.md) fixes
admitted runtime APIs, receipt-journal authority, trusted caller scope, strict
configuration, dependency declarations, limits and retry behavior. Its HTTP
surface admits nine routes with explicit tenant input and startup-bound service
credentials. Experimental features, legacy RAG files, generation, semantic
checkpoints and backend projection operations are outside that selected runtime.
Profile admission on Linux/macOS/Windows and CPython 3.10–3.13 is separate from
demonstrated portability and filesystem/power-loss qualification. Independent
final review accepted **9.6/10**, with **441 focused tests passing**. The final root
full suite passed **3,342 tests**, with **9 skips** and **3 warnings**, in 163.26
seconds; maintained/new-surface Ruff, the 63-file mypy selection, Hermes hygiene
and the diff check passed. The [review record](artifacts/production-profile-review-2026-09-18.json)
and hardening document retain the evidence and limits. Publication and exact-commit
CI remain pending at this source snapshot; outcomes will be recorded in PR #118.
Eleven milestones remain: nine for the first release and two deferred. Exact
model-input and tokenizer budget binding is next; no maturity promotion occurred.

## Serial exact model-input follow-up

Before this gate began, the published stage-16 source `dc85123` passed all nine
jobs in CI run 315 (`35341742222`). Both Linux full-suite jobs passed 3,343 tests
with 8 skips, the production lane passed 394 profile tests, and all six portability
jobs passed. The tested PR merge tree matched the published source tree. These
remote results supplement the earlier local source snapshot; they do not qualify
the later stage-17 implementation or imply that PR #118 has merged.

The [exact-input companion](model-input-hardening-2026-09-18.md) is the seventeenth
serial gate and addresses milestone 3. It verifies a local GPT-2 snapshot, pins the
real tokenizer and fixed messages/tools template, and rejects an oversized complete
input plus output reservation before inference. Its owning consumer dispatches the
immutable compiled token IDs directly. A dedicated pinned CPU lane exercises a real
tiny model and tokenizer with zero permitted skips. Independent review accepted
**9.6/10**, with **256 focused tests passing and 0 skips** in 5.59 seconds.
The final root full suite passed **3,598 tests**, with **9 skips** and **3 warnings**,
in **146.54 seconds**; all 256 new model-input cases passed without skips.
Maintained/strict new-surface Ruff, the 66-file mypy selection, Hermes hygiene and
the diff check passed. The [review record](artifacts/model-input-review-2026-09-18.json)
retains source hashes and separately attributed evidence. Milestone 3 is closed
at the reviewed-source gate. Publication and exact-commit CI remain pending at this
source snapshot; outcomes will be recorded in PR #118. The ledger has ten remaining
milestones: eight first-release and two deferred. Crash/filesystem qualification
is next. The memory profile retains its nine HTTP routes and no
generation, and retrieval token counts remain labeled estimates.
