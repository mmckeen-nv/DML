# DML productionization plan

Date: 2026-09-12. Repository: `mmckeen-nv/DML`.
Reviewed code: `f28c7475d5d4a8c455c26a0d159cfb8fa6844d5c`.
Status: implementation backlog and proposed release gates; this document does not declare existing alpha features production-ready.

## Repository consolidation

All five non-main branch tips are already ancestors of the reviewed `main`. There are no open pull requests and no outstanding commits on these branches to merge. Git ancestry, remote heads, and object connectivity were checked. Existing merge commits and history remain intact; no squash, rebase, force push, or runtime code change is needed.

| Branch | Preserved tip | Commits absent from main |
| --- | --- | ---: |
| `feat/daystrom-context-manager` | `1c792fd38e0b8c6b15365d8239dd4e7232e3a025` | 0 |
| `feat/dcm-kv-fabric-contract` | `bcf9cfd930fd3d9b54b14997c686ad65f62c2d9c` | 0 |
| `fix/current-bugs-and-packaging` | `ad840e85776864f9071b60a0cfc2f4872fd4726a` | 0 |
| `fix/glm-review-remediation` | `25cf49ef5273de52ef4d68bd210861b3722599e1` | 0 |
| `perf/complete-performance-work-queue` | `48b2ba21ebc4f2b479b72062557afdf9b4398be4` | 0 |

The five remote branch names remain. The available GitHub connection supports commits but exposes no branch-deletion operation; command-line push authentication is unavailable. This limits branch-name cleanup, not code consolidation. If retiring them later, fetch again, verify each current tip is still an ancestor of current `main`, and delete only with an expected-tip lease. Stop if any tip moved. The recorded SHAs allow recreating those branch names from preserved history.

The reviewed main commit has [successful GitHub CI](https://github.com/mmckeen-nv/DML/actions/runs/34086858946). No existing user memory stores were opened or migrated.

Local validation on Linux/Python 3.12: **104 passed**, one upstream Starlette/AnyIO deprecation warning. The focused selection covers `test_atomic_io`, `test_store_lock`, `test_fail_closed_persistence`, `test_persistence`, `test_performance`, `test_scope_regressions`, `context/test_page_catalog`, `context/test_execution_checkpoints` and `context/test_kv_fabric`. An isolated environment used the server/faiss/mcp/dev/tokenizer extras; `socksio` was added for this host's proxy environment after two initial client-construction failures. No repository dependency changes were made. The full suite, live inference/GPU behavior, power-loss recovery and future production gates were not rerun or newly proven by this documentation change.

## What the code already provides

Paths below are relative to the repository. Existing mechanisms should be extended rather than replaced wholesale.

| Area | Current implementation | Remaining production gap |
| --- | --- | --- |
| Orchestration | `dml_core/daystrom_dml/dml_adapter.py`, 3,380 lines; facade, retrieval, persistence, lifecycle, context and policy wiring | Responsibilities and private-state access remain tightly coupled. |
| Durable files | `atomic_io.py`, `persistence.py`, `persistent_index.py`; atomic replacement, fsync, checksums, rollback tests | Multi-component process-kill atomicity and platform-specific durability need explicit guarantees. Directory synchronization has platform-dependent behavior. |
| Incremental state | `journal.py`: optional SQLite WAL, FULL synchronous commits, live records/lineage and logical snapshots | A transaction covers that database, not all adapter side effects. Database revision is not a schema migration policy. |
| Mutation ownership | `store_lock.py`, adapter `_mutation_transaction`, `atomic_batch`, `_rollback_mutation` | Existing compensating writes handle caught errors; a killed process cannot run compensation between component commits. Read snapshots and all background writers need one ownership model. |
| Memory formats | `MemoryItem.to_dict()`, JSON/JSONL, journal, RAG manifests, checkpoint files | JSONL writes version 1 but `load_state()` does not reject unsupported header versions. Memory records lack an explicit schema version. |
| Native checkpoint safety | `context/checkpoints.py`, `context/kv_fabric.py`, runtime adapters and bridge tests | Strong identity checks already exist. Prove every restore route enforces the complete model/tokenizer/runtime/topology/layout/position/context boundary. |
| Checkpoint operations | `checkpoint.py` | Second-resolution filenames can collide; the background loop catches errors without surfacing checkpoint failure there. Audit and test both behaviors. |
| Evaluation and telemetry | `metrics.py`, wrapper recall/stress harnesses, DCM workload benchmark, performance scripts | Synthetic retrieval speedups and small deterministic suites do not establish long-running agent benefit or comprehensive decision replay. |

See [existing performance implementation](performance-implementation-2026-09-06.md), [prior work queue](refactor-and-performance-2026-09-05.md), and [agent test plan](../TEST_PLAN.md). Keep their historical results; this plan supplies new acceptance gates rather than treating older claims as current proof.

## 1. Define the smallest production contract

Proposed stable v1: **DML accepts scoped, attributed memory durably and returns bounded, attributable context from a specified committed state, with explicit recovery and compatibility outcomes.**

The first supported production profile should be a single host, local filesystem, one authoritative transactional store, and cooperating trusted clients through the provider or documented adapter API. Reuse the existing SQLite journal as the starting implementation. Do not silently change existing JSON installations or claim network-filesystem/multi-host writer support. Existing formats remain explicit import/export and compatibility paths until separately qualified. Current service tokens can select scopes; they do not establish per-token tenant ownership. Untrusted multi-tenant access requires a separate authenticated-principal-to-scope authorization contract before being supported.

| Guarantee | Required behavior and acceptance test |
| --- | --- |
| Durable ingestion | Validate before mutation. Return a receipt containing memory IDs, operation/idempotency key, scope and committed revision only after the authoritative transaction commits. Restart must retain every acknowledged operation or stop with a specific integrity failure. A rejected operation must not become visible. |
| Uncertain outcomes | A disconnect or crash after commit but before reply is an unknown outcome, not a rejection. Retry with the same key returns the original receipt; changed content under the same key is rejected. Never infer success from an empty error response. |
| Reproducible retrieval | Inputs include exact query or vector, embedding identity, scope, store revision, policy version, effective clock, filters, top-k and token budget. Exact mode uses a deterministic tie-breaker. Record these inputs and returned IDs; reproduce IDs/order for the pinned backend and numeric environment. ANN remains experimental and must not promise exact reproducibility. |
| Bounded context | Count the final assembled model input with the pinned tokenizer/chat template, including system/tool framing, and reserve output tokens. Return a typed budget error if mandatory content alone cannot fit. Never silently substitute a different tokenizer or random embeddings in the stable profile. |
| Honest recovery | Healthy, degraded-read-only, recovery-required and unavailable states have explicit permitted operations. Corruption must never appear as an empty successful store. Missing state at a previously initialized location is an error, distinct from explicit new-store creation. |
| Provenance and compatibility | Retain source identity, trust, valid time, scope, lineage and schema version. Unknown formats and incompatible checkpoints fail closed. Native cache loss may fall back to fresh inference only after explicit rejection and a recorded fallback decision. |

“Never invent memory” means no fabricated persisted records, provenance, or unsupported authority changes. It cannot guarantee that an imported claim or an LLM summary is true. Preserve original evidence under the retention policy and label derived summaries; semantic correctness is measured separately. Explicit deletion, retention expiry and capacity eviction must have durable tombstones/reasons, not silent loss.

Deliver `docs/contracts/production-v1.md`, versioned receipt/query/result/error schemas, a stable profile, and executable contract tests. Stable v1 must work with experimental modules disabled.

## 2. Break up DMLAdapter incrementally

Keep public adapter signatures as a compatibility facade during extraction. Add dependency-injected protocols under a small `daystrom_dml/services/` package; reuse existing stores and context objects behind them. Avoid introducing a second parallel architecture or a generic plugin framework.

| Interface | Owns | Existing code to wrap/extract |
| --- | --- | --- |
| `MemoryRepository` / transaction | Commit, read snapshot, revisions, idempotency, tombstones and recovery | Adapter persistence/rollback methods, `journal.py`, `persistence.py`, `store_lock.py` |
| `RetrievalService` | Embedding identity/cache, scoped candidate selection, ranking and suppression reasons | Adapter query methods, `memory_store.py`, `persistent_index.py`, RAG stores |
| `MemoryLifecycleService` | Ingest validation, deduplication, conflict/supersession, promotion, decay, lineage | Adapter memory methods, `memory_store.py`, `maintenance.py`, `agent_schema.py` |
| `ContextBuilder` | Pure construction from a snapshot plus explicit policy/tokenizer/budget | Adapter context assembly, `context/`, `inference/`, STM integration |
| `CheckpointService` | Semantic snapshots, native checkpoint compatibility and lifecycle through separate typed methods | `checkpoint.py`, `context/checkpoints.py`, native runtime adapters |
| `DecisionRecorder` | Versioned audit events, decision manifests, metric emission | `metrics.py`, current wrapper/curation evidence and context manifests |

First capture golden public behavior for ordering, scope, receipts/errors and context budgets. Extract one responsibility per PR with unchanged behavior; then implement semantics in later PRs. Workers must call the same services and transaction boundary. No service may reach into another service's private lock or mutable store. LLM, embedding and vector-network calls occur outside the writer lock; commit checks the revision they were computed against.

Completion: the adapter contains composition and sequencing, no direct persistence serialization, ranking loops, promotion mutations or checkpoint-compatibility policy. Compare against golden outputs and check imports/API snapshots. Line count is a review aid, not the acceptance criterion.

## 3. Make crash consistency a release gate

Choose one authoritative commit boundary. Store memory, lineage, idempotency receipts, lifecycle changes and a decision/outbox record in the same transaction. Treat vector indexes and external RAG mirrors as rebuildable projections with recorded revisions. They must never become an alternative source of truth. Explicit degraded fallback is allowed only within the same scope and committed state; otherwise return unavailable.

Inventory every mutation entry point: single/batch ingest, merge, promote, decay/repair, curation/delete, import/migrate, checkpoint save/restore/prune, index rebuild and close. Add named fault hooks around validation, staging, serialization, write/flush/fsync, replace, database commit, projection publication, compensation and response delivery.

Build a subprocess harness with an independent expected-operation ledger. Kill workers at each hook (including after commit/before response), truncate/corrupt data and manifests, inject short writes/ENOSPC/EIO, exhaust an isolated test volume, deny writes, fail serialization, kill during promotion and take the vector backend offline. Never target a user's live store. Test JSON/JSONL compatibility paths and the journal separately; run platform-specific cases on Linux, Windows and macOS.

On restart assert acknowledged records and provenance survive; rejected records are absent; uncertain operations resolve by key; batches are whole; lineage is consistent; no cross-scope state appears; indexes rebuild only from committed records; integrity failures are visible. Record recovery duration and the failing hook. Distinguish process-kill evidence from power-loss evidence; qualify filesystem/VM power-loss behavior separately before advertising that guarantee.

Gate: every inventoried hook has a deterministic passing case, plus a seeded randomized campaign. Zero silent loss, phantom commits, partial published promotions, or scope leaks. Publish the supported failure matrix and recovery runbook.

## 4. Add an adversarial memory regression corpus

Create `tests/fixtures/memory_adversarial/v1/` and a deterministic runner with fixed clock, embeddings, policies and expected state/context. Each case includes input events, provenance/trust, scope, expected active/suppressed IDs, lineage, authority and reason codes. Run through the adapter and provider; keep model-output evaluation separate from deterministic state checks.

| Case | Expected outcome |
| --- | --- |
| Conflicting facts over time | Preserve both source claims and valid times; apply explicit authority/supersession policy or report unresolved conflict. Do not fabricate a reconciled fact. |
| Superseded preference | New authorized explicit preference wins in its scope; older preference remains historical and cannot reappear through recency fallback or summaries. |
| Near duplicates | Merge only scope/trust/policy-compatible records; retain every source and respect `no_merge`. Similar wording is insufficient to merge contradictions. |
| Instruction-like memory | Retrieved text stays evidence; it cannot acquire system/tool authority, alter policy, or authorize actions. Preserve literal content only under the import policy. |
| Stale high-salience fact | Expiry/supersession precedes salience ranking; explain suppression and test clock-boundary behavior. |
| Two agents updating related state | Revision conflict or deterministic serialized result; no lost update, half-promotion, or duplicate source. |
| Self-reinforcing wrong retrieval | Repeating an agent's own recalled claim does not count as independent corroboration or raise authority; correction remains effective after summaries and restart. |
| Untrusted import | Quarantine or explicitly mark untrusted; imported claimed scope/authority cannot override authenticated import policy. Promotion requires validated provenance. |

Add cross-tenant/session canaries to each case. Gate: 100% of deterministic expected outcomes pass, including restart and context assembly. Failures become named regression cases before fixes land.

## 5. Establish an independent boring baseline

Implement one baseline first: SQLite + the same embeddings + exact cosine top-k + fixed recency weighting + prompt compaction. Use an independent minimal implementation, not the DML adapter with flags changed. Give it equivalent scope isolation, persistence durability, source metadata and token budgets. Postgres is a later deployment baseline if multi-host operation becomes a requirement.

Compare no-memory, baseline and DML stable; add experimental features as individual ablations. Hold model revision, embedding revision, tokenizer, prompts, tools, dataset/order, hardware, inference concurrency, budget and compaction model constant. Account for summary/embedding calls, retries and background maintenance. Separate cold-start, warm-cache, rebuild and steady-state results. Freeze tuning on a development corpus and use held-out workloads for the decision.

Proposed value gate to freeze before collecting acceptance results: either at least 5 percentage points higher task success, or at least 15% fewer total tokens per completed task with a task-success non-inferiority margin of 2 percentage points. Use paired runs across at least 20 independent seeded episodes per workload and 95% confidence intervals; the interval must support the claimed gate. Also require no new deterministic safety failures, no material worsening of semantic false-memory/contradiction rates, and p95 latency within the predeclared workload SLO. Extend sample size if uncertainty is too large; do not pick a winning seed or metric afterward. If DML does not pass, simplify or keep the added feature experimental.

## 6. Continuously measure agent outcomes

Extend `openclaw-wrapper/scripts/benchmark_openclaw_memory.py`, `recall_eval.py`, `stress_harness.py`, `dml_core/scripts/dcm_workload_benchmark.py` and `performance_benchmark.py` with a common episode/event schema. Reuse useful workloads from `TEST_PLAN.md`; distinguish actual tool-driven agent runs from offline simulations.

| Metric | Definition |
| --- | --- |
| Task success | Completed tasks satisfying a pinned verifier / attempted tasks; count failures and timeouts. |
| Repeated-error rate | Repetition of a previously observed, still-applicable avoidable error / relevant retry opportunities. |
| Contradictions introduced | New outputs conflicting with established valid evidence / evaluated factual outputs. |
| False-memory rate | Unsupported recalled claims / evaluated recalled claims; report persistent phantom records separately with a zero-tolerance gate. |
| Tokens per completed task | All input/output/compaction/maintenance tokens across the run, including failed attempts, / successful tasks; undefined if none succeed, never report zero. |
| TTFT / total latency | From accepted agent request to first model token / final task response; include memory/context work. Report p50/p95/p99, errors and timeouts. |
| Retrieval overhead | Embedding, index search, ranking, context assembly and lock wait, separately and total. |
| Store growth | Records, evidence/lineage/tombstones, index/database/WAL/snapshot bytes and RSS at each milestone. |
| Recovery time | From restart to verified service readiness after the injected fault; record unavailable/degraded duration. |

Run checkpoints at 1k, 10k and 100k turns, with separate memory counts so turns are not confused with records. Save raw events, seeds, dependency/runtime identities, hardware, effective configuration, task verifiers and report-generation version. Keep secrets out of artifacts. Report quality confidence intervals and latency distributions, not just means.

Planned CI cadence: PRs run deterministic contract/adversarial/recovery cases and a small offline episode; nightly runs cover 1k/10k-turn workloads and concurrent stress; scheduled release campaigns cover 100k turns and real pinned inference backends. Expensive/live checks report separately and require provisioned runners. A skipped GPU/long-horizon lane is incomplete evidence, not a pass. This plan does not create scheduled jobs yet.

## 7. Separate stable, experimental and research surfaces

Publish an API/status inventory in the contract registry, docs and capability response. Existing “implemented” features retain that description but gain explicit maturity labels. Do not silently break legacy defaults; introduce the stable profile and a documented transition.

| Tier | Initial assignment |
| --- | --- |
| Stable candidate | Scoped receipt-based ingest, exact retrieval, bounded context, transactional persistence, semantic snapshots and validated migration. Promote only after this plan's gates pass. |
| Experimental | Native KV reuse/DCM runtime paths, ANN, automated abstraction/promotion, DPM/DCN learning/evolution, distributed cache routing. Keep independently switchable and absent from stable-profile correctness dependencies. |
| Research | Unimplemented transfer/data-plane integrations and unproven self-improving policies. No availability or compatibility promise. |

Native KV reuse needs one authoritative compatibility decision shared by save, restore, continuation, transition and import. Bind model weights/revision and adapters, tokenizer and template, runtime/build/adapter, endpoint identity, TP/PP/device topology, dtype/cache layout, positional/RoPE state, exact prefix tokens/context identity, scope/authority and payload digest. Existing checkpoint and fabric identities supply much of this; audit their end-to-end binding rather than replacing them blindly.

Gate: mutate every dimension independently; missing, unknown, stale or mismatched identity must reject before native restore is called. Test v1/v2 identity handling, replay/expiry, changed authority, purge and parent/child lineage. Recompute native state instead of heuristically converting incompatible KV. Run capability and hardware canaries on each explicitly supported runtime build before adding it to the support matrix.

## 8. Introduce schema and migration discipline

Inventory every persisted family: memories, lineage, source evidence, receipts, JSON/JSONL, journal tables/metadata, vector manifests, semantic/native checkpoints, STM/DCM state, DPM/DCN state and audit events. Separate object schema version, database schema version, API version, policy version and mutable data revision.

Require explicit versions and validate required fields, counts/checksums and supported versions before accepting data. Legacy unversioned files enter only through an explicit validated importer; unsupported future versions fail closed. First regression: a checksummed JSONL payload with an unsupported header version must be rejected; existing `load_state()` needs this change.

Migrations run under exclusive maintenance ownership with a verified backup and enough disk space. Perform database DDL/data changes transactionally; for file stores, stage and validate a complete generation before atomically switching its manifest. Make resumability/idempotency explicit and fault-inject before/after each transition. Rebuild derived indexes from the migrated authoritative revision.

Never downgrade in place. Either supply a verified compatible export with an explicit loss report, or require restoring the old binary plus pre-upgrade backup while accounting for writes since upgrade. Keep golden fixtures created by the actual supported release tags. Until multiple releases exist, use named commit-pinned legacy fixtures and label them accurately. Before supporting N+1, run every supported source-version-to-N+1 path, interruption/restart, export/restore, and future-version rejection.

## 9. Define concurrency semantics explicitly

Stable v1 uses serialized commits per store and immutable revisioned read snapshots. Ingest/promote/delete return committed revision receipts; retrieval and checkpoint creation pin one revision. Retrieval-side access/decay updates become explicit later transactions, not mutations of the reader's snapshot. No distributed multi-host writer guarantee is implied.

Specify lock acquisition order, ownership, reentrancy, cancellation, deadlines and shutdown/draining. Preserve cross-process advisory locks for cooperating local processes; Python's GIL is not a transaction mechanism. Apply the same ownership to background aging/repair, imports and checkpoint/pruning. External model/embedding calls run outside locks and use compare-and-swap/revalidation before committing their result. Revision changes trigger bounded retries or a typed conflict, never overwrite unseen state.

Stress 1/16/64/256 clients using threads, separate processes and HTTP workers. Exercise same-scope contention and many scopes, inject seeded delays between reads/writes/commit/publication, cancel requests and kill lock holders. Use an independent history checker for acknowledged writes, idempotency, snapshot consistency and related-state invariants. Measure lock hold/wait, throughput, p95/p99, starvation and recovery. Gate: zero lost updates, dirty reads, deadlocks or scope leakage; timeout/retry results must match the documented state machine.

## 10. Make every memory decision explainable

Define versioned decision events for ingest, reject, merge, supersede, retrieve, suppress, promote, expire/delete, context assembly, checkpoint compatibility and recovery. Carry trace/operation ID, committed revision, policy/schema versions, scope, source/parent IDs, trust/authority before/after, candidate scores/components, filters/reason codes and context manifest identity. Compatibility events identify the checked dimensions and exact rejection reason without disclosing keys or native payloads.

Write mutation evidence with the authoritative transaction/outbox so an exporter outage cannot erase the audit trail. An external metrics outage must not change memory semantics. Record bounded candidate evidence and reason counts; an explicit diagnostic mode may retain full candidates under access and retention controls. Never use memory IDs or text as unbounded Prometheus labels.

Add an operator explanation/replay command: given a request ID, resolve its immutable memory revision, input/policy identities and final context manifest, then show why each selected/suppressed item affected the response. Retain exact rendered context in protected evidence storage where enabled, or reconstruct from retained immutable source versions and the pinned renderer. Digests alone cannot reconstruct content. Clearly report when retention/redaction makes exact reconstruction unavailable. Credentials are never logged; payload access follows the original scope and operator permissions.

Gate: use a deliberately bad agent answer to reconstruct the actual context, retrieval/suppression/merge history, authority changes and checkpoint decision without a debugger. Explain events must remain available across crash/recovery. Audit checkpoint failures and retention actions as well as successful decisions.

## Delivery order and review gates

Estimate: roughly 8–12 engineer-weeks for the initial stable candidate with one primary implementer and review, excluding provisioning and the duration of hardware/100k-turn campaigns. Re-estimate after the mutation inventory; these are planning estimates, not delivery commitments. Each row can span several small PRs. No named assignees or external issues are created by this plan.

| Order | PR-sized deliverable | Depends on | Exit gate | Items covered |
| --- | --- | --- | --- | --- |
| P01 | Contract ADR, stable profile specification, maturity inventory and mutation/format map | Current main | Agree observable invariants, supported platforms and typed errors; add failing characterization cases for known gaps | 1, 7, 8, 9 |
| P02 | Public behavior fixtures, transaction/read-snapshot protocols and minimal decision-event schema | P01 | Existing API/golden behavior preserved; ownership documented | 2, 9, 10 |
| P03 | Extract persistence service; authoritative commit, idempotency receipts and projection outbox | P02 | Reopen, duplicate retry and after-commit/before-reply fault cases pass | 2, 3, 9, 10 |
| P04 | Version validation and explicit transactional migrations | P03 | Legacy/current/future-version and interrupted migration matrix passes | 3, 8 |
| P05 | Complete mutation fault inventory, crash runner and recovery runbook | P03–P04 | All mutation hooks pass; unsupported durability cases explicitly listed | 3 |
| P06 | Extract retrieval/context services; pinned inputs and final token-budget enforcement | P02–P03 | Golden scope/order/context outputs and budget rejection tests pass | 1, 2 |
| P07 | Extract lifecycle and checkpoint services; adversarial fixtures and native compatibility audit | P04–P06 | Eight semantic cases pass; every KV identity mismatch rejects | 2, 4, 7 |
| P08 | Concurrent history checker and 256-client fault/delay stress lane | P05–P07 | Ownership/recovery invariants and predeclared latency/timeouts pass | 3, 9 |
| P09 | Independent SQLite baseline and common episode/metric harness | P01–P02 for scaffold; P06–P07 for comparison | Fairness manifest, verifier, seeds and value thresholds frozen before evaluation | 5, 6 |
| P10 | Operator decision replay and durable audit export | P03, P06–P07 | Reconstruct bad-answer fixture, including crash and retention limitations | 10 |
| P11 | Continuous 1k/10k lanes, 100k release campaign and stability promotion | P08–P10 | All integrity/semantic/migration gates pass; value gate and supported-runtime evidence published | All |

P09's independent baseline scaffold can be developed while persistence work proceeds. Promote no experimental feature merely because its extraction is complete. Keep the existing CI core/Hermes/OpenClaw and portability suites; add focused gates rather than replacing them. Add maintained lint/type coverage as services are extracted.

Immediate first implementation: P01, followed by a regression for unsupported persistence versions and the P02 transaction seam. Production deployment, default storage migration, and the architectural changes themselves are future implementation work; this commit records the plan only.
