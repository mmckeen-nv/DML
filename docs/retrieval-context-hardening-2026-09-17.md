# Scoped retrieval and context service boundary

This fourteenth serial gate advances adapter decomposition and the production
contract in P06, with concurrency evidence for area 9. It extracts the supported
scoped context path behind narrow interfaces while preserving characterized
output. It introduces no storage format, public endpoint or maturity promotion.

## Responsibilities and ownership

`services.scoped_retrieval.ScopedRetrievalRequest` freezes the resolved four-member
scope, kinds, phase, top-k, effective time and inspection flag. The adapter still
owns input normalization, router/configuration defaults, finite-clock validation,
model preparation and operation ownership. The same frozen kinds drive selection
and evidence, even if a callback changes the original caller/router list.

`select_scoped_context` receives only the public filtered-retrieval capability, a
lazy candidate supplier, the query vector and the frozen request. It supplies
lifecycle eligibility before top-k. The existing `MemoryStore.retrieve_filtered`
implementation retains scoring, recency weights, thresholds, dimension handling,
deterministic ties and optional experimental retrieval modes. Reader exceptions
propagate; an exception cannot produce a successful recent-memory fallback.

Recent candidates are read exactly once after an empty successful ranked result.
Semantic success adds no full-store fallback copy. Both capabilities refer to the
same store under the adapter's existing ownership boundary. Collection snapshots
do not freeze `MemoryItem` objects: records are borrowed until context/report
construction finishes. The services acquire no locks and do no embedding, durable
storage I/O or clock reads. Rendering uses the timestamp already on each item.

The public adapter performs embedding I/O before acquiring ownership. Receipt
schema-2/3/4 retrieval requires an explicit nonempty tenant, validates embedding
identity/dimension, refreshes changed authority and rechecks compatibility inside
ownership before ranking. A commit during model preparation is observed when the
read takes ownership. A cooperating adapter mutation cannot interleave with
selection, compaction or evidence materialization. Report revision describes the
owned runtime snapshot, not a promise that later reads will have that revision.

`services.context.compact_context` keeps its existing renderer unchanged.
`build_context_report` builds the existing response and retrieval-decision schema
from explicit values, with no adapter/store access. It deep-copies nested entries,
scope, kinds, suppression evidence and personality overlay. Returned dictionaries
remain mutable for compatibility; their digests describe construction-time values
and do not update after caller edits. The query vector is not retained. Routing,
DPM lookup/rendering, metrics and the existing latency measurement boundary remain
adapter responsibilities. Latency still ends after compaction, before suppression
scanning and evidence/report construction; it is not total request latency.

## Preserved behavior

The characterization harness was run against the unchanged base adapter before
wiring these services. It pins complete reports, literal context and decision
digests across legacy storage, journal schema 1 and receipt schemas 2, 3 and 4.
The contract deliberately preserves these compatibility details:

- Scoped matching checks all four fields, including omitted optional fields as
  null. Semantic selection uses resolved kinds; only recent fallback additionally
  checks item phase. An empty kinds list differs from omitted kinds during ranking
  and keeps its existing fallback meaning.
- Recent fallback orders by timestamp descending, then memory ID ascending.
  Unscoped legacy fallback remains global unless its existing internal caller
  explicitly requires unscoped records. Receipt callers cannot use global reads.
- Suppression evidence covers all matching-scope candidates in source order,
  independently of phase, kinds and rank. Inspection bypass behavior is preserved.
- The latest survival ledger is chosen before lifecycle admission; ledger lookup
  does not fall back to an older ledger when the latest is suppressed. Normal
  semantic ranking may still select an eligible older ledger. Equal ledger
  timestamps preserve source order and scope retains existing string normalization.
- An admitted ledger is prepended and deduplicated before compaction; the ledger
  inclusion flag records that admission even if the budget later omits it.
- Context accounting includes heading, sources and personality prefix. Prefix
  overflow fails explicitly; the first memory may be shortened, and a subsequent
  memory that does not fit ends admission. Counts remain labeled estimates.

The intentional hardening is input/output isolation: later mutation of caller
kinds, personality data or returned nested metadata cannot alter the borrowed
source or retrospectively change response evidence. Explicit callback-mutation
regressions verify that selected kinds and reported kinds agree.

## Qualification and limits

Separate builders implemented selection policy and report construction. A third
agent captured baseline behavior; an independent grader added hostile-capability,
alias and controlled-interleaving regressions. The final review and test evidence
are recorded with accepted source hashes in the
[review artifact](artifacts/retrieval-context-review-2026-09-17.json).

The independent reviewer accepted this bounded gate at **9.8/10**, with no blocking
findings. Its combined focused run passed **104 tests**: 28 selection-service,
27 context-service, 29 baseline integration and 20 independently authored
adversarial checks. A separate 13-case existing compatibility selection passed.
Controlled interleavings cover commits during embedding, embedding identity changes
at preparation/ownership boundaries, and actual retirement contention through
report materialization on receipt schemas 2, 3 and 4.

Final local integration on Python 3.12.14: **2,818 passed**, nine optional-dependency
skips and two known warnings in 280.25 seconds. Ruff, all 58 maintained mypy files
and Hermes hygiene passed. The warnings are the existing Starlette/AnyIO alias
deprecation and the intentional huge-query norm overflow regression. Accepted
runtime/test/CI hashes were verified against the final local source. Exact-commit
CI results are recorded separately on PR #118; the review artifact distinguishes
independent focused review from root integration evidence.

Focused CI runs all four new test files in the production-evidence lane and on
Linux, macOS and Windows with Python 3.10/3.13. The full regression lane also covers
the existing provider, receipt, lifecycle, Hermes and wrapper surfaces. No user
memory stores are opened or changed by the harness.

This is a bounded extraction gate. Legacy hybrid retrieval, automatic quality
adjustments, ANN, DPM and native KV remain experimental. This stage does not add
physical erasure, durable context replay, source-content access control, complete
model/tokenizer binding, lock-latency guarantees or a large-store performance
claim. Existing retrieval evidence still declares incomplete replay inputs.
Full model-input budget qualification and the remaining persistence/lifecycle
orchestration extraction stay in the original production plan.
