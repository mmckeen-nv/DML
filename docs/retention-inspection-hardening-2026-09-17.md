# Verified memory retention inspection

This thirteenth serial gate adds a scoped, revision-pinned inspection of retained
memory content. It advances the retention contract and operator evidence in areas
1, 8 and 10. Physical erasure and bounded history retention remain separate gates;
this feature cannot delete content or certify its absence from external copies.

## Why this boundary exists

Receipt journals preserve exact historical results for idempotent retries.
Decisions bind those receipts, and schemas 3/4 retain complete outbox states.
First-level promotion also preserves source records inside the derived record.
Removing current memory cannot remove these copies. Rewriting or pruning this
history would violate the existing receipt and ordered-delivery contracts.

The current contract therefore retains receipt and outbox history without a
configured expiry. Retirement suppresses ordinary retrieval while retaining
content, provenance and capacity. This stage adds neither physical erasure nor
capacity reclamation. A later erasure design must explicitly define changes to
historical retries, source proofs, consumer identity and independently held copies.

## Supported inspection

`DMLAdapter.inspect_memory_retention(memory_id, *, tenant_id, client_id=None,
session_id=None, instance_id=None)` requires an existing configured receipt
journal using schema 2, 3 or 4. The narrow `services.retention` service validates
and freezes the nonnegative integer ID and exact four-member scope; booleans,
coerced values, unknown fields and oversized scope strings are refused.
Stored records follow existing receipt scope semantics: an omitted optional scope
field means null; supplied values must match by both type and value. The canonical
request itself always contains all four scope fields.

One verified SQLite read transaction captures the current state, archived journal
snapshot, all receipts and all applicable outbox events. The result identifies the
store, journal schema, committed revision and current state digest. A writer may
commit during or after inspection: the report describes its pinned revision and
does not become permission to mutate a later revision. Inspection takes no adapter
writer lock and calls no embedding model, hydration callback or external backend.

The `dml-retention-report-v1` report contains bounded counts for five surfaces:

| Surface | Content counted |
| --- | --- |
| `current_items` | Matching current records, including retired records. |
| `current_lineage` | Matching archived lineage records. |
| `journal_snapshot` | Matching records in the persisted snapshot, which may predate the current revision. |
| `receipts` | Matching full historical result records. |
| `outbox_states` | Matching records in every retained full-state event. |

Each surface separately counts `direct_records` and `embedded_source_records`.
The latter are complete source records in recognized first-level promotion proofs.
`known_reference_count` sums these serialized record occurrences. It is neither a
unique-memory count nor a byte/file count. Outbox receipt bindings carry hashes
and keys, rather than another serialized memory, and are not counted again.

Outer records and receipts must match the exact requested scope before their
proofs are inspected. Proof sources must also match it to count. Unknown, malformed
or recursive same-scope promotion proofs refuse inspection instead of producing a
misleading zero. Bare lineage IDs, arbitrary metadata and semantic copies are not
interpreted as complete source records. No text, vectors, arbitrary metadata,
receipt keys, operator reasons or filesystem paths appear in the report.

A record present only in receipts, an older snapshot, an outbox state or a retained
proof is inspectable. An ID with no known structured occurrence in the requested
scope has the same not-found outcome as a foreign-scope ID. That outcome does not
prove physical absence or erasure.

## Capabilities, API and failure states

`GET /api/contracts` includes a detached, data-independent `retention` contract.
`dml-journal retention-contract` prints the same contract without opening a store.
`POST /api/memory/retention/inspect` exposes the scoped report under the existing
bearer authentication. Token holders remain trusted to choose scopes; this feature
does not introduce per-token tenant authorization.

The endpoint returns 200 for a report, 404 for no accessible known occurrence, 409
for unsupported proof inspection, 400/422 for unsupported configuration or invalid
input, and 503 when storage cannot establish a verified outcome. Error codes are
fixed and omit underlying storage messages. Corruption or incomplete history
refuses inspection without repair or pruning.

Every successful report sets `physical_erasure_supported=false`,
`erasure_proven=false` and `retirement_is_erasure=false`. Its fixed uninspected
inventory covers projections, outbox consumers, external checkpoints, backups and
migration sources, SQLite WAL/free pages, filesystem snapshots, runtime/caller
copies, and arbitrary metadata/semantic duplicates. The tool does not discover
whether any particular external copy exists.

Read-only means no logical authority mutation: no memory, receipt, decision,
snapshot or outbox update. Normal journal construction and SQLite connection
management may touch coordination files or WAL/SHM. Checksums detect inconsistency,
not an attacker replacing the entire authority and every linked checksum. History
validation and inspection costs grow with retained history.

## Operator procedure

1. Inspect the exact ID and scope and retain the report's revision with the result.
2. Use existing receipted retirement when the goal is to suppress future ordinary
   retrieval; derived memories require their own lifecycle decisions.
3. Treat history, backups, checkpoints, consumers and previously returned context
   as separate retained copies. Do not delete journal rows, sidecars or outbox
   events to simulate erasure: the next verified operation must reject such damage.
4. If actual erasure is required, stop claiming that this profile meets it. Define
   the history/retry transition and external-copy policy before implementing a
   separately reviewed purge operation.

## Qualification

Separate builders owned storage/service, API/CLI and recovery/concurrency coverage.
An independent grader accepted the frozen feature at **9.6/10**, with no blocking
findings. Its combined corpus passed **213 tests**: 55 service, 65 integration,
61 recovery/concurrency and 32 independently authored adversarial cases. The
[review artifact](artifacts/retention-review-2026-09-17.json) pins 12 accepted
runtime/test/CI hashes and preserves the review history.

Evidence includes six actual process kills, six deterministic read/write races
and 37 corruption/history/identity/migration refusals. Each journal schema passed
256 simultaneous readers without changing logical authority. Nine existing
provider tests passed; a separate regression preserves other endpoints' validation
responses. The longest collected feature node ID is 140 UTF-16 units, keeping
oversized-input cases bounded on Windows.

Final local integration: **2,714 passed**, nine optional-dependency skips and two
known warnings in 295.32 seconds. Ruff, 57 maintained mypy files (local Python 3.12)
and Hermes hygiene passed. The warnings are the existing Starlette AnyIO alias
deprecation and intentional huge-query-vector norm overflow. CI runs the corpus
on Linux/macOS/Windows and retains recovery and 256-reader evidence; exact-commit
results are recorded on PR #118. No user memory stores were changed. The overall
maturity remains candidate.
