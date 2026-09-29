# Receipted memory retirement

This serial lifecycle gate adds one explicit operation: retire an existing scoped
memory from normal retrieval while preserving its historical record. It advances
areas 1, 2, 3, 4, 9 and 10 of the production plan. It remains a candidate feature.

## Contract

`DMLAdapter.retire_memory_receipted(memory_id, *, expected_memory_digest, reason,
idempotency_key, tenant_id, client_id=None, session_id=None, instance_id=None)`
requires an opt-in receipt journal. Journal schemas 2, 3 and 4 retain their existing
formats. The new narrow service is `services.receipt_lifecycle`; the adapter handles
configuration and runtime hydration.

A new retirement requires an exact tenant/client/session/instance scope match and
the SHA-256 digest of the complete current memory record. Use
`services.receipt_lifecycle.memory_digest(record)` on the exact persisted record
returned in a receipt. A historical ingestion receipt may be stale after a later
change; the expected digest is deliberately not a request to retire any future
version of that ID. Different metadata types such as true, 1 and 1.0 remain distinct.

The memory ID must be a nonnegative integer, never a boolean. Scope members are
nonempty strings of at most 256 UTF-8 bytes, with null allowed for optional members.
The reason is a nonempty string of at most 1,024 UTF-8 bytes. Digests are 64 lowercase
hexadecimal characters. Canonical requests are frozen and their digest is checked
again at the service boundary before storage access.

The transaction sets `meta.memory_state="deleted"` and adds a versioned retirement
decision carrying the prior record digest and operator reason. It preserves the
ID, text, vector, timestamps and other record fields and provenance. The changed
record, immutable result receipt, journal decision and any enabled outbox event
commit atomically. No embedding model or external backend is called. A target
changed since the caller's decision conflicts; unrelated concurrent writes can be
retried under journal compare-and-swap.

Ingestion and retirement share the same scoped idempotency-key namespace. An exact
retry returns the historical receipt before checking current memory state. Reusing
a key for a different operation or request conflicts. A new key for an already
retired record conflicts. Missing, lineage-only and wrong-scope targets produce the
same not-found response. A live record with a preexisting retirement decision is
also refused, preserving its provenance for inspection.

## Usage and failure states

```python
from daystrom_dml.services.receipt_lifecycle import memory_digest

original = adapter.ingest_memory_receipted(
    "My old preference", tenant_id="example", idempotency_key="append-1",
)
record = original["result"]["memory"]
retired = adapter.retire_memory_receipted(
    record["id"], tenant_id="example", idempotency_key="retire-1",
    expected_memory_digest=memory_digest(record), reason="Preference withdrawn",
)
```

`POST /api/memory/retire/receipt` exposes the same arguments under existing bearer
authentication. It rejects unknown fields and type coercion. Service token holders
remain trusted to choose scopes; this does not add per-token tenant authorization.

| Status | Meaning |
| --- | --- |
| 200 | Historical retirement receipt, including an exact retry. |
| 400 / 422 | Invalid request or unsupported adapter configuration. |
| 404 | No memory available in that exact scope. |
| 409 | Key conflict, stale memory/lifecycle decision, or exhausted journal revision retries. |
| 503 | Ownership or durable receipt outcome unavailable; retry the identical request and key after recovery. |

Error bodies contain fixed codes rather than storage exception messages. A failed
acknowledgement is reconciled against the durable ledger, never compensated by
restoring a memory. A successful receipt remains success if runtime hydration
fails; health records degradation and a later owned read refreshes state. Nested
legacy compensating transactions and external RAG mirroring are refused.

## Retention and scope limits

Retirement is a tombstone, not physical erasure. It does not release capacity, purge
text/vectors from receipts or outbox history, modify RAG stores, or reclaim disk.
Normal scoped retrieval and projection queries suppress the tombstone through the
existing lifecycle policy. Trusted `include_quarantined=True` inspection can still
surface it. A stale projection must reconcile before it can answer against the new
source revision. Previously constructed context packets remain historical outputs;
this operation cannot retract context already sent to a model.

The journal and receipt checks detect inconsistent storage, not authenticated
tampering by a writer that can replace authority and every checksum. Remaining
public update, supersession, promotion/merge and physical-erasure APIs require
separate lifecycle gates. Other legacy mutations stay disabled for receipt journals.
No whole-platform production readiness or long-history performance claim is made.

## Qualification

The recovery harness kills actual processes at all eight receipt commit hooks in
schema 2 and all nine outbox commit hooks in schemas 3 and 4: 26 cases. Its parent
oracle requires the original acknowledged receipt, either unchanged state or the
complete retirement, exact retry, one revision and the corresponding outbox event.
Concurrent same-key tests use a start barrier with 256 worker threads, plus eight
competing processes, for each supported schema. CI retains recovery and concurrency
JUnit and runs the retirement corpus in Linux/macOS/Windows portability lanes.

Independent review accepted this feature at **9.6/10**, with **151 new tests**
passing independently: 65 service, 43 integration, 32 recovery/concurrency and 11
adversarial cases. The [review record](artifacts/retirement-review-2026-09-14.json)
pins accepted source hashes and retained findings. An existing rejection-message
compatibility test caught a diagnostic-only regression in the full suite; the
wording was corrected and all 23 embedding-contract tests passed independent review.
Exact-commit CI is tracked on PR #118.
Process-death tests do not qualify physical power loss or every filesystem.

Final local integration: **1,752 passed**, nine optional-dependency skips and two
known warnings in 76.26 seconds. Ruff, 52 maintained mypy files (local Python 3.12)
and Hermes hygiene passed. All three schema scenarios also passed the separate
256-client retirement stress run.
