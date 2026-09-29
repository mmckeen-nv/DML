# Explicit receipted memory supersession

This is the tenth serial hardening gate after the production foundations tranche.
It extends P03's authoritative lifecycle receipts, with P04 migration compatibility
and P07 semantic-policy coverage. It does not complete the ten production areas.
Full mutation coverage, further adapter extraction, native KV compatibility,
growing-store baseline value, real-agent outcome campaigns and complete decision
replay remain separate gates. DML and this feature remain candidates.

## Contract

`supersede_memory_receipted(memory_id, *, replacement_memory_id,
expected_memory_digest, expected_replacement_digest, reason, idempotency_key,
tenant_id, client_id=None, session_id=None, instance_id=None)` links two existing
memories in the same exact scope. The narrow `services.receipt_supersession` service
reuses the accepted journal transaction and error types. Journal schemas 2, 3 and 4,
receipt envelopes and outbox event formats are unchanged.

Both IDs must be different nonnegative integers. Both records must exist in the
live-item collection under the complete tenant/client/session/instance tuple.
Missing, lineage-only and wrong-scope records share the same not-found result.
The caller supplies the exact complete-record digest for each memory, calculated
by `services.receipt_lifecycle.memory_digest`. Both records are checked in one
snapshot, and journal compare-and-swap prevents either changing before commit.
An unrelated revision can be retried; a changed source or replacement conflicts.

The source cannot already be retired or superseded. Existing lifecycle decisions
and terminal-state aliases are protected against overwrite. The replacement must
be eligible for normal retrieval when checked: untrusted, quarantined, suppressed,
expired, retired or superseded replacements are refused. Conflicting lifecycle
aliases cannot conceal a terminal state. This operation does not promote a source
or grant trust to a replacement; its metadata remains unchanged.

Only the source changes:

- `meta.memory_state` becomes `superseded`.
- `meta.superseded_by` records the replacement's integer ID.
- `meta.supersession_decision` records version `dml-supersession-decision-v1`, the
  prior source digest, replacement ID and digest, and operator reason.

All other record fields and provenance are preserved. The changed source, receipt,
decision (`supersede-receipt-v1`) and optional outbox event commit atomically. The
receipt contains the historical changed source record; its decision binds the
replacement version that was checked. No model, embedding or external backend is
called. No records are erased or capacity released.

## Requests, retries and chains

Scope and idempotency-key bounds match the receipt profile (256 UTF-8 bytes per
string); reason is nonempty and at most 1,024 UTF-8 bytes. Digests must be canonical
64-character lowercase SHA-256 values. Requests are frozen, shape-checked and
verified against their canonical digest before any authority access. The API
rejects coercion, booleans as IDs and extra fields.

Ingestion, retirement and supersession share a scoped key namespace. An identical
retry returns its original receipt before checking present lifecycle state. It can
replay even after the replacement is later retired or removed from live state.
A key reused for a different request conflicts. Replayed generic journal
receipts must actually contain the requested source, link and decision.

An explicit A→B followed by B→C is allowed. The original A→B link remains historical;
the service does not rewrite it to A→C or resolve chains during retrieval. A reverse
edge to an already superseded memory is refused. Concurrent opposing edges cannot
both commit because the loser must revalidate changed records. The replacement
may later expire or be retired; success does not promise permanent retrievability.
Eligibility is checked during the decision, not guaranteed at the instant a client
receives an acknowledgement.

## Adapter and HTTP surface

```python
from daystrom_dml.services.receipt_lifecycle import memory_digest

old = original_receipt["result"]["memory"]
new = replacement_receipt["result"]["memory"]
receipt = adapter.supersede_memory_receipted(
    old["id"], replacement_memory_id=new["id"], tenant_id="example",
    expected_memory_digest=memory_digest(old),
    expected_replacement_digest=memory_digest(new),
    reason="Preference changed", idempotency_key="preference-change-1",
)
```

`POST /api/memory/supersede/receipt` exposes the same arguments with existing bearer
authentication. It returns 200 for the historical receipt, 404 for unavailable
scoped records, 409 for lifecycle/digest/key conflicts, 400/422 for invalid or
unsupported requests, and 503 when ownership or durable outcome is unavailable.
Storage errors expose fixed codes rather than underlying exception messages.
Service token holders remain trusted to choose scopes; no per-token authorization
is added. The caller needs exact authoritative records; this gate adds no record
inspection endpoint.

Receipt mode must be explicitly enabled. Nested compensating transactions and RAG
mirroring remain refused. A failed acknowledgement is reconciled against the
durable ledger without compensating away a possible commit. Hydration failure
marks runtime degradation but cannot revoke a durable receipt; a later owned query
refreshes authoritative state.

Normal retrieval suppresses superseded memories. Verified projection queries
require reconciliation to the new revision and apply the same policy. Trusted
`include_quarantined=True` inspection may still expose historical records. Already
assembled context packets cannot be retracted from a model by this operation.

## Qualification boundaries

The process-death harness checks all eight receipt commit hooks in schema 2 and
all nine outbox hooks in each of schemas 3 and 4 (26 cases). Parent-held oracles
require both original ingestion receipts, unchanged replacement bytes, the exact
source-only transition, one decision/receipt/event, and retry without another
revision. Concurrency coverage uses 256 threads released by a start barrier and
eight competing processes for each supported schema. CI retains recovery and
concurrency JUnit and covers Linux, macOS and Windows.

Independent review accepted the feature at **9.6/10**, with **199 focused tests**
passing: 78 service, 57 integration, 32 recovery/concurrency and 32 independent
adversarial cases. Both separately authored opposing-link races require exactly
one committed edge. The [review record](artifacts/supersession-review-2026-09-14.json)
pins accepted source hashes; final exact-commit CI is tracked on PR #118.
Checksums detect inconsistency,
not malicious replacement of authority and every linked checksum. Process death
does not establish hardware power-loss durability. Full-history verification and
full-state outbox events retain growing costs; this gate claims no performance or
agent-quality advantage.

Final local verification: **1,951 passed**, nine optional-dependency skips and two
known warnings in 82.46 seconds. Ruff, 53 maintained mypy files (local Python 3.12)
and Hermes hygiene passed. All three supported schema scenarios passed the separate
256-client start-barrier stress run.
