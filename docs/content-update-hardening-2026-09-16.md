# Receipted content updates

This eleventh serial hardening gate extends P03's durable mutation boundary and
P07's lifecycle coverage. It adds content correction to the existing receipted
ingestion, retirement and supersession profile. DML remains a candidate platform;
this does not complete the original ten production areas.

## Atomic update contract

`DMLAdapter.update_memory_receipted(memory_id, *, text, expected_memory_digest,
reason, idempotency_key, tenant_id, client_id=None, session_id=None,
instance_id=None)` updates one existing item in the exact receipt scope. The narrow
`services.receipt_update` service commits text, its newly prepared vector and a
receipt through the accepted journal transaction. Journal schemas 2, 3 and 4,
receipt envelopes and outbox event formats are unchanged.

A new request requires the complete current memory-record digest, calculated with
`services.receipt_lifecycle.memory_digest(record)`. Scope and digest are checked
before model access and again on every commit attempt. Missing, lineage-only and
wrong-scope IDs have the same not-found result. A changed target conflicts; unrelated
concurrent writes may be rebased through at most three journal CAS attempts.

Only three fields change: `text`, `embedding`, and
`meta.content_update_decision`. The decision has version
`dml-content-update-decision-v1`, the prior memory digest and the operator's reason.
A later update can replace that recognized latest decision; previous versions
remain in immutable receipts and journal history. Unknown or malformed existing
decision metadata causes a conflict. This metadata is not an authentication
mechanism for imported provenance.

The ID, creation timestamp, salience, fidelity, level, scope, trust, expiry,
lifecycle state, remaining metadata, capacity charge and ID allocator are preserved.
Content correction does not make a memory more recent or authoritative. Retired
and superseded records are refused, including conflicting legacy state aliases or
existing terminal decision markers. Quarantined, suppressed, expired or untrusted
memories can be corrected while retaining their existing retrieval restrictions.

Stored cached summaries remain part of that provenance, but context construction
uses the corrected text after a recognized content update. The shared summary
reader bypasses the historical summary; direct summaries, bounded context and LTM
formatting therefore cannot silently reuse the prior claim through that cache.

The memory, receipt, journal decision (`update-receipt-v1`) and enabled outbox event
commit together. A same-text request with a new key conflicts before embedding;
this endpoint is not a re-embedding or model-migration facility.

## Embedding preparation and compatibility

The persisted embedding contract must already identify the configured embedding
space. There is no implicit contract adoption or dimension/model conversion.
Identity descriptors are copied as canonical finite JSON; typed equality preserves
differences such as true, 1 and 1.0. The service checks identity before preparation,
after preparation and inside each commit attempt, along with the persisted contract
and vector dimension. A changed model declaration or target cannot silently reuse
an old prepared vector.

Model work runs outside store ownership. Returned vectors are copied immediately
and must be nonempty one-dimensional numeric values representable as finite float32
values. Strings, booleans, complex/object vectors, nonfinite values and dimension
mismatches are rejected. Multiple competing requests may compute vectors; only one
shared-key request can commit. Backend latency is not bounded by the journal's
ownership timeout; backend deadlines remain the caller/provider's responsibility.

Embedding identity is an operator/backend assertion, not cryptographic attestation
of model weights or proof that the vector semantically represents the new text.
The checks qualify declared-space consistency, not model quality.

## Retries and public API

Requests use exact integer IDs, strict strings, a nonempty reason of at most 1,024
UTF-8 bytes and the existing 256-byte scope/key bounds. Canonical finite JSON is
limited to 1 MiB and revalidated at the service boundary. Caller-owned request
objects cannot change the intent during storage or embedding work.

All receipted operations share the same scoped idempotency-key namespace. An exact
retry returns the original historical receipt before current lifecycle, digest or
model checks—even after a later update, retirement or supersession. A different
request under the same key conflicts. The adapter resolves model callbacks lazily,
so replay does not access an unavailable embedding property. Generic replayed
receipts must acknowledge the requested ID, text, scope and update decision.

```python
from daystrom_dml.services.receipt_lifecycle import memory_digest

record = previous_receipt["result"]["memory"]
receipt = adapter.update_memory_receipted(
    record["id"], text="Corrected content", tenant_id="example",
    expected_memory_digest=memory_digest(record), reason="Corrected a typo",
    idempotency_key="correction-1",
)
```

`POST /api/memory/update/receipt` exposes the same arguments under existing bearer
authentication. It rejects unknown fields and coercion. Success returns the receipt;
missing/wrong-scope IDs return 404, stale/lifecycle/key conflicts 409, invalid or
unsupported requests 400/422, and unavailable embedding/ownership/durable outcome
503. Fixed error codes omit underlying storage/model exception text. Service token
holders remain trusted to choose scopes; per-token authorization is unchanged.

The adapter requires opt-in receipt mode, refuses nested compensating transactions
and RAG mirroring, and adds no projection I/O to the commit. Uncertain acknowledgements
are reconciled through durable receipts, never compensated by restoring old text.
If a competing caller committed while embedding failed, the receipt can still
resolve success. Failed hydration marks runtime degradation but cannot revoke a
valid durable receipt; later owned retrieval refreshes authoritative state.
Preparation and commit guards also reconcile a competing receipt before reporting
a conflict, including a commit between an initial lookup and the next snapshot.

Projection/delta/outbox delivery carries the new text and vector. Projection queries
must match the source revision before answering. Historical receipts, events and
already-built context packets retain prior content. This is not physical erasure,
history compaction, a trust promotion or a way to retract context sent to a model.

## Qualification and remaining gates

The process harness kills actual child processes during embedding and at every
accepted receipt/outbox mutation hook: 29 cases across schemas 2, 3 and 4. Parent-held
oracles require the original acknowledged receipt, either unchanged state or the
complete content/vector update, exact provenance preservation and one revision.
Each schema also runs 256 simultaneous clients released through a start barrier,
plus eight competing processes. CI retains recovery/concurrency JUnit and covers
Linux, macOS and Windows.

The [independent review](artifacts/content-update-review-2026-09-16.json) rejected
the first implementation at 9.1/10 for a duplicate-request preparation race and
stale cached summaries. Both received regression cases and were corrected; the
frozen implementation was accepted at **9.6/10**. All **265 focused cases** passed:
116 service, 73 adapter/API, 35 recovery/concurrency and 41 independent adversarial
cases. The separate 256-client campaign passed for every supported schema.

The final full local suite passed **2,216 tests**, with 9 optional-dependency skips
and 2 known warnings, in 135.44 seconds. Ruff, the 54 maintained mypy source files
and Hermes hygiene passed. The review pins ten source/test/CI hashes; exact-commit
CI is recorded on PR #118. No user memory stores were modified.

The first exact-commit CI run (309) passed seven jobs but exposed a Windows
test-harness defect: pytest expanded the oversized Unicode input into its test ID,
exceeding the Windows environment-variable limit before that case could execute.
The other Windows job was cancelled by matrix fail-fast. Explicit short parameter
IDs retain the same input and assertions; all 73 integration cases pass after the
correction, and all 265 gate IDs are at most 190 UTF-16 units. Runtime code is
unchanged. The review records this rejected CI attempt separately, and the corrected
commit requires a fresh complete CI run on PR #118.

Process death does not qualify physical
power loss. Full-history verification and full-state events retain growing costs.
Promotion/merge, retention and physical erasure, remaining adapter extraction,
complete native KV compatibility, real-agent value/outcome campaigns and full
operator replay remain separate production gates.
