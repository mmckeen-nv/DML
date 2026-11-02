# M8 independent-review packet: two proposed contracts — November 2, 2026

Subject: commit `522d66404976febba84303f341c94410bd2bace7` on base `b7b1fedcd41cfed195f7cfc48613572f94a9f46c` (base CI run [36641399096](https://github.com/mmckeen-nv/DML/actions/runs/36641399096) green; head CI run 41479829510 green; 536 tests, 0 failures/errors/skips locally, JUnit SHA-256 `6975c572c6bd1e076d8c7e87bcfa350ff38e5a3dbe2b640e445a1b85164b810f`). Companion narrative: [m8-fairness-blockers-2026-11-02.md](m8-fairness-blockers-2026-11-02.md); machine-readable artifact: [artifacts/m8-fairness-blockers-2026-11-02.json](artifacts/m8-fairness-blockers-2026-11-02.json). Exact change set: `git diff b7b1fed..522d664` (4 source files, 3 test files, 1 doc, 1 artifact; 583 insertions, 15 deletions).

**Both contracts are proposals pending this review. Nothing here is admitted, frozen, or qualified.** `event_clock_qualified` remains `False` everywhere; M8 remains open.

## 1. Scope and non-goals

The reviewer is asked to admit or reject, independently:

1. **Contract 1 — `dml-append-request-v2` event-time ingestion.** The API semantics (canonical request shape, digest binding, commit-path guards, record stamping) and the separately versioned `DMLDevelopmentEventMemory` bridge profile (`dml-m8-development-memory-v2`).
2. **Contract 2 — `m8-external-correction-authority-v1` common correction authority.** The envelope-authentication semantics and the identical pre-dispatch rejection installed in all three arms.

Admission means: the semantics as implemented and pinned by the tests in §4 are acceptable as the fairness basis for the M8 three-arm comparison. Admission of contract 1 additionally requires a decision on whether `ingest_memory_event_receipted` may join the frozen production profile contract; it is currently decorated `@admitted_profile_operation` on the adapter but its docstring and the bridge both record admission as pending.

Explicit non-goals (remain open regardless of the review outcome): live neutral GPU runtime qualification; pinned semantic embedding admission; durable execution/cost instrumentation; leakage audit and held-out sealing; statistical calibration; resource forecast, exact-source CI and independent execution approval. No training, model loading, learned-embedding inference or held-out work occurred in this tranche. M7 profiles, results and replay behavior are untouched; the historical commit-time ingestion API is pinned byte-for-byte by tests (§4).

## 2. Contract 1 walkthrough — `dml-append-request-v2` event-time ingestion

### API surface

- `services/receipt_ingestion.py:144` — new constant `EVENT_APPEND_SCHEMA = "dml-append-request-v2"`.
- `receipt_ingestion.py:147-151` — `validate_event_time(value)`: accepts only `int`/`float` (bool rejected by the exact `type()` check), finite, nonnegative; returns `float(value)`. Never coerces text.
- `receipt_ingestion.py:154-171` — `canonical_event_request(text, *, event_time, tenant_id, ...)`: builds the historical v1 canonical request via `canonical_request`, then overrides `schema_version` to v2 and adds the validated `event_time`; re-runs `_strict_json` and the 1 MiB cap; returns the request and its SHA-256 over the sorted, separator-tight, `allow_nan=False` serialization. Because `event_time` is a field of the canonical request, it is inside the idempotency digest by construction.
- `receipt_ingestion.py:174-185` — `append_receipted` gains optional `event_time=None` with three consistency guards executed before any journal read: a v2 request without `event_time` is rejected ("cannot be committed without its bound event_time"); a non-v2 request with `event_time` is rejected; and `validate_event_time(event_time) != request["event_time"]` is rejected ("differs from its canonical request binding").
- `receipt_ingestion.py:224-233` — at commit: `commit_time = time.time()` always; when `event_time is None` the record is stamped `timestamp=commit_time` with `meta=request["meta"]` exactly as before; when bound, `timestamp=float(event_time)` and `meta={**request["meta"], "commit_time": commit_time}`. The journal `save_with_receipt` call is otherwise unchanged.
- `dml_adapter.py:2162-2196` — new `DMLAdapter.ingest_memory_event_receipted(text, *, event_time, idempotency_key, tenant_id, ...)`: same guard set as the historical `ingest_memory_receipted` (no nesting in a compensating transaction, journal/receipts persistence required, external RAG mirroring rejected), same hydrate/degraded closures, same retry semantics via `append_receipted`. The historical method is unmodified.
- `services/m8_dml_memory.py:263-284` — `DMLDevelopmentEventMemory(DMLDevelopmentMemory)`, `version = "dml-m8-development-memory-v2"`, `event_time_profile = True`. Construction fails closed if the adapter lacks `ingest_memory_event_receipted` (line 276-277). The only behavioral override is `_receipt_append`, which routes through the v2 API with `event_time=value["timestamp"]` (the validated event timestamp). `DMLDevelopmentMemory.ingest` is refactored so v1 and v2 share every validation, evidence-chain and supersession step (`_receipt_append` hook at lines 144-147).

### Digest binding and stamping invariants

- **Digest binding.** `event_time` is a member of the canonical request, so two requests differing only in event time have different SHA-256 digests, and `int`/`float` spellings of the same value canonicalize identically (`1000` and `1000.0` both become `1000.0` before serialization). Pinned by `test_event_time_bound_into_canonical_request_and_digest` and `test_conflicting_key_with_different_event_time_rejected` (same idempotency key, different event time → `IdempotencyConflict`, one item retained).
- **No clock patching.** Wall time is read once per commit attempt via `time.time()` (`receipt_ingestion.py:224`); no global clock is monkeypatched anywhere in the change. The one monkeypatch in the new tests targets `journal.save_with_receipt` to inject a CAS race, not time.
- **Old-API preservation.** The v1 path is byte-identical: `canonical_request` is untouched, and the `event_time is None` branch reproduces the historical record exactly. Pinned by `test_historical_commit_time_api_unchanged` (timestamp within the wall window, no `commit_time` key in meta) and by the pre-existing v1 bridge assertion `records[10]["timestamp"] != events[10]["timestamp"]` (`test_m8_dml_memory.py:62`), which still passes.
- **Chain version isolation.** The evidence-chain loader rejects any row whose `version` differs from the opening bridge's version (`m8_dml_memory.py:72-74`), so a v1 bridge cannot open a v2 chain and vice versa. Pinned by `test_v1_and_v2_chains_are_version_isolated` in both directions.
- **Authority/commit separation.** Record `timestamp` equals the event time; wall commit time is retained only as `meta.commit_time` and never rewrites committed history (append-only journal; the change adds no mutation path). Pinned by `test_record_authority_timestamp_is_event_time_commit_time_retained`, `test_v2_bridge_event_times_supersession_and_restart` (per-record `timestamp == event timestamp`, `commit_time` a float, supersession intact, bridge restart restores identical records) and `test_adapter_restart_preserves_event_time_authority` (authority timestamp survives adapter close/reopen).

### Risks the reviewer should probe

- **Journal operation label.** `save_with_receipt` is still called with `operation="append-receipt-v1"` for v2 appends (`receipt_ingestion.py:240`). The receipt's `request_digest` distinguishes schemas, but the operation label does not. Decide whether receipt provenance must name the v2 schema.
- **`commit_time` meta collision.** `{**request["meta"], "commit_time": commit_time}` silently overwrites a caller-supplied `meta["commit_time"]` (v1 preserves caller meta verbatim). `canonical_request` rejects meta keys colliding with scope/`kind`/`no_merge` but not `commit_time`. Probe whether silent overwrite is acceptable or should be a rejection.
- **Event-time authentication.** The adapter accepts any finite nonnegative `event_time`; nothing binds it to an authenticated envelope at the adapter layer. Under the bridge, `event_time` is the validated event timestamp of an envelope that passed contract 2's authentication, so adapter-level trust reduces to driver trust (see contract 2 risks).
- **Float precision.** `validate_event_time` coerces `int` to `float`; ints above 2^53 lose precision deterministically, and the digest binds the coerced value. Boundary 0 and 2^53 are pinned; behavior beyond 2^53 is coerced-but-deterministic, not rejected.
- **Guard completeness.** The three `append_receipted` guards pin the argument/request/schema consistency surface, but only the adapter and tests call `append_receipted`; confirm no other in-repo caller can now pass `event_time` (the parameter is keyword-only and defaults to `None`, so existing callers are untouched — verify by callers, not by assumption).

## 3. Contract 2 walkthrough — `m8-external-correction-authority-v1`

### Authority semantics

`services/m8_memory_baseline.py:50-69`:

- `AUTHORITY_PROFILE = "m8-external-correction-authority-v1"`.
- `source_authenticated(envelope)`: reads `envelope["metadata"]["source_trust"]`, defaulting absent metadata or absent key to `"untrusted"`. Only the exact strings `"trusted"`/`"untrusted"` are legal; anything else (`None`, `1`, `"expired"`) raises `ValueError("invalid explicit source trust")`. Returns `True` only for `"trusted"`.
- `require_authenticated_source(envelope)`: raises `ValueError("unauthenticated source cannot be admitted to any arm")` when not authenticated.
- This is envelope authentication only: source **text remains untrusted as instructions** in every arm's prompts (docstring, lines 54-59).

### All-arm admission equality

The identical call is installed pre-dispatch in all three arms:

- RAG: `PersistentRAGBaseline.ingest`, `m8_memory_baseline.py:161` — after `validate_event`, before vector validation and before `BEGIN IMMEDIATE` (line 167), so rejection precedes any table mutation.
- DML bridge: `DMLDevelopmentMemory.ingest`, `m8_dml_memory.py:156` — after `validate_event` and the owned-scope check, before duplicate/chronology/correction checks and before `self.failed = True` (line 184), so rejection neither mutates the evidence chain nor poisons the fail-closed bridge. Inherited unchanged by `DMLDevelopmentEventMemory`.
- Context-only: `ContextBuffer.ingest`, `m8_memory_baseline.py:239` — after `validate_event` and the scope check, before the duplicate/nonchronological check and before any buffer mutation.

In all three arms an unauthenticated event is rejected with `ValueError` before any state change: it cannot enter a current view, replace a predecessor, or be presented as evidence. Authenticated corrections keep pre-existing semantics in both persistent arms (monotonic same-source replacement before ranking/top-k; immutable history retained; conflicting independent sources preserved). No record is hidden from RAG and nothing is silently marked trusted.

### What changed in the untrusted-rejection path

Previously the only untrusted-source test was `test_untrusted_replacement_remains_authentically_rejected`: an untrusted v2-correction passed bridge validation, was appended to the DML store, then the native supersession layer raised `ReceiptLifecycleConflict` mid-ingestion — leaving the bridge fail-closed with two retained append receipts, while the RAG arm could accept the same correction into its current view. That was the genuine asymmetry.

That test is **replaced by two tests** (`git diff b7b1fed..522d664 -- dml_core/tests/test_m8_dml_memory.py`):

1. `test_unauthenticated_events_rejected_before_dispatch_without_poisoning` — untrusted new event, untrusted correction, and invalid trust value are each rejected pre-dispatch; the evidence file is byte-identical afterward, `bridge.failed` is `False`, only the original receipt exists, and reads return the original record.
2. `test_native_supersession_still_rejects_untrusted_replacement` — the native DML authority's authentic rejection is retained and exercised at the layer where it lives: an untrusted record written directly via `adapter.ingest_memory_receipted` exists in the store, but `supersede_memory_receipted` toward it raises `ReceiptLifecycleConflict` and the predecessor is not marked superseded. The neutral contract simply never requests such a replacement.

### Risks the reviewer should probe

- **Does pre-dispatch rejection hide any arm asymmetry?** The rejection itself is now identical, but ordering relative to other checks is not fully uniform: the DML and context arms check scope before authentication; the RAG arm has no ingest-time scope-equality check (scope is validated, not compared against a stored owner) and authenticates before vector validation. Rejection *outcomes* for unauthenticated events are identical in all arms; rejection *precedence* for events that are simultaneously malformed and unauthenticated is not pinned cross-arm. Probe whether the contract should specify precedence, or only the admit/reject decision.
- **`source_trust` is caller-asserted metadata.** Any driver can set `metadata.source_trust="trusted"`; the envelope carries no signature or provenance. This matches the baseline-audit driver model ("authenticated non-model drivers deliver the same scoped event stream", [m8-baseline-audit-2026-09-29.md](m8-baseline-audit-2026-09-29.md):40), where authentication is the driver's responsibility. The reviewer should rule on whether driver-asserted trust is in-scope for M8 or whether the envelope needs verifiable provenance.
- **Breaking change for existing drivers.** Absent `source_trust` now defaults to rejected in every arm; fixtures were updated accordingly (`test_m8_memory_baseline.py:19-23`). Confirm no M8 driver plan relied on implicit trust.
- **Native rejection now unreachable under the neutral contract.** The fail-closed mid-ingestion path that previously fired on untrusted corrections can no longer be triggered through the bridge; it is exercised only by direct adapter calls in the retained native test. Decide whether that is sufficient evidence of the native authority, or whether the harness must still exercise it.
- **Metadata snapshot.** `validate_event` snapshots the envelope via JSON round-trip (`m8_memory_baseline.py:98`), so post-ingest caller mutation cannot alter retained state — pinned by the updated `test_context_returned_envelopes_cannot_mutate_retained_state`. Confirm this covers the trust field in all arms (it does: the check reads the validated snapshot, not the caller object).

## 4. Test matrix mapping

New file `dml_core/tests/test_m8_event_ingestion.py` (19 tests: 12 functions, one parametrized ×7):

| Test | Invariant pinned |
|---|---|
| `test_event_time_bound_into_canonical_request_and_digest` | `event_time` inside canonical request and SHA-256 digest; int/float canonicalize identically; v1 schema string and shape unchanged |
| `test_event_time_validation_rejects_nonfinite_negative_and_text` (×7: `True`, `"1000"`, `None`, NaN, inf, `-1`, `-0.5`) | strict typing; no bool/text coercion; finite nonnegative only |
| `test_event_time_boundary_zero_and_large_values` | boundary 0 accepted; 2^53 exact |
| `test_record_authority_timestamp_is_event_time_commit_time_retained` | authority `timestamp == event_time`; `meta.commit_time` within wall window |
| `test_historical_commit_time_api_unchanged` | old-API preservation: commit-time stamp, no `commit_time` key in meta |
| `test_idempotent_event_retry_returns_original_receipt` | idempotent replay returns the original receipt; one item committed |
| `test_conflicting_key_with_different_event_time_rejected` | digest binding: same key + different event time → `IdempotencyConflict`, no second item |
| `test_append_consistency_guards` | all three argument/request/schema guards fire; zero mutation |
| `test_revision_conflict_retries_then_commits_event_time` | CAS retry preserves the bound event time (injected `RevisionConflict`, 2 attempts) |
| `test_v2_bridge_event_times_supersession_and_restart` | end-to-end bridge parity: record timestamps equal event timestamps, `commit_time` retained, supersession intact, bridge restart restores identical records; `event_clock_qualified is False`, `event_time_profile is True` |
| `test_v2_bridge_preserves_validation_and_no_clock_patching` | duplicate/nonchronological/implicit-version/untrusted rejections leave evidence bytes unchanged, bridge unpoisoned; stamped time is event time (1001.0), not wall time |
| `test_v1_and_v2_chains_are_version_isolated` | mixed-version chain opens fail closed in both directions |
| `test_adapter_restart_preserves_event_time_authority` | event-time authority and `commit_time` survive adapter close/reopen |

Changed `dml_core/tests/test_m8_memory_baseline.py`:

| Test | Invariant pinned |
|---|---|
| `event()` fixture (lines 19-23) | default `metadata={"source_trust": "trusted"}` — existing tests now run under explicit authentication |
| `test_context_returned_envelopes_cannot_mutate_retained_state` (updated) | retained-state immutability now pinned with explicit trusted metadata present |
| `test_unauthenticated_source_rejected_without_mutation` (×4: `untrusted`, `expired`, `None`, `1`) | RAG arm: new event and correction both rejected pre-dispatch; `history()` unchanged; memory table row count stays 1 |
| `test_context_buffer_rejects_unauthenticated_without_mutation` | context-only arm: identical rejection; presented evidence unchanged |

Changed `dml_core/tests/test_m8_dml_memory.py`:

| Test | Invariant pinned |
|---|---|
| `test_unauthenticated_events_rejected_before_dispatch_without_poisoning` (new) | DML arm: pre-dispatch rejection of untrusted/invalid events; evidence chain byte-identical; bridge not failed; receipt count unchanged |
| `test_native_supersession_still_rejects_untrusted_replacement` (new, replaces `test_untrusted_replacement_remains_authentically_rejected`) | native DML authority retained: untrusted record may exist in the store but supersession toward it raises `ReceiptLifecycleConflict`; predecessor not superseded |
| `test_real_dml_supersession_projection_and_restart` (pre-existing, line 62) | v1 bridge record timestamp ≠ event timestamp — old-API behavior preserved |

Net new tests: 25 (19 + 4 + 2 authority tests, minus the 1 replaced), matching the artifact's `new_tests: 25`; total suite 536/0 failures.

## 5. Open questions for the reviewer

1. Should `dml-append-request-v2` / `ingest_memory_event_receipted` be admitted to the frozen production profile contract, or confined to the development bridge pending runtime qualification? What evidence beyond this tranche would admission require?
2. Is the unchanged journal operation label `"append-receipt-v1"` for v2 appends acceptable provenance, or must receipts name the v2 schema?
3. Should a caller-supplied `meta["commit_time"]` be rejected on the v2 path instead of silently overwritten?
4. Is driver-asserted `metadata.source_trust` sufficient authentication for M8 (per the baseline-audit driver model), or must the envelope carry verifiable provenance? If the latter, contract 2's semantics change.
5. Must rejection *precedence* (authentication vs scope/shape/vector validation order) be identical across arms, or is identical admit/reject outcome sufficient?
6. Is retaining the native untrusted-supersession rejection as an adapter-layer test sufficient evidence, given the neutral contract can no longer trigger it through any bridge?
7. What qualification evidence would flip `event_clock_qualified` from `False`, and is that decision part of this review or the later runtime-qualification gate?
8. Are event times themselves authenticated (contract 2 authenticates the envelope; the adapter accepts any finite `event_time`), and does anything need to bind the two at the adapter layer?
