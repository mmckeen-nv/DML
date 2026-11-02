# M8 fairness blockers: event-clock and correction-authority development — November 2, 2026

Continues from the [M8 development foundation](m8-development-foundation-2026-09-29.md) at commit `b7b1fedcd41cfed195f7cfc48613572f94a9f46c` (CI run [36641399096](https://github.com/mmckeen-nv/DML/actions/runs/36641399096) verified green, 23/23 jobs, before this work began). **M8 remains open. These are proposed contracts pending independent review; nothing here is an execution freeze or a live model qualification.** No training, model search, model loading, learned-embedding inference or held-out work occurred. M7 profiles, results, replay behavior and acceptance gates are unchanged; the historical commit-time ingestion API is byte-for-byte pinned by tests.

## Blocker 1 — historical event-clock parity (proposed resolution)

The supported append API stamped `time.time()` as the authority timestamp, so ingested history ranked and presented by commit time while RAG used explicit event time; metadata timestamps and retrieval `as_of` never equalized them.

Implemented proposal:

- `receipt_ingestion.py`: new `canonical_event_request` with schema `dml-append-request-v2` binds a finite nonnegative `event_time` into the canonical request and its SHA-256 idempotency digest. `append_receipted` gains an optional `event_time`; consistency guards reject a v2 request without its bound time, a v1 request with one, and any mismatch between argument and request. When bound, the stored record's authority `timestamp` is the event time and the wall commit time is retained in record metadata as `commit_time`. When unbound, behavior is exactly the historical path (`timestamp=time.time()`, no metadata additions). Global clocks are never patched and committed snapshots are never rewritten.
- `dml_adapter.py`: new opt-in `ingest_memory_event_receipted`, mirroring the historical method's admission guards, retry and hydrate/degraded semantics. The frozen v1 profile contract is not modified; admission of this API to the production profile remains **pending independent API review**.
- `m8_dml_memory.py`: new `DMLDevelopmentEventMemory` (version `dml-m8-development-memory-v2`) ingests through the v2 API. Its evidence chain is version-isolated from v1 chains (mixed-version opens fail closed). `event_clock_qualified` remains `False`; `retrieve` reports `event_time_profile` explicitly.

## Blocker 2 — common correction authority (proposed resolution)

Previously DML's native authority authentically rejected an untrusted replacement (leaving the bridge fail-closed mid-ingestion) while the RAG arm could accept the same correction into its current view — a genuine asymmetry. Trusted-source fixtures did not establish parity.

Implemented proposal, `m8-external-correction-authority-v1` in `m8_memory_baseline.py`:

- An event envelope with `metadata.source_trust == "trusted"` is externally authenticated; anything else (absent, `untrusted`, or any invalid value) is not. This is authentication of the source/correction envelope only — source **text remains untrusted as instructions** in every arm's prompts.
- Identical pre-dispatch admission in all three arms (`PersistentRAGBaseline`, `DMLDevelopmentMemory` and subclasses, `ContextBuffer`): an unauthenticated event is rejected with `ValueError` before any mutation, partial ingestion or fail-closed bridge state. Deterministic rejection matches the manifest's zero-tolerance authorization gate rather than creating an arm divergence.
- Authenticated corrections keep the existing semantics in both persistent arms: monotonic same-source replacement applied before ranking/top-k, immutable history retained, conflicting independent sources preserved. Nothing is silently marked trusted, and RAG is not weakened: no records are hidden from it.
- The authentic native rejection is preserved and tested at the layer where it lives: an untrusted record can exist in the DML store but remains retrieval-invisible and supersession-ineligible (`ReceiptLifecycleConflict`); the neutral contract simply never requests such a replacement. Rejection evidence is retained.

## Verification

Environment: local macOS, Homebrew Python 3.12.13, NumPy 2.5.4, pinned SQLite 3.53.4 rebuilt from the CI-pinned amalgamation (SHA3-256 verified, `DYLD_LIBRARY_PATH`), `DML_SKIP_VENV_REEXEC=1`, single BLAS/OMP threads.

- Focused M8 suites plus receipt/journal regression surface: **536 tests passed, zero failures/errors/skips** across 17 files (50.2s, JUnit SHA-256 `6975c572c6bd1e076d8c7e87bcfa350ff38e5a3dbe2b640e445a1b85164b810f`) (`test_m8_*` ×6, `test_outcome_metrics`, `test_receipt_ingestion`, `test_receipt_embedding_contract`, `test_retention_recovery`, `test_projection`, `test_crash_qualification_adversarial`, `test_supersession_integration`, `test_receipt_retirement`, `test_journal_hardening`, `test_adversarial_memory`, `test_episode_tools`).
- New `test_m8_event_ingestion.py` (19 tests): digest binding, int/float canonicalization, schema isolation from v1, boundary/NaN/inf/negative/bool/text event-time rejection, record timestamp equality with commit-time retention, historical-API pinning, idempotent replay, cross-event-time key conflict, append consistency guards, injected `RevisionConflict` CAS retry, v2 bridge supersession/restart/replay, v1↔v2 chain isolation, adapter restart authority.
- New authority tests: identical unauthenticated rejection without mutation in RAG and context arms; DML bridge pre-dispatch rejection without poisoning; native untrusted-supersession rejection retained.
- Ruff: all touched M8-scope files pass (`dml_adapter.py` has 4 pre-existing unused-import findings on HEAD, untouched and unchanged by this work). mypy 3.12 target: clean on `m8_memory_baseline`, `m8_dml_memory`, `receipt_ingestion`.
- The v1 bridge test pinning `record timestamp != event timestamp` still passes, proving the old API's behavior is preserved.

## Still-open gates (unchanged or explicitly unchanged)

Independent review of both proposed contracts is **required before admission**; this tranche implements and tests proposals only. `event_clock_qualified=False` stands. Live neutral GPU runtime qualification, pinned semantic embedding admission, durable execution/cost instrumentation, leakage audit/seal, statistical calibration, resource forecast, exact-source CI and independent execution approval all remain pending before any pilot or held-out campaign. The station was reachable during this work (up 18 minutes at first contact after its September 29 shutdown); no station services or evidence were modified.
