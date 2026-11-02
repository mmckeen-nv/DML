# M8 competence battery — first development run — November 2, 2026

First multi-scenario development battery of the neutral three-arm protocol on the pinned station environment. **Development protocol-competence evidence only: fixture workloads, no held-out data, no benchmark, no quality claim. M8 remains open.** Raw evidence on the station at `/home/nvidia/dml-debug/m8-battery-20261102/` (`battery.jsonl` SHA-256 `e25833e19ddcdb2e48097da9953ad84020ebbbb9958eba6f02cc6e4a13af7fa3`, `battery.summary.json` `3014b82ae6b616e1ab86e9771d651f5c7928e5c964ab690192e294094a571b8c`, `battery.log` `7ac3a26079439c19a8d51a041b57e49dc69060e902b0f9542ca7ed50f1380882`).

## Setup

- 12 development fixtures across all six families (schema `dml.m8.competence-battery.v1`), 3 declared sampler seeds (0 = pinned; 1, 2 = development diagnostics labeled in every episode), 3 arms = **108 episodes**, wall 3,092s. Same dedicated-GPU discipline as the smoke: no competing GPU work during the run (1 MiB used after exit).
- Harness: battery runner (`scripts/m8_battery/`) with the pinned MiniLM embedder, `DMLDevelopmentEventMemory` (v2 event-time) DML bridge, durable RAG, context buffer, `M8NeutralConsumer` per arm-episode, strict neutral parsing and `validate_final` with fixture truth oracles. Zero runner/plumbing failures; all 102 failures are strict protocol rejections (M8ProtocolError), retained, no retries, no repair.

## Headline result

**Persistent-arm non-retrieval is systematic, not sampling noise.** Across 72 persistent-arm episodes (24 per seed × 3 seeds), retrieval was attempted **0/72**; every episode ended in an immediate unsupported final — either empty-evidence "unknown" claims or fabricated citations (e.g. `evidence_ids: [1]` against record IDs in the hundreds), all rejected by strict validation. By the a priori decision rule in the [competence analysis](m8-protocol-competence-analysis-2026-11-02.md) (H0: single-sample artifact), **H0 is excluded**: the smoke's non-retrieval replicates across scenarios and seeds at rate 1.0.

Context-only behaved as designed: inline evidence, no retrieval, 6/36 valid cited finals (the changing-facts and atlas-gateway fixtures, consistently per seed).

## Secondary finding — verifier-vs-model value-shape friction (context arm)

All 30 context-arm failures are `Citation does not support claim`. Inspection shows two systematic shape mismatches, not absent evidence: (a) claim **key** mismatch — the model answers with a natural key (`status`, `tax_rate`) while fixture truth uses namespaced keys (`status_bulletin`, `ingest_retry_backoff_cap`), and the verifier admits only declared keys; (b) exact **value-form** mismatch — `300` vs `300 seconds`, `8.25` vs `8.25%`, where the cited record states the fact in the truth's exact form. These are battery-verifier contract questions (how strict key/value identity should be), recorded for the review process — they do not excuse the persistent arms' fabrication, and they must not be relaxed to manufacture passes.

## Interpretation vs the analysis decision rules

- H0 (single-sample artifact): **excluded** (0/72 across seeds/scenarios).
- H1 (policy-text dependence) vs H2 (tool-surface conditioning): **not yet separable** — both predict what we see. The next decisive experiment is the DML-policy control arm from the analysis design (dev-only diagnostic): does the adapter retrieve under its trained surface on these fixtures, and does it retrieve even when unnecessary (surface-cued) or only when needed (need-driven)? That answer gates whether any fairness-eligible neutral revision (R1–R3) can work.
- H3 (citation-schema mismatch): further weakened — context-only cites neutral envelopes correctly in 6 valid finals; failures are key/value-shape, not schema, issues.

## Harness qualifications confirmed

Battery runner, fixture schema, per-scenario fresh stores, per-episode single-use consumers, seed labeling, evidence JSONL + per-family summaries, strict rejection paths — all performed as designed at 108-episode scale. The remaining protocol-competence gate is model behavior, addressed above.

## Next steps

1. DML-policy control diagnostic (dev-only, declared): retrieve-rate and need-vs-surface-cue measurement on the same fixtures. Requires a declared dev path to the DML policy surface with the neutral harness — designed, not yet reviewed.
2. Review the verifier key/value-shape contract questions with the contract reviewer (battery-only issue; the manifest's independent domain verifier remains the acceptance path).
3. Neutral SYSTEM/transport revision decision (R1–R3) **only after** (1) lands; any revision goes through the reviewed contract process with the analysis doc's fairness questions.
4. Deferred, unblocked but not started: embedding admission formalization (wrapper exists, admission pending), durable execution/cost instrumentation wiring into the runner, statistical coverage calibration for `m8_statistics`/`m8_bootstrap` (implementations landed, calibration pending).
