# Independent review: M8 transport-native fairness amendment — November 9, 2026

**Status: independent review of [the proposed amendment](m8-transport-native-fairness-2026-11-05.md). Not an approval gate action by itself; the amendment remains PROPOSED until the owner resolves the conditions below.** Reviewer independence: this review was conducted outside the amendment's authoring team; the adversarial first pass was run via `opencode` on `nvidia/deepseek-ai/deepseek-v4-pro` against the repo docs, and every cited claim in that pass was re-verified against the source documents before adoption. No files were modified; nothing here is committed.

**Evidence read:** the amendment; [policy-control diagnostic](m8-policy-control-2026-11-05.md); [fairness manifest](m8-fairness-manifest-2026-09-29.md) (original arm design + acceptance gates); [verifier-shape analysis](m8-verifier-shape-analysis-2026-11-02.md); [M9 charter](m9-charter-2026-11-06.md); [M9b candidate-2 gate record](m9b-gate-candidate2-2026-11-09.md).

## Verdict: APPROVE-WITH-CONDITIONS

The core move is correct and evidence-forced: surface-conditioned retrieval (36/36 retrieve-first under the DML surface, 24/24 of those needless; 0/72 under the neutral protocol — policy-control §Headline) is a measured fact, and it falsifies the neutral-protocol comparison's central assumption. Re-binding all arms to the trained surface converts retrieve-always from a confound into a controlled constant *for the persistent-arm comparison*, and the DML-vs-RAG estimand survives intact. The amendment is also honest about what it does not claim (§"Explicitly not claimed") and about finalization weakness being measured, not hidden.

But as written it has one near-fatal flaw and several material gaps. Conditions C1–C3 are **mandatory** (the amendment is a REJECT without C1 and C2); C4–C7 are **required before campaign freeze**.

## Answers to the review questions

### 1. Does the re-binding remove the confound, or rename it?

**Removes it for `dml_memory` vs `rag_memory`; renames it for `no_memory`.** For the two persistent arms, the treatment (what retrieve returns) is cleanly isolated given presentation parity (see C4). For `no_memory`, the policy retrieves unconditionally into a dead tool — and the control diagnostic already measured what happens: 9/12 step-limit identical-retrieve loops on empty stores at baseline (policy-control §Secondary finding). That arm does not measure "policy without memory"; it measures "policy + dead tool," i.e. the surface-cue confound re-enters as a *behavioral* confound. The amendment concedes this ("not context-only in the old sense", amendment §Fairness properties 4) but still tables the arm as "the context floor." It is not the context floor; the context floor is the evidence-fed `inline_evidence` cell, which the diagnostic found "correct as built" (policy-control §Fixture repair, WS-B). The amendment's defense — "loop statistics are first-class outcomes" — is necessary but not sufficient: it makes the contamination visible without making the arm a valid control.

### 2. Are the stated fairness properties sufficient? What is missing?

The four stated properties (identical model/prompt/sampler/surface; backend-only variation; event-clock parity; cost accounting) are necessary but incomplete. Missing:

- **Retrieve-*return* parity.** Property 2 binds only ingestion ("identical event streams"). Nothing binds the *returned* records' ordering, metadata fields, provenance, formatting, or top-k between the DML receipt store and the RAG wrapper — the original manifest explicitly required identical whole-record presentation, top-k ≤ 8, and the 2,048-token ceiling (manifest §Fixed candidate). A receipt-metadata-rich DML return vs a receipt-free RAG return is a presentation confound in exactly the citation channel the strict verifier scores.
- **Pairing validity after divergence.** "The only varying component is the retrieve backend" is true only at step 1; after the first backend response, trajectories legitimately diverge. The manifest's paired design needs a mechanical anchor under the new binding: assert (and record) that the **first emitted retrieve query is byte-identical across arms per scenario** (same prompt, same seed-reset sampler — this must hold, and its failure is a harness defect).
- **Acceptance-gate re-derivation.** The amendment changes arm semantics but never re-states the manifest's gate table under them. See issue 3 below.
- **Cost-gate restatement.** "Retrieval cost is now incurred by every arm (constant)" (amendment §3) is false: `no_memory` pays ~nothing, RAG pays embedding + vector search, DML pays receipt overhead. Costs are comparable, not constant; the DML/RAG ratio and p95 gates survive, but "one query per store" amortization (manifest §Cache/order/cost) has no store for `no_memory`, and the claim of constancy should be struck.

### 3. Hidden metrics that would look bad

- **Abstention-cause conflation (the main risk).** A dead-tool arm will abstain on *answer-present* cells. Whether those abstentions are reported as "calibrated caution" (flattering) or as control-arm failures (correct) is not pre-specified. The amendment lists valid-final/unsupported-final/step-limit/loop rates as first-class but never fixes the classification of control-arm abstention on answer-present cells. The manifest's rule — "abstention can avoid a stale claim but still fail correctness" — handles scoring, but pooled abstention-rate reporting could still launder a disabled tool into prudence. **Pre-register abstention-by-cause per arm before unblinding** (C6).
- **The −2pt context non-inferiority gate quietly changes meaning.** Against a dead-tool arm that loops/abstains on answer-present cells, DML clears "DML − context > −2pt" trivially. That gate was derived against an *evidence-fed* context-only arm (manifest §Proposed acceptance) and encoded the milestone's "no material regression" claim. Left unrevised, the amendment would let a vacuous gate inherit the original gate's evidentiary weight. This is the closest thing to a hidden-metric problem in the proposal.
- Not hidden, but worth stating: the absolute ≥90% DML correctness gate remains the binding constraint (qualified candidate sits at 8/36 valid on the dev diagnostic, exactly at its own gate threshold with no margin — m9b §Scorecard note). The amendment makes the campaign *runnable*; it does not make it *winnable*, and review materials should not imply otherwise.

### 4. Cheaper/better alternatives not considered

- **Promote `inline_evidence` to a full arm (the obvious fix, mentioned but not adopted).** The amendment itself notes it "can be promoted to an arm with the same transport" (§Fairness properties 4) and then doesn't. Adopting it restores the manifest's non-inferiority intent at near-zero new machinery — the cells already exist and are validated. This is the single largest avoidable weakness.
- **Fixed-query single-retrieve pairing** for the persistent arms: one fixed retrieve query per scenario, identical across arms, then free continuation — tighter pairing than fully free-running episodes, cheaper than re-running the full battery for pairing diagnostics. Not evaluated.
- **Re-verify the premise under the qualified policy.** The amendment was written 2026-11-05 against llama3-sft-v3; the campaign policy is now qwen3.5-9b-sft-v1 (qualified 2026-11-09). Gate 6 shows retrieve-first 12/12 on *answer-present* cells (m9b §Scorecard), but the needless-retrieval and neutral-protocol behavior of the qwen adapter — the exact evidence the amendment's "no third option" argument stands on — is unreported. The premise is probably still true (gate 6 exists precisely because retrieve-always was retained), but "probably" is not evidence. A 36-episode diagnostic re-run under the qualified profile is cheap and already tooled.

### 5. Verdict rationale

APPROVE-WITH-CONDITIONS, not REJECT, because: the diagnosis is correct and forced by retained evidence; the primary estimand (DML vs RAG) is sound; the dishonest options (neutral-prompt revision to manufacture retrieval) are explicitly and correctly excluded; and every defect found is repairable by specification, not by redesign. Not unconditional APPROVE, because C1 and C2 as written would let the campaign claim a context-non-inferiority result against a control that measures a dead tool.

## Conditions

| # | Condition | Severity |
|---|---|---|
| C1 | **Re-verify the surface-cued-retrieval premise under the qualified qwen3.5-9b-sft-v1 profile** (needless-retrieval cells + neutral-protocol probe) before freeze; record as an amendment addendum. | Mandatory |
| C2 | **Promote `inline_evidence` to a full arm** through the same transport and re-derive the context non-inferiority gate against it. `no_memory` may be retained only as a fourth, explicitly-labeled *policy-floor* arm; it may not carry the manifest's context-only gates. | Mandatory |
| C3 | **Re-derive the full acceptance-gate table under the new arm semantics** — state per gate: survives / void / re-estimand — and re-run the power/coverage planning for any re-derived gate before freeze. | Mandatory |
| C4 | **Specify and enforce retrieve-return presentation parity** between `dml_memory` and `rag_memory`: ordering, metadata/provenance fields, whole-record formatting, top-k, token ceiling (inherit manifest §Fixed-candidate values). | Required before freeze |
| C5 | **Add the first-action pairing check**: per scenario, the first emitted retrieve query must be byte-identical across arms; record it per episode; treat violations as harness defects. | Required before freeze |
| C6 | **Pre-register abstention-by-cause classification** per arm (genuine-insufficiency vs dead-tool vs loop-termination), including the rule that control-arm abstention on answer-present cells fails correctness and is reported as a control failure. | Required before freeze |
| C7 | **Strike the cost-constancy claim** and restate cost gates against the actual backends (DML receipt overhead vs RAG embedding+search; no store for `no_memory`). | Required before freeze |

## Top 3 issues (summary)

1. **Stale premise**: the amendment's falsification argument was measured on llama3-sft-v3; the campaign policy is now the qwen3.5-9b adapter qualified four days later, whose needless-retrieval/neutral behavior is unmeasured (C1).
2. **`no_memory` is not a context floor**: a dead retrieve under an unconditionally-retrieving trained policy measures "policy + dead tool" (9/12 empty-store loops at baseline), and it silently vacates the manifest's −2pt context non-inferiority gate (C2, C3).
3. **Presentation/pairing gaps in the primary comparison**: return-format parity between DML and RAG retrieve backends is unspecified, and the "backend-only variation" claim holds only at step 1 — pairing needs the mechanical first-action identity check (C4, C5).
