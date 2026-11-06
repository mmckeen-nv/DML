# M8 DML-policy control diagnostic — first x86 run, November 5, 2026

**Status: development evidence. No competence claim; informs the neutral-protocol revision decision (H1 vs H2) only.** First qualification-adjacent run on the x86 development runtime (192.168.50.133) after the station shutdown/migration.

## Run identity

- 36 episodes: 12 battery fixtures × 3 controls (`none`, `inline_evidence`, `empty_store`) × seed 0; wall 587 s (~16 s/episode on the RTX PRO 5000).
- Policy: authenticated llama3-sft-v3 snapshot (unchanged weights), runtime `dml-llama3-sft-action-runtime-v2:c4b76a42…19cb`, consumer profile `llama3-8b-instruct-sft-v3-bf16-action-json-v1`.
- Runtime: **declared development identity** `development-x86-20261102` (x86_64, driver 595.99.02, RTX PRO 5000 72GB, capability (12,0)), recorded in every episode record; vllm image `vllm/vllm-openai@sha256:04563c30…b5641d` (amd64), torch 2.11.0+cu130 / transformers 5.6.2 / tokenizers 0.22.2 identical to the GB300 pins; pinned SQLite 3.53.4 rebuilt for x86 (lib sha256 `34886f3d…`).
- Evidence (station-local, raw): `/home/test/dml-debug/m8-control-20261102/` — `control.jsonl` sha256 `f93b2e1839e70651faa2d253a9757dc59d8f0f27ee470afa0019528a77c7b37c`, `control.summary.json` sha256 `b4eb5fdeef4641b9040c13b126aae96279f95a1a4b860e853c4447c597134a34`.

## Headline result

**Retrieval under the trained DML surface is unconditional: 36/36 episodes opened with `retrieve` (retrieve_first rate 1.0) — including 24/24 episodes where retrieval was needless by design** (12 inline-evidence, 12 empty-store; needless_retrieval_rate 1.0 in every cell).

| cell | episodes | retrieval | needless retrieval | valid cited final | step_limit |
|---|---|---|---|---|---|
| answer_present | 4 | 4 | 2/2 | 0 | 1 |
| correction_chain | 18 | 18 | 9/9 | 2 | 3 |
| conflicting | 2 | 2 | 1/1 | 0 | 0 |
| empty_store | 12 | 12 | 12/12 | 0 | 9 |

Contrast with the competence battery (commit `dba301c`): under the neutral protocol the same policy made **0/72 retrieval attempts**. The behavior flips entirely with the action/transport surface.

## Interpretation against the predeclared hypotheses

- **H2 (surface-cued retrieval) — supported.** The policy retrieves whenever the DML surface is present, including when retrieval cannot help (empty store) and when it is unnecessary (answer inline). Retrieval is conditioned on the trained transport, not on need.
- **H1 (policy-text dependence) — dead.** A need-driven policy would skip retrieval in the needless cells; it does not. No neutral-prompt revision (R1–R3) can manufacture need-driven behavior that the trained surface itself does not produce — and any neutral change that *does* restore retrieval would be measuring prompt familiarity, not protocol competence.
- H0 (single-sample artifact) was already excluded by the battery's a priori rule.

**Decision this informs (for the reviewer, not this document):** the neutral protocol cannot be made fairness-eligible by prompt revision. Options narrow to (a) accept that retrieval-competence comparisons require the DML-native transport on every arm (a fairness-framework redesign), or (b) treat neutral-protocol competence as out of scope for M8 and rescope the milestone's protocol gate accordingly. This run does not choose between them.

## Secondary finding: finalization weakness under the DML surface

Only **2/36 episodes produced a valid cited final** under the trained surface, with 13 step_limit terminations. The pattern in needless cells is a retrieval loop: on an empty store the policy retrieves repeatedly rather than finalizing (9/12 step_limits); with inline evidence it retrieves first and sometimes loops (4/12). In `none` controls 12/12 completed (2 valid, the rest unsupported/empty-claims finals — consistent with the battery's context-arm findings being verifier-shape issues, not plumbing). The trained policy's competence is thus *retrieve-always, finalize-weakly* — relevant to how any future retrieval gate should be written (a retrieval gate alone would credit surface-conditioned loops).

## Amendment probe program (same evening, same runtime)

After the baseline run exposed the loop attractor, four named, recorded prompt amendments were run through the same diagnostic (36 episodes each; all retained on 133 under `/home/test/dml-debug/m8-control-{,am1-,am2-,am3-,am4-}20261105/`):

| Run | Amendment | completed | step_limit | empty_claims (of 12 empty-store) | phantom-cited fabrications (empty-store) | unsupported | valid |
|---|---|---|---|---|---|---|---|
| baseline | none | 23 | 13 | 0 | 2 | 21 | 2 |
| am1 | terminate-on-empty-v1 | 32 | 4 | 0 | 10 | 29 | 3 |
| am2 | abstain-on-empty-v2 | 33 | 3 | 2 | 3 | 29 | 2 |
| am3 | exact-shape-abstain-v3 | 33 | 3 | 6 | 4 | 24 | 3 |
| am4 | no-phantom-citation-v4 | 34 | 2 | 8 | 2 | 23 | 3 |

Evidence hashes: am1 jsonl `c98f47ca…1cbc`, am2 `79a89236…3b5b`, am3 `72132a1a…38`, am4 jsonl `49e4c569…a87e` / summary `118a50c2…173c` (all under `/home/test/dml-debug/m8-control-am{1,2,3,4}-20261105/`).

Mechanics by iteration: baseline loops identical retrieves 6× on empty stores (temperature-0 attractor). v1's termination rules broke the loops but converted them to **fabrication** (unsupported 21→29 — finals asserting facts no record supports). v2's abstention rule produced value-level abstention (`"value": "unknown"/null` claims with empty evidence_ids) but not the declared insufficiency shape. v3's exact-shape few-shot example produced the declared `claims: []` abstention in 6/12 empty-store episodes — but 5/12 still fabricated, **citing phantom record ids that no retrieve call ever returned** (the dangerous failure class: confident, falsely-cited wrong answers; the strict verifier correctly rejects every one). am4's explicit citation-validity rule closed the arc: **zero step-limits and 8/12 declared-shape abstentions on empty stores**, with the residual being 2/12 phantom-cited fabrications plus 2/12 null-valued pseudo-abstentions (semantically abstaining, shape-wrong).

Reading for the milestone: prompt-level rules are a strong lever against *shape* failures (loops 13→0, declared abstention 0→8/12) and a weak lever against *parametric* failures (phantom-cited fabrication persists at ~2-4/36 across all variants; the valid-cited rate on answer-present cells stays at 3/36, dominated by verifier-shape and conflict-handling issues). The system — verifier, schema, fail-closed contracts — never once credited a fabrication in 180 episodes; the model is the weak component. The prompt program has converged: the residual gap is trained-in behavior, and the honest next step is a bounded training decision (out of M8's no-training scope — owner decision as a new milestone), with this program's evidence as its specification.

## Reproduce

On x86-dev: repo at `ffda259`, `bash /tmp/m8-control-run-133.sh` (script + pins recorded in the evidence dir log; requires `DML_DEV_RUNTIME=development-x86-20261102`). Caveats: test_injected episode envelope (not live_local), fixture workloads, single seed, development runtime identity (not the M7-qualified aarch64 identity), GPU shared with an unrelated small-model server during the run (no evidence of interference; per-episode wall times consistent).
