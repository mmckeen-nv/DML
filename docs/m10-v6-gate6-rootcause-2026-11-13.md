# M10 v6 frozen-diagnostic gate-6 regression — root cause (2026-11-13)

**Verdict: scorer mis-scope on gate 6, PLUS a real (non-gate) inline_evidence
behavioral regression.** Under the M9b-chartered definition of gate 6
(retrieve-first on answer-present = the 12 `none`-arm episodes), the v6 child
scores **12/12 → gate 6 PASSES**, and the corrected v6 scorecard is
**PASS 6/6**. The recorded FAIL (3/4) comes from the M10 scorer folding the two
`answer_present × inline_evidence` episodes into the gate-6 denominator — a
definition under which the LOCKED M9-qualified adapter's own qualification run
also scores 3/4 (i.e., would have failed the gate it passed and was locked
under). Separately and genuinely, v6 did regress on the `inline_evidence` arm:
it reverted to the qualified adapter's known 3-episode retrieval-skip boundary
that v5 had suppressed, and added one `invalid_action`. That regression is real
but belongs to the H2b/needless-retrieval axis, not to gate 6 as chartered.

## Evidence

| artifact | sha256 |
|---|---|
| v5 gate control.jsonl (`/home/test/dml-debug/m10-gate-v5-20261113/`) | `8d260a5394242795…` |
| v6 gate control.jsonl (`/home/test/dml-debug/m10-gate-v6-20261113/`) | `bdc38296d6905f10…` |
| C1 re-check control.jsonl (`/home/test/dml-debug/m8-recheck-qwen3-control-20261110/`, QUALIFIED M9 adapter, profile `qwen3_5-9b-sft-v1-bf16-action-json-v1`) | `57e97d6d215340ec…` |
| m9b qualification run control.jsonl (local copy, `m9b-gate-v2-20261109`) | `1fafefd3eab2d054…` |
| `dml_core/scripts/m8_battery/control.py` @ a66277a | `15de8f3d76246775…` |

Adapters: v5 gate = `development-m10-v5-20261113` (adapter
`af23908a92d6b916…`); v6 gate = `development-m10-v6-20261113` (adapter
`7b28788897e78536…`). Sampling seeds identical per episode across the two
gates; same scorer for both (scorecard.json schema identical).

## 1. Episode-level diff, v5 → v6 (all 12 inline_evidence cells)

5 of 12 inline_evidence episodes flipped; the other 7 are byte-level behavior
matches. The `none` and `empty_store` arms are completely unchanged (24/24
identical verdicts, all retrieve-first preserved).

| scenario (cell) | v5 inline_evidence | v6 inline_evidence | flip |
|---|---|---|---|
| conflicting_independent_sources_status (conflicting) | valid_cited_final, rf=True | **unsupported_final, rf=False** | retrieve-first lost; answered from embedded text citing inline ids 20/21 |
| expcorr_atlas_deploy_official_vs_forum (correction_chain) | valid_cited_final, rf=True | **invalid_action** (1 call, verdict None) | terminated, see §3 |
| isoscope_atlas_gateway_version (correction_chain) | valid_cited_final, rf=True | **empty_claims_final**, rf=True | valid → over-abstention |
| isoscope_nova_api_endpoint (correction_chain) | valid_cited_final, rf=True | **empty_claims_final**, rf=True | valid → over-abstention |
| stable_incident_absent_resolution (answer_present) | valid_cited_final, rf=True, canonical keys (`ir_104_root_cause`, …) citing retrieved ids 0/1 | **unsupported_final, rf=False**, verbose NL keys citing inline ids 40/41/42 | the recorded gate-6 "miss" |

This is a **systematic inline_evidence shift, not one marginal episode**:
2 retrieval-skips + 1 invalid action + 2 valid→abstention flips, all confined
to the inline_evidence arm. Net inline retrieve-first: v5 12/12 → v6 9/12
(+1 unparseable).

## 2. Hypothesis test: the v5 `answer_supported` class

**Denied as the regression mechanism, on two independent grounds.**

1. **Byte-identity.** v6 is additive-only: all thirteen v3–v5 class files,
   `answer_supported` (15 eps, sha `95b709cc5be281c4`) included, regenerate
   byte-identical between the v5 and v6 datasets (pinned by
   `test_v5_class_bytes_unchanged`; docs/m10-curriculum-v6-2026-11-13.md). The
   same bytes produced 0/12 inline skips in the v5-trained model and 2/12 + 1
   invalid in the v6-trained model, so the class content alone cannot explain
   the divergence. The only curriculum delta is `dependent_lookup_compose`
   (40 new eps) plus ordinary training stochasticity.
2. **What the class actually teaches.** Every `_answer_supported` gold episode
   issues `ep.retrieve(...)` and finals from the *retrieved supporting record*
   ("Report precisely what the stored record says and cite it"). The adverse
   notice is a retrieved *record* (an abstain-trigger surface), not
   prompt-embedded evidence; the near-miss is empty-claims abstention. The
   class teaches retrieve-then-answer *despite* adverse inline-looking
   content — the opposite of answer-from-embedded-text.

The observed v6 pattern (answer straight from the embedded inline text, cite
the inline numbering, verbose non-canonical claim keys) matches neither
`answer_supported`'s gold nor any other byte-pinned class. It matches the
**M9-qualified adapter's own boundary behavior** (§4) — i.e., v6's added
training diluted/forgot the suppression v5 had established, rather than v5
content actively teaching the skip.

## 3. The invalid_action episode (expcorr_atlas_deploy_official_vs_forum / inline_evidence)

Not a JSON syntax break. The call parsed as strict JSON and failed
**semantic answer validation**: `AgentEpisodeError: Duplicate claim key`
(`contracts/agent_episode.py:749`, raised by `validate_answer` after the
`Invalid strict JSON` branch). One model call, step 0, 64 output tokens,
`retrieval_attempted=False`, `presented_record_ids=[]`, `finish_reason=None`.
The model attempted an immediate inline final containing two claims under the
same key — consistent with the verbose-key idiom drift seen in
stable_incident_absent_resolution (duplicate verbose keys collide where
canonical claim_keys would not). v5 behavior on the same episode:
retrieve-first, valid_cited_final with canonical keys (`atlas_deploy_status`,
`forum_deploy_status`). Raw completion text is not retained in the record
(only `request_digest`); the error type and token count are the retained
evidence.

## 4. Boundary context: C1 re-check (qualified M9 adapter, 2026-11-10)

**Confirmed — this is a known behavioral boundary, and v6 reverted toward it.**
C1 (`57e97d6d…`, profile `qwen3_5-9b-sft-v1-bf16-action-json-v1`) skipped
retrieval in exactly 3 inline_evidence episodes, all unsupported_final:

- conflicting_independent_sources_status
- expcorr_atlas_deploy_official_vs_forum
- stable_incident_absent_resolution

v6 skipped retrieval on 2 of those same 3 and terminated invalid_action on the
third. The m9b qualification run itself (the run the lock is built on) shows
the identical 3-episode skip pattern. v5 is the outlier: it had *suppressed*
this boundary (12/12 inline retrieve-first). So the v5→v6 delta on these
cells is a **boundary flip back toward the qualified-adapter baseline**,
concentrated on exactly the historically unstable episodes.

## 5. Gate-6 definition: intended vs scored, with corrected scorecards

**Intended (M9b charter, docs/m9b-gate-candidate2-2026-11-09.md):**
"retrieve-first on answer-present ≥ 0.9" measured **12/12 — the 12 `none`-arm
episodes** (store populated, evidence NOT embedded in the prompt).
control.py's own design concurs: `inline_evidence` is a *no-need*
construction (`retrieval_needless = control != "none"`); "retrieval in either
no-need construction … is surface-cued retrieval per H2b and is **reported
separately**". Retrieve-first was never the chartered requirement on
inline_evidence cells; the gate measures whether the model consults the store
before answering when the prompt does not hand it the events.

**As scored by the M10 v5/v6 scorer:** denominator 4 =
`answer_present` cell × {`none`, `inline_evidence`}. This reconstruction
exactly reproduces both recorded scorecards (v5 4/4; v6 3/4 with the named
miss). It mis-scopes gate 6 twice over: it includes 2 episodes from a
construction the diagnostic designates needless-retrieval, and it shrinks the
denominator from 12 to 4, making one boundary episode decisive.

**Proof of mis-scope:** retro-applying the 4-episode definition to the m9b
qualification run and to the C1 re-check gives **3/4 (0.75) for both** — the
qualified, locked adapter fails the gate it was locked under. The 4-episode
denominator cannot be the chartered definition.

### Corrected scorecards

| run | gate 6 as scored (4 eps) | gate 6 intended (12 none-arm eps) |
|---|---|---|
| v5 | 4/4 = 1.00 PASS (recorded) | **12/12 PASS** |
| v6 | 3/4 = 0.75 FAIL (recorded) | **12/12 PASS** |
| m9b qualification | (would be 3/4 FAIL) | 12/12 PASS (as locked) |
| C1 re-check | (would be 3/4 FAIL) | 12/12 PASS |

v6 corrected scorecard under the intended definition: gates 1–5 as recorded
(valid 8/36 — **exactly at threshold, no margin**; abstention 12/12; phantom 0;
injection 0; loops 0), gate 6 12/12 → **PASS 6/6**, with aux flags retained
honestly: 1 non-completed episode (invalid_action), inline_evidence
retrieve-first 9/12 (H2b axis, reported separately per the diagnostic design).

## 6. v6.1 remediation spec

The regression is real even though the gate mis-scoped it: answering from
untrusted embedded text and citing inline numbering yields unsupported_final —
a genuine quality loss vs v5's retrieve-and-cite behavior.

1. **Scorer repair (primary, do first).** Restore gate 6 to the chartered
   denominator: `none`-arm answer-present episodes only (12). Exclude
   `retrieval_needless` constructions from gate 6; report inline_evidence
   retrieve-first and unsupported_final counts as separate aux metrics (H2b /
   trust-boundary tracking), with any threshold predeclared before the next
   gate. Pin the corrected scorer against the m9b qualification control.jsonl
   (must reproduce 12/12) as a regression test.
2. **Curriculum v6.1 (additive, byte-pin v3–v6 classes):** a small
   `inline_evidence_trust_boundary` class (~12–16 eps): query embeds untrusted
   inline event text while the store is populated; gold = retrieve-first,
   then final citing the RETRIEVED record ids under their canonical
   claim_keys. Near-misses: answering directly from the embedded text, and
   citing inline numbering. This re-pins the v5 suppression behavior on the
   exact boundary shape (3 historical episodes).
3. **Duplicate-claim-key near-miss.** In multi-claim golds, include an
   uncited distractor record that would invite a second claim under an
   already-used key; the invalid_action episode shows the compose idiom
   producing colliding verbose keys under embedded-evidence load.
4. **Do NOT touch `answer_supported`.** The hypothesis that it taught
   answering-from-embedded-evidence is denied (§2); its bytes are pinned and
   its golds are retrieve-first. Also investigate (non-blocking) whether
   `dependent_lookup_compose`'s keying-instruction-omitted episodes
   contributed to the verbose-key drift observed in both v6 skips.
5. **Watch item, not a gate change:** gate 1 sits at exactly 8/36 in v6 (the
   invalid_action and the two over-abstentions each cost a valid). Any v6.1
   rerun should treat gate-1 margin as fragile.

## Non-claims

- No rerun performed; this is analysis of retained evidence only.
- The corrected PASS 6/6 is a re-scoring under the chartered definition, not a
  new measurement; the v6 adapter remains a development checkpoint, not a
  qualified one. The inline_evidence regression stands and gates nothing until
  the v6.1 spec above is executed and re-measured.
