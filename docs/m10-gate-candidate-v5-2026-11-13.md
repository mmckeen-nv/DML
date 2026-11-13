# M10 gate result — candidate v5 (2026-11-13, fixed heldout2)

**Verdict: FAIL on clause 2 (paired bound). Clause 1 (quality bar) PASSED. Floors PASS. Evidence retained; no re-run.**

## Numbers (sealed heldout2 `15ed5697…`, 576 episodes, zero harness failures)

| clause | number | verdict |
|---|---|---|
| dml_memory valid ≥ 60% | **92/144 = 63.9%** | **PASS** |
| paired lower bound ≥ 0 (inflated rule) | wins 4 / losses 4 / ties 136; δ̂=0.0, lower **−0.0404** | **FAIL** |
| floors (context_inline 0 valid/88.9% abstain; no_memory 144/144; no harness failures) | one step-limit episode (retained, dml dependent_lookups_0076) | PASS (caveat recorded) |
| first-query identity | 144/144 | PASS |

## Per-family (dml/rag valid)

changing_facts 12/12 · explicit_corrections 24/24 · conflicts 11/12 · isolated_scopes 22/24 · **dependent_lookups 0/48 both arms** · stable_adverse 23/20

## Analysis

- Trajectory: 21.5% → 41.0% → **63.9%**. The curriculum mechanism works; the user's quality objection is met (clause 1).
- The parity clause fails on a tie: 4/4 discordant pairs. **48/144 scenarios (dependent_lookups) are mutual zeros** — winnable by construction post-re-seal, unsolved by either backend: the chained-lookup capability did not transfer from the v5 `dependent_lookup_chain` class. Outside that family: dml 92/96 vs rag 91/96 — and dml leads the adverse family 23/20.
- dependent_lookups non-transfer is now THE question: did the policy attempt the second retrieve keyed on the discovered token? Decomposition of the 48 episodes decides v6's target.

Evidence: `/home/test/dml-debug/m10-campaign-v5-20261113/` (sha256-pinned): campaign.jsonl, summary, scorer, statistics.
