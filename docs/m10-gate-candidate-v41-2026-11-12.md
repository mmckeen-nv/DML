# M10 gate result — candidate v4.1 (2026-11-12)

**Verdict: FAIL — quality bar not met. Floors intact. Evidence retained; nothing re-run or tuned.**

## Chain state before the campaign

- Curriculum v4.1 (seed 20261116, stem-purged pools, verifier-credited golds): merged `c2147a2`
- Training: adapter `df60527011c10610…`, loss 0.047→0.023, PEFT_LOAD_OK
- Frozen diagnostic: PASS 6/6 (valid 12/36, up from M9's 8/36)
- Leakage re-audit: PASS (`m10-leakage-v41-20261112.json`)
- Admission both-ways verified; candidate manifest `96df7272…`

## Campaign (heldout2, sealed `7016a781…`, 576 episodes, zero harness failures)

| arm | valid | abstain | unsupported |
|---|---|---|---|
| dml_memory | **59/144 (41.0%)** | 15 | 70 |
| rag_memory | 62/144 (43.1%) | — | — |
| context_inline | 0 (79.9% abstain) | — | — |
| no_memory | 0 (144/144 abstain) | — | — |

Paired trinomial (inflated rule, α=0.05, family-stratified): wins 4 / losses 7 / ties 133 → δ̂ = **−0.0208**, lower bound **−0.0659**. First-query identity 144/144.

## Decision rule

1. valid ≥ 60% — **FAIL (41.0%)**
2. paired lower bound ≥ 0 — **FAIL (−0.0659)**
3. floors — PASS

## Failure signals for the next tranche (retained evidence)

- `dependent_lookups`: 0/48 valid on BOTH arms — total-miss class (multi-hop decomposition; investigate winnability AND capability)
- `stable_memory_adverse`: 0 wins / 5 losses vs rag — injection/adverse resistance weaker than rag's on this set
- Trajectory: 21.5% (M9 adapter, partially-unwinnable set) → 41.0% (v4.1, winnable set). The bar stands.

Evidence: `/home/test/dml-debug/m10-campaign-v41-20261112/` (sha256-pinned, `sha256sum -c` verified) incl. `m10_gate_statistics.json`, `score_m10_gate.py`.
