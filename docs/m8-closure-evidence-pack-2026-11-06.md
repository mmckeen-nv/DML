# M8 closure evidence pack — November 6, 2026

**Status: closure-READY for owner decision. M8 cannot close as a pass (the trained policy's correctness is far below gate thresholds); it can close as a fully-evidenced negative result with a specified unblock, or stay open pending the training decision.** All contracts remain proposals pending independent review.

## What M8 set out to prove, and what the evidence says

The milestone's core question: does durable DML memory help the qualified Llama3-8B-SFT-v3 policy under a fair comparison? The evidence chain, every step retained:

1. **The policy's retrieval is surface-conditioned, not need-driven** (H2): 0/72 retrieval under the neutral protocol vs 36/36 retrieve-first under the trained DML surface — including 24/24 episodes where retrieval was needless. The neutral-protocol fairness design is falsified by construction; the transport-native redesign is drafted ([amendment](m8-transport-native-fairness-2026-11-05.md), review pending).
2. **The policy's finalization was broken and is now partially repaired:** retrieval loops eliminated (13→0-2 per 36), calibrated abstention learned at the prompt level (0→8/12 at v4), fixture authoring debt repaired (keys now correct), value-unit forms repaired (v5).
3. **The residual failure is trained-in parametric behavior:** partial coverage of multi-part questions (16/19 unsupported episodes at v5), conflict-source splitting, injection-following (the adversarial family working as designed — it caught real compliance), rare phantom citations. Prompt rules at five iterations now displace each other (v5 fixed units, regressed abstention) — the documented ceiling.
4. **The machinery is complete and verified:** strict verifier never credited a fabrication in 216 retained episodes; fixtures hygiene-gated in CI; runner 2.5× faster; x86 runtime primary (station retired); adapter verified live on x86; calibration nominal at smoke scale; CI green at tip.

## Gate-by-gate status

| Gate | Status | Evidence |
|---|---|---|
| Fairness blockers 1&2 (event-clock, correction parity) | Implemented, proposed-contract | `docs/m8-review-packet-2026-11-02.md` |
| Protocol competence (neutral) | **Failed by policy behavior** (0/72 retrieval); framework falsified | battery doc + control doc |
| Transport-native redesign | Proposed, review pending | `docs/m8-transport-native-fairness-2026-11-05.md` |
| Control diagnostic (H1/H2) | **Done** — H2 supported, H1 dead | `docs/m8-policy-control-2026-11-05.md` |
| Prompt-amendment program | **Done, converged at v4** | same doc (5 iterations, 216 episodes) |
| Fixture integrity | Repaired + CI-gated | hygiene tests, this pack |
| Verifier strictness | Tightened (word-boundary) | suite green |
| Calibration machinery | Built; smoke nominal; station grids pending | `docs/m8-calibration-plan-2026-11-02.md` |
| Adapter correctness | Verified live on x86 | probe report (scratch) |
| Campaign (216-dev, 4,320 held-out) | **Not run** — policy below gate threshold | this pack |
| Independent review | **Pending** — all contracts proposals | review packet |

## The unblock (owner decision)

A bounded training milestone (M9) teaching: abstention shape discipline, multi-part coverage, conflict-source splitting, injection resistance. Its specification is this program's retained evidence: 216 control episodes across 6 prompt surfaces with per-episode failure classification, plus the 108-episode battery. Nothing else stands between the current state and a campaign: once the policy passes the control diagnostic at a declared threshold, the transport-native runner, calibration, and evidence machinery are ready to execute.

## Pointers

- Policy-control record (all runs, hashes, mechanics): `docs/m8-policy-control-2026-11-05.md`
- Transport-native fairness amendment: `docs/m8-transport-native-fairness-2026-11-05.md`
- Review packet: `docs/m8-review-packet-2026-11-02.md`
- Calibration plan: `docs/m8-calibration-plan-2026-11-02.md`
- Raw evidence: `/home/test/dml-debug/m8-control-*2026110*/` on x86-dev (station evidence mirrored intact)
