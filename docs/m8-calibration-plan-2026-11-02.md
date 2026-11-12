# M8 statistical calibration plan — November 2, 2026

**Status: plan plus machinery; boundary rule measured at station scale, not yet admitted.** Station-scale re-measurement (2026-11-10, x86-dev, R=400 margins grid, seeds 20260929, `/home/test/dml-debug/m8-calibration-20261109/`): the default `wilks` rule measured anti-conservative (28/36 grid points below nominal 0.95); the pinned `wilks_plus_inflation` boundary rule measured **36/36 grid points at or above nominal (0.9525–0.9825)** (`margins-r400-inflated.json`) and is the measured-qualifying conservative rule. It remains a **proposal pending independent review** — no freeze is authorized and the acceptance statistics are not admitted. Machinery: `dml_core/scripts/m8_calibration/simulation.py` (deterministic harness; memoized + process-pooled bound evaluations; 16 tests in `dml_core/tests/test_m8_calibration.py`, ~1m45s), over `m8_statistics.py` (stratified paired trinomial profile-likelihood) and `m8_bootstrap.py` (paired cell-stratified gates). All simulations use `SeedSequence([20260929])` spawning; identical inputs are bit-for-bit reproducible.

## What must be qualified before any freeze

1. **Null coverage at the declared margins.** The one-sided 95% lower bound must cover true Delta at rate ≥95% when the true weighted Delta equals each margin (+0.05 DML−RAG gate, −0.02 DML−context gate) — across cell counts 6…216 and discordance levels (q ∈ {0.15, 0.3, 0.6}). Wilks asymptotics are unproven under boundary-saturated cells; if measured coverage is materially below nominal, a conservative boundary rule must be pinned (e.g., width inflation or exact-enumeration substitution for small-N cells) and re-measured.
2. **Boundary behavior.** all-tie/all-win/all-loss and rare-discordance structures: finite nondegenerate bounds, all-loss floor −1, no zero-width intervals, swapping symmetry, monotonicity in added wins/losses (unit-pinned; sweep re-verified at grid scale).
3. **Power at the acceptance margins.** For the +0.05 gate: fraction of replicates whose lower bound clears +0.05 at true deltas {0.05, 0.10, 0.20} per cell count — the manifest's planning target is ~95% at +0.10 with 4,320 pairs; the 216-cell development shape is measured separately. Full-conjunction power simulation (all acceptance gates jointly) follows after the correctness gates, using the bootstrap gate harness for cost/latency components.
4. **Bootstrap tail coverage.** Paired cell-stratified resample gates (lifecycle p95 ratio ≤1.25 per cache condition, tokens-per-correct ratio ≤1.10 pooled): one-sided bound coverage at true ratios {1.0, 1.05, 1.25}, blocked-gate behavior under zero-denominator/unknown-cost, and quantile-method sensitivity.

## Execution order

1. Local smoke (this repo, <60s): `python -m scripts.m8_calibration.simulation --preset smoke` — exact single-cell enumeration at the margins, re-verified in CI-adjacent runs.
2. Station margins grid: `--preset station-margins --replicates 400 --cells 6,12,24,48,96,216` (memoized; ~400 × distinct-outcome evaluations per grid point; runtime dominated by unique (wins,losses,ties) keys, not replicates).
3. Station power grid: `--preset station-power --replicates 400`.
4. Bootstrap coverage sweeps via `bootstrap_gate_simulation` at datasets ≥200, resamples ≥1000.
5. Independent review of measured coverage/power; pin any conservative boundary rules as a reviewed contract change; only then freeze.

## Caveats carried

- Monte Carlo error: coverage estimates carry `sqrt(p(1-p)/R)` error; report R and SE beside every number.
- Common random numbers: grid points sharing a cell-count prefix reuse streams on the prefix (variance reduction); cross-row comparisons are not MC-independent.
- Wilks threshold is asymptotic; boundary cells violate regularity — measured, not assumed.
- The per-call bound evaluation costs seconds; grids are sized by unique count-vectors, and station runs are expected for anything beyond smoke scale.
- Nothing here authorizes a freeze; the acceptance gates remain pending independent calibration review.

## Station-scale re-measurement (2026-11-10, x86-dev)

R=400 margins grid, seeds 20260929, `/home/test/dml-debug/m8-calibration-20261109/`: under the default `wilks` rule the one-sided lower bound measured 0.9175–0.9625 point coverage (28/36 below nominal 0.95) — mildly anti-conservative, worst at small cells. Under the pinned `wilks_plus_inflation` rule (`margins-r400-inflated.json`): **all 36/36 grid points at or above nominal (0.9525–0.9825)** — the conservative rule restores coverage at the acceptance margins, at the cost of ~20% wider bounds (power grid unchanged; power at 216 cells is 0.99 at true δ=0.10). The inflated rule is the measured-qualifying boundary rule; it remains a proposal pending independent review like every contract here.
