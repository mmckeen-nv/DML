"""Doubles-level tests for the M8 calibration machinery.

No model calls, no large simulations: every grid here is smoke-scale
(seconds). These tests pin determinism, estimator correctness against exact
small-N enumeration, boundary-sweep execution, and output schema stability.
They are not themselves a coverage calibration; the calibration protocol is
docs/m8-calibration-plan-2026-11-02.md.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
import pytest

from daystrom_dml.services.m8_statistics import paired_trinomial_inference
from scripts.m8_calibration.simulation import (
    SEED_ENTROPY,
    ArmMetricConfig,
    BootstrapCellConfig,
    CellConfig,
    all_tie_cell,
    bootstrap_gate_simulation,
    boundary_structures,
    cached_bound_evaluations,
    cell_generators,
    exact_single_cell_coverage,
    rare_discordance_cell,
    run_boundary_sweep,
    run_coverage_grid,
    simulate_trinomial_counts,
    trinomial_coverage,
    true_delta,
    uniform_structure,
)

ALPHA = 0.05


def fair_coin_structure(n=8):
    """Single cell, q=1, delta=0: a fair coin with no ties."""
    return uniform_structure(1, q=1.0, delta=0.0, scenarios=n)


def test_determinism_counts_and_reports():
    configs = uniform_structure(4, q=0.3, delta=0.1, scenarios=8)
    first_counts = simulate_trinomial_counts(configs, replicates=16)
    second_counts = simulate_trinomial_counts(configs, replicates=16)
    assert first_counts.dtype == np.int64
    assert first_counts.shape == (16, 4, 3)
    assert np.array_equal(first_counts, second_counts)
    assert np.all(first_counts.sum(axis=2) == 8)

    first = trinomial_coverage(configs, structure="det", replicates=16, keep_outcomes=True)
    second = trinomial_coverage(configs, structure="det", replicates=16, keep_outcomes=True)
    assert first == second
    assert first.seed == SEED_ENTROPY
    assert first.outcomes is not None and len(first.outcomes) == 16


def test_determinism_generators_spawn_scheme():
    gens_a = cell_generators(3)
    gens_b = cell_generators(3)
    draws_a = [g.random(5) for g in gens_a]
    draws_b = [g.random(5) for g in gens_b]
    assert all(np.array_equal(a, b) for a, b in zip(draws_a, draws_b))
    # Distinct children produce distinct streams.
    assert not np.array_equal(draws_a[0], draws_a[1])


def test_true_delta_weighting():
    configs = (
        CellConfig(name="a", weight=1.0, scenarios=8, q=0.5, delta=0.2),
        CellConfig(name="b", weight=3.0, scenarios=8, q=0.5, delta=-0.1),
    )
    assert true_delta(configs) == pytest.approx((0.2 * 1.0 - 0.1 * 3.0) / 4.0)
    with pytest.raises(ValueError):
        true_delta(())


def test_cell_config_validation_and_from_probabilities():
    cfg = CellConfig.from_probabilities("p", 2.0, 8, p_win=0.35, p_loss=0.15, p_tie=0.5)
    assert cfg.q == pytest.approx(0.5)
    assert cfg.delta == pytest.approx(0.2)
    assert cfg.p_win == pytest.approx(0.35)
    assert cfg.p_loss == pytest.approx(0.15)
    assert cfg.p_tie == pytest.approx(0.5)
    with pytest.raises(ValueError):
        CellConfig.from_probabilities("bad", 1.0, 8, p_win=0.5, p_loss=0.4, p_tie=0.2)
    with pytest.raises(ValueError):
        CellConfig(name="x", weight=1.0, scenarios=8, q=0.2, delta=0.3)  # |delta| > q
    with pytest.raises(ValueError):
        CellConfig(name="x", weight=0.0, scenarios=8, q=0.2, delta=0.1)
    with pytest.raises(ValueError):
        CellConfig(name="x", weight=1.0, scenarios=0, q=0.2, delta=0.1)


def test_rare_discordance_factory():
    cfg = rare_discordance_cell("r", scenarios=8)
    assert cfg.q == pytest.approx(1.0 / 8)
    assert cfg.delta == pytest.approx(cfg.q)  # all discordance is wins
    lossy = rare_discordance_cell("r", scenarios=8, win_share=0.0)
    assert lossy.delta == pytest.approx(-lossy.q)
    with pytest.raises(ValueError):
        rare_discordance_cell("r", scenarios=8, win_share=1.5)


def test_exact_single_cell_coverage_matches_brute_force():
    """Analytically-known case: single fair-coin cell, n=8.

    exact_single_cell_coverage must agree with an independent brute-force
    enumeration written here against the service directly.
    """
    n = 8
    exact = exact_single_cell_coverage(n, 1.0, 0.0, alpha=ALPHA)
    brute = 0.0
    for w in range(n + 1):
        lo = n - w
        result = paired_trinomial_inference([1.0], [w], [lo], [0], alpha=ALPHA)
        if result.lower_bound <= 0.0:
            brute += comb(n, w) / 2**n
    assert exact == pytest.approx(brute, abs=1e-12)
    # One-sided 95% bound on a fair coin: conservative but not degenerate.
    assert 0.9 <= exact <= 1.0


def test_exact_single_cell_coverage_with_ties():
    # q < 1 exercises the tie branch of the enumeration (0 * log 0 handling).
    exact = exact_single_cell_coverage(6, 0.5, 0.1, alpha=ALPHA)
    assert 0.0 <= exact <= 1.0
    # True delta above the MLE support should still be covered often.
    high = exact_single_cell_coverage(6, 0.5, 0.4, alpha=ALPHA)
    assert high >= exact


def test_empirical_coverage_consistent_with_exact():
    """Monte Carlo coverage estimate must land within sampling error of the
    exact enumeration on the fair-coin cell (deterministic seed)."""
    configs = fair_coin_structure(n=8)
    exact = exact_single_cell_coverage(8, 1.0, 0.0, alpha=ALPHA)
    replicates = 1000
    report = trinomial_coverage(configs, structure="fair", replicates=replicates)
    mc_se = math.sqrt(exact * (1.0 - exact) / replicates)
    assert abs(report.coverage - exact) <= 6.0 * mc_se + 1e-9
    assert report.true_delta == pytest.approx(0.0)
    assert report.cell_count == 1
    assert report.scenarios_per_replicate == 8
    # Fair coin, no ties: every draw is all-win, all-loss, or mixed discordant;
    # never all-tie.
    assert report.all_tie_fraction == pytest.approx(0.0)
    assert 0.0 < report.all_win_fraction < 1.0
    assert 0.0 < report.all_loss_fraction < 1.0


def test_boundary_sweep_executes_and_saturates():
    reports = run_boundary_sweep(cell_count=6, scenarios=8, replicates=32)
    assert [r.structure for r in reports] == [
        "all_tie",
        "all_win",
        "all_loss",
        "rare_discordance",
        "mixed_boundary",
    ]
    by_name = {r.structure: r for r in reports}
    # Pure structures saturate their boundary in every (replicate, cell) draw.
    assert by_name["all_tie"].all_tie_fraction == pytest.approx(1.0)
    assert by_name["all_win"].all_win_fraction == pytest.approx(1.0)
    assert by_name["all_loss"].all_loss_fraction == pytest.approx(1.0)
    # Rare discordance: all-tie probability per cell draw is (1-q)^scenarios
    # = (7/8)^8 ~= 0.3436; assert within 4 binomial SE of theory (192 draws).
    rare = by_name["rare_discordance"]
    theory = (7.0 / 8.0) ** 8
    se = math.sqrt(theory * (1.0 - theory) / (32 * 6))
    assert rare.all_tie_fraction == pytest.approx(theory, abs=4.0 * se)
    assert rare.all_loss_fraction == pytest.approx(0.0)
    # Every report: finite bounds, sane coverage, all-loss hits the floor.
    for report in reports:
        assert 0.0 <= report.coverage <= 1.0
        assert math.isfinite(report.width_mean)
        assert 0.0 <= report.width_min <= report.width_max
        assert report.width_mean > 0.0  # bounds never collapse to the point estimate
    all_loss = by_name["all_loss"]
    assert all_loss.floor_fraction == pytest.approx(1.0)
    assert all_loss.coverage == pytest.approx(1.0)  # floor -1 covers true delta -1
    # Boundary structures never error and always produce finite widths.
    mixed = by_name["mixed_boundary"]
    assert math.isfinite(mixed.width_median)


def test_boundary_structures_validation():
    with pytest.raises(ValueError):
        boundary_structures(cell_count=3)


def test_coverage_report_schema_stability():
    configs = uniform_structure(2, q=0.3, delta=0.1, scenarios=8)
    report = trinomial_coverage(configs, structure="schema", replicates=8)
    assert set(report.to_dict()) == {
        "structure",
        "replicates",
        "alpha",
        "boundary_rule",
        "true_delta",
        "coverage",
        "cell_count",
        "scenarios_per_replicate",
        "seed",
        "width",
        "floor_fraction",
        "ceil_fraction",
        "boundary",
    }
    assert report.to_dict()["boundary_rule"] == "wilks"
    assert set(report.to_dict()["width"]) == {"mean", "median", "min", "max"}
    assert set(report.to_dict()["boundary"]) == {"all_tie", "all_win", "all_loss"}
    assert report.to_dict()["seed"] == [20260929]


def bootstrap_configs(ratio=1.0):
    """Two cells, two arms, ratio-scaled DML metrics vs RAG."""

    def cell(name, condition):
        return BootstrapCellConfig(
            name=name,
            weight=1.0,
            scenarios=6,
            cache_condition=condition,
            arms={
                "dml": ArmMetricConfig(
                    latency_ms=10.0 * ratio,
                    lifecycle_ms=10.0 * ratio,
                    tokens=int(round(100 * ratio)),
                    success_rate=1.0,  # deterministic successes: ratio tests isolate metric ratios
                ),
                "rag": ArmMetricConfig(latency_ms=10.0, lifecycle_ms=10.0, tokens=100, success_rate=1.0),
            },
        )

    return (cell("c0", "cold"), cell("c1", "warm"))


def test_bootstrap_gate_simulation_determinism_and_gates():
    configs = bootstrap_configs(ratio=1.0)
    first = bootstrap_gate_simulation(configs, structure="boot", datasets=2, resamples=100)
    second = bootstrap_gate_simulation(configs, structure="boot", datasets=2, resamples=100)
    assert first == second
    gate_names = [g.gate for g in first.gates]
    assert gate_names == ["lifecycle_ratio:cold", "lifecycle_ratio:warm", "tokens_ratio"]
    # True ratio 1.0 with thresholds 1.25/1.10: gates should pass every dataset.
    for gate in first.gates:
        assert gate.pass_fraction == pytest.approx(1.0)
        assert gate.blocked_fraction == pytest.approx(0.0)
        assert gate.mean_upper_bound is not None
        assert 0.9 < gate.mean_upper_bound < 1.1


def test_bootstrap_gate_simulation_failing_ratio():
    configs = bootstrap_configs(ratio=2.0)
    report = bootstrap_gate_simulation(configs, structure="boot-bad", datasets=2, resamples=100)
    for gate in report.gates:
        assert gate.pass_fraction == pytest.approx(0.0)
        assert gate.blocked_fraction == pytest.approx(0.0)  # fails on bound, not blocking
        assert gate.mean_upper_bound is not None and gate.mean_upper_bound > 1.5


def test_bootstrap_gate_report_schema_stability():
    report = bootstrap_gate_simulation(bootstrap_configs(), structure="schema", datasets=1, resamples=100)
    assert set(report.to_dict()) == {
        "structure",
        "seed",
        "datasets",
        "resamples",
        "dml_arm",
        "rag_arm",
        "gates",
        "conjunction_pass_fraction",
    }
    gate_dicts = report.to_dict()["gates"]
    assert isinstance(gate_dicts, list) and len(gate_dicts) == 3
    for gate_dict in gate_dicts:
        assert set(gate_dict) == {
            "gate",
            "threshold",
            "datasets",
            "pass_fraction",
            "blocked_fraction",
            "mean_upper_bound",
            "upper_bounds",
        }


def test_bootstrap_gate_simulation_validation():
    configs = bootstrap_configs()
    with pytest.raises(ValueError):
        bootstrap_gate_simulation(configs, datasets=0, resamples=100)
    with pytest.raises(ValueError):
        bootstrap_gate_simulation(configs, datasets=1, resamples=50)
    with pytest.raises(ValueError):
        bootstrap_gate_simulation(configs, datasets=1, resamples=100, seed=[])
    mismatched = BootstrapCellConfig(
        name="m",
        weight=1.0,
        scenarios=4,
        cache_condition="cold",
        arms={
            "dml": ArmMetricConfig(latency_ms=1.0, lifecycle_ms=1.0, tokens=10, success_rate=0.5),
        },
    )
    with pytest.raises(ValueError):
        bootstrap_gate_simulation((mismatched,), datasets=1, resamples=100)


def test_all_tie_all_win_all_loss_factories():
    tie = all_tie_cell("t")
    win = rare_discordance_cell("w")  # ensure no name confusion; checked elsewhere
    assert tie.q == 0.0 and tie.delta == 0.0 and tie.kind == "all_tie"
    assert win.kind == "rare_discordance"
    configs = uniform_structure(3, q=1.0, delta=1.0, scenarios=4, kind="all_win")
    report = trinomial_coverage(configs, structure="wins", replicates=8)
    assert report.all_win_fraction == pytest.approx(1.0)
    assert report.coverage == pytest.approx(1.0)  # true delta 1 is always covered


# --- Conservative boundary rule wiring ('wilks_plus_inflation') ---


def test_cached_bound_evaluations_cache_key_includes_boundary_rule():
    request = ([1.0, 1.0], (3, 1), (1, 2), (4, 5))
    wilks_first = cached_bound_evaluations([request])
    inflated = cached_bound_evaluations([request], boundary_rule="wilks_plus_inflation")
    wilks_second = cached_bound_evaluations([request])
    # Same counts under the same rule hit the cache: identical values.
    assert wilks_first == wilks_second
    # The rule is part of the key: the inflated entry is a distinct,
    # wider-or-equal bound, not the memoized wilks value.
    (w_hat, w_lower, w_upper) = wilks_first[0]
    (i_hat, i_lower, i_upper) = inflated[0]
    assert i_hat == w_hat
    assert i_lower <= w_lower + 1e-12
    assert i_upper >= w_upper - 1e-12
    with pytest.raises(ValueError):
        cached_bound_evaluations([request], boundary_rule="clopper_pearson")


def test_coverage_grid_rejects_unknown_boundary_rule():
    configs = uniform_structure(2, q=0.3, delta=0.05, scenarios=8)
    with pytest.raises(ValueError):
        trinomial_coverage(configs, structure="bad-rule", replicates=2, boundary_rule="clopper_pearson")
    with pytest.raises(ValueError):
        exact_single_cell_coverage(4, 0.3, 0.05, boundary_rule="clopper_pearson")
    with pytest.raises(ValueError):
        run_boundary_sweep(cell_count=4, replicates=2, boundary_rule="clopper_pearson")


def test_inflated_rule_coverage_not_below_wilks_small_grid():
    """Tiny seeded margins grid: the inflated rule must never undercut wilks.

    Both runs share the seed and structures, so every replicate draws the
    SAME counts under both rules; per-replicate dominance (inflated lower
    bound <= wilks lower bound) then implies coverage ordering exactly, with
    no Monte Carlo ambiguity. This is machinery capability evidence, not a
    station re-measurement.
    """
    structures = [
        ("cells6-q0.15-d0.05", uniform_structure(6, q=0.15, delta=0.05, scenarios=8)),
        ("cells6-q0.3-d-0.02", uniform_structure(6, q=0.3, delta=-0.02, scenarios=8)),
    ]
    wilks_reports = run_coverage_grid(structures, replicates=12, keep_outcomes=True)
    inflated_reports = run_coverage_grid(
        structures, replicates=12, boundary_rule="wilks_plus_inflation", keep_outcomes=True
    )
    strict_widening_seen = False
    for wilks, inflated in zip(wilks_reports, inflated_reports):
        assert wilks.structure == inflated.structure
        assert wilks.boundary_rule == "wilks"
        assert inflated.boundary_rule == "wilks_plus_inflation"
        assert wilks.seed == inflated.seed == SEED_ENTROPY
        wilks_outcomes = wilks.outcomes or ()
        inflated_outcomes = inflated.outcomes or ()
        assert len(wilks_outcomes) == len(inflated_outcomes) == 12
        for w_out, i_out in zip(wilks_outcomes, inflated_outcomes):
            # Identical draws: the point estimate is rule-independent.
            assert i_out.delta_hat == w_out.delta_hat
            assert i_out.lower_bound <= w_out.lower_bound + 1e-12
            assert i_out.upper_bound >= w_out.upper_bound - 1e-12
            if i_out.lower_bound < w_out.lower_bound - 1e-9:
                strict_widening_seen = True
            if w_out.covers:
                assert i_out.covers  # per-replicate coverage dominance
        assert inflated.coverage >= wilks.coverage
        assert inflated.width_mean >= wilks.width_mean - 1e-12
    assert strict_widening_seen  # the inflation moved at least one bound
