"""Behavioral checks for the M8 stratified paired trinomial inference.

These tests pin numerical behavior (boundary cells, symmetry, monotonicity,
determinism, weight handling, scaling) and characterize the gap to exact
Clopper-Pearson enumeration for small single-cell samples. They are not a
coverage calibration; null coverage at the declared margins must be
established by separate simulation before acceptance use.
"""

from __future__ import annotations

import math
import time
from math import comb

import pytest

from daystrom_dml.services.m8_statistics import (
    likelihood_ratio,
    paired_trinomial_inference,
    profile_loglikelihood,
)

THR_05 = 1.6448536269514722 ** 2  # NormalDist().inv_cdf(0.95) ** 2


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0], [1, 2], [0, 0], [0, 0])


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        paired_trinomial_inference([], [], [], [])


def test_rejects_negative_count():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0], [-1], [0], [2])


def test_rejects_non_integer_count():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0], [1.5], [0], [2])  # type: ignore[list-item]


def test_rejects_zero_weight():
    with pytest.raises(ValueError):
        paired_trinomial_inference([0.0], [1], [0], [2])


def test_rejects_negative_weight():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0, -2.0], [1, 0], [0, 1], [2, 0])


def test_rejects_nan_weight():
    with pytest.raises(ValueError):
        paired_trinomial_inference([float("nan")], [1], [0], [2])


def test_rejects_infinite_weight():
    with pytest.raises(ValueError):
        paired_trinomial_inference([float("inf")], [1], [0], [2])


def test_rejects_zero_total_cell():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0, 1.0], [1, 0], [0, 0], [2, 0])


def test_rejects_bad_alpha():
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0], [1], [0], [2], alpha=0.0)
    with pytest.raises(ValueError):
        paired_trinomial_inference([1.0], [1], [0], [2], alpha=0.5)


def test_all_tie_cell_matches_closed_form():
    # Single all-tie cell: profile penalty is n * log(1 - |delta|) exactly, so
    # the bound solves -2 * n * log(1 - |delta|) = threshold.
    n = 20
    result = paired_trinomial_inference([1.0], [0], [0], [n])
    expected = -(1.0 - math.exp(-result.threshold / (2.0 * n)))
    assert result.delta_hat == 0.0
    assert result.lower_bound == pytest.approx(expected, abs=1e-12)
    assert -1.0 < result.lower_bound < 0.0 < result.upper_bound
    assert result.upper_bound == pytest.approx(-expected, abs=1e-12)


def test_all_win_cell_matches_closed_form():
    # Single all-win cell: q_hat = 1 binds, so the bound solves
    # -2 * n * log((1 + delta) / 2) = threshold.
    n = 20
    result = paired_trinomial_inference([1.0], [n], [0], [0])
    expected = 2.0 * math.exp(-result.threshold / (2.0 * n)) - 1.0
    assert result.delta_hat == 1.0
    assert result.lower_bound == pytest.approx(expected, abs=1e-12)
    assert result.upper_bound == 1.0
    assert -1.0 < result.lower_bound < 1.0


def test_all_loss_cell_hits_feasible_floor():
    result = paired_trinomial_inference([1.0], [0], [20], [0])
    assert result.delta_hat == -1.0
    assert result.lower_bound == -1.0
    assert result.upper_bound == pytest.approx(
        -(2.0 * math.exp(-THR_05 / 40.0) - 1.0), abs=1e-12
    )


def test_swap_symmetry_flips_sign():
    weights = [0.5, 0.3, 0.2]
    wins, losses, ties = [7, 3, 0], [2, 5, 12], [1, 2, 8]
    fwd = paired_trinomial_inference(weights, wins, losses, ties)
    rev = paired_trinomial_inference(weights, losses, wins, ties)
    assert fwd.delta_hat == -rev.delta_hat
    # The one-sided lower bound mirrors into the upper bound of the swapped
    # problem; agreement is to bisection resolution, not bit-exact.
    assert fwd.lower_bound == pytest.approx(-rev.upper_bound, abs=1e-9)
    assert fwd.upper_bound == pytest.approx(-rev.lower_bound, abs=1e-9)
    assert fwd.loglik_mle == rev.loglik_mle


def test_monotonicity_in_data():
    base = paired_trinomial_inference([1.0], [5], [3], [4])
    more_wins = paired_trinomial_inference([1.0], [7], [3], [2])
    more_losses = paired_trinomial_inference([1.0], [5], [5], [2])
    assert more_wins.lower_bound > base.lower_bound > more_losses.lower_bound
    assert more_wins.delta_hat > base.delta_hat > more_losses.delta_hat


def test_single_cell_against_binomial_enumeration():
    # Enumeration check: single tie-free cell vs the exact Clopper-Pearson
    # one-sided bound on delta = 2p - 1, computed by summing binomial tails.
    # The Wilks-type profile bound must be monotone in the observed wins and
    # stay within a characterized band of the exact bound; the residual gap at
    # n = 12 quantifies the pending coverage calibration.
    n = 12

    def exact_lower(w: int) -> float:
        if w == 0:
            return -1.0

        def tail(p: float) -> float:
            return sum(comb(n, k) * p ** k * (1.0 - p) ** (n - k) for k in range(w, n + 1))

        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if tail(mid) < 0.05:
                lo = mid
            else:
                hi = mid
        return 2.0 * (0.5 * (lo + hi)) - 1.0

    previous = -1.0
    max_gap = 0.0
    for w in range(0, n + 1, 3):
        result = paired_trinomial_inference([1.0], [w], [n - w], [0])
        exact = exact_lower(w)
        assert -1.0 <= result.lower_bound <= result.delta_hat
        assert result.lower_bound >= previous - 1e-12  # enumerated monotonicity
        previous = result.lower_bound
        max_gap = max(max_gap, abs(result.lower_bound - exact))
    assert max_gap < 0.25  # characterization bound, not a coverage claim


def test_determinism_bit_for_bit():
    args = ([0.5, 0.3, 0.2], [7, 3, 1], [2, 5, 1], [1, 2, 8])
    first = paired_trinomial_inference(*args)
    second = paired_trinomial_inference(*args)
    assert first == second
    assert first.lower_bound == second.lower_bound  # exact float equality


def test_weight_sensitivity_and_normalization():
    wins, losses, ties = [8, 2], [2, 8], [0, 0]
    toward_winner = paired_trinomial_inference([0.9, 0.1], wins, losses, ties)
    toward_loser = paired_trinomial_inference([0.1, 0.9], wins, losses, ties)
    assert toward_winner.delta_hat > 0.0 > toward_loser.delta_hat
    assert toward_winner.lower_bound > toward_loser.lower_bound
    # Weights are normalized: positive rescaling must not change anything.
    scaled = paired_trinomial_inference([9.0, 1.0], wins, losses, ties)
    assert scaled == toward_winner


def test_point_estimate_is_weighted_cell_effects():
    result = paired_trinomial_inference([2.0, 1.0, 1.0], [6, 0, 0], [2, 0, 6], [0, 4, 4])
    expected = 0.5 * 0.5 + 0.25 * 0.0 + 0.25 * (-0.6)
    assert result.delta_hat == pytest.approx(expected)
    assert result.weights == pytest.approx((0.5, 0.25, 0.25))
    assert result.cell_count == 3
    assert result.total_observations == 22


def test_alpha_widens_bound():
    args = ([0.6, 0.4], [30, 5], [20, 4], [0, 1])
    loose = paired_trinomial_inference(*args, alpha=0.10)
    strict = paired_trinomial_inference(*args, alpha=0.01)
    assert loose.lower_bound > strict.lower_bound


def test_likelihood_ratio_consistency_at_bound():
    weights, wins, losses, ties = [0.5, 0.5], [8, 0], [2, 0], [0, 10]
    result = paired_trinomial_inference(weights, wins, losses, ties)
    assert likelihood_ratio(result.delta_hat, weights, wins, losses, ties) == pytest.approx(0.0, abs=1e-9)
    # The reported lower bound is the last rejected point: its statistic sits
    # at or just above the threshold, and the bound rounds conservatively.
    assert likelihood_ratio(result.lower_bound, weights, wins, losses, ties) >= result.threshold - 1e-6
    assert result.lower_bound <= result.delta_hat <= result.upper_bound
    assert result.loglik_at_lower_bound <= result.loglik_mle
    assert profile_loglikelihood(result.lower_bound, weights, wins, losses, ties) == pytest.approx(
        result.loglik_at_lower_bound
    )


def test_216_cell_synthetic_case_time():
    # Fixed arithmetic pattern, no RNG: 216 cells shaped like the manifest's
    # stratified scenario grid.
    weights = [1.0 + (i % 7) for i in range(216)]
    wins = [(i * 5 + 3) % 11 for i in range(216)]
    losses = [(i * 7 + 1) % 9 for i in range(216)]
    ties = [(i * 3 + 2) % 5 for i in range(216)]
    started = time.perf_counter()
    result = paired_trinomial_inference(weights, wins, losses, ties)
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0
    assert -1.0 <= result.lower_bound <= result.delta_hat <= result.upper_bound <= 1.0
    assert result.cell_count == 216
    assert result.total_observations == sum(wins) + sum(losses) + sum(ties)
