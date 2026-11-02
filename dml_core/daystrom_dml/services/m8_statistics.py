"""Stratified paired trinomial constrained profile-likelihood inference.

Implements the M8 fairness-manifest contract for binary correctness
differences between paired arms per scenario cell. Each cell contributes
nonnegative integer counts (wins, losses, ties) modeled as a trinomial with
per-cell parameters

    p_win  = (q_c + delta_c) / 2
    p_loss = (q_c - delta_c) / 2
    p_tie  = 1 - q_c,        abs(delta_c) <= q_c <= 1.

The estimand is Delta = sum(w_c * delta_c) over fixed, positive cell weights
(normalized here to sum to 1). No common effect is imposed across cells: every
cell keeps its own (q_c, delta_c). The reported lower bound is the smallest
null value Delta_0 not rejected by a one-sided profile-likelihood-ratio test
at level alpha, profiling all cell-specific nuisance parameters subject to
sum(w_c * delta_c) = Delta_0.

Numerical method (deterministic, numpy/stdlib only):

1. Per-cell profile over q for fixed delta is exact and closed form. The cell
   log-likelihood is strictly concave in q on (abs(delta), 1); its derivative
   numerator is a quadratic, so the maximizer is the larger quadratic root
   clamped to [abs(delta), 1]. The clamp only binds where the count of the
   vanishing probability is zero, so clamping is boundary-exact, not an
   approximation (all-tie cells give q_hat = abs(delta); all-win/all-loss
   cells give q_hat = 1).
2. The constrained problem over per-cell deltas is solved by Lagrangian
   decoupling: for multiplier lambda each cell independently solves
   max_delta [ell_c(delta) - lambda * w_c * delta] via bisection on the
   (monotone decreasing, concavity) envelope score
   d ell_c / d delta = W/(q_hat + delta) - L/(q_hat - delta).
   lambda is then bisected until sum(w_c * delta_c(lambda)) = Delta_0.
3. The lower bound bisects Delta_0 in [-1, delta_hat] for the sign change of
   LR(Delta_0) = 2 * (ell_mle - ell_profile(Delta_0)) - threshold, with
   threshold = NormalDist().inv_cdf(1 - alpha) ** 2 (one-sided Wilks).
   The returned bound is the last rejected point (conservative rounding).

All bisections use fixed iteration counts and no randomness, so results are
bit-for-bit reproducible for identical inputs.

KNOWN LIMITATIONS (this is an implementation pending independent coverage
calibration; no coverage guarantee is claimed here):

- The one-sided Wilks threshold is asymptotic. For boundary-saturated cells
  (all-tie, all-win, all-loss) the parameter sits on the |delta| <= q
  constraint boundary, where regularity conditions fail and the chi-square
  approximation can misbehave in either direction. Null coverage at the
  declared margins remains to be checked by simulation before any acceptance
  use.
- No finite-sample correction (Bartlett, bootstrap, or exact enumeration) is
  applied. Small-cell bounds are likelihood-based approximations, not exact
  sign-test bounds, though they are checkable against enumeration.
- The score uses the chain rule where the q = abs(delta) clamp binds and the
  envelope theorem elsewhere; the residual one-sided selection at the seam
  between regimes is bounded by final bisection bracket width and is
  numerically irrelevant.
- Weights are treated as fixed known constants; weight estimation error is
  out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

_LOG_FLOOR = 1e-280
_DELTA_BOUND = 1.0 - 1e-9
_INNER_ITERS = 48
_LAMBDA_ITERS = 48
_OUTER_ITERS = 48
_BRACKET_DOUBLINGS = 64


@dataclass(frozen=True)
class StratifiedTrinomialResult:
    """Point estimate and one-sided profile-likelihood confidence bounds.

    upper_bound is the mirrored one-sided bound at the same alpha, included so
    win/loss swap symmetry (lower_bound_swapped == -upper_bound_original) is
    directly checkable.
    """

    delta_hat: float
    lower_bound: float
    upper_bound: float
    alpha: float
    threshold: float
    loglik_mle: float
    loglik_at_lower_bound: float
    cell_count: int
    total_observations: int
    weights: tuple[float, ...]


def _validated(
    weights: Sequence[float],
    wins: Sequence[int],
    losses: Sequence[int],
    ties: Sequence[int],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if not (len(wins) == len(losses) == len(ties) == len(weights)):
        raise ValueError("weights, wins, losses, ties must have equal lengths")
    if len(wins) == 0:
        raise ValueError("at least one cell is required")
    for value in list(wins) + list(losses) + list(ties):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if int(value) < 0:
            raise ValueError("counts must be nonnegative")
    wgt: FloatArray = np.asarray([float(v) for v in weights], dtype=np.float64)
    if not np.all(np.isfinite(wgt)) or not np.all(wgt > 0.0):
        raise ValueError("weights must be finite and strictly positive")
    wgt = wgt / float(np.sum(wgt))
    w: FloatArray = np.asarray([float(v) for v in wins], dtype=np.float64)
    lo: FloatArray = np.asarray([float(v) for v in losses], dtype=np.float64)
    ti: FloatArray = np.asarray([float(v) for v in ties], dtype=np.float64)
    if np.any(w + lo + ti < 1.0):
        raise ValueError("empty cell (zero total count) is not allowed")
    return wgt, w, lo, ti


def _profile_q_root(delta: FloatArray, a: FloatArray, b: FloatArray, tie: FloatArray, n: FloatArray) -> FloatArray:
    """Unconstrained concavity root for q_hat(delta) before clamping."""
    linear = a + b * delta
    disc = linear * linear + 4.0 * n * (tie * delta * delta - b * delta)
    return (linear + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * n)


def _profile_q(delta: FloatArray, a: FloatArray, b: FloatArray, tie: FloatArray, n: FloatArray) -> FloatArray:
    """Exact q_hat(delta): larger concavity root clamped to [abs(delta), 1]."""
    root = _profile_q_root(delta, a, b, tie, n)
    return np.minimum(np.maximum(root, np.abs(delta)), 1.0)


def _delta_score(w: FloatArray, lo: FloatArray, ti: FloatArray, q: FloatArray,
                 delta: FloatArray, low_bind: FloatArray) -> FloatArray:
    """Total score d ell/d delta at q_hat(delta); floored denominators.

    Where the lower clamp q = abs(delta) binds, dq/d delta = sign(delta) adds
    the chain-rule term d ell/d q; where q is interior the envelope theorem
    zeroes that term; where q = 1 binds, dq/d delta = 0.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        win_term = w / np.maximum(q + delta, _LOG_FLOOR)
        loss_term = lo / np.maximum(q - delta, _LOG_FLOOR)
        q_deriv = win_term + loss_term - ti / np.maximum(1.0 - q, _LOG_FLOOR)
    return win_term - loss_term + np.where(low_bind, np.sign(delta) * q_deriv, 0.0)


def _cell_loglik(w: FloatArray, lo: FloatArray, ti: FloatArray, q: FloatArray, delta: FloatArray) -> FloatArray:
    """Cell log-likelihood with 0 * log(0) contributions floored to zero."""
    p_win = np.maximum(0.5 * (q + delta), _LOG_FLOOR)
    p_loss = np.maximum(0.5 * (q - delta), _LOG_FLOOR)
    p_tie = np.maximum(1.0 - q, _LOG_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return w * np.log(p_win) + lo * np.log(p_loss) + ti * np.log(p_tie)


def _solve_deltas(lam: float, wgt: FloatArray, w: FloatArray, lo: FloatArray, ti: FloatArray,
                  a: FloatArray, b: FloatArray, n: FloatArray) -> FloatArray:
    """Per-cell argmax of ell_c(delta) - lam * w_c * delta (vectorized bisection)."""
    lower = np.full(w.shape, -_DELTA_BOUND, dtype=np.float64)
    upper = np.full(w.shape, _DELTA_BOUND, dtype=np.float64)
    lam_w = lam * wgt
    for _ in range(_INNER_ITERS):
        mid = 0.5 * (lower + upper)
        q = _profile_q(mid, a, b, ti, n)
        # The lower clamp q = abs(delta) binds iff d ell/d q at q = abs(delta)
        # is nonpositive (the unconstrained root wants q below the clamp).
        # This is exact at equality, e.g. all-tie cells where root == |delta|.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            q_deriv_low = (
                w / np.maximum(np.abs(mid) + mid, _LOG_FLOOR)
                + lo / np.maximum(np.abs(mid) - mid, _LOG_FLOOR)
                - ti / np.maximum(1.0 - np.abs(mid), _LOG_FLOOR)
            )
        go_right = _delta_score(w, lo, ti, q, mid, q_deriv_low <= 0.0) > lam_w
        lower = np.where(go_right, mid, lower)
        upper = np.where(go_right, upper, mid)
    return 0.5 * (lower + upper)


def _weighted_sum(lam: float, wgt: FloatArray, w: FloatArray, lo: FloatArray, ti: FloatArray,
                  a: FloatArray, b: FloatArray, n: FloatArray) -> float:
    return float(np.dot(wgt, _solve_deltas(lam, wgt, w, lo, ti, a, b, n)))


def profile_loglikelihood(
    delta0: float,
    weights: Sequence[float],
    wins: Sequence[int],
    losses: Sequence[int],
    ties: Sequence[int],
) -> float:
    """Max log-likelihood subject to sum(w_c * delta_c) = delta0."""
    wgt, w, lo, ti = _validated(weights, wins, losses, ties)
    return _profile_loglikelihood_arrays(delta0, wgt, w, lo, ti)


def _profile_loglikelihood_arrays(delta0: float, wgt: FloatArray, w: FloatArray,
                                  lo: FloatArray, ti: FloatArray) -> float:
    a = w + lo
    b = w - lo
    n = a + ti
    target = float(min(max(delta0, -_DELTA_BOUND), _DELTA_BOUND))
    lam_lo, lam_hi = -1.0, 1.0
    for _ in range(_BRACKET_DOUBLINGS):
        if _weighted_sum(lam_lo, wgt, w, lo, ti, a, b, n) >= target:
            break
        lam_lo *= 2.0
    for _ in range(_BRACKET_DOUBLINGS):
        if _weighted_sum(lam_hi, wgt, w, lo, ti, a, b, n) <= target:
            break
        lam_hi *= 2.0
    for _ in range(_LAMBDA_ITERS):
        mid = 0.5 * (lam_lo + lam_hi)
        if _weighted_sum(mid, wgt, w, lo, ti, a, b, n) > target:
            lam_lo = mid
        else:
            lam_hi = mid
    lam = 0.5 * (lam_lo + lam_hi)
    delta = _solve_deltas(lam, wgt, w, lo, ti, a, b, n)
    q = _profile_q(delta, a, b, ti, n)
    return float(np.sum(_cell_loglik(w, lo, ti, q, delta)))


def _mle_arrays(w: FloatArray, lo: FloatArray, ti: FloatArray) -> tuple[FloatArray, float]:
    a = w + lo
    n = a + ti
    delta_hat = np.where(a > 0.0, (w - lo) / n, 0.0)
    q_hat = np.where(a > 0.0, a / n, 0.0)
    return delta_hat, float(np.sum(_cell_loglik(w, lo, ti, q_hat, delta_hat)))


def likelihood_ratio(
    delta0: float,
    weights: Sequence[float],
    wins: Sequence[int],
    losses: Sequence[int],
    ties: Sequence[int],
) -> float:
    """Profile likelihood-ratio statistic 2 * (ell_mle - ell_profile(delta0))."""
    wgt, w, lo, ti = _validated(weights, wins, losses, ties)
    _, ll_mle = _mle_arrays(w, lo, ti)
    ll_null = _profile_loglikelihood_arrays(delta0, wgt, w, lo, ti)
    return max(2.0 * (ll_mle - ll_null), 0.0)


def paired_trinomial_inference(
    weights: Sequence[float],
    wins: Sequence[int],
    losses: Sequence[int],
    ties: Sequence[int],
    alpha: float = 0.05,
) -> StratifiedTrinomialResult:
    """Delta point estimate and one-sided level-alpha profile-likelihood lower bound.

    Weights are normalized to sum to 1. The lower bound is the smallest null
    Delta_0 whose profile likelihood-ratio statistic does not exceed the
    one-sided Wilks threshold. Boundary behavior: all-loss data yields the
    feasible-floor bound -1; all-tie/all-win cells contribute finite penalties
    and never collapse the bound to the point estimate.
    """
    if not isinstance(alpha, float) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be a float in (0, 0.5)")
    wgt, w, lo, ti = _validated(weights, wins, losses, ties)
    delta_mle, ll_mle = _mle_arrays(w, lo, ti)
    delta_hat = float(np.dot(wgt, delta_mle))
    threshold = NormalDist().inv_cdf(1.0 - alpha) ** 2

    def lr(delta0: float) -> float:
        return max(2.0 * (ll_mle - _profile_loglikelihood_arrays(delta0, wgt, w, lo, ti)), 0.0)

    floor = -1.0
    if delta_hat <= floor or lr(floor) <= threshold:
        lower_bound = floor
    else:
        rejected, accepted = floor, delta_hat
        for _ in range(_OUTER_ITERS):
            mid = 0.5 * (rejected + accepted)
            if lr(mid) <= threshold:
                accepted = mid
            else:
                rejected = mid
        lower_bound = rejected
    ceiling = 1.0
    if delta_hat >= ceiling or lr(ceiling) <= threshold:
        upper_bound = ceiling
    else:
        accepted, rejected = delta_hat, ceiling
        for _ in range(_OUTER_ITERS):
            mid = 0.5 * (accepted + rejected)
            if lr(mid) <= threshold:
                accepted = mid
            else:
                rejected = mid
        upper_bound = accepted
    return StratifiedTrinomialResult(
        delta_hat=delta_hat,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        alpha=alpha,
        threshold=threshold,
        loglik_mle=ll_mle,
        loglik_at_lower_bound=_profile_loglikelihood_arrays(lower_bound, wgt, w, lo, ti),
        cell_count=int(w.shape[0]),
        total_observations=int(float(np.sum(w + lo + ti))),
        weights=tuple(float(v) for v in wgt),
    )
