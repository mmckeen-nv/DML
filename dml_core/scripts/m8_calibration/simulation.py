"""Deterministic Monte Carlo calibration harness for the M8 evidence gates.

Purpose: measure, before any manifest freeze, the empirical behavior of the
two pending-calibration evidence components:

* ``daystrom_dml.services.m8_statistics.paired_trinomial_inference`` — the
  stratified paired trinomial profile-likelihood lower bound. The harness
  draws synthetic cell structures (per-cell win/loss/tie probabilities,
  weights, scenario counts), runs the lower bound over R replicates, and
  reports empirical coverage of the true weighted Delta, bound-width
  statistics, and boundary-saturation behavior (all-tie / all-win / all-loss
  draws, rare-discordance cells).
* ``daystrom_dml.services.m8_bootstrap.paired_bootstrap`` — the lifecycle and
  tokens ratio gates. The harness synthesizes deterministic arm records and
  reports per-gate pass/block fractions plus the full-conjunction pass
  fraction over independent synthetic datasets.

Spawning scheme (identical construction to m8_bootstrap):
``numpy.random.SeedSequence([20260929])`` is spawned once per cell in
caller-supplied cell order; each child seeds
``numpy.random.Generator(numpy.random.PCG64(child))``. Per cell, one
``multinomial`` draw of shape ``(replicates, 3)`` produces all trinomial
replicates; bootstrap datasets draw per-arm correctness/latency/lifecycle
sequentially from the same per-cell generator. Everything is deterministic
for fixed cell order, fixed seed, and fixed numpy version; reordering cells
changes the stream. No model calls anywhere.

Bound-evaluation cost discipline: one ``paired_trinomial_inference`` call
costs seconds (fixed-iteration bisections, independent of cell count), so all
bound evaluations in this module route through ``cached_bound_evaluations``,
which (a) memoizes by exact inputs — identical count vectors across
replicates, structures, tests, and grid points are evaluated once — and
(b) evaluates uncached keys in a stdlib ``multiprocessing`` process pool
(default context ``fork`` on POSIX, ``spawn`` elsewhere). The service is
deterministic, so caching and parallelism never change any returned value;
they only skip or relocate repeated identical computation. Pool size defaults to
``min(16, os.cpu_count())`` and is overridable with
``DML_M8_CALIBRATION_WORKERS``; the context with ``DML_M8_CALIBRATION_MP``
(``spawn`` | ``fork`` | ``off``). Any pool failure falls back to sequential
evaluation. The module-level cache is capped at ``_CACHE_LIMIT`` entries and
cleared wholesale past the cap (station grids are one-shot processes).

This module is calibration machinery only. It asserts no coverage guarantee;
empirical coverage estimates carry Monte Carlo error of order
``sqrt(p * (1 - p) / R)`` and must be interpreted against the protocol in
docs/m8-calibration-plan-2026-11-02.md. Weights are fixed known constants, as
in the underlying services.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from daystrom_dml.services.m8_bootstrap import (
    MIN_RESAMPLES,
    ArmRecord,
    CellSpec,
    paired_bootstrap,
)
from daystrom_dml.services.m8_statistics import paired_trinomial_inference

IntCounts: TypeAlias = npt.NDArray[np.int64]

SEED_ENTROPY: tuple[int, ...] = (20260929,)
DEFAULT_ALPHA = 0.05
_BOUNDARY_TOL = 1e-9
_PROB_SUM_TOL = 1e-9
_CACHE_LIMIT = 200_000
_POOL_MIN_JOBS = 6
_MAX_WORKERS = 16
_WORKERS_ENV = "DML_M8_CALIBRATION_WORKERS"
_MP_ENV = "DML_M8_CALIBRATION_MP"

# (alpha, weights, wins, losses, ties) -> (delta_hat, lower_bound, upper_bound)
BoundKey: TypeAlias = tuple[float, tuple[float, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
BoundValues: TypeAlias = tuple[float, float, float]
BoundRequest: TypeAlias = tuple[Sequence[float], Sequence[int], Sequence[int], Sequence[int]]

__all__ = [
    "ArmMetricConfig",
    "BootstrapCellConfig",
    "BootstrapGateReport",
    "CellConfig",
    "CoverageReport",
    "DEFAULT_ALPHA",
    "GatePassSummary",
    "ReplicateOutcome",
    "SEED_ENTROPY",
    "all_loss_cell",
    "all_tie_cell",
    "all_win_cell",
    "bootstrap_gate_simulation",
    "boundary_structures",
    "cached_bound_evaluations",
    "cell_generators",
    "exact_single_cell_coverage",
    "rare_discordance_cell",
    "run_boundary_sweep",
    "run_coverage_grid",
    "simulate_trinomial_counts",
    "trinomial_coverage",
    "true_delta",
    "uniform_structure",
]


def _finite(value: float, field: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{field}: finite number required")
    return float(value)


def _validate_seed(seed: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(seed, (tuple, list)) or not seed or any(type(s) is not int or s < 0 for s in seed):
        raise ValueError("seed: nonempty sequence of nonnegative integers required")
    return tuple(seed)


def _validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, float) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be a float in (0, 0.5)")
    return alpha


@dataclass(frozen=True)
class CellConfig:
    """One synthetic stratum for the trinomial simulation.

    q is the discordance probability P(win) + P(loss); delta is
    P(win) - P(loss), so p_win = (q + delta) / 2, p_loss = (q - delta) / 2,
    p_tie = 1 - q, matching the m8_statistics parameterization.
    """

    name: str
    weight: float
    scenarios: int
    q: float
    delta: float
    kind: str = "standard"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name: nonempty string required")
        if _finite(self.weight, "weight") <= 0.0:
            raise ValueError("weight: finite positive number required")
        if type(self.scenarios) is not int or self.scenarios < 1:
            raise ValueError("scenarios: positive integer required")
        q = _finite(self.q, "q")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        delta = _finite(self.delta, "delta")
        if not -q <= delta <= q:
            raise ValueError("delta must satisfy abs(delta) <= q")
        if type(self.kind) is not str or not self.kind:
            raise ValueError("kind: nonempty string required")

    @property
    def p_win(self) -> float:
        return 0.5 * (float(self.q) + float(self.delta))

    @property
    def p_loss(self) -> float:
        return 0.5 * (float(self.q) - float(self.delta))

    @property
    def p_tie(self) -> float:
        return 1.0 - float(self.q)

    @classmethod
    def from_probabilities(
        cls,
        name: str,
        weight: float,
        scenarios: int,
        *,
        p_win: float,
        p_loss: float,
        p_tie: float,
        kind: str = "standard",
    ) -> CellConfig:
        """Build a cell directly from win/loss/tie probabilities."""
        probs = [
            _finite(p_win, "p_win"),
            _finite(p_loss, "p_loss"),
            _finite(p_tie, "p_tie"),
        ]
        if any(p < 0.0 for p in probs):
            raise ValueError("probabilities must be nonnegative")
        if abs(sum(probs) - 1.0) > _PROB_SUM_TOL:
            raise ValueError("p_win + p_loss + p_tie must equal 1")
        pw, pl, _ = probs
        return cls(name=name, weight=weight, scenarios=scenarios, q=pw + pl, delta=pw - pl, kind=kind)


def all_tie_cell(name: str, weight: float = 1.0, scenarios: int = 8) -> CellConfig:
    """Cell whose every scenario is a tie (q = 0)."""
    return CellConfig(name=name, weight=weight, scenarios=scenarios, q=0.0, delta=0.0, kind="all_tie")


def all_win_cell(name: str, weight: float = 1.0, scenarios: int = 8) -> CellConfig:
    """Cell whose every scenario is a DML win (q = 1, delta = 1)."""
    return CellConfig(name=name, weight=weight, scenarios=scenarios, q=1.0, delta=1.0, kind="all_win")


def all_loss_cell(name: str, weight: float = 1.0, scenarios: int = 8) -> CellConfig:
    """Cell whose every scenario is a DML loss (q = 1, delta = -1)."""
    return CellConfig(name=name, weight=weight, scenarios=scenarios, q=1.0, delta=-1.0, kind="all_loss")


def rare_discordance_cell(name: str, weight: float = 1.0, scenarios: int = 8, win_share: float = 1.0) -> CellConfig:
    """Cell with one expected discordant observation (q = 1 / scenarios).

    win_share is the fraction of discordance that is wins: 1.0 means every
    rare discordant observation is a win, 0.0 every one a loss.
    """
    if type(scenarios) is not int or scenarios < 1:
        raise ValueError("scenarios: positive integer required")
    share = _finite(win_share, "win_share")
    if not 0.0 <= share <= 1.0:
        raise ValueError("win_share must lie in [0, 1]")
    q = 1.0 / scenarios
    return CellConfig(
        name=name,
        weight=weight,
        scenarios=scenarios,
        q=q,
        delta=q * (2.0 * share - 1.0),
        kind="rare_discordance",
    )


def uniform_structure(
    cell_count: int,
    *,
    q: float,
    delta: float,
    scenarios: int,
    weight: float = 1.0,
    kind: str = "standard",
) -> tuple[CellConfig, ...]:
    """Identical cells; the workhorse for coverage grid points."""
    if type(cell_count) is not int or cell_count < 1:
        raise ValueError("cell_count: positive integer required")
    return tuple(
        CellConfig(name=f"cell-{i}", weight=weight, scenarios=scenarios, q=q, delta=delta, kind=kind)
        for i in range(cell_count)
    )


def true_delta(configs: Sequence[CellConfig]) -> float:
    """Weighted estimand Delta = sum(w_c * delta_c) / sum(w_c)."""
    if not configs:
        raise ValueError("at least one cell config required")
    if any(not isinstance(c, CellConfig) for c in configs):
        raise ValueError("CellConfig instances required")
    total_weight = sum(float(c.weight) for c in configs)
    return sum(float(c.weight) * float(c.delta) for c in configs) / total_weight


def cell_generators(cell_count: int, *, seed: Sequence[int] = SEED_ENTROPY) -> list[np.random.Generator]:
    """Per-cell PCG64 generators from SeedSequence(seed).spawn(cell_count)."""
    if type(cell_count) is not int or cell_count < 1:
        raise ValueError("cell_count: positive integer required")
    sequence = np.random.SeedSequence(list(_validate_seed(seed)))
    return [np.random.Generator(np.random.PCG64(child)) for child in sequence.spawn(cell_count)]


def simulate_trinomial_counts(
    configs: Sequence[CellConfig],
    *,
    replicates: int,
    seed: Sequence[int] = SEED_ENTROPY,
) -> IntCounts:
    """Draw (replicates, cells, 3) counts; last axis is (wins, losses, ties).

    One multinomial draw of shape (replicates, 3) per cell from that cell's
    own spawned generator; deterministic for fixed config order and seed.
    """
    if not configs:
        raise ValueError("at least one cell config required")
    if any(not isinstance(c, CellConfig) for c in configs):
        raise ValueError("CellConfig instances required")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates: positive integer required")
    generators = cell_generators(len(configs), seed=seed)
    counts = np.empty((replicates, len(configs), 3), dtype=np.int64)
    for ci, (gen, cfg) in enumerate(zip(generators, configs)):
        probs = np.array([cfg.p_win, cfg.p_loss, cfg.p_tie], dtype=np.float64)
        probs = probs / float(np.sum(probs))  # guard 1-ulp rounding in (q +/- delta) / 2
        counts[:, ci, :] = gen.multinomial(cfg.scenarios, probs, size=replicates)
    return counts


def _bound_worker(job: BoundKey) -> tuple[BoundKey, BoundValues]:
    """One deterministic service call; top-level for process-pool pickling."""
    alpha, weights, wins, losses, ties = job
    result = paired_trinomial_inference(list(weights), list(wins), list(losses), list(ties), alpha=alpha)
    return job, (result.delta_hat, result.lower_bound, result.upper_bound)


def _pool_workers() -> int:
    raw = os.environ.get(_WORKERS_ENV)
    if raw is not None:
        try:
            workers = int(raw)
        except ValueError:
            workers = 0
        if workers >= 1:
            return workers
    return max(1, min(_MAX_WORKERS, os.cpu_count() or 1))


def _mp_context() -> str:
    """Pool start method. Default fork on POSIX (no main-module re-import, so
    safe under pytest, REPLs, and -c/stdin drivers); spawn elsewhere. Override
    with DML_M8_CALIBRATION_MP = spawn | fork | off."""
    raw = os.environ.get(_MP_ENV, "").strip().lower()
    if raw in ("spawn", "fork", "forkserver"):
        return raw
    if raw == "off":
        return "off"
    return "fork" if os.name == "posix" else "spawn"


def _evaluate_uncached(jobs: list[BoundKey]) -> list[tuple[BoundKey, BoundValues]]:
    """Evaluate unique uncached keys, in a process pool when worthwhile.

    Sequential fallback on any pool failure; the service is deterministic so
    the execution lane never affects the values.
    """
    context = _mp_context()
    workers = min(_pool_workers(), len(jobs))
    if len(jobs) < _POOL_MIN_JOBS or workers < 2 or context == "off":
        return [_bound_worker(j) for j in jobs]
    try:
        ctx = mp.get_context(context)
        with ctx.Pool(processes=workers) as pool:
            return pool.map(_bound_worker, jobs, chunksize=1)
    except Exception:
        return [_bound_worker(j) for j in jobs]


_BOUND_CACHE: dict[BoundKey, BoundValues] = {}


def _request_key(request: BoundRequest, alpha: float) -> BoundKey:
    if not isinstance(request, (tuple, list)) or len(request) != 4:
        raise ValueError("each request must be a (weights, wins, losses, ties) 4-tuple")
    weights, wins, losses, ties = request
    for seq, field in ((weights, "weights"), (wins, "wins"), (losses, "losses"), (ties, "ties")):
        if not isinstance(seq, (tuple, list)) or not seq:
            raise ValueError(f"{field}: nonempty sequence required")
    if not (len(weights) == len(wins) == len(losses) == len(ties)):
        raise ValueError("weights, wins, losses, ties must have equal lengths")
    weight_tuple = tuple(_finite(w, "weight") for w in weights)
    if any(w <= 0.0 for w in weight_tuple):
        raise ValueError("weights must be positive")
    count_tuples: list[tuple[int, ...]] = []
    for seq, field in ((wins, "wins"), (losses, "losses"), (ties, "ties")):
        values: list[int] = []
        for v in seq:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise ValueError(f"{field}: nonnegative integers required")
            values.append(int(v))
        count_tuples.append(tuple(values))
    w_t, l_t, t_t = count_tuples
    if any(w + lo + t < 1 for w, lo, t in zip(w_t, l_t, t_t)):
        raise ValueError("empty cell (zero total count) is not allowed")
    return (alpha, weight_tuple, w_t, l_t, t_t)


def cached_bound_evaluations(
    requests: Sequence[BoundRequest],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[BoundValues, ...]:
    """Batched profile-likelihood bound evaluations with memoization.

    Each request is ``(weights, wins, losses, ties)``; returns
    ``(delta_hat, lower_bound, upper_bound)`` per request in request order.
    Identical requests (within or across calls) are evaluated once per
    process; uncached keys are evaluated in a process pool. Deterministic:
    values are exactly what ``paired_trinomial_inference`` returns for the
    same inputs.
    """
    _validate_alpha(alpha)
    if not isinstance(requests, (tuple, list)):
        raise ValueError("requests: sequence of (weights, wins, losses, ties) required")
    keys = [_request_key(req, alpha) for req in requests]
    missing = [k for k in dict.fromkeys(keys) if k not in _BOUND_CACHE]
    if missing:
        if len(_BOUND_CACHE) + len(missing) > _CACHE_LIMIT:
            _BOUND_CACHE.clear()
        for key, values in _evaluate_uncached(missing):
            _BOUND_CACHE[key] = values
    return tuple(_BOUND_CACHE[k] for k in keys)


@dataclass(frozen=True)
class ReplicateOutcome:
    """One simulated replicate: point estimate, bounds, coverage indicator."""

    delta_hat: float
    lower_bound: float
    upper_bound: float
    covers: bool


@dataclass(frozen=True)
class CoverageReport:
    """Empirical coverage summary for one synthetic cell structure.

    coverage is the fraction of replicates whose lower bound is at or below
    the true Delta; it carries Monte Carlo error sqrt(p(1-p)/replicates) and
    is not itself a guarantee. Boundary fractions are over (replicate, cell)
    draws that saturate a constraint boundary.
    """

    structure: str
    replicates: int
    alpha: float
    true_delta: float
    coverage: float
    cell_count: int
    scenarios_per_replicate: int
    seed: tuple[int, ...]
    width_mean: float
    width_median: float
    width_min: float
    width_max: float
    floor_fraction: float
    ceil_fraction: float
    all_tie_fraction: float
    all_win_fraction: float
    all_loss_fraction: float
    outcomes: tuple[ReplicateOutcome, ...] | None

    def to_dict(self) -> dict[str, object]:
        """Stable serialization schema (pinned by tests)."""
        return {
            "structure": self.structure,
            "replicates": self.replicates,
            "alpha": self.alpha,
            "true_delta": self.true_delta,
            "coverage": self.coverage,
            "cell_count": self.cell_count,
            "scenarios_per_replicate": self.scenarios_per_replicate,
            "seed": list(self.seed),
            "width": {
                "mean": self.width_mean,
                "median": self.width_median,
                "min": self.width_min,
                "max": self.width_max,
            },
            "floor_fraction": self.floor_fraction,
            "ceil_fraction": self.ceil_fraction,
            "boundary": {
                "all_tie": self.all_tie_fraction,
                "all_win": self.all_win_fraction,
                "all_loss": self.all_loss_fraction,
            },
        }


def _counts_requests(counts: IntCounts, weights: tuple[float, ...]) -> list[BoundRequest]:
    """One bound request per replicate row, in replicate order."""
    return [
        (
            weights,
            tuple(int(v) for v in counts[r, :, 0]),
            tuple(int(v) for v in counts[r, :, 1]),
            tuple(int(v) for v in counts[r, :, 2]),
        )
        for r in range(counts.shape[0])
    ]


def _aggregate_report(
    *,
    structure: str,
    configs: Sequence[CellConfig],
    counts: IntCounts,
    alpha: float,
    entropy: tuple[int, ...],
    bounds: Sequence[BoundValues],
    keep_outcomes: bool,
) -> CoverageReport:
    delta_true = true_delta(configs)
    outcomes = [
        ReplicateOutcome(
            delta_hat=delta_hat,
            lower_bound=lower,
            upper_bound=upper,
            covers=lower <= delta_true,
        )
        for delta_hat, lower, upper in bounds
    ]
    coverage = sum(1 for o in outcomes if o.covers) / len(outcomes)
    widths = np.array([o.upper_bound - o.lower_bound for o in outcomes], dtype=np.float64)
    floor_hits = sum(1 for o in outcomes if o.lower_bound <= -1.0 + _BOUNDARY_TOL)
    ceil_hits = sum(1 for o in outcomes if o.upper_bound >= 1.0 - _BOUNDARY_TOL)

    wins_a = counts[:, :, 0]
    losses_a = counts[:, :, 1]
    ties_a = counts[:, :, 2]
    draws = float(counts.shape[0] * counts.shape[1])
    all_tie = float(np.count_nonzero((wins_a == 0) & (losses_a == 0))) / draws
    all_win = float(np.count_nonzero((losses_a == 0) & (ties_a == 0) & (wins_a > 0))) / draws
    all_loss = float(np.count_nonzero((wins_a == 0) & (ties_a == 0) & (losses_a > 0))) / draws

    return CoverageReport(
        structure=structure,
        replicates=int(counts.shape[0]),
        alpha=alpha,
        true_delta=delta_true,
        coverage=coverage,
        cell_count=int(counts.shape[1]),
        scenarios_per_replicate=int(sum(c.scenarios for c in configs)),
        seed=entropy,
        width_mean=float(np.mean(widths)),
        width_median=float(np.median(widths)),
        width_min=float(np.min(widths)),
        width_max=float(np.max(widths)),
        floor_fraction=floor_hits / len(outcomes),
        ceil_fraction=ceil_hits / len(outcomes),
        all_tie_fraction=all_tie,
        all_win_fraction=all_win,
        all_loss_fraction=all_loss,
        outcomes=tuple(outcomes) if keep_outcomes else None,
    )


def _validate_structures(
    structures: Sequence[tuple[str, Sequence[CellConfig]]],
) -> tuple[tuple[str, tuple[CellConfig, ...]], ...]:
    if not isinstance(structures, (tuple, list)) or not structures:
        raise ValueError("structures: nonempty sequence of (name, configs) pairs required")
    validated: list[tuple[str, tuple[CellConfig, ...]]] = []
    seen: set[str] = set()
    for entry in structures:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each structure must be a (name, configs) pair")
        name, configs = entry
        if type(name) is not str or not name:
            raise ValueError("structure name: nonempty string required")
        if name in seen:
            raise ValueError(f"duplicate structure name {name!r}")
        seen.add(name)
        if not configs or any(not isinstance(c, CellConfig) for c in configs):
            raise ValueError(f"structure {name!r}: nonempty CellConfig sequence required")
        validated.append((name, tuple(configs)))
    return tuple(validated)


def run_coverage_grid(
    structures: Sequence[tuple[str, Sequence[CellConfig]]],
    *,
    replicates: int,
    alpha: float = DEFAULT_ALPHA,
    seed: Sequence[int] = SEED_ENTROPY,
    keep_outcomes: bool = False,
) -> tuple[CoverageReport, ...]:
    """Run ``trinomial_coverage`` over many structures with ONE batched
    bound-evaluation round.

    All replicate rows from all structures are deduped and evaluated together
    (memoized + process pool), so shared cell parameters across grid points
    cost nothing extra. Structures are drawn independently per cell position
    from the same seed, so grid points sharing a cell-count prefix use common
    random numbers on the shared prefix — coverage differences between such
    points are not Monte-Carlo independent, by design (CRN variance
    reduction); document this when comparing grid rows.
    """
    _validate_alpha(alpha)
    entropy = _validate_seed(seed)
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates: positive integer required")
    validated = _validate_structures(structures)

    per_structure: list[tuple[str, tuple[CellConfig, ...], IntCounts, tuple[float, ...]]] = []
    requests: list[BoundRequest] = []
    slices: list[slice] = []
    offset = 0
    for name, configs in validated:
        counts = simulate_trinomial_counts(configs, replicates=replicates, seed=entropy)
        weights = tuple(float(c.weight) for c in configs)
        rows = _counts_requests(counts, weights)
        per_structure.append((name, configs, counts, weights))
        requests.extend(rows)
        slices.append(slice(offset, offset + len(rows)))
        offset += len(rows)

    bounds = cached_bound_evaluations(requests, alpha=alpha)
    return tuple(
        _aggregate_report(
            structure=name,
            configs=configs,
            counts=counts,
            alpha=alpha,
            entropy=entropy,
            bounds=bounds[sl],
            keep_outcomes=keep_outcomes,
        )
        for (name, configs, counts, _), sl in zip(per_structure, slices)
    )


def trinomial_coverage(
    configs: Sequence[CellConfig],
    *,
    structure: str = "structure",
    replicates: int,
    alpha: float = DEFAULT_ALPHA,
    seed: Sequence[int] = SEED_ENTROPY,
    keep_outcomes: bool = False,
) -> CoverageReport:
    """Empirical lower-bound coverage of the true Delta over R replicates."""
    if type(structure) is not str or not structure:
        raise ValueError("structure: nonempty string required")
    (report,) = run_coverage_grid(
        ((structure, tuple(configs)),),
        replicates=replicates,
        alpha=alpha,
        seed=seed,
        keep_outcomes=keep_outcomes,
    )
    return report


def exact_single_cell_coverage(
    scenarios: int,
    q: float,
    delta: float,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Exact lower-bound coverage for one cell by full trinomial enumeration.

    Sums the multinomial probability of every (wins, losses, ties) outcome
    whose profile-likelihood lower bound covers the true delta. This is the
    ground truth the Monte Carlo harness is checked against at small N. Bound
    evaluations are batched and memoized; the enumeration for a given
    (scenarios, alpha) is evaluated once per process regardless of (q, delta).
    """
    _validate_alpha(alpha)
    cfg = CellConfig(name="exact", weight=1.0, scenarios=scenarios, q=q, delta=delta)
    n = cfg.scenarios
    pw, pl, pt = cfg.p_win, cfg.p_loss, cfg.p_tie
    log_pw = math.log(pw) if pw > 0.0 else None
    log_pl = math.log(pl) if pl > 0.0 else None
    log_pt = math.log(pt) if pt > 0.0 else None
    log_n_factorial = math.lgamma(n + 1)

    logps: list[float] = []
    requests: list[BoundRequest] = []
    for w in range(n + 1):
        for lo in range(n + 1 - w):
            t = n - w - lo
            if (w > 0 and log_pw is None) or (lo > 0 and log_pl is None) or (t > 0 and log_pt is None):
                continue  # zero-probability outcome
            logp = log_n_factorial - math.lgamma(w + 1) - math.lgamma(lo + 1) - math.lgamma(t + 1)
            logp += w * log_pw if w > 0 and log_pw is not None else 0.0
            logp += lo * log_pl if lo > 0 and log_pl is not None else 0.0
            logp += t * log_pt if t > 0 and log_pt is not None else 0.0
            logps.append(logp)
            requests.append(((1.0,), (w,), (lo,), (t,)))

    bounds = cached_bound_evaluations(requests, alpha=alpha)
    total = 0.0
    for logp, (_, lower, _) in zip(logps, bounds):
        if lower <= cfg.delta:
            total += math.exp(logp)
    return total


def boundary_structures(
    *,
    cell_count: int = 6,
    scenarios: int = 8,
) -> tuple[tuple[str, tuple[CellConfig, ...]], ...]:
    """Named boundary-saturated structures: all-tie, all-win, all-loss,
    rare discordance (one expected discordant observation per cell), and a
    mixed structure combining all four cell kinds."""
    if type(cell_count) is not int or cell_count < 4:
        raise ValueError("cell_count: integer >= 4 required (mixed structure needs four cells)")
    ties = tuple(all_tie_cell(f"tie-{i}", scenarios=scenarios) for i in range(cell_count))
    wins = tuple(all_win_cell(f"win-{i}", scenarios=scenarios) for i in range(cell_count))
    losses = tuple(all_loss_cell(f"loss-{i}", scenarios=scenarios) for i in range(cell_count))
    rare = tuple(rare_discordance_cell(f"rare-{i}", scenarios=scenarios) for i in range(cell_count))
    mixed = (
        all_tie_cell("mixed-tie", scenarios=scenarios),
        all_win_cell("mixed-win", scenarios=scenarios),
        all_loss_cell("mixed-loss", scenarios=scenarios),
        rare_discordance_cell("mixed-rare", scenarios=scenarios),
    ) + tuple(
        CellConfig(name=f"mixed-std-{i}", weight=1.0, scenarios=scenarios, q=0.3, delta=0.1)
        for i in range(cell_count - 4)
    )
    return (
        ("all_tie", ties),
        ("all_win", wins),
        ("all_loss", losses),
        ("rare_discordance", rare),
        ("mixed_boundary", mixed),
    )


def run_boundary_sweep(
    *,
    cell_count: int = 6,
    scenarios: int = 8,
    replicates: int = 64,
    alpha: float = DEFAULT_ALPHA,
    seed: Sequence[int] = SEED_ENTROPY,
) -> tuple[CoverageReport, ...]:
    """Run trinomial_coverage over every boundary structure in one batched
    evaluation round. Smoke-scale defaults complete in seconds once the
    boundary draws are deduped (pure structures evaluate one unique draw
    each)."""
    return run_coverage_grid(
        boundary_structures(cell_count=cell_count, scenarios=scenarios),
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


@dataclass(frozen=True)
class ArmMetricConfig:
    """Synthetic per-arm scenario metrics for bootstrap gate simulation."""

    latency_ms: float
    lifecycle_ms: float
    tokens: int
    success_rate: float
    cost: float = 0.5

    def __post_init__(self) -> None:
        if _finite(self.latency_ms, "latency_ms") <= 0.0:
            raise ValueError("latency_ms: finite positive number required")
        if _finite(self.lifecycle_ms, "lifecycle_ms") <= 0.0:
            raise ValueError("lifecycle_ms: finite positive number required")
        if type(self.tokens) is not int or self.tokens < 1:
            raise ValueError("tokens: positive integer required")
        rate = _finite(self.success_rate, "success_rate")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("success_rate must lie in [0, 1]")
        if _finite(self.cost, "cost") < 0.0:
            raise ValueError("cost: finite nonnegative number required")


@dataclass(frozen=True)
class BootstrapCellConfig:
    """One synthetic stratum for the bootstrap gate simulation."""

    name: str
    weight: float
    scenarios: int
    cache_condition: str
    arms: Mapping[str, ArmMetricConfig]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name: nonempty string required")
        if _finite(self.weight, "weight") <= 0.0:
            raise ValueError("weight: finite positive number required")
        if type(self.scenarios) is not int or self.scenarios < 1:
            raise ValueError("scenarios: positive integer required")
        if type(self.cache_condition) is not str or not self.cache_condition:
            raise ValueError("cache_condition: nonempty string required")
        if not isinstance(self.arms, Mapping) or not self.arms:
            raise ValueError("arms: nonempty mapping of arm name to ArmMetricConfig required")
        if any(not isinstance(m, ArmMetricConfig) for m in self.arms.values()):
            raise ValueError("arms: ArmMetricConfig values required")


@dataclass(frozen=True)
class GatePassSummary:
    """Pass/block behavior of one bootstrap gate over synthetic datasets.

    upper_bounds is the per-dataset one-sided upper bound (None entries for
    unbounded datasets), present only when keep_upper_bounds=True; tail
    coverage of the one-sided bound is computed from it.
    """

    gate: str
    threshold: float
    datasets: int
    pass_fraction: float
    blocked_fraction: float
    mean_upper_bound: float | None
    upper_bounds: tuple[float | None, ...] | None

    def to_dict(self) -> dict[str, object]:
        """Stable serialization schema (pinned by tests)."""
        return {
            "gate": self.gate,
            "threshold": self.threshold,
            "datasets": self.datasets,
            "pass_fraction": self.pass_fraction,
            "blocked_fraction": self.blocked_fraction,
            "mean_upper_bound": self.mean_upper_bound,
            "upper_bounds": list(self.upper_bounds) if self.upper_bounds is not None else None,
        }


@dataclass(frozen=True)
class BootstrapGateReport:
    """Aggregate bootstrap gate behavior over independent synthetic datasets.

    conjunction_pass_fraction is the fraction of datasets on which EVERY
    gate passed — the full acceptance-gate conjunction, whose power under
    an alternative is bounded by the per-gate margins (Frechet bounds).
    """

    structure: str
    seed: tuple[int, ...]
    datasets: int
    resamples: int
    dml_arm: str
    rag_arm: str
    gates: tuple[GatePassSummary, ...]
    conjunction_pass_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Stable serialization schema (pinned by tests)."""
        return {
            "structure": self.structure,
            "seed": list(self.seed),
            "datasets": self.datasets,
            "resamples": self.resamples,
            "dml_arm": self.dml_arm,
            "rag_arm": self.rag_arm,
            "gates": [g.to_dict() for g in self.gates],
            "conjunction_pass_fraction": self.conjunction_pass_fraction,
        }


def synthesize_bootstrap_dataset(
    configs: Sequence[BootstrapCellConfig],
    generators: Sequence[np.random.Generator],
    *,
    arms: Sequence[str],
) -> list[CellSpec]:
    """One deterministic synthetic dataset: per-scenario correctness is
    Bernoulli(success_rate); latency/lifecycle get +/-5% uniform jitter so
    quantiles are nondegenerate. Draw order per cell per arm is correctness,
    then latency jitter, then lifecycle jitter — deterministic for fixed
    config order and generator state."""
    specs: list[CellSpec] = []
    for gen, cfg in zip(generators, configs):
        per_arm: dict[str, list[ArmRecord]] = {}
        for arm in arms:
            metric = cfg.arms[arm]
            correct = gen.random(cfg.scenarios) < metric.success_rate
            latency_jitter = gen.uniform(0.95, 1.05, size=cfg.scenarios)
            lifecycle_jitter = gen.uniform(0.95, 1.05, size=cfg.scenarios)
            per_arm[arm] = [
                ArmRecord(
                    scenario_id=f"{cfg.name}-{i}",
                    latency_ms=metric.latency_ms * float(latency_jitter[i]),
                    lifecycle_ms=metric.lifecycle_ms * float(lifecycle_jitter[i]),
                    tokens=metric.tokens,
                    correct=bool(correct[i]),
                    cost=metric.cost,
                    cache_condition=cfg.cache_condition,
                )
                for i in range(cfg.scenarios)
            ]
        specs.append(CellSpec(name=cfg.name, weight=cfg.weight, arms=per_arm))
    return specs


def bootstrap_gate_simulation(
    configs: Sequence[BootstrapCellConfig],
    *,
    structure: str = "structure",
    datasets: int,
    resamples: int,
    arms: Sequence[str] = ("dml", "rag"),
    dml_arm: str = "dml",
    rag_arm: str = "rag",
    lifecycle_threshold: float = 1.25,
    tokens_threshold: float = 1.10,
    seed: Sequence[int] = SEED_ENTROPY,
    keep_upper_bounds: bool = False,
) -> BootstrapGateReport:
    """Per-gate pass/block fractions over independent synthetic datasets.

    Each dataset is bootstrapped with the service's own fixed default seed;
    dataset-to-dataset variation comes from the harness generators, so the
    whole run is deterministic for fixed config order and seed.
    """
    entropy = _validate_seed(seed)
    if not configs:
        raise ValueError("at least one cell config required")
    if any(not isinstance(c, BootstrapCellConfig) for c in configs):
        raise ValueError("BootstrapCellConfig instances required")
    if type(structure) is not str or not structure:
        raise ValueError("structure: nonempty string required")
    if type(datasets) is not int or datasets < 1:
        raise ValueError("datasets: positive integer required")
    if type(resamples) is not int or resamples < MIN_RESAMPLES:
        raise ValueError(f"resamples: integer >= {MIN_RESAMPLES} required")
    if (
        not isinstance(arms, (tuple, list))
        or len(arms) < 2
        or any(type(a) is not str or not a for a in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError("arms: at least two distinct nonempty names required")
    arm_tuple = tuple(arms)
    if dml_arm not in arm_tuple or rag_arm not in arm_tuple or dml_arm == rag_arm:
        raise ValueError("dml_arm and rag_arm must be distinct members of arms")
    if _finite(lifecycle_threshold, "lifecycle_threshold") <= 0.0:
        raise ValueError("lifecycle_threshold: finite positive number required")
    if _finite(tokens_threshold, "tokens_threshold") <= 0.0:
        raise ValueError("tokens_threshold: finite positive number required")
    for cfg in configs:
        if set(cfg.arms) != set(arm_tuple):
            raise ValueError(f"cell {cfg.name!r}: arms {sorted(cfg.arms)} != required {sorted(arm_tuple)}")

    conditions = tuple(sorted({cfg.cache_condition for cfg in configs}))
    gate_names = tuple(f"lifecycle_ratio:{cond}" for cond in conditions) + ("tokens_ratio",)
    passes = {name: 0 for name in gate_names}
    blocked = {name: 0 for name in gate_names}
    upper_bounds: dict[str, list[float | None]] = {name: [] for name in gate_names}
    conjunction_passes = 0

    generators = cell_generators(len(configs), seed=entropy)
    for _ in range(datasets):
        specs = synthesize_bootstrap_dataset(configs, generators, arms=arm_tuple)
        result = paired_bootstrap(
            specs,
            arms=arm_tuple,
            dml_arm=dml_arm,
            rag_arm=rag_arm,
            resamples=resamples,
            lifecycle_threshold=lifecycle_threshold,
            tokens_threshold=tokens_threshold,
        )
        gate_map = {f"lifecycle_ratio:{cond}": result.lifecycle_ratio[cond] for cond in conditions}
        gate_map["tokens_ratio"] = result.tokens_ratio
        if all(gate.passes for gate in gate_map.values()):
            conjunction_passes += 1
        for name, gate in gate_map.items():
            if gate.passes:
                passes[name] += 1
            if gate.blocked_by:
                blocked[name] += 1
            upper_bounds[name].append(gate.upper_bound)

    thresholds = {name: lifecycle_threshold for name in gate_names}
    thresholds["tokens_ratio"] = tokens_threshold
    summaries = tuple(
        GatePassSummary(
            gate=name,
            threshold=thresholds[name],
            datasets=datasets,
            pass_fraction=passes[name] / datasets,
            blocked_fraction=blocked[name] / datasets,
            mean_upper_bound=(
                sum(u for u in upper_bounds[name] if u is not None)
                / sum(1 for u in upper_bounds[name] if u is not None)
                if any(u is not None for u in upper_bounds[name])
                else None
            ),
            upper_bounds=tuple(upper_bounds[name]) if keep_upper_bounds else None,
        )
        for name in gate_names
    )
    return BootstrapGateReport(
        structure=structure,
        seed=entropy,
        datasets=datasets,
        resamples=resamples,
        dml_arm=dml_arm,
        rag_arm=rag_arm,
        gates=summaries,
        conjunction_pass_fraction=conjunction_passes / datasets,
    )


def _smoke_grid(alpha: float) -> dict[str, object]:
    """Local smoke grid: exact single-cell enumeration at the declared
    margins. One enumeration of n=8 (45 outcomes) serves every (q, delta)
    row via memoization; completes in well under 60 seconds."""
    rows = []
    for q in (0.15, 0.3, 0.6):
        for delta in (-0.02, 0.0, 0.05, 0.2):
            if abs(delta) > q:
                continue
            rows.append(
                {
                    "type": "exact_single_cell",
                    "scenarios": 8,
                    "q": q,
                    "delta": delta,
                    "alpha": alpha,
                    "coverage": exact_single_cell_coverage(8, q, delta, alpha=alpha),
                }
            )
    return {"preset": "smoke", "alpha": alpha, "rows": rows}


def _margin_grid(cell_counts: Sequence[int], replicates: int, alpha: float) -> dict[str, object]:
    """Station grid: null coverage at the declared delta margins (+0.05,
    -0.02) across cell counts 6..216 and discordance levels."""
    structures: list[tuple[str, tuple[CellConfig, ...]]] = []
    for cells in cell_counts:
        for q in (0.15, 0.3, 0.6):
            for delta in (-0.02, 0.05):
                structures.append(
                    (
                        f"cells{cells}-q{q}-d{delta}",
                        uniform_structure(cells, q=q, delta=delta, scenarios=8),
                    )
                )
    reports = run_coverage_grid(structures, replicates=replicates, alpha=alpha)
    return {
        "preset": "station-margins",
        "alpha": alpha,
        "replicates": replicates,
        "reports": [r.to_dict() for r in reports],
    }


def _power_grid(cell_counts: Sequence[int], replicates: int, alpha: float) -> dict[str, object]:
    """Station grid: power of the lower bound clearing the +0.05 margin at
    true deltas above the margin (acceptance-gate power design input)."""
    structures: list[tuple[str, tuple[CellConfig, ...]]] = []
    for cells in cell_counts:
        for delta in (0.05, 0.1, 0.2):
            structures.append(
                (
                    f"cells{cells}-q0.3-d{delta}",
                    uniform_structure(cells, q=0.3, delta=delta, scenarios=8),
                )
            )
    reports = run_coverage_grid(structures, replicates=replicates, alpha=alpha, keep_outcomes=True)
    rows = []
    for report in reports:
        outcomes = report.outcomes or ()
        clears = sum(1 for o in outcomes if o.lower_bound >= 0.05)
        rows.append(
            {
                "structure": report.structure,
                "true_delta": report.true_delta,
                "replicates": report.replicates,
                "power_clear_+0.05": clears / report.replicates,
                "coverage": report.coverage,
            }
        )
    return {"preset": "station-power", "alpha": alpha, "replicates": replicates, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    """Calibration grid runner. Station runs execute the big grids; the local
    smoke preset completes in under 60 seconds. Example:
    python -m scripts.m8_calibration.simulation --preset smoke
    """
    parser = argparse.ArgumentParser(description="M8 calibration grid runner (deterministic).")
    parser.add_argument(
        "--preset",
        choices=("smoke", "station-margins", "station-power"),
        default="smoke",
        help="smoke: exact small-N enumeration (<60s). station-*: heavy MC grids.",
    )
    parser.add_argument("--replicates", type=int, default=400, help="MC replicates per grid point")
    parser.add_argument(
        "--cells",
        type=str,
        default="6,12,24,48,96,216",
        help="comma-separated cell counts for station grids",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out", type=str, default="", help="write JSON here instead of stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if type(args.replicates) is not int or args.replicates < 1:
        raise ValueError("--replicates: positive integer required")
    alpha = _validate_alpha(float(args.alpha))
    cells = [int(part) for part in str(args.cells).split(",") if part.strip()]
    if not cells or any(c < 1 for c in cells):
        raise ValueError("--cells: comma-separated positive integers required")

    if args.preset == "smoke":
        payload = _smoke_grid(alpha)
    elif args.preset == "station-margins":
        payload = _margin_grid(cells, args.replicates, alpha)
    else:
        payload = _power_grid(cells, args.replicates, alpha)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
