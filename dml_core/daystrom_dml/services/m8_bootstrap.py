"""Paired cell-stratified scenario bootstrap for M8 cost/latency evidence.

Manifest contract: 20,000 fixed-seed paired cell-stratified scenario bootstrap
resamples. Scenarios are resampled with replacement *within each cell* (never
across cells), and one index draw per cell per resample applies to every arm,
so all arms of a scenario stay together. Any scenario missing any arm is
rejected (hard validation error, never silently dropped).

Statistics per resample, summarized as empirical quantile intervals:

* per-arm latency p50/p95/p99 (descriptive),
* lifecycle p95 ratio DML/RAG per cache condition (gate, default <= 1.25),
* tokens-per-correct-success ratio DML/RAG pooled across cells with the fixed
  caller-supplied cell weights (gate, default <= 1.10).

Two-sided 95% intervals (2.5%/97.5% empirical quantiles) are descriptive only
and are reported separately from the one-sided 95% upper gate bound. A gate
passes only when its one-sided bound is at or under threshold AND nothing
blocks it. Zero-denominator statistics and unknown costs are surfaced as
explicit sentinels and block passage; intervals with any invalid (sentinel)
resample are unstable and cannot pass; intervals with no valid resamples are
unbounded and cannot pass.

Spawning scheme (deterministic): ``numpy.random.SeedSequence([20260929])`` is
spawned once per cell in caller-supplied cell order; each child SeedSequence
seeds an independent ``numpy.random.Generator(numpy.random.PCG64(child))``.
Every resample draws, per cell in the same order, ``len(cell)`` scenario
indices with replacement from that cell's generator. Deterministic for fixed
cell order, fixed scenario order within cells, fixed seed and fixed numpy
version; reordering cells or scenarios changes the stream.

Boundary/tail calibration is PENDING before any freeze: quantile method,
interval levels, resample count and gate bounds here are not yet calibrated
against held-out tail behavior. Do not freeze a manifest against these
intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import numpy as np

SEED_ENTROPY: tuple[int, ...] = (20260929,)
DEFAULT_RESAMPLES = 20_000
MIN_RESAMPLES = 100
TWO_SIDED_LEVEL = 0.95
ONE_SIDED_LEVEL = 0.95
DEFAULT_LIFECYCLE_RATIO_THRESHOLD = 1.25
DEFAULT_TOKENS_RATIO_THRESHOLD = 1.10
LATENCY_QUANTILES: tuple[float, ...] = (0.50, 0.95, 0.99)
LATENCY_QUANTILE_NAMES: tuple[str, ...] = ("p50", "p95", "p99")

SPAWNING_SCHEME = (
    "SeedSequence([20260929]).spawn(n_cells) in caller-supplied cell order; "
    "each child seeds numpy.random.Generator(numpy.random.PCG64(child)); "
    "every resample draws, per cell in the same order, len(cell) scenario "
    "indices with replacement. One index draw per cell per resample applies "
    "to every arm, so arms stay paired. Deterministic for fixed cell order, "
    "scenario order, seed and numpy version."
)


class ZeroDenominator:
    """Sentinel: statistic undefined because its denominator is zero.

    Zero-success cells (tokens-per-correct-success) and zero baseline p95
    (lifecycle ratio) produce this. It blocks gates and is never silently
    coerced to a number.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail

    def __repr__(self) -> str:
        return f"ZeroDenominator({self.detail!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZeroDenominator) and other.detail == self.detail

    def __hash__(self) -> int:
        return hash((ZeroDenominator, self.detail))


class UnknownCost:
    """Sentinel: cost/usage unknown for a cell. Blocks every gate."""

    def __init__(self, detail: str) -> None:
        self.detail = detail

    def __repr__(self) -> str:
        return f"UnknownCost({self.detail!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnknownCost) and other.detail == self.detail

    def __hash__(self) -> int:
        return hash((UnknownCost, self.detail))


Stat = Union[float, ZeroDenominator]


@dataclass(frozen=True)
class ArmRecord:
    """One scenario outcome for one arm. cost=None means cost unknown."""

    scenario_id: str
    latency_ms: float
    lifecycle_ms: float
    tokens: int
    correct: bool
    cost: float | None
    cache_condition: str


@dataclass(frozen=True)
class CellSpec:
    """One stratum: fixed weight plus per-arm records keyed for pairing."""

    name: str
    weight: float
    arms: Mapping[str, Sequence[ArmRecord]]


@dataclass(frozen=True)
class CellSummary:
    name: str
    weight: float
    scenarios: int
    cost_total: float | UnknownCost
    zero_success_arms: tuple[str, ...]


@dataclass(frozen=True)
class Interval:
    """Descriptive two-sided interval; never a gate by itself."""

    point: Stat
    two_sided: tuple[float, float] | None
    valid_fraction: float
    stable: bool


@dataclass(frozen=True)
class GateResult:
    """One-sided gate bound plus explicit blocking reasons."""

    threshold: float
    point: Stat
    upper_bound: float | None
    two_sided: tuple[float, float] | None
    valid_fraction: float
    blocked_by: tuple[str, ...]
    passes: bool


@dataclass(frozen=True)
class BootstrapResult:
    seed: tuple[int, ...]
    resamples: int
    spawning: str
    cells: tuple[CellSummary, ...]
    latency: dict[str, dict[str, Interval]]
    lifecycle_ratio: dict[str, GateResult]
    tokens_ratio: GateResult
    samples: dict[str, list[Stat]] | None


@dataclass
class _Cell:
    name: str
    weight: float
    latency: dict[str, np.ndarray]
    lifecycle: dict[str, np.ndarray]
    tokens: dict[str, np.ndarray]
    correct: dict[str, np.ndarray]
    condition_code: dict[str, np.ndarray]
    zero_success_arms: tuple[str, ...]
    cost_total: float | UnknownCost


def _number(value: float | int, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field}: finite nonnegative number required")
    return float(value)


def _positive(value: float, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field}: finite positive number required")
    return float(value)


def _validate_record(cell: str, arm: str, rec: ArmRecord) -> None:
    where = f"cell {cell!r} arm {arm!r}"
    if not isinstance(rec, ArmRecord):
        raise ValueError(f"{where}: ArmRecord required")
    if type(rec.scenario_id) is not str or not rec.scenario_id:
        raise ValueError(f"{where}: nonempty scenario_id required")
    _number(rec.latency_ms, f"{where} scenario {rec.scenario_id!r} latency_ms")
    _number(rec.lifecycle_ms, f"{where} scenario {rec.scenario_id!r} lifecycle_ms")
    if type(rec.tokens) is not int:
        raise ValueError(f"{where} scenario {rec.scenario_id!r}: integer tokens required")
    if type(rec.correct) is not bool:
        raise ValueError(f"{where} scenario {rec.scenario_id!r}: boolean correct required")
    if rec.cost is not None:
        _number(rec.cost, f"{where} scenario {rec.scenario_id!r} cost")
    if type(rec.cache_condition) is not str or not rec.cache_condition:
        raise ValueError(f"{where} scenario {rec.scenario_id!r}: nonempty cache_condition required")


def _validate_resamples(resamples: int) -> int:
    if type(resamples) is not int or resamples < MIN_RESAMPLES:
        raise ValueError(f"resamples: integer >= {MIN_RESAMPLES} required")
    return resamples


def _validate_seed(seed: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(seed, (tuple, list)) or not seed or any(type(s) is not int or s < 0 for s in seed):
        raise ValueError("seed: nonempty sequence of nonnegative integers required")
    return tuple(seed)


def _prepare_cells(
    cells: Sequence[CellSpec],
    arms: tuple[str, ...],
    conditions: tuple[str, ...],
) -> list[_Cell]:
    if not cells:
        raise ValueError("at least one cell required")
    cond_index = {c: i for i, c in enumerate(conditions)}
    seen: set[str] = set()
    prepped: list[_Cell] = []
    for spec in cells:
        if not isinstance(spec, CellSpec):
            raise ValueError("CellSpec required")
        if type(spec.name) is not str or not spec.name or spec.name in seen:
            raise ValueError("distinct nonempty cell names required")
        seen.add(spec.name)
        weight = _positive(spec.weight, f"cell {spec.name!r} weight")
        if set(spec.arms) != set(arms):
            raise ValueError(f"cell {spec.name!r}: arms {sorted(spec.arms)} != required {sorted(arms)}")
        per_arm: dict[str, dict[str, ArmRecord]] = {}
        for arm in arms:
            recs = list(spec.arms[arm])
            if not recs:
                raise ValueError(f"cell {spec.name!r} arm {arm!r}: at least one scenario required")
            by_id: dict[str, ArmRecord] = {}
            for rec in recs:
                _validate_record(spec.name, arm, rec)
                if rec.scenario_id in by_id:
                    raise ValueError(f"cell {spec.name!r} arm {arm!r}: duplicate scenario {rec.scenario_id!r}")
                by_id[rec.scenario_id] = rec
            per_arm[arm] = by_id
        base_ids = [r.scenario_id for r in spec.arms[arms[0]]]
        for arm in arms[1:]:
            missing = [sid for sid in base_ids if sid not in per_arm[arm]]
            extra = [sid for sid in per_arm[arm] if sid not in per_arm[arms[0]]]
            if missing or extra:
                raise ValueError(
                    f"cell {spec.name!r}: scenario pairing broken for arm {arm!r}: "
                    f"missing {missing!r} extra {extra!r}"
                )
        latency: dict[str, np.ndarray] = {}
        lifecycle: dict[str, np.ndarray] = {}
        tokens: dict[str, np.ndarray] = {}
        correct: dict[str, np.ndarray] = {}
        codes: dict[str, np.ndarray] = {}
        zero_success: list[str] = []
        for arm in arms:
            paired = [per_arm[arm][sid] for sid in base_ids]
            latency[arm] = np.array([r.latency_ms for r in paired], dtype=float)
            lifecycle[arm] = np.array([r.lifecycle_ms for r in paired], dtype=float)
            tokens[arm] = np.array([r.tokens for r in paired], dtype=float)
            correct[arm] = np.array([1.0 if r.correct else 0.0 for r in paired], dtype=float)
            codes[arm] = np.array([cond_index[r.cache_condition] for r in paired], dtype=int)
            if correct[arm].sum() == 0.0:
                zero_success.append(arm)
        unknown = sorted(
            {f"{arm}:{sid}" for arm in arms for sid in base_ids if per_arm[arm][sid].cost is None}
        )
        cost: float | UnknownCost
        if unknown:
            cost = UnknownCost(f"cell {spec.name!r}: cost unknown for {unknown!r}")
        else:
            total = 0.0
            for arm in arms:
                for sid in base_ids:
                    known = per_arm[arm][sid].cost
                    total += float(known) if known is not None else 0.0
            cost = total
        prepped.append(
            _Cell(
                name=spec.name,
                weight=weight,
                latency=latency,
                lifecycle=lifecycle,
                tokens=tokens,
                correct=correct,
                condition_code=codes,
                zero_success_arms=tuple(zero_success),
                cost_total=cost,
            )
        )
    return prepped


def draw_resample_indices(
    cell_sizes: Sequence[int],
    *,
    seed: Sequence[int] = SEED_ENTROPY,
    resamples: int = DEFAULT_RESAMPLES,
) -> list[tuple[np.ndarray, ...]]:
    """Per-resample per-cell within-cell index draws (see SPAWNING_SCHEME)."""
    _validate_resamples(resamples)
    entropy = _validate_seed(seed)
    if not cell_sizes or any(type(n) is not int or n < 1 for n in cell_sizes):
        raise ValueError("cell_sizes: nonempty sequence of positive integers required")
    sequence = np.random.SeedSequence(list(entropy))
    generators = [np.random.Generator(np.random.PCG64(child)) for child in sequence.spawn(len(cell_sizes))]
    return [
        tuple(gen.integers(0, n, size=n) for gen, n in zip(generators, cell_sizes)) for _ in range(resamples)
    ]


def _pooled_stats(
    prepped: list[_Cell],
    idxs: tuple[np.ndarray, ...],
    arms: tuple[str, ...],
    conditions: tuple[str, ...],
    dml_arm: str,
    rag_arm: str,
) -> tuple[dict[str, np.ndarray], dict[str, Stat], Stat]:
    latency: dict[str, np.ndarray] = {}
    for arm in arms:
        vals = np.concatenate([p.latency[arm][ix] for p, ix in zip(prepped, idxs)])
        latency[arm] = np.quantile(vals, LATENCY_QUANTILES)
    lifecycle: dict[str, Stat] = {}
    for ci, cond in enumerate(conditions):
        p95: dict[str, float] = {}
        for arm in (dml_arm, rag_arm):
            vals = np.concatenate(
                [p.lifecycle[arm][ix][p.condition_code[arm][ix] == ci] for p, ix in zip(prepped, idxs)]
            )
            p95[arm] = float(np.quantile(vals, 0.95))
        if p95[rag_arm] == 0.0:
            lifecycle[cond] = ZeroDenominator(
                f"cache condition {cond!r}: {rag_arm!r} lifecycle p95 is zero"
            )
        else:
            lifecycle[cond] = p95[dml_arm] / p95[rag_arm]
    per_arm: dict[str, Stat] = {}
    for arm in (dml_arm, rag_arm):
        weighted_tokens = sum(p.weight * float(p.tokens[arm][ix].sum()) for p, ix in zip(prepped, idxs))
        weighted_successes = sum(p.weight * float(p.correct[arm][ix].sum()) for p, ix in zip(prepped, idxs))
        if weighted_successes == 0.0:
            per_arm[arm] = ZeroDenominator(f"arm {arm!r}: zero weighted correct successes")
        else:
            per_arm[arm] = weighted_tokens / weighted_successes
    dml_stat = per_arm[dml_arm]
    rag_stat = per_arm[rag_arm]
    tokens: Stat
    if isinstance(dml_stat, ZeroDenominator):
        tokens = dml_stat
    elif isinstance(rag_stat, ZeroDenominator):
        tokens = rag_stat
    elif rag_stat == 0.0:
        tokens = ZeroDenominator(f"arm {rag_arm!r}: zero weighted tokens")
    else:
        tokens = dml_stat / rag_stat
    return latency, lifecycle, tokens


def _interval(point: Stat, samples: list[Stat]) -> Interval:
    valid = np.array([s for s in samples if type(s) is float], dtype=float)
    fraction = valid.size / len(samples) if samples else 0.0
    stable = fraction == 1.0
    two_sided: tuple[float, float] | None = None
    if valid.size:
        alpha = (1.0 - TWO_SIDED_LEVEL) / 2.0
        two_sided = (float(np.quantile(valid, alpha)), float(np.quantile(valid, 1.0 - alpha)))
    return Interval(point=point, two_sided=two_sided, valid_fraction=fraction, stable=stable)


def _gate(point: Stat, samples: list[Stat], threshold: float, unknown_cost: bool) -> GateResult:
    interval = _interval(point, samples)
    valid = np.array([s for s in samples if type(s) is float], dtype=float)
    upper = float(np.quantile(valid, ONE_SIDED_LEVEL)) if valid.size else None
    blocked: list[str] = []
    if isinstance(point, ZeroDenominator):
        blocked.append(f"zero_denominator: {point.detail}")
    if not interval.stable:
        blocked.append("unstable_interval")
    if upper is None or not math.isfinite(upper):
        blocked.append("unbounded_interval")
    if unknown_cost:
        blocked.append("unknown_cost")
    passes = not blocked and upper is not None and upper <= threshold
    return GateResult(
        threshold=threshold,
        point=point,
        upper_bound=upper,
        two_sided=interval.two_sided,
        valid_fraction=interval.valid_fraction,
        blocked_by=tuple(blocked),
        passes=passes,
    )


def paired_bootstrap(
    cells: Sequence[CellSpec],
    *,
    arms: Sequence[str],
    dml_arm: str,
    rag_arm: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: Sequence[int] = SEED_ENTROPY,
    lifecycle_threshold: float = DEFAULT_LIFECYCLE_RATIO_THRESHOLD,
    tokens_threshold: float = DEFAULT_TOKENS_RATIO_THRESHOLD,
    keep_samples: bool = False,
) -> BootstrapResult:
    """Run the paired cell-stratified bootstrap. Manifest defaults are fixed;
    only tests should override seed/resamples/thresholds."""
    _validate_resamples(resamples)
    entropy = _validate_seed(seed)
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
    lifecycle_threshold = _positive(lifecycle_threshold, "lifecycle_threshold")
    tokens_threshold = _positive(tokens_threshold, "tokens_threshold")
    conditions = tuple(
        sorted({r.cache_condition for spec in cells for recs in spec.arms.values() for r in recs if isinstance(r, ArmRecord)})
    )
    if not conditions:
        raise ValueError("at least one record required")
    prepped = _prepare_cells(cells, arm_tuple, conditions)
    unknown_cost = any(isinstance(p.cost_total, UnknownCost) for p in prepped)

    identity = tuple(np.arange(p.latency[arm_tuple[0]].size) for p in prepped)
    point_latency, point_lifecycle, point_tokens = _pooled_stats(
        prepped, identity, arm_tuple, conditions, dml_arm, rag_arm
    )

    lat_samples: dict[str, dict[str, list[Stat]]] = {
        arm: {q: [] for q in LATENCY_QUANTILE_NAMES} for arm in arm_tuple
    }
    life_samples: dict[str, list[Stat]] = {c: [] for c in conditions}
    tok_samples: list[Stat] = []
    for idxs in draw_resample_indices(
        [p.latency[arm_tuple[0]].size for p in prepped], seed=entropy, resamples=resamples
    ):
        lat, life, tok = _pooled_stats(prepped, idxs, arm_tuple, conditions, dml_arm, rag_arm)
        for arm in arm_tuple:
            for qi, qname in enumerate(LATENCY_QUANTILE_NAMES):
                lat_samples[arm][qname].append(float(lat[arm][qi]))
        for cond in conditions:
            life_samples[cond].append(life[cond])
        tok_samples.append(tok)

    latency = {
        arm: {q: _interval(float(point_latency[arm][qi]), lat_samples[arm][q]) for qi, q in enumerate(LATENCY_QUANTILE_NAMES)}
        for arm in arm_tuple
    }
    lifecycle_ratio = {
        cond: _gate(point_lifecycle[cond], life_samples[cond], lifecycle_threshold, unknown_cost)
        for cond in conditions
    }
    tokens_ratio = _gate(point_tokens, tok_samples, tokens_threshold, unknown_cost)
    summaries = tuple(
        CellSummary(
            name=p.name,
            weight=p.weight,
            scenarios=int(p.latency[arm_tuple[0]].size),
            cost_total=p.cost_total,
            zero_success_arms=p.zero_success_arms,
        )
        for p in prepped
    )
    samples: dict[str, list[Stat]] | None = None
    if keep_samples:
        samples = {f"latency:{arm}:{q}": lat_samples[arm][q] for arm in arm_tuple for q in LATENCY_QUANTILE_NAMES}
        samples.update({f"lifecycle_ratio:{cond}": life_samples[cond] for cond in conditions})
        samples["tokens_ratio"] = tok_samples
    return BootstrapResult(
        seed=entropy,
        resamples=resamples,
        spawning=SPAWNING_SCHEME,
        cells=summaries,
        latency=latency,
        lifecycle_ratio=lifecycle_ratio,
        tokens_ratio=tokens_ratio,
        samples=samples,
    )
