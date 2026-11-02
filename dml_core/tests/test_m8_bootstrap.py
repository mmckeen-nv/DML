import numpy as np
import pytest

from daystrom_dml.services.m8_bootstrap import (
    SEED_ENTROPY,
    SPAWNING_SCHEME,
    ArmRecord,
    BootstrapResult,
    CellSpec,
    UnknownCost,
    ZeroDenominator,
    draw_resample_indices,
    paired_bootstrap,
)

ARMS = ("dml", "rag", "noop")


def rec(sid, *, latency=1.0, lifecycle=1.0, tokens=10, correct=True, cost=0.5, condition="cold"):
    return ArmRecord(
        scenario_id=sid,
        latency_ms=latency,
        lifecycle_ms=lifecycle,
        tokens=tokens,
        correct=correct,
        cost=cost,
        cache_condition=condition,
    )


def make_cell(name="cell", weight=1.0, n=8, condition="cold", arm_overrides=None):
    overrides = arm_overrides or {}
    per_arm = {}
    for ai, arm in enumerate(ARMS):
        kw = dict(
            latency=1.0 + ai + 0.1,
            lifecycle=1.0 + 0.5 * ai,
            tokens=10 + ai,
            correct=True,
            cost=0.5,
            condition=condition,
        )
        kw.update(overrides.get(arm, {}))
        per_arm[arm] = [rec(f"{name}-{i}", **{k: (v if not callable(v) else v(i)) for k, v in kw.items()}) for i in range(n)]
    return CellSpec(name=name, weight=weight, arms=per_arm)


def test_determinism_same_seed_same_result():
    cells = [make_cell("a", 1.0, 8), make_cell("b", 2.0, 6, condition="warm")]
    first = paired_bootstrap(cells, arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200, keep_samples=True)
    second = paired_bootstrap(cells, arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200, keep_samples=True)
    assert first == second
    assert first.seed == SEED_ENTROPY
    draws_a = draw_resample_indices([8, 6], resamples=200)
    draws_b = draw_resample_indices([8, 6], resamples=200)
    assert all(np.array_equal(x, y) for ra, rb in zip(draws_a, draws_b) for x, y in zip(ra, rb))


def test_spawning_scheme_documented():
    assert "SeedSequence([20260929])" in SPAWNING_SCHEME
    assert "PCG64" in SPAWNING_SCHEME
    result = paired_bootstrap([make_cell()], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100)
    assert result.spawning == SPAWNING_SCHEME


def test_pairing_keeps_arms_together():
    # Identical per-scenario latencies in every arm, record order shuffled per
    # arm. If resamples keep arms paired, per-resample quantiles are identical
    # across arms; independent per-arm resampling would diverge almost surely.
    values = [0.5, 1.0, 2.0, 5.0, 50.0, 500.0]
    orders = {"dml": [0, 1, 2, 3, 4, 5], "rag": [5, 3, 4, 1, 0, 2], "noop": [2, 0, 5, 4, 3, 1]}
    per_arm = {
        arm: [rec(f"s{i}", latency=values[i], lifecycle=1.0 + i) for i in order] for arm, order in orders.items()
    }
    cell = CellSpec(name="paired", weight=1.0, arms=per_arm)
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=300, keep_samples=True)
    assert result.samples is not None
    for q in ("p50", "p95", "p99"):
        assert result.samples[f"latency:dml:{q}"] == result.samples[f"latency:rag:{q}"] == result.samples[
            f"latency:noop:{q}"
        ]
    assert result.samples["latency:dml:p95"] != [result.samples["latency:dml:p95"][0]] * 300  # not degenerate


def test_cell_stratification_never_crosses_cells():
    # Constant cells: any cross-cell pooling would perturb the pooled p50.
    low = make_cell("low", 1.0, 4, arm_overrides={a: {"latency": 1.0} for a in ARMS})
    high = make_cell("high", 1.0, 6, arm_overrides={a: {"latency": 1000.0} for a in ARMS})
    result = paired_bootstrap([low, high], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200, keep_samples=True)
    assert result.samples is not None
    for arm in ARMS:
        samples = result.samples[f"latency:{arm}:p50"]
        assert set(samples) == {1000.0}
        interval = result.latency[arm]["p50"]
        assert interval.two_sided == (1000.0, 1000.0)
    draws = draw_resample_indices([4, 6], resamples=200)
    assert all(len(d[0]) == 4 and len(d[1]) == 6 for d in draws)
    assert all(int(d[0].max()) < 4 and int(d[1].max()) < 6 for d in draws)


def test_zero_denominator_surfaced_and_blocking():
    cell = make_cell(
        "z",
        1.0,
        6,
        arm_overrides={
            "rag": {"correct": False, "lifecycle": 0.0},
        },
    )
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200)
    summary = result.cells[0]
    assert "rag" in summary.zero_success_arms
    tokens = result.tokens_ratio
    assert isinstance(tokens.point, ZeroDenominator)
    assert tokens.passes is False
    assert any(b.startswith("zero_denominator") for b in tokens.blocked_by)
    lifecycle = result.lifecycle_ratio["cold"]
    assert isinstance(lifecycle.point, ZeroDenominator)
    assert lifecycle.passes is False
    assert any(b.startswith("zero_denominator") for b in lifecycle.blocked_by)


def test_unstable_interval_cannot_pass():
    # One success in four scenarios: many resamples have zero successes.
    cell = make_cell("u", 1.0, 4, arm_overrides={"dml": {"correct": lambda i: i == 0}})
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=300, keep_samples=True)
    assert result.samples is not None
    assert any(isinstance(s, ZeroDenominator) for s in result.samples["tokens_ratio"])
    gate = result.tokens_ratio
    assert not isinstance(gate.point, ZeroDenominator)
    assert gate.valid_fraction < 1.0
    assert "unstable_interval" in gate.blocked_by
    assert gate.passes is False


def test_unknown_cost_surfaced_and_blocking():
    good = make_cell("good", 1.0, 6)
    bad_records = {
        arm: [rec(f"bad-{i}", cost=None if i == 0 else 0.5) for i in range(6)] for arm in ARMS
    }
    bad = CellSpec(name="bad", weight=1.0, arms=bad_records)
    result = paired_bootstrap([good, bad], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200)
    assert isinstance(result.cells[1].cost_total, UnknownCost)
    assert isinstance(result.cells[0].cost_total, float)
    assert "unknown_cost" in result.tokens_ratio.blocked_by
    assert "unknown_cost" in result.lifecycle_ratio["cold"].blocked_by
    assert result.tokens_ratio.passes is False
    assert result.lifecycle_ratio["cold"].passes is False


def test_degenerate_constant_data_degenerate_interval_and_gate_pass():
    cell = make_cell(
        "c",
        1.0,
        5,
        arm_overrides={
            "dml": {"latency": 5.0, "lifecycle": 1.0, "tokens": 10},
            "rag": {"latency": 5.0, "lifecycle": 1.5, "tokens": 11},
            "noop": {"latency": 5.0, "lifecycle": 2.0, "tokens": 12},
        },
    )
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200)
    interval = result.latency["dml"]["p95"]
    assert interval.point == 5.0
    assert interval.two_sided == (5.0, 5.0)
    assert interval.stable is True
    gate = result.lifecycle_ratio["cold"]
    assert gate.point == pytest.approx(1.0 / 1.5)
    assert gate.upper_bound == pytest.approx(1.0 / 1.5)
    assert gate.two_sided == pytest.approx((1.0 / 1.5, 1.0 / 1.5))
    assert gate.blocked_by == ()
    assert gate.passes is True
    tokens = result.tokens_ratio
    assert tokens.point == pytest.approx(10 / 11)
    assert tokens.passes is True


def test_skewed_data_sensible_quantiles():
    skewed = [1.0] * 18 + [100.0, 200.0]
    per_arm = {arm: [rec(f"s{i}", latency=v, lifecycle=v) for i, v in enumerate(skewed)] for arm in ARMS}
    cell = CellSpec(name="skew", weight=1.0, arms=per_arm)
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=300)
    p50, p95, p99 = (result.latency["dml"][q] for q in ("p50", "p95", "p99"))
    assert p50.point == 1.0
    assert p95.point == pytest.approx(np.quantile(skewed, 0.95))
    assert p99.point >= p95.point >= p50.point
    for interval in (p50, p95, p99):
        lo, hi = interval.two_sided
        assert 1.0 <= lo <= hi <= 200.0
    assert p50.two_sided[0] <= p95.two_sided[0]


def test_gate_fails_above_threshold_and_weighting_matters():
    cell = make_cell(
        "slow",
        1.0,
        5,
        arm_overrides={"dml": {"lifecycle": 2.0}, "rag": {"lifecycle": 1.0}},
    )
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200)
    gate = result.lifecycle_ratio["cold"]
    assert gate.point == pytest.approx(2.0)
    assert gate.passes is False
    assert gate.blocked_by == ()
    # Fixed weights: heavy cell dominates pooled tokens-per-correct-success.
    cheap = make_cell("cheap", 1.0, 5, arm_overrides={"dml": {"tokens": 10}, "rag": {"tokens": 10}})
    pricey = make_cell("pricey", 9.0, 5, arm_overrides={"dml": {"tokens": 20}, "rag": {"tokens": 10}})
    weighted = paired_bootstrap([cheap, pricey], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=200)
    assert weighted.tokens_ratio.point == pytest.approx((1 * 50 + 9 * 100) / (1 * 50 + 9 * 50))
    assert weighted.tokens_ratio.passes is False  # 1.9 > 1.10


def test_two_sided_reported_separately_from_one_sided_bound():
    per_arm = {arm: [rec(f"s{i}", lifecycle=1.0 + 0.1 * i + (0.05 if arm == "dml" else 0.0)) for i in range(20)] for arm in ARMS}
    cell = CellSpec(name="spread", weight=1.0, arms=per_arm)
    result = paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=300)
    gate = result.lifecycle_ratio["cold"]
    lo, hi = gate.two_sided
    assert lo <= gate.upper_bound <= hi or gate.upper_bound == pytest.approx(hi, rel=0.05)
    assert lo < hi  # genuinely two-sided descriptive spread
    assert gate.threshold == 1.25


def test_resample_count_and_seed_validation():
    cell = make_cell()
    for bad in (0, 99, 100.5, True, "20000"):
        with pytest.raises(ValueError):
            paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=bad)
    for bad_seed in ([], [-1], [1.5], "20260929"):
        with pytest.raises(ValueError):
            paired_bootstrap([cell], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100, seed=bad_seed)
    with pytest.raises(ValueError):
        draw_resample_indices([0], resamples=100)


def test_weight_and_cell_validation():
    for bad_weight in (0.0, -1.0, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            paired_bootstrap(
                [make_cell("w", bad_weight)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
            )
    with pytest.raises(ValueError):  # duplicate cell names
        paired_bootstrap([make_cell("x"), make_cell("x")], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100)
    with pytest.raises(ValueError):  # no cells
        paired_bootstrap([], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100)


def test_pairing_and_record_validation():
    arms_missing = {"dml": [rec("a"), rec("b")], "rag": [rec("a")], "noop": [rec("a"), rec("b")]}
    with pytest.raises(ValueError, match="pairing"):
        paired_bootstrap(
            [CellSpec(name="m", weight=1.0, arms=arms_missing)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
        )
    dup = {arm: [rec("a"), rec("a")] for arm in ARMS}
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(
            [CellSpec(name="d", weight=1.0, arms=dup)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
        )
    wrong_arms = {"dml": [rec("a")], "rag": [rec("a")]}
    with pytest.raises(ValueError):
        paired_bootstrap(
            [CellSpec(name="w", weight=1.0, arms=wrong_arms)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
        )
    with pytest.raises(ValueError):
        paired_bootstrap([make_cell()], arms=ARMS, dml_arm="dml", rag_arm="dml", resamples=100)
    bad_latency = {arm: [rec("a", latency=-1.0)] for arm in ARMS}
    with pytest.raises(ValueError):
        paired_bootstrap(
            [CellSpec(name="bl", weight=1.0, arms=bad_latency)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
        )
    bool_tokens = {arm: [rec("a", tokens=True)] for arm in ARMS}
    with pytest.raises(ValueError):
        paired_bootstrap(
            [CellSpec(name="bt", weight=1.0, arms=bool_tokens)], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100
        )
    assert isinstance(
        paired_bootstrap([make_cell()], arms=ARMS, dml_arm="dml", rag_arm="rag", resamples=100), BootstrapResult
    )
