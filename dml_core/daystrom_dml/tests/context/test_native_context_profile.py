import hashlib

import pytest

from daystrom_dml.api_contracts import ContractError
from daystrom_dml.context.native_profile import (
    NativeContextProfileConfig,
    NativeContextProfiler,
    NativeContextSpan,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def span(
    span_id: str,
    start: int,
    tokens: int,
    *,
    resident: bool = True,
    age: int = 0,
    priority: str = "reference",
    authority: str = "reference",
    exact_required: bool = False,
    summary_tokens: int | None = None,
) -> NativeContextSpan:
    return NativeContextSpan(
        span_id=span_id,
        content_digest=digest(span_id),
        start_token=start,
        token_count=tokens,
        resident=resident,
        age_turns=age,
        reference_count=1,
        priority=priority,
        authority=authority,
        exact_required=exact_required,
        summary_digest=digest(f"summary:{span_id}") if summary_tokens is not None else None,
        summary_tokens=summary_tokens,
    )


def config(**overrides) -> NativeContextProfileConfig:
    values = {
        "model_id": "model",
        "runtime_id": "vllm-0.20",
        "model_native_limit": 1_000,
        "served_limit": 800,
        "target_hot_tokens": 600,
        "stale_after_turns": 3,
        "freeze_after_turns": 8,
        "runtime_state_bytes_per_token": 16,
    }
    values.update(overrides)
    return NativeContextProfileConfig(**values)


def action_map(report) -> dict[str, str]:
    return {item.span_id: item.action for item in report.actions}


def test_profiles_retain_compress_freeze_thaw_and_recompute_boundary() -> None:
    spans = [
        span("system", 0, 100, authority="immutable", priority="critical", exact_required=True),
        span("active", 100, 200, age=1, priority="working"),
        span("stale", 300, 400, age=4, summary_tokens=40),
        span("old", 700, 100, age=10),
        span("requested", 800, 200, resident=False, age=12),
    ]

    report = NativeContextProfiler(config()).profile(spans, requested_span_ids=["requested"])

    assert report.feasible is True
    assert action_map(report) == {
        "system": "retain",
        "active": "retain",
        "stale": "compress",
        "old": "freeze",
        "requested": "thaw",
    }
    assert report.logical_tokens == 1_000
    assert report.resident_tokens_before == 800
    assert report.resident_tokens_after == 540
    assert report.compressed_tokens_saved == 360
    assert report.frozen_exact_tokens == 100
    assert report.thawed_exact_tokens == 200
    assert report.stable_prefix_tokens == 300
    assert report.recompute_from_token == 300
    assert report.recompute_tokens == 240
    assert report.tier_bytes_out == 1_600
    assert report.tier_bytes_in == 3_200
    assert report.native_window_utilization == 0.54
    assert report.served_window_utilization == 0.675


def test_requested_thaw_hot_swaps_cold_resident_span_under_pressure() -> None:
    spans = [
        span("system", 0, 100, authority="immutable", priority="critical", exact_required=True),
        span("cold", 100, 400, age=6),
        span("requested", 500, 300, resident=False, age=20),
    ]

    report = NativeContextProfiler(config(target_hot_tokens=500)).profile(
        spans,
        requested_span_ids=["requested"],
    )

    assert report.feasible is True
    assert action_map(report) == {
        "system": "retain",
        "cold": "hot_swap_out",
        "requested": "thaw",
    }
    assert report.hot_swap_in == ["requested"]
    assert report.hot_swap_out == ["cold"]
    assert report.resident_tokens_after == 400
    assert report.stable_prefix_tokens == 100
    assert report.recompute_from_token == 100
    assert report.recompute_tokens == 300


def test_reports_infeasible_when_protected_native_context_exceeds_runtime_budget() -> None:
    spans = [
        span("system", 0, 400, authority="immutable", priority="critical", exact_required=True),
        span("current", 400, 300, authority="current_instruction", priority="critical", exact_required=True),
    ]

    report = NativeContextProfiler(config(target_hot_tokens=600, served_limit=650)).profile(spans)

    assert report.feasible is False
    assert report.resident_tokens_after == 700
    assert report.hot_overflow_tokens == 100
    assert report.served_overflow_tokens == 50
    assert "protected_context_exceeds_hot_budget" in report.reason_codes
    assert "planned_context_exceeds_served_limit" in report.reason_codes


def test_distinguishes_model_native_limit_from_reduced_served_limit() -> None:
    spans = [
        span("hot", 0, 400, priority="working"),
        span("frozen", 400, 500, resident=False, age=20),
    ]

    report = NativeContextProfiler(
        config(model_native_limit=1_000, served_limit=512, target_hot_tokens=500)
    ).profile(spans)

    assert report.feasible is True
    assert report.logical_tokens == 900
    assert report.resident_tokens_after == 400
    assert report.model_native_limit == 1_000
    assert report.served_limit == 512
    assert report.served_limit_shortfall == 488
    assert "runtime_serves_less_than_model_native_limit" in report.reason_codes


def test_contract_rejects_invalid_digests_gaps_overlaps_and_unknown_thaw_requests() -> None:
    with pytest.raises(ContractError, match="content_digest"):
        NativeContextSpan(
            span_id="bad",
            content_digest="not-a-digest",
            start_token=0,
            token_count=1,
        )

    profiler = NativeContextProfiler(config())
    with pytest.raises(ContractError, match="contiguous"):
        profiler.profile([span("a", 0, 10), span("b", 11, 10)])
    with pytest.raises(ContractError, match="model-native"):
        profiler.profile([span("too-large", 0, 1_001)])
    with pytest.raises(ContractError, match="unknown requested"):
        profiler.profile([span("a", 0, 10)], requested_span_ids=["missing"])


def test_report_is_payload_free_and_deterministic() -> None:
    spans = [
        span("system", 0, 10, authority="immutable", exact_required=True),
        span("stale", 10, 20, age=4, summary_tokens=5),
    ]
    profiler = NativeContextProfiler(config())

    left = profiler.profile(spans).to_dict()
    right = profiler.profile(spans).to_dict()

    assert left == right
    assert "content" not in str(left).lower()
    assert left["profile_digest"] == right["profile_digest"]
    assert len(left["profile_digest"]) == 64
