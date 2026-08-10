#!/usr/bin/env python3
"""Render the verified Nemotron context-exhaustion evidence chart."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
COLD_ARTIFACT = ROOT / "docs/artifacts/vllm-cold-context-boundary-pilot-2026-08-10.json"
CHAIN_ARTIFACT = ROOT / "docs/artifacts/vllm-native-transition-chain-poc-2026-08-10.json"
OUTPUT = ROOT / "docs/artifacts/vllm-context-exhaustion-chart-2026-08-10.svg"

W, H = 1800, 1050
BG = "#071018"
PANEL = "#0D1923"
PANEL_2 = "#101F2B"
INK = "#F4F8FB"
MUTED = "#91A4B2"
GRID = "#263946"
CYAN = "#43D9C5"
GREEN = "#8BE36A"
AMBER = "#FFCB66"
RED = "#FF6B6B"
BLUE = "#6EAEFF"


def text(x: float, y: float, value: str, size: int, *, fill: str = INK, weight: int = 400, anchor: str = "start", family: str = "Inter, Helvetica Neue, Arial, sans-serif") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" font-family="{family}">'
        f'{escape(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, *, radius: int = 18, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"{stroke_attr}/>'


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def main() -> None:
    cold = json.loads(COLD_ARTIFACT.read_text(encoding="utf-8"))
    chain = json.loads(CHAIN_ARTIFACT.read_text(encoding="utf-8"))
    accepted = [case for case in cold["cases"] if case["http_status"] == 200]
    rejected = next(case for case in cold["cases"] if case["http_status"] != 200)
    enabled = next(event for event in chain["events"] if event["operation"] == "transition" and event["prompt_tokens"] == 18500)
    disabled = next(event for event in chain["events"] if event["operation"] == "save" and event["prompt_tokens"] == 18500)

    enabled_ms = float(enabled["latency_ms"])
    disabled_ms = float(disabled["latency_ms"])
    saved_ms = disabled_ms - enabled_ms
    faster_pct = saved_ms / disabled_ms * 100.0
    speedup = disabled_ms / enabled_ms
    reused = int(enabled["cpu_offload_matched_tokens"])
    prompt_tokens = int(enabled["prompt_tokens"])
    remaining = prompt_tokens - reused
    reused_pct = reused / prompt_tokens * 100.0
    served_limit = int(cold["served_limit"])
    output_equal = enabled["output_digest"] == disabled["output_digest"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        rect(0, 0, W, H, BG, radius=0),
        text(70, 78, "DCM reduces repeated prefill—until the served window ends", 48, weight=720),
        text(70, 120, "Verified Nemotron 3 Super evidence · vLLM 0.20 · concurrency 1", 22, fill=MUTED),
    ]

    # KPI strip.
    kpis = [
        (70, "62.7%", "CPU fallback vs cold at 18.5K", GREEN),
        (430, "12,672", "tokens reprocessing avoided", CYAN),
        (850, "733 ms", "saved on the paired request", BLUE),
        (1250, "65,536", "served logical limit", AMBER),
    ]
    for x, value, label, color in kpis:
        svg.extend([
            text(x, 187, value, 38, fill=color, weight=720),
            text(x, 217, label, 17, fill=MUTED, weight=500),
        ])

    # Left panel: cold context scaling.
    px, py, pw, ph = 70, 255, 1045, 625
    svg.append(rect(px, py, pw, ph, PANEL, stroke=GRID))
    svg.extend([
        text(px + 34, py + 48, "Cold full-history recomputation", 27, weight=680),
        text(px + 34, py + 80, "Cold-control isolation · APC remained enabled at server", 17, fill=MUTED),
    ])
    chart_x0, chart_y0 = px + 82, py + 125
    chart_w, chart_h = pw - 132, ph - 205
    xmin, xmax, ymax = 30000, 66500, 4.2

    def sx(tokens: float) -> float:
        return chart_x0 + (tokens - xmin) / (xmax - xmin) * chart_w

    def sy(seconds: float) -> float:
        return chart_y0 + chart_h - seconds / ymax * chart_h

    for seconds in [0, 1, 2, 3, 4]:
        y = sy(seconds)
        svg.extend([
            line(chart_x0, y, chart_x0 + chart_w, y, stroke=GRID),
            text(chart_x0 - 18, y + 6, f"{seconds}s", 15, fill=MUTED, anchor="end"),
        ])
    for tokens in [32000, 40000, 48000, 56000, 64000]:
        x = sx(tokens)
        svg.extend([
            line(x, chart_y0, x, chart_y0 + chart_h, stroke=GRID),
            text(x, chart_y0 + chart_h + 29, f"{tokens // 1000}K", 15, fill=MUTED, anchor="middle"),
        ])

    points = [(int(case["actual_prompt_tokens"]), float(case["native_ttft_delta_ms"]) / 1000.0) for case in accepted]
    path = " ".join(("M" if i == 0 else "L") + f" {sx(t):.1f} {sy(v):.1f}" for i, (t, v) in enumerate(points))
    svg.append(f'<path d="{path}" fill="none" stroke="{CYAN}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    for tokens, seconds in points:
        x, y = sx(tokens), sy(seconds)
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{BG}" stroke="{CYAN}" stroke-width="4"/>'
        )
        if tokens == 65000:
            svg.append(text(x - 12, y - 18, f"{seconds:.2f}s @ 65K", 15, fill=INK, weight=650, anchor="end"))
        elif tokens < 65000:
            svg.append(text(x, y - 18, f"{seconds:.2f}s", 15, fill=INK, weight=650, anchor="middle"))

    limit_x = sx(served_limit)
    svg.extend([
        line(limit_x, chart_y0, limit_x, chart_y0 + chart_h, stroke=AMBER, width=3, dash="8 8"),
        text(limit_x - 12, chart_y0 + 145, "65,536 hard limit", 15, fill=AMBER, weight=700, anchor="end"),
        f'<circle cx="{sx(rejected["actual_prompt_tokens"]):.1f}" cy="{sy(0.35):.1f}" r="10" fill="none" stroke="{RED}" stroke-width="4"/>',
        line(sx(rejected["actual_prompt_tokens"]) - 7, sy(0.35) - 7, sx(rejected["actual_prompt_tokens"]) + 7, sy(0.35) + 7, stroke=RED, width=3),
        line(sx(rejected["actual_prompt_tokens"]) + 7, sy(0.35) - 7, sx(rejected["actual_prompt_tokens"]) - 7, sy(0.35) + 7, stroke=RED, width=3),
        text(limit_x - 20, sy(0.35) + 6, "65,538 combined → HTTP 400", 15, fill=RED, weight=650, anchor="end"),
        text(chart_x0 + chart_w / 2, py + ph - 25, "Actual prompt tokens", 16, fill=MUTED, anchor="middle"),
    ])

    # Right top: one valid paired A/B point.
    rx, ry, rw, rh = 1145, 255, 585, 350
    svg.append(rect(rx, ry, rw, rh, PANEL_2, stroke=GRID))
    svg.extend([
        text(rx + 32, ry + 48, "18.5K CPU-fallback canary", 27, weight=680),
        text(rx + 32, ry + 79, "Same prompt · deterministic output equal", 17, fill=MUTED),
    ])
    bar_x0, bar_y0, bar_w, max_ms = rx + 38, ry + 135, rw - 76, 1250.0
    bars = [
        ("With DCM", "managed CPU restore", enabled_ms, GREEN),
        ("Without DCM", "cold full recomputation", disabled_ms, MUTED),
    ]
    for i, (label, sublabel, value, color) in enumerate(bars):
        y = bar_y0 + i * 88
        width = value / max_ms * bar_w
        svg.extend([
            text(bar_x0, y - 12, label, 18, weight=680),
            text(bar_x0 + 145, y - 12, sublabel, 14, fill=MUTED),
            rect(bar_x0, y, bar_w, 30, "#172B38", radius=7),
            rect(bar_x0, y, width, 30, color, radius=7),
            text(bar_x0 + width - 10 if width > 110 else bar_x0 + width + 10, y + 22, f"{value:.0f} ms", 16, fill=BG if width > 110 else INK, weight=750, anchor="end" if width > 110 else "start"),
        ])
    svg.extend([
        text(rx + 32, ry + rh - 35, f"DCM faster by {faster_pct:.1f}%  ·  {speedup:.2f}× cold/managed ratio", 19, fill=GREEN, weight=700),
    ])

    # Right bottom: processing avoided and boundary truth.
    by, bh = 630, 250
    svg.append(rect(rx, by, rw, bh, PANEL, stroke=GRID))
    svg.extend([
        text(rx + 32, by + 46, "Repeated context processing", 25, weight=680),
        text(rx + 32, by + 77, "Native counters from the managed 18.5K request", 16, fill=MUTED),
    ])
    stack_x, stack_y, stack_w = rx + 32, by + 110, rw - 64
    reused_w = stack_w * reused / prompt_tokens
    svg.extend([
        rect(stack_x, stack_y, stack_w, 42, "#21313D", radius=9),
        rect(stack_x, stack_y, reused_w, 42, CYAN, radius=9),
        text(stack_x + reused_w / 2, stack_y + 28, f"{reused:,} avoided", 16, fill=BG, weight=750, anchor="middle"),
        text(stack_x + reused_w + (stack_w - reused_w) / 2, stack_y + 28, f"{remaining:,} new", 15, fill=INK, weight=650, anchor="middle"),
        text(rx + 32, by + 184, f"{reused_pct:.1f}% of prompt reprocessing avoided—not {reused_pct:.1f}% faster", 17, fill=CYAN, weight=650),
        text(rx + 32, by + 215, f"Output digest equal: {'yes' if output_equal else 'no'}", 16, fill=GREEN if output_equal else RED, weight=650),
    ])

    # Footer / limitations.
    svg.extend([
        line(70, 920, 1730, 920, stroke=GRID),
        text(70, 960, "Normal route", 17, fill=AMBER, weight=750),
        text(205, 960, "GPU APC first → managed CPU fallback → cold recomputation only on a true miss", 18, weight=620),
        text(70, 1000, "Shared-host observations · route-isolation resets were test-only · production APC stays enabled · DCM does not extend the 65,536 served limit", 15, fill=MUTED),
        text(1730, 1000, "2026-08-10", 15, fill=MUTED, anchor="end"),
    ])

    svg.append("</svg>")
    OUTPUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
