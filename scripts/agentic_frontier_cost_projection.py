#!/usr/bin/env python3
"""Project frontier API cost for long agentic runs with and without DML.

The projection uses a measured DML compression smoke result as the compression
profile, then applies it to a hypothetical full agentic run. It does not call
paid frontier APIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


MODEL_PRICES_PER_MTOK = [
    {
        "provider": "OpenAI",
        "model": "gpt-5.3-codex",
        "input": 1.75,
        "cached_input": 0.175,
        "output": 14.00,
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    {
        "provider": "OpenAI",
        "model": "gpt-5.4",
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    {
        "provider": "OpenAI",
        "model": "gpt-5.4-mini",
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.50,
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    {
        "provider": "OpenAI",
        "model": "gpt-5.4-nano",
        "input": 0.20,
        "cached_input": 0.02,
        "output": 1.25,
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Opus 4.7",
        "input": 5.00,
        "cached_input": 0.50,
        "output": 25.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Sonnet 4.6",
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
    },
    {
        "provider": "Anthropic",
        "model": "Claude Haiku 4.5",
        "input": 1.00,
        "cached_input": 0.10,
        "output": 5.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
    },
]


def money(value: float) -> str:
    return f"${value:,.2f}"


def cost(input_tokens: float, output_tokens: float, model: dict[str, Any], *, cached_input_share: float = 0.0) -> float:
    cached = input_tokens * cached_input_share
    uncached = max(0.0, input_tokens - cached)
    return (
        (uncached / 1_000_000.0) * float(model["input"])
        + (cached / 1_000_000.0) * float(model["cached_input"])
        + (output_tokens / 1_000_000.0) * float(model["output"])
    )


def load_smoke_profile(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    input_ratio = float(summary["proxy_input_tokens"]) / max(1.0, float(summary["direct_input_tokens"]))
    output_ratio = (
        float(summary["direct_output_tokens_assumed"] - summary["output_token_savings"])
        / max(1.0, float(summary["direct_output_tokens_assumed"]))
        if "direct_output_tokens_assumed" in summary
        else 420.0 / 900.0
    )
    return {
        "input_ratio": input_ratio,
        "output_ratio": output_ratio,
        "source_direct_input_tokens": float(summary["direct_input_tokens"]),
        "source_proxy_input_tokens": float(summary["proxy_input_tokens"]),
        "source_dml_context_recall_pct": float(summary.get("dml_context_recall_pct", 0.0)),
        "source_draft_recall_pct": float(summary.get("draft_recall_pct", 0.0)),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["scenario"]
    lines = [
        "# Agentic Frontier Cost Projection",
        "",
        "This projection does not call paid frontier APIs. It applies the measured DML compression smoke profile to a hypothetical long agentic run.",
        "",
        "## Scenario",
        "",
        f"- Traditional run length: {s['duration_hours']} hours",
        f"- Traditional total tokens: {s['traditional_total_tokens']:,}",
        f"- Assumed direct input tokens: {s['direct_input_tokens']:,}",
        f"- Assumed direct output tokens: {s['direct_output_tokens']:,}",
        f"- DML proxy input tokens: {s['proxy_input_tokens']:,}",
        f"- DML proxy output tokens: {s['proxy_output_tokens']:,}",
        f"- Input compression ratio: {s['input_compression_ratio_pct']}%",
        f"- Output compression ratio: {s['output_compression_ratio_pct']}%",
        "",
        "## Model Cost Table",
        "",
        "| Provider | Model | Direct cost | DML proxy cost | Savings | Output savings | Savings % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["models"]:
        lines.append(
            f"| {row['provider']} | {row['model']} | {money(row['direct_cost'])} | "
            f"{money(row['proxy_cost'])} | {money(row['savings'])} | "
            f"{money(row['output_cost_savings'])} | {row['savings_pct']}% |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- DML proxy cost means frontier verifier/finalizer tokens only. Local model compute is not priced here.",
            "- The projection uses the smoke-test output assumption that verifier/finalizer output is 420 tokens where direct frontier generation would be 900 tokens.",
            "- The local draft is treated as quality assist, not as trusted final output.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["models"]
    width = 1240
    height = 140 + len(rows) * 54
    max_cost = max(row["direct_cost"] for row in rows)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='1240' height='100%' fill='#101314'/>",
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#edf7f2}.muted{fill:#9ba8a5}.small{font-size:14px}.label{font-size:16px;font-weight:750}.title{font-size:32px;font-weight:850}.track{fill:#26302d}.direct{fill:#f87171}.proxy{fill:#68d391}</style>",
        "<text x='42' y='50' class='title'>4-Hour Agentic Run Frontier Cost Projection</text>",
        f"<text x='42' y='80' class='muted small'>Traditional 1.7B tokens · DML proxy input {payload['scenario']['proxy_input_tokens']:,} · DML proxy output {payload['scenario']['proxy_output_tokens']:,}</text>",
        "<text x='42' y='118' class='muted small'>Direct</text><rect x='96' y='107' width='28' height='12' rx='3' class='direct'/>",
        "<text x='144' y='118' class='muted small'>DML proxy</text><rect x='226' y='107' width='28' height='12' rx='3' class='proxy'/>",
    ]
    y = 158
    for row in rows:
        direct_w = max(2, 520 * row["direct_cost"] / max_cost)
        proxy_w = max(2, 520 * row["proxy_cost"] / max_cost)
        lines.extend(
            [
                f"<text x='42' y='{y}' class='label'>{row['model']}</text>",
                f"<text x='42' y='{y + 21}' class='muted small'>{row['provider']} · save {row['savings_pct']}%</text>",
                f"<rect x='330' y='{y - 14}' width='520' height='16' rx='4' class='track'/>",
                f"<rect x='330' y='{y - 14}' width='{direct_w:.1f}' height='16' rx='4' class='direct'/>",
                f"<text x='870' y='{y}' class='small'>{money(row['direct_cost'])}</text>",
                f"<rect x='330' y='{y + 9}' width='520' height='16' rx='4' class='track'/>",
                f"<rect x='330' y='{y + 9}' width='{proxy_w:.1f}' height='16' rx='4' class='proxy'/>",
                f"<text x='870' y='{y + 23}' class='small'>{money(row['proxy_cost'])}</text>",
                f"<text x='1015' y='{y + 8}' class='small'>saves {money(row['savings'])}</text>",
            ]
        )
        y += 54
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project costs for a compressed long agentic run")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "agentic_frontier_cost_projection")
    parser.add_argument("--smoke-results", type=Path, default=REPO_ROOT / "out" / "frontier_compression_smoke" / "results.json")
    parser.add_argument("--traditional-total-tokens", type=int, default=1_700_000_000)
    parser.add_argument("--duration-hours", type=float, default=4.0)
    parser.add_argument("--output-share", type=float, default=0.20)
    parser.add_argument("--cached-input-share", type=float, default=0.0)
    args = parser.parse_args(argv)

    profile = load_smoke_profile(args.smoke_results)
    total = int(args.traditional_total_tokens)
    direct_output = int(total * args.output_share)
    direct_input = total - direct_output
    proxy_input = int(direct_input * profile["input_ratio"])
    proxy_output = int(direct_output * profile["output_ratio"])
    models = []
    for model in MODEL_PRICES_PER_MTOK:
        direct = cost(direct_input, direct_output, model, cached_input_share=args.cached_input_share)
        proxy = cost(proxy_input, proxy_output, model, cached_input_share=args.cached_input_share)
        direct_output_cost = (direct_output / 1_000_000.0) * float(model["output"])
        proxy_output_cost = (proxy_output / 1_000_000.0) * float(model["output"])
        output_cost_savings = direct_output_cost - proxy_output_cost
        savings = direct - proxy
        models.append(
            {
                "provider": model["provider"],
                "model": model["model"],
                "input_price_per_mtok": model["input"],
                "cached_input_price_per_mtok": model["cached_input"],
                "output_price_per_mtok": model["output"],
                "direct_cost": round(direct, 2),
                "proxy_cost": round(proxy, 2),
                "direct_output_cost": round(direct_output_cost, 2),
                "proxy_output_cost": round(proxy_output_cost, 2),
                "output_cost_savings": round(output_cost_savings, 2),
                "output_cost_savings_pct": round((output_cost_savings / direct_output_cost) * 100, 1)
                if direct_output_cost
                else 0.0,
                "savings": round(savings, 2),
                "savings_pct": round((savings / direct) * 100, 1) if direct else 0.0,
                "source": model["source"],
            }
        )
    payload = {
        "scenario": {
            "duration_hours": args.duration_hours,
            "traditional_total_tokens": total,
            "direct_input_tokens": direct_input,
            "direct_output_tokens": direct_output,
            "proxy_input_tokens": proxy_input,
            "proxy_output_tokens": proxy_output,
            "input_compression_ratio_pct": round(profile["input_ratio"] * 100, 2),
            "output_compression_ratio_pct": round(profile["output_ratio"] * 100, 2),
            "cached_input_share": args.cached_input_share,
            "smoke_dml_context_recall_pct": profile["source_dml_context_recall_pct"],
            "smoke_local_draft_recall_pct": profile["source_draft_recall_pct"],
        },
        "models": models,
        "price_sources": sorted({model["source"] for model in MODEL_PRICES_PER_MTOK}),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / "README.md", payload)
    write_svg(output_dir / "results.svg", payload)
    print(json.dumps({"output_dir": str(output_dir), "scenario": payload["scenario"], "models": models}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
