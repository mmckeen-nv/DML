"""Pure bounded memory-context assembly, including all rendered framing."""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Sequence

import numpy as np

from ..memory_store import MemoryItem
from .telemetry import retrieval_decision


class ContextBudgetError(ValueError):
    """Required context cannot fit the caller's supplied token budget."""


def compact_context(items: Sequence[MemoryItem], *, budget: int, item_limit: int,
                    summary_chars: int, ledger_chars: int,
                    count_tokens: Callable[[str], int], prefix: str = "") -> tuple[list[dict], str, int]:
    def count(text):
        value = count_tokens(text) if text else 0
        if type(value) is not int or value < 0:
            raise ContextBudgetError("Token counter must return a nonnegative integer")
        return value

    if count(prefix) > budget:
        raise ContextBudgetError("Context prefix exceeds token budget")
    lines = ([prefix] if prefix else []) + ["=== Retrieved Context ==="]
    entries: list[dict] = []
    for item in items[:item_limit]:
        meta = copy.deepcopy(item.meta or {})
        max_len = max(summary_chars, ledger_chars) if meta.get("kind") == "survival_ledger" else summary_chars
        summary = item.cached_summary(max_len=max_len)
        source = meta.get("source", "unknown")
        timestamp = time.strftime("%Y-%m-%d", time.gmtime(item.timestamp))
        def rendered(text):
            return "\n".join(lines + [f"- ({timestamp}) [source={source}]\n  {text}"])
        # Preserve the leading excerpt when a first item does not fit. Count the
        # actual result every time; never clamp a reported count to the budget.
        if count(rendered(summary)) > budget:
            if entries:
                break
            while summary and count(rendered(summary)) > budget:
                summary = summary[:len(summary) // 2].rstrip()
            if not summary:
                continue
        lines.append(f"- ({timestamp}) [source={source}]\n  {summary}")
        entries.append({"id": str(item.id), "text": summary, "summary": summary,
                        "meta": meta, "timestamp": float(item.timestamp), "level": item.level,
                        "fidelity": float(item.fidelity), "salience": float(item.salience),
                        "tokens": count(summary)})
    context = "\n".join(lines) if entries else prefix
    return entries, context, count(context)


def build_context_report(
    *,
    entries: list[dict[str, Any]],
    context: str,
    tokens_used: int,
    prompt: str,
    scope: dict[str, str | None],
    revision: int | None,
    as_of: float,
    top_k: int,
    kinds: list[str] | None,
    embedding: np.ndarray,
    suppressed: list[dict[str, str]],
    replayable: bool,
    phase: str | None,
    include_quarantined: bool,
    survival_ledger_included: bool,
    personality_overlay: dict[str, Any] | None,
    latency_ms: int,
) -> dict[str, Any]:
    """Build context evidence and a detached compatibility report.

    All inputs are borrowed read-only. Selection, lifecycle decisions, rendering,
    token counting, revision ownership, and timing remain the caller's work.
    The returned report owns copies of mutable input data, including nested item
    metadata; it neither retains the query vector nor reads runtime state.
    Digests describe the values at construction time. The returned dictionaries
    remain mutable for compatibility, and later edits do not recompute evidence.
    """
    report_entries = copy.deepcopy(entries)
    report_scope = copy.deepcopy(scope)
    report_kinds = copy.deepcopy(kinds)
    report_suppressed = copy.deepcopy(suppressed)
    evidence = retrieval_decision(
        prompt=prompt,
        scope=report_scope,
        revision=revision,
        as_of=as_of,
        entries=report_entries,
        context=context,
        top_k=top_k,
        kinds=report_kinds,
        embedding=embedding,
        suppressed=report_suppressed,
        replayable=replayable,
    )
    return {
        "decision": evidence,
        "token_count_kind": "estimate",
        "raw_context": context,
        "context_tokens": tokens_used,
        "top_k": top_k,
        "kinds": report_kinds,
        "phase": phase,
        "include_quarantined": include_quarantined,
        "items": report_entries,
        "survival_ledger_included": survival_ledger_included,
        "personality_overlay": copy.deepcopy(personality_overlay),
        "latency_ms": latency_ms,
    }
