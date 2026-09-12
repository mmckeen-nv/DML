"""Pure bounded memory-context assembly, including all rendered framing."""
from __future__ import annotations

import copy
import time
from typing import Callable, Sequence

from ..memory_store import MemoryItem


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
