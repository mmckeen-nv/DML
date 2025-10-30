"""Intent routing utilities for retrieval mode selection."""
from __future__ import annotations

from typing import Literal

RetrievalMode = Literal["semantic", "literal", "hybrid"]


def decide_mode(query: str) -> RetrievalMode:
    """Infer the retrieval mode for a query using lightweight heuristics."""
    normalized = query.lower()
    # Detect explicit code and structured query signals.
    literal_triggers = (
        "(",
        ")",
        "::",
        "->",
    )
    if any(trigger in query for trigger in literal_triggers):
        return "literal"
    if "select" in normalized:
        return "literal"
    if "fetchuserprofile(" in normalized:
        return "literal"

    semantic_keywords = (
        "average",
        "trend",
        "summarize",
    )
    if any(keyword in normalized for keyword in semantic_keywords):
        return "semantic"

    return "hybrid"
