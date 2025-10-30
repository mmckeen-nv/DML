"""Intent routing utilities for retrieval mode selection."""
from __future__ import annotations

from typing import Literal

RetrievalMode = Literal["semantic", "literal", "hybrid"]


def decide_mode(query: str) -> RetrievalMode:
    """Infer the retrieval mode for a query using lightweight heuristics."""
    normalized = query.lower()
    # Detect explicit code and structured query signals.
    strong_literal_triggers = (
        "::",
        "->",
    )
    if any(trigger in query for trigger in strong_literal_triggers):
        return "literal"
    literal_keywords = (
        "select",
        "fetchuserprofile(",
    )
    if any(keyword in normalized for keyword in literal_keywords):
        return "literal"

    semantic_keywords = (
        "average",
        "trend",
        "summarize",
    )
    if any(keyword in normalized for keyword in semantic_keywords):
        return "semantic"

    # Treat parentheses as a literal cue only when no semantic intent was found.
    if "(" in query or ")" in query:
        return "literal"

    return "hybrid"
