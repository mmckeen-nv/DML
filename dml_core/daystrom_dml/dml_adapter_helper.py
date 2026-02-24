"""Helper method to format context items for retrieval."""
from typing import List

def _format_context_items(self, items: List) -> str:
    """Format retrieved items into context string."""
    if not items:
        return ""
    lines: List[str] = ["=== Retrieved Context ==="]
    for item in items:
        meta = item.meta or {}
        source = meta.get("source", "unknown")
        timestamp = time.strftime("%Y-%m-%d", time.gmtime(item.timestamp))
        summary = item.cached_summary(max_len=220)
        lines.append(f"- ({timestamp}) [source={source}]\n  {summary}")
    return "\n".join(lines)