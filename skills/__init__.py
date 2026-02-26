"""OpenClaw integration for Daystrom Memory Lattice (DML)."""

from typing import Any, Dict, List, Optional
import sys
sys.path.insert(0, '/home/nvidia/.npm-global/lib/node_modules/openclaw/node_modules/daystrom-dml')

from dml_core.daystrom_dml.dml_adapter import DMLAdapter
from dml_core.daystrom_dml.agent_schema import MemoryKind


class DMLAgent:
    """OpenClaw agent with DML memory integration."""

    def __init__(
        self,
        config_path: str = None,
        *,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize DML agent.

        Args:
            config_path: Optional path to DML config file
            config_overrides: Optional dict of config overrides
        """
        self.adapter = DMLAdapter(
            config_path=config_path,
            config_overrides=config_overrides or {}
        )
        self._initialized = True

    def ingest(
        self,
        text: str,
        kind: str = "action",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Ingest new information into DML memory.

        Args:
            text: Text to store
            kind: Memory kind (action, observation, insight, etc.)
            meta: Optional metadata dictionary

        Returns:
            Ingestion result
        """
        kind_map = {
            "action": MemoryKind.ACTION,
            "observation": MemoryKind.OBSERVATION,
            "insight": MemoryKind.INSIGHT,
            "planning": MemoryKind.PLANNING,
            "execution": MemoryKind.EXECUTION,
            "result": MemoryKind.RESULT,
        }
        memory_kind = kind_map.get(kind.lower(), MemoryKind.ACTION)

        result = self.adapter.ingest(text, meta={**meta or {}, "kind": memory_kind.name})
        return result

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Retrieve context from DML memory.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            Retrieval result with context and metadata
        """
        result = self.adapter.retrieve_context(query, top_k=top_k)
        return result

    def get_context(
        self,
        query: str,
        max_tokens: int = 1000,
    ) -> str:
        """Get formatted context string for LLM prompts.

        Args:
            query: Query string
            max_tokens: Maximum context tokens

        Returns:
            Formatted context string
        """
        report = self.adapter.retrieve_context(query)
        raw_context = report.get('raw_context', '')

        # Truncate to max tokens
        estimated_tokens = len(raw_context) // 4
        if estimated_tokens > max_tokens:
            raw_context = raw_context[: max_tokens * 4]

        return raw_context

    def memory_count(self) -> int:
        """Get total number of memories stored."""
        # Check adapter methods
        if hasattr(self.adapter, 'get_stm_state'):
            try:
                state = self.adapter.get_stm_state()
                return state.get('total_memories', 0)
            except RuntimeError:
                pass
        return 0

    def shutdown(self) -> None:
        """Clean up DML adapter."""
        if self._initialized:
            self.adapter.close()
            self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def dml_retrieve(query: str, top_k: int = 4) -> Dict[str, Any]:
    """Quick function to retrieve context from DML.

    Args:
        query: Query string
        top_k: Number of results to return

    Returns:
        Retrieval result
    """
    adapter = DMLAdapter()
    try:
        return adapter.retrieve_context(query, top_k=top_k)
    finally:
        adapter.close()


def dml_ingest(text: str, kind: str = "action", meta: Optional[Dict[str, Any]] = None) -> Any:
    """Quick function to ingest into DML.

    Args:
        text: Text to store
        kind: Memory kind
        meta: Optional metadata

    Returns:
        Ingestion result
    """
    adapter = DMLAdapter()
    try:
        kind_map = {
            "action": MemoryKind.ACTION,
            "observation": MemoryKind.OBSERVATION,
            "insight": MemoryKind.INSIGHT,
            "planning": MemoryKind.PLANNING,
            "execution": MemoryKind.EXECUTION,
            "result": MemoryKind.RESULT,
        }
        memory_kind = kind_map.get(kind.lower(), MemoryKind.ACTION)
        return adapter.ingest(text, meta={**meta or {}, "kind": memory_kind.name})
    finally:
        adapter.close()


__all__ = ["DMLAgent", "dml_retrieve", "dml_ingest"]