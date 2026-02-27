#!/usr/bin/env python3
"""
DML (Daystrom Memory Lattice) Integration for OpenClaw
This script provides utilities for integrating DML into the agent's workflow.
"""

import sys
from typing import Any, Dict, Optional

# Add DML lib to path
sys.path.insert(0, '/home/nvidia/.npm-global_lib/node_modules/openclaw/node_modules/daystrom-dml')

from skills.daystrom_dml import DMLAgent


class DMLManager:
    """Manager for DML (Daystrom Memory Lattice) integration."""

    def __init__(self):
        """Initialize DML manager with GPU support."""
        self.dml = DMLAgent()
        self._initialized = True

    def ingest_memory(
        self,
        text: str,
        kind: str = "action",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Ingest memory into DML.

        Args:
            text: Memory content
            kind: Memory kind (action, observation, insight, preference, etc.)
            meta: Optional metadata

        Returns:
            Ingestion result
        """
        if not self._initialized:
            raise RuntimeError("DML not initialized")

        result = self.dml.ingest(text, kind, meta)
        return result

    def get_context(self, query: str, max_tokens: int = 1000) -> str:
        """Get context for decision-making.

        Args:
            query: Query string
            max_tokens: Maximum tokens

        Returns:
            Formatted context
        """
        if not self._initialized:
            raise RuntimeError("DML not initialized")

        context = self.dml.get_context(query, max_tokens)
        return context

    def memory_count(self) -> int:
        """Get total memory count."""
        if not self._initialized:
            return 0
        return self.dml.memory_count()

    def shutdown(self) -> None:
        """Shutdown DML."""
        if self._initialized:
            self.dml.shutdown()
            self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def dml_summary() -> str:
    """Get DML usage summary."""
    manager = DMLManager()
    count = manager.memory_count()
    manager.shutdown()
    return f"DML has {count} memories stored"


# Global DML manager instance (for quick access)
_dml_manager: Optional[DMLManager] = None


def get_dml() -> DMLManager:
    """Get or create global DML manager.

    Note: This creates a new instance each call. For persistent sessions,
    use DMLManager with context manager or manage lifecycle manually.
    """
    global _dml_manager
    if _dml_manager is None:
        _dml_manager = DMLManager()
    return _dml_manager


def reset_dml() -> None:
    """Reset global DML manager (for testing)."""
    global _dml_manager
    if _dml_manager:
        _dml_manager.shutdown()
    _dml_manager = None