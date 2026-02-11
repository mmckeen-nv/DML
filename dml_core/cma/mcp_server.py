"""Compatibility wrapper for the CMA MCP server."""
from __future__ import annotations

from dml_mcp.cma_mcp_server import create_mcp_server, main, run

__all__ = ["create_mcp_server", "run", "main"]
