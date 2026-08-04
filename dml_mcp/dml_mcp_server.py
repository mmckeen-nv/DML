"""Expose the Daystrom Memory Lattice via the Model Context Protocol."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, TYPE_CHECKING

from daystrom_dml.dml_adapter import DMLAdapter
from daystrom_dml.metrics import record_operation

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from mcp.server.fastmcp import FastMCP as FastMCPType
else:  # pragma: no cover - runtime import guard
    FastMCPType = Any

try:  # pragma: no cover - optional dependency
    from mcp.server.fastmcp import FastMCP as _FastMCP
except Exception:  # pragma: no cover - best effort import guard
    _FastMCP = None
    MCP_AVAILABLE = False
else:
    MCP_AVAILABLE = True


def _build_adapter(
    config_path: Path | None,
    storage_dir: Path | None,
) -> DMLAdapter:
    overrides: dict[str, Any] = {"persistence": {"interval_sec": 0}}
    if storage_dir:
        overrides["storage_dir"] = str(storage_dir)
    return DMLAdapter(
        config_path=str(config_path) if config_path else None,
        config_overrides=overrides,
        start_aging_loop=False,
    )


class _LazyAdapter:
    """Single-flight construction plus leased, serialized adapter operations."""

    def __init__(self, config_path: Path | None, storage_dir: Path | None) -> None:
        self._config_path = config_path
        self._storage_dir = storage_dir
        self._condition = threading.Condition()
        self._operation_lock = threading.RLock()
        self._adapter: DMLAdapter | None = None
        self._preload_thread: threading.Thread | None = None
        self._initializing = False
        self._active_operations = 0
        self._closed = False

    def get(self) -> DMLAdapter:
        with self._condition:
            while self._initializing and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("DML MCP adapter holder is closed")
            if self._adapter is not None:
                return self._adapter
            self._initializing = True

        adapter: DMLAdapter | None = None
        try:
            started = time.perf_counter()
            adapter = _build_adapter(self._config_path, self._storage_dir)
            latency_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info("DML MCP adapter initialized latency_ms=%.2f", latency_ms)
            record_operation("adapter_initialization", latency_ms=latency_ms)
        finally:
            with self._condition:
                self._initializing = False
                if adapter is not None and not self._closed:
                    self._adapter = adapter
                self._condition.notify_all()

        if adapter is None:
            raise RuntimeError("DML MCP adapter initialization failed")
        with self._condition:
            if self._closed:
                adapter.close(persist=False)
                raise RuntimeError("DML MCP adapter holder is closed")
        return adapter

    def preload(self) -> None:
        """Construct and warm the adapter without affecting registered tools on failure."""

        try:
            adapter = self.get()
            started = time.perf_counter()
            adapter.embedder.embed("daystrom dml readiness")
            latency_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info("DML MCP embedding warm-up complete latency_ms=%.2f", latency_ms)
            record_operation("embedding_warmup", latency_ms=latency_ms)
        except Exception:
            LOGGER.exception("DML MCP background preload failed; tools remain registered")
            record_operation("adapter_preload_failure")

    def start_preload(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._preload_thread and self._preload_thread.is_alive():
                return
            self._preload_thread = threading.Thread(
                target=self.preload,
                name="dml-mcp-preload",
                daemon=True,
            )
            self._preload_thread.start()

    @contextmanager
    def operation(self, *, mutate: bool = False) -> Iterator[DMLAdapter]:
        del mutate  # mutation locking belongs to DMLAdapter's durable boundary
        adapter = self.get()
        with self._condition:
            if self._closed:
                raise RuntimeError("DML MCP adapter holder is closed")
            self._active_operations += 1
        try:
            # MemoryStore and FAISS are not safe for concurrent refresh/mutation.
            with self._operation_lock:
                adapter.refresh_if_changed()
                yield adapter
        finally:
            deferred_close: DMLAdapter | None = None
            with self._condition:
                self._active_operations -= 1
                if self._closed and self._active_operations == 0 and not self._initializing:
                    deferred_close = self._adapter
                    self._adapter = None
                self._condition.notify_all()
            if deferred_close is not None:
                deferred_close.close(persist=False)

    def close(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while (self._initializing or self._active_operations) and time.monotonic() < deadline:
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            if self._initializing or self._active_operations:
                LOGGER.warning(
                    "DML MCP shutdown timed out; leaving adapter open for active workers "
                    "initializing=%s active_operations=%s",
                    self._initializing,
                    self._active_operations,
                )
                return
            adapter = self._adapter
            self._adapter = None
        if adapter is not None:
            adapter.close(persist=False)


def create_server(
    *,
    config_path: Path | None = None,
    storage_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCPType:
    """Register MCP tool schemas immediately and initialize DML lazily."""

    registration_started = time.perf_counter()
    if not MCP_AVAILABLE:  # pragma: no cover - import guard
        raise RuntimeError("mcp extras are not installed; install with '.[mcp]'")
    if _FastMCP is None:  # pragma: no cover - defensive import guard
        raise RuntimeError("FastMCP runtime is unavailable")

    holder = _LazyAdapter(config_path, storage_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastMCPType) -> Any:
        holder.start_preload()
        try:
            yield
        finally:
            await asyncio.to_thread(holder.close)

    server = _FastMCP(
        name="daystrom-dml",
        instructions="Augment prompts with Daystrom Memory Lattice context",
        host=host,
        port=port,
        lifespan=lifespan,
    )

    @server.tool(name="ingest", description="Explicitly ingest a document file or directory")
    async def ingest(path: str) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            target = Path(path).expanduser()
            if not target.exists():
                raise ValueError(f"Path does not exist: {target}")
            files = list(_iter_ingest_targets(target))
            if not files:
                raise ValueError(f"No ingestible files found in {target}")
            count = 0
            with holder.operation(mutate=True) as adapter:
                for file_path in files:
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue
                    adapter.ingest(text, meta={"doc_path": str(file_path)})
                    count += 1
            return {"files": count, "target": str(target)}

        return await asyncio.to_thread(_run)

    @server.tool(name="query", description="Query the lattice and receive structured context")
    async def query(prompt: str, mode: str = "auto") -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            with holder.operation() as adapter:
                report = adapter.query_database(prompt, mode=mode or "auto")
            return {
                "mode": report["mode"],
                "context": report["context"],
                "tokens": int(report.get("tokens", 0)),
                "latency_ms": int(report.get("latency_ms", 0)),
                "sources": report.get("source_docs", []),
                "selected_source_count": int(report.get("selected_source_count", 0)),
                "expanded_context_chars": int(report.get("expanded_context_chars", 0)),
            }

        return await asyncio.to_thread(_run)

    @server.tool(name="search", description="Search DML memory and return result handles")
    async def search(query: str, tenant_id: str = "openclaw", session_id: str | None = None, top_k: int = 6) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            with holder.operation() as adapter:
                report = adapter.retrieve_context(query, tenant_id=tenant_id, session_id=session_id, top_k=top_k)
            results = []
            for item in report.get("items", []):
                meta = item.get("meta") or {}
                results.append(
                    {
                        "id": str(item.get("id") or ""),
                        "title": meta.get("source") or f"memory:{item.get('id')}",
                        "snippet": item.get("summary") or item.get("text") or "",
                        "metadata": meta,
                    }
                )
            return {"query": query, "results": results}

        return await asyncio.to_thread(_run)

    @server.tool(name="fetch", description="Fetch one DML memory by id")
    async def fetch(memory_id: str) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            with holder.operation() as adapter:
                for item in adapter.store.items():
                    if str(item.id) == str(memory_id):
                        return {
                            "id": str(item.id),
                            "text": item.text,
                            "summary": item.cached_summary(max_len=400),
                            "metadata": item.meta or {},
                            "timestamp": float(item.timestamp),
                        }
            raise ValueError(f"Memory not found: {memory_id}")

        return await asyncio.to_thread(_run)

    @server.tool(name="stats", description="Return basic adapter statistics")
    async def stats() -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            with holder.operation() as adapter:
                return adapter.stats()

        return await asyncio.to_thread(_run)

    registration_ms = (time.perf_counter() - registration_started) * 1000.0
    LOGGER.info("DML MCP tools registered latency_ms=%.2f", registration_ms)
    record_operation("mcp_tool_registration", latency_ms=registration_ms)
    setattr(server, "_dml_adapter_holder", holder)
    return server


def _iter_ingest_targets(root: Path) -> Iterable[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md", ".log", ".json"}:
            files.append(candidate)
    return files


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DML MCP server")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file override")
    parser.add_argument("--storage", type=Path, default=None, help="Persistent storage directory override")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to expose",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for HTTP transports")
    args = parser.parse_args(argv)

    server = create_server(
        config_path=args.config,
        storage_dir=args.storage,
        host=args.host,
        port=args.port,
    )
    server.run(transport=args.transport)  # type: ignore[union-attr]


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
