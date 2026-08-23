"""Regression tests for immediate MCP discovery and lazy adapter startup."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from dml_mcp import dml_mcp_server as mcp_server


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, _text: str):
        self.calls += 1
        return [1.0, 0.0]


class _FakeItem:
    def __init__(self, item_id, text, meta=None, timestamp=None):
        self.id = item_id
        self.text = text
        self.meta = meta or {}
        self.timestamp = timestamp or time.time()

    def cached_summary(self, max_len=400):
        return self.text[:max_len]


class _FakeStore:
    def __init__(self, items=None):
        self._items = items or []

    def items(self):
        return list(self._items)


class _FakeAdapter:
    def __init__(self, store_items=None) -> None:
        self.embedder = _FakeEmbedder()
        self.store = _FakeStore(store_items or [])
        self.storage_dir = "/tmp/dml-mcp-test"
        self.closed_with = None
        self.ingested = []

    def refresh_if_changed(self):
        return False

    def stats(self):
        return {"count": len(self.store.items())}

    def ingest(self, text, meta=None, persist=True):
        self.ingested.append((text, meta or {}))

    def close(self, persist=True):
        self.closed_with = persist


def _tool_structured_payload(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    if isinstance(result, dict):
        return result
    raise AssertionError(f"unexpected FastMCP tool result shape: {type(result)!r}")


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_cold_tool_discovery_does_not_construct_adapter(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: calls.append(True) or _FakeAdapter())

    server = mcp_server.create_server()
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {"ingest", "query", "search", "fetch", "stats"}
    assert calls == []


def test_lazy_initialization_is_single_flight(monkeypatch):
    calls = []
    gate = threading.Barrier(8)

    def build(*_args):
        calls.append(time.perf_counter())
        time.sleep(0.02)
        return _FakeAdapter()

    monkeypatch.setattr(mcp_server, "_build_adapter", build)
    holder = mcp_server._LazyAdapter(None, None)

    def get_one(_index):
        gate.wait()
        return holder.get()

    with ThreadPoolExecutor(max_workers=8) as pool:
        adapters = list(pool.map(get_one, range(8)))

    assert len(calls) == 1
    assert len({id(adapter) for adapter in adapters}) == 1


def test_close_waits_for_inflight_preload_and_closes_created_adapter(monkeypatch):
    build_started = threading.Event()
    allow_build = threading.Event()
    adapter = _FakeAdapter()

    def build(*_args):
        build_started.set()
        assert allow_build.wait(timeout=2)
        return adapter

    monkeypatch.setattr(mcp_server, "_build_adapter", build)
    holder = mcp_server._LazyAdapter(None, None)
    holder.start_preload()
    assert build_started.wait(timeout=2)

    closer = threading.Thread(target=holder.close)
    closer.start()
    allow_build.set()
    closer.join(timeout=2)

    assert not closer.is_alive()
    assert adapter.closed_with is False
    with pytest.raises(RuntimeError, match="closed"):
        holder.get()


def test_close_waits_for_active_operation(monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    holder = mcp_server._LazyAdapter(None, None)
    entered = threading.Event()
    release = threading.Event()

    def operate():
        with holder.operation():
            entered.set()
            assert release.wait(timeout=2)

    worker = threading.Thread(target=operate)
    worker.start()
    assert entered.wait(timeout=2)
    closer = threading.Thread(target=lambda: holder.close(timeout=2))
    closer.start()
    time.sleep(0.05)
    assert adapter.closed_with is None
    release.set()
    worker.join(timeout=2)
    closer.join(timeout=2)
    assert not worker.is_alive()
    assert not closer.is_alive()
    assert adapter.closed_with is False


def test_timed_out_close_defers_adapter_close_until_active_operation_finishes(monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    holder = mcp_server._LazyAdapter(None, None)
    entered = threading.Event()
    release = threading.Event()

    def operate():
        with holder.operation():
            entered.set()
            assert release.wait(timeout=2)

    worker = threading.Thread(target=operate)
    worker.start()
    assert entered.wait(timeout=2)
    holder.close(timeout=0.01)
    assert adapter.closed_with is None
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert adapter.closed_with is False


def test_close_is_bounded_when_initialization_stalls(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    adapter = _FakeAdapter()

    def build(*_args):
        started.set()
        assert release.wait(timeout=2)
        return adapter

    monkeypatch.setattr(mcp_server, "_build_adapter", build)
    holder = mcp_server._LazyAdapter(None, None)
    initializer = threading.Thread(target=lambda: pytest.raises(RuntimeError, holder.get))
    initializer.start()
    assert started.wait(timeout=2)
    began = time.monotonic()
    holder.close(timeout=0.05)
    assert time.monotonic() - began < 0.5
    release.set()
    initializer.join(timeout=2)
    assert not initializer.is_alive()
    assert adapter.closed_with is False


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_background_preload_failure_does_not_hide_tools(monkeypatch):
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: (_ for _ in ()).throw(RuntimeError("cold backend")))
    server = mcp_server.create_server()
    holder = getattr(server, "_dml_adapter_holder")

    holder.preload()
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {"ingest", "query", "search", "fetch", "stats"}


def test_scope_matches_helper():
    assert mcp_server._scope_matches(
        {"tenant_id": "tenant-a"},
        tenant_id="tenant-a", client_id=None, session_id=None, instance_id=None,
    )
    assert not mcp_server._scope_matches(
        {"tenant_id": "tenant-a"},
        tenant_id="tenant-b", client_id=None, session_id=None, instance_id=None,
    )
    assert not mcp_server._scope_matches(
        {"tenant_id": "tenant-a", "client_id": "c1"},
        tenant_id="tenant-a", client_id="c2", session_id=None, instance_id=None,
    )
    assert not mcp_server._scope_matches(
        {"tenant_id": "tenant-a", "client_id": "c1"},
        tenant_id="tenant-a", client_id=None, session_id=None, instance_id=None,
    )
    assert mcp_server._scope_matches(
        {"tenant_id": "tenant-a", "client_id": "c1"},
        tenant_id="tenant-a", client_id="c1", session_id=None, instance_id=None,
    )


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_mcp_fetch_rejects_cross_tenant(monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    items = [
        _FakeItem(1, "tenant-a secret", {"tenant_id": "tenant-a"}),
        _FakeItem(2, "tenant-b secret", {"tenant_id": "tenant-b"}),
    ]
    adapter = _FakeAdapter(store_items=items)
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    server = mcp_server.create_server()

    own = asyncio.run(server.call_tool("fetch", {"memory_id": "1", "tenant_id": "tenant-a"}))
    structured = _tool_structured_payload(own)
    assert structured["text"] == "tenant-a secret"
    assert structured["metadata"]["tenant_id"] == "tenant-a"

    with pytest.raises(ToolError, match="not found"):
        asyncio.run(
            server.call_tool(
                "fetch",
                {
                    "memory_id": "1",
                    "tenant_id": "tenant-a",
                    "session_id": "other-session",
                },
            )
        )

    with pytest.raises(ToolError, match="not found"):
        asyncio.run(server.call_tool("fetch", {"memory_id": "1", "tenant_id": "tenant-b"}))


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_mcp_ingest_propagates_scope_into_metadata(monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    server = mcp_server.create_server()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = Path(tmpdir) / "doc.txt"
        doc.write_text("hello world")
        asyncio.run(server.call_tool("ingest", {
            "path": str(doc),
            "tenant_id": "tenant-x",
            "session_id": "sess-1",
        }))

    assert len(adapter.ingested) == 1
    meta = adapter.ingested[0][1]
    assert meta["tenant_id"] == "tenant-x"
    assert meta["session_id"] == "sess-1"


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_mcp_ingest_cross_tenant_isolation(monkeypatch):
    items_a = [_FakeItem(1, "tenant-a file", {"tenant_id": "tenant-a"})]
    adapter = _FakeAdapter(store_items=list(items_a))
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    server = mcp_server.create_server()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = Path(tmpdir) / "doc.txt"
        doc.write_text("tenant-b file content")
        asyncio.run(server.call_tool("ingest", {
            "path": str(doc),
            "tenant_id": "tenant-b",
            "client_id": "client-b",
        }))

    assert adapter.ingested[0][1]["tenant_id"] == "tenant-b"
    assert adapter.ingested[0][1]["client_id"] == "client-b"

    fetched_a = asyncio.run(server.call_tool("fetch", {"memory_id": "1", "tenant_id": "tenant-a"}))
    structured_a = _tool_structured_payload(fetched_a)
    assert structured_a["text"] == "tenant-a file"

    from mcp.server.fastmcp.exceptions import ToolError
    with pytest.raises(ToolError, match="not found"):
        asyncio.run(server.call_tool("fetch", {"memory_id": "1", "tenant_id": "tenant-b"}))


@pytest.mark.skipif(not mcp_server.MCP_AVAILABLE, reason="mcp extra is unavailable")
def test_mcp_fetch_enforces_stored_secondary_scope(monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    items = [
        _FakeItem(1, "client secret", {"tenant_id": "tenant-a", "client_id": "c1"}),
        _FakeItem(2, "session secret", {"tenant_id": "tenant-a", "session_id": "s1"}),
        _FakeItem(3, "instance secret", {"tenant_id": "tenant-a", "instance_id": "i1"}),
        _FakeItem(4, "shared tenant memory", {"tenant_id": "tenant-a"}),
    ]
    adapter = _FakeAdapter(store_items=items)
    monkeypatch.setattr(mcp_server, "_build_adapter", lambda *_args: adapter)
    server = mcp_server.create_server()

    def fetch(memory_id, **kwargs):
        return asyncio.run(
            server.call_tool("fetch", {"memory_id": str(memory_id), **kwargs})
        )

    # client_id: omitted and mismatch -> not found; exact supplied scope -> succeeds
    with pytest.raises(ToolError, match="not found"):
        fetch(1, tenant_id="tenant-a")
    with pytest.raises(ToolError, match="not found"):
        fetch(1, tenant_id="tenant-a", client_id="c-other")
    assert _tool_structured_payload(fetch(1, tenant_id="tenant-a", client_id="c1"))["text"] == "client secret"

    # session_id: omitted and mismatch -> not found; exact supplied scope -> succeeds
    with pytest.raises(ToolError, match="not found"):
        fetch(2, tenant_id="tenant-a")
    with pytest.raises(ToolError, match="not found"):
        fetch(2, tenant_id="tenant-a", session_id="s-other")
    assert _tool_structured_payload(fetch(2, tenant_id="tenant-a", session_id="s1"))["text"] == "session secret"

    # instance_id: omitted and mismatch -> not found; exact supplied scope -> succeeds
    with pytest.raises(ToolError, match="not found"):
        fetch(3, tenant_id="tenant-a")
    with pytest.raises(ToolError, match="not found"):
        fetch(3, tenant_id="tenant-a", instance_id="i-other")
    assert _tool_structured_payload(fetch(3, tenant_id="tenant-a", instance_id="i1"))["text"] == "instance secret"

    # tenant-wide shared memory (stored secondary scope absent) remains fetchable
    assert _tool_structured_payload(fetch(4, tenant_id="tenant-a"))["text"] == "shared tenant memory"

    # no existence disclosure: omitted-scope and cross-tenant both raise ToolError "not found"
    with pytest.raises(ToolError, match="not found"):
        fetch(1, tenant_id="tenant-b")
