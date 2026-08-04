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


class _FakeStore:
    def items(self):
        return []


class _FakeAdapter:
    def __init__(self) -> None:
        self.embedder = _FakeEmbedder()
        self.store = _FakeStore()
        self.storage_dir = "/tmp/dml-mcp-test"
        self.closed_with = None

    def refresh_if_changed(self):
        return False

    def stats(self):
        return {"count": 0}

    def close(self, persist=True):
        self.closed_with = persist


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
