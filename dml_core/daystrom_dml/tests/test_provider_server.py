from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from daystrom_dml.provider_server import create_app


@dataclass
class DummyItem:
    id: int
    text: str
    meta: dict
    timestamp: float = field(default_factory=time.time)

    def cached_summary(self, max_len: int = 400) -> str:
        return self.text[:max_len]


class DummyStore:
    def __init__(self, items=None) -> None:
        if items is not None:
            self._items = items
        else:
            self._items = [DummyItem(1, "Provider memory text", {"tenant_id": "openclaw", "source": "unit"})]

    def items(self):
        return list(self._items)


class DummyAdapter:
    def __init__(self, store=None) -> None:
        self.store = store or DummyStore()
        self.ingested = []

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {"count": len(self.store.items()), "storage_dir": "/tmp/dml-provider-test"}

    def ingest(self, text: str, meta: dict | None = None) -> None:
        self.ingested.append((text, meta or {}))

    def retrieve_context(self, query: str, **kwargs) -> dict:
        return {
            "raw_context": "=== Retrieved Context ===\nProvider memory text",
            "context_tokens": 5,
            "items": [
                {
                    "id": "1",
                    "summary": "Provider memory text",
                    "text": "Provider memory text",
                    "meta": {"tenant_id": kwargs.get("tenant_id"), "source": "unit"},
                    "salience": 1.0,
                }
            ],
            "query": query,
        }


def test_provider_server_health_recall_search_and_fetch() -> None:
    app = create_app(adapter_factory=DummyAdapter)
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    recall = client.post("/api/recall", json={"query": "provider", "tenant_id": "openclaw"}).json()
    assert recall["items"][0]["id"] == "1"
    search = client.get("/api/search", params={"q": "provider"}).json()
    assert search["results"][0]["id"] == "1"
    fetched = client.get("/api/fetch/1").json()
    assert fetched["text"] == "Provider memory text"


def test_provider_server_remember() -> None:
    adapter = DummyAdapter()
    app = create_app(adapter_factory=lambda: adapter)
    client = TestClient(app)

    response = client.post(
        "/api/remember",
        json={"text": "Remember this", "tenant_id": "openclaw", "session_id": "s1", "meta": {"source": "unit"}},
    )

    assert response.status_code == 200
    assert adapter.ingested[0][0] == "Remember this"
    assert adapter.ingested[0][1]["session_id"] == "s1"


def test_provider_server_ollama_style_endpoints() -> None:
    app = create_app(adapter_factory=DummyAdapter)
    client = TestClient(app)

    tags = client.get("/api/tags").json()
    assert tags["models"][0]["name"] == "daystrom-dml:memory"
    show = client.post("/api/show", json={"model": "daystrom-dml:memory"}).json()
    assert show["details"]["family"] == "memory-provider"
    generated = client.post("/api/generate", json={"prompt": "provider memory", "model": "daystrom-dml:memory"}).json()
    assert generated["done"] is True
    assert "Provider memory text" in generated["response"]


def test_provider_server_ollama_chat_embed_and_management_endpoints() -> None:
    app = create_app(adapter_factory=DummyAdapter)
    client = TestClient(app)

    assert client.get("/api/version").json()["version"].startswith("dml-ollama-compatible")
    assert client.get("/api/ps").json()["models"] == []
    chat = client.post(
        "/api/chat",
        json={
            "model": "daystrom-dml:memory",
            "messages": [{"role": "user", "content": "provider memory"}],
        },
    ).json()
    assert chat["done"] is True
    assert "Provider memory text" in chat["message"]["content"]
    embeddings = client.post("/api/embeddings", json={"model": "daystrom-dml:memory", "prompt": "hello"}).json()
    assert len(embeddings["embedding"]) == 384
    embed = client.post("/api/embed", json={"model": "daystrom-dml:memory", "input": ["hello", "world"]}).json()
    assert len(embed["embeddings"]) == 2
    assert client.post("/api/pull", json={"model": "daystrom-dml:memory"}).json()["status"] == "success"


def test_provider_frontier_prepare_builds_verifier_prompt() -> None:
    app = create_app(adapter_factory=DummyAdapter)
    client = TestClient(app)

    payload = client.post(
        "/api/frontier/prepare",
        json={
            "prompt": "What should the agent remember?",
            "tenant_id": "openclaw",
            "session_id": "s1",
            "top_k": 4,
            "direct_input_tokens_estimate": 2000,
        },
    ).json()

    assert payload["mode"] == "frontier_with_dml_context"
    assert "Provider memory text" in payload["dml_context"]
    assert "frontier finalizer" in payload["frontier_prompt"]
    assert payload["telemetry"]["frontier_input_tokens"] > 0
    assert payload["telemetry"]["input_tokens_saved_estimate"] > 0


def test_provider_server_root_supports_browser_ui_and_ollama_probe() -> None:
    app = create_app(adapter_factory=DummyAdapter)
    client = TestClient(app)

    assert client.get("/", headers={"accept": "application/json"}).text == "Ollama is running"
    assert "local memory provider" in client.get("/", headers={"accept": "text/html"}).text


def test_provider_fetch_rejects_cross_tenant() -> None:
    items = [
        DummyItem(1, "tenant-a secret", {"tenant_id": "tenant-a", "session_id": "s1", "client_id": "c1"}),
        DummyItem(2, "tenant-b secret", {"tenant_id": "tenant-b"}),
    ]
    store = DummyStore(items=items)
    app = create_app(adapter_factory=lambda: DummyAdapter(store=store))
    client = TestClient(app)

    own = client.get("/api/fetch/1", params={"tenant_id": "tenant-a"})
    assert own.status_code == 404

    own_scoped = client.get("/api/fetch/1", params={"tenant_id": "tenant-a", "session_id": "s1", "client_id": "c1"})
    assert own_scoped.status_code == 200
    assert own_scoped.json()["text"] == "tenant-a secret"

    mismatched_session = client.get("/api/fetch/1", params={"tenant_id": "tenant-a", "session_id": "s2"})
    assert mismatched_session.status_code == 404

    cross = client.get("/api/fetch/1", params={"tenant_id": "tenant-b"})
    assert cross.status_code == 404

    cross_default = client.get("/api/fetch/2", params={"tenant_id": "tenant-a"})
    assert cross_default.status_code == 404


def test_provider_scope_matches_helper() -> None:
    from daystrom_dml.provider_server import _scope_matches

    assert not _scope_matches(
        {"tenant_id": "a", "client_id": "c1"}, tenant_id="a", client_id=None, session_id=None, instance_id=None,
    )
    assert _scope_matches(
        {"tenant_id": "a", "client_id": "c1"}, tenant_id="a", client_id="c1", session_id=None, instance_id=None,
    )
    assert not _scope_matches(
        {"tenant_id": "a"}, tenant_id="b", client_id=None, session_id=None, instance_id=None,
    )
    assert not _scope_matches(
        {"tenant_id": "a", "session_id": "s1"},
        tenant_id="a", client_id=None, session_id="s2", instance_id=None,
    )
    assert not _scope_matches(
        {"tenant_id": "a", "session_id": "s1"},
        tenant_id="a", client_id=None, session_id=None, instance_id=None,
    )
    assert _scope_matches(
        {"tenant_id": "a", "session_id": "s1"},
        tenant_id="a", client_id=None, session_id="s1", instance_id=None,
    )


def test_provider_fetch_enforces_stored_secondary_scope() -> None:
    items = [
        DummyItem(1, "client secret", {"tenant_id": "tenant-a", "client_id": "c1"}),
        DummyItem(2, "session secret", {"tenant_id": "tenant-a", "session_id": "s1"}),
        DummyItem(3, "instance secret", {"tenant_id": "tenant-a", "instance_id": "i1"}),
        DummyItem(4, "shared tenant memory", {"tenant_id": "tenant-a"}),
    ]
    store = DummyStore(items=items)
    app = create_app(adapter_factory=lambda: DummyAdapter(store=store))
    client = TestClient(app)

    # client_id: omitted and mismatch -> 404; exact supplied scope -> 200
    assert client.get("/api/fetch/1", params={"tenant_id": "tenant-a"}).status_code == 404
    assert client.get("/api/fetch/1", params={"tenant_id": "tenant-a", "client_id": "c-other"}).status_code == 404
    ok = client.get("/api/fetch/1", params={"tenant_id": "tenant-a", "client_id": "c1"})
    assert ok.status_code == 200
    assert ok.json()["text"] == "client secret"

    # session_id: omitted and mismatch -> 404; exact supplied scope -> 200
    assert client.get("/api/fetch/2", params={"tenant_id": "tenant-a"}).status_code == 404
    assert client.get("/api/fetch/2", params={"tenant_id": "tenant-a", "session_id": "s-other"}).status_code == 404
    ok = client.get("/api/fetch/2", params={"tenant_id": "tenant-a", "session_id": "s1"})
    assert ok.status_code == 200
    assert ok.json()["text"] == "session secret"

    # instance_id: omitted and mismatch -> 404; exact supplied scope -> 200
    assert client.get("/api/fetch/3", params={"tenant_id": "tenant-a"}).status_code == 404
    assert client.get("/api/fetch/3", params={"tenant_id": "tenant-a", "instance_id": "i-other"}).status_code == 404
    ok = client.get("/api/fetch/3", params={"tenant_id": "tenant-a", "instance_id": "i1"})
    assert ok.status_code == 200
    assert ok.json()["text"] == "instance secret"

    # tenant-wide shared memory (stored secondary scope absent) remains fetchable
    shared = client.get("/api/fetch/4", params={"tenant_id": "tenant-a"})
    assert shared.status_code == 200
    assert shared.json()["text"] == "shared tenant memory"

    # no existence disclosure: omitted-scope not-found matches cross-tenant not-found
    scoped_not_found = client.get("/api/fetch/1", params={"tenant_id": "tenant-a"})
    cross_tenant_not_found = client.get("/api/fetch/1", params={"tenant_id": "tenant-b"})
    assert scoped_not_found.status_code == cross_tenant_not_found.status_code == 404
    assert scoped_not_found.json() == cross_tenant_not_found.json()
