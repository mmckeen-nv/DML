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
    def __init__(self) -> None:
        self._items = [DummyItem(1, "Provider memory text", {"tenant_id": "openclaw", "source": "unit"})]

    def items(self):
        return list(self._items)


class DummyAdapter:
    def __init__(self) -> None:
        self.store = DummyStore()
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
