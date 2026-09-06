"""Reusable connection-pooled client for scoped provider workloads."""
from __future__ import annotations

import os
import httpx


class ProviderClient:
    def __init__(self, base_url="http://127.0.0.1:8765", *, token=None, timeout=30.0, transport=None):
        credential = token if token is not None else os.environ.get("DML_API_TOKEN") or os.environ.get("DML_ADMIN_TOKEN")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {credential}"} if credential else {},
        )

    def _post(self, path, payload):
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def remember(self, text, *, tenant_id, **scope_and_metadata):
        return self._post("/api/remember", {**scope_and_metadata, "text": text, "tenant_id": tenant_id})

    def remember_many(self, records, *, batch_size=64):
        """Commit one bounded batch atomically; larger workloads choose their own checkpoints."""
        return self._post("/api/remember/batch", {"records": records, "batch_size": batch_size})

    def recall(self, query, *, tenant_id, **scope_and_options):
        return self._post("/api/recall", {**scope_and_options, "query": query, "tenant_id": tenant_id})

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
