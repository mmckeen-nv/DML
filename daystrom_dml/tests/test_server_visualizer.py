from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from fastapi import HTTPException

from daystrom_dml import server


def test_visualizer_redirect_gracefully_handles_launch_failure(monkeypatch):
    def fail_launch() -> None:
        raise HTTPException(status_code=500, detail="pip exploded")

    monkeypatch.setattr(server, "_launch_visualizer_server", fail_launch)

    with TestClient(server.app) as client:
        response = client.get("/visualizer/redirect")

    assert response.status_code == 503
    payload = response.json()
    assert "Visualizer unavailable" in payload["detail"]


def test_visualizer_launch_gracefully_handles_launch_failure(monkeypatch):
    def fail_launch() -> None:
        raise HTTPException(status_code=500, detail="pip exploded")

    monkeypatch.setattr(server, "_launch_visualizer_server", fail_launch)

    with TestClient(server.app) as client:
        response = client.post("/visualizer/launch")

    assert response.status_code == 503
    payload = response.json()
    assert "Visualizer unavailable" in payload["detail"]
