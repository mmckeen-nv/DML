from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from daystrom_dml.auth import BearerAuthMiddleware


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/knowledge")
    def knowledge():
        return {"private": True}

    @app.post("/nim/start")
    def nim_start():
        return {"started": True}

    @app.websocket("/visualizer/embed/socket")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("ok")

    return app


def test_auth_is_disabled_when_tokens_are_unconfigured(monkeypatch):
    monkeypatch.delenv("DML_API_TOKEN", raising=False)
    monkeypatch.delenv("DML_ADMIN_TOKEN", raising=False)
    assert TestClient(_app()).get("/knowledge").status_code == 200


def test_public_routes_do_not_require_configured_token(monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "api-secret")
    assert TestClient(_app()).get("/health").status_code == 200


def test_private_route_requires_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "api-secret")
    client = TestClient(_app())

    response = client.get("/knowledge")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/knowledge", headers={"Authorization": "Basic api-secret"}).status_code == 401
    assert client.get("/knowledge", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/knowledge", headers={"Authorization": "Bearer api-secret"}).status_code == 200


def test_admin_token_is_required_for_management_routes(monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "api-secret")
    monkeypatch.setenv("DML_ADMIN_TOKEN", "admin-secret")
    client = TestClient(_app())

    assert client.post("/nim/start", headers={"Authorization": "Bearer api-secret"}).status_code == 403
    assert client.post("/nim/start", headers={"Authorization": "Bearer admin-secret"}).status_code == 200
    assert client.get("/knowledge", headers={"Authorization": "Bearer admin-secret"}).status_code == 200


def test_configured_token_protects_websockets(monkeypatch):
    monkeypatch.setenv("DML_API_TOKEN", "api-secret")
    client = TestClient(_app())

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/visualizer/embed/socket"):
            pass
    assert exc_info.value.code == 4401

    with client.websocket_connect(
        "/visualizer/embed/socket", headers={"Authorization": "Bearer api-secret"}
    ) as websocket:
        assert websocket.receive_text() == "ok"
