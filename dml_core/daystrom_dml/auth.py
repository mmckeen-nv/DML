"""Authentication middleware for the DML HTTP and WebSocket API."""
from __future__ import annotations

import hmac
import os
from collections.abc import Iterable

from starlette.responses import JSONResponse


PUBLIC_PATHS = frozenset({"/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json"})
PUBLIC_PREFIXES = ("/static/",)
ADMIN_PATHS = frozenset({"/visualizer/launch"})
ADMIN_PREFIXES = ("/nim/",)


def _bearer_token(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """Return a strictly parsed bearer credential from ASGI headers."""

    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        try:
            scheme, token = value.decode("latin-1").split(" ", 1)
        except ValueError:
            return None
        if scheme.lower() != "bearer" or not token or token != token.strip():
            return None
        return token
    return None


def _matches(candidate: str | None, expected: str | None) -> bool:
    """Compare non-empty credentials without leaking comparison timing."""

    return bool(candidate and expected) and hmac.compare_digest(candidate, expected)


class BearerAuthMiddleware:
    """Protect DML routes when an API or administrator token is configured.

    ``DML_API_TOKEN`` enables authentication. ``DML_ADMIN_TOKEN`` may be set to
    a distinct credential for management routes; when it is the only configured
    token, it protects the entire API. If no token is configured, authentication
    remains disabled for backwards-compatible local-only operation.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in PUBLIC_PATHS or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return

        api_token = os.environ.get("DML_API_TOKEN") or None
        admin_token = os.environ.get("DML_ADMIN_TOKEN") or None
        if api_token is None and admin_token is None:
            await self.app(scope, receive, send)
            return

        supplied = _bearer_token(scope.get("headers", ()))
        is_admin = _matches(supplied, admin_token)
        authenticated = is_admin or _matches(supplied, api_token)
        management = path in ADMIN_PATHS or any(path.startswith(prefix) for prefix in ADMIN_PREFIXES)
        status = 403 if authenticated and management and admin_token and not is_admin else 401

        if not authenticated or status == 403:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403 if status == 403 else 4401})
            else:
                response = JSONResponse(
                    {"detail": "Administrator authorization required" if status == 403 else "Authentication required"},
                    status_code=status,
                    headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
                )
                await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
