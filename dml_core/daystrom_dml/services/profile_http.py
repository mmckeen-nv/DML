"""Closed HTTP surface for the explicitly selected receipted local profile."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import os
import re
import sqlite3
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from ..auth import _bearer_token
from ..contracts.profile import (
    PROFILE_ID, production_profile_contract, validate_profile_authority,
    validate_profile_config,
)
from ..contracts.retention import retention_contract
from ..journal import IdempotencyConflict, JournalIntegrityError, RevisionConflict
from ..settings import DMLSettings
from .receipt_ingestion import (
    ReceiptCapacityError, ReceiptCommitRejected, ReceiptCommitUncertain,
    ReceiptEmbeddingError,
)
from .receipt_lifecycle import ReceiptLifecycleConflict, ReceiptMemoryNotFound
from .retention import RetentionInspectionUnsupported


class ScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: StrictStr = Field(min_length=1, max_length=256)
    client_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    session_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    instance_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("tenant_id", "client_id", "session_id", "instance_id")
    @classmethod
    def scope_bytes(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or len(value.encode("utf-8")) > 256):
            raise ValueError("Invalid scope")
        return value


class RecallRequest(ScopeRequest):
    query: StrictStr = Field(min_length=1, max_length=1024 * 1024)
    top_k: StrictInt = Field(default=6, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_bytes(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Invalid query")
        return value


class ReceiptRequest(ScopeRequest):
    idempotency_key: StrictStr = Field(min_length=1, max_length=256)

    @field_validator("idempotency_key")
    @classmethod
    def key_bytes(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > 256:
            raise ValueError("Invalid idempotency key")
        return value


class RememberRequest(ReceiptRequest):
    text: StrictStr = Field(min_length=1, max_length=1024 * 1024)
    kind: StrictStr = Field(default="note", min_length=1, max_length=256)
    meta: dict[str, Any] = Field(default_factory=dict)


class RetentionRequest(ScopeRequest):
    memory_id: StrictInt = Field(ge=0)


class RetireRequest(ReceiptRequest):
    memory_id: StrictInt = Field(ge=0)
    expected_memory_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    reason: StrictStr = Field(min_length=1, max_length=1024)


class SupersedeRequest(RetireRequest):
    replacement_memory_id: StrictInt = Field(ge=0)
    expected_replacement_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class UpdateRequest(RetireRequest):
    text: StrictStr = Field(min_length=1, max_length=1024 * 1024)


class PromotionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_id: StrictInt = Field(ge=0)
    expected_memory_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class PromoteRequest(ReceiptRequest):
    sources: list[PromotionSource] = Field(min_length=1, max_length=32)
    text: StrictStr = Field(min_length=1, max_length=1024 * 1024)
    reason: StrictStr = Field(min_length=1, max_length=1024)


def profile_credentials() -> tuple[str, ...]:
    """Freeze credentials before constructing an adapter or invoking a factory."""
    tokens = tuple(value for name in ("DML_API_TOKEN", "DML_ADMIN_TOKEN")
                   if (value := os.environ.get(name)) is not None)
    if not tokens or any(re.fullmatch(r"[A-Za-z0-9._~+/\-]+=*", token) is None for token in tokens):
        raise ValueError("Production profile requires a nonempty HTTP bearer DML_API_TOKEN or DML_ADMIN_TOKEN")
    return tokens


class ProfileAuthMiddleware:
    """Authenticate against startup credentials; environment changes cannot open access."""

    def __init__(self, app, *, tokens: tuple[str, ...]):
        self.app = app
        self.tokens = tuple(token.encode("utf-8") for token in tokens)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return
        supplied = _bearer_token(scope.get("headers", ()))
        if not supplied or not any(hmac.compare_digest(supplied.encode("utf-8"), token) for token in self.tokens):
            await JSONResponse(status_code=401, content={"detail": {"code": "authentication_required"}},
                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _fail(status: int, code: str, *, retry_same_key: bool = False) -> HTTPException:
    detail: dict[str, Any] = {"code": code}
    if retry_same_key:
        detail["retry_same_key"] = True
    return HTTPException(status_code=status, detail=detail)


def _invoke(operation: Callable, arguments: dict[str, Any], *, mode: str = "receipt") -> dict[str, Any]:
    """Return stable, payload-free errors without changing successful receipt envelopes."""
    receipt = mode == "receipt"
    try:
        result = operation(**arguments)
        try:
            if not isinstance(result, dict):
                raise TypeError("Profile operations return object envelopes")
            # Validate and freeze the response within the retry boundary, including
            # UTF-8 and finite numbers, before FastAPI attempts serialization.
            encoded = json.dumps(result, allow_nan=False, ensure_ascii=False).encode("utf-8")
            return json.loads(encoded)
        except Exception as exc:
            raise RuntimeError("Profile response serialization failed") from exc
    except ReceiptMemoryNotFound as exc:
        raise _fail(404, "receipt_memory_not_found") from exc
    except RetentionInspectionUnsupported as exc:
        raise _fail(409, "retention_inspection_unsupported") from exc
    except ReceiptLifecycleConflict as exc:
        raise _fail(409, "receipt_lifecycle_conflict") from exc
    except ReceiptCapacityError as exc:
        raise _fail(409, "receipt_capacity_exceeded") from exc
    except IdempotencyConflict as exc:
        raise _fail(409, "idempotency_conflict") from exc
    except RevisionConflict as exc:
        raise _fail(409, "revision_conflict", retry_same_key=receipt) from exc
    except TimeoutError as exc:
        raise _fail(503, "receipt_ownership_unavailable" if receipt else f"{mode}_outcome_unavailable",
                    retry_same_key=receipt) from exc
    except ReceiptCommitRejected as exc:
        raise _fail(503, "receipt_not_committed", retry_same_key=receipt) from exc
    except ReceiptEmbeddingError as exc:
        raise _fail(503, "embedding_unavailable", retry_same_key=receipt) from exc
    except (ReceiptCommitUncertain, JournalIntegrityError, OSError, sqlite3.Error) as exc:
        raise _fail(503, f"{mode}_outcome_unavailable", retry_same_key=receipt) from exc
    except ValueError as exc:
        raise _fail(400, f"invalid_or_unsupported_{mode}_request") from exc
    except Exception as exc:
        # A lost acknowledgement must never instruct the caller to choose a new key.
        raise _fail(503, f"{mode}_outcome_unavailable", retry_same_key=receipt) from exc


def build_profile_app(
    adapter: Any, *, tokens: tuple[str, ...], expected_settings: DMLSettings | None = None,
) -> FastAPI:
    """Build only admitted routes; omitted experimental handlers never initialize."""
    if getattr(adapter, "production_profile_id", None) != PROFILE_ID:
        raise ValueError("Production profile adapter does not match the selected profile")
    status = adapter.production_profile_status()
    if (not isinstance(status, dict) or status.get("profile_id") != PROFILE_ID
            or status.get("production_ready") is not False
            or status.get("validated") is not True or status.get("status") != "candidate"):
        raise ValueError("Production profile adapter status is invalid")
    authority = status.get("authority")
    identity = status.get("embedding_identity")
    if not isinstance(authority, dict) or not isinstance(identity, dict):
        raise ValueError("Production profile adapter admission evidence is missing")
    validate_profile_authority(PROFILE_ID, authority.get("journal_schema_version"), authority.get("outbox_enabled"))
    revision = identity.get("revision")
    if (identity.get("mode") != "native" or type(revision) is not str
            or not revision.strip() or len(revision.encode("utf-8")) > 1024):
        raise ValueError("Production profile adapter embedding admission is invalid")
    if expected_settings is not None:
        actual_settings = getattr(adapter, "settings", None)
        if not isinstance(actual_settings, DMLSettings):
            raise ValueError("Production profile factory must expose validated settings")
        if actual_settings.as_dict() != expected_settings.as_dict():
            raise ValueError("Production profile factory settings do not match provider configuration")
        validate_profile_config(actual_settings.as_dict())
        if (authority["outbox_enabled"] != expected_settings.persistence.outbox
                or revision != expected_settings.persistence.receipt_embedding_identity):
            raise ValueError("Production profile factory authority does not match provider configuration")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            adapter.close()

    app = FastAPI(title="Daystrom receipted local candidate", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ProfileAuthMiddleware, tokens=tokens)
    app.state.adapter = adapter

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": {"code": "profile_validation_failed"}})

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            durability = adapter.durability_status()
            healthy = isinstance(durability, dict) and durability.get("status") == "ok"
        except Exception:
            healthy = False
        return {"status": "ok" if healthy else "degraded", "profile_id": PROFILE_ID,
                "maturity": "candidate", "production_ready": False}

    @app.get("/api/contracts")
    def contracts() -> dict[str, Any]:
        return {**production_profile_contract(), "retention": retention_contract()}

    @app.post("/api/remember/receipt")
    def remember(payload: RememberRequest) -> dict[str, Any]:
        return _invoke(adapter.ingest_memory_receipted, payload.model_dump())

    @app.post("/api/recall")
    def recall(payload: RecallRequest) -> dict[str, Any]:
        arguments = payload.model_dump()
        arguments["prompt"] = arguments.pop("query")
        return _invoke(adapter.retrieve_context, arguments, mode="retrieval")

    @app.post("/api/memory/retention/inspect")
    def retention(payload: RetentionRequest) -> dict[str, Any]:
        return _invoke(adapter.inspect_memory_retention, payload.model_dump(), mode="retention")

    @app.post("/api/memory/retire/receipt")
    def retire(payload: RetireRequest) -> dict[str, Any]:
        return _invoke(adapter.retire_memory_receipted, payload.model_dump())

    @app.post("/api/memory/supersede/receipt")
    def supersede(payload: SupersedeRequest) -> dict[str, Any]:
        return _invoke(adapter.supersede_memory_receipted, payload.model_dump())

    @app.post("/api/memory/update/receipt")
    def update(payload: UpdateRequest) -> dict[str, Any]:
        return _invoke(adapter.update_memory_receipted, payload.model_dump())

    @app.post("/api/memory/promote/receipt")
    def promote(payload: PromoteRequest) -> dict[str, Any]:
        return _invoke(adapter.promote_memories_receipted, payload.model_dump())

    return app
