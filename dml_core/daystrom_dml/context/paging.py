"""Memory-only exact context page cache for DML2 paging."""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from daystrom_dml.api_contracts import DaystromScope


class ContextPageError(ValueError):
    """Raised when a context page operation must fail closed."""


@dataclass(frozen=True)
class ContextPage:
    """JSON-friendly exact payload page.

    Text payloads are stored as text. Byte payloads are stored as base64 text so
    the dataclass can be serialized without custom JSON encoders.
    """

    page_id: str
    scope: DaystromScope
    payload_encoding: str
    media_type: str
    content_type: str
    source_segment_id: Optional[str]
    created_at: float
    accessed_at: float
    expires_at: Optional[float]
    content_digest: str
    sensitivity_label: str = "unspecified"
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload_text: Optional[str] = None
    payload_bytes_b64: Optional[str] = None

    @classmethod
    def from_text(
        cls,
        *,
        scope: DaystromScope,
        text: str,
        media_type: str = "text/plain; charset=utf-8",
        content_type: str = "context-page",
        source_segment_id: Optional[str] = None,
        created_at: float = 0.0,
        accessed_at: float = 0.0,
        expires_at: Optional[float] = None,
        content_digest: Optional[str] = None,
        sensitivity_label: str = "unspecified",
        metadata: Optional[Dict[str, Any]] = None,
        page_id: Optional[str] = None,
    ) -> "ContextPage":
        payload = text.encode("utf-8")
        return cls(
            page_id=page_id or _new_page_id(),
            scope=_copy_scope(scope),
            payload_encoding="text",
            media_type=media_type,
            content_type=content_type,
            source_segment_id=source_segment_id,
            created_at=float(created_at),
            accessed_at=float(accessed_at),
            expires_at=float(expires_at) if expires_at is not None else None,
            content_digest=content_digest or _digest(payload),
            sensitivity_label=sensitivity_label,
            metadata=_copy_metadata(metadata),
            payload_text=text,
        )

    @classmethod
    def from_bytes(
        cls,
        *,
        scope: DaystromScope,
        payload: bytes,
        media_type: str = "application/octet-stream",
        content_type: str = "context-page",
        source_segment_id: Optional[str] = None,
        created_at: float = 0.0,
        accessed_at: float = 0.0,
        expires_at: Optional[float] = None,
        content_digest: Optional[str] = None,
        sensitivity_label: str = "unspecified",
        metadata: Optional[Dict[str, Any]] = None,
        page_id: Optional[str] = None,
    ) -> "ContextPage":
        return cls(
            page_id=page_id or _new_page_id(),
            scope=_copy_scope(scope),
            payload_encoding="base64",
            media_type=media_type,
            content_type=content_type,
            source_segment_id=source_segment_id,
            created_at=float(created_at),
            accessed_at=float(accessed_at),
            expires_at=float(expires_at) if expires_at is not None else None,
            content_digest=content_digest or _digest(payload),
            sensitivity_label=sensitivity_label,
            metadata=_copy_metadata(metadata),
            payload_bytes_b64=base64.b64encode(payload).decode("ascii"),
        )

    def bytes(self) -> bytes:
        if self.payload_encoding == "text":
            if self.payload_text is None:
                raise ContextPageError("text page missing payload_text")
            return self.payload_text.encode("utf-8")
        if self.payload_encoding == "base64":
            if self.payload_bytes_b64 is None:
                raise ContextPageError("byte page missing payload_bytes_b64")
            try:
                return base64.b64decode(self.payload_bytes_b64.encode("ascii"), validate=True)
            except Exception as exc:
                raise ContextPageError("invalid base64 page payload") from exc
        raise ContextPageError(f"unsupported payload_encoding: {self.payload_encoding}")

    def text(self) -> Optional[str]:
        if self.payload_encoding == "text":
            return self.payload_text
        return None

    @property
    def size_bytes(self) -> int:
        return len(self.bytes())

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "page_id": self.page_id,
            "scope": _copy_scope(self.scope).to_dict(),
            "payload_encoding": self.payload_encoding,
            "media_type": self.media_type,
            "content_type": self.content_type,
            "source_segment_id": self.source_segment_id,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "expires_at": self.expires_at,
            "content_digest": self.content_digest,
            "sensitivity_label": self.sensitivity_label,
            "metadata": _copy_metadata(self.metadata),
        }
        if self.payload_encoding == "text":
            payload["payload_text"] = self.payload_text
        else:
            payload["payload_bytes_b64"] = self.payload_bytes_b64
        return payload


@dataclass(frozen=True)
class ContextPageCacheStats:
    pages: int
    bytes: int
    hits: int
    misses: int
    evictions: int


@dataclass
class _CacheEntry:
    page: ContextPage
    size_bytes: int
    last_access: float


class MemoryContextPageCache:
    """Bounded memory-only context page cache with strict scope isolation."""

    def __init__(
        self,
        *,
        max_pages: int,
        max_bytes: int,
        max_page_bytes: Optional[int] = None,
        now: Optional[Callable[[], float]] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        if max_pages < 1:
            raise ContextPageError("max_pages must be positive")
        if max_bytes < 1:
            raise ContextPageError("max_bytes must be positive")
        if max_page_bytes is not None and max_page_bytes < 1:
            raise ContextPageError("max_page_bytes must be positive")
        self._max_pages = int(max_pages)
        self._max_bytes = int(max_bytes)
        self._max_page_bytes = int(max_page_bytes) if max_page_bytes is not None else int(max_bytes)
        self._now = now or _default_now
        self._monotonic = monotonic or _default_monotonic
        self._entries: Dict[Tuple[Tuple[Optional[str], ...], str], _CacheEntry] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def put_text(
        self,
        scope: DaystromScope,
        text: str,
        *,
        ttl_seconds: Optional[float],
        media_type: str = "text/plain; charset=utf-8",
        content_type: str = "context-page",
        source_segment_id: Optional[str] = None,
        sensitivity_label: str = "unspecified",
        metadata: Optional[Dict[str, Any]] = None,
        page_id: Optional[str] = None,
    ) -> ContextPage:
        created_at, expires_at = self._times_for_ttl(ttl_seconds)
        page = ContextPage.from_text(
            scope=scope,
            text=text,
            media_type=media_type,
            content_type=content_type,
            source_segment_id=source_segment_id,
            created_at=created_at,
            accessed_at=created_at,
            expires_at=expires_at,
            sensitivity_label=sensitivity_label,
            metadata=metadata,
            page_id=page_id,
        )
        return self.put(scope, page)

    def put_bytes(
        self,
        scope: DaystromScope,
        payload: bytes,
        *,
        ttl_seconds: Optional[float],
        media_type: str = "application/octet-stream",
        content_type: str = "context-page",
        source_segment_id: Optional[str] = None,
        sensitivity_label: str = "unspecified",
        metadata: Optional[Dict[str, Any]] = None,
        page_id: Optional[str] = None,
    ) -> ContextPage:
        created_at, expires_at = self._times_for_ttl(ttl_seconds)
        page = ContextPage.from_bytes(
            scope=scope,
            payload=payload,
            media_type=media_type,
            content_type=content_type,
            source_segment_id=source_segment_id,
            created_at=created_at,
            accessed_at=created_at,
            expires_at=expires_at,
            sensitivity_label=sensitivity_label,
            metadata=metadata,
            page_id=page_id,
        )
        return self.put(scope, page)

    def put(self, scope: DaystromScope, page: ContextPage) -> ContextPage:
        self._validate_scope(scope)
        if not page.page_id:
            raise ContextPageError("page_id must be non-empty")
        if _scope_key(scope) != _scope_key(page.scope):
            raise ContextPageError("page scope mismatch")
        now = float(self._now())
        if page.expires_at is not None and page.expires_at <= now:
            raise ContextPageError("page expires_at must be in the future")
        self._validate_digest(page)
        size = page.size_bytes
        if size > self._max_page_bytes:
            raise ContextPageError("page exceeds per-page maximum")
        if size > self._max_bytes:
            raise ContextPageError("page exceeds cache byte maximum")

        key = (_scope_key(scope), page.page_id)
        self._evict_expired(now)
        existing = self._entries.get(key)
        projected_bytes = self._bytes - (existing.size_bytes if existing else 0) + size
        while (len(self._entries) - (1 if existing else 0)) >= self._max_pages or projected_bytes > self._max_bytes:
            evicted = self._evict_lru(skip_key=key)
            if evicted is None:
                break
            projected_bytes -= evicted.size_bytes
        if projected_bytes > self._max_bytes:
            raise ContextPageError("unable to fit page within byte maximum")

        if existing:
            self._bytes -= existing.size_bytes
        accessed = float(self._now())
        cached_page = _copy_page(replace(page, accessed_at=accessed))
        entry = _CacheEntry(page=cached_page, size_bytes=size, last_access=float(self._monotonic()))
        self._entries[key] = entry
        self._bytes += size
        return _copy_page(cached_page)

    def get(self, scope: DaystromScope, page_id: str) -> Optional[ContextPage]:
        self._validate_scope(scope)
        key = (_scope_key(scope), str(page_id))
        entry = self._entries.get(key)
        now = float(self._now())
        if entry is None:
            self._misses += 1
            return None
        if self._is_expired(entry.page, now):
            self._drop(key)
            self._evictions += 1
            self._misses += 1
            return None
        self._hits += 1
        return self._touch_entry(key, entry, now)

    def touch(self, scope: DaystromScope, page_id: str) -> bool:
        self._validate_scope(scope)
        key = (_scope_key(scope), str(page_id))
        entry = self._entries.get(key)
        now = float(self._now())
        if entry is None:
            return False
        if self._is_expired(entry.page, now):
            self._drop(key)
            self._evictions += 1
            return False
        self._touch_entry(key, entry, now)
        return True

    def evict(self, scope: DaystromScope, page_id: str) -> bool:
        self._validate_scope(scope)
        key = (_scope_key(scope), str(page_id))
        if key not in self._entries:
            return False
        self._drop(key)
        self._evictions += 1
        return True

    def list_pages(self, scope: DaystromScope) -> List[ContextPage]:
        self._validate_scope(scope)
        now = float(self._now())
        self._evict_expired(now)
        scope_key = _scope_key(scope)
        entries = [entry for key, entry in self._entries.items() if key[0] == scope_key]
        return [_copy_page(entry.page) for entry in sorted(entries, key=lambda item: (item.last_access, item.page.page_id))]

    def clear_scope(self, scope: DaystromScope) -> int:
        self._validate_scope(scope)
        scope_key = _scope_key(scope)
        keys = [key for key in self._entries if key[0] == scope_key]
        for key in keys:
            self._drop(key)
        self._evictions += len(keys)
        return len(keys)

    def stats(self) -> ContextPageCacheStats:
        self._evict_expired(float(self._now()))
        return ContextPageCacheStats(
            pages=len(self._entries),
            bytes=self._bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    def _times_for_ttl(self, ttl_seconds: Optional[float]) -> Tuple[float, Optional[float]]:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ContextPageError("ttl_seconds must be positive")
        created_at = float(self._now())
        expires_at = created_at + float(ttl_seconds) if ttl_seconds is not None else None
        return created_at, expires_at

    def _validate_scope(self, scope: DaystromScope) -> None:
        if not scope.tenant_id:
            raise ContextPageError("scope tenant_id must be non-empty")

    def _validate_digest(self, page: ContextPage) -> None:
        expected = _digest(page.bytes())
        if page.content_digest != expected:
            raise ContextPageError("digest mismatch")

    def _touch_entry(self, key: Tuple[Tuple[Optional[str], ...], str], entry: _CacheEntry, now: float) -> ContextPage:
        page = replace(entry.page, accessed_at=now)
        self._entries[key] = _CacheEntry(page=page, size_bytes=entry.size_bytes, last_access=float(self._monotonic()))
        return _copy_page(page)

    def _evict_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if self._is_expired(entry.page, now)]
        for key in expired:
            self._drop(key)
        self._evictions += len(expired)

    def _is_expired(self, page: ContextPage, now: float) -> bool:
        return page.expires_at is not None and page.expires_at <= now

    def _evict_lru(self, *, skip_key: Tuple[Tuple[Optional[str], ...], str]) -> Optional[_CacheEntry]:
        candidates = [(key, entry) for key, entry in self._entries.items() if key != skip_key]
        if not candidates:
            return None
        key, entry = min(candidates, key=lambda item: (item[1].last_access, item[0][1]))
        self._drop(key)
        self._evictions += 1
        return entry

    def _drop(self, key: Tuple[Tuple[Optional[str], ...], str]) -> None:
        entry = self._entries.pop(key)
        self._bytes -= entry.size_bytes


def _new_page_id() -> str:
    return f"ctxpg_{uuid.uuid4().hex}"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _scope_key(scope: DaystromScope) -> Tuple[Optional[str], ...]:
    return (
        scope.tenant_id,
        scope.client_id,
        scope.session_id,
        scope.instance_id,
        scope.thread_id,
        scope.project_id,
        scope.relationship_id,
    )


def _copy_scope(scope: DaystromScope) -> DaystromScope:
    return DaystromScope.from_dict(scope.to_dict())


def _copy_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ContextPageError("metadata must be JSON-compatible")
    try:
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ContextPageError("metadata must be JSON-compatible") from exc
    if not isinstance(copied, dict):
        raise ContextPageError("metadata must be JSON-compatible")
    return copied


def _copy_page(page: ContextPage) -> ContextPage:
    return replace(page, scope=_copy_scope(page.scope), metadata=_copy_metadata(page.metadata))


def _default_now() -> float:
    import time

    return time.time()


def _default_monotonic() -> float:
    import time

    return time.monotonic()
