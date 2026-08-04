from __future__ import annotations

import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope
from daystrom_dml.context.paging import ContextPage, ContextPageError, MemoryContextPageCache


class FakeClock:
    def __init__(self) -> None:
        self.wall = 1_000.0
        self.mono = 10.0

    def now(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        self.mono += 1.0
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds


def scope(**overrides: str) -> DaystromScope:
    values = {
        "tenant_id": "tenant-a",
        "client_id": "client-a",
        "session_id": "session-a",
        "thread_id": "thread-a",
        "project_id": "project-a",
    }
    values.update(overrides)
    return DaystromScope(**values)


def test_text_page_exact_unicode_round_trip_and_json_shape() -> None:
    page = ContextPage.from_text(
        scope=scope(),
        text="alpha\nsnowman=\u2603\nemoji=\U0001f680",
        media_type="text/plain; charset=utf-8",
        content_type="hot-context",
        source_segment_id="seg-1",
        created_at=1.0,
        accessed_at=1.0,
        expires_at=9.0,
        sensitivity_label="internal",
        metadata={"rank": 1},
        page_id="page-fixed",
    )

    assert page.text() == "alpha\nsnowman=\u2603\nemoji=\U0001f680"
    assert page.bytes() == "alpha\nsnowman=\u2603\nemoji=\U0001f680".encode()
    assert page.to_dict()["payload_text"] == page.text()
    assert "payload_bytes_b64" not in page.to_dict()
    assert page.to_dict()["scope"]["tenant_id"] == "tenant-a"


def test_bytes_page_exact_round_trip() -> None:
    payload = b"\x00\x01exact\xff"
    page = ContextPage.from_bytes(
        scope=scope(),
        payload=payload,
        media_type="application/octet-stream",
        content_type="exact-context",
        source_segment_id="seg-bytes",
        created_at=2.0,
        accessed_at=2.0,
        expires_at=None,
    )

    assert page.bytes() == payload
    assert page.text() is None
    assert page.to_dict()["payload_encoding"] == "base64"


def test_cache_isolates_scope_without_metadata_leak() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    original = scope()
    other = scope(session_id="session-b")

    page = cache.put_text(original, "secret payload", ttl_seconds=60)

    assert cache.get(other, page.page_id) is None
    assert cache.touch(other, page.page_id) is False
    assert cache.evict(other, page.page_id) is False
    assert cache.list_pages(other) == []
    assert cache.stats().misses == 1


def test_ttl_expiration_behaves_as_unavailable() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    page = cache.put_text(scope(), "short lived", ttl_seconds=5)

    clock.advance(6)

    assert cache.get(scope(), page.page_id) is None
    assert cache.list_pages(scope()) == []
    assert cache.stats().misses == 1
    assert cache.stats().evictions == 1


def test_lru_eviction_is_deterministic_by_scope_and_access() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=2, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    tenant_scope = scope()
    first = cache.put_text(tenant_scope, "first", ttl_seconds=60, page_id="first")
    second = cache.put_text(tenant_scope, "second", ttl_seconds=60, page_id="second")
    assert cache.get(tenant_scope, first.page_id) is not None

    cache.put_text(tenant_scope, "third", ttl_seconds=60, page_id="third")

    assert cache.get(tenant_scope, first.page_id) is not None
    assert cache.get(tenant_scope, second.page_id) is None
    assert [page.page_id for page in cache.list_pages(tenant_scope)] == ["third", "first"]


def test_byte_limit_eviction_and_oversize_fail_closed() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(
        max_pages=10,
        max_bytes=10,
        max_page_bytes=8,
        now=clock.now,
        monotonic=clock.monotonic,
    )
    cache.put_text(scope(), "12345", ttl_seconds=60, page_id="a")
    cache.put_text(scope(), "67890", ttl_seconds=60, page_id="b")

    cache.put_text(scope(), "xyz", ttl_seconds=60, page_id="c")

    assert [page.page_id for page in cache.list_pages(scope())] == ["b", "c"]
    assert cache.stats().bytes == 8
    with pytest.raises(ContextPageError, match="exceeds per-page maximum"):
        cache.put_text(scope(), "too-large", ttl_seconds=60)


def test_digest_mismatch_fails_closed_without_mutation() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    page = ContextPage.from_text(scope=scope(), text="tampered", content_digest="sha256:not-real")

    with pytest.raises(ContextPageError, match="digest mismatch"):
        cache.put(scope(), page)

    assert cache.list_pages(scope()) == []
    assert cache.stats().pages == 0


def test_invalid_ttl_and_empty_tenant_fail_closed() -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024)
    with pytest.raises(ContextPageError, match="ttl_seconds"):
        cache.put_text(scope(), "payload", ttl_seconds=0)
    with pytest.raises(ContractError, match="tenant_id"):
        DaystromScope(tenant_id="")


def test_clear_scope_only_removes_matching_scope() -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024)
    first_scope = scope()
    second_scope = scope(project_id="project-b")
    cache.put_text(first_scope, "one", ttl_seconds=60, page_id="one")
    cache.put_text(second_scope, "two", ttl_seconds=60, page_id="two")

    assert cache.clear_scope(first_scope) == 1

    assert cache.get(first_scope, "one") is None
    assert cache.get(second_scope, "two") is not None
