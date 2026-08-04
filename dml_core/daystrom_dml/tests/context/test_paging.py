from __future__ import annotations

import json

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
        "instance_id": "instance-a",
        "thread_id": "thread-a",
        "project_id": "project-a",
        "relationship_id": "relationship-a",
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


@pytest.mark.parametrize("kind", ["text", "bytes"])
def test_page_json_roundtrip(kind: str) -> None:
    if kind == "text":
        page = ContextPage.from_text(scope=scope(), text="exact text", created_at=1, accessed_at=1, expires_at=2)
    else:
        page = ContextPage.from_bytes(scope=scope(), payload=b"\x00exact\xff", created_at=1, accessed_at=1, expires_at=2)

    restored = ContextPage.from_dict(json.loads(json.dumps(page.to_dict())))

    assert restored == page
    assert restored.bytes() == page.bytes()


def test_page_from_dict_rejects_payload_and_digest_tampering() -> None:
    page = ContextPage.from_text(scope=scope(), text="exact", created_at=1, accessed_at=1, expires_at=2)
    payload = page.to_dict()

    with pytest.raises(ContextPageError, match="only payload_text"):
        ContextPage.from_dict({**payload, "payload_bytes_b64": "ZXhhY3Q="})
    without_payload = dict(payload)
    without_payload.pop("payload_text")
    with pytest.raises(ContextPageError, match="only payload_text"):
        ContextPage.from_dict(without_payload)
    with pytest.raises(ContextPageError, match="digest mismatch"):
        ContextPage.from_dict({**payload, "content_digest": "sha256:not-real"})

    byte_payload = ContextPage.from_bytes(scope=scope(), payload=b"exact", created_at=1, accessed_at=1).to_dict()
    with pytest.raises(ContextPageError, match="invalid base64"):
        ContextPage.from_dict({**byte_payload, "payload_bytes_b64": "***"})


def test_page_rejects_invalid_temporal_ordering() -> None:
    with pytest.raises(ContextPageError, match="timestamps"):
        ContextPage.from_text(scope=scope(), text="x", created_at=-1, accessed_at=0)
    with pytest.raises(ContextPageError, match="accessed_at"):
        ContextPage.from_text(scope=scope(), text="x", created_at=2, accessed_at=1)
    with pytest.raises(ContextPageError, match="expires_at"):
        ContextPage.from_text(scope=scope(), text="x", created_at=1, accessed_at=2, expires_at=1)


@pytest.mark.parametrize(
    ("field", "other_value"),
    [
        ("tenant_id", "tenant-b"),
        ("client_id", "client-b"),
        ("session_id", "session-b"),
        ("instance_id", "instance-b"),
        ("thread_id", "thread-b"),
        ("project_id", "project-b"),
        ("relationship_id", "relationship-b"),
    ],
)
def test_cache_isolates_scope_without_metadata_leak(field: str, other_value: str) -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    original = scope()
    other = scope(**{field: other_value})

    page = cache.put_text(original, "secret payload", ttl_seconds=60)

    assert cache.get(other, page.page_id) is None
    assert cache.touch(other, page.page_id) is False
    assert cache.evict(other, page.page_id) is False
    assert cache.list_pages(other) == []
    assert cache.stats().misses == 1


@pytest.mark.parametrize(
    ("field", "other_value"),
    [
        ("tenant_id", "tenant-b"),
        ("client_id", "client-b"),
        ("session_id", "session-b"),
        ("instance_id", "instance-b"),
        ("thread_id", "thread-b"),
        ("project_id", "project-b"),
        ("relationship_id", "relationship-b"),
    ],
)
def test_put_rejects_page_scope_mismatch_on_all_isolation_fields(field: str, other_value: str) -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024)
    page = ContextPage.from_text(scope=scope(**{field: other_value}), text="payload")

    with pytest.raises(ContextPageError, match="page scope mismatch"):
        cache.put(scope(), page)

    assert cache.list_pages(scope()) == []


def test_cache_copies_scope_and_metadata_on_ingress_and_egress() -> None:
    clock = FakeClock()
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024, now=clock.now, monotonic=clock.monotonic)
    caller_scope = scope()
    metadata = {"rank": 1, "nested": {"labels": ["initial"]}}

    page = cache.put_text(caller_scope, "stable payload", ttl_seconds=60, metadata=metadata, page_id="stable")
    metadata["nested"]["labels"].append("caller-mutated")
    object.__setattr__(caller_scope, "session_id", "caller-mutated")
    page.metadata["nested"]["labels"].append("returned-mutated")
    object.__setattr__(page.scope, "thread_id", "returned-mutated")

    cached = cache.get(scope(), "stable")

    assert cached is not None
    assert cached.scope == scope()
    assert cached.metadata == {"rank": 1, "nested": {"labels": ["initial"]}}

    cached.metadata["nested"]["labels"].append("egress-mutated")
    again = cache.get(scope(), "stable")

    assert again is not None
    assert again.metadata == {"rank": 1, "nested": {"labels": ["initial"]}}


def test_put_copies_supplied_page_and_rejects_non_json_metadata() -> None:
    cache = MemoryContextPageCache(max_pages=4, max_bytes=1024)
    caller_scope = scope()
    caller_metadata = {"nested": {"labels": ["initial"]}}
    page = ContextPage.from_text(scope=caller_scope, text="stable payload", metadata=caller_metadata, page_id="stable")

    stored = cache.put(caller_scope, page)
    caller_metadata["nested"]["labels"].append("caller-mutated")
    page.metadata["nested"]["labels"].append("page-mutated")
    object.__setattr__(caller_scope, "client_id", "caller-mutated")

    cached = cache.get(scope(), stored.page_id)

    assert cached is not None
    assert cached.scope == scope()
    assert cached.metadata == {"nested": {"labels": ["initial"]}}

    with pytest.raises(ContextPageError, match="metadata must be JSON-compatible"):
        ContextPage.from_text(scope=scope(), text="payload", metadata={"bad": object()})


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
