from __future__ import annotations

import io
import zipfile

import pytest

from daystrom_dml import server
from daystrom_dml.ingestion_limits import (
    IngestionBudget,
    IngestionLimitExceeded,
    IngestionLimits,
)


def _limits(**overrides: int) -> IngestionLimits:
    values = {
        "max_upload_bytes": 1_000,
        "max_file_bytes": 1_000,
        "max_decompressed_bytes": 1_000,
        "max_archive_member_bytes": 1_000,
        "max_archive_members": 10,
        "max_archive_depth": 2,
        "max_documents": 10,
        "max_chunks": 10,
        "max_tokens": 1_000,
    }
    values.update(overrides)
    return IngestionLimits(**values)


def _zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return output.getvalue()


def test_archive_rejects_excess_aggregate_decompressed_bytes() -> None:
    budget = IngestionBudget(_limits(max_decompressed_bytes=7))
    contents = _zip({"one.txt": b"1234", "two.txt": b"5678"})

    with pytest.raises(IngestionLimitExceeded, match="decompressed size"):
        list(server._iter_ingest_documents("docs.zip", contents, "application/zip", budget=budget))


def test_archive_rejects_excess_member_count() -> None:
    budget = IngestionBudget(_limits(max_archive_members=1))
    contents = _zip({"one.txt": b"one", "two.txt": b"two"})

    with pytest.raises(IngestionLimitExceeded, match="archive member count"):
        list(server._iter_ingest_documents("docs.zip", contents, "application/zip", budget=budget))


def test_archive_rejects_excess_nesting_depth() -> None:
    innermost = _zip({"doc.txt": b"hello"})
    middle = _zip({"inner.zip": innermost})
    outer = _zip({"middle.zip": middle})
    budget = IngestionBudget(_limits(max_archive_depth=2))

    with pytest.raises(IngestionLimitExceeded, match="nesting depth"):
        list(server._iter_ingest_documents("outer.zip", outer, "application/zip", budget=budget))


def test_document_chunk_and_token_limits_are_request_scoped() -> None:
    budget = IngestionBudget(_limits(max_documents=1, max_chunks=2, max_tokens=3))
    budget.add_document()
    budget.add_chunks(2)
    budget.add_tokens(3)

    with pytest.raises(IngestionLimitExceeded, match="document count"):
        budget.add_document()
    with pytest.raises(IngestionLimitExceeded, match="chunk count"):
        budget.add_chunks(1)
    with pytest.raises(IngestionLimitExceeded, match="token count"):
        budget.add_tokens(1)
