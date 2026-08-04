"""Request-scoped resource budgets for document ingestion."""
from __future__ import annotations

import os
from dataclasses import dataclass


class IngestionLimitExceeded(ValueError):
    """Raised when an upload would exceed its configured processing budget."""


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class IngestionLimits:
    """Limits applied to one ``/upload`` request."""

    max_upload_bytes: int
    max_file_bytes: int
    max_decompressed_bytes: int
    max_archive_member_bytes: int
    max_archive_members: int
    max_archive_depth: int
    max_documents: int
    max_chunks: int
    max_tokens: int

    @classmethod
    def from_env(cls) -> "IngestionLimits":
        return cls(
            max_upload_bytes=_env_int("DML_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            max_file_bytes=_env_int("DML_MAX_UPLOAD_FILE_SIZE", 10 * 1024 * 1024),
            max_decompressed_bytes=_env_int(
                "DML_MAX_DECOMPRESSED_BYTES", 50 * 1024 * 1024
            ),
            max_archive_member_bytes=_env_int(
                "DML_MAX_ARCHIVE_MEMBER_SIZE", 5 * 1024 * 1024
            ),
            max_archive_members=_env_int("DML_MAX_ARCHIVE_MEMBERS", 1_000),
            max_archive_depth=_env_int("DML_MAX_ARCHIVE_DEPTH", 3),
            max_documents=_env_int("DML_MAX_INGEST_DOCUMENTS", 500),
            max_chunks=_env_int("DML_MAX_INGEST_CHUNKS", 5_000),
            max_tokens=_env_int("DML_MAX_INGEST_TOKENS", 1_000_000),
        )


@dataclass
class IngestionBudget:
    """Mutable accounting state shared by all files in an upload request."""

    limits: IngestionLimits
    upload_bytes: int = 0
    decompressed_bytes: int = 0
    archive_members: int = 0
    documents: int = 0
    chunks: int = 0
    tokens: int = 0

    def _add(self, field: str, amount: int, maximum: int, label: str) -> None:
        value = getattr(self, field) + amount
        if value > maximum:
            raise IngestionLimitExceeded(f"Upload exceeds {label} limit ({maximum}).")
        setattr(self, field, value)

    def add_upload_bytes(self, amount: int) -> None:
        self._add("upload_bytes", amount, self.limits.max_upload_bytes, "total size")

    def add_decompressed_bytes(self, amount: int) -> None:
        self._add(
            "decompressed_bytes",
            amount,
            self.limits.max_decompressed_bytes,
            "decompressed size",
        )

    def add_archive_member(self) -> None:
        self._add(
            "archive_members", 1, self.limits.max_archive_members, "archive member count"
        )

    def check_archive_member_size(self, size: int) -> None:
        if size > self.limits.max_archive_member_bytes:
            raise IngestionLimitExceeded(
                "Archive member exceeds size limit "
                f"({self.limits.max_archive_member_bytes})."
            )

    def add_document(self) -> None:
        self._add("documents", 1, self.limits.max_documents, "document count")

    def add_chunks(self, amount: int) -> None:
        self._add("chunks", amount, self.limits.max_chunks, "chunk count")

    def add_tokens(self, amount: int) -> None:
        self._add("tokens", amount, self.limits.max_tokens, "token count")

    def check_depth(self, depth: int) -> None:
        if depth > self.limits.max_archive_depth:
            raise IngestionLimitExceeded(
                f"Upload exceeds archive nesting depth limit ({self.limits.max_archive_depth})."
            )
