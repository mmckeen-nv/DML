"""Pinned MiniLM embedding runtime for the M8 persistent arms.

One pinned embedder identity for every persistent M8 arm: CPU float32,
batch one, mean pooling over attention tokens, L2 normalized, 384 dims.
Silent truncation is rejected everywhere: inputs above 192 content tokens
raise; chunk_text is the only admitted path to fit longer text.

No model is loaded by importing this module; torch/transformers stay lazy
inside from_pretrained. The pure pipeline (validate, pool, normalize) takes
an injected backend so tests never touch the real model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SNAPSHOT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_IDENTITY = (
    "sentence-transformers/all-MiniLM-L6-v2@c9745ed1d9f207416be6d2e6f8de32d1f16199bf+cpu-fp32-meanpool-l2-384-v1"
)
EMBEDDING_DIMENSIONS = 384
MAX_CONTENT_TOKENS = 192
MODEL_TOKEN_LIMIT = 256
NUM_THREADS = 4

_PARAGRAPH_BREAK = re.compile(r"(\n\s*\n)")
_SENTENCE_BOUNDARY = re.compile(r"((?<=[.!?])\s+)")


@dataclass(frozen=True)
class EncodedInput:
    """Untruncated encoding of ONE text: hidden states, mask, special count.

    The backend contract: encode with truncation disabled and report every
    special token inside the attention span, so the pipeline can enforce the
    content-token limit instead of inheriting a silently truncated row.
    """

    hidden_states: np.ndarray  # (sequence_length, EMBEDDING_DIMENSIONS) float32
    attention_mask: np.ndarray  # (sequence_length,) of 0/1
    special_tokens: int  # special tokens inside the attention span


def _measured(count_tokens: Callable[[str], int], text: str) -> int:
    used = count_tokens(text)
    if type(used) is not int or not 0 <= used <= 2**63 - 1:
        raise ValueError("invalid token count")
    return used


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    *,
    max_tokens: int = MAX_CONTENT_TOKENS,
) -> list[str]:
    """Split text on deterministic boundaries; every chunk fits, or raise.

    Paragraphs (blank-line separated) are hard boundaries; oversized
    paragraphs fall back to sentences packed greedily in order. Separators
    stay attached to the preceding chunk, so "".join(chunk_text(t, c)) == t.
    A single sentence over max_tokens is unsplittable and raises; nothing is
    ever truncated or split mid-sentence.
    """
    if type(text) is not str or not text.strip():
        raise ValueError("chunking requires nonempty text")
    if not callable(count_tokens):
        raise ValueError("token counter required")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_CONTENT_TOKENS:
        raise ValueError("invalid max_tokens")
    chunks: list[str] = []
    pending = ""  # unattached leading separator whitespace
    parts = _PARAGRAPH_BREAK.split(text)
    index = 0
    while index < len(parts):
        paragraph = pending + parts[index]
        pending = ""
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        index += 2
        if not paragraph.strip():
            pending = paragraph + separator
            continue
        if _measured(count_tokens, paragraph) <= max_tokens:
            chunks.append(paragraph + separator)
            continue
        segments = _SENTENCE_BOUNDARY.split(paragraph)
        units = []
        position = 0
        while position < len(segments):
            unit = segments[position] + (segments[position + 1] if position + 1 < len(segments) else "")
            position += 2
            if unit:
                units.append(unit)
        current = ""
        for unit in units:
            if _measured(count_tokens, unit) > max_tokens:
                raise ValueError("single unsplittable sentence exceeds the content token limit")
            candidate = current + unit
            if current and _measured(count_tokens, candidate) > max_tokens:
                chunks.append(current)
                current = unit
            else:
                current = candidate
        if current or separator:
            chunks.append(current + separator)
    if pending:
        if not chunks:
            raise ValueError("chunking requires nonempty text")
        chunks[-1] += pending
    if not chunks or any(not chunk.strip() for chunk in chunks):
        raise ValueError("chunking produced an empty chunk")
    return chunks


class PinnedMiniLMEmbedder:
    """Pure pinned pipeline over an injected token-embedding backend.

    The backend is a callable str -> EncodedInput encoding ONE text with
    truncation disabled. embed() validates the encoding, rejects anything
    over 192 content tokens (or 256 total tokens), mean-pools over attention
    tokens and L2-normalizes. It makes no quality claim beyond the pinned
    mechanics in EMBEDDING_IDENTITY.
    """

    model_name = EMBEDDING_MODEL
    embedding_identity = EMBEDDING_IDENTITY
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(
        self,
        backend: Callable[[str], EncodedInput],
        count_tokens: Callable[[str], int] | None = None,
    ):
        if not callable(backend):
            raise ValueError("token-embedding backend required")
        if count_tokens is not None and not callable(count_tokens):
            raise ValueError("invalid token counter")
        self._backend = backend
        self._count_tokens = count_tokens

    @classmethod
    def from_pretrained(cls) -> "PinnedMiniLMEmbedder":
        """Construct the real pinned CPU backend; heavy imports stay here."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(NUM_THREADS)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, revision=SNAPSHOT_REVISION, local_files_only=True)
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            revision=SNAPSHOT_REVISION,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()

        def count_tokens(text: str) -> int:
            if type(text) is not str or not text.strip():
                raise ValueError("token counting requires nonempty text")
            batch = tokenizer(text, truncation=False, return_special_tokens_mask=True)
            return int(
                sum(
                    1
                    for attended, special in zip(batch["attention_mask"], batch["special_tokens_mask"])
                    if attended and not special
                )
            )

        def backend(text: str) -> EncodedInput:
            batch = tokenizer(text, return_tensors="pt", truncation=False)
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state[0]
            special = tokenizer(text, return_tensors="pt", truncation=False, return_special_tokens_mask=True)[
                "special_tokens_mask"
            ][0]
            mask = batch["attention_mask"][0]
            return EncodedInput(
                hidden_states=hidden.float().numpy(),
                attention_mask=mask.numpy(),
                special_tokens=int((special.bool() & mask.bool()).sum()),
            )

        return cls(backend, count_tokens)

    def _encoded(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoded = self._backend(text)
        if not isinstance(encoded, EncodedInput):
            raise ValueError("backend must return an EncodedInput")
        hidden = np.asarray(encoded.hidden_states, dtype=np.float32)
        mask = np.asarray(encoded.attention_mask, dtype=np.float32)
        if hidden.ndim != 2 or hidden.shape[1] != self.dimensions or hidden.shape[0] < 1:
            raise ValueError("invalid hidden states")
        if not np.all(np.isfinite(hidden)):
            raise ValueError("nonfinite hidden states")
        if mask.shape != (hidden.shape[0],) or not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("invalid attention mask")
        attended = int(mask.sum())
        if attended < 1:
            raise ValueError("empty attention span")
        if type(encoded.special_tokens) is not int or not 0 <= encoded.special_tokens <= attended:
            raise ValueError("invalid special token count")
        if attended > MODEL_TOKEN_LIMIT:
            raise ValueError("encoding exceeds the 256-token model limit; silent truncation rejected")
        if attended - encoded.special_tokens > MAX_CONTENT_TOKENS:
            raise ValueError("text exceeds 192 content tokens; chunk first, truncation rejected")
        return hidden, mask

    def embed(self, text: str) -> np.ndarray:
        """Embed ONE nonempty text into a finite L2-normalized 384-vector."""
        if type(text) is not str or not text.strip():
            raise ValueError("embedding requires nonempty text")
        hidden, mask = self._encoded(text)
        weights = mask[:, None]
        pooled = (hidden * weights).sum(0) / weights.sum()
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("degenerate pooled embedding")
        vector = (pooled / norm).astype(np.float32)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("invalid embedding row")
        return vector

    def chunk(self, text: str) -> list[str]:
        """chunk_text with this embedder's pinned tokenizer counter."""
        if self._count_tokens is None:
            raise ValueError("this embedder has no token counter; pass count_tokens to chunk_text")
        return chunk_text(text, self._count_tokens)
