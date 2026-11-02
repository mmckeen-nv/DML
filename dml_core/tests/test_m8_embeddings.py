"""No-model controls for the pinned M8 embedding pipeline; doubles only."""

import sys

import numpy as np
import pytest

from daystrom_dml.services.m8_embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_IDENTITY,
    EMBEDDING_MODEL,
    MAX_CONTENT_TOKENS,
    MODEL_TOKEN_LIMIT,
    SNAPSHOT_REVISION,
    EncodedInput,
    PinnedMiniLMEmbedder,
    chunk_text,
)


def words(text):
    # Injected deterministic counter for chunk mechanics only. The real arms
    # must pass the pinned tokenizer counter, never this fixture.
    return len(text.split())


def encoded(hidden, mask, special=2):
    return EncodedInput(
        hidden_states=np.asarray(hidden, dtype=np.float32),
        attention_mask=np.asarray(mask, dtype=np.float32),
        special_tokens=special,
    )


def one_hot_rows(length, dimension=EMBEDDING_DIMENSIONS, value=1.0):
    rows = np.zeros((length, dimension), dtype=np.float32)
    rows[:, 0] = value
    return rows


def embedder_for(hidden, mask, special=2, calls=None):
    def backend(text):
        if calls is not None:
            calls.append(text)
        return encoded(hidden, mask, special)

    return PinnedMiniLMEmbedder(backend)


def test_identity_pins():
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert SNAPSHOT_REVISION == "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    assert EMBEDDING_IDENTITY == (
        "sentence-transformers/all-MiniLM-L6-v2@c9745ed1d9f207416be6d2e6f8de32d1f16199bf+cpu-fp32-meanpool-l2-384-v1"
    )
    assert (EMBEDDING_DIMENSIONS, MAX_CONTENT_TOKENS, MODEL_TOKEN_LIMIT) == (384, 192, 256)


def test_import_and_pure_pipeline_load_no_heavy_backend():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    embedder_for(one_hot_rows(4), [1, 1, 1, 1]).embed("text")
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_embed_shape_dtype_l2_and_finite():
    hidden = np.arange(4 * EMBEDDING_DIMENSIONS, dtype=np.float32).reshape(4, -1) + 1.0
    vector = embedder_for(hidden, [1, 1, 1, 1]).embed("some text")
    assert vector.shape == (EMBEDDING_DIMENSIONS,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-5, atol=1e-6)


def test_mean_pooling_uses_attention_tokens_only():
    hidden = np.zeros((4, EMBEDDING_DIMENSIONS), dtype=np.float32)
    hidden[0, 0], hidden[1, 1], hidden[2, 2] = 1.0, 1.0, 1.0
    hidden[3] = 1e30  # padded row must not leak into the pooled vector
    vector = embedder_for(hidden, [1, 1, 1, 0], special=1).embed("text")
    expected = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
    expected[:3] = 1.0 / 3.0
    expected /= np.linalg.norm(expected)
    assert np.allclose(vector, expected, rtol=1e-5, atol=1e-6)


def test_content_token_boundary_192_admitted_193_rejected():
    embedder_for(one_hot_rows(194), [1] * 194).embed("at the limit")  # 192 content
    with pytest.raises(ValueError, match="192 content tokens"):
        embedder_for(one_hot_rows(195), [1] * 195).embed("over the limit")


def test_model_limit_rejects_oversized_encoding_without_truncation():
    with pytest.raises(ValueError, match="256-token model limit"):
        embedder_for(one_hot_rows(257), [1] * 257, special=0).embed("huge")


def test_embed_rejects_empty_whitespace_and_nonstring():
    embedder = embedder_for(one_hot_rows(4), [1, 1, 1, 1])
    for bad in ("", "   \n\t ", None, 7, b"bytes"):
        with pytest.raises(ValueError, match="nonempty text"):
            embedder.embed(bad)


def test_batch_one_semantics():
    calls = []
    embedder = embedder_for(one_hot_rows(4), [1, 1, 1, 1], calls=calls)
    vector = embedder.embed("exactly one text")
    assert calls == ["exactly one text"]
    assert vector.ndim == 1
    with pytest.raises(ValueError, match="nonempty text"):
        embedder.embed(["two", "texts"])


def test_embed_rejects_invalid_backend_rows():
    cases = [
        (one_hot_rows(4), [1, 1, 1], 2),  # mask/hidden length mismatch
        (one_hot_rows(4), [1, 1, 1, 2], 2),  # non-binary mask
        (one_hot_rows(4), [0, 0, 0, 0], 0),  # empty attention span
        (one_hot_rows(4), [1, 1, 1, 1], 5),  # special count above span
        (one_hot_rows(4), [1, 1, 1, 1], -1),  # negative special count
        (one_hot_rows(4, dimension=8), [1, 1, 1, 1], 2),  # wrong dimension
        (np.full((4, EMBEDDING_DIMENSIONS), np.nan, dtype=np.float32), [1, 1, 1, 1], 2),
        (np.zeros((4, EMBEDDING_DIMENSIONS), dtype=np.float32), [1, 1, 1, 1], 2),  # zero norm
    ]
    for hidden, mask, special in cases:
        with pytest.raises(ValueError):
            embedder_for(hidden, mask, special).embed("text")
    with pytest.raises(ValueError, match="EncodedInput"):
        PinnedMiniLMEmbedder(lambda text: (one_hot_rows(4), [1, 1, 1, 1])).embed("text")
    with pytest.raises(ValueError, match="backend required"):
        PinnedMiniLMEmbedder(None)


def test_chunk_boundary_192_whole_paragraph_and_193_rejection():
    text = " ".join(f"w{i}" for i in range(192))
    assert chunk_text(text, words) == [text]
    with pytest.raises(ValueError, match="unsplittable sentence"):
        chunk_text(text + " extra", words)  # one 193-word sentence, no boundary


def test_chunk_sentence_packing_reassembles_exactly():
    text = "One two. Three four. Five six."
    chunks = chunk_text(text, words, max_tokens=4)
    assert chunks == ["One two. Three four. ", "Five six."]
    assert "".join(chunks) == text
    assert all(words(chunk) <= 4 for chunk in chunks)


def test_chunk_paragraphs_are_hard_boundaries_and_reassemble():
    text = "Alpha one. Beta two.\n\nGamma three. Delta four."
    chunks = chunk_text(text, words, max_tokens=6)
    assert chunks == ["Alpha one. Beta two.\n\n", "Gamma three. Delta four."]
    assert "".join(chunks) == text
    forced = chunk_text(text, words, max_tokens=3)
    assert forced == ["Alpha one. ", "Beta two.\n\n", "Gamma three. ", "Delta four."]
    assert "".join(forced) == text


def test_chunk_is_deterministic():
    text = "A b. C d. E f.\n\nG h. I j. K l."
    first = chunk_text(text, words, max_tokens=3)
    assert first == chunk_text(text, words, max_tokens=3)


def test_chunk_rejects_empty_whitespace_and_bad_arguments():
    for bad in ("", "  \n\n  ", None, 5):
        with pytest.raises(ValueError, match="nonempty text"):
            chunk_text(bad, words)
    with pytest.raises(ValueError, match="token counter required"):
        chunk_text("text", None)
    for bad in (0, -1, 193, 1.5, "192"):
        with pytest.raises(ValueError, match="max_tokens"):
            chunk_text("text", words, max_tokens=bad)
    with pytest.raises(ValueError, match="invalid token count"):
        chunk_text("text", lambda text: -1)


def test_chunk_requires_sentence_boundary_for_oversized_units():
    with pytest.raises(ValueError, match="unsplittable sentence"):
        chunk_text("no punctuation anywhere in this paragraph", words, max_tokens=3)


def test_embedder_chunk_uses_injected_counter():
    embedder = PinnedMiniLMEmbedder(lambda text: encoded(one_hot_rows(4), [1, 1, 1, 1]), words)
    assert embedder.chunk("One two. Three four.") == ["One two. Three four."]
    with pytest.raises(ValueError, match="no token counter"):
        embedder_for(one_hot_rows(4), [1, 1, 1, 1]).chunk("text")


def test_embedder_identity_attributes():
    embedder = embedder_for(one_hot_rows(4), [1, 1, 1, 1])
    assert embedder.model_name == EMBEDDING_MODEL
    assert embedder.embedding_identity == EMBEDDING_IDENTITY
    assert embedder.dimensions == EMBEDDING_DIMENSIONS
