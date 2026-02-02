from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from daystrom_dml.llm_backends.transformers_backend import TransformersBackend


def test_transformers_backend_generates_text() -> None:
    try:
        backend = TransformersBackend(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
            dtype="float32",
        )
    except Exception as exc:
        pytest.skip(f"Unable to load tiny model: {exc}")
    output = backend.generate(
        prompt="Hello",
        max_new_tokens=5,
        temperature=0.0,
        top_p=1.0,
    )
    assert isinstance(output, str)
    assert output.strip() != ""
