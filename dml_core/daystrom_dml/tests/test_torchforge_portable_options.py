from __future__ import annotations

import pytest

from daystrom_dml.llm_backends.transformers_backend import (
    _build_portable_load_options,
    portable_to_torchforge_options,
)


def test_portable_load_options_basic_shape() -> None:
    options = _build_portable_load_options(
        model_name="sshleifer/tiny-gpt2",
        device="CPU",
        dtype="FLOAT32",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert options == {
        "loader": "transformers",
        "model_name": "sshleifer/tiny-gpt2",
        "device": "cpu",
        "dtype": "float32",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "load_in_4bit": False,
        "load_in_8bit": False,
    }


def test_portable_load_options_enables_device_map_for_quantized_loads() -> None:
    options = _build_portable_load_options(
        model_name="meta-llama/Llama-3.2-1B",
        device="auto",
        dtype="auto",
        trust_remote_code=True,
        use_fast_tokenizer=False,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    assert options["load_in_4bit"] is True
    assert options["load_in_8bit"] is False
    assert options["device_map"] == "auto"


def test_portable_load_options_rejects_conflicting_quantization_flags() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_portable_load_options(
            model_name="meta-llama/Llama-3.2-1B",
            device="auto",
            dtype="auto",
            trust_remote_code=True,
            use_fast_tokenizer=False,
            load_in_4bit=True,
            load_in_8bit=True,
        )


def test_portable_load_options_rejects_blank_model_name() -> None:
    with pytest.raises(ValueError, match="model_name is required"):
        _build_portable_load_options(
            model_name="   ",
            device="auto",
            dtype="auto",
            trust_remote_code=False,
            use_fast_tokenizer=True,
            load_in_4bit=False,
            load_in_8bit=False,
        )


def test_portable_load_options_normalizes_dtype_aliases() -> None:
    options = _build_portable_load_options(
        model_name="sshleifer/tiny-gpt2",
        device="cpu",
        dtype="bf16",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert options["dtype"] == "bfloat16"


def test_portable_load_options_normalizes_device_aliases() -> None:
    options = _build_portable_load_options(
        model_name="sshleifer/tiny-gpt2",
        device="GPU",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert options["device"] == "cuda"


def test_portable_to_torchforge_options_maps_quantization_and_aliases() -> None:
    portable = _build_portable_load_options(
        model_name="meta-llama/Llama-3.2-1B",
        device="cuda:0",
        dtype="fp16",
        trust_remote_code=True,
        use_fast_tokenizer=False,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge == {
        "model": "meta-llama/Llama-3.2-1B",
        "device": "cuda",
        "dtype": "float16",
        "trust_remote_code": True,
        "tokenizer_fast": False,
        "quantization": "4bit",
        "device_map": "auto",
    }


def test_portable_to_torchforge_options_rejects_unknown_loader() -> None:
    with pytest.raises(ValueError, match="unsupported loader"):
        portable_to_torchforge_options(
            {
                "loader": "torchforge",
                "model_name": "meta-llama/Llama-3.2-1B",
            }
        )
