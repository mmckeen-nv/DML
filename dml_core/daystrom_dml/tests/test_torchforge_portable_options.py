from __future__ import annotations

from daystrom_dml.llm_backends.transformers_backend import _build_portable_load_options


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
