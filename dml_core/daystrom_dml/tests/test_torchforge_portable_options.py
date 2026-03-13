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


def test_portable_load_options_normalizes_indexed_devices() -> None:
    cuda_options = _build_portable_load_options(
        model_name="sshleifer/tiny-gpt2",
        device="CUDA:1",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    cpu_options = _build_portable_load_options(
        model_name="sshleifer/tiny-gpt2",
        device="CPU:2",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert cuda_options["device"] == "cuda"
    assert cpu_options["device"] == "cpu"


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


def test_portable_to_torchforge_options_accepts_torch_dtype_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "torch_dtype": "torch.bfloat16",
        }
    )

    assert torchforge["dtype"] == "bfloat16"


def test_portable_to_torchforge_options_normalizes_torch_float_alias_to_float32() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "torch_dtype": "torch.float",
        }
    )

    assert torchforge["dtype"] == "float32"


def test_portable_to_torchforge_options_preserves_explicit_device_map() -> None:
    portable = _build_portable_load_options(
        model_name="meta-llama/Llama-3.2-1B",
        device="auto",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    portable["device_map"] = {"": "cuda:0"}

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["device_map"] == {"": "cuda:0"}


def test_portable_to_torchforge_options_splits_model_revision_suffix() -> None:
    portable = _build_portable_load_options(
        model_name="meta-llama/Llama-3.2-1B@refs/pr/42",
        device="auto",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"
    assert torchforge["revision"] == "refs/pr/42"


def test_portable_to_torchforge_options_accepts_model_revision_alias() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "meta-llama/Llama-3.2-1B",
        "model_revision": "refs/pr/42",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["revision"] == "refs/pr/42"


def test_portable_to_torchforge_options_rejects_conflicting_revision_aliases() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "meta-llama/Llama-3.2-1B",
        "revision": "main",
        "model_revision": "refs/pr/42",
    }

    with pytest.raises(ValueError, match="conflicting model revision"):
        portable_to_torchforge_options(portable)


def test_portable_to_torchforge_options_rejects_conflicting_revision_sources() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "meta-llama/Llama-3.2-1B@main",
        "revision": "refs/pr/42",
    }

    with pytest.raises(ValueError, match="conflicting model revision"):
        portable_to_torchforge_options(portable)


def test_portable_load_options_normalizes_hf_uri_prefix() -> None:
    options = _build_portable_load_options(
        model_name="hf://meta-llama/Llama-3.2-1B",
        device="auto",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert options["model_name"] == "meta-llama/Llama-3.2-1B"


def test_portable_to_torchforge_options_normalizes_huggingface_uri_prefix_with_revision() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "huggingface://meta-llama/Llama-3.2-1B@refs/pr/12",
        }
    )

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"
    assert torchforge["revision"] == "refs/pr/12"


def test_portable_to_torchforge_options_accepts_matching_revision_sources() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "meta-llama/Llama-3.2-1B@main",
        "revision": "main",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"
    assert torchforge["revision"] == "main"


def test_portable_to_torchforge_options_coerces_string_booleans() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "device": "auto",
            "dtype": "auto",
            "trust_remote_code": "true",
            "use_fast_tokenizer": "0",
            "load_in_4bit": "false",
            "load_in_8bit": "1",
        }
    )

    assert torchforge["trust_remote_code"] is True
    assert torchforge["tokenizer_fast"] is False
    assert torchforge["quantization"] == "8bit"


def test_portable_to_torchforge_options_sets_auto_device_map_for_quantized_loads() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "load_in_4bit": True,
        }
    )

    assert torchforge["quantization"] == "4bit"
    assert torchforge["device_map"] == "auto"


def test_portable_to_torchforge_options_uses_explicit_revision_when_suffix_is_empty() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B@",
            "revision": " refs/pr/42 ",
        }
    )

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"
    assert torchforge["revision"] == "refs/pr/42"


def test_portable_to_torchforge_options_carries_local_files_only_flag() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "local_files_only": "yes",
        }
    )

    assert torchforge["local_files_only"] is True


def test_portable_to_torchforge_options_passes_cache_dir_subfolder_tokenizer_revision_and_attn_implementation() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "cache_dir": " /models/hf-cache ",
            "subfolder": " text-generation ",
            "tokenizer_revision": " refs/pr/17 ",
            "attn_implementation": " flash_attention_2 ",
            "token": " hf_token_abc123 ",
        }
    )

    assert torchforge["cache_dir"] == "/models/hf-cache"
    assert torchforge["subfolder"] == "text-generation"
    assert torchforge["tokenizer_revision"] == "refs/pr/17"
    assert torchforge["attn_implementation"] == "flash_attention_2"
    assert torchforge["token"] == "hf_token_abc123"


def test_portable_to_torchforge_options_accepts_use_auth_token_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "use_auth_token": " hf_alias_token ",
        }
    )

    assert torchforge["token"] == "hf_alias_token"


def test_portable_to_torchforge_options_prefers_token_over_use_auth_token_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "token": " hf_primary_token ",
            "use_auth_token": " hf_alias_token ",
        }
    )

    assert torchforge["token"] == "hf_primary_token"


def test_portable_to_torchforge_options_ignores_blank_cache_dir_subfolder_tokenizer_revision_and_attn_implementation() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "cache_dir": "   ",
            "subfolder": "",
            "tokenizer_revision": "  ",
            "attn_implementation": "   ",
        }
    )

    assert "cache_dir" not in torchforge
    assert "subfolder" not in torchforge
    assert "tokenizer_revision" not in torchforge
    assert "attn_implementation" not in torchforge


def test_portable_to_torchforge_options_rejects_non_boolean_flags() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        portable_to_torchforge_options(
            {
                "loader": "transformers",
                "model_name": "meta-llama/Llama-3.2-1B",
                "load_in_4bit": "maybe",
            }
        )


def test_portable_to_torchforge_options_rejects_conflicting_quantization_flags() -> None:
    with pytest.raises(ValueError, match="both load_in_4bit and load_in_8bit"):
        portable_to_torchforge_options(
            {
                "loader": "transformers",
                "model_name": "meta-llama/Llama-3.2-1B",
                "load_in_4bit": True,
                "load_in_8bit": True,
            }
        )


def test_portable_to_torchforge_options_rejects_non_boolean_local_files_only() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        portable_to_torchforge_options(
            {
                "loader": "transformers",
                "model_name": "meta-llama/Llama-3.2-1B",
                "local_files_only": "sometimes",
            }
        )


def test_portable_to_torchforge_options_accepts_model_alias_when_model_name_is_missing() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model": "meta-llama/Llama-3.2-1B",
        }
    )

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"


def test_portable_to_torchforge_options_accepts_pretrained_model_name_or_path_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B",
        }
    )

    assert torchforge["model"] == "meta-llama/Llama-3.2-1B"


def test_portable_to_torchforge_options_accepts_use_fast_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "model_name": "meta-llama/Llama-3.2-1B",
            "use_fast": "false",
        }
    )

    assert torchforge["tokenizer_fast"] is False


def test_portable_to_torchforge_options_rejects_unknown_loader() -> None:
    with pytest.raises(ValueError, match="unsupported loader"):
        portable_to_torchforge_options(
            {
                "loader": "torchforge",
                "model_name": "meta-llama/Llama-3.2-1B",
            }
        )
