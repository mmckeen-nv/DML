import importlib
import sys
import types
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1] / "daystrom_dml"
_BACKENDS = _ROOT / "llm_backends"

pkg = types.ModuleType("daystrom_dml")
pkg.__path__ = [str(_ROOT)]
sys.modules.setdefault("daystrom_dml", pkg)

subpkg = types.ModuleType("daystrom_dml.llm_backends")
subpkg.__path__ = [str(_BACKENDS)]
sys.modules.setdefault("daystrom_dml.llm_backends", subpkg)

_module = importlib.import_module("daystrom_dml.llm_backends.transformers_backend")

_build_portable_load_options = _module._build_portable_load_options
portable_to_torchforge_options = _module.portable_to_torchforge_options


def test_portable_options_map_to_torchforge_shape() -> None:
    portable = _build_portable_load_options(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="CUDA",
        dtype="float16",
        trust_remote_code=True,
        use_fast_tokenizer=False,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge == {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cuda",
        "dtype": "float16",
        "trust_remote_code": True,
        "tokenizer_fast": False,
        "quantization": "4bit",
        "device_map": "auto",
    }


def test_torchforge_mapping_rejects_missing_model_name() -> None:
    with pytest.raises(ValueError, match="missing model_name"):
        portable_to_torchforge_options({"loader": "transformers"})


def test_torchforge_mapping_accepts_pretrained_model_name_or_path_alias() -> None:
    torchforge = portable_to_torchforge_options(
        {
            "loader": "transformers",
            "pretrained_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        }
    )

    assert torchforge["model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_torchforge_mapping_rejects_unknown_loader() -> None:
    with pytest.raises(ValueError, match="unsupported loader"):
        portable_to_torchforge_options({"loader": "ollama", "model_name": "x"})


def test_torchforge_mapping_rejects_conflicting_quantization() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "load_in_4bit": True,
        "load_in_8bit": True,
    }
    with pytest.raises(ValueError, match="both load_in_4bit and load_in_8bit"):
        portable_to_torchforge_options(portable)


def test_torchforge_mapping_normalizes_dtype_aliases() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "dtype": "bf16",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["dtype"] == "bfloat16"


def test_torchforge_mapping_normalizes_torch_dtype_aliases() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "dtype": "torch.float16",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["dtype"] == "float16"


def test_torchforge_mapping_normalizes_indexed_device_aliases() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "device": "cuda:3",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["device"] == "cuda"


def test_torchforge_mapping_trims_whitespace_around_device_and_dtype() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "device": "  CUDA:3  ",
        "dtype": "  BF16  ",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["device"] == "cuda"
    assert torchforge["dtype"] == "bfloat16"


def test_portable_builder_trims_whitespace_around_device_and_dtype() -> None:
    portable = _build_portable_load_options(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="  CUDA:1  ",
        dtype="  FP16  ",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert portable["device"] == "cuda"
    assert portable["dtype"] == "float16"


def test_portable_builder_normalizes_torch_dtype_aliases() -> None:
    portable = _build_portable_load_options(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="auto",
        dtype="torch.bfloat16",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert portable["dtype"] == "bfloat16"


def test_torchforge_mapping_splits_model_revision_suffix() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "openai/whisper-large-v3-turbo@refs/pr/7",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["model"] == "openai/whisper-large-v3-turbo"
    assert torchforge["revision"] == "refs/pr/7"


def test_portable_builder_splits_model_revision_suffix() -> None:
    portable = _build_portable_load_options(
        model_name="openai/whisper-large-v3-turbo@refs/pr/7",
        device="auto",
        dtype="auto",
        trust_remote_code=False,
        use_fast_tokenizer=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    assert portable["model_name"] == "openai/whisper-large-v3-turbo"
    assert portable["revision"] == "refs/pr/7"


def test_torchforge_mapping_uses_explicit_revision_field() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "openai/whisper-large-v3-turbo",
        "revision": "refs/pr/8",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["model"] == "openai/whisper-large-v3-turbo"
    assert torchforge["revision"] == "refs/pr/8"


def test_torchforge_mapping_accepts_model_revision_alias() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "openai/whisper-large-v3-turbo",
        "model_revision": "refs/pr/8",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["revision"] == "refs/pr/8"


def test_torchforge_mapping_rejects_conflicting_revision_aliases() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "openai/whisper-large-v3-turbo",
        "revision": "main",
        "model_revision": "refs/pr/8",
    }

    with pytest.raises(ValueError, match="conflicting model revision"):
        portable_to_torchforge_options(portable)


def test_torchforge_mapping_rejects_conflicting_revision_sources() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "openai/whisper-large-v3-turbo@refs/pr/7",
        "revision": "refs/pr/8",
    }

    with pytest.raises(ValueError, match="conflicting model revision"):
        portable_to_torchforge_options(portable)


def test_torchforge_mapping_preserves_explicit_empty_device_map() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "device_map": {},
    }

    torchforge = portable_to_torchforge_options(portable)

    assert "device_map" in torchforge
    assert torchforge["device_map"] == {}


def test_torchforge_mapping_passes_token_when_present() -> None:
    portable = {
        "loader": "transformers",
        "model_name": "x",
        "token": " hf_secret_token ",
    }

    torchforge = portable_to_torchforge_options(portable)

    assert torchforge["token"] == "hf_secret_token"
