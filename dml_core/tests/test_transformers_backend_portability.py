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
