from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1] / "daystrom_dml"
_BACKENDS = _ROOT / "llm_backends"


pkg = types.ModuleType("daystrom_dml")
pkg.__path__ = [str(_ROOT)]
sys.modules.setdefault("daystrom_dml", pkg)

subpkg = types.ModuleType("daystrom_dml.llm_backends")
subpkg.__path__ = [str(_BACKENDS)]
sys.modules.setdefault("daystrom_dml.llm_backends", subpkg)


class _FakeTensor:
    shape = (1, 1)

    def to(self, _device: str) -> "_FakeTensor":
        return self


class _FakeTokenizer:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> "_FakeTokenizer":
        cls.calls.append((model_name, kwargs))
        return cls()

    def __call__(self, _prompt: str, return_tensors: str = "pt") -> dict[str, _FakeTensor]:
        return {"input_ids": _FakeTensor()}


class _FakeModel:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> "_FakeModel":
        cls.calls.append((model_name, kwargs))
        return cls()

    def eval(self) -> None:
        return None

    def to(self, _device: str) -> "_FakeModel":
        return self


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeMPS:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeBackends:
    mps = _FakeMPS()


def test_transformers_backend_passes_revision_to_hf_loaders(monkeypatch) -> None:
    _FakeTokenizer.calls.clear()
    _FakeModel.calls.clear()

    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda(),
        backends=_FakeBackends(),
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        AutoModelForCausalLM=_FakeModel,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    module = importlib.import_module("daystrom_dml.llm_backends.transformers_backend")
    backend_cls = module.TransformersBackend

    backend = backend_cls(
        model_name="openai/whisper-large-v3-turbo@refs/pr/7",
        device="cpu",
        dtype="float32",
    )

    assert backend is not None
    assert _FakeTokenizer.calls == [
        (
            "openai/whisper-large-v3-turbo",
            {
                "use_fast": True,
                "trust_remote_code": False,
                "revision": "refs/pr/7",
            },
        )
    ]
    assert _FakeModel.calls == [
        (
            "openai/whisper-large-v3-turbo",
            {
                "trust_remote_code": False,
                "revision": "refs/pr/7",
                "torch_dtype": "float32",
            },
        )
    ]
