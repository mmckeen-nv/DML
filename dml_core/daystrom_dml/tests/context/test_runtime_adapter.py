import pytest

from daystrom_dml.context.adapters.api_messages import APIMessageAdapter
from daystrom_dml.context.adapters.base import BaseRuntimeContextAdapter


def test_api_message_adapter_estimates_tokens_with_injected_estimator():
    calls = []

    def estimator(value):
        calls.append(value)
        if isinstance(value, list):
            return 41
        return 7

    adapter = APIMessageAdapter(token_estimator=estimator)

    assert adapter.estimate_tokens("hello world") == 7
    assert adapter.estimate_tokens([{"role": "user", "content": "hello"}]) == 41
    assert calls == ["hello world", [{"role": "user", "content": "hello"}]]


def test_api_message_adapter_renders_roles_and_keeps_authority_in_manifest_only():
    adapter = APIMessageAdapter(token_estimator=lambda value: 5)
    segments = [
        {"id": "policy", "role": "system", "content": "policy text", "metadata": {"authority": "system"}},
        {
            "id": "retrieved",
            "role": "system",
            "content": "retrieved memory",
            "source": "retrieval",
            "metadata": {"authority": "untrusted"},
        },
    ]

    rendered = adapter.render_messages(segments)

    assert rendered["messages"] == [
        {"role": "system", "content": "policy text"},
        {"role": "user", "content": "retrieved memory"},
    ]
    assert "authority" not in rendered["messages"][0]
    assert rendered["manifest"][0]["metadata"]["authority"] == "system"
    assert rendered["manifest"][1]["role_requested"] == "system"
    assert rendered["manifest"][1]["role_rendered"] == "user"
    assert rendered["manifest"][1]["metadata"]["authority"] == "untrusted"


def test_base_runtime_adapter_kv_methods_are_explicitly_unsupported():
    adapter = BaseRuntimeContextAdapter()

    assert adapter.capabilities()["kv"] is False
    assert adapter.kv_get("scope", "key") == {
        "supported": False,
        "error": "kv_get_unsupported",
        "value": None,
    }
    assert adapter.kv_put("scope", "key", "value")["error"] == "kv_put_unsupported"

    with pytest.raises(NotImplementedError):
        adapter.render_messages([])
