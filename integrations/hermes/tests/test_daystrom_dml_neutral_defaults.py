"""A generic installation must not assume an operator or assistant identity."""
from integrations.hermes.plugins import daystrom_dml as plugin


def test_provider_defaults_are_neutral_and_explicit_config_wins(monkeypatch):
    monkeypatch.setattr(plugin.DaystromDMLProvider, "_load_provider_config", lambda self: {})
    provider = plugin.DaystromDMLProvider()
    assert provider.client_id == "hermes"
    assert provider._relationship_id == "relationship:runtime"
    assert provider._project_id == "project:runtime"
    monkeypatch.setattr(plugin.DaystromDMLProvider, "_load_provider_config", lambda self: {
        "client_id": "client:configured", "relationship_id": "relationship:configured",
        "project_id": "project:configured",
    })
    configured = plugin.DaystromDMLProvider()
    assert configured.client_id == "client:configured"
    assert configured._relationship_id == "relationship:configured"
    assert configured._project_id == "project:configured"


def test_hygiene_filters_unknown_persona_wrappers_without_a_name_allowlist():
    for persona in ("Fixture Robot", "Sample_Agent"):
        assert plugin._semantic_value(f"{persona} durable turn memory. I prefer numbered steps.") == "I prefer numbered steps."
        classified = plugin._classify_turn_memory(
            f"Completed {persona} turn. User: I prefer numbered steps.", "Acknowledged.")
        assert classified["keep"] is False
        result = plugin.DaystromDMLProvider()._safe_context_from_payload({
            "raw_context": f"- Memory policy: Identity: {persona}. Preferences: fixture text."
        })
        assert result == ""


def test_current_focus_is_not_replaced_with_a_developer_workflow():
    value = "Offload the sample invoice export to Codex."
    assert plugin._semantic_label_value("Current focus", value) == value
