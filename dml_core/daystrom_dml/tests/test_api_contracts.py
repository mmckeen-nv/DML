import pytest

from daystrom_dml.api_contracts import ContractError, DaystromScope, RiskInfo, TokenBudget


def test_scope_serialization_roundtrip_with_default_tenant():
    scope = DaystromScope(client_id="client", session_id="session", thread_id="thread")

    data = scope.to_dict()
    restored = DaystromScope.from_dict(data)

    assert data["tenant_id"] == "openclaw"
    assert restored == scope


def test_scope_rejects_empty_tenant():
    with pytest.raises(ContractError):
        DaystromScope(tenant_id="")


def test_contract_from_dict_ignores_future_unknown_fields():
    scope = DaystromScope.from_dict({"tenant_id": "tenant", "future_field": "ignored"})

    assert scope.tenant_id == "tenant"
    assert not hasattr(scope, "future_field")


def test_contract_from_dict_rejects_non_dict_payload():
    with pytest.raises(ContractError):
        DaystromScope.from_dict(42)  # type: ignore[arg-type]


def test_token_budget_rejects_negative_values():
    with pytest.raises(ContractError):
        TokenBudget(limit_tokens=-1)

    with pytest.raises(ContractError):
        TokenBudget(limit_tokens=10, used_tokens=-1)


def test_token_budget_remaining_tokens():
    budget = TokenBudget(limit_tokens=100, used_tokens=25, reserved_tokens=10)
    assert budget.remaining_tokens == 65


def test_risk_info_serialization_and_validation():
    risk = RiskInfo(level="medium", reasons=["delete"], requires_confirmation=True, side_effect_classes=["delete"])
    assert RiskInfo.from_dict(risk.to_dict()) == risk

    with pytest.raises(ContractError):
        RiskInfo(level="catastrophic")
