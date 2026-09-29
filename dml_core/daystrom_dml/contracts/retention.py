"""Data-independent retention limits for the candidate receipt-journal profile."""
from copy import deepcopy

from ..services.retention import UNINSPECTED_SURFACES


_RETENTION_CONTRACT = {
    "schema_version": "dml-retention-contract-v1",
    "maturity": "candidate",
    "journal_schema_versions": [2, 3, 4],
    "retirement_is_erasure": False,
    "physical_erasure_supported": False,
    "erasure_proven": False,
    "retirement": {
        "effect": "suppress_from_normal_scoped_retrieval",
        "preserves_text_vectors_metadata_and_history": True,
        "releases_capacity": False,
        "cascades_to_derived_memories": False,
        "retracts_previously_returned_context": False,
    },
    "history": {
        "receipt_replay": "exact_immutable_historical_result",
        "retention": "indefinite_under_supported_operations",
        "receipt_or_outbox_expiry_supported": False,
        "receipt_or_outbox_compaction_supported": False,
    },
    "inspection": {
        "operation": "inspect_memory_retention",
        "coverage": "known_structured_references_in_one_journal",
        "scope": "exact_tenant_client_session_instance",
        "revision_pinned": True,
        "returns_payloads": False,
        "mutates_authoritative_state": False,
        "known_reference_forms": ["direct_records", "first_level_promotion_source_proofs"],
        "surfaces": ["current_items", "current_lineage", "journal_snapshot", "receipts",
                     "outbox_states"],
        "no_match_does_not_prove_erasure": True,
        "sqlite_runtime_sidecar_activity_may_occur": True,
    },
    "uninspected_surfaces": list(UNINSPECTED_SURFACES),
}


def retention_contract() -> dict:
    """Return a detached capability description, without opening any store."""
    return deepcopy(_RETENTION_CONTRACT)
