"""Machine-readable maturity inventory; implementation is not graduation."""
from copy import deepcopy

from .profile import PROFILE_ID, production_profile_contract

_PRODUCTION_STATUS = {
    "schema_version": "dml-production-status-v1",
    "production_ready": False,
    "stable": [],
    "candidate": [PROFILE_ID, "exact-model-input-v1", "jsonl-v1-validation", "journal-v1-integrity", "append-only-receipts-v1", "sqlite-snapshot-projection-v1", "coalesced-projection-delta-v1", "bounded-projection-worker-v1", "transactional-outbox-v1", "explicit-outbox-migration-v1", "receipted-memory-retirement-v1", "receipted-memory-supersession-v1", "receipted-content-update-v1", "receipted-first-level-promotion-v1", "scoped-retention-inspection-v1", "scoped-exact-retrieval", "bounded-rendered-context"],
    "experimental": ["native-kv-reuse", "ann", "automatic-abstraction-promotion", "dpm", "dcn", "kv-fabric-routing"],
    "research": ["physical-rdma-gds-transfer", "self-improving-memory-policy"],
    "remaining_release_gates": [
        "supported-profile-crash-recovery-filesystems",
        "persisted-format-migration-coverage", "mixed-operation-concurrency",
        "live-agent-semantic-outcome-harness", "fair-baseline-value",
        "continuous-1k-10k-and-100k-campaign", "durable-decision-replay-audit-retention",
        "release-qualification-support",
    ],
    "deferred_milestones": ["remaining-legacy-retrieval-lifecycle-extraction", "native-kv-restore-identity"],
    "remaining_first_release_milestones": 8,
    "remaining_deferred_milestones": 2,
}


def production_status() -> dict:
    result = deepcopy(_PRODUCTION_STATUS)
    result["supported_profiles"] = [production_profile_contract()]
    return result
