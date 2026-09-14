"""Machine-readable maturity inventory; implementation is not graduation."""
from copy import deepcopy

_PRODUCTION_STATUS = {
    "schema_version": "dml-production-status-v1",
    "production_ready": False,
    "stable": [],
    "candidate": ["jsonl-v1-validation", "journal-v1-integrity", "append-only-receipts-v1", "sqlite-snapshot-projection-v1", "coalesced-projection-delta-v1", "bounded-projection-worker-v1", "transactional-outbox-v1", "explicit-outbox-migration-v1", "receipted-memory-retirement-v1", "receipted-memory-supersession-v1", "scoped-exact-retrieval", "bounded-rendered-context"],
    "experimental": ["native-kv-reuse", "ann", "automatic-abstraction-promotion", "dpm", "dcn", "kv-fabric-routing"],
    "research": ["physical-rdma-gds-transfer", "self-improving-memory-policy"],
    "remaining_release_gates": ["idempotent-write-receipts", "all-component-atomicity", "complete-adapter-extraction",
                                "all-mutation-crash-campaign", "real-agent-baseline-value", "100k-turn-agent-campaign",
                                "supported-platform-power-loss", "durable-full-decision-replay"],
}


def production_status() -> dict:
    return deepcopy(_PRODUCTION_STATUS)
