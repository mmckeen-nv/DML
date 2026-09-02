# Daystrom KV Fabric Control Plane

## Scope of this change

- The existing DCM + vLLM cooperative data path (GPU APC residency with pinned CPU offload) remains the proven runtime path and is unchanged.
- `daystrom_dml.context.kv_fabric` is control-plane only: strict contracts and deterministic transfer planning metadata.
- No transport runtime is implemented here. This module never performs RDMA, GDS, filesystem, object-store, or network payload movement.

## Why this seam exists

- Future adapters (KVBM/NIXL/SGLang/TensorRT-LLM and engine-native implementations) need a shared payload-free contract for tier declarations, object descriptors, and route plans.
- Endpoints declare implementation identity, authority domain digests, admission bounds, accepted runtime-compatibility identities, and supported mechanisms; the planner deterministically validates compatibility and returns an integrity-bound path.
- Adapters should delegate physical movement to KVBM/NIXL or engine-native capabilities instead of reimplementing transport stacks in DCM.

## Object model boundaries

- KV checkpoint block objects are represented by `KVObjectDescriptor`, bound to `ExecutionCheckpointIdentity` digests, and derive their authority-domain digest from the checkpoint's complete `DaystromScope` rather than accepting a caller-supplied domain label.
- Runtime compatibility is explicit and opt-in: each object carries a `runtime_compatibility_digest` from exact `(runtime_id, runtime_version, adapter_id)`, and each endpoint must declare that digest in `accepted_runtime_identity_digests` for eligibility.
- Route planning uses deterministic bounded graph search (max hops, max endpoints, no cycles) with nonnegative-cost minimization, descending-priority preference, and lexicographic tie-breaks.
- Transfer tickets are integrity-bound and HMAC-authenticated with an injected process-local signing key; replay prevention remains process-local single-use tracking.
- DEC expert tensors are intentionally out of scope and should be introduced later as a separate typed object class with its own compatibility and authority contracts.
