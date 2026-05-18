# DPM Lifecycle/Config Contract

Status: canonical draft
Layer: post-foundation additive plugin layer on top of continuity metadata

## Purpose
This artifact defines the operator-facing lifecycle/config contract for enabling DPM safely without runtime wiring. It standardizes the allowed lifecycle modes, config fields, defaults, and validation rules that future runtime integrations must honor.

This contract is intentionally implementation-agnostic:
- no durable/main integration is required
- no storage backend is mandated
- no inference or retrieval implementation is implied

## Canonical lifecycle enum
The DPM lifecycle mode is a closed enum with exactly four allowed values:
- `disabled`
- `observe-only`
- `active-read`
- `active-write`

Any other value is invalid.

## Mode semantics

### `disabled`
- DPM does not participate in retrieval
- DPM does not emit overlays
- DPM does not write plugin state
- all DPM feature flags are effectively inert

### `observe-only`
- DPM may inspect candidate sources for evaluation
- DPM must not shape runtime behavior
- DPM must not write durable plugin state
- audit output may describe what would have happened

### `active-read`
- DPM may read eligible plugin state
- DPM may emit bounded overlay output
- DPM must not write or mutate durable plugin state
- audit output is required for any non-empty read result

### `active-write`
- DPM may read eligible plugin state
- DPM may emit bounded overlay output
- DPM may write plugin state, subject to explicit write-safe contracts
- all writes must be attributable, scoped, and auditable

## Canonical config object
A future runtime integration must treat the DPM config object as having this conceptual structure:

```yaml
version: 1
plugin: dpm
mode: disabled
read:
  allow_thread: true
  allow_project: true
  allow_relationship: true
  allow_preference_graph: true
write:
  enabled: false
  require_explicit_runtime_support: true
  allowed_scopes: []
audit:
  enabled: true
  include_excluded_sources: true
  max_sources: 20
  max_overlay_chars: 2000
safety:
  explicit_user_override: true
  fail_closed_on_invalid_mode: true
  cross_scope_fallback_requires_compatibility: true
```

## Required fields
- `version`: integer config contract version
- `plugin`: must equal `dpm`
- `mode`: one of the four canonical lifecycle values
- `read`: read controls for eligible source families
- `write`: write controls; still subordinate to lifecycle mode
- `audit`: audit controls and output bounds
- `safety`: mandatory safety invariants

## Field rules

### Top-level
- `version` must be `1` for this contract revision
- `plugin` must be the literal string `dpm`
- `mode` must be one of the canonical lifecycle values

### `read`
Read flags may narrow retrieval eligibility, but they must not broaden scope beyond the base DPM contract.

Fields:
- `allow_thread`: boolean
- `allow_project`: boolean
- `allow_relationship`: boolean
- `allow_preference_graph`: boolean

Rules:
- in `disabled`, all read activity is suppressed regardless of these values
- in `observe-only`, reads may occur for evaluation but must not influence runtime behavior
- in `active-read` and `active-write`, read flags determine which source families are eligible for consideration

### `write`
Fields:
- `enabled`: boolean
- `require_explicit_runtime_support`: boolean
- `allowed_scopes`: list of strings from `thread`, `project`, `relationship`, `preference-graph`

Rules:
- in `disabled`, writes are forbidden
- in `observe-only`, writes are forbidden even if `enabled: true`
- in `active-read`, writes are forbidden even if `enabled: true`
- in `active-write`, writes are allowed only when `enabled: true` and runtime support exists
- `allowed_scopes` must be empty unless `mode` is `active-write`
- future runtimes may support only a strict subset of scopes and must fail closed on unsupported write scopes

### `audit`
Fields:
- `enabled`: boolean
- `include_excluded_sources`: boolean
- `max_sources`: integer, minimum `1`, recommended default `20`
- `max_overlay_chars`: integer, minimum `1`, recommended default `2000`

Rules:
- any mode other than `disabled` should keep `audit.enabled: true`
- audit bounds are hard limits, not hints
- future runtimes may emit less than the configured maximum but must not exceed it

### `safety`
Fields:
- `explicit_user_override`: boolean
- `fail_closed_on_invalid_mode`: boolean
- `cross_scope_fallback_requires_compatibility`: boolean

Rules:
- all three fields must be `true` in compliant configurations
- a runtime must reject or fail closed on configs that set these invariants to `false`

## Safe defaults
The default safe posture is:
- `mode: disabled`
- writes disabled
- audit enabled
- explicit user override enforced
- compatibility checks required for fallback

This makes DPM opt-in and reviewable by default.

## Validation rules
A config is valid under this contract only if all of the following hold:
1. top-level required fields are present
2. `plugin == "dpm"`
3. `version == 1`
4. `mode` is one of the four canonical values
5. write scopes are empty unless `mode == "active-write"`
6. `write.enabled` does not grant write authority outside `active-write`
7. all safety invariants are `true`
8. audit bounds are positive integers

## Example compliance matrix

### Valid: disabled
- no reads take effect
- no writes take effect
- safe default for shipping a dormant plugin

### Valid: observe-only
- reads allowed for evaluation only
- runtime behavior unchanged by DPM
- safe for dry-run and audit comparisons

### Valid: active-read
- read-only retrieval and overlay generation allowed
- durable state mutation forbidden
- safe for initial behavioral rollout with no persistence

### Valid: active-write
- full plugin lifecycle allowed
- still subordinate to safety invariants and explicit user instructions
- requires explicit runtime support before any write path becomes effective

## Invalid examples
- unknown mode such as `enabled`
- `plugin: continuity` instead of `plugin: dpm`
- `version: 2` under this contract revision
- `active-read` with non-empty `write.allowed_scopes`
- any config with `safety.explicit_user_override: false`
- any config with `audit.max_sources: 0`

## Relationship to the plugin contract
This lifecycle/config contract narrows operator inputs to a safe, deterministic shape. The existing DPM plugin contract remains the source of truth for:
- retrieval precedence
- merge/conflict behavior
- audit semantics
- explicit-user-instruction override behavior

Future runtime integrations must satisfy both contracts.
