# Daystrom Contract Compatibility Policy

Daystrom contract artifacts are versioned compatibility boundaries between DML,
DPM, DCN, DIP, host-agent plugins, provider APIs, and future MCP tools.

## Current stable contracts

- `daystrom-cognitive-packet-v1`
  - JSON Schema artifact: `docs/contracts/cognitive-packet-v1.schema.json`
  - Runtime registry: `daystrom_dml.contracts.ContractRegistry`
  - Validator: `daystrom_dml.contracts.validate_cognitive_packet_v1(...)`

## Versioning rules

1. **V1 is additive-only.** New optional fields may be added when older clients
   can ignore them safely.
2. **Breaking changes require V2.** Renaming fields, changing required field
   semantics, removing fields, or tightening previously accepted payloads beyond
   explicit validation gates requires a new packet version.
3. **Runtime constants are canonical.** Code should use exported constants such
   as `COGNITIVE_PACKET_V1`, not duplicate string literals.
4. **Schemas are audit artifacts.** JSON schemas are checked in so API, MCP, and
   gateway integrations can validate payload shape without importing internal
   implementation modules.
5. **Current-turn intent still wins.** Contracts describe wire shape; they do not
   grant authority for memory, preference, cognition, or inference writeback.

## Compatibility posture

- Unknown fields remain allowed in schema artifacts for forward-compatible
  adapters.
- Strict validators enforce only core safety invariants in runtime gates.
- Provider OpenAPI snapshots should be regenerated intentionally and reviewed
  when public route shapes change.
