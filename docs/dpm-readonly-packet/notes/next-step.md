# Next Step

Current lane state:
- this note describes the archived read-only packet lane
- DPM has since moved beyond the packet snapshot into runtime implementation
- active-read overlays and active-write preference graph updates now exist in `daystrom_dml.personality_matrix`
- the current implementation plan is tracked in `docs/dpm-runtime-roadmap.md`

Immediate next action:
- follow `docs/dpm-runtime-roadmap.md`
- start with live wrapper/config integration, then API/CLI inspection and graph governance

Guardrails:
- preserve explicit current-turn user instruction precedence over any plugin-derived guidance
- keep retrieval/order/audit behavior aligned across the plugin contract, lifecycle/schema docs, and runtime coherence artifacts
- keep archived packet claims distinct from current runtime claims
- do not let packet fixtures imply production behavior unless a runtime test or smoke exists
- keep `runtime/`, `out/`, `scratch/`, and cache artifacts out of the tracked packet
