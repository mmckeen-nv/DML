# Packet 2 boundary contract

Date: 2026-04-15
Lane: project only (`projects/dpm`)
Status: canonical packet boundary note for scratch/runtime/output reconciliation

## Purpose
Classify which paths are canonical packet inputs for the current Packet 2 read-lane reconciliation and which paths are explicitly excluded because they are runtime state, generated output, cache, or scaffolding.

This note is authoritative for narrow packet review inside `projects/dpm` and is intended to keep manifest, notes, and validation surfaces aligned.

## Canonical packet inputs
The packet may include only source-controlled, reviewable artifacts that define or validate the read-only contract seam:

### Notes
- `notes/sprint-freeze-readonly-plugin-state.md`
- `notes/read-lane-promotion-runbook-note.md`
- `notes/read-lane-promotion-manifest.md`
- `notes/read-lane-rollback-note.md`
- `notes/next-step.md`
- `notes/packet-2-boundary-contract.md`

### Contracts, schemas, and examples
- `specs/`
- `examples/dpm/config.active-read.json`
- `examples/dpm/config.observe-only.json`
- `examples/dpm/config.disabled.json`

### Validation and seam artifacts
- `scripts/continuity_recall.py`
- `scripts/run_continuity_validation.sh`
- `scripts/thread_safe_key.py`
- `scripts/dml_checkpoint_thread.sh`
- `scripts/find_thread_checkpoint.sh`
- `scripts/recall_thread_checkpoint.sh`
- `tests/`
- `PROJECT.md`

## Explicit exclusions
The following paths are outside the packet and must not be treated as promotion inputs for this reconciliation:

### Runtime state
- `runtime/`
- any live checkpoint, registry, or continuity state produced under runtime-owned paths

### Generated output
- `out/`
- rendered overlays, exports, logs, transcripts, or replay outputs produced by validation or runtime execution

### Scratch and temporary work
- `scratch/`
- ad hoc rehearsal copies, staging scratchpads, and temporary investigation material

### Cache and interpreter byproducts
- `__pycache__/`
- `*.pyc`
- other Python/interpreter cache artifacts

### Scaffolding not required as packet input
- helper/generated directories created only as execution byproducts
- local environment state not committed as contract evidence
- `examples/dpm/config.active-write.json` as an enabled target for this packet

## Review rule
If a path cannot be reviewed as a stable contract/spec/script/test/note input, classify it as excluded until a refreshed manifest explicitly promotes it into the packet.

## Alignment rule
The packet boundary is aligned only when all of the following stay true:
- the manifest checklist includes this note
- the smoke/layout test requires this note
- excluded runtime/output/cache/scaffolding paths are described consistently in notes and manifest
- validation evidence is gathered from source artifacts in `projects/dpm`, not from excluded byproducts

## Current packet claim
For Packet 2 reconciliation in this lane:
- packet inputs are limited to reviewable source artifacts under `notes/`, `specs/`, `scripts/`, `tests/`, selected read-safe example configs, and `PROJECT.md`
- `runtime/`, `out/`, `scratch/`, cache artifacts, and active-write enablement are excluded from the packet
- any future change to this boundary requires a manifest refresh before promotion review
