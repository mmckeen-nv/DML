# Hermes / Citizen Snips integration

This directory contains the Hermes memory provider plugin used to connect
Citizen Snips to Daystrom DML memory and DPM/personality overlays.

The plugin is intentionally memory/personality-only. It does not route model
inference through the DML frontier pipeline.

Operational notes:
- Install under a Hermes profile as `plugins/daystrom_dml/`.
- Configure Hermes with `memory.provider: daystrom_dml` and point
  `memory.daystrom_dml.integration_dir` / `storage_dir` at the desired DML
  runtime bundle and store.
- `maintenance_scan.py` is dry-run by default and reports obvious polluted
  records. Use `--apply` only after reviewing the report; it quarantines via
  metadata rather than deleting records.
