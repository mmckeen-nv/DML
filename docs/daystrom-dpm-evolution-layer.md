# Daystrom Personality Matrix evolution layer

The DPM evolution layer gives an agent personality room to breathe without making safety or user obedience negotiable. It records interaction signals into a sliding graph and renders a bounded personality overlay at runtime.

## What evolves

DPM maintains fast and slow values for traits such as warmth, directness, playfulness, technicality, initiative, social restraint, mechanicality, and continuity drive. Fast state reacts to recent interactions; slow self moves cautiously. Context edges let the same agent adapt differently for creative voice work, build/debug work, reef support, and general collaboration.

## What does not evolve

Every evolution overlay carries immutable hard laws:

- Explicit current-turn user instructions override personality tendencies.
- Safety, privacy, and secret-hygiene constraints are immutable.
- Personality may choose tone and initiative, not disobedience or harm.
- When uncertain, ask or take the safest useful action rather than asserting autonomy against the human.

These are rendered with the overlay and reasserted on save so the evolution graph cannot silently mutate them away.

## CLI

Record an observed interaction:

```bash
python3 openclaw-wrapper/scripts/dml_memory.py \
  --storage-dir /path/to/store \
  --no-require-gpu \
  dpm-observe \
  --prompt "The user asked for a warmer rewrite" \
  --response "Produced a warmer rewrite" \
  --meta '{"task_type":"creative_personality","feedback_valence":0.4}'
```

The command writes `dpm_evolution_graph.json` in the selected store when DPM is enabled in active-write mode.

## Hermes integration

The Hermes memory provider calls `dpm-observe` during `sync_turn` for durable, hygienic turns. It also extracts clean current-user preference text for the DPM preference graph so DPM learns from the user's correction rather than from wrapper summaries.

## Verification

Focused regression bundle:

```bash
python3 -m pytest dml_core/daystrom_dml/tests/test_dml.py -k "personality_evolution or personality_matrix"
```

The tests cover interaction recording, fast/slow trait updates, context adaptation, and hard-law rendering.
