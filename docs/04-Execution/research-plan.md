# Research Plan (Execution Index)

This file is the execution index for the HUSAI research program.

## Current Goal

Establish whether SAE variants can improve internal consistency while also improving external benchmark deltas (SAEBench and CE-Bench) under strict stress-gated release criteria.

## Primary Artifacts

- `EXPERIMENT_PLAN.md`: canonical phase plan and hypotheses.
- `EXPERIMENT_LOG.md`: run-by-run commands, run IDs, and outcomes.
- `EXECUTIVE_SUMMARY.md`: current scientific bottom line.
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`: ranked next-step priorities.

## Current High-Impact Tracks

1. External-aware assignment objective (`scripts/experiments/run_assignment_consistency_v3.py`).
2. Routed and Matryoshka frontier sweeps under matched budget.
3. Grouped uncertainty-aware release selection (LCB-based).
4. Stress-gated release policy (transcoder + OOD + external deltas).

## Queue Scripts

- `scripts/experiments/run_b200_high_impact_queue.sh`
- `scripts/experiments/run_cycle4_followups_after_queue.sh`
- `scripts/experiments/run_cycle5_external_push.sh`
- `scripts/experiments/run_cycle6_saeaware_push.sh`

## Evidence Folders

- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/`
- `docs/evidence/cycle3_queue_run_20260215T165724Z/`

## Reading Order

1. `START_HERE.md`
2. `README.md`
3. `LEARNING_PATH.md`
4. `EXPERIMENT_LOG.md`
5. `EXECUTIVE_SUMMARY.md`
