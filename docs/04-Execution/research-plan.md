# Research Plan (Execution Index)

This file is the execution index for the HUSAI program.

## Current Goal

Establish whether SAE variants can improve internal consistency while also improving external benchmark deltas (SAEBench and CE-Bench) under strict stress-gated release criteria.

## Primary Artifacts

- `archive/project_docs_2026_02/EXPERIMENT_PLAN.md`: historical phase plan and hypotheses.
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
- `scripts/experiments/run_cycle7_pareto_push.sh`
- `scripts/experiments/run_cycle8_robust_pareto_push.sh`
- `scripts/experiments/run_cycle9_novelty_push.sh`
- `scripts/experiments/run_cycle10_external_recovery.sh`

## Evidence Folders

- `docs/evidence/`
- `results/final_packages/cycle10_final_20260218T141310Z` (remote RunPod storage)

## Reading Order

1. `START_HERE.md`
2. `EVIDENCE_STATUS.md`
3. `EXECUTIVE_SUMMARY.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
