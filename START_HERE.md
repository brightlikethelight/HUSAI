# Start Here

Updated: 2026-02-17

This is the canonical entrypoint for HUSAI.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- Cycle-8 robust queue has completed routed stage (`b0`, `r1`, `r2`, `r3`, `r4`) and assignment stage through `a3`.
- Cycle-8 assignment `a3` final acceptance is still failing (`pass_all=false`) with external deficits.
- Cycle-9 novelty queue (with supervised-proxy assignment objective) is now actively running routed stage1.
- Cycle-10 external-recovery queue is prepared for immediate launch after cycle-9:
  - `scripts/experiments/run_cycle10_external_recovery.sh`
- Last fully completed strict release gate remains failing (`pass_all=false`) due external LCB criteria.

Canonical current-status files:
- `EXECUTIVE_SUMMARY.md`
- `CYCLE7_PARETO_PLAN.md`
- `CYCLE8_ROBUST_PLAN.md`
- `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`
- `docs/evidence/cycle8_cycle9_live_snapshot_20260217T152123Z/monitoring_summary.md`
- `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`

## 2) Read in This Order

1. `START_HERE.md`
2. `CURRENT_STATUS_AND_STUDY_GUIDE.md`
3. `LEARNING_PATH.md`
4. `PROJECT_STUDY_GUIDE.md`
5. `EXECUTIVE_SUMMARY.md`
6. `CYCLE7_PARETO_PLAN.md`
7. `CYCLE8_ROBUST_PLAN.md`
8. `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`
9. `RUNBOOK.md`
10. `EXPERIMENT_LOG.md`

## 3) Core Research Questions

1. Can trained SAEs beat random controls on consistency?
2. Which architecture/sparsity/width choices improve consistency under fixed budget?
3. Do internal gains transfer to SAEBench/CE-Bench gains?
4. Can any candidate pass strict gates (random + transcoder + OOD + external)?

## 4) Current Blockers

1. External gate failure (SAEBench and CE-Bench are still negative at LCB threshold for selected candidate).
2. No candidate currently satisfies all strict release criteria simultaneously.
3. Known-circuit closure gate is still below threshold on trained-vs-random deltas.

## 5) Fastest Way to Understand the Project

1. Read `CURRENT_STATUS_AND_STUDY_GUIDE.md`.
2. Read `docs/evidence/cycle8_cycle9_live_snapshot_20260217T152123Z/monitoring_summary.md`.
3. Read `EXECUTIVE_SUMMARY.md`.
4. Read `CYCLE7_PARETO_PLAN.md`, `CYCLE8_ROBUST_PLAN.md`, and `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`.
5. Use `RUNBOOK.md` for rerun commands.
