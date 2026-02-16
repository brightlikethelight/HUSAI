# Start Here

Updated: 2026-02-16

This is the canonical entrypoint for HUSAI.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- Cycle-7 is actively testing Pareto tradeoffs; routed stage (`p1..p5`) is complete, assignment stage is running.
- Cycle-8 robustness queue is launched and waiting behind cycle-7.
- Last fully completed strict release gate is still failing (`pass_all=false`) due external LCB criteria.

Canonical current-status files:
- `EXECUTIVE_SUMMARY.md`
- `CYCLE7_PARETO_PLAN.md`
- `CYCLE8_ROBUST_PLAN.md`
- `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`

## 2) Read in This Order

1. `START_HERE.md`
2. `LEARNING_PATH.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `EXECUTIVE_SUMMARY.md`
5. `CYCLE7_PARETO_PLAN.md`
6. `CYCLE8_ROBUST_PLAN.md`
7. `RUNBOOK.md`
8. `EXPERIMENT_LOG.md`

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

1. Read `LEARNING_PATH.md`.
2. Read `PROJECT_STUDY_GUIDE.md`.
3. Read `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`.
4. Read `CYCLE7_PARETO_PLAN.md` and `CYCLE8_ROBUST_PLAN.md`.
5. Use `RUNBOOK.md` for rerun commands.
