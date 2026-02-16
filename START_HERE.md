# Start Here

Updated: 2026-02-16

This is the canonical entrypoint for HUSAI.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- CE-Bench improved in cycle-5 routed/assignment sweeps, but SAEBench remains negative.
- Strict release gate is still failing (`pass_all=false`).

Canonical current-status files:
- `EXECUTIVE_SUMMARY.md`
- `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md` (historical baseline)
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`

## 2) Read in This Order

1. `START_HERE.md`
2. `LEARNING_PATH.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `EXECUTIVE_SUMMARY.md`
5. `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

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
3. Read `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`.
4. Read `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`.
5. Use `RUNBOOK.md` for rerun commands.
