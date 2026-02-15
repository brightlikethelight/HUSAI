# Start Here

Updated: 2026-02-15

This is the canonical entrypoint for the current state of HUSAI.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- External deltas vs matched baselines are still negative in release-candidate settings.
- Strict release gate currently fails (`pass_all=false`).

Canonical current-status files:
- `EXECUTIVE_SUMMARY.md`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`

## 2) Read in This Order

1. `START_HERE.md`
2. `EXECUTIVE_SUMMARY.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
5. `REPO_NAVIGATION.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

## 3) Core Research Questions

1. Can trained SAEs beat random controls on consistency?
2. Which architecture/sparsity/width choices improve consistency?
3. Do internal gains transfer to SAEBench/CE-Bench gains?
4. Can any candidate pass strict gates (random + transcoder + OOD + external)?

## 4) Current Blockers

1. External gate failure (SAEBench and CE-Bench both negative at LCB level).
2. New-family track (Matryoshka) requires rerun after collapse/adapter fixes.
3. Known-circuit closure requires rerun after basis-space fix.
4. Assignment-v3 needs external-compatible `d_model` setup.

## 5) Fastest Way to Understand the Project

1. Read `PROJECT_STUDY_GUIDE.md`.
2. Read cycle4 evidence summaries under `docs/evidence/cycle4_followups_run_20260215T190004Z/`.
3. Use `RUNBOOK.md` to rerun only the high-impact followup queue.
