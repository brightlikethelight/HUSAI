# Start Here

Updated: 2026-02-15

This is the canonical entrypoint for the current state of HUSAI.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- External deltas vs matched baselines remain negative in release-candidate settings.
- Strict release gate currently fails (`pass_all=false`).

Canonical current-status files:
- `EXECUTIVE_SUMMARY.md`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
- `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.md`
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_summary.md`

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
2. Assignment-v3 external-aware run still needs `d_model`-compatible setup.
3. No current candidate satisfies all strict release criteria simultaneously.

## 5) Fastest Way to Understand the Project

1. Read `PROJECT_STUDY_GUIDE.md`.
2. Read `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`.
3. Read cycle4 evidence folders under `docs/evidence/`.
4. Use `RUNBOOK.md` for rerun commands.
