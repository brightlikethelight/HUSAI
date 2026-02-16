# Repository Navigation (Canonical Index)

Updated: 2026-02-16

## 1) Primary Entry Points

1. `START_HERE.md`
2. `LEARNING_PATH.md`
3. `EXECUTIVE_SUMMARY.md`
4. `PROJECT_STUDY_GUIDE.md`
5. `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

## 2) Canonical Status Files

- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/selector/selection_summary.json`
- `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`

## 3) Core Code Path

Main training/eval flow:
1. Baseline model training: `scripts/training/train_baseline.py`
2. Activation extraction: `scripts/analysis/extract_activations.py`
3. SAE training: `scripts/training/train_sae.py`
4. Experiment programs: `scripts/experiments/`
5. Analysis/consistency checks: `scripts/analysis/`

## 4) Major Experiment Programs

- `results/experiments/phase4a_trained_vs_random/`
- `results/experiments/phase4c_core_ablations/`
- `results/experiments/phase4d_assignment_consistency_v3_external_sweep/`
- `results/experiments/phase4b_routed_frontier_external_sweep/`
- `results/experiments/phase4b_architecture_frontier_external_multiseed/`
- `results/experiments/phase4e_external_scaling_study_multiseed/`
- `results/experiments/release_candidate_selection_cycle5/`
- `results/experiments/release_stress_gates/`

## 5) Evidence Mirrors (Local)

- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/`
- `docs/evidence/cycle4_postfix_reruns/`
- `docs/evidence/phase4e_cebench_matched200/`

## 6) Reliability / Audit Docs

- `ARCHITECTURE.md`
- `AUDIT.md`
- `BUGS.md`
- `LIT_REVIEW.md`
- `NOVEL_CONTRIBUTIONS.md`

## 7) CI and Dev

- CI workflow: `.github/workflows/ci.yml`
- Task runner: `Makefile`
- Tests: `tests/`

## 8) Final Writeups

- Blog: `FINAL_BLOG.md`
- Paper: `FINAL_PAPER.md`
- Cycle-5 review: `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
