# Repository Navigation

Updated: 2026-03-05

## 1) First Five Files To Read

1. `START_HERE.md`
2. `EVIDENCE_STATUS.md`
3. `EXECUTIVE_SUMMARY.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`

## 2) Critical Research Path (Core 10 Files)

1. `scripts/experiments/run_phase4a_reproduction.py`
2. `scripts/experiments/run_core_ablations.py`
3. `scripts/experiments/run_assignment_consistency_v3.py`
4. `scripts/experiments/run_architecture_frontier_external.py`
5. `scripts/experiments/run_routed_frontier_external.py`
6. `scripts/experiments/run_husai_saebench_custom_eval.py`
7. `scripts/experiments/run_husai_cebench_custom_eval.py`
8. `scripts/experiments/select_release_candidate.py`
9. `scripts/experiments/run_stress_gated_release_policy.py`
10. `scripts/experiments/run_official_external_benchmarks.py`

## 3) Core Library Code

- Data: `src/data/modular_arithmetic.py`
- Models: `src/models/simple_sae.py`
- Training loop: `src/training/train_sae.py`
- Stability metrics: `src/analysis/feature_matching.py`
- Configs: `src/utils/config.py`

## 4) Quality and Reproducibility

- CI workflow: `.github/workflows/ci.yml`
- Smoke script: `scripts/ci/smoke_pipeline.sh`
- Tests: `tests/`

## 5) Writing and Presentation

- Blog: `FINAL_BLOG.md`
- Paper-style summary: `FINAL_PAPER.md`
- Literature: `LIT_REVIEW.md`
- Experiment plan: `EXPERIMENT_PLAN.md`
- Slide package: `docs/05-Presentation/cycle10_readout/`

## 6) Evidence

- Local evidence root: `docs/evidence/`
- Evidence policy: `EVIDENCE_STATUS.md`
