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

## 5) Follow-Up Experiments (Paper Section 4.11)

- Runner: `scripts/experiments/run_all_followup_experiments.sh`
- 1-layer ground truth: `scripts/experiments/exp_1layer_ground_truth.py`
- Subspace stability: `scripts/experiments/exp_subspace_stability.py`
- Effective rank predictor: `scripts/experiments/exp_effective_rank_predictor.py`
- Contrastive stability: `scripts/experiments/exp_contrastive_stability.py`
- Intervention stability: `scripts/experiments/exp_intervention_stability.py`
- Dictionary pinning: `scripts/experiments/exp_dictionary_pinning.py`
- Pythia-70M scaling: `scripts/experiments/exp_pythia70m_stability.py`
- Tests: `tests/unit/test_followup_experiments.py`

## 6) Writing and Presentation

- Paper: `paper/sae_stability_paper.md`
- Technical report: `paper/FINAL_PAPER.md`
- Blog: `FINAL_BLOG.md`
- Literature: `LIT_REVIEW.md`
- Experiment roadmap: `docs/04-Execution/EXPERIMENT_PLAN_2026_02_20.md`
- Slide package: `docs/05-Presentation/cycle10_readout/`

## 7) Evidence

- Local evidence root: `docs/evidence/`
- Evidence policy: `EVIDENCE_STATUS.md`

## 8) Historical Documents

Planning and review documents from earlier cycles are archived to `archive/project_docs_2026_02/` for traceability.
