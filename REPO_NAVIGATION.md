# Repository Navigation (Canonical Index)

Updated: 2026-02-15

## 1) Primary Entry Points

1. `START_HERE.md`
2. `EXECUTIVE_SUMMARY.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`

## 2) Canonical Status Files

- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.md`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/selector/selection_summary.json`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`

## 3) Core Code Path

Main training/eval flow:
1. Baseline model training: `scripts/training/train_baseline.py`
2. Activation extraction: `scripts/analysis/extract_activations.py`
3. SAE training: `scripts/training/train_sae.py`
4. Experiment programs: `scripts/experiments/`
5. Analysis/consistency checks: `scripts/analysis/`

Core package:
- `src/data/`
- `src/models/`
- `src/training/`
- `src/analysis/`
- `src/utils/`

## 4) Major Experiment Programs

- Phase 4a trained-vs-random: `results/experiments/phase4a_trained_vs_random/`
- Core ablations: `results/experiments/phase4c_core_ablations/`
- Assignment objective: `results/experiments/phase4d_assignment_consistency_v2/`, `results/experiments/phase4d_assignment_consistency_v3/`, `results/experiments/phase4d_assignment_consistency_v3_external/`
- External official/custom: `results/experiments/phase4e_external_benchmark_official/`
- Architecture frontier multiseed: `results/experiments/phase4b_architecture_frontier_external_multiseed/`
- Matryoshka frontier: `results/experiments/phase4b_matryoshka_frontier_external/`
- Routed frontier: `results/experiments/phase4b_routed_frontier_external/`
- External scaling multiseed: `results/experiments/phase4e_external_scaling_study_multiseed/`
- Transcoder sweep: `results/experiments/phase4e_transcoder_stress_sweep_b200/`
- OOD stress: `results/experiments/phase4e_ood_stress_b200/`
- Release gates: `results/experiments/release_stress_gates/`
- Known-circuit closure: `results/experiments/known_circuit_recovery_closure/`

## 5) Evidence Mirrors (Local)

- `docs/evidence/cycle4_followups_run_20260215T220728Z/`
- `docs/evidence/cycle4_followups_run_20260215T212614Z/`
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

Useful targets:
- `make smoke`
- `make reproduce-phase4a`
- `make ablate-core`
- `make benchmark-official`
- `make transcoder-stress`
- `make ood-stress`
- `make release-gate-strict`
- `make audit-results`

## 8) Final Writeups

- Blog: `FINAL_BLOG.md`
- Paper: `FINAL_PAPER.md`

## 9) Legacy Material

Historical artifacts are kept for provenance and should not override canonical status:
- `archive/`
- `archive/session_notes/dec_2025/root_legacy/`
