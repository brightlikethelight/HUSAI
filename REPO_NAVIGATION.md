# Repository Navigation (Canonical Index)

Updated: 2026-02-15

Use this file as the canonical map of the repository.

## 1) Entry Points

1. `START_HERE.md` (recommended first)
2. `README.md` (project snapshot)
3. `RUNBOOK.md` (execution and reproduction)
4. `EXECUTIVE_SUMMARY.md` (cycle-level conclusions)
5. `EXPERIMENT_LOG.md` (run-by-run provenance)

## 2) Canonical Status and Gate Truth

- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- `results/analysis/experiment_consistency_report.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`
- `FINAL_READINESS_REVIEW.md`

## 3) Core Code Path

Main training/eval flow:
1. Baseline model training: `scripts/training/train_baseline.py`
2. Activation extraction: `scripts/analysis/extract_activations.py`
3. SAE training: `scripts/training/train_sae.py`
4. Analyses and experiments: `scripts/analysis/`, `scripts/experiments/`

Core package:
- `src/data/`
- `src/models/`
- `src/training/`
- `src/analysis/`
- `src/utils/`

## 4) Major Experiment Programs

- Phase 4a trained-vs-random: `results/experiments/phase4a_trained_vs_random/`
- Core ablations: `results/experiments/phase4c_core_ablations/`
- Assignment consistency v2: `results/experiments/phase4d_assignment_consistency_v2/`
- External official/custom benchmark runs: `results/experiments/phase4e_external_benchmark_official/`
- Architecture frontier multiseed: `results/experiments/phase4b_architecture_frontier_external_multiseed/`
- External scaling multiseed: `results/experiments/phase4e_external_scaling_study_multiseed/`
- Transcoder stress: `results/experiments/phase4e_transcoder_stress_b200/`
- OOD stress: `results/experiments/phase4e_ood_stress_b200/`
- Release gates: `results/experiments/release_stress_gates/`

## 5) Evidence Mirrors (Pulled Local)

- `docs/evidence/cycle3_queue_final/`
- `docs/evidence/phase4b_architecture_frontier_external/`
- `docs/evidence/phase4e_external_scaling_study/`
- `docs/evidence/phase4e_cebench_matched200/`

## 6) Reliability and Audit Docs

- `ARCHITECTURE.md`
- `AUDIT.md`
- `BUGS.md`
- `LIT_REVIEW.md`
- `NOVEL_CONTRIBUTIONS.md`
- `PHASE0_SUBAGENT_REPORTS.md`

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

Historical/superseded artifacts are intentionally preserved for provenance:
- `archive/`
- `archive/session_notes/dec_2025/root_legacy/`

Do not treat legacy content as current source-of-truth unless explicitly referenced.
