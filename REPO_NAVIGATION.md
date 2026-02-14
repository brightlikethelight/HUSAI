# Repository Navigation (Canonical Index)

Updated: 2026-02-14

Use this file as the entry point for understanding and operating this repository.

## 1) Start Here

- Project overview: `README.md`
- Study guide (recommended first): `PROJECT_STUDY_GUIDE.md`
- Fast setup and run path: `RUNBOOK.md`
- Quick setup and run path: `QUICK_START.md`
- Experiment provenance log: `EXPERIMENT_LOG.md`
- Current final writeups:
  - Blog: `FINAL_BLOG.md`
  - Paper: `FINAL_PAPER.md`

## 2) Core Research Pipeline

Main code path:
1. Baseline model training: `scripts/training/train_baseline.py`
2. Activation extraction: `scripts/analysis/extract_activations.py`
3. SAE training: `scripts/training/train_sae.py`
4. Stability evaluation and analyses: `scripts/analysis/` and `scripts/experiments/`

Core library modules:
- `src/data/`
- `src/models/`
- `src/training/`
- `src/analysis/`
- `src/utils/`

## 3) Highest-Impact Experiment Artifacts

- Phase 4a trained vs random:
  - `results/experiments/phase4a_trained_vs_random/`
- Core ablations:
  - `results/experiments/phase4c_core_ablations/`
- Benchmark-aligned slice:
  - `results/experiments/phase4e_external_benchmark_slice/`
- Official benchmark harness runs:
  - `results/experiments/phase4e_external_benchmark_official/`
- Tracked HUSAI custom SAEBench multi-seed evidence:
  - `docs/evidence/phase4e_husai_custom_multiseed/`
- Adaptive L0 follow-up:
  - `results/experiments/adaptive_l0_calibration/`
- Consistency-objective sweep:
  - `results/experiments/consistency_objective_sweep/`
- Transcoder stress artifacts:
  - `results/experiments/phase4e_transcoder_stress/`
- OOD stress artifacts:
  - `results/experiments/phase4e_ood_stress/`
- Release gate evaluations:
  - `results/experiments/release_stress_gates/`

## 4) Reliability and Audit Documents

- Architecture map: `ARCHITECTURE.md`
- Health audit: `AUDIT.md`
- Bug/risk list: `BUGS.md`
- Literature and landscape: `LIT_REVIEW.md`
- Novel contributions and ranked next steps: `NOVEL_CONTRIBUTIONS.md`
- Subagent-track phase-0 report: `PHASE0_SUBAGENT_REPORTS.md`

## 5) Benchmark and Claim Hygiene

- Official benchmark harness script:
  - `scripts/experiments/run_official_external_benchmarks.py`
- Stress benchmark scripts:
  - `scripts/experiments/run_transcoder_stress_eval.py`
  - `scripts/experiments/run_ood_stress_eval.py`
  - `scripts/experiments/run_stress_gated_release_policy.py`
- Artifact-claim consistency check:
  - `scripts/analysis/verify_experiment_consistency.py`
- Generated consistency report:
  - `results/analysis/experiment_consistency_report.md`

## 6) Development and CI

- CI workflow: `.github/workflows/ci.yml`
- Primary task runner: `Makefile`
- Unit/integration tests: `tests/`

Useful targets:
- `make smoke`
- `make reproduce-phase4a`
- `make ablate-core`
- `make benchmark-slice`
- `make benchmark-official`
- `make transcoder-stress`
- `make ood-stress`
- `make release-gate-strict`
- `make audit-results`

## 7) Archive and Legacy Materials

Historical or superseded artifacts are stored under:
- `archive/`
- `archive/session_notes/dec_2025/root_legacy/`

These are preserved for provenance and should not be treated as current source-of-truth unless explicitly referenced.
