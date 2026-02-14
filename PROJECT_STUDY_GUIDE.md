# Project Study Guide

Updated: 2026-02-14

This is the canonical onboarding guide for understanding HUSAI end-to-end.

## 1) Is the repo cleaned, updated, organized?

Short answer: mostly yes for the active research path, not fully minimal for legacy material.

What is clean and current:
- Canonical run path and commands are documented in `RUNBOOK.md`.
- Canonical map is in `REPO_NAVIGATION.md`.
- Final state and critical status are in `EXECUTIVE_SUMMARY.md` and `FINAL_READINESS_REVIEW.md`.
- High-impact cycle artifacts are under `docs/evidence/` and referenced from summaries.
- CI smoke + incremental quality gates are active in `.github/workflows/ci.yml`.

What is intentionally still present:
- Historical exploration scripts/docs are kept for provenance (`archive/`, exploratory scripts in `scripts/experiments/`).
- This means the repo is research-complete and navigable, but not aggressively pruned.

## 2) Reading Order (Do This First)

1. `README.md`
- High-level project framing and top-level findings.

2. `REPO_NAVIGATION.md`
- Canonical map of where things live.

3. `RUNBOOK.md`
- How to run the pipeline and key experiments now.

4. `EXECUTIVE_SUMMARY.md`
- Current cycle-level conclusions and top issues.

5. `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- Detailed closure report of the top-5 high-impact follow-ups.

6. `FINAL_READINESS_REVIEW.md`
- Critical answer to “are we actually finished?” and what remains.

7. `EXPERIMENT_LOG.md`
- Run-by-run provenance, commands, and outcomes.

8. `docs/evidence/` (phase folders)
- Artifact-backed evidence for each major claim.

## 3) Core Research Questions and Hypotheses

Primary question:
- Can we improve SAE feature consistency in a way that also improves external benchmark performance?

Hypotheses tested:
1. Training signal hypothesis:
- Trained SAEs should outperform random controls on stability metrics.

2. Sparsity/width tradeoff hypothesis:
- `k` and `d_sae` sweeps should show consistency-quality tradeoffs.

3. External transfer hypothesis:
- Internal gains should transfer to SAEBench/CE-Bench deltas.

4. Architecture frontier hypothesis:
- Alternative SAE families can move the external frontier under matched budgets.

5. Assignment-aware objective hypothesis:
- Matching-aware regularization can improve internal consistency without unacceptable EV drop.

6. Stress-gated release hypothesis:
- Claims should be blocked unless random-model, transcoder, OOD, and external gates pass.

## 4) Experiments We Ran (What, Setup, Findings, Why It Matters)

### A) Phase 4a: Trained vs Random Reproduction
- Script: `scripts/experiments/run_phase4a_reproduction.py`
- Artifact: `results/experiments/phase4a_trained_vs_random/results.json`
- Setup: TopK SAE (`d_model=128`, `d_sae=1024`, `k=32`) with multi-seed checkpoint comparison.
- Key finding:
  - trained PWMCC mean `0.300059`
  - random PWMCC mean `0.298829`
  - delta `+0.001230`, p-value `0.00863`
- Why important:
  - Confirms non-zero training signal, but effect size is small; this sets the baseline realism for later claims.

### B) Phase 4c: Core Ablations (`k` and `d_sae`)
- Script: `scripts/experiments/run_core_ablations.py`
- Artifact: `results/experiments/phase4c_core_ablations/run_20260212T091848Z/results.json`
- Setup: controlled sweeps over sparsity and width.
- Key finding:
  - best `k` condition: `k=8` with small positive delta (`+0.00341`)
  - best width condition: `d_sae=64` with larger delta (`+0.12009`, ratio `1.52x`)
- Why important:
  - Shows stability can be moved by structure choices; tradeoff surface is real, not noise.

### C) Adaptive L0 Follow-up (Fair Control)
- Script: `scripts/experiments/run_adaptive_l0_calibration.py`
- Artifacts:
  - `results/experiments/adaptive_l0_calibration/run_20260212T145416Z/results.json`
  - `results/experiments/adaptive_l0_calibration/run_20260212T145727Z/results.json`
- Setup: selected low-`k` vs matched control (`k=32`).
- Key finding:
  - selected `k=4`: delta `+0.07567`, conservative LCB `+0.07256`
  - matched `k=32`: delta `+0.01866`, LCB `+0.01676`
- Why important:
  - Strongest validated internal-gain lever in this repo.

### D) Official External Benchmark Harness
- Script: `scripts/experiments/run_official_external_benchmarks.py`
- Artifacts:
  - SAEBench run: `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
  - CE-Bench official evidence: `docs/evidence/phase4e_cebench_official/`
- Setup: official-style harness with reproducibility manifests and logs.
- Key finding:
  - Infrastructure now executes benchmark protocols reproducibly.
- Why important:
  - Removed a major credibility blocker (preflight-only state).

### E) Direct HUSAI CE-Bench + Matched Baseline
- Scripts:
  - `scripts/experiments/run_husai_cebench_custom_eval.py`
  - `scripts/experiments/run_official_external_benchmarks.py`
- Artifacts:
  - baseline: `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`
  - custom deltas: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`
- Setup: matched CE-Bench budget (`max_rows=200`) for fair comparison.
- Key finding:
  - baseline interpretability max `47.95`
  - tested custom checkpoints are much lower (roughly `-40` to `-44` delta).
- Why important:
  - External gap is quantified clearly; no overclaiming.

### F) Architecture Frontier (External)
- Script: `scripts/experiments/run_architecture_frontier_external.py`
- Artifact: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md`
- Setup: matched-budget sweep over `topk,relu,batchtopk,jumprelu`.
- Key finding:
  - SAEBench ranking favors ReLU/JumpReLU.
  - CE-Bench ranking favors TopK/BatchTopK.
- Why important:
  - No single winner; objective conflict must be handled explicitly.

### G) External Scaling Study
- Script: `scripts/experiments/run_external_metric_scaling_study.py`
- Artifact: `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md`
- Setup: `token_budget x hook_layer x d_sae` grid (8 conditions).
- Key finding:
  - layer 1 / larger width helps CE-Bench but worsens SAEBench delta on average.
- Why important:
  - Confirms structured cross-metric tradeoff, not random drift.

### H) Assignment-Aware Consistency Objective v2
- Script: `scripts/experiments/run_assignment_consistency_v2.py`
- Artifact: `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`
- Setup: Hungarian-assignment consistency regularization sweep.
- Key finding:
  - best lambda `0.2`
  - delta PWMCC `+0.070804`
  - LCB `+0.054419`
  - EV drop `0.000878`
  - external gate still fails (`external_delta=-0.132836`)
- Why important:
  - Strong internal improvement is possible but still not sufficient for external acceptance.

### I) Stress-Gated Release Policy
- Script: `scripts/experiments/run_stress_gated_release_policy.py`
- Artifact: `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`
- Setup: strict gates over random-model, transcoder, OOD, and external criteria.
- Key finding:
  - random gate pass
  - transcoder/OOD missing at that run
  - external gate fail
  - overall `pass_all=False`
- Why important:
  - Prevents narrative overreach and enforces evidence discipline.

## 5) What Is Newly Added for Final Polish

New scripts to close stress-plumbing gaps:
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_ood_stress_eval.py`

New make targets:
- `make transcoder-stress`
- `make ood-stress`
- `make release-gate-strict`

These feed release gating directly and convert missing-artifact failures into measurable outcomes.

## 6) Why This Project Is Interesting

1. It is a realistic negative/nuanced result story.
- Internal consistency gains do not automatically imply external interpretability gains.

2. It demonstrates claim hygiene in practice.
- Benchmarks, manifests, and strict gates are first-class, not an afterthought.

3. It yields a concrete research frontier.
- The open problem is now clear: find methods that improve both internal consistency and external scores under matched budgets.

## 7) Fast Commands for You

```bash
# sanity + tests
make smoke
pytest tests -q

# core evidence refresh
make reproduce-phase4a
make ablate-core

# stress artifacts and strict gate
make transcoder-stress
make ood-stress
make release-gate-strict \
  TRANSCODER_RESULTS=<path/to/transcoder_stress_summary.json> \
  OOD_RESULTS=<path/to/ood_stress_summary.json> \
  EXTERNAL_SUMMARY=<path/to/external_summary.json>
```
