# Learning Path: What To Read, What We Ran, and What It Means

Updated: 2026-02-18

## 1) Original Goal

Core question:

- Are SAE features stable and trustworthy across seeds, and do internal consistency gains transfer to external benchmarks?

Operational success condition:

- pass strict release gates with positive external criteria.

## 2) Read These Files In Order

1. `START_HERE.md`
2. `CANONICAL_DOCS.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `EXECUTIVE_SUMMARY.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`
7. `FINAL_PAPER.md`
8. `HIGH_IMPACT_FOLLOWUPS_REPORT.md`

## 3) Hypotheses We Tested

1. Trained-vs-random hypothesis.
- Trained SAEs should outperform random controls if they learn real structure.

2. Architecture frontier hypothesis.
- Architecture choice (`topk`, `relu`, `batchtopk`, `jumprelu`, `matryoshka`, `routed`) changes internal/external tradeoffs.

3. Scaling hypothesis.
- Token budget, hook layer, and `d_sae` influence external transfer and robustness.

4. Assignment-aware objective hypothesis.
- Consistency-aware objective variants may improve reproducibility and external transfer jointly.

5. Stress-gated claim hypothesis.
- A candidate is only claim-ready if random/transcoder/OOD/external gates all pass.

## 4) Main Experiment Programs and Setup

- Internal baselines:
  - `scripts/experiments/run_phase4a_reproduction.py`
  - `scripts/experiments/run_core_ablations.py`

- External frontier/scaling:
  - `scripts/experiments/run_architecture_frontier_external.py`
  - `scripts/experiments/run_matryoshka_frontier_external.py`
  - `scripts/experiments/run_routed_frontier_external.py`
  - `scripts/experiments/run_external_metric_scaling_study.py`

- Assignment objective:
  - `scripts/experiments/run_assignment_consistency_v3.py`

- Selection + stress gate:
  - `scripts/experiments/select_release_candidate.py`
  - `scripts/experiments/run_transcoder_stress_sweep.py`
  - `scripts/experiments/run_ood_stress_eval.py`
  - `scripts/experiments/run_stress_gated_release_policy.py`

## 5) Final Findings

1. Internal consistency improvements are real and replicated.
2. Stress controls pass for selected candidates.
3. External transfer remains unresolved under strict criteria.
4. Final strict gate is `pass_all=false` in cycle-10.

## 6) Why This Is Scientifically Important

- It separates internal interpretability progress from externally validated interpretability.
- It enforces claim discipline with confidence-aware and stress-gated criteria.
- It isolates the main bottleneck: jointly improving SAEBench and CE-Bench under strict gates.

## 7) Final Evidence Anchor

- `/workspace/HUSAI/results/final_packages/cycle10_final_20260218T141310Z/meta/FINAL_INDEX.md`
