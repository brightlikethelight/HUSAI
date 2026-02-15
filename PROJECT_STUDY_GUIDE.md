# Project Study Guide

Updated: 2026-02-15

This is the learner-first guide for understanding what HUSAI did, what we found, and what remains.

## 1) Is the Repo Fully Cleaned and Organized?

Short answer: functionally yes for the active research path; legacy materials are still kept intentionally.

What is now clean and current:
- Canonical entrypoint: `START_HERE.md`
- Canonical index: `REPO_NAVIGATION.md`
- Repro commands: `RUNBOOK.md`
- Cycle-3 final truth: `EXECUTIVE_SUMMARY.md`
- Artifact-backed closure: `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- Claim audit: `results/analysis/experiment_consistency_report.md`

What remains intentionally non-minimal:
- Historical scripts/results/docs under `archive/` and older experiment folders are preserved for provenance.

## 2) Reading Order (Best Path)

1. `START_HERE.md`
- Current truth and experiment map in one file.

2. `EXECUTIVE_SUMMARY.md`
- Top issues, current gate status, and prioritized follow-ups.

3. `PROPOSAL_COMPLETENESS_REVIEW.md`
- Honest comparison against original proposal goals.

4. `RUNBOOK.md`
- How to run and verify experiments.

5. `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- Final queue outcomes and gate breakdown.

6. `EXPERIMENT_LOG.md`
- Exact command and artifact provenance.

## 3) Core Research Question and Hypotheses

Primary question:
- Can SAE consistency improvements transfer to external benchmark improvements?

Hypotheses tested:
1. Training signal hypothesis
- Trained SAEs should beat random controls on consistency metrics.

2. Structure hypothesis
- `k` and `d_sae` control a consistency-quality tradeoff.

3. Architecture frontier hypothesis
- Different SAE families shift external metrics differently.

4. Transfer hypothesis
- Internal consistency gains should correlate with better SAEBench/CE-Bench performance.

5. Stress-gate hypothesis
- Strong claims should require random-model, transcoder, OOD, and external gates to pass.

## 4) Experiments We Ran (Setup, Findings, Why Important)

### A) Phase 4a: Trained vs Random Reproduction
- Script: `scripts/experiments/run_phase4a_reproduction.py`
- Artifact: `results/experiments/phase4a_trained_vs_random/results.json`
- Setup: TopK SAE with multi-seed trained/random comparison.
- Found: small but non-zero trained-over-random gain.
- Why important: establishes a real baseline signal.

### B) Phase 4c: Core Ablations (`k`, `d_sae`)
- Script: `scripts/experiments/run_core_ablations.py`
- Artifact: `results/experiments/phase4c_core_ablations/`
- Setup: controlled sweeps over sparsity and width.
- Found: meaningful tradeoff surface, not noise.
- Why important: identifies levers that actually move internal consistency.

### C) Assignment-Aware Consistency v2
- Script: `scripts/experiments/run_assignment_consistency_v2.py`
- Artifact: `results/experiments/phase4d_assignment_consistency_v2/`
- Setup: Hungarian-assignment regularization sweep.
- Found: internal consistency improved strongly; external acceptance still failed.
- Why important: confirms internal objective progress alone is insufficient.

### D) Official/Custom External Benchmark Integration
- Scripts:
  - `scripts/experiments/run_official_external_benchmarks.py`
  - `scripts/experiments/run_husai_saebench_custom_eval.py`
  - `scripts/experiments/run_husai_cebench_custom_eval.py`
- Artifacts:
  - `results/experiments/phase4e_external_benchmark_official/`
  - `docs/evidence/phase4e_cebench_matched200/`
- Setup: matched-budget CE-Bench baseline + custom HUSAI checkpoints.
- Found: large negative deltas vs matched CE-Bench baseline.
- Why important: prevents invalid external-performance claims.

### E) Architecture Frontier (Multiseed, B200)
- Script: `scripts/experiments/run_architecture_frontier_external.py`
- Artifact: `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/`
- Setup: `topk,relu,batchtopk,jumprelu` over 5 seeds, matched budget.
- Found:
  - SAEBench best delta: `relu = -0.024691`.
  - CE-Bench best mean interpretability: `topk = 7.726768`.
  - All CE-Bench deltas vs matched baseline remain strongly negative.
- Why important: shows cross-benchmark objective conflict is robust.

### F) External Scaling Study (Multiseed, B200)
- Script: `scripts/experiments/run_external_metric_scaling_study.py`
- Artifact: `results/experiments/phase4e_external_scaling_study_multiseed/run_20260214T212435Z/`
- Setup: grid over token budget (`10k,30k`), hook layer (`0,1`), `d_sae` (`1024,2048`), 3 seeds.
- Found:
  - CE-Bench improves with layer 1 and larger `d_sae`.
  - SAEBench deltas worsen in those same regions.
- Why important: transfer tension is structured and reproducible.

### G) Stress-Gated Release (B200)
- Scripts:
  - `scripts/experiments/run_transcoder_stress_eval.py`
  - `scripts/experiments/run_ood_stress_eval.py`
  - `scripts/experiments/run_stress_gated_release_policy.py`
- Artifacts:
  - `results/experiments/phase4e_transcoder_stress_b200/run_20260214T224242Z/transcoder_stress_summary.json`
  - `results/experiments/phase4e_ood_stress_b200/run_20260214T224309Z/ood_stress_summary.json`
  - `results/experiments/release_stress_gates/run_20260214T225029Z/release_policy.json`
- Found:
  - transcoder gate fails (`transcoder_delta = -0.002227966984113039`)
  - OOD gate passes (`ood_drop = 0.01445406161520213`)
  - external gate fails; overall `pass_all=False`
- Why important: strict gate prevents overclaiming and defines exact blockers.

## 5) What We Learned (Bottom Line)

1. Internal progress is real but does not yet transfer externally.
2. SAEBench and CE-Bench favor different design regions.
3. Reliability engineering and claim gating are now strong.
4. The unresolved scientific problem is now precise: improve both internal and external objectives simultaneously.

## 6) Highest-Impact Remaining Work

1. Multi-objective training/selection to improve external deltas without losing internal consistency.
2. Add one new architecture family (Matryoshka/RouteSAE/HierarchicalTopK) under identical matched protocol.
3. Close known-ground-truth circuit recovery promised in the original proposal.
4. Tighten deterministic CUDA reproducibility (`CUBLAS_WORKSPACE_CONFIG` in run scripts).
5. Keep release language hard-coupled to strict gate outcomes.

## 7) Fast Verification Commands

```bash
# quick quality check
pytest tests -q
make smoke

# strict gate check
make release-gate-strict \
  TRANSCODER_RESULTS=<path/to/transcoder_stress_summary.json> \
  OOD_RESULTS=<path/to/ood_stress_summary.json> \
  EXTERNAL_SUMMARY=<path/to/external_summary.json>

# consistency audit
python scripts/analysis/verify_experiment_consistency.py
```
