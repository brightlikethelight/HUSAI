# Learning Path: What To Read, What We Ran, and What It Means

Updated: 2026-02-16

## 1) Original Goal (From Proposal)

Core question:
- Are SAE features stable and trustworthy across seeds, or are they mostly training artifacts?

Operationalized objective in this repo:
- Improve internal consistency metrics, then verify whether those improvements transfer to external benchmarks (SAEBench, CE-Bench) under strict release gates.

## 2) Read These Files In Order

1. `START_HERE.md`
2. `PROJECT_STUDY_GUIDE.md`
3. `EXECUTIVE_SUMMARY.md`
4. `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`
7. `LIT_REVIEW.md`
8. `FINAL_BLOG.md` and `FINAL_PAPER.md`

## 3) Hypotheses We Tested

1. Trained-vs-random hypothesis.
- If SAEs are learning real structure, trained models should beat random controls.

2. Architecture frontier hypothesis.
- Architecture choice (`topk`, `relu`, `batchtopk`, `jumprelu`, `matryoshka`, `routed`) changes consistency/external tradeoffs.

3. Scaling hypothesis.
- Token budget, hook layer, and `d_sae` affect external transfer and robustness.

4. Assignment-aware objective hypothesis.
- Consistency-aware objective variants can improve reproducibility and external transfer jointly.

5. Stress-gated claim hypothesis.
- A candidate is only claim-ready if random/transcoder/OOD/external gates all pass.

6. Known-circuit closure hypothesis.
- On controlled circuit-style tasks, trained checkpoints should show positive trained-vs-random recovery margins.

## 4) Main Experiment Programs and Setup

- `scripts/experiments/run_phase4a_reproduction.py`
  - Trained vs random multiseed baseline.

- `scripts/experiments/run_core_ablations.py`
  - Sweeps over `k`, `d_sae`, and training settings.

- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_matryoshka_frontier_external.py`
- `scripts/experiments/run_routed_frontier_external.py`
  - Matched-budget architecture comparison with external eval.

- `scripts/experiments/run_external_metric_scaling_study.py`
  - Token-budget / layer / width external scaling runs.

- `scripts/experiments/run_assignment_consistency_v3.py`
  - Assignment-aware objective and checkpoint selection.

- `scripts/experiments/select_release_candidate.py`
  - Grouped-LCB candidate selection over external metrics.

- `scripts/experiments/run_transcoder_stress_sweep.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`
  - Stress and strict final gating.

- `scripts/experiments/run_known_circuit_recovery_closure.py`
  - Proposal-closure track for controlled circuit recovery.

## 5) Most Important Findings (As of Cycle 5)

Canonical evidence:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`

Findings:
1. Internal consistency improvements are real and replicated.
2. CE-Bench improved in routed/assignment sweeps.
3. SAEBench remains negative for selected release candidates.
4. Strict gate still fails (`pass_all=false`) due external criteria.
5. Selector threshold policy (`min_seeds_per_group`) can change candidate identity.

## 6) Why This Is Scientifically Important

- It separates internal interpretability progress from externally validated interpretability.
- It demonstrates that claim discipline (strict gates + CIs + controls) changes conclusions.
- It identifies the exact bottleneck: jointly improving SAEBench and CE-Bench while keeping internal consistency.

## 7) What To Do Next

1. Add SAEBench-aware objective terms to assignment-v3.
2. Increase seed support and align grouped selector thresholds.
3. Re-run joint Pareto selection with explicit SAEBench floor.
4. Re-run strict release gate on expanded candidate pool.
5. Improve known-circuit closure with trained-vs-random confidence-bound targets.
