# Project Study Guide

Updated: 2026-02-16

This guide explains what HUSAI is trying to discover, what we ran, what we found, and what remains open.

## 1) What Problem Are We Solving?

We want to know whether SAE features are trustworthy:
- If we retrain with different seeds, do we recover consistent features?
- If consistency improves internally, does that improve external interpretability benchmarks?

## 2) Current Bottom Line

- Internal consistency gains: real.
- External gains vs matched baselines: still negative in strict release settings.
- CE-Bench has improved in cycle-5 assignment/routed sweeps, but SAEBench is still the blocking gap.
- Stress-gated release policy remains correctly blocking promotion (`pass_all=False`).

Primary evidence:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`

## 3) Read in This Order

1. `START_HERE.md`
2. `LEARNING_PATH.md`
3. `EXECUTIVE_SUMMARY.md`
4. `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
5. `PROPOSAL_COMPLETENESS_REVIEW.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

## 4) Hypotheses and How We Tested Them

1. Trained-vs-random hypothesis.
- `scripts/experiments/run_phase4a_reproduction.py`

2. Architecture and scaling frontier hypothesis.
- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_external_metric_scaling_study.py`
- `scripts/experiments/run_matryoshka_frontier_external.py`
- `scripts/experiments/run_routed_frontier_external.py`

3. Assignment-aware objective hypothesis.
- `scripts/experiments/run_assignment_consistency_v2.py`
- `scripts/experiments/run_assignment_consistency_v3.py`

4. Stress-gated release hypothesis.
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_transcoder_stress_sweep.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

5. Proposal closure (known-circuit) hypothesis.
- `scripts/experiments/run_known_circuit_recovery_closure.py`

## 5) What We Found (Most Important)

1. Internal gains are reproducible.
2. External transfer remains unresolved.
3. Routed `expert_topk` fixed routed sparsity collapse and improved CE-Bench.
4. Assignment `d_sae=2048` produced best CE-Bench among assignment sweeps.
5. Default grouped selector settings can hide promising groups if seed threshold is too high.

## 6) Why This Is Interesting Scientifically

- Internal interpretability proxies and external benchmark behavior are not aligned by default.
- Objective design and selector policy interact strongly with scientific conclusions.
- Reliability-first gating prevents unsupported claims and makes negative results actionable.

## 7) What To Run Next (Highest Impact)

1. SAEBench-aware assignment objective extension.
2. Larger-seed assignment/routed reruns with aligned selector grouping thresholds.
3. Joint Pareto selection requiring explicit SAEBench floor.
4. Selector diagnostics for dropped groups.
5. Determinism env hardening across all queue scripts.

## 8) Quick Validation Commands

```bash
pytest tests/unit/test_release_policy_selector.py -q
pytest tests/unit/test_routed_frontier_modes.py -q
pytest tests/unit/test_assignment_consistency_v3.py -q
python scripts/analysis/verify_experiment_consistency.py
```
