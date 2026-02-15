# Project Study Guide

Updated: 2026-02-15

This guide explains what HUSAI is trying to discover, what we ran, what we found, and what remains open.

## 1) What Problem Are We Solving?

We want to know whether SAE features are trustworthy:
- If we retrain with different seeds, do we recover consistent features?
- If consistency improves internally, does that improve external interpretability benchmarks?

This directly tests whether SAE-based interpretability claims are robust or artifact-prone.

## 2) Current Bottom Line

- Internal consistency gains: real.
- External gains vs matched baselines: still negative for current release candidates.
- Stress-gated release policy: correctly blocks promotion (`pass_all=False`).

Primary evidence:
- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.md`

## 3) Read in This Order

1. `START_HERE.md`
2. `EXECUTIVE_SUMMARY.md`
3. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
4. `PROPOSAL_COMPLETENESS_REVIEW.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`

## 4) Hypotheses and How We Tested Them

1. Trained-vs-random hypothesis.
- Script: `scripts/experiments/run_phase4a_reproduction.py`

2. Architecture and scaling frontier hypothesis.
- Scripts:
  - `scripts/experiments/run_architecture_frontier_external.py`
  - `scripts/experiments/run_external_metric_scaling_study.py`
  - `scripts/experiments/run_matryoshka_frontier_external.py`
  - `scripts/experiments/run_routed_frontier_external.py`

3. Assignment-aware objective hypothesis.
- Scripts:
  - `scripts/experiments/run_assignment_consistency_v2.py`
  - `scripts/experiments/run_assignment_consistency_v3.py`

4. Stress-gated release hypothesis.
- Scripts:
  - `scripts/experiments/run_transcoder_stress_eval.py`
  - `scripts/experiments/run_transcoder_stress_sweep.py`
  - `scripts/experiments/run_ood_stress_eval.py`
  - `scripts/experiments/run_stress_gated_release_policy.py`

5. Proposal closure (known-circuit) hypothesis.
- Script: `scripts/experiments/run_known_circuit_recovery_closure.py`

## 5) What We Found (Most Important)

1. Internal gains are reproducible.
- Trained-vs-random and assignment-v3 tracks show stable positive internal margins.

2. External transfer remains unresolved.
- Strict gate fails on SAEBench/CE-Bench LCB criteria.

3. Assignment-v3 external stage now runs end-to-end.
- Best lambda (`0.3`) still fails external and EV-drop acceptance criteria.

4. New architecture family added.
- Routed frontier completed under matched budget, but external deltas remain negative.

5. Stress checks are not the current bottleneck.
- Random-model, transcoder, and OOD gates pass for selected candidate.

## 6) Why This Is Interesting Scientifically

- Improving internal interpretability proxies is not sufficient for external benchmark validity.
- The negative external deltas are robust to seed grouping and LCB selection, not just point-estimate noise.
- The open problem is now precise: optimize internal consistency and external benchmark deltas jointly.

## 7) What To Run Next (Highest Impact)

1. Routed-family hyper-sweep to avoid under-activation (`train_l0` too low).
2. Assignment-v3 external-aware multi-objective sweep with larger seed set.
3. External-aware Pareto checkpointing with CI-lower-bound thresholds as hard constraints.
4. Known-circuit closure improvement track with explicit trained-vs-random confidence targets.
5. Unified W&B instrumentation in all queue scripts for live monitoring and comparability.

## 8) Quick Validation Commands

```bash
pytest tests/unit/test_husai_custom_sae_adapter.py -q
pytest tests/unit/test_assignment_consistency_v3.py -q
pytest tests/unit/test_known_circuit_recovery_closure.py -q
python scripts/analysis/verify_experiment_consistency.py
```
