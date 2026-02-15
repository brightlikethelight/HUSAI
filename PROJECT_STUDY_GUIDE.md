# Project Study Guide

Updated: 2026-02-15

This guide explains what HUSAI was trying to discover, what experiments were run, what we found, and what is still open.

## 1) What Problem Are We Solving?

We want to know whether SAE features are trustworthy:
- If we retrain with different seeds, do we recover consistent features?
- If consistency improves internally, does that improve external interpretability benchmarks?

This directly tests whether SAE-based interpretability claims are robust or artifact-prone.

## 2) Current Bottom Line

- Internal consistency gains: real.
- External gains vs matched baselines: still negative for current release-candidate settings.
- Stress-gated release policy: correctly blocks promotion (`pass_all=False`).

Primary evidence:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

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
- Multiple tracks show positive trained-vs-random consistency signals.

2. External transfer remains unresolved.
- Current strict gate uses LCB criteria and fails because SAEBench/CE-Bench deltas are negative.

3. Transcoder and OOD stress pass for the selected candidate.
- External gate remains the main blocker.

4. Post-fix reruns resolved two important validity issues.
- Known-circuit closure now evaluates all checkpoints (20/20).
- Matryoshka now trains with non-degenerate sparsity (`l0=32`) and completes external evals.

## 6) Why This Is Interesting Scientifically

- It demonstrates that improving an internal interpretability proxy is not enough for external validity.
- It gives a benchmarked tradeoff surface rather than anecdotal conclusions.
- It enforces strong claim hygiene via strict gates, reducing false-positive conclusions.

## 7) What To Run Next (Highest Impact)

1. Assignment-v3 rerun with external-compatible `d_model` setup.
2. Add RouteSAE under matched-budget protocol.
3. Re-run grouped-LCB selection with new families.
4. Re-run stress gates and strict release gate.
5. Refresh canonical summaries from latest artifacts.

## 8) Quick Validation Commands

```bash
pytest tests/unit/test_husai_custom_sae_adapter.py -q
pytest tests/unit/test_known_circuit_recovery_closure.py -q
python scripts/analysis/verify_experiment_consistency.py
```
