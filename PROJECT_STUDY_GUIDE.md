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
- External gains vs matched baselines: still negative for current release candidate.
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
- Question: do trained SAEs beat random controls on consistency?

2. Architecture and scaling frontier hypothesis.
- Scripts:
  - `scripts/experiments/run_architecture_frontier_external.py`
  - `scripts/experiments/run_external_metric_scaling_study.py`
- Question: can architecture/scale choices improve external scores while preserving internal gains?

3. Assignment-aware objective hypothesis.
- Scripts:
  - `scripts/experiments/run_assignment_consistency_v2.py`
  - `scripts/experiments/run_assignment_consistency_v3.py`
- Question: can assignment-aware regularization improve consistency with low reconstruction cost and still pass external criteria?

4. Stress-gated release hypothesis.
- Scripts:
  - `scripts/experiments/run_transcoder_stress_eval.py`
  - `scripts/experiments/run_transcoder_stress_sweep.py`
  - `scripts/experiments/run_ood_stress_eval.py`
  - `scripts/experiments/run_stress_gated_release_policy.py`
- Question: do candidates remain acceptable under strict random/transcoder/OOD/external controls?

5. Proposal closure (known-circuit) hypothesis.
- Script: `scripts/experiments/run_known_circuit_recovery_closure.py`
- Question: do trained models/SAEs recover Fourier-like circuit structure above random baselines?

## 5) What We Found (Most Important)

1. Internal gains are reproducible.
- Multiple tracks show positive trained-vs-random consistency signals.

2. External transfer remains unresolved.
- Current strict gate uses LCB criteria and fails because SAEBench/CE-Bench deltas are negative.

3. Transcoder and OOD stress are currently passing for selected candidate.
- This tightened confidence in robustness controls, but external gate still blocks release.

4. New-family and closure tracks need post-fix reruns.
- Matryoshka run in cycle4 artifacts collapsed (`l0=0`) and crashed external adapter checks.
- Known-circuit closure artifacts were produced before basis-space fix and need rerun.

## 6) Why This Is Interesting Scientifically

- It demonstrates that improving an internal interpretability proxy is not enough for external validity.
- It gives a concrete benchmarked tradeoff surface rather than anecdotal results.
- It enforces strong claim hygiene via strict gates, reducing false-positive research conclusions.

## 7) What To Run Next (Highest Impact)

1. Re-run Matryoshka frontier after fixed training+adapter path.
2. Re-run known-circuit closure after model-space basis fix.
3. Run assignment-v3 with external-compatible `d_model` setup.
4. Add RouteSAE under matched budget and run grouped-LCB selection.
5. Re-run strict release gate and update canonical summaries.

## 8) Quick Validation Commands

```bash
pytest tests/unit/test_husai_custom_sae_adapter.py -q
pytest tests/unit/test_known_circuit_recovery_closure.py -q
python scripts/analysis/verify_experiment_consistency.py
```
