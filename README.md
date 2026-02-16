# HUSAI: Stable and Trustworthy SAE Features

HUSAI studies whether sparse autoencoder (SAE) features are reproducible across seeds and whether internal consistency gains transfer to external interpretability benchmarks.

## Current Status (Cycle 5 Update, 2026-02-16)

- Internal consistency improvements: supported.
- CE-Bench deltas: improved in cycle-5 routed/assignment sweeps.
- SAEBench deltas: still negative in release-candidate settings.
- Strict release gate: failing (`pass_all=False`).

Canonical status artifacts:
- `START_HERE.md`
- `EXECUTIVE_SUMMARY.md`
- `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`

## Main Research Question

Can we improve SAE feature consistency in ways that also improve external benchmark performance?

## What We Ran

1. Internal baseline and ablations
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_assignment_consistency_v2.py`
- `scripts/experiments/run_assignment_consistency_v3.py`

2. External benchmark program
- `scripts/experiments/run_husai_saebench_custom_eval.py`
- `scripts/experiments/run_husai_cebench_custom_eval.py`
- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_external_metric_scaling_study.py`
- `scripts/experiments/run_matryoshka_frontier_external.py`
- `scripts/experiments/run_routed_frontier_external.py`

3. Stress and release policy
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_transcoder_stress_sweep.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

4. Queue orchestration
- `scripts/experiments/run_b200_high_impact_queue.sh`
- `scripts/experiments/run_cycle4_followups_after_queue.sh`
- `scripts/experiments/run_cycle5_external_push.sh`

## Latest Gate Metrics (Cycle 5)

From `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`:

- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Selected metrics:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## Why This Matters

- Internal consistency gains alone are not enough for external validity.
- CE-Bench and SAEBench respond differently to candidate improvements.
- Strict gate policy prevents unsupported release claims.

## Start Here (Reading Order)

1. `START_HERE.md`
2. `LEARNING_PATH.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `EXECUTIVE_SUMMARY.md`
5. `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

## Quick Commands

```bash
pytest tests -q
make smoke

make release-gate-strict \
  TRANSCODER_RESULTS=<path/to/transcoder_stress_summary.json> \
  OOD_RESULTS=<path/to/ood_stress_summary.json> \
  EXTERNAL_SUMMARY=<path/to/external_summary.json>
```

## License

MIT (`LICENSE`).
