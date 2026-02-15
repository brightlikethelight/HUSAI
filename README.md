# HUSAI: Stable and Trustworthy SAE Features

HUSAI studies whether sparse autoencoder (SAE) features are reproducible across seeds and whether internal consistency gains transfer to external interpretability benchmarks.

## Current Status (Cycle 4 Reflective Update, 2026-02-15)

- Internal consistency improvements: supported.
- External superiority claims: not supported.
- Strict release gate: failing (`pass_all=False`).

Canonical status artifacts:
- `START_HERE.md`
- `EXECUTIVE_SUMMARY.md`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

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

3. Stress and release policy
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_transcoder_stress_sweep.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

4. Queue orchestration
- `scripts/experiments/run_b200_high_impact_queue.sh`
- `scripts/experiments/run_cycle4_followups_after_queue.sh`

## Latest Gate Metrics (Cycle 4)

From `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`:

- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Selected metrics:
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## Why This Matters

- Internal consistency gains alone are not enough for external validity.
- SAEBench and CE-Bench reward different regions of the design space.
- Strict gate policy prevents unsupported release claims.

## Start Here (Reading Order)

1. `START_HERE.md`
2. `PROJECT_STUDY_GUIDE.md`
3. `EXECUTIVE_SUMMARY.md`
4. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`

## Quick Commands

```bash
# quality and smoke
pytest tests -q
make smoke

# strict release gate
make release-gate-strict \
  TRANSCODER_RESULTS=<path/to/transcoder_stress_summary.json> \
  OOD_RESULTS=<path/to/ood_stress_summary.json> \
  EXTERNAL_SUMMARY=<path/to/external_summary.json>
```

## Reproducibility Notes

- Keep run manifests and config hashes with artifacts.
- Always compare against random controls and matched external baselines.
- Treat `pass_all=True` as a prerequisite for strong external claims.

## License

MIT (`LICENSE`).
