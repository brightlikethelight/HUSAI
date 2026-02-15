# HUSAI: Stable and Trustworthy SAE Features

HUSAI studies whether sparse autoencoder (SAE) features are reproducible across seeds and whether internal consistency gains transfer to external interpretability benchmarks.

## Current Status (Cycle 3 Final, 2026-02-15)

- Internal consistency improvements: supported.
- External superiority claims: not supported.
- Strict release gate: failing (`pass_all=False`).

Canonical status artifacts:
- `START_HERE.md`
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- `results/analysis/experiment_consistency_report.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`

## Main Research Question

Can we improve SAE feature consistency in ways that also improve external benchmark performance?

## What We Ran

1. Internal baseline and ablations
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_adaptive_l0_calibration.py`
- `scripts/experiments/run_assignment_consistency_v2.py`

2. External benchmark program
- `scripts/experiments/run_husai_saebench_custom_eval.py`
- `scripts/experiments/run_husai_cebench_custom_eval.py`
- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_external_metric_scaling_study.py`

3. Stress and release policy
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

4. B200 queue orchestration
- `scripts/experiments/run_b200_high_impact_queue.sh`

## Headline Results (Cycle 3 Queue)

Source: `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`

- Frontier multiseed completed (`4 architectures x 5 seeds`, 20 conditions).
- Scaling multiseed completed (24 conditions).
- Transcoder stress completed.
- OOD stress completed.
- Strict release gate executed and failed.

Selected metrics:
- Best SAEBench delta among tested architectures: `relu = -0.024691`.
- Best CE-Bench interpretability among tested architectures: `topk = 7.726768`.
- Transcoder delta: `-0.002227966984113039` (gate fail).
- OOD drop: `0.01445406161520213` (gate pass).
- Release gates: random pass, transcoder fail, OOD pass, external fail, `pass_all=False`.

## Why This Matters

- Internal consistency gains alone are not enough for external validity.
- SAEBench and CE-Bench reward different regions of the design space.
- The project now has strict claim hygiene: claims are blocked when gates fail.

## Start Here (Reading Order)

1. `START_HERE.md`
2. `REPO_NAVIGATION.md`
3. `RUNBOOK.md`
4. `EXECUTIVE_SUMMARY.md`
5. `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
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
