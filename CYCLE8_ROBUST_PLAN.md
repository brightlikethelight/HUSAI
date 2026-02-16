# Cycle-8 Robust Pareto Push Plan

Date: 2026-02-16
Status: prepared (queue script added)
Queue script: `scripts/experiments/run_cycle8_robust_pareto_push.sh`

## Why this cycle

Cycle-5/6/7 evidence repeatedly shows the same bottleneck:
- CE-Bench can be improved, but SAEBench remains negative in release candidates.
- Strict release gate stays blocked by external LCB criteria.

Cycle-8 targets this with a robustness-first routed sweep and SAEBench-prioritized selection.

## Core changes in code

`run_routed_frontier_external.py` now supports two new regularization controls:
1. Route consistency regularization on noisy inputs
- `--robust-noise-std`
- `--route-consistency-coef`

2. Decoder diversity regularization
- `--decoder-diversity-coef`
- `--decoder-diversity-sample`

These are logged in per-seed metrics and run aggregates as:
- `route_consistency_loss`
- `decoder_diversity_loss`

## Stage design

### Stage 1: Robust routed sweep
Run family:
- `results/experiments/phase4b_routed_frontier_external_sweep_cycle8_robust/run_*`

Conditions (`b0`,`r1`..`r4`):
- baseline anchor + robustness/diversity variants around the current Pareto region.
- 4 seeds each (`42,123,456,789`) to enforce grouped uncertainty handling.

### Stage 2: Assignment-v3 external-aware Pareto sweep
Run family:
- `results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_*`

Changes:
- SAEBench-prioritized external candidate scoring.
- stricter external floors than cycle-7 candidate pass-through.

### Stage 3: Grouped-LCB selection (strict + sensitivity)
Selector outputs:
- `release_candidate_selection_cycle8_robust`
- `release_candidate_selection_cycle8_robust_min3`

Primary objective weights:
- SAEBench: 0.75
- CE-Bench: 0.20
- train EV: 0.05

### Stage 4: OOD + strict release gate
Run OOD on selected checkpoint and enforce strict external LCB gate:
- `min_saebench_delta_lcb >= 0`
- `min_cebench_delta_lcb >= 0`

## Success criteria

Cycle-8 is considered successful only if:
1. external gate passes under grouped-LCB selection, and
2. stress controls remain green (`random_model`, `transcoder`, `ood`).

## Evidence pointers

- Current cycle-7 queue log: `results/experiments/cycle7_pareto_push/run_20260216T062213Z/cycle7.log`
- Cycle-8 manifest output (when run): `results/experiments/cycle8_robust_pareto_push/run_*/manifest.json`
- Routed robustness runs: `results/experiments/phase4b_routed_frontier_external_sweep_cycle8_robust/run_*/results.json`

## Literature anchors

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://arxiv.org/abs/2509.00691
- Route SAE: https://arxiv.org/abs/2503.08200
- Robust Sparse Autoencoders: https://arxiv.org/abs/2505.24473
- Supervised Sparse Autoencoders: https://arxiv.org/abs/2505.16004
