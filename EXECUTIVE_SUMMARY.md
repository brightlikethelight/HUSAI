# Executive Summary (Cycle 4 Reflective Update)

Date: 2026-02-15

## Repo Purpose

HUSAI tests a central mechanistic-interpretability question: whether SAE feature consistency gains are real across seeds, and whether those gains transfer to external benchmark validity (SAEBench, CE-Bench) under strict release gates.

## Current Scientific Bottom Line

- Internal consistency progress: **real and replicated**.
- External competitiveness: **not yet achieved**.
- Reliability and reproducibility hygiene: **strong**.
- Strict release decision: **fail** (`pass_all=False`).

Canonical current-status artifacts:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
- `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.md`
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_summary.md`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`

## Cycle 4 Key Metrics (Gate Run)

Run IDs:
- Followups orchestrator: `run_20260215T190004Z`
- Release gate: `run_20260215T191137Z`
- Transcoder sweep: `run_20260215T184609Z`
- OOD stress: `run_20260215T190404Z`

Gate outcomes:
- random_model: `True`
- transcoder: `True`
- ood: `True`
- external_saebench: `False`
- external_cebench: `False`
- pass_all: `False`

Numerical highlights:
- `trained_random_delta_lcb = 6.18e-05`
- `transcoder_delta = +0.0049161`
- `ood_drop = 0.0209946`
- `saebench_delta_ci95_low = -0.0447896`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.4670`

## Post-Fix Rerun Highlights (This Pass)

1. Known-circuit rerun now evaluates all checkpoints.
- `checkpoints_evaluated: 20` (previously 0)
- gate still fails, but now result is valid and interpretable.

2. Matryoshka rerun no longer crashes and no longer collapses.
- `train_l0_mean = 32.0` (previous cycle4 artifact had `l0=0`).
- `train_ev_mean = 0.6166`.
- external summaries produced for all 3 seeds.
- external deltas still negative.

## Top Issues (Current)

1. `P0` External benchmark gap remains large (CE-Bench delta heavily negative).
2. `P0` No candidate currently satisfies both external gates jointly.
3. `P1` Assignment-v3 external stage remains unresolved due dimensional mismatch in current artifact run.
4. `P1` Strict release gate still blocks promotion (`pass_all=False`).
5. `P2` W&B instrumentation remains inconsistent across scripts.

## What Changed in This Update

Code fixes:
- `scripts/experiments/husai_custom_sae_adapter.py`
  - dead-decoder-row repair + encoder masking before custom-SAE norm checks.
- `scripts/experiments/run_known_circuit_recovery_closure.py`
  - corrected SAE Fourier overlap geometry to model-space projection.
- `scripts/experiments/run_matryoshka_frontier_external.py`
  - switched training path to HUSAI TopK with dead-feature recovery auxiliary loss.

Tests added:
- `tests/unit/test_husai_custom_sae_adapter.py`
- `tests/unit/test_known_circuit_recovery_closure.py`

Docs synchronized to cycle4 truth and reflective status.

## Highest-Leverage Next 5 (Ranked)

1. Assignment-v3 rerun with external-compatible `d_model` configuration.
2. Add RouteSAE under matched budget and run full external protocol.
3. Re-run grouped-LCB candidate selection including new family runs.
4. Re-run OOD/transcoder stress on the new selected candidate.
5. Re-run strict gate and update canonical summaries only from latest artifacts.

## Read Next

1. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
2. `START_HERE.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
