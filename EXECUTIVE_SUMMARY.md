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
- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/selector/selection_summary.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/assignment_external/results.json`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`

## Cycle 4 Key Metrics (Latest Gate)

Run IDs:
- Followups orchestrator (step 3-5): `run_20260215T220728Z`
- Release gate: `run_20260215T223154Z`
- Assignment-v3 external: `run_20260215T220737Z`
- Routed frontier: `run_20260215T213621Z`
- Matryoshka frontier: `run_20260215T212623Z`

Gate outcomes:
- random_model: `True`
- transcoder: `True`
- ood: `True`
- external_saebench: `False`
- external_cebench: `False`
- pass_all: `False`

Numerical highlights:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.015173514260201082`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## New High-Impact Coverage in Latest Pass

1. Assignment-v3 external path is now complete (no external `d_model` mismatch blocker).
- Best lambda selected: `0.3`.
- Internal LCB remains strong (`0.823699951171875`) but external deltas still fail acceptance.

2. New family added and benchmarked: routed frontier.
- `scripts/experiments/run_routed_frontier_external.py`
- External deltas remain negative despite routing regularization and matched budget.

3. Grouped LCB selection and strict gate were rerun on updated pool.
- Selected candidate remains `topk_seed123` from frontier multiseed.

## Top Issues (Current)

1. `P0` External benchmark gap remains large (especially CE-Bench delta vs matched baseline).
2. `P0` No candidate currently satisfies both external gates jointly.
3. `P1` Known-circuit closure gate still fails trained-vs-random thresholds.
4. `P2` W&B instrumentation is still not enabled in current remote runs.
5. `P2` Some historical docs referenced stale cycle4 run IDs (updated in this pass).

## Highest-Leverage Next 5 (Ranked)

1. Improve external transfer while preserving internal consistency (multi-objective training/selection track).
2. Run routed-family hyper-sweep to fix under-utilized sparsity (`train_l0` too low) before further comparison.
3. Expand external-aware assignment objective with Pareto checkpointing over larger seed pool.
4. Close known-circuit gap with targeted architecture/feature-space changes plus confidence bounds.
5. Enable uniform W&B logging + run dashboard for all queue scripts.

## Read Next

1. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
2. `START_HERE.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
