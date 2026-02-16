# Executive Summary (Live Cycle-7/Cycle-8 State)

Date: 2026-02-16

## Repo Purpose

HUSAI tests whether SAE feature consistency gains are real across seeds, and whether those gains transfer to external benchmark validity (SAEBench, CE-Bench) under strict release gates.

## Current Scientific Bottom Line

- Internal consistency progress: **real and replicated**.
- External competitiveness: **still unresolved** (latest fully completed strict gate remains fail).
- Reliability and reproducibility hygiene: **strong** (unit tests passing, queue manifests, strict gates).
- Live execution state: **cycle-7 running (assignment stage), cycle-8 queued with robustness upgrades**.

Canonical current-status artifacts:
- `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`
- `CYCLE7_PARETO_PLAN.md`
- `CYCLE8_ROBUST_PLAN.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`

## Cycle 5 Key Outcomes

1. Routed family sweep (`expert_topk` mode) fixed effective sparsity collapse.
- Prior routed baseline had low effective `l0` due global-topk masking.
- New mode restored `l0=32/48` in routed runs.

2. Best routed CE-Bench delta improved to:
- `-37.260996` (`run_20260215T234257Z`)

3. Assignment-v3 high-capacity sweep (`d_sae=2048`) improved CE-Bench delta further:
- `cebench_delta = -34.345572`
- but `saebench_delta = -0.049864` remained negative.

4. Default grouped selector (`min_seeds_per_group=3`) still selected baseline topk.
- Assignment groups were undercounted at that threshold.

5. Corrected selector check (`min_seeds_per_group=2`) selected assignment candidate.
- Still fails strict external-positive gate because both external deltas remain < 0.

## Latest Strict Gate Metrics (Cycle 5 canonical run)

From `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`:

- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Numerical highlights:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## Top Issues (Current)

1. `P0` External benchmark gate still fails.
2. `P0` No candidate is jointly external-positive on SAEBench and CE-Bench under strict thresholds.
3. `P1` Selector thresholding can exclude promising groups when seed-count requirements are mismatched.
4. `P1` Determinism env warnings still appear in assignment stages (cuBLAS config consistency).

## Highest-Leverage Next 5 (Ranked)

1. Finish cycle-7 and evaluate whether assignment stage can produce any external non-negative-LCB candidate.
2. Execute cycle-8 robust routed sweep and compare robustness/diversity regularization against cycle-7 routed Pareto points.
3. Re-run grouped-LCB selector with cycle-8 outputs and quantify sensitivity at `min_seeds_per_group=4` vs `3`.
4. If external gate still fails, run a focused assignment-only cycle with SAEBench-heavier checkpoint policy and stricter CE floor.
5. Close known-circuit track with trained-vs-random CI reporting in the same release packet.

## Read Next

1. `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`
2. `CYCLE7_PARETO_PLAN.md`
3. `CYCLE8_ROBUST_PLAN.md`
4. `START_HERE.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`
