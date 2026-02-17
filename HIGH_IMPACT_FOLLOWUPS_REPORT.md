# High-Impact Follow-Ups Report (Cycle 7/8/9/10 Live)

Date: 2026-02-17

## Current Queue Status

- `cycle7` run (`results/experiments/cycle7_pareto_push/run_20260216T062213Z`): complete.
- `cycle8` run (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`): complete through `a3`.
  - final assignment run: `results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_20260217T111709Z`
  - external completion: `sae_done=28`, `ce_done=28`
  - final acceptance: `pass_all=false`
- `cycle9` run (`results/experiments/cycle9_novelty_push/run_20260217T052929Z`): active.
  - current stage: routed sweep active (`run_20260217T151852Z` completed, `run_20260217T153308Z` in progress)
  - supervised-proxy assignment config (for stage2):
    - `SUPERVISED_PROXY_MODE=file_id`
    - `SUPERVISED_PROXY_WEIGHT=0.10`
    - `SUPERVISED_PROXY_NUM_CLASSES=0`
- `cycle10` queue script is prepared and syntax-validated for immediate post-cycle9 launch:
  - `scripts/experiments/run_cycle10_external_recovery.sh`

## What Is Already Complete (artifact-backed)

1. Direct HUSAI-checkpoint CE-Bench adapter with matched baseline.
- Evidence: `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json`

2. Matched-budget architecture frontier on external benchmarks.
- Evidence: `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/`
- Additional cycle7 routed sweep complete in:
  - `results/experiments/phase4b_routed_frontier_external_sweep_cycle7_pareto/`

3. External-metric scaling study (`token budget`, `hook layer`, `d_sae`).
- Evidence: `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`

4. Assignment-aware objective with external-aware checkpoint policy.
- Evidence: `scripts/experiments/run_assignment_consistency_v3.py`

5. Stress-gated release policy.
- Evidence: `scripts/experiments/run_stress_gated_release_policy.py`
- Prior strict-gate outputs: `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`

## Current Scientific Bottom Line

- Internal consistency: strong and reproducible.
- External competitiveness: still below strict release gates.
- Main open problem: improve SAEBench and CE-Bench deltas jointly under grouped-LCB selection.

## Cycle8 a3 Final Evidence

From `run_20260217T111709Z`:
- best lambda: `0.10`
- internal LCB: `0.8349106998182834` (passes internal)
- `ev_drop`: `0.2736066937446594` (fails EV gate)
- `saebench_delta`: `-0.04743732688749169` (fails SAEBench gate)
- `cebench_delta`: `-33.67718325614929` (passes CE gate)
- final gate: `pass_all=false`

Primary live snapshot:
- `docs/evidence/cycle8_cycle9_live_snapshot_20260217T152123Z/monitoring_summary.md`

## Updated Highest-Leverage Next 5 (Execution-Ready)

1. Complete cycle9 routed + assignment + selector + strict gate, then compare directly against cycle8 a3 outcomes.
2. Launch cycle10 external-recovery queue (`scripts/experiments/run_cycle10_external_recovery.sh`) immediately after cycle9.
3. Use grouped-LCB selector + strict gate as immutable criterion for cycle10 candidate acceptance.
4. If cycle10 remains external-negative, run assignment-v4 (relation-constrained objective + supervised proxy) under the same gate policy.
5. Close known-circuit recovery with trained-vs-random confidence bounds as release prerequisite.

Detailed plans:
- `CYCLE9_NOVELTY_PLAN.md`
- `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`
