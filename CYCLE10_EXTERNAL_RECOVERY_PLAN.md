# Cycle-10 External Recovery Plan

Updated: 2026-02-17

## Goal

Recover external benchmark deltas (SAEBench + CE-Bench) without losing internal consistency, and enforce strict uncertainty-aware release gating.

## Why Cycle-10 Exists

Cycle-8 improved internal consistency but external metrics remained negative under strict gates. Cycle-10 is the first queue explicitly optimized for external recovery while preserving grouped-LCB selection discipline.

## Scientific Hypotheses

1. H10.1: Routed robustness settings around the current best routed condition (`r4`) can improve external deltas under matched compute.
2. H10.2: Assignment-v3 with supervised proxy can improve SAEBench deltas while keeping CE-Bench degradation bounded.
3. H10.3: Grouped-LCB selector over combined frontier+assignment outputs reduces overfitting to point-estimate winners.
4. H10.4: Stress-gated release policy should remain the final claim filter even if selector metrics improve.

## Execution Script

Primary entrypoint:
- `scripts/experiments/run_cycle10_external_recovery.sh`

Queue behavior:
- waits for conflicting runners (`cycle4`..`cycle9` + direct experiment runners)
- performs `git pull origin main` before execution
- writes run manifest under:
  - `results/experiments/cycle10_external_recovery/run_<timestamp>/manifest.json`

## Stage Plan

### Stage 1: Routed external recovery sweep

Output root:
- `results/experiments/phase4b_routed_frontier_external_sweep_cycle10_recovery`

Conditions (from script):
1. `c1`: `d_sae=1024`, `k=48`, `experts=4`, `noise=0.03`, `consistency=0.12`, `diversity=0.015`, `lr=8e-4`, `epochs=12`
2. `c2`: `d_sae=1024`, `k=48`, `experts=4`, `noise=0.02`, `consistency=0.15`, `diversity=0.020`, `lr=8e-4`, `epochs=12`
3. `c3`: `d_sae=1280`, `k=48`, `experts=4`, `noise=0.025`, `consistency=0.12`, `diversity=0.020`, `lr=7.5e-4`, `epochs=12`
4. `c4`: `d_sae=1024`, `k=56`, `experts=4`, `noise=0.03`, `consistency=0.12`, `diversity=0.015`, `lr=7.5e-4`, `epochs=12`

Shared setup:
- seeds: `42,123,456,789`
- activation cache: `pythia-70m-deduped` layer-0 residual
- external eval: SAEBench 16-dataset slice + CE-Bench matched-baseline mode

### Stage 2: Assignment-v3 supervised-proxy recovery sweep

Output root:
- `results/experiments/phase4d_assignment_consistency_v3_cycle10_recovery`

Conditions (from script):
1. `s1`: `d_sae=2048`, `k=48`, `epochs=24`, `lr=5e-4`, `supervised_proxy_weight=0.05`
2. `s2`: `d_sae=2048`, `k=48`, `epochs=24`, `lr=5e-4`, `supervised_proxy_weight=0.10`
3. `s3`: `d_sae=3072`, `k=48`, `epochs=24`, `lr=4e-4`, `supervised_proxy_weight=0.10`

Shared setup:
- seeds: `123,456,789,1011`
- lambdas: `0.0,0.02,0.04,0.06,0.08,0.1,0.15`
- checkpoint policy: external-aware (`external_score`), per-lambda candidate narrowing
- SAEBench/CE-Bench both required for checkpoint acceptance

### Stage 3: Grouped-LCB selector

Selector root:
- `results/experiments/release_candidate_selection_cycle10_recovery`

Selection policy:
- require both external metrics
- group by condition
- uncertainty mode: `lcb`
- minimum seeds per group: `4`
- weights: SAEBench `0.80`, CE-Bench `0.15`, train-EV `0.05`

### Stage 4: OOD + strict release gate

- Run OOD stress on selected checkpoint.
- Run strict release policy requiring:
  - random-model gate
  - transcoder gate
  - OOD gate
  - external LCB gates

## Acceptance Criteria

1. Selector candidate must have non-degenerate grouped-LCB estimates (no missing group stats).
2. Strict release gate must return `pass_all=true` to claim external readiness.
3. If `pass_all=false`, write explicit failure reason and transition to assignment-v4 objective branch.

## Runtime Estimate (B200, single GPU)

Approximate wall-clock (order-of-magnitude):
1. Stage 1 routed sweep: `10-16h` (depends on external eval throughput).
2. Stage 2 assignment sweep: `16-30h` (training + external eval dominates).
3. Stage 3 selector + Stage 4 gating: `<2h`.

Total expected: `~1-2 days` continuous queue time.

## Monitoring Checklist

1. Process health:
- ensure one active worker per stage; no stale wrapper deadlock.

2. External eval progress:
- monitor `husai_custom_sae_summary.json` and `husai_custom_cebench_summary.json` counts.

3. Error scan:
- check logs for `Traceback`, `CUDA OOM`, `nan`, `inf`, `killed`.

4. Evidence sync:
- copy manifests, summaries, and launch logs into `docs/evidence/` snapshots during live runs.

## Fallback Branch (if Cycle-10 fails)

1. Add relation-constrained assignment objective branch (assignment-v4).
2. Keep grouped-LCB selector and strict gate unchanged.
3. Run known-circuit closure refresh before any narrative upgrade.
