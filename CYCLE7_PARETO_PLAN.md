# Cycle-7 Pareto Push Plan

Date: 2026-02-16
Status: queued (waiting behind cycle-6)
Queue script: `scripts/experiments/run_cycle7_pareto_push.sh`

## Why This Cycle Exists

Cycle-5 and early cycle-6 evidence showed a stable trade-off:
- Some routed settings improved CE-Bench deltas but hurt SAEBench.
- Other settings improved SAEBench but regressed CE-Bench.
- External gate remained failing under strict release criteria.

Cycle-7 targets this trade-off directly with a Pareto-zone sweep plus SAEBench-prioritized assignment-v3 selection.

## Evidence Basis for Condition Selection

Empirical anchors (local evidence snapshots):
- Best CE-oriented routed condition in cycle-5 came from lower-width/lower-expert setting:
  - `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/run_20260215T233353Z_results.json`
- Best SAEBench-oriented routed condition in cycle-5 came from higher-k routed setting:
  - `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/run_20260215T235219Z_results.json`
- Assignment-v3 had strongest CE around `d_sae=2048`, but SAEBench stayed negative:
  - `docs/evidence/cycle5_external_push_run_20260215T232351Z/assignment/run_20260216T005618Z_results.json`

Literature anchors:
- SAEBench benchmark discipline: https://arxiv.org/abs/2503.09532
- CE-Bench benchmark design: https://arxiv.org/abs/2509.00691
- Route SAE design motivation: https://arxiv.org/abs/2503.08200
- Transcoder control signal: https://arxiv.org/abs/2501.18823

## Cycle-7 Design

### Stage 1: Routed Pareto-zone sweep
- Run family: `phase4b_routed_frontier_external_sweep_cycle7_pareto`
- Seeds: `42,123,456,789,1011`
- Conditions:
  - `p1`: `d_sae=1024`, `k=32`, `experts=4`, `lr=1e-3`, `epochs=10`
  - `p2`: `d_sae=2048`, `k=48`, `experts=8`, `lr=7e-4`, `epochs=12`
  - `p3`: `d_sae=1536`, `k=40`, `experts=6`, `lr=8e-4`, `epochs=12`
  - `p4`: `d_sae=2048`, `k=40`, `experts=6`, `lr=7e-4`, `epochs=12`
  - `p5`: `d_sae=1024`, `k=48`, `experts=4`, `lr=8e-4`, `epochs=12`

### Stage 2: Assignment-v3 SAEBench-prioritized sweep
- Run family: `phase4d_assignment_consistency_v3_cycle7_pareto`
- Train seeds: `123,456,789,1011,1213,1415`
- Lambda grid: `0.0,0.03,0.05,0.08,0.1,0.15,0.2,0.3`
- Condition set:
  - `a1`: `d_sae=2048`, `k=32`, `epochs=20`, `lr=7e-4`
  - `a2`: `d_sae=2048`, `k=48`, `epochs=20`, `lr=6e-4`
  - `a3`: `d_sae=3072`, `k=48`, `epochs=24`, `lr=5e-4`
- External checkpoint policy:
  - `external_score`
  - candidate weights: `saebench=0.70`, `cebench=0.20`, `alignment=0.05`, `ev=0.05`

### Stage 3: Grouped LCB reselection
- Selector output roots:
  - `release_candidate_selection_cycle7`
  - `release_candidate_selection_cycle7_min3`
- Main threshold: `--min-seeds-per-group 4`
- Sensitivity threshold: `--min-seeds-per-group 3`
- Weighting: `saebench=0.70`, `cebench=0.25`, `train_ev=0.05`

### Stage 4: OOD + strict gate
- OOD eval on selected checkpoint
- Strict release gate with external LCB requirements retained (`>= 0.0`)

## Decision Criteria

Cycle-7 is considered scientifically successful only if both are true:
1. External deltas improve jointly (SAEBench and CE-Bench) under grouped uncertainty-aware selection.
2. Strict gate outcome improves versus cycle-5/cycle-6 baseline, without breaking stress controls.

## Outputs to Watch

- `results/experiments/cycle7_pareto_push/run_*/manifest.json`
- `results/experiments/phase4b_routed_frontier_external_sweep_cycle7_pareto/run_*/results.json`
- `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_*/results.json`
- `results/experiments/release_candidate_selection_cycle7/run_*/selection_summary.json`
- `results/experiments/release_stress_gates/run_*/release_policy.json`
