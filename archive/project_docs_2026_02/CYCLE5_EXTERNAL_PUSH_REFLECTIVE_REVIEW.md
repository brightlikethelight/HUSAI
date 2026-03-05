# Cycle 5 External Push Reflective Review

Date: 2026-02-16

## Scope

Cycle-5 targeted the highest-impact external-gap bottleneck with:

1. Routed-family hyper-sweep with a new `expert_topk` routing mode.
2. Assignment-v3 external-aware sweep at higher capacity (`d_sae=2048`).
3. Grouped LCB reselection with assignment integration.
4. OOD + strict release gate rerun.

Primary evidence bundle:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`

## Main Results

### Routed sweep

Best routed CE-Bench delta improved to:
- `-37.260996` at `run_20260215T234257Z` (`expert_topk`, `d_sae=2048`, `k=32`, `experts=8`)

Routed key correction worked:
- `expert_topk` restored effective sparsity (`train_l0=32/48`) versus old `global_mask` collapse (`train_l0≈4.3`).

But external gate target still missed:
- SAEBench deltas stayed negative for all routed conditions.

### Assignment sweep

Best assignment condition:
- `run_20260216T005618Z` (`d_sae=2048`, `k=32`, `best_lambda=0.05`)
- `internal_lcb = 0.838793`
- `cebench_delta = -34.345572` (material CE-Bench improvement vs prior topk baseline)
- `saebench_delta = -0.049864` (still negative)

So assignment improved CE-Bench substantially but still failed joint external positivity.

### Selection + gate

Default grouped selector (`min_seeds_per_group=3`) selected:
- `topk_seed123` (unchanged from prior cycle)

Reason:
- Assignment groups had only 2 eligible grouped samples and were filtered out.

Corrected selector check (`min_seeds_per_group=2`) selected:
- `assignv3_lambda0.05` group

But strict external gate still fails because both external deltas remain < 0.

## Reliability Findings

1. Queue completed end-to-end and produced all stage outputs.
2. One orchestration bug was discovered in this run's queue manifest serialization (null metadata fields).
3. Bug has been fixed in `main` (`run_cycle5_external_push.sh` now passes manifest fields as script args, not non-exported env vars).

## Scientific Bottom Line After Cycle 5

- Internal consistency: still strong.
- CE-Bench competitiveness: improved in routed/assignment tracks.
- SAEBench competitiveness: still negative.
- Strict release gate: still `pass_all=False`.

The core open problem is now sharper:
- improve SAEBench without giving back CE-Bench/internal consistency.

## Highest-Impact Next Steps

1. Run assignment-v3 + routed joint Pareto selection with `min_seeds_per_group=2` and larger seed support to avoid exclusion artifacts.
2. Add SAEBench-targeted regularization/selection term in assignment objective (currently CE improves more than SAE AUC).
3. Expand routed `expert_topk` sweep on `k` and expert count with SAEBench-focused acceptance criteria.
4. Add automatic selector warning when groups are dropped due seed-count thresholds.
5. Add strict deterministic env export (`CUBLAS_WORKSPACE_CONFIG`) in every queue stage to remove cuBLAS nondeterminism warnings.
