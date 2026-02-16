# Cycle-5 External Push Synthesis

## Routed Sweep

| run | mode | d_sae | k | experts | train_ev | train_l0 | saebench_delta | cebench_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_20260215T232359Z` | `expert_topk` | 1024 | 32 | 8 | 0.247353 | 32.000 | -0.067025 | -39.020924 |
| `run_20260215T233353Z` | `expert_topk` | 1024 | 32 | 4 | 0.366918 | 32.000 | -0.067687 | -37.853417 |
| `run_20260215T234257Z` | `expert_topk` | 2048 | 32 | 8 | 0.290360 | 32.000 | -0.073375 | -37.260996 |
| `run_20260215T235219Z` | `expert_topk` | 2048 | 48 | 8 | 0.314012 | 48.000 | -0.068142 | -38.709273 |
| `run_20260216T000156Z` | `global_mask` | 1024 | 32 | 8 | 0.226188 | 4.317 | -0.069580 | -39.787987 |

Best routed CE delta: `-37.260996` at `run_20260215T234257Z`

## Assignment Sweep

| run | d_sae | k | epochs | lr | best_lambda | internal_lcb | ev_drop | saebench_delta | cebench_delta | pass_all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260216T001059Z` | 1024 | 32 | 16 | 0.001 | 0.05 | 0.846420 | 0.236335 | -0.055411 | -37.594082 | `False` |
| `run_20260216T005618Z` | 2048 | 32 | 16 | 0.0007 | 0.05 | 0.838793 | 0.257788 | -0.049864 | -34.345572 | `False` |

Best assignment CE delta: `-34.345572` at `run_20260216T005618Z`

## Selection + Gate

- Selector (min_seeds=3) picked: `topk` / checkpoint `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/checkpoints/topk_seed123/sae_final.pt`
- Selector (min_seeds=2) picked: `assignv3_lambda0.05` / checkpoint `results/experiments/phase4d_assignment_consistency_v3_external_sweep/run_20260216T005618Z/checkpoints/lambda_0.05/sae_seed456.pt`
- Release pass_all: `False`
- External SAEBench LCB metric used: `-0.04478959689939781`
- External CE-Bench LCB metric used: `-40.467037470119465`

## Key Finding

- Routed `expert_topk` restored effective sparsity (`l0=32`/`48`) and improved CE-Bench relative to prior routed baseline, but all external deltas remain negative.
- Assignment sweep at `d_sae=2048` improved CE-Bench delta materially (`-34.35`), but SAEBench delta remains negative and strict external gate still fails.
