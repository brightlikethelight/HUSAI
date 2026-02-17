# Cycle8/Cycle9 Live Monitoring Summary

Timestamp (UTC): 2026-02-17T13:36:19Z

## Queue Status

- `cycle8` orchestrator run: `results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`
  - routed stage complete (`b0`, `r1`, `r2`, `r3`, `r4`)
  - assignment stage:
    - `a1` complete (`run_20260217T061919Z`)
    - `a2` complete (`run_20260217T084709Z`)
    - `a3` active (`run_20260217T111709Z`, checkpoints `33`, external eval not started yet)
- `cycle9` orchestrator run: `results/experiments/cycle9_novelty_push/run_20260217T052929Z`
  - waiting behind cycle8
  - confirmed config: `SUPERVISED_PROXY_MODE=file_id`, `SUPERVISED_PROXY_WEIGHT=0.10`, `SUPERVISED_PROXY_NUM_CLASSES=0`

## Routed Stage Comparison (Cycle8)

From routed summary artifacts in `routed/*_summary.md`:

| run | setting | train_ev | SAEBench delta | CE-Bench delta |
|---|---|---:|---:|---:|
| `run_20260217T051230Z` | `b0` (noise 0.0, cons 0.0, div 0.0, d_sae 1536, k 40) | 0.3628 | -0.0662 | -36.4181 |
| `run_20260217T052555Z` | `r1` (noise 0.02, cons 0.10, div 0.0, d_sae 1536, k 40) | 0.3631 | -0.0699 | -36.4446 |
| `run_20260217T053854Z` | `r2` (noise 0.02, cons 0.10, div 0.02, d_sae 1536, k 40) | 0.3623 | -0.0660 | -36.5889 |
| `run_20260217T055236Z` | `r3` (noise 0.015, cons 0.08, div 0.02, d_sae 2048, k 40) | 0.3858 | -0.0717 | -37.2600 |
| `run_20260217T060602Z` | `r4` (noise 0.03, cons 0.12, div 0.015, d_sae 1024, k 48) | 0.3498 | **-0.0632** | **-36.1828** |

Observation:
- Best external deltas among routed conditions came from `r4`, but both are still below strict external gate thresholds.
- Higher capacity (`r3`) improved `train_ev` but hurt both external deltas.

## Assignment Stage Results So Far (Cycle8)

From assignment summaries in `assignment/*_summary.md`:

- `a1` (`run_20260217T061919Z`):
  - best lambda `0.10`
  - `internal_lcb=0.83957`
  - `ev_drop=0.27625` (fails EV-drop gate)
  - `saebench_delta=-0.04060` (fails SAEBench gate)
  - `cebench_delta=-34.86151` (passes CE floor in this configuration)
  - `pass_all=False`

- `a2` (`run_20260217T084709Z`):
  - best lambda `0.05`
  - `internal_lcb=0.83895`
  - `ev_drop=0.24146` (fails EV-drop gate)
  - `saebench_delta=-0.03976` (fails SAEBench gate)
  - `cebench_delta=-35.48915` (passes CE floor in this configuration)
  - `pass_all=False`

- `a3` (`run_20260217T111709Z`):
  - training active (`checkpoints=33`)
  - external eval stage has not started yet.

## W&B/Telemetry

- Remote queue currently has no active `WANDB_*` env telemetry.
- Canonical observability remains manifest/log JSON/markdown artifacts in `results/experiments/*` and synced `docs/evidence/*`.

## Decision-Relevant Interim Conclusion

- Cycle8 is generating measurable external movement but has not reached strict gate viability.
- The strongest immediate hope is whether `a3` can reduce EV drop while preserving assignment external gains.
- If `a3` does not improve SAEBench enough, cycle9 supervised-proxy stage becomes the next critical test.
