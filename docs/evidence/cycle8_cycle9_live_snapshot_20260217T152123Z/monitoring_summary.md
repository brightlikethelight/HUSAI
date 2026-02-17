# Monitoring Summary (Cycle8/9 Live Snapshot)

Snapshot timestamp: 2026-02-17T15:40:42Z

## Queue state
- `cycle8`: stage2 assignment run `a3` completed: `results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_20260217T111709Z`
- `cycle9`: active routed sweep.
  - completed routed run: `results/experiments/phase4b_routed_frontier_external_sweep_cycle9_novelty/run_20260217T151852Z` (`checkpoints=4`, `saebench=4`, `cebench=4`)
  - currently active routed run: `results/experiments/phase4b_routed_frontier_external_sweep_cycle9_novelty/run_20260217T153308Z`
- `cycle10`: prepared, not started yet (`scripts/experiments/run_cycle10_external_recovery.sh`)

## Cycle8 a3 final acceptance
- best lambda: `0.1`
- best checkpoint: `results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_20260217T111709Z/checkpoints/lambda_0.1/sae_seed1011.pt`
- gate_internal_lcb: `True`
- gate_ev_drop: `False`
- gate_saebench: `False`
- gate_cebench: `True`
- pass_all: `False`

Selected candidate metrics:
- internal_lcb: `0.8349106998182834`
- ev_drop: `0.2736066937446594`
- saebench_delta: `-0.04743732688749169`
- cebench_delta: `-33.67718325614929`

Lambda-level summary (cycle8 a3):
| lambda | internal_lcb | ev_drop | saebench_delta | cebench_delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.024064 | 0.000000 | -0.071178 | -37.510606 |
| 0.0300 | 0.833238 | 0.238974 | -0.049378 | -33.821840 |
| 0.0500 | 0.834255 | 0.254527 | -0.049830 | -33.519217 |
| 0.0800 | 0.834743 | 0.267090 | -0.053521 | -33.587462 |
| 0.1000 | 0.834911 | 0.273607 | -0.047437 | -33.677183 |
| 0.1500 | 0.835161 | 0.287488 | -0.049400 | -34.062932 |
| 0.2000 | 0.835287 | 0.297212 | -0.050714 | -31.660304 |

## Live process/GPU poll
```text
ts=2026-02-17T15:40:42Z
run=results/experiments/phase4b_routed_frontier_external_sweep_cycle9_novelty/run_20260217T151852Z
checkpoints=4
saebench=4
cebench=4
root      320343  265642 57 15:32 ?        00:04:26 python scripts/experiments/run_routed_frontier_external.py ... --output-dir results/experiments/phase4b_routed_frontier_external_sweep_cycle9_novelty
root      321380  320343 99 15:38 ?        00:03:12 /usr/local/bin/python scripts/experiments/run_husai_cebench_custom_eval.py ... run_20260217T153308Z ...
```

## W&B status
- No active `WANDB_*` environment variables observed in queue process checks.
- No queue-linked `wandb/run-*` directories observed during live polls.
- Canonical telemetry remains log + JSON artifact based.
