# Assignment-Aware Consistency v3

- Run ID: `run_20260215T190154Z`
- Best lambda: `0.3`
- Best checkpoint: `results/experiments/phase4d_assignment_consistency_v3/run_20260215T190154Z/checkpoints/lambda_0.3/sae_seed456.pt`
- pass_all: `False`

## Acceptance

- gate_internal_lcb: `True`
- gate_ev_drop: `True`
- gate_saebench: `False`
- gate_cebench: `False`

| lambda | internal_lcb | ev_drop | saebench_delta | cebench_delta | joint_score | pareto |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3000 | 0.04702860862016678 | 0.0023497790098190308 | None | None | 0.45 | True |
| 0.2000 | 0.042979625364144625 | 0.0009657591581344604 | None | None | 0.39604889129384524 | True |
| 0.1000 | 0.03814409921566647 | 0.00019761919975280762 | None | None | 0.2756250217149941 | True |
| 0.0500 | 0.03640575458606082 | -2.0369887351989746e-05 | None | None | 0.22865297920256267 | True |
| 0.0000 | 0.03415578603744507 | 0.0 | None | None | 0.14871084761533526 | False |
