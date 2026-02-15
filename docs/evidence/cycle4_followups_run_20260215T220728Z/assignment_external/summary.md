# Assignment-Aware Consistency v3

- Run ID: `run_20260215T220737Z`
- Best lambda: `0.3`
- Best checkpoint: `results/experiments/phase4d_assignment_consistency_v3_external/run_20260215T220737Z/checkpoints/lambda_0.3/sae_seed456.pt`
- pass_all: `False`

## Acceptance

- gate_internal_lcb: `True`
- gate_ev_drop: `False`
- gate_saebench: `False`
- gate_cebench: `False`

| lambda | internal_lcb | ev_drop | saebench_delta | cebench_delta | joint_score | pareto |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3000 | 0.823699951171875 | 0.2714995543162028 | -0.048109553196925114 | -37.8070786523819 | 0.8143083925196677 | True |
| 0.0500 | 0.821007251739502 | 0.18272970120112103 | -0.05136131785281817 | -37.52352042675018 | 0.8116133386558022 | True |
| 0.2000 | 0.8233182430267334 | 0.24048542976379395 | -0.04951346816107971 | -37.66102049827576 | 0.8109786531193441 | True |
| 0.1000 | 0.8224997520446777 | 0.20967739820480347 | -0.05439812084198403 | -37.21468829870224 | 0.7533636502787815 | True |
| 0.0000 | 0.04151754081249237 | 0.0 | -0.05777670234074217 | -40.53418212652206 | 0.15 | True |
