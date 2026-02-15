# Cycle 3 Final Synthesis (B200 Queue)

- Queue run: `run_20260214T210734Z`
- Status: `completed`
- Strict release policy exit code: `2`

## Execution Summary

- Frontier run: `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z`
- Scaling run: `results/experiments/phase4e_external_scaling_study_multiseed/run_20260214T212435Z`
- Transcoder stress: `results/experiments/phase4e_transcoder_stress_b200/run_20260214T224242Z/transcoder_stress_summary.json`
- OOD stress: `results/experiments/phase4e_ood_stress_b200/run_20260214T224309Z/ood_stress_summary.json`
- Release policy: `results/experiments/release_stress_gates/run_20260214T225029Z/release_policy.json`

## Frontier (Multiseed External)

- Rows used: `144042`
- Source activation files: `80`
- SAEBench datasets: `8`
- Architectures: `['topk', 'relu', 'batchtopk', 'jumprelu']`
- Seeds: `[42, 123, 456, 789, 1011]`

| architecture | train EV mean | SAEBench best-LLM AUC mean | SAEBench std | CE-Bench interp mean | CE-Bench interp std | CE-Bench delta vs matched baseline mean |
|---|---:|---:|---:|---:|---:|---:|
| topk | 0.745090 | -0.040593 | 0.004788 | 7.726768 | 0.276307 | -40.224843 |
| relu | 0.995546 | -0.024691 | 0.005091 | 4.257686 | 0.021609 | -43.693925 |
| batchtopk | 0.677543 | -0.043356 | 0.003982 | 6.537639 | 0.159780 | -41.413973 |
| jumprelu | 0.995933 | -0.030577 | 0.008254 | 4.379002 | 0.057746 | -43.572609 |

## Scaling (Multiseed External)

- Conditions: `24`
- Effective training rows per condition (min/max): `10000` / `29432`

### by_token_budget

| key | SAEBench mean | CE-Bench interp mean | CE-Bench delta mean | n |
|---|---:|---:|---:|---:|
| 10000 | -0.086291 | 7.862273 | -40.089339 | 12 |
| 30000 | -0.085132 | 8.026933 | -39.924679 | 12 |

### by_hook_layer

| key | SAEBench mean | CE-Bench interp mean | CE-Bench delta mean | n |
|---|---:|---:|---:|---:|
| 0 | -0.077427 | 6.749414 | -41.202197 | 12 |
| 1 | -0.093996 | 9.139791 | -38.811820 | 12 |

### by_d_sae

| key | SAEBench mean | CE-Bench interp mean | CE-Bench delta mean | n |
|---|---:|---:|---:|---:|
| 1024 | -0.082122 | 7.167310 | -40.784301 | 12 |
| 2048 | -0.089301 | 8.721896 | -39.229716 | 12 |

## Stress and Gate Outcomes

| metric | value |
|---|---:|
| transcoder_delta | -0.002227966984113039 |
| transcoder_pwmcc_mean | 0.26833198467890423 |
| sae_pwmcc_mean | 0.2705599516630173 |
| random_pwmcc_mean | 0.2479203666249911 |
| transcoder_samples | 12769 |
| id_best_minus_llm_auc | -0.028988677631220372 |
| ood_best_minus_llm_auc | -0.0434427392464225 |
| ood_drop | 0.01445406161520213 |
| ood_relative_drop | 0.49861058855735185 |
| release_pass_all | False |

## Gate Breakdown

| gate | pass |
|---|---:|
| random_model | True |
| transcoder | False |
| ood | True |
| external | False |
| pass_all | False |

## Monitoring and Logging Quality

- No active training/eval processes at final check; GPU idle (`0%`, `0 MB`).
- Frontier log coverage: expected `40` logs (`20` SAEBench + `20` CE-Bench).
- Scaling log coverage: expected `72` logs for `24` conditions.
- W&B status: no `WANDB_*` environment variables and no remote `wandb/` run artifacts for this queue cycle.

## Interpretation

1. External deltas remain negative versus matched baseline across all tested architectures and scaling settings.
2. SAEBench and CE-Bench still prefer different regions (tradeoff persists).
3. OOD gate now passes; transcoder and external gates fail, so strict release gate remains blocked.

