# Monitoring Summary (Cycle8/9 Live Snapshot)

Snapshot timestamp: 2026-02-17T13:59:08Z

## Queue state
- `cycle8`: active (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`)
- `cycle8` active stage: assignment `a3` external-eval sweep (`run_20260217T111709Z`)
- `cycle9`: waiting (`results/experiments/cycle9_novelty_push/run_20260217T052929Z`)

## a3 partial progress (live poll)
```text
ts=2026-02-17T14:32:43Z
sae_done=16
ce_done=15
total 351
-rw-rw-rw- 1 root root 11381 Feb 17 14:32 lambda_0.08_seed1011_saebench.log
-rw-rw-rw- 1 root root 11115 Feb 17 14:31 lambda_0.08_seed123_cebench.log
-rw-rw-rw- 1 root root 11427 Feb 17 14:29 lambda_0.08_seed123_saebench.log
-rw-rw-rw- 1 root root 11115 Feb 17 14:28 lambda_0.08_seed456_cebench.log
-rw-rw-rw- 1 root root 11427 Feb 17 14:26 lambda_0.08_seed456_saebench.log
-rw-rw-rw- 1 root root 11115 Feb 17 14:25 lambda_0.08_seed789_cebench.log
-rw-rw-rw- 1 root root 11363 Feb 17 14:23 lambda_0.08_seed789_saebench.log
-rw-rw-rw- 1 root root 11119 Feb 17 14:22 lambda_0.05_seed1011_cebench.log
-rw-rw-rw- 1 root root 11317 Feb 17 14:20 lambda_0.05_seed1011_saebench.log
-rw-rw-rw- 1 root root 11667 Feb 17 14:19 lambda_0.05_seed123_cebench.log
-rw-rw-rw- 1 root root 11363 Feb 17 14:16 lambda_0.05_seed123_saebench.log
root      302337  215681 99 11:17 ?        17:20:37 python scripts/experiments/run_assignment_consistency_v3.py --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped --activation-glob *_blocks.0.hook_resid_pre.pt --max-files 80 --max-rows-per-file 2048 --max-total-rows 150000 --d-sae 3072 --k 48 --device cuda --epochs 24 --batch-size 4096 --learning-rate 0.0004 --train-seeds 123,456,789,1011 --lambdas 0.0,0.03,0.05,0.08,0.1,0.15,0.2 --run-saebench --run-cebench --cebench-repo /workspace/CE-Bench --cebench-max-rows 200 --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json --saebench-datasets 100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom --saebench-results-path /tmp/husai_saebench_probe_results_cycle8_assignment --saebench-model-cache-path /tmp/sae_bench_model_cache --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle8_assignment --external-checkpoint-policy external_score --external-checkpoint-candidates-per-lambda 4 --external-candidate-require-both --external-candidate-min-saebench-delta -0.04 --external-candidate-min-cebench-delta -35.5 --external-candidate-weight-saebench 0.75 --external-candidate-weight-cebench 0.15 --external-candidate-weight-alignment 0.05 --external-candidate-weight-ev 0.05 --weight-internal-lcb 0.20 --weight-ev 0.05 --weight-saebench 0.55 --weight-cebench 0.20 --force-rerun-external --require-external --min-saebench-delta -0.02 --min-cebench-delta -35.5 --output-dir results/experiments/phase4d_assignment_consistency_v3_cycle8_robust
root      311225  302337 99 14:32 ?        00:00:17 /usr/local/bin/python scripts/experiments/run_husai_cebench_custom_eval.py --cebench-repo /workspace/CE-Bench --checkpoint /workspace/HUSAI/results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_20260217T111709Z/checkpoints/lambda_0.08/sae_seed1011.pt --architecture topk --sae-release husai_assignv3_run_20260217T111709Z_lambda0.08_seed1011 --model-name pythia-70m-deduped --hook-layer 0 --hook-name blocks.0.hook_resid_pre --device cuda --sae-dtype float32 --output-folder /workspace/HUSAI/results/experiments/phase4d_assignment_consistency_v3_cycle8_robust/run_20260217T111709Z/external_eval/0.08/seed1011/cebench --artifacts-path /tmp/ce_bench_artifacts_cycle8_assignment --max-rows 200 --matched-baseline-summary /workspace/HUSAI/docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json
```

## Completed lambda=0.0 seed metrics (partial, 3 seeds)
| Seed | SAEBench best_minus_llm_auc | SAEBench best_k | SAEBench best_auc | LLM auc | CE avg delta vs matched baseline | CE contrastive delta | CE independent delta | CE interpretability delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1011 | -0.074601 | 5 | 0.618215 | 0.692816 | -39.467707 | -40.634103 | -39.984679 | -37.784340 |
| 456 | -0.071178 | 5 | 0.621638 | 0.692816 | -39.465358 | -40.315215 | -40.570251 | -37.510606 |
| 789 | -0.074740 | 5 | 0.618076 | 0.692816 | -39.214361 | -40.149758 | -40.215947 | -37.277378 |

Interpretation: this partial batch is still strongly negative on external deltas; full a3 completion is required before selector/gate updates.

## W&B status
- No `WANDB_*` env vars observed in the active queue process environment during live checks.
- No `wandb/run-*` directories observed for the active cycle8/9 queue in `/workspace/HUSAI/wandb` at check time.
- Current telemetry source is filesystem logs + JSON summaries under `results/experiments/...`.

