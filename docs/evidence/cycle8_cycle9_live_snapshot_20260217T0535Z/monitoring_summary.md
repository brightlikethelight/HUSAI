# Cycle8/Cycle9 Live Snapshot

Timestamp (UTC): 2026-02-17T05:35:00Z

## Queue State

- Cycle8 runner: `results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`
  - stage1 condition `b0` completed.
  - stage1 condition `r1` active.
- Cycle9 runner: `results/experiments/cycle9_novelty_push/run_20260217T052929Z`
  - waiting behind cycle8.
  - supervised-proxy config confirmed in log header:
    - `SUPERVISED_PROXY_MODE=file_id`
    - `SUPERVISED_PROXY_WEIGHT=0.10`
    - `SUPERVISED_PROXY_NUM_CLASSES=0`

## Stage1 b0 Early Metrics

From `cycle8_routed_b0_summary.md`:
- `train_ev_mean = 0.3628`
- `train_l0_mean = 40.0`
- `saebench_best_minus_llm_auc_mean = -0.0662`
- `cebench_interp_delta_vs_baseline_mean = -36.4181`

## Reliability Notes

- Queue scripts now use anchored conflict checks (`d1ac12d`) to prevent stale-wrapper false positives.
- W&B telemetry was checked on remote runner and remains inactive for this queue (artifact/log-file tracking is canonical).
