# Cycle-7 Live Monitoring Snapshot

- Snapshot UTC: 2026-02-16T17:05:36+00:00
- Source run: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
- Cycle-8 queue: waiting behind cycle-7 (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`)

## Routed Stage (Completed p1..p5)

| run | d_sae | k | experts | lr | epochs | train_ev | saebench_delta | cebench_delta | cebench_abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run_20260216T130431Z | 1024 | 32 | 4 | 0.001 | 10 | 0.342547 | -0.069497 | -36.675231 | 11.276380 |
| run_20260216T132055Z | 2048 | 48 | 8 | 0.0007 | 12 | 0.354931 | -0.071644 | -37.356861 | 10.594751 |
| run_20260216T133659Z | 1536 | 40 | 6 | 0.0008 | 12 | 0.362673 | -0.063807 | -36.232309 | 11.719302 |
| run_20260216T135316Z | 2048 | 40 | 6 | 0.0007 | 12 | 0.366857 | -0.065077 | -37.028728 | 10.922884 |
| run_20260216T140951Z | 1024 | 48 | 4 | 0.0008 | 12 | 0.343716 | -0.065715 | -36.183461 | 11.768150 |

- Best SAEBench delta: `-0.063807` at `run_20260216T133659Z`
- Best CE-Bench delta: `-36.183461` at `run_20260216T140951Z`

## Assignment Stage Progress (run_20260216T142558Z)

- checkpoints: `56`
- completed SAEBench evals: `23`
- completed CE-Bench evals: `22`

| lambda | checkpoints | saebench | cebench |
|---|---:|---:|---:|
| 0.0 | 7 | 4 | 4 |
| 0.03 | 7 | 4 | 4 |
| 0.05 | 7 | 4 | 4 |
| 0.08 | 7 | 4 | 4 |
| 0.1 | 7 | 4 | 4 |
| 0.15 | 7 | 3 | 2 |
| 0.2 | 7 | 0 | 0 |
| 0.3 | 7 | 0 | 0 |

## Weights & Biases

- No `WANDB_*` env vars detected in remote run shell.
- Current telemetry is artifact/log-file based.
