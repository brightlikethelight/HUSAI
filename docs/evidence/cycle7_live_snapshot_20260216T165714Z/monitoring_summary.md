# Cycle-7 Live Monitoring Snapshot

- Snapshot UTC: 2026-02-16T18:25:00Z
- Source run: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
- Cycle-8 queue: waiting behind cycle-7 (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`)
- Selection-robustness patch pushed to `main`: `14b6c59` (cycle-8 will pull this before starting)

## Routed Stage (Completed p1..p5)

| run | d_sae | k | experts | lr | epochs | train_ev | saebench_delta | cebench_delta | cebench_abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run_20260216T130431Z | 1024 | 32 | 4 | 0.001 | 10 | 0.342547 | -0.069497 | -36.675231 | 11.276380 |
| run_20260216T132055Z | 2048 | 48 | 8 | 0.0007 | 12 | 0.354931 | -0.071644 | -37.356861 | 10.594751 |
| run_20260216T133659Z | 1536 | 40 | 6 | 0.0008 | 12 | 0.362673 | -0.063807 | -36.232309 | 11.719302 |
| run_20260216T135316Z | 2048 | 40 | 6 | 0.0007 | 12 | 0.366857 | -0.065077 | -37.028728 | 10.922884 |
| run_20260216T140951Z | 1024 | 48 | 4 | 0.0008 | 12 | 0.343716 | -0.065715 | -36.183461 | 11.768150 |

- Best SAEBench delta (routed): `-0.063807` at `run_20260216T133659Z`
- Best CE-Bench delta (routed): `-36.183461` at `run_20260216T140951Z`

## Assignment Stage

### a1 (completed)
- Run: `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T142558Z`
- Best lambda: `0.15`
- Internal LCB: `0.8398407121499379`
- SAEBench delta: `-0.04354644998752344`
- CE-Bench delta: `-34.468481616973875`
- `pass_all=False` (external still below strict zero-LCB gates)

### a2 (in progress)
- Run: `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T173317Z`
- Checkpoints complete: `50 / 56`
- External eval summaries: `0 / 32` SAEBench, `0 / 32` CE-Bench
- Most recent artifact age during snapshot: ~36s (active forward progress)

## Process/GPU Health

- Active processes:
  - `bash scripts/experiments/run_cycle7_pareto_push.sh`
  - `python scripts/experiments/run_assignment_consistency_v3.py` (a2 condition)
  - `bash scripts/experiments/run_cycle8_robust_pareto_push.sh` (waiting)
- GPU snapshot: `NVIDIA B200, 5% util, 4686 MiB / 183359 MiB`

## Weights & Biases

- No `WANDB_*` environment variables detected on remote shell.
- No active `wandb/run-*` directories for current queue.
- Current telemetry remains artifact/log-file based.
