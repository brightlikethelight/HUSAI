# Cycle-7 Live Monitoring Snapshot

- Snapshot UTC: 2026-02-16T22:50:00Z
- Source run: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
- Cycle-8 queue: waiting behind cycle-7 (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`)
- Cycle-9 queue: waiting behind cycle-8/cycle-7 (`results/experiments/cycle9_novelty_push/run_20260216T184628Z`)
- Latest assignment-throughput patch on `main`: `95f567c` (cycle-8 and cycle-9 will pull latest before execution)

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
- `pass_all=False`

### a2 (completed)
- Run: `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T173317Z`
- Summary artifact present at `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T173317Z/summary.md`

### a3 (in progress)
- Run: `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T201509Z`
- Checkpoints complete: `46 / 56`
- External eval summaries: `0 / 32` SAEBench, `0 / 32` CE-Bench
- Current stage is training/checkpointing; external eval has not started yet.

## Process/GPU Health

- Active processes:
  - `bash scripts/experiments/run_cycle7_pareto_push.sh`
  - `python scripts/experiments/run_assignment_consistency_v3.py` (a3 condition)
  - `bash scripts/experiments/run_cycle8_robust_pareto_push.sh` (waiting)
  - `bash scripts/experiments/run_cycle9_novelty_push.sh` (waiting)
- GPU snapshot (`nvidia-smi`): `NVIDIA B200, 0% util, ~6060 MiB / 183359 MiB` (captured during a low-util point in training loop)

## Weights & Biases

- Remote shell `WANDB_*`/`WB_*` environment variables: `0`
- Remote `wandb/run-*` directories in project root: `0`
- Telemetry remains artifact/log-file based.
