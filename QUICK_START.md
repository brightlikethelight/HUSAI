# Quick Start

Updated: 2026-02-12

For the full operational path and troubleshooting, use `RUNBOOK.md`.
For repository orientation, use `REPO_NAVIGATION.md`.

## 1) Setup

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
```

If needed on this machine:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/mpl
```

## 2) Fail-Fast Smoke

```bash
make smoke
```

## 3) Core Reproduction Commands

```bash
# Phase 4a: trained vs random
make reproduce-phase4a

# Phase 4c: core ablations
make ablate-core

# Phase 4e: benchmark-aligned slice
make benchmark-slice

# Official benchmark harness preflight/execute
make benchmark-official
```

## 4) Highest-Impact Follow-Ups

```bash
make adaptive-l0
make adaptive-l0-control
make consistency-sweep
make audit-results
```

## 5) Key Outputs

- `results/experiments/phase4a_trained_vs_random/`
- `results/experiments/phase4c_core_ablations/`
- `results/experiments/adaptive_l0_calibration/`
- `results/experiments/consistency_objective_sweep/`
- `results/analysis/experiment_consistency_report.md`

## 6) Quality Gate

```bash
pytest tests -q
```

## 7) Current Writeups

- `FINAL_BLOG.md`
- `FINAL_PAPER.md`
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
