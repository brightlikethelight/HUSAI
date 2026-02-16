# Setup Guide (Execution)

This guide provides the minimum setup and launch path for reproducible HUSAI experiments.

## Local Environment

1. Create environment:
   - `python -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `python -m pip install -r requirements.txt -r requirements-dev.txt`
3. Optional quality hooks:
   - `pre-commit install`

## Determinism Defaults

Use deterministic settings when running major sweeps:

- `export CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `export KMP_DUPLICATE_LIB_OK=TRUE`
- `export MPLCONFIGDIR=/tmp/mpl`

## Core Validation

- `pytest -q tests/unit/test_assignment_consistency_v3.py tests/unit/test_release_policy_selector.py tests/unit/test_routed_frontier_modes.py`
- `python -m py_compile scripts/experiments/run_assignment_consistency_v3.py`

## Remote (RunPod B200)

SSH command:

- `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@198.13.252.68 -p 34426 -i ~/.ssh/runpod_key`

Launch queue (example):

- `bash scripts/experiments/run_cycle6_saeaware_push.sh`

Monitor queue:

- `tail -f results/experiments/cycle6_saeaware_push/run_*/cycle6.log`
- `nvidia-smi`

## Canonical Outputs

- Queue manifest: `results/experiments/cycle6_saeaware_push/run_*/manifest.json`
- Selector result: `results/experiments/release_candidate_selection_cycle6/run_*/selected_candidate.json`
- Release gate: `results/experiments/release_stress_gates/run_*/release_policy.json`
