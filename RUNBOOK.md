# Runbook (Current)

Updated: 2026-02-15

Canonical map: `START_HERE.md` and `REPO_NAVIGATION.md`.

## 1) Environment

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
pre-commit install
```

Common local env flags used in this repo:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export TMPDIR=/tmp
```

## 2) Quality Checks

```bash
pytest tests -q
make smoke
python scripts/analysis/verify_experiment_consistency.py
```

## 3) Core Experiment Commands

### Internal baseline and ablations

```bash
python scripts/experiments/run_phase4a_reproduction.py
python scripts/experiments/run_core_ablations.py
python scripts/experiments/run_adaptive_l0_calibration.py
python scripts/experiments/run_assignment_consistency_v2.py --device cpu
```

### External benchmark programs

```bash
python scripts/experiments/run_official_external_benchmarks.py --execute

python scripts/experiments/run_husai_saebench_custom_eval.py \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda

python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo <path/to/CE-Bench> \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda
```

### Architecture frontier and scaling

```bash
python scripts/experiments/run_architecture_frontier_external.py \
  --architectures topk,relu,batchtopk,jumprelu \
  --seeds 42,123,456 \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench>

python scripts/experiments/run_external_metric_scaling_study.py \
  --token-budgets 10000,30000 \
  --hook-layers 0,1 \
  --d-sae-values 1024,2048 \
  --seeds 42,123,456 \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench>
```

### Stress and strict release gate

```bash
python scripts/experiments/run_transcoder_stress_eval.py --output-dir <out_dir>
python scripts/experiments/run_ood_stress_eval.py --output-dir <out_dir>

python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results <transcoder_summary.json> \
  --ood-results <ood_summary.json> \
  --external-summary <external_summary.json> \
  --require-transcoder --require-ood --require-external \
  --fail-on-gate-fail
```

## 4) B200 Queue Execution

End-to-end queue script:

```bash
bash scripts/experiments/run_b200_high_impact_queue.sh
```

Cycle-3 queue evidence mirror:
- `docs/evidence/cycle3_queue_final/`

## 5) Artifact Expectations

For every major run:
- `summary.md` or `summary.json`
- run config + command manifest
- logs with return codes
- explicit link in `EXPERIMENT_LOG.md`

## 6) Claim Policy

Do not promote external claims unless strict gate passes (`pass_all=true`).

Current known status (cycle-3): `pass_all=false`.
