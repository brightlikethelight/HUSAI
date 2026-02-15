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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
python scripts/experiments/run_assignment_consistency_v3.py --device cpu
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

python scripts/experiments/run_matryoshka_frontier_external.py \
  --seeds 42,123,456 \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench>

python scripts/experiments/run_external_metric_scaling_study.py \
  --token-budgets 10000,30000 \
  --hook-layers 0,1 \
  --d-sae-values 1024,2048 \
  --seeds 42,123,456 \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench> \
  --cebench-matched-baseline-map docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json
```

### Candidate Selection + Stress and Strict Release Gate

```bash
# default mode is now grouped + uncertainty-aware LCB over conditions
python scripts/experiments/select_release_candidate.py \
  --frontier-results <frontier_results.json> \
  --scaling-results <scaling_results.json> \
  --require-both-external

# optional: force old seed-level point-estimate selection
python scripts/experiments/select_release_candidate.py \
  --frontier-results <frontier_results.json> \
  --scaling-results <scaling_results.json> \
  --require-both-external \
  --seed-level-selection \
  --uncertainty-mode point

# stress runners
python scripts/experiments/run_transcoder_stress_eval.py --output-dir <out_dir>
python scripts/experiments/run_transcoder_stress_sweep.py \
  --min-delta-lcb 0.0 \
  --fail-on-gate-fail \
  --output-dir <out_dir>
python scripts/experiments/run_ood_stress_eval.py --output-dir <out_dir>

# strict joint external gate (LCB mode)
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results <transcoder_summary.json> \
  --ood-results <ood_summary.json> \
  --external-candidate-json <selected_candidate.json> \
  --external-mode joint \
  --use-external-lcb \
  --min-saebench-delta-lcb 0.0 \
  --min-cebench-delta-lcb 0.0 \
  --require-transcoder --require-ood --require-external \
  --fail-on-gate-fail
```

### Proposal Closure (Known-Circuit)

```bash
python scripts/experiments/run_known_circuit_recovery_closure.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --sae-checkpoint-glob 'results/experiments/phase4d_assignment_consistency_v3/run_*/checkpoints/lambda_*/sae_seed*.pt'
```

## 4) B200 Queue Execution

End-to-end queue script:

```bash
MIN_SAEBENCH_DELTA=0.0 MIN_CEBENCH_DELTA=0.0 \
MIN_SAEBENCH_DELTA_LCB=0.0 MIN_CEBENCH_DELTA_LCB=0.0 \
USE_EXTERNAL_LCB_GATES=1 \
SELECTOR_GROUP_BY_CONDITION=1 SELECTOR_UNCERTAINTY_MODE=lcb MIN_SEEDS_PER_GROUP=3 \
CEBENCH_BASELINE_MAP=docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json \
bash scripts/experiments/run_b200_high_impact_queue.sh
```

Queue behavior includes:
1. scaling run,
2. multi-objective candidate selection,
3. stress runs,
4. joint external gate evaluation.

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
