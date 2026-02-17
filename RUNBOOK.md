# Runbook (Current)

Updated: 2026-02-17

Canonical map: `START_HERE.md`, `REPO_NAVIGATION.md`, `CYCLE7_PARETO_PLAN.md`, `CYCLE8_ROBUST_PLAN.md`, `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`.

## 1) Environment

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
pre-commit install
```

Recommended deterministic flags:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export TMPDIR=/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Optional W&B telemetry (disabled by default in current remote queues):

```bash
export WANDB_PROJECT=husai
export WANDB_ENTITY=<your_entity>
export WANDB_MODE=online
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

python scripts/experiments/run_routed_frontier_external.py \
  --seeds 42,123,456 \
  --route-topk-mode expert_topk \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench>

python scripts/experiments/run_external_metric_scaling_study.py \
  --token-budgets 10000,30000 \
  --hook-layers 0,1 \
  --d-sae-values 1024,2048 \
  --seeds 42,123,456 \
  --run-saebench --run-cebench --cebench-repo <path/to/CE-Bench> \
  --cebench-matched-baseline-map docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json
```

### Candidate selection + stress + strict release gate

```bash
python scripts/experiments/select_release_candidate.py \
  --frontier-results <frontier_results.json> \
  --scaling-results <scaling_results.json> \
  --assignment-results <assignment_results.json> \
  --group-by-condition --uncertainty-mode lcb --min-seeds-per-group 3 \
  --require-both-external

python scripts/experiments/run_transcoder_stress_sweep.py \
  --min-delta-lcb 0.0 \
  --fail-on-gate-fail \
  --output-dir <out_dir>

python scripts/experiments/run_ood_stress_eval.py --output-dir <out_dir>

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

### Proposal closure (known-circuit)

```bash
python scripts/experiments/run_known_circuit_recovery_closure.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --sae-checkpoint-glob 'results/experiments/phase4d_assignment_consistency_v3/run_*/checkpoints/lambda_*/sae_seed*.pt'
```

## 4) Queue Execution (B200)

Queue performance note (assignment stage):

```bash
export ASSIGN_UPDATE_INTERVAL=4
```

Use `ASSIGN_UPDATE_INTERVAL=1` if you need exact per-step Hungarian updates.

Cycle-9 supervised-proxy defaults (assignment-v4 path):

```bash
export SUPERVISED_PROXY_MODE=file_id
export SUPERVISED_PROXY_WEIGHT=0.10
export SUPERVISED_PROXY_NUM_CLASSES=0
```

Queue conflict detection in cycle8/cycle9 uses anchored process checks to avoid stale-wrapper false positives.

Cycle-3 queue:

```bash
MIN_SAEBENCH_DELTA=0.0 MIN_CEBENCH_DELTA=0.0 \
MIN_SAEBENCH_DELTA_LCB=0.0 MIN_CEBENCH_DELTA_LCB=0.0 \
USE_EXTERNAL_LCB_GATES=1 \
SELECTOR_GROUP_BY_CONDITION=1 SELECTOR_UNCERTAINTY_MODE=lcb MIN_SEEDS_PER_GROUP=3 \
CEBENCH_BASELINE_MAP=docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json \
bash scripts/experiments/run_b200_high_impact_queue.sh
```

Cycle-4 followups:

```bash
bash scripts/experiments/run_cycle4_followups_after_queue.sh
```

Cycle-5 external push:

```bash
bash scripts/experiments/run_cycle5_external_push.sh
```

Cycle-6 SAE-aware push:

```bash
bash scripts/experiments/run_cycle6_saeaware_push.sh
```

Cycle-7 Pareto push:

```bash
bash scripts/experiments/run_cycle7_pareto_push.sh
```

Cycle-8 robust Pareto push:

```bash
bash scripts/experiments/run_cycle8_robust_pareto_push.sh
```

Cycle-9 novelty push:

```bash
bash scripts/experiments/run_cycle9_novelty_push.sh
```

Cycle-10 external recovery push:

```bash
bash scripts/experiments/run_cycle10_external_recovery.sh
```

## 5) Current Claim Policy

Do not promote external claims unless strict gate passes (`pass_all=true`).

Latest fully completed strict-gate status remains `pass_all=false` (cycle-5 canonical run).
Live queue status is tracked in `docs/evidence/cycle8_cycle9_live_snapshot_20260217T152123Z/monitoring_summary.md`.
Canonical completed-gate artifact: `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`.
