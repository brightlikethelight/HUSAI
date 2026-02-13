# Runbook (Current State)

Updated: 2026-02-13

Primary navigation index: `REPO_NAVIGATION.md`

## 1) Environment Setup

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
pre-commit install
```

Machine-specific env workarounds observed in this workspace:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export TMPDIR=/tmp
```

## 2) Core Smoke Workflow

### A. Baseline transformer train (1 epoch smoke)
```bash
python -m scripts.training.train_baseline \
  --config configs/examples/baseline_relu.yaml \
  --epochs 1 \
  --batch-size 128 \
  --no-wandb \
  --save-dir /tmp/husai_demo
```

### B. Activation extraction
```bash
python -m scripts.analysis.extract_activations \
  --model-path /tmp/husai_demo/transformer_final.pt \
  --layer 1 \
  --position answer \
  --batch-size 128 \
  --output /tmp/husai_demo/acts.pt
```

### C. SAE training smoke
```bash
python scripts/training/train_sae.py \
  --transformer-checkpoint /tmp/husai_demo/transformer_final.pt \
  --config configs/sae/topk_8x_k32.yaml \
  --layer 1 \
  --use-cached-activations /tmp/husai_demo/acts.pt \
  --epochs 1 \
  --batch-size 128 \
  --save-dir /tmp/husai_train_sae \
  --no-wandb \
  --device cpu \
  --quiet
```

### D. End-to-end pipeline check
```bash
python tests/test_sae_pipeline.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt
```

## 3) Test Commands

```bash
pytest tests/unit -q
pytest tests/integration -q
pytest tests -q
```

Current status in this workspace: full test suite passes.

## 4) Recommended Repro Controls

- set explicit seeds for Python/NumPy/Torch
- persist full config with each run
- include commit hash in run metadata
- log trained-vs-random baseline for every stability metric
- write command-to-artifact entries into `EXPERIMENT_LOG.md`

## 5) Remaining Risks

- CE-Bench compatibility execution is completed for a public SAE target (`run_20260213T103218Z`), with evidence tracked under `docs/evidence/phase4e_cebench_official/`.
- HUSAI custom SAEBench path is now integrated and reproducible, but current external AUC remains below baseline (see `docs/evidence/phase4e_husai_custom_multiseed/summary.json`).
- Environment specs remain split across `environment.yml`, `requirements*.txt`, and `pyproject.toml` without lockfile pinning.
- CI lint/typecheck are intentionally incremental because repository-wide static-analysis debt is still high.

## 6) CI and Follow-up Automation

### CI workflows
- Main workflow: `.github/workflows/ci.yml`
  - `smoke` job: fail-fast end-to-end smoke (`scripts/ci/smoke_pipeline.sh`)
  - `quality` job: lint (`flake8`), typecheck (`mypy`), tests (`pytest`)

### Local smoke target
```bash
make smoke
# or
scripts/ci/smoke_pipeline.sh /tmp/husai_ci_smoke
```

### Reproduction / Ablation / Benchmark commands
```bash
# Phase 4a: trained vs random reproduction with manifest
python scripts/experiments/run_phase4a_reproduction.py

# Phase 4c: core ablations (k sweep + d_sae sweep) with CIs
python scripts/experiments/run_core_ablations.py --device cpu --epochs 20

# Phase 4e: SAEBench/CE-Bench-aligned local slice
python scripts/experiments/run_external_benchmark_slice.py

# Phase 4e: official benchmark harness preflight (no execution)
python scripts/experiments/run_official_external_benchmarks.py

# Phase 4e: official benchmark harness execute mode
python scripts/experiments/run_official_external_benchmarks.py \
  --saebench-repo /path/to/SAEBench \
  --cebench-repo /path/to/CE-Bench \
  --saebench-command "<official SAEBench command>" \
  --cebench-command "<official CE-Bench command>" \
  --execute
```

HUSAI custom-checkpoint SAEBench execution example:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --skip-cebench \
  --husai-saebench-checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --husai-saebench-release husai_pythia70m_topk_seed42 \
  --husai-saebench-model-name pythia-70m-deduped \
  --husai-saebench-hook-layer 0 \
  --husai-saebench-hook-name blocks.0.hook_resid_pre \
  --husai-saebench-ks 1,2,5 \
  --husai-saebench-force-rerun \
  --execute
```

CE-Bench official command pattern (from CE-Bench repo):
```bash
python ce_bench/CE_Bench.py \
  --sae_regex_pattern "<pattern>" \
  --sae_block_pattern "<block_pattern>" \
  --output_folder <output_dir> \
  --artifacts_path <artifacts_dir>
```

### Artifact roots
- `results/experiments/phase4a_trained_vs_random/`
- `results/experiments/phase4c_core_ablations/`
- `results/experiments/phase4e_external_benchmark_slice/`
- `results/experiments/phase4e_external_benchmark_official/`

## 7) Highest-Impact Follow-up Commands

Adaptive L0 calibration (search + retrain):
```bash
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu
```

Matched-control retrain at fixed `k=32`:
```bash
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu --k-candidates 32
```

Consistency-objective sweep:
```bash
python scripts/experiments/run_consistency_regularization_sweep.py --device cpu --k 4
```

Results-consistency audit against artifact JSONs:
```bash
python scripts/analysis/verify_experiment_consistency.py
```

Follow-up artifacts:
- `results/experiments/adaptive_l0_calibration/`
- `results/experiments/consistency_objective_sweep/`
- `results/analysis/experiment_consistency_report.json`
- `results/analysis/experiment_consistency_report.md`
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`


## 8) Git Hygiene

- Keep `main` as the stable narrative branch; merge only artifact-backed updates.
- Use small, reviewable commits grouped by concern (infra, experiments, writeups).
- Before push: run `pytest tests -q` and update `EXPERIMENT_LOG.md` for any executed runs.
- Prefer adding new experiment runs under `results/experiments/<phase>/<run_id>/` with `manifest.json`.

Suggested sync commands:
```bash
git status -sb
git log --oneline --decorate -n 12
git push origin main
```

## 9) New High-Impact Pipelines (Cycle 2)

Direct HUSAI-checkpoint CE-Bench adapter:
```bash
python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo /path/to/CE-Bench \
  --checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre
```

Combined official + custom CE-Bench harness run (matched baseline mode):
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --cebench-repo /path/to/CE-Bench \
  --cebench-use-compat-runner \
  --cebench-sae-regex-pattern pythia-70m-deduped-res-sm \
  --cebench-sae-block-pattern blocks.0.hook_resid_pre \
  --cebench-force-rerun \
  --husai-cebench-checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --husai-cebench-match-baseline \
  --execute
```

Matched-budget architecture frontier (external metrics):
```bash
python scripts/experiments/run_architecture_frontier_external.py \
  --run-saebench \
  --run-cebench \
  --cebench-repo /path/to/CE-Bench \
  --architectures topk,relu,batchtopk,jumprelu \
  --seeds 42,123,456
```

External scaling study (token budget, hook layer, `d_sae`):
```bash
python scripts/experiments/run_external_metric_scaling_study.py \
  --run-saebench \
  --run-cebench \
  --cebench-repo /path/to/CE-Bench \
  --token-budgets 50000,100000,150000 \
  --hook-layers 0,1 \
  --d-sae-values 1024,2048
```

Assignment-aware consistency objective v2:
```bash
python scripts/experiments/run_assignment_consistency_v2.py \
  --device cpu \
  --lambdas 0.0,0.01,0.05,0.1,0.2 \
  --external-summary <path/to/external/summary.json>
```

Stress-gated release policy:
```bash
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results <path/to/transcoder.json> \
  --ood-results <path/to/ood.json> \
  --external-summary <path/to/external/summary.json> \
  --require-transcoder --require-ood --require-external
```
