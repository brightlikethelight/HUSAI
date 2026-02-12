# Runbook (Current State)

Updated: 2026-02-12

## 1) Environment Setup

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
pre-commit install
```

Machine-specific env workarounds observed here:
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
- write command->artifact entries into `EXPERIMENT_LOG.md`

## 5) Remaining Risks

- absolute paths still exist in several auxiliary experiment scripts
- no CI workflow checked into `.github/workflows` yet
- environment remains split across multiple spec files without lockfile
