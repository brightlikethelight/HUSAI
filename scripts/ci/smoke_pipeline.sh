#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/mpl}"
mkdir -p "$MPLCONFIGDIR"

SMOKE_DIR="${1:-$TMPDIR/husai_ci_smoke}"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

echo "[smoke] train baseline transformer (1 epoch)"
python -m scripts.training.train_baseline \
  --config configs/examples/baseline_relu.yaml \
  --epochs 1 \
  --batch-size 2048 \
  --no-wandb \
  --device cpu \
  --save-dir "$SMOKE_DIR/transformer"

echo "[smoke] extract activations"
python -m scripts.analysis.extract_activations \
  --model-path "$SMOKE_DIR/transformer/transformer_final.pt" \
  --layer 1 \
  --position answer \
  --batch-size 2048 \
  --device cpu \
  --output "$SMOKE_DIR/acts.pt"

echo "[smoke] train SAE (1 epoch)"
python -m scripts.training.train_sae \
  --transformer-checkpoint "$SMOKE_DIR/transformer/transformer_final.pt" \
  --config configs/sae/topk_8x_k32.yaml \
  --layer 1 \
  --use-cached-activations "$SMOKE_DIR/acts.pt" \
  --epochs 1 \
  --batch-size 2048 \
  --save-dir "$SMOKE_DIR/sae" \
  --no-wandb \
  --device cpu \
  --quiet

echo "[smoke] verify outputs"
test -f "$SMOKE_DIR/transformer/transformer_final.pt"
test -f "$SMOKE_DIR/acts.pt"
test -f "$SMOKE_DIR/sae/sae_final.pt"

echo "[smoke] success: artifacts in $SMOKE_DIR"
