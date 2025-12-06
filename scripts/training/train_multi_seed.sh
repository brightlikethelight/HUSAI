#!/bin/bash
# Train multiple SAEs with different seeds for stability analysis
# This automates the multi-seed experiments for Week 3

set -e  # Exit on error

# Configuration
TRANSFORMER_CHECKPOINT="${1:-results/transformer_5000ep/transformer_best.pt}"
LAYER="${2:-1}"
ARCHITECTURE="${3:-topk}"  # topk or relu
SEEDS=(42 123 456 789 1011)

# Validate inputs
if [ ! -f "$TRANSFORMER_CHECKPOINT" ]; then
    echo "Error: Transformer checkpoint not found: $TRANSFORMER_CHECKPOINT"
    echo "Usage: $0 <transformer_checkpoint> <layer> <architecture>"
    exit 1
fi

# Set config based on architecture
if [ "$ARCHITECTURE" = "topk" ]; then
    CONFIG="configs/sae/topk_8x_k32.yaml"
    BASE_DIR="results/saes/topk"
elif [ "$ARCHITECTURE" = "relu" ]; then
    CONFIG="configs/sae/relu_8x.yaml"
    BASE_DIR="results/saes/relu"
else
    echo "Error: Unknown architecture: $ARCHITECTURE"
    echo "Must be 'topk' or 'relu'"
    exit 1
fi

echo "============================================================"
echo "MULTI-SEED SAE TRAINING"
echo "============================================================"
echo "Architecture: $ARCHITECTURE"
echo "Config: $CONFIG"
echo "Transformer: $TRANSFORMER_CHECKPOINT"
echo "Layer: $LAYER"
echo "Seeds: ${SEEDS[@]}"
echo "============================================================"
echo ""

# Train SAEs with different seeds
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Training SAE with seed $seed ($(date +%H:%M:%S))"
    echo "------------------------------------------------------------"
    
    SAVE_DIR="${BASE_DIR}_seed${seed}"
    
    python scripts/train_sae.py \
        --transformer-checkpoint "$TRANSFORMER_CHECKPOINT" \
        --config "$CONFIG" \
        --layer "$LAYER" \
        --seed "$seed" \
        --save-dir "$SAVE_DIR" \
        || {
            echo "ERROR: Training failed for seed $seed"
            exit 1
        }
    
    echo "✅ Completed seed $seed"
done

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE!"
echo "============================================================"
echo "Trained ${#SEEDS[@]} SAEs with architecture: $ARCHITECTURE"
echo "Checkpoints saved to: $BASE_DIR*"
echo ""
echo "Next steps:"
echo "  1. Analyze feature stability:"
echo "     python scripts/analyze_feature_stability.py \\"
echo "         --sae-dir $BASE_DIR \\"
echo "         --pattern '${ARCHITECTURE}_seed*.pt'"
echo ""
echo "  2. View results in W&B:"
echo "     https://wandb.ai/brightliu-harvard-university/husai-sae-stability"
echo "============================================================"
