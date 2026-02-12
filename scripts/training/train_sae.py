#!/usr/bin/env python3
"""Train Sparse Autoencoder (SAE) on transformer activations.

This script provides a CLI interface for training SAEs on activations
extracted from a trained transformer. It handles the full pipeline:
1. Load trained transformer checkpoint
2. Extract activations from target layer
3. Create and configure SAE
4. Train SAE with W&B logging
5. Save trained SAE checkpoint

Usage:
    # Train TopK SAE on layer 1
    python scripts/train_sae.py \\
        --transformer-checkpoint results/transformer_best.pt \\
        --config configs/examples/topk_16x.yaml \\
        --layer 1 \\
        --seed 42

    # Train ReLU SAE with custom settings
    python scripts/train_sae.py \\
        --transformer-checkpoint results/transformer_best.pt \\
        --config configs/examples/baseline_relu.yaml \\
        --layer 1 \\
        --seed 123 \\
        --epochs 20 \\
        --batch-size 1024

    # Train without W&B logging
    python scripts/train_sae.py \\
        --transformer-checkpoint results/transformer_best.pt \\
        --config configs/examples/topk_16x.yaml \\
        --layer 1 \\
        --no-wandb

Example Output:
    $ python scripts/train_sae.py --transformer-checkpoint results/transformer_best.pt \\
          --config configs/examples/topk_16x.yaml --layer 1 --seed 42

    Loading config from configs/examples/topk_16x.yaml...
    Loading transformer from results/transformer_best.pt...
    Extracting activations from layer 1...
    Creating TopK SAE (16x expansion, k=32)...
    Starting SAE training (20 epochs)...
    Epoch 1/20: loss=0.0234, l0=31.2, explained_var=0.812
    ...
    Training complete! Saved to results/saes/topk_layer1_seed42/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.models.sae import SAEWrapper
from src.training.train_sae import train_sae
from src.utils.config import ExperimentConfig, SAEConfig
from scripts.analysis.extract_activations import extract_activations


def main():
    parser = argparse.ArgumentParser(
        description='Train SAE on transformer activations'
    )

    # Required arguments
    parser.add_argument(
        '--transformer-checkpoint',
        type=Path,
        required=True,
        help='Path to trained transformer checkpoint'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to experiment config YAML file'
    )
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer to extract activations from (0 or 1 for 2-layer model)'
    )

    # Optional arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config if specified)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config if specified)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for SAE training (overrides config if specified)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config if specified)'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=None,
        help='Directory to save SAE checkpoints (default: results/saes/{architecture}_layer{layer}_seed{seed}/)'
    )
    parser.add_argument(
        '--position',
        type=str,
        default='answer',
        choices=['answer', 'all', 'bos', 'first_operand'],
        help='Sequence position to extract activations from (default: answer)'
    )
    parser.add_argument(
        '--use-cached-activations',
        type=Path,
        default=None,
        help='Path to pre-extracted activations (skips extraction step)'
    )
    parser.add_argument(
        '--cache-activations',
        action='store_true',
        help='Save extracted activations for future use'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='husai-sae-stability',
        help='W&B project name (default: husai-sae-stability)'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='W&B run name (default: sae_{architecture}_layer{layer}_seed{seed})'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=5,
        help='Save checkpoint every N epochs (default: 5)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n{'='*70}")
    print(f"HUSAI SAE Training Pipeline")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Transformer: {args.transformer_checkpoint}")
    print(f"Config: {args.config}")
    print(f"Layer: {args.layer}")
    print(f"Position: {args.position}")

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = ExperimentConfig.from_yaml(args.config)

    # Override config with command-line args
    if args.seed is not None:
        config.sae.seed = args.seed
        print(f"  Seed override: {args.seed}")

    if args.epochs is not None:
        config.sae.num_epochs = args.epochs
        print(f"  Epochs override: {args.epochs}")

    if args.batch_size is not None:
        config.sae.batch_size = args.batch_size
        print(f"  Batch size override: {args.batch_size}")

    if args.learning_rate is not None:
        config.sae.learning_rate = args.learning_rate
        print(f"  Learning rate override: {args.learning_rate}")

    # Set random seed for reproducibility
    if config.sae.seed is not None:
        torch.manual_seed(config.sae.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.sae.seed)
        print(f"  Random seed set: {config.sae.seed}")

    # Determine save directory
    if args.save_dir is None:
        save_dir = Path("results/saes") / f"{config.sae.architecture}_layer{args.layer}_seed{config.sae.seed}"
    else:
        save_dir = args.save_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSave directory: {save_dir}")

    # Extract or load activations
    if args.use_cached_activations is not None:
        print(f"\nLoading cached activations from {args.use_cached_activations}...")
        activations = torch.load(args.use_cached_activations)
        print(f"  Loaded activations: {list(activations.shape)}")
    else:
        print(f"\nExtracting activations from layer {args.layer} ({args.position} position)...")
        activations = extract_activations(
            model_path=args.transformer_checkpoint,
            layer=args.layer,
            position=args.position,
            batch_size=256,
            device=device,
            modulus=config.dataset.modulus,
            seed=config.dataset.seed
        )

        # Cache activations if requested
        if args.cache_activations:
            cache_path = save_dir / f"activations_layer{args.layer}_{args.position}.pt"
            print(f"  Caching activations to {cache_path}...")
            torch.save(activations, cache_path)

    # Create SAE
    print(f"\nCreating {config.sae.architecture.upper()} SAE...")
    print(f"  Input dim: {config.sae.input_dim}")
    print(f"  Expansion factor: {config.sae.expansion_factor}x")
    print(f"  SAE dim: {config.sae.input_dim * config.sae.expansion_factor}")

    if config.sae.architecture == "topk":
        print(f"  TopK k: {config.sae.k}")
    elif config.sae.architecture == "relu":
        print(f"  L1 coefficient: {config.sae.l1_coefficient}")

    sae = SAEWrapper(config.sae, device=device)

    # Determine W&B run name
    if args.wandb_run_name is None:
        wandb_run_name = f"sae_{config.sae.architecture}_layer{args.layer}_seed{config.sae.seed}"
    else:
        wandb_run_name = args.wandb_run_name

    # Train SAE
    print(f"\n{'='*70}")
    print(f"Starting SAE Training")
    print(f"{'='*70}")
    print(f"  Epochs: {config.sae.num_epochs}")
    print(f"  Batch size: {config.sae.batch_size}")
    print(f"  Learning rate: {config.sae.learning_rate}")
    print(f"  W&B: {'enabled' if not args.no_wandb else 'disabled'}")
    if not args.no_wandb:
        print(f"  W&B project: {args.wandb_project}")
        print(f"  W&B run: {wandb_run_name}")
    print()

    metrics = train_sae(
        sae=sae,
        activations=activations,
        config=config.sae,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
        device=device,
        checkpoint_dir=save_dir,
        checkpoint_freq=args.checkpoint_freq,
        verbose=not args.quiet
    )

    # Save final summary
    summary_path = save_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"HUSAI SAE Training Summary\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Architecture: {config.sae.architecture}\n")
        f.write(f"  Layer: {args.layer}\n")
        f.write(f"  Position: {args.position}\n")
        f.write(f"  Seed: {config.sae.seed}\n")
        f.write(f"  Input dim: {config.sae.input_dim}\n")
        f.write(f"  Expansion factor: {config.sae.expansion_factor}x\n")
        f.write(f"  SAE dim: {config.sae.input_dim * config.sae.expansion_factor}\n")
        f.write(f"\n")
        f.write(f"Training:\n")
        f.write(f"  Epochs: {config.sae.num_epochs}\n")
        f.write(f"  Batch size: {config.sae.batch_size}\n")
        f.write(f"  Learning rate: {config.sae.learning_rate}\n")
        f.write(f"\n")
        f.write(f"Final Metrics:\n")
        f.write(f"  Loss: {metrics.loss[-1]:.6f}\n")
        f.write(f"  MSE Loss: {metrics.mse_loss[-1]:.6f}\n")
        f.write(f"  Sparsity Loss: {metrics.sparsity_loss[-1]:.6f}\n")
        f.write(f"  L0: {metrics.l0[-1]:.2f}\n")
        f.write(f"  L1: {metrics.l1[-1]:.4f}\n")
        f.write(f"  Explained Variance: {metrics.explained_variance[-1]:.4f}\n")
        f.write(f"  Dead Neurons: {metrics.dead_neuron_count[-1]}/{config.sae.input_dim * config.sae.expansion_factor}")
        f.write(f" ({metrics.dead_neuron_pct[-1]:.2f}%)\n")

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"\nCheckpoints saved to: {save_dir}/")
    print(f"  - sae_final.pt (final model)")
    print(f"  - sae_epoch_*.pt (periodic checkpoints)")
    print(f"  - training_summary.txt (metrics summary)")
    if args.cache_activations:
        print(f"  - activations_layer{args.layer}_{args.position}.pt (cached activations)")

    print(f"\nFinal Metrics:")
    print(f"  L0 Sparsity: {metrics.l0[-1]:.2f} active features")
    print(f"  Explained Variance: {metrics.explained_variance[-1]:.4f}")
    print(f"  Dead Neurons: {metrics.dead_neuron_count[-1]}/{config.sae.input_dim * config.sae.expansion_factor}")
    print(f" ({metrics.dead_neuron_pct[-1]:.2f}%)")

    # Quality assessment
    print(f"\nQuality Assessment:")
    if metrics.explained_variance[-1] > 0.85:
        print(f"  ✓ Excellent reconstruction (EV > 0.85)")
    elif metrics.explained_variance[-1] > 0.75:
        print(f"  ✓ Good reconstruction (EV > 0.75)")
    elif metrics.explained_variance[-1] > 0.60:
        print(f"  ⚠ Moderate reconstruction (EV > 0.60)")
    else:
        print(f"  ✗ Poor reconstruction (EV < 0.60) - consider adjusting hyperparameters")

    if metrics.dead_neuron_pct[-1] < 10:
        print(f"  ✓ Low dead neuron rate (<10%)")
    elif metrics.dead_neuron_pct[-1] < 25:
        print(f"  ⚠ Moderate dead neuron rate (10-25%)")
    else:
        print(f"  ✗ High dead neuron rate (>25%) - consider auxiliary loss or resampling")

    if config.sae.architecture == "topk":
        expected_l0 = config.sae.k
        if abs(metrics.l0[-1] - expected_l0) < expected_l0 * 0.2:
            print(f"  ✓ L0 close to target k={config.sae.k}")
        else:
            print(f"  ⚠ L0 ({metrics.l0[-1]:.1f}) differs from target k={config.sae.k}")

    print(f"\n✓ SAE training complete!")


if __name__ == '__main__':
    main()
