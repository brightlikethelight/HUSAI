#!/usr/bin/env python3
"""Train baseline transformer on modular arithmetic.

This script trains a transformer model on modular arithmetic until it
exhibits grokking behavior (train accuracy → 100% fast, test accuracy →
100% slow). The trained model is then used for SAE training.

Usage:
    python scripts/train_baseline.py --config configs/examples/baseline_relu.yaml
    python scripts/train_baseline.py --config configs/examples/baseline_relu.yaml --no-wandb
    python scripts/train_baseline.py --config configs/examples/baseline_relu.yaml --epochs 10000

Example:
    $ python scripts/train_baseline.py --config configs/examples/baseline_relu.yaml
    Loading config from configs/examples/baseline_relu.yaml...
    Creating model with 1.2M parameters...
    Training for 5000 epochs...
    Epoch 100/5000: train_acc=1.000, val_acc=0.543, train_loss=0.001
    Grokking detected at epoch 1234! 🎉
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import create_dataloaders
from src.utils.config import ExperimentConfig


def train_epoch(
    model: ModularArithmeticTransformer,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for tokens, labels in train_loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(tokens)  # [batch, seq, vocab]

        # Get logits for answer position (position -2: before EOS token)
        # Sequence: [BOS, a, +, b, =, c, EOS]
        # We predict 'c' at position -2
        answer_logits = logits[:, -2, :]  # [batch, vocab]

        # Compute loss
        loss = criterion(answer_logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * tokens.size(0)
        predictions = answer_logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += tokens.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: ModularArithmeticTransformer,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Args:
        model: Transformer model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for tokens, labels in val_loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(tokens)
        answer_logits = logits[:, -2, :]

        # Compute loss
        loss = criterion(answer_logits, labels)

        # Track metrics
        total_loss += loss.item() * tokens.size(0)
        predictions = answer_logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += tokens.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def detect_grokking(
    train_acc: float,
    val_acc: float,
    train_acc_threshold: float = 0.99,
    val_acc_threshold: float = 0.95
) -> bool:
    """Detect if grokking has occurred.

    Grokking is when the model achieves high train accuracy AND high val accuracy.

    Args:
        train_acc: Current training accuracy
        val_acc: Current validation accuracy
        train_acc_threshold: Threshold for train accuracy (default 0.99)
        val_acc_threshold: Threshold for val accuracy (default 0.95)

    Returns:
        True if grokking detected
    """
    return train_acc >= train_acc_threshold and val_acc >= val_acc_threshold


def main():
    parser = argparse.ArgumentParser(description='Train baseline transformer')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to experiment config YAML file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size (default: 512)'
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=None,
        help='Directory to save checkpoints (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=500,
        help='Save checkpoint every N epochs (default: 500)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = ExperimentConfig.from_yaml(args.config)

    # Override config with command-line args
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.epochs is not None:
        num_epochs = args.epochs
    else:
        # Default to 5000 epochs if not specified
        num_epochs = 5000

    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=config.wandb_project,
                name=f"{config.experiment_name}_transformer",
                config={
                    **config.to_dict(),
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'num_epochs': num_epochs,
                },
                tags=['transformer', 'baseline', 'modular-arithmetic']
            )
            print(f"W&B initialized: {wandb.run.get_url()}")
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            print("Continuing without W&B logging...")
            use_wandb = False

    # Create model
    print(f"\nCreating model...")
    model = ModularArithmeticTransformer(config.transformer, device=device)
    num_params = model.count_parameters()
    print(f"Model created with {num_params:,} parameters")
    print(model)

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        modulus=config.dataset.modulus,
        fraction=1.0,  # Use full dataset
        train_fraction=config.dataset.train_split,
        batch_size=args.batch_size,
        seed=config.dataset.seed,
        format="sequence",  # Use sequence format
        num_workers=0  # Single-threaded for reproducibility
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create optimizer and loss function
    optimizer = optim.AdamW(model.model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'='*60}\n")

    grokking_epoch = None
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log metrics
        metrics = {
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'learning_rate': args.lr,
            'epoch': epoch,
        }

        if use_wandb:
            wandb.log(metrics, step=epoch)

        # Print progress
        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Detect grokking
        if grokking_epoch is None and detect_grokking(train_acc, val_acc):
            grokking_epoch = epoch
            print(f"\n🎉 Grokking detected at epoch {epoch}! 🎉")
            print(f"   Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}\n")

            if use_wandb:
                wandb.log({'grokking_epoch': epoch}, step=epoch)

        # Save checkpoint
        if epoch % args.checkpoint_freq == 0:
            checkpoint_path = save_dir / f"transformer_epoch_{epoch}.pt"
            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = save_dir / "transformer_best.pt"
            model.save_checkpoint(
                best_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics
            )

    # Save final checkpoint
    final_checkpoint_path = save_dir / "transformer_final.pt"
    model.save_checkpoint(
        final_checkpoint_path,
        optimizer=optimizer,
        epoch=num_epochs,
        metrics=metrics
    )
    print(f"\nFinal checkpoint saved to {final_checkpoint_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final val accuracy: {val_acc:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    if grokking_epoch is not None:
        print(f"Grokking detected at epoch: {grokking_epoch}")
    else:
        print("Warning: Grokking not detected. May need more epochs or different hyperparameters.")
    print(f"Checkpoints saved to: {save_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
