#!/usr/bin/env python3
"""Extract activations from trained transformer for SAE training.

This script extracts activations from a trained transformer model to use
for training Sparse Autoencoders (SAEs). It supports extracting from
specific layers and positions.

Usage:
    python scripts/extract_activations.py \\
        --model-path results/transformer_best.pt \\
        --layer 1 \\
        --position answer \\
        --output results/activations/layer1_answer.pt

    python scripts/extract_activations.py \\
        --model-path results/transformer_best.pt \\
        --layer 0 \\
        --position all \\
        --batch-size 128

Example:
    $ python scripts/extract_activations.py \\
          --model-path results/transformer_best.pt --layer 1
    Extracting activations from layer 1 (answer position)...
    Loaded transformer from results/transformer_best.pt
    Processing 50000 samples...
    Saved activations: [50000, 128] to results/activations/layer1_answer.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Literal
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import create_dataloaders
from src.utils.config import ExperimentConfig


def extract_activations(
    model_path: Path,
    layer: int,
    position: Literal["answer", "all", "bos", "first_operand"] = "answer",
    batch_size: int = 256,
    device: str = "cuda",
    modulus: int = 113,
    seed: int = 42
) -> torch.Tensor:
    """Extract activations from trained transformer.

    Args:
        model_path: Path to trained transformer checkpoint
        layer: Which layer to extract from (0 = first, 1 = second for 2-layer model)
        position: Which sequence position(s) to extract:
            - "answer": Position -2 (before EOS, where answer is predicted)
            - "all": All sequence positions
            - "bos": Position 0 (BOS token)
            - "first_operand": Position 1 (first number in equation)
        batch_size: Batch size for extraction
        device: Device to use
        modulus: Modulus for dataset (default 113)
        seed: Random seed (default 42)

    Returns:
        activations: [num_samples, d_model] if position != "all"
                     [num_samples * seq_len, d_model] if position == "all"

    Example:
        >>> acts = extract_activations(
        ...     Path("results/transformer_best.pt"),
        ...     layer=1,
        ...     position="answer"
        ... )
        >>> acts.shape
        torch.Size([50000, 128])
    """
    print(f"Extracting activations from layer {layer} ({position} position)...")

    # Load model
    print(f"Loading transformer from {model_path}...")
    model, extras = ModularArithmeticTransformer.load_checkpoint(
        model_path,
        device=device
    )
    model.eval()

    print(f"Model loaded successfully (epoch {extras.get('epoch', 'unknown')})")

    # Create dataloaders
    print(f"Creating dataloaders (modulus={modulus}, batch_size={batch_size})...")
    train_loader, val_loader = create_dataloaders(
        modulus=modulus,
        fraction=1.0,  # Use full dataset
        train_fraction=0.8,
        batch_size=batch_size,
        seed=seed,
        format="sequence",
        num_workers=0
    )

    # Combine train and val for SAE training
    all_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([
            train_loader.dataset,
            val_loader.dataset
        ]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    total_samples = len(all_loader.dataset)
    print(f"Processing {total_samples} samples...")

    # Extract activations
    all_activations = []

    with torch.no_grad():
        for tokens, _ in tqdm(all_loader, desc="Extracting"):
            tokens = tokens.to(device)

            # Get activations from target layer
            acts = model.get_activations(
                tokens,
                layer=layer,
                activation_name="resid_post"
            )  # [batch, seq_len, d_model]

            # Select position(s)
            if position == "answer":
                # Position -2 (before EOS token, where answer is predicted)
                acts = acts[:, -2, :]  # [batch, d_model]
            elif position == "bos":
                # Position 0 (BOS token)
                acts = acts[:, 0, :]  # [batch, d_model]
            elif position == "first_operand":
                # Position 1 (first number in equation: [BOS, a, +, b, =, c, EOS])
                acts = acts[:, 1, :]  # [batch, d_model]
            elif position == "all":
                # All positions
                batch_size_actual, seq_len, d_model = acts.shape
                acts = acts.reshape(-1, d_model)  # [batch * seq_len, d_model]
            else:
                raise ValueError(f"Unknown position: {position}")

            all_activations.append(acts.cpu())

    # Concatenate all batches
    activations = torch.cat(all_activations, dim=0)

    print(f"Extracted activations: {list(activations.shape)}")
    print(f"  Mean: {activations.mean().item():.4f}")
    print(f"  Std: {activations.std().item():.4f}")
    print(f"  Min: {activations.min().item():.4f}")
    print(f"  Max: {activations.max().item():.4f}")

    return activations


def main():
    parser = argparse.ArgumentParser(
        description='Extract activations from trained transformer'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to trained transformer checkpoint'
    )
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer to extract from (0 or 1 for 2-layer model)'
    )
    parser.add_argument(
        '--position',
        type=str,
        default='answer',
        choices=['answer', 'all', 'bos', 'first_operand'],
        help='Sequence position to extract (default: answer)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for activations (default: results/activations/layer{layer}_{position}.pt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for extraction (default: 256)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--modulus',
        type=int,
        default=113,
        help='Modulus for dataset (default: 113)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Extract activations
    activations = extract_activations(
        model_path=args.model_path,
        layer=args.layer,
        position=args.position,
        batch_size=args.batch_size,
        device=device,
        modulus=args.modulus,
        seed=args.seed
    )

    # Determine output path
    if args.output is None:
        output_dir = Path("results/activations")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer{args.layer}_{args.position}.pt"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save activations
    print(f"\nSaving activations to {output_path}...")
    torch.save(activations, output_path)

    # Save metadata
    metadata_path = output_path.with_suffix('.meta.pt')
    metadata = {
        'model_path': str(args.model_path),
        'layer': args.layer,
        'position': args.position,
        'shape': list(activations.shape),
        'device': device,
        'modulus': args.modulus,
        'seed': args.seed,
        'statistics': {
            'mean': activations.mean().item(),
            'std': activations.std().item(),
            'min': activations.min().item(),
            'max': activations.max().item()
        }
    }
    torch.save(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")

    print("\n✓ Activation extraction complete!")


if __name__ == '__main__':
    main()
