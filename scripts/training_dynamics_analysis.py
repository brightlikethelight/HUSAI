#!/usr/bin/env python3
"""Training Dynamics Analysis - CRITICAL EXPERIMENT (Task 4.3)

Analyzes WHEN SAE features diverge during training to test the underconstrained
reconstruction hypothesis.

Key Questions:
1. Do features start aligned and diverge? Or diverge from the start?
2. Is there a critical epoch where divergence accelerates?
3. Does PWMCC stabilize or continue changing?

Methodology:
- Train 3 TopK SAEs (seeds 42, 123, 456) from scratch
- Save checkpoints every 5 epochs
- Compute decoder-based PWMCC at each checkpoint
- Visualize PWMCC evolution during training

Expected Insight:
- If underconstrained: PWMCC should decrease over training (features diverge)
- If overconstrained: PWMCC should increase over training (features converge)
- If optimal: PWMCC should stabilize to a unique solution
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import sys

# Add src to path
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
sys.path.insert(0, str(BASE_DIR))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import create_dataloaders
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# Configuration
# ============================================================================

SEEDS = [42, 123, 456]
LAYER = 0  # Train on Layer 0 activations
NUM_EPOCHS = 50
CHECKPOINT_FREQ = 5  # Save every 5 epochs
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
EXPANSION_FACTOR = 8
K = 32  # TopK parameter

# Paths
RESULTS_DIR = BASE_DIR / 'results'
TRANSFORMER_PATH = RESULTS_DIR / 'transformer_5000ep' / 'transformer_final.pt'
ACTIVATIONS_PATH = RESULTS_DIR / 'training_dynamics' / 'activations_layer0.pt'
CHECKPOINT_BASE_DIR = RESULTS_DIR / 'training_dynamics'
FIGURES_DIR = BASE_DIR / 'figures'

# Output
OUTPUT_JSON = RESULTS_DIR / 'analysis' / 'training_dynamics_results.json'
OUTPUT_FIGURE = FIGURES_DIR / 'pwmcc_training_dynamics.pdf'

# ============================================================================
# PWMCC Computation (Decoder-Based)
# ============================================================================

def compute_pwmcc_decoder(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC using decoder weights (CORRECT method).

    Args:
        decoder1: Decoder weights [d_model, d_sae]
        decoder2: Decoder weights [d_model, d_sae]

    Returns:
        PWMCC score (0-1)
    """
    # Normalize each feature column to unit norm
    d1_norm = F.normalize(decoder1, dim=0)  # [d_model, d_sae]
    d2_norm = F.normalize(decoder2, dim=0)

    # Cosine similarity matrix [d_sae, d_sae]
    cos_sim = d1_norm.T @ d2_norm

    # Symmetric PWMCC: average of max in both directions
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()

    return (max_1to2 + max_2to1) / 2


def load_decoder_from_checkpoint(checkpoint_path: Path) -> torch.Tensor:
    """Load decoder weights from SAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'sae_state_dict' in checkpoint:
        state_dict = checkpoint['sae_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # SAELens stores decoder as W_dec [d_sae, d_in], but we need [d_in, d_sae]
    if 'W_dec' in state_dict:
        return state_dict['W_dec'].T
    elif 'decoder.weight' in state_dict:
        return state_dict['decoder.weight']
    else:
        raise KeyError(f"Could not find decoder weights in checkpoint: {list(state_dict.keys())}")


# ============================================================================
# Activation Extraction
# ============================================================================

def extract_layer_activations(
    model_path: Path,
    layer: int,
    save_path: Path,
    device: str = 'cuda',
    modulus: int = 113,
    batch_size: int = 256
) -> torch.Tensor:
    """Extract activations from a specific layer of the transformer.

    Args:
        model_path: Path to trained transformer
        layer: Layer index to extract from
        save_path: Where to save extracted activations
        device: Device to use
        modulus: Modulus for dataset
        batch_size: Batch size for extraction

    Returns:
        Extracted activations [num_samples, d_model]
    """
    print(f"\nExtracting Layer {layer} activations from transformer...")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    # Load transformer
    model, extras = ModularArithmeticTransformer.load_checkpoint(
        model_path,
        device=device
    )
    model.eval()
    print(f"  ✓ Loaded transformer (epoch {extras.get('epoch', 'unknown')})")

    # Create dataloader
    train_loader, _ = create_dataloaders(
        modulus=modulus,
        fraction=1.0,
        train_fraction=0.8,
        batch_size=batch_size,
        seed=42
    )

    # Extract activations
    all_activations = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting activations"):
            inputs = batch[0].to(device)

            # Get activations using TransformerLens interface
            layer_acts = model.get_activations(inputs, layer=layer)

            # Get activations at answer position (position -2)
            # layer_acts: [batch, seq, d_model]
            answer_acts = layer_acts[:, -2, :]  # Answer position

            all_activations.append(answer_acts.cpu())

    # Concatenate all batches
    activations = torch.cat(all_activations, dim=0)
    print(f"  ✓ Extracted activations: {activations.shape}")

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(activations, save_path)
    print(f"  ✓ Saved to: {save_path}")

    return activations


# ============================================================================
# Training with Checkpoints
# ============================================================================

def train_sae_with_checkpoints(
    activations: torch.Tensor,
    seed: int,
    checkpoint_dir: Path,
    verbose: bool = True
) -> SAETrainingMetrics:
    """Train SAE from scratch and save checkpoints every N epochs.

    Args:
        activations: Training activations [num_samples, d_model]
        seed: Random seed for initialization
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print progress

    Returns:
        Training metrics
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get dimensions
    num_samples, d_model = activations.shape

    # Create SAE config
    config = SAEConfig(
        architecture="topk",
        input_dim=d_model,
        expansion_factor=EXPANSION_FACTOR,
        sparsity_level=K,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        k=K,
        seed=seed
    )

    # Create SAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"\nTraining SAE (seed={seed}) on {device}...")
        print(f"  Input dim: {d_model}")
        print(f"  SAE dim: {d_model * EXPANSION_FACTOR}")
        print(f"  Num samples: {num_samples}")
        print(f"  Epochs: {NUM_EPOCHS}")
        print(f"  Checkpoint freq: {CHECKPOINT_FREQ}")

    # Train with checkpointing
    sae = create_sae(config, device=device)
    metrics = train_sae(
        sae=sae,
        activations=activations,
        config=config,
        use_wandb=False,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=CHECKPOINT_FREQ,
        verbose=verbose
    )

    return metrics


# ============================================================================
# PWMCC Evolution Analysis
# ============================================================================

def compute_pwmcc_evolution(
    checkpoint_dirs: Dict[int, Path],
    epochs: List[int]
) -> Dict[str, List[float]]:
    """Compute PWMCC between all seed pairs at each checkpoint epoch.

    Args:
        checkpoint_dirs: Mapping from seed to checkpoint directory
        epochs: List of epochs to analyze

    Returns:
        Dictionary mapping seed_pair to PWMCC values over epochs
    """
    results = {
        'epochs': epochs,
        'pwmcc_42_vs_123': [],
        'pwmcc_42_vs_456': [],
        'pwmcc_123_vs_456': []
    }

    print("\nComputing PWMCC evolution across training...")

    for epoch in tqdm(epochs, desc="Analyzing epochs"):
        # Load decoders for this epoch
        decoders = {}

        for seed in SEEDS:
            checkpoint_path = checkpoint_dirs[seed] / f"sae_epoch_{epoch}.pt"
            if not checkpoint_path.exists():
                # Try final checkpoint if this is the last epoch
                if epoch == NUM_EPOCHS:
                    checkpoint_path = checkpoint_dirs[seed] / "sae_final.pt"

                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            decoders[seed] = load_decoder_from_checkpoint(checkpoint_path)

        # Compute pairwise PWMCC
        pwmcc_42_123 = compute_pwmcc_decoder(decoders[42], decoders[123])
        pwmcc_42_456 = compute_pwmcc_decoder(decoders[42], decoders[456])
        pwmcc_123_456 = compute_pwmcc_decoder(decoders[123], decoders[456])

        results['pwmcc_42_vs_123'].append(pwmcc_42_123)
        results['pwmcc_42_vs_456'].append(pwmcc_42_456)
        results['pwmcc_123_vs_456'].append(pwmcc_123_456)

    return results


# ============================================================================
# Visualization
# ============================================================================

def visualize_pwmcc_dynamics(
    results: Dict[str, List[float]],
    save_path: Path
):
    """Create publication-quality visualization of PWMCC evolution.

    Args:
        results: Dictionary with epochs and PWMCC values
        save_path: Path to save figure
    """
    epochs = results['epochs']

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PWMCC evolution for each seed pair
    ax.plot(epochs, results['pwmcc_42_vs_123'],
            marker='o', linewidth=2, markersize=6,
            label='Seed 42 vs 123', alpha=0.8)
    ax.plot(epochs, results['pwmcc_42_vs_456'],
            marker='s', linewidth=2, markersize=6,
            label='Seed 42 vs 456', alpha=0.8)
    ax.plot(epochs, results['pwmcc_123_vs_456'],
            marker='^', linewidth=2, markersize=6,
            label='Seed 123 vs 456', alpha=0.8)

    # Compute mean PWMCC across all pairs
    mean_pwmcc = [
        np.mean([
            results['pwmcc_42_vs_123'][i],
            results['pwmcc_42_vs_456'][i],
            results['pwmcc_123_vs_456'][i]
        ])
        for i in range(len(epochs))
    ]
    ax.plot(epochs, mean_pwmcc,
            linewidth=3, linestyle='--', color='black',
            label='Mean PWMCC', alpha=0.6)

    # Add reference lines
    ax.axhline(y=0.30, color='red', linestyle=':', linewidth=2, alpha=0.5,
               label='Random Baseline (0.30)')

    # Styling
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decoder-Based PWMCC', fontsize=14, fontweight='bold')
    ax.set_title('SAE Feature Alignment During Training\n'
                 'TopK SAE (k=32, 8x expansion) on Layer 0',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])

    # Add text annotation for interpretation
    initial_pwmcc = mean_pwmcc[0]
    final_pwmcc = mean_pwmcc[-1]
    delta_pwmcc = final_pwmcc - initial_pwmcc

    interpretation = (
        f"Initial PWMCC: {initial_pwmcc:.3f}\n"
        f"Final PWMCC: {final_pwmcc:.3f}\n"
        f"Change: {delta_pwmcc:+.3f}"
    )

    ax.text(0.02, 0.98, interpretation,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {save_path}")

    # Also save as PNG for easy viewing
    png_path = save_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ PNG saved: {png_path}")

    plt.close()


# ============================================================================
# Analysis and Interpretation
# ============================================================================

def analyze_dynamics(results: Dict[str, List[float]]) -> Dict:
    """Analyze PWMCC dynamics and answer key questions.

    Returns:
        Dictionary with analysis results
    """
    epochs = results['epochs']
    mean_pwmcc = [
        np.mean([
            results['pwmcc_42_vs_123'][i],
            results['pwmcc_42_vs_456'][i],
            results['pwmcc_123_vs_456'][i]
        ])
        for i in range(len(epochs))
    ]

    # Compute statistics
    initial_pwmcc = mean_pwmcc[0]
    final_pwmcc = mean_pwmcc[-1]
    delta_pwmcc = final_pwmcc - initial_pwmcc
    max_pwmcc = max(mean_pwmcc)
    min_pwmcc = min(mean_pwmcc)

    # Detect divergence point (epoch where PWMCC changes most)
    pwmcc_changes = [abs(mean_pwmcc[i+1] - mean_pwmcc[i])
                     for i in range(len(mean_pwmcc)-1)]
    max_change_idx = np.argmax(pwmcc_changes)
    divergence_epoch = epochs[max_change_idx]

    # Interpret dynamics
    if delta_pwmcc < -0.05:
        interpretation = "DIVERGENT: Features start aligned but diverge over training (underconstrained)"
    elif delta_pwmcc > 0.05:
        interpretation = "CONVERGENT: Features start misaligned but converge over training (overconstrained)"
    else:
        interpretation = "STABLE: PWMCC remains relatively constant (optimal constraint level)"

    # Check if PWMCC stabilizes (variance in last 20% of training)
    late_training_idx = int(0.8 * len(mean_pwmcc))
    late_pwmcc = mean_pwmcc[late_training_idx:]
    late_variance = np.var(late_pwmcc)
    is_stable = late_variance < 0.001

    analysis = {
        'initial_pwmcc': float(initial_pwmcc),
        'final_pwmcc': float(final_pwmcc),
        'delta_pwmcc': float(delta_pwmcc),
        'max_pwmcc': float(max_pwmcc),
        'min_pwmcc': float(min_pwmcc),
        'divergence_epoch': int(divergence_epoch),
        'max_change': float(max(pwmcc_changes)),
        'late_training_variance': float(late_variance),
        'is_stable': bool(is_stable),
        'interpretation': interpretation,
        'mean_pwmcc_by_epoch': {int(e): float(p) for e, p in zip(epochs, mean_pwmcc)}
    }

    return analysis


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("TRAINING DYNAMICS ANALYSIS - CRITICAL EXPERIMENT (Task 4.3)")
    print("="*80)
    print()
    print("This experiment tests the underconstrained reconstruction hypothesis by")
    print("analyzing WHEN SAE features diverge during training.")
    print()
    print(f"Configuration:")
    print(f"  Seeds: {SEEDS}")
    print(f"  Layer: {LAYER}")
    print(f"  Architecture: TopK (k={K})")
    print(f"  Expansion: {EXPANSION_FACTOR}x")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Checkpoint frequency: every {CHECKPOINT_FREQ} epochs")
    print()

    # Load or extract activations
    if ACTIVATIONS_PATH.exists():
        print(f"Loading cached activations from: {ACTIVATIONS_PATH}")
        activations = torch.load(ACTIVATIONS_PATH)
        print(f"✓ Loaded activations: {activations.shape}")
    else:
        print(f"Activations not found, extracting from transformer...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        activations = extract_layer_activations(
            model_path=TRANSFORMER_PATH,
            layer=LAYER,
            save_path=ACTIVATIONS_PATH,
            device=device
        )
    print()

    # Create checkpoint directories
    checkpoint_dirs = {
        seed: CHECKPOINT_BASE_DIR / f'seed{seed}'
        for seed in SEEDS
    }

    # Train SAEs (or load existing checkpoints)
    print("="*80)
    print("PHASE 1: Training SAEs with Checkpoints")
    print("="*80)

    for seed in SEEDS:
        checkpoint_dir = checkpoint_dirs[seed]

        # Check if training already done
        final_checkpoint = checkpoint_dir / "sae_final.pt"
        if final_checkpoint.exists():
            print(f"\n✓ Seed {seed}: Checkpoints already exist, skipping training")
            continue

        # Train from scratch
        print(f"\n{'='*80}")
        print(f"Training seed {seed}")
        print(f"{'='*80}")
        train_sae_with_checkpoints(
            activations=activations,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            verbose=True
        )

    print("\n" + "="*80)
    print("PHASE 2: Computing PWMCC Evolution")
    print("="*80)

    # Get checkpoint epochs
    checkpoint_epochs = list(range(CHECKPOINT_FREQ, NUM_EPOCHS + 1, CHECKPOINT_FREQ))
    if NUM_EPOCHS not in checkpoint_epochs:
        checkpoint_epochs.append(NUM_EPOCHS)

    print(f"\nAnalyzing epochs: {checkpoint_epochs}")

    # Compute PWMCC evolution
    pwmcc_results = compute_pwmcc_evolution(checkpoint_dirs, checkpoint_epochs)

    # Analyze dynamics
    print("\n" + "="*80)
    print("PHASE 3: Analysis and Interpretation")
    print("="*80)

    analysis = analyze_dynamics(pwmcc_results)

    print("\nKEY FINDINGS:")
    print("-" * 80)
    print(f"Initial PWMCC (epoch {checkpoint_epochs[0]}): {analysis['initial_pwmcc']:.4f}")
    print(f"Final PWMCC (epoch {NUM_EPOCHS}): {analysis['final_pwmcc']:.4f}")
    print(f"Change: {analysis['delta_pwmcc']:+.4f}")
    print(f"Max PWMCC: {analysis['max_pwmcc']:.4f}")
    print(f"Min PWMCC: {analysis['min_pwmcc']:.4f}")
    print(f"Divergence point: Epoch {analysis['divergence_epoch']} (max change: {analysis['max_change']:.4f})")
    print(f"Late training variance: {analysis['late_training_variance']:.6f}")
    print(f"Stable: {analysis['is_stable']}")
    print()
    print(f"INTERPRETATION: {analysis['interpretation']}")
    print("-" * 80)

    # Save results
    full_results = {
        'config': {
            'seeds': SEEDS,
            'layer': LAYER,
            'architecture': 'topk',
            'k': K,
            'expansion_factor': EXPANSION_FACTOR,
            'num_epochs': NUM_EPOCHS,
            'checkpoint_freq': CHECKPOINT_FREQ,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'pwmcc_evolution': pwmcc_results,
        'analysis': analysis
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Results saved: {OUTPUT_JSON}")

    # Visualize
    print("\n" + "="*80)
    print("PHASE 4: Visualization")
    print("="*80)

    visualize_pwmcc_dynamics(pwmcc_results, OUTPUT_FIGURE)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print()
    print("ANSWER TO KEY QUESTION:")
    print("-" * 80)

    if abs(analysis['delta_pwmcc']) < 0.05:
        print("Features appear STABLE throughout training - PWMCC does not change significantly.")
        print("This suggests the SAE is finding a consistent solution regardless of initialization.")
        print()
        print("Implication: The underconstrained hypothesis may NOT explain the low PWMCC.")
        print("The problem may be that there are genuinely MANY valid feature sets.")
    elif analysis['delta_pwmcc'] < 0:
        print("Features START aligned but DIVERGE during training.")
        print("This supports the UNDERCONSTRAINED hypothesis - training is free to explore")
        print("different solutions that all reconstruct equally well.")
        print()
        print(f"Critical divergence occurs around epoch {analysis['divergence_epoch']}.")
    else:
        print("Features START misaligned but CONVERGE during training.")
        print("This suggests the SAE is finding a unique attractor solution.")
        print("The low final PWMCC may be due to slow convergence, not multiple solutions.")

    print("-" * 80)
    print()
    print(f"Figure location: {OUTPUT_FIGURE}")
    print(f"Results location: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
