#!/usr/bin/env python3
"""Task Generalization Experiment: Test if PWMCC ≈ 0.30 baseline generalizes.

This script tests whether the finding that trained SAEs perform similar to random
baselines (PWMCC ≈ 0.30) is task-specific or universal. We train a transformer
on a different task (sequence copying), extract activations, train multiple SAEs,
and compare PWMCC to the modular arithmetic results.

Critical Question:
    Does PWMCC ≈ 0.30 replicate across tasks, or is it specific to modular arithmetic?

Experiment Design:
    1. Train transformer on sequence copying task
    2. Extract activations from middle layer
    3. Train 5 TopK SAEs with different seeds
    4. Compute PWMCC matrix
    5. Compare to modular arithmetic PWMCC (0.309)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/task_generalization.py \
        --task copy \
        --vocab-size 20 \
        --seq-len 10 \
        --epochs 500 \
        --sae-epochs 20 \
        --seeds 42 123 456 789 1011

Expected Output:
    - New task PWMCC: X.XXX ± X.XXX
    - Modular arithmetic PWMCC: 0.309 ± 0.023
    - Generalization: [YES/NO] (whether PWMCC values are similar)
    - Figure: figures/task_comparison_pwmcc.pdf

References:
    - Paulo & Belrose (2025): "Do SAEs Converge to Stable Features?"
    - Critical finding from CRITICAL_FINDINGS.md: Random baseline = trained SAE
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.models.simple_sae import TopKSAE
from src.analysis.feature_matching import compute_feature_overlap_matrix, compute_feature_statistics
from src.utils.config import TransformerConfig


class SequenceCopyDataset:
    """Generate sequence copying dataset.

    Task: Given input sequence [a, b, c, SEP], predict [a, b, c]

    This is a simple memorization task that tests if the transformer
    can learn to copy sequences through its residual stream.

    Example:
        Input:  [5, 3, 8, SEP, ?, ?, ?]
        Target: [?, ?, ?, ?,   5, 3, 8]

    Args:
        vocab_size: Number of unique tokens (excluding special tokens)
        seq_len: Length of sequence to copy
        n_samples: Number of training examples
    """

    def __init__(self, vocab_size: int = 20, seq_len: int = 10, n_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        # Special tokens
        self.SEP = vocab_size  # Separator token
        self.PAD = vocab_size + 1  # Padding token
        self.total_vocab = vocab_size + 2

        # Generate all examples
        self.inputs, self.targets = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all training examples.

        Returns:
            inputs: [n_samples, 2*seq_len + 1] token sequences
            targets: [n_samples, 2*seq_len + 1] target sequences
        """
        inputs = []
        targets = []

        for _ in range(self.n_samples):
            # Generate random sequence
            sequence = torch.randint(0, self.vocab_size, (self.seq_len,))

            # Create input: [seq, SEP, PAD, PAD, ...]
            input_seq = torch.cat([
                sequence,
                torch.tensor([self.SEP]),
                torch.full((self.seq_len,), self.PAD)
            ])

            # Create target: [PAD, ..., PAD, SEP, seq]
            target_seq = torch.cat([
                torch.full((self.seq_len,), self.PAD),
                torch.tensor([self.SEP]),
                sequence
            ])

            inputs.append(input_seq)
            targets.append(target_seq)

        return torch.stack(inputs), torch.stack(targets)

    def create_dataloaders(self, train_fraction: float = 0.9, batch_size: int = 256):
        """Create train and validation dataloaders.

        Args:
            train_fraction: Fraction of data for training
            batch_size: Batch size

        Returns:
            train_loader, val_loader
        """
        n_train = int(self.n_samples * train_fraction)

        train_dataset = TensorDataset(self.inputs[:n_train], self.targets[:n_train])
        val_dataset = TensorDataset(self.inputs[n_train:], self.targets[n_train:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


def train_transformer_copy_task(
    vocab_size: int = 20,
    seq_len: int = 10,
    n_samples: int = 10000,
    epochs: int = 500,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> ModularArithmeticTransformer:
    """Train transformer on sequence copying task.

    Args:
        vocab_size: Vocabulary size (excluding special tokens)
        seq_len: Sequence length
        n_samples: Number of training examples
        epochs: Training epochs
        d_model: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on

    Returns:
        model: Trained transformer
    """
    print("="*60)
    print("TRAINING TRANSFORMER ON SEQUENCE COPYING TASK")
    print("="*60)

    # Create dataset
    dataset = SequenceCopyDataset(vocab_size, seq_len, n_samples)
    train_loader, val_loader = dataset.create_dataloaders(batch_size=batch_size)

    print(f"\nDataset:")
    print(f"  Vocab size: {vocab_size} (+ 2 special tokens = {dataset.total_vocab})")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Total sequence length: {2 * seq_len + 1}")

    # Create model config
    config = TransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=4 * d_model,
        vocab_size=dataset.total_vocab,
        max_seq_len=2 * seq_len + 1,
        activation='gelu'
    )

    print(f"\nModel config:")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_mlp: {config.d_mlp}")

    # Create model
    model = ModularArithmeticTransformer(config, device=device)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            # Compute loss only on output positions (after SEP)
            # Positions: [0..seq_len-1] = input, [seq_len] = SEP, [seq_len+1..end] = output
            loss = F.cross_entropy(
                logits[:, seq_len+1:, :].reshape(-1, dataset.total_vocab),
                targets[:, seq_len+1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy on output positions
            predictions = logits[:, seq_len+1:, :].argmax(dim=-1)
            train_correct += (predictions == targets[:, seq_len+1:]).sum().item()
            train_total += seq_len * len(inputs)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)

                loss = F.cross_entropy(
                    logits[:, seq_len+1:, :].reshape(-1, dataset.total_vocab),
                    targets[:, seq_len+1:].reshape(-1)
                )

                val_loss += loss.item()

                predictions = logits[:, seq_len+1:, :].argmax(dim=-1)
                val_correct += (predictions == targets[:, seq_len+1:]).sum().item()
                val_total += seq_len * len(inputs)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
                  f"Best={best_val_acc:.4f}")

    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")

    if best_val_acc > 0.99:
        print("   🎉 Model achieved near-perfect copying!")
    elif best_val_acc > 0.9:
        print("   ✅ Model learned the task well")
    else:
        print("   ⚠️  Model performance is suboptimal - consider training longer")

    return model


def extract_activations(
    model: ModularArithmeticTransformer,
    vocab_size: int = 20,
    seq_len: int = 10,
    layer: int = 1,
    position: int = -1,
    max_samples: int = 50000,
    batch_size: int = 256,
    device: str = 'cpu'
) -> torch.Tensor:
    """Extract activations from trained transformer.

    Args:
        model: Trained transformer
        vocab_size: Vocabulary size
        seq_len: Sequence length
        layer: Layer to extract from
        position: Position to extract (-1 = last, 0 = first, etc.)
        max_samples: Maximum number of samples
        batch_size: Batch size
        device: Device

    Returns:
        activations: [n_samples, d_model]
    """
    print("\n" + "="*60)
    print("EXTRACTING ACTIVATIONS")
    print("="*60)

    # Create dataset for extraction
    dataset = SequenceCopyDataset(vocab_size, seq_len, n_samples=max_samples)
    loader = DataLoader(
        TensorDataset(dataset.inputs, dataset.targets),
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Layer: {layer}")
    print(f"Position: {position}")
    print(f"Max samples: {max_samples}")

    activations = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Extracting"):
            inputs = inputs.to(device)

            # Get activations from specified layer
            act = model.get_activations(inputs, layer=layer, activation_name="resid_post")

            # Extract from specified position
            if position == -1:
                act = act[:, -1, :]
            else:
                act = act[:, position, :]

            activations.append(act.cpu())

    activations = torch.cat(activations, dim=0)[:max_samples]

    print(f"✅ Extracted {activations.shape[0]} activations: {activations.shape}")
    print(f"   Mean: {activations.mean():.4f}, Std: {activations.std():.4f}")

    return activations


def train_sae(
    sae: TopKSAE,
    activations: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train single SAE on activations.

    Args:
        sae: SAE model
        activations: Training data
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device

    Returns:
        metrics: Final training metrics
    """
    sae.to(device)
    activations = activations.to(device)

    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        sae.train()
        sae.reset_feature_counts()

        epoch_loss = 0
        epoch_l0 = 0
        n_batches = 0

        for (batch,) in dataloader:
            optimizer.zero_grad()

            # Forward pass
            reconstruction, latents, aux_loss = sae(batch)

            # MSE loss
            mse_loss = F.mse_loss(reconstruction, batch)

            # Total loss
            loss = mse_loss + aux_loss

            # Backward
            loss.backward()
            optimizer.step()

            # Normalize decoder
            sae.normalize_decoder()

            # Metrics
            l0 = sae.get_l0(latents)
            epoch_loss += loss.item()
            epoch_l0 += l0
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_l0 = epoch_l0 / n_batches

    # Compute final metrics
    sae.eval()
    with torch.no_grad():
        all_recon = []
        for (batch,) in DataLoader(dataset, batch_size=1024):
            recon, _, _ = sae(batch)
            all_recon.append(recon)
        all_recon = torch.cat(all_recon, dim=0)

        data_var = activations.var()
        error_var = (activations - all_recon).var()
        explained_var = 1 - (error_var / data_var)
        explained_var = explained_var.item()

    dead_neurons = sae.get_dead_neurons()
    dead_pct = len(dead_neurons) / sae.d_sae * 100

    return {
        'loss': avg_loss,
        'l0': avg_l0,
        'explained_variance': explained_var,
        'dead_neuron_pct': dead_pct
    }


def train_multiple_saes(
    activations: torch.Tensor,
    seeds: List[int],
    d_model: int = 128,
    d_sae: int = 1024,
    k: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cpu',
    save_dir: Path = None
) -> List:
    """Train multiple SAEs with different seeds.

    Args:
        activations: Training data
        seeds: List of random seeds
        d_model: Model dimension
        d_sae: SAE dimension
        k: TopK k parameter
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        save_dir: Directory to save SAEs

    Returns:
        saes: List of trained SAE wrappers
    """
    print("\n" + "="*60)
    print("TRAINING MULTIPLE SAEs")
    print("="*60)

    class SimpleWrapper:
        """Wrapper to match feature_matching interface."""
        def __init__(self, sae):
            self.sae = sae

    saes = []

    for seed in seeds:
        print(f"\n🌱 Training SAE with seed={seed}")
        print("-" * 40)

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create SAE
        sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

        # Train
        metrics = train_sae(
            sae,
            activations,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )

        print(f"   Loss: {metrics['loss']:.4f}")
        print(f"   L0: {metrics['l0']:.2f}")
        print(f"   Explained variance: {metrics['explained_variance']:.4f}")
        print(f"   Dead neurons: {metrics['dead_neuron_pct']:.1f}%")

        # Save if requested
        if save_dir:
            seed_dir = save_dir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            sae.save(seed_dir / "sae_final.pt")
            print(f"   💾 Saved to {seed_dir}")

        # Wrap and append
        saes.append(SimpleWrapper(sae))

    print("\n✅ All SAEs trained!")
    return saes


def create_comparison_figure(
    new_task_pwmcc: float,
    new_task_std: float,
    new_task_matrix: np.ndarray,
    modular_pwmcc: float = 0.309,
    modular_std: float = 0.023,
    save_path: Path = None
) -> plt.Figure:
    """Create comparison figure between tasks.

    Args:
        new_task_pwmcc: Mean PWMCC for new task
        new_task_std: Std PWMCC for new task
        new_task_matrix: PWMCC matrix for new task
        modular_pwmcc: Mean PWMCC for modular arithmetic (default: 0.309)
        modular_std: Std PWMCC for modular arithmetic (default: 0.023)
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Bar comparison
    ax = axes[0]
    tasks = ['Modular\nArithmetic', 'Sequence\nCopying']
    means = [modular_pwmcc, new_task_pwmcc]
    stds = [modular_std, new_task_std]

    x = np.arange(len(tasks))
    bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e'])

    # Add baseline and threshold lines
    ax.axhline(0.30, color='red', linestyle='--', linewidth=2,
               label='Paulo & Belrose baseline (0.30)', alpha=0.7)
    ax.axhline(0.70, color='green', linestyle='--', linewidth=2,
               label='Stability threshold (0.70)', alpha=0.7)

    ax.set_ylabel('Mean PWMCC', fontsize=12, fontweight='bold')
    ax.set_title('Task Comparison: PWMCC Stability', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.03,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. PWMCC matrix heatmap for new task
    ax = axes[1]
    n = len(new_task_matrix)
    labels = [f"Seed{i+1}" for i in range(n)]

    im = ax.imshow(new_task_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Sequence Copying PWMCC Matrix', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PWMCC', fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{new_task_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white" if new_task_matrix[i, j] < 0.5 else "black",
                          fontsize=9)

    # 3. Distribution histogram
    ax = axes[2]

    # Get off-diagonal values
    triu_indices = np.triu_indices(n, k=1)
    off_diagonal = new_task_matrix[triu_indices]

    ax.hist(off_diagonal, bins=15, alpha=0.7, color='#ff7f0e', edgecolor='black')
    ax.axvline(new_task_pwmcc, color='orange', linestyle='-', linewidth=2.5,
               label=f'Seq Copy: {new_task_pwmcc:.3f}')
    ax.axvline(modular_pwmcc, color='blue', linestyle='-', linewidth=2.5,
               label=f'Modular: {modular_pwmcc:.3f}')
    ax.axvline(0.30, color='red', linestyle='--', linewidth=2,
               label='Baseline (0.30)', alpha=0.7)
    ax.axvline(0.70, color='green', linestyle='--', linewidth=2,
               label='Threshold (0.70)', alpha=0.7)

    ax.set_xlabel('PWMCC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('PWMCC Distribution (Sequence Copying)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved comparison figure to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Task Generalization: Test if PWMCC ≈ 0.30 generalizes beyond modular arithmetic"
    )

    # Task args
    parser.add_argument('--task', type=str, default='copy', choices=['copy'],
                       help='Task type (currently only "copy" supported)')
    parser.add_argument('--vocab-size', type=int, default=20,
                       help='Vocabulary size for copy task')
    parser.add_argument('--seq-len', type=int, default=10,
                       help='Sequence length for copy task')

    # Transformer training args
    parser.add_argument('--epochs', type=int, default=500,
                       help='Transformer training epochs')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Transformer d_model')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Transformer layers')
    parser.add_argument('--n-heads', type=int, default=4,
                       help='Transformer attention heads')

    # SAE training args
    parser.add_argument('--sae-epochs', type=int, default=20,
                       help='SAE training epochs')
    parser.add_argument('--expansion', type=int, default=8,
                       help='SAE expansion factor')
    parser.add_argument('--k', type=int, default=32,
                       help='TopK k parameter')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1011],
                       help='Random seeds for SAE training')

    # Comparison args
    parser.add_argument('--modular-pwmcc', type=float, default=0.309,
                       help='Reference PWMCC from modular arithmetic')
    parser.add_argument('--modular-std', type=float, default=0.023,
                       help='Reference std from modular arithmetic')

    # Output args
    parser.add_argument('--save-dir', type=Path, default=Path('results/saes/copy_task'),
                       help='Directory to save SAEs')
    parser.add_argument('--figure-path', type=Path,
                       default=Path('figures/task_comparison_pwmcc.pdf'),
                       help='Path to save comparison figure')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*60)
    print("TASK GENERALIZATION EXPERIMENT")
    print("="*60)
    print(f"\nCritical Question:")
    print(f"  Does PWMCC ≈ 0.30 generalize beyond modular arithmetic?")
    print(f"\nReference (Modular Arithmetic):")
    print(f"  PWMCC: {args.modular_pwmcc:.3f} ± {args.modular_std:.3f}")
    print(f"\nNew Task: {args.task}")
    print(f"  Seeds: {args.seeds}")
    print("="*60)

    # Step 1: Train transformer
    print("\n📝 STEP 1: Training Transformer")
    model = train_transformer_copy_task(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        device=args.device
    )

    # Step 2: Extract activations
    print("\n📝 STEP 2: Extracting Activations")
    activations = extract_activations(
        model,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        layer=args.n_layers - 1,  # Middle layer
        device=args.device
    )

    # Step 3: Train multiple SAEs
    print("\n📝 STEP 3: Training Multiple SAEs")
    d_sae = args.expansion * args.d_model
    saes = train_multiple_saes(
        activations,
        seeds=args.seeds,
        d_model=args.d_model,
        d_sae=d_sae,
        k=args.k,
        epochs=args.sae_epochs,
        device=args.device,
        save_dir=args.save_dir
    )

    # Step 4: Compute PWMCC
    print("\n📝 STEP 4: Computing PWMCC Matrix")
    print("="*60)
    overlap_matrix = compute_feature_overlap_matrix(saes)
    stats = compute_feature_statistics(saes, overlap_matrix)

    new_task_pwmcc = stats['mean_overlap']
    new_task_std = stats['std_overlap']

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n📊 Sequence Copying Task:")
    print(f"   PWMCC: {new_task_pwmcc:.3f} ± {new_task_std:.3f}")
    print(f"   Range: [{stats['min_overlap']:.3f}, {stats['max_overlap']:.3f}]")
    print(f"   Median: {stats['median_overlap']:.3f}")

    print(f"\n📊 Modular Arithmetic Task (Reference):")
    print(f"   PWMCC: {args.modular_pwmcc:.3f} ± {args.modular_std:.3f}")

    # Step 5: Compare
    print("\n" + "="*60)
    print("GENERALIZATION ANALYSIS")
    print("="*60)

    diff = abs(new_task_pwmcc - args.modular_pwmcc)
    combined_std = np.sqrt(new_task_std**2 + args.modular_std**2)

    print(f"\nDifference: {diff:.3f}")
    print(f"Combined std: {combined_std:.3f}")
    print(f"Z-score: {diff/combined_std:.2f}")

    # Determine if it generalizes (within 1 std)
    generalizes = diff < combined_std

    if generalizes:
        print("\n✅ GENERALIZATION: YES")
        print(f"   The PWMCC ≈ 0.30 baseline replicates across tasks!")
        print(f"   This suggests the finding is UNIVERSAL, not task-specific.")
        print(f"   SAEs show similar instability regardless of the task.")
    else:
        print("\n❌ GENERALIZATION: NO")
        print(f"   The PWMCC differs significantly between tasks.")
        print(f"   This suggests the finding may be TASK-SPECIFIC.")
        print(f"   Different tasks may have different SAE stability properties.")

    # Both low?
    both_low = (new_task_pwmcc < 0.4) and (args.modular_pwmcc < 0.4)
    if both_low:
        print("\n🔴 CRITICAL: Both tasks show LOW PWMCC (<0.4)")
        print(f"   This indicates a REPRODUCIBILITY CRISIS in SAE training!")
        print(f"   Different seeds learn fundamentally different features.")

    # Step 6: Create figure
    print("\n📝 STEP 5: Creating Comparison Figure")
    create_comparison_figure(
        new_task_pwmcc=new_task_pwmcc,
        new_task_std=new_task_std,
        new_task_matrix=overlap_matrix,
        modular_pwmcc=args.modular_pwmcc,
        modular_std=args.modular_std,
        save_path=args.figure_path
    )

    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\n📊 Key Findings:")
    print(f"   1. Sequence Copying PWMCC: {new_task_pwmcc:.3f} ± {new_task_std:.3f}")
    print(f"   2. Modular Arithmetic PWMCC: {args.modular_pwmcc:.3f} ± {args.modular_std:.3f}")
    print(f"   3. Generalization: {'YES' if generalizes else 'NO'}")
    print(f"   4. Both low (<0.4): {'YES - REPRODUCIBILITY CRISIS' if both_low else 'NO'}")

    print(f"\n💾 Outputs:")
    print(f"   SAEs: {args.save_dir}")
    print(f"   Figure: {args.figure_path}")

    print("\n" + "="*60)

    return {
        'new_task_pwmcc': new_task_pwmcc,
        'new_task_std': new_task_std,
        'modular_pwmcc': args.modular_pwmcc,
        'modular_std': args.modular_std,
        'generalizes': generalizes,
        'both_low': both_low,
        'overlap_matrix': overlap_matrix,
        'stats': stats
    }


if __name__ == "__main__":
    main()
