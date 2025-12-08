#!/usr/bin/env python3
"""Sparse Ground Truth Validation - Extension 1

This experiment validates Cui et al. (2025)'s identifiability theory by testing
whether SAEs achieve high stability (PWMCC > 0.70) when the ground truth IS sparse.

Background:
-----------
Our main finding: SAE PWMCC = 0.309 (matches random baseline 0.300)
Identifiability theory (Cui et al.): Predicts PWMCC ≈ 0.30 when ground truth is DENSE
Our setup: Effective rank = 80/128 = 62.5% (VIOLATES Condition 1: extreme sparsity)

Hypothesis:
-----------
If we train on activations with SPARSE ground truth → PWMCC should be > 0.70

Experimental Design:
-------------------
1. Train 1-layer transformer on modular arithmetic
   - Known to learn Fourier circuits (Nanda et al., 2023)
   - Ground truth: ~10-20 key frequencies (SPARSE)
   - Target R² > 0.90 (validate Fourier learning)

2. Train SAEs with matched configuration
   - d_sae = 256 (enough for ~20-30 Fourier features)
   - k = 16 (matched to sparse Fourier structure)
   - 5 seeds: 42, 123, 456, 789, 1011

3. Measure PWMCC
   - Prediction: PWMCC > 0.70 (vs dense setup: 0.309)
   - This would definitively validate identifiability theory

Expected Outcome:
----------------
| Setup | Ground Truth | Theory Predicts | Empirical | Status |
|-------|--------------|-----------------|-----------|--------|
| 2-layer (ours) | Dense (eff_rank=80) | PWMCC ≈ 0.30 | 0.309 | ✅ Validated |
| 1-layer | Sparse (~15 freqs) | PWMCC > 0.70 | ??? | **Testing** |

Usage:
------
python scripts/sparse_ground_truth_experiment.py \
    --output-dir results/sparse_ground_truth \
    --n-epochs 5000 \
    --n-sae-seeds 5 \
    --device cuda

References:
-----------
- Cui et al. (2025): Identifiability theory [arXiv:2506.15963]
- Nanda et al. (2023): Fourier circuits in 1-layer transformers
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_sae import TopKSAE


# ==============================================================================
# 1-Layer Transformer for Modular Arithmetic
# ==============================================================================

class SimpleTransformer(nn.Module):
    """Minimal 1-layer transformer for modular arithmetic.

    Simplified architecture to encourage Fourier circuit learning:
    - 1 attention layer (not 2)
    - No LayerNorm (simpler optimization landscape)
    - Small d_model to match Nanda et al. setup

    This architecture is known to learn Fourier circuits that grok.
    """

    def __init__(
        self,
        vocab_size: int = 117,  # 113 numbers + 4 special tokens
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Positional embedding
        self.pos_embed = nn.Embedding(7, d_model)  # max_len = 7

        # Single attention layer
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            batch_first=True
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Output projection
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input token IDs [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
            activations: Dict of intermediate activations for SAE training
        """
        batch_size, seq_len = x.shape

        # Embed tokens
        token_embed = self.embed(x)  # [batch, seq_len, d_model]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embed = self.pos_embed(positions)

        h = token_embed + pos_embed  # [batch, seq_len, d_model]

        # Attention
        attn_out, _ = self.attn(h, h, h)
        h_after_attn = h + attn_out

        # MLP
        mlp_out = self.mlp(h_after_attn)
        h_after_mlp = h_after_attn + mlp_out

        # Unembed
        logits = self.unembed(h_after_mlp)

        # Store activations for SAE training
        activations = {
            'embed': token_embed + pos_embed,
            'after_attn': h_after_attn,
            'after_mlp': h_after_mlp,
        }

        return logits, activations


def generate_modular_arithmetic_data(
    modulus: int = 113,
    n_samples: int = 10000,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate modular arithmetic dataset: a + b = c (mod p).

    Args:
        modulus: Prime modulus
        n_samples: Number of samples
        device: Device

    Returns:
        inputs: [n_samples, 6] - [BOS, a, +, b, =, answer]
        targets: [n_samples, 6] - shifted for next-token prediction
    """
    # Special tokens
    BOS = modulus + 2
    PLUS = modulus
    EQUALS = modulus + 1

    inputs = []
    targets = []

    for _ in range(n_samples):
        a = np.random.randint(0, modulus)
        b = np.random.randint(0, modulus)
        c = (a + b) % modulus

        # Input: [BOS, a, +, b, =, answer]
        seq = [BOS, a, PLUS, b, EQUALS, c]

        # Target: predict next token (shift by 1)
        # We only care about predicting 'answer' at position -1
        target = [a, PLUS, b, EQUALS, c, c]  # last token doesn't matter

        inputs.append(seq)
        targets.append(target)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def train_transformer(
    model: SimpleTransformer,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    n_epochs: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = 'cpu',
    save_dir: Path = None
) -> Dict:
    """Train 1-layer transformer on modular arithmetic.

    Args:
        model: SimpleTransformer
        train_inputs: Training inputs
        train_targets: Training targets
        n_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        save_dir: Directory to save checkpoints

    Returns:
        training_stats: Dict of training statistics
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Create dataloader
    dataset = TensorDataset(train_inputs, train_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    training_stats = {
        'losses': [],
        'accuracies': [],
        'epochs': []
    }

    print(f"\nTraining 1-layer transformer for {n_epochs} epochs...")
    print(f"Dataset size: {len(train_inputs)}, Batch size: {batch_size}")

    best_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for inputs, targets in dataloader:
            optimizer.zero_grad()

            logits, _ = model(inputs)

            # Compute loss only on answer position (-1)
            loss = F.cross_entropy(
                logits[:, -1, :],  # [batch, vocab_size]
                targets[:, -1]     # [batch]
            )

            loss.backward()
            optimizer.step()

            # Track stats
            epoch_loss += loss.item()

            # Accuracy
            preds = logits[:, -1, :].argmax(dim=-1)
            epoch_correct += (preds == targets[:, -1]).sum().item()
            epoch_total += inputs.size(0)

        # Epoch stats
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / epoch_total

        training_stats['losses'].append(avg_loss)
        training_stats['accuracies'].append(accuracy)
        training_stats['epochs'].append(epoch)

        # Track best model
        if accuracy > best_acc:
            best_acc = accuracy
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': avg_loss,
                }, save_dir / 'transformer_best.pt')

        # Print progress
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Best: {best_acc:.4f}")

        # Save periodic checkpoints
        if save_dir and epoch % 1000 == 0 and epoch > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
                'loss': avg_loss,
            }, save_dir / f'transformer_epoch_{epoch}.pt')

    # Save final model
    if save_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': n_epochs,
            'accuracy': accuracy,
            'loss': avg_loss,
            'training_stats': training_stats,
        }, save_dir / 'transformer_final.pt')

    return training_stats


def validate_fourier_structure(
    model: SimpleTransformer,
    modulus: int = 113,
    device: str = 'cpu'
) -> Dict:
    """Validate that transformer learned Fourier circuits.

    Uses Nanda et al.'s method: DFT on embedding matrix, compute R².

    Args:
        model: Trained transformer
        modulus: Modulus
        device: Device

    Returns:
        results: Dict with R² and key frequencies
    """
    print("\n" + "="*80)
    print("VALIDATING FOURIER STRUCTURE (Nanda et al. method)")
    print("="*80)

    # Extract embedding matrix for numerical tokens
    W_E = model.embed.weight.data[:modulus, :]  # [modulus, d_model]

    # Apply DFT along vocab dimension
    W_E_fourier = torch.fft.fft(W_E, dim=0)  # [modulus, d_model] complex
    freq_magnitudes = torch.abs(W_E_fourier).mean(dim=1)  # [modulus]

    # Find top frequencies
    top_k = 10
    top_freqs = torch.argsort(freq_magnitudes, descending=True)[:top_k].tolist()

    print(f"\nTop {top_k} frequencies (by magnitude):")
    for i, freq_idx in enumerate(top_freqs):
        print(f"  {i+1}. k={freq_idx:3d}: magnitude = {freq_magnitudes[freq_idx]:.4f}")

    # Compute R² for top-2 frequencies (typical for modular arithmetic)
    def get_fourier_basis(modulus, freqs):
        """Get Fourier basis for given frequencies."""
        components = []
        for k in freqs:
            indices = torch.arange(modulus, dtype=torch.float32)
            angles = 2 * torch.pi * k * indices / modulus
            components.append(torch.cos(angles).unsqueeze(1))
            components.append(torch.sin(angles).unsqueeze(1))
        return torch.cat(components, dim=1)  # [modulus, 2*len(freqs)]

    # Compute R² for different numbers of frequencies
    r_squared_results = {}

    for n_freqs in [2, 5, 10]:
        if n_freqs > len(top_freqs):
            continue

        basis = get_fourier_basis(modulus, top_freqs[:n_freqs]).to(device)

        # Center W_E
        W_E_centered = W_E - W_E.mean(dim=0)

        # Project onto Fourier basis
        projection = basis @ (basis.T @ W_E_centered)

        # Compute R²
        ss_total = (W_E_centered ** 2).sum()
        ss_residual = ((W_E_centered - projection) ** 2).sum()
        r_squared = 1 - (ss_residual / ss_total)

        r_squared_results[f'r2_top{n_freqs}'] = r_squared.item()

        print(f"\nR² (top {n_freqs} frequencies): {r_squared.item():.4f} ({r_squared.item()*100:.2f}%)")

        if r_squared > 0.9:
            print(f"  ✅ EXCELLENT - Strong sparse Fourier structure!")
        elif r_squared > 0.6:
            print(f"  ⚠️  MODERATE - Partial Fourier structure")
        else:
            print(f"  ❌ WEAK - Not suitable for sparse ground truth validation")

    # Compare to Nanda et al.
    primary_r2 = r_squared_results.get('r2_top2', r_squared_results.get('r2_top5', 0))

    print(f"\n" + "-"*80)
    print(f"Nanda et al. (2023): R² = 0.93 - 0.98 (grokked models)")
    print(f"Our result:          R² = {primary_r2:.4f}")
    print(f"Difference:          {(primary_r2 - 0.95)*100:+.2f} percentage points")

    if primary_r2 > 0.9:
        print(f"\n✅ VALIDATION SUCCESSFUL - Transformer learned sparse Fourier circuits!")
        print(f"   Suitable for testing SAE stability with sparse ground truth")
        validation_status = "success"
    elif primary_r2 > 0.6:
        print(f"\n⚠️  PARTIAL SUCCESS - Some Fourier structure, but not ideal")
        validation_status = "partial"
    else:
        print(f"\n❌ VALIDATION FAILED - Transformer did not learn Fourier circuits")
        print(f"   Cannot use for sparse ground truth validation")
        print(f"   Consider: longer training, different hyperparameters, or synthetic data")
        validation_status = "failed"

    return {
        'top_frequencies': top_freqs,
        'frequency_magnitudes': freq_magnitudes.tolist(),
        'r_squared': r_squared_results,
        'primary_r2': primary_r2,
        'validation_status': validation_status,
    }


def extract_activations(
    model: SimpleTransformer,
    inputs: torch.Tensor,
    layer: str = 'after_mlp',
    position: int = -2,
    device: str = 'cpu'
) -> torch.Tensor:
    """Extract activations from transformer for SAE training.

    Args:
        model: Trained transformer
        inputs: Input sequences [n_samples, seq_len]
        layer: Which layer to extract ('embed', 'after_attn', 'after_mlp')
        position: Which position to extract (-2 = answer position)
        device: Device

    Returns:
        activations: [n_samples, d_model]
    """
    model.eval()
    all_acts = []

    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size].to(device)
            _, acts_dict = model(batch)

            # Extract at specified position
            acts = acts_dict[layer][:, position, :]  # [batch, d_model]
            all_acts.append(acts.cpu())

    return torch.cat(all_acts, dim=0)


def train_sae(
    activations: torch.Tensor,
    d_sae: int = 256,
    k: int = 16,
    n_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = 'cpu',
    seed: int = 42,
) -> TopKSAE:
    """Train a single SAE on activations.

    Args:
        activations: Input activations [n_samples, d_model]
        d_sae: SAE hidden dimension
        k: TopK sparsity
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        seed: Random seed

    Returns:
        sae: Trained SAE
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_model = activations.size(1)
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstruction, latents, aux_loss = sae(batch)

            # Total loss
            mse_loss = F.mse_loss(reconstruction, batch)
            loss = mse_loss + aux_loss

            loss.backward()
            optimizer.step()

            # Normalize decoder
            sae.normalize_decoder()

            epoch_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    return sae


def compute_pwmcc(
    sae1: TopKSAE,
    sae2: TopKSAE
) -> float:
    """Compute Pairwise Maximum Cosine Correlation (PWMCC) between two SAEs.

    Args:
        sae1: First SAE
        sae2: Second SAE

    Returns:
        pwmcc: PWMCC score [0, 1]
    """
    # Get decoder weights [d_model, d_sae]
    D1 = sae1.decoder.weight.data.T  # [d_model, d_sae]
    D2 = sae2.decoder.weight.data.T  # [d_model, d_sae]

    # Normalize
    D1 = F.normalize(D1, dim=0)
    D2 = F.normalize(D2, dim=0)

    # Cosine similarity matrix
    cos_sim = D1.T @ D2  # [d_sae1, d_sae2]

    # PWMCC: mean of max similarities
    max_sim1 = cos_sim.abs().max(dim=1)[0].mean()
    max_sim2 = cos_sim.abs().max(dim=0)[0].mean()

    pwmcc = (max_sim1 + max_sim2) / 2

    return pwmcc.item()


def main():
    parser = argparse.ArgumentParser(
        description="Sparse Ground Truth Validation - Extension 1"
    )
    parser.add_argument('--output-dir', type=Path,
                       default='results/sparse_ground_truth',
                       help='Output directory')
    parser.add_argument('--modulus', type=int, default=113,
                       help='Prime modulus')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--n-epochs', type=int, default=5000,
                       help='Transformer training epochs')
    parser.add_argument('--n-sae-seeds', type=int, default=5,
                       help='Number of SAE seeds to train')
    parser.add_argument('--d-sae', type=int, default=256,
                       help='SAE hidden dimension')
    parser.add_argument('--k', type=int, default=16,
                       help='TopK sparsity for SAE')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*80)
    print("SPARSE GROUND TRUTH VALIDATION - Extension 1")
    print("="*80)
    print(f"\nHypothesis: Sparse ground truth → High SAE stability (PWMCC > 0.70)")
    print(f"\nConfiguration:")
    print(f"  Modulus: {args.modulus}")
    print(f"  Training samples: {args.n_samples}")
    print(f"  Transformer epochs: {args.n_epochs}")
    print(f"  SAE seeds: {args.n_sae_seeds}")
    print(f"  SAE architecture: d_sae={args.d_sae}, k={args.k}")
    print(f"  Device: {args.device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: Train 1-layer Fourier Transformer
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 1: Training 1-Layer Fourier Transformer")
    print("="*80)

    # Generate data
    print("\nGenerating modular arithmetic data...")
    train_inputs, train_targets = generate_modular_arithmetic_data(
        modulus=args.modulus,
        n_samples=args.n_samples,
        device=args.device
    )
    print(f"✅ Generated {len(train_inputs)} training samples")

    # Create model
    model = SimpleTransformer(
        vocab_size=args.modulus + 4,  # numbers + special tokens
        d_model=128,
        n_heads=4,
        d_ff=512
    )

    # Train
    start_time = time.time()
    training_stats = train_transformer(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        n_epochs=args.n_epochs,
        batch_size=512,
        lr=1e-3,
        device=args.device,
        save_dir=args.output_dir / 'transformer'
    )
    train_time = time.time() - start_time

    print(f"\n✅ Training complete in {train_time/60:.2f} minutes")
    print(f"   Final accuracy: {training_stats['accuracies'][-1]:.4f}")

    # ==============================================================================
    # PHASE 2: Validate Fourier Structure
    # ==============================================================================

    fourier_results = validate_fourier_structure(
        model=model,
        modulus=args.modulus,
        device=args.device
    )

    # Save Fourier validation results
    with open(args.output_dir / 'fourier_validation.json', 'w') as f:
        json.dump(fourier_results, f, indent=2)

    # Check if validation passed
    if fourier_results['validation_status'] != 'success':
        print("\n" + "="*80)
        print("⚠️  WARNING: Transformer did not learn strong Fourier structure")
        print("="*80)
        print("\nOptions:")
        print("1. Train longer (increase --n-epochs)")
        print("2. Adjust hyperparameters (lr, architecture)")
        print("3. Use synthetic sparse data instead")
        print("\nAborting SAE training. Results saved for analysis.")
        return

    # ==============================================================================
    # PHASE 3: Extract Activations
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 3: Extracting Activations for SAE Training")
    print("="*80)

    print("\nExtracting activations from 'after_mlp' layer...")
    activations = extract_activations(
        model=model,
        inputs=train_inputs,
        layer='after_mlp',
        position=-2,  # Answer position
        device=args.device
    )

    print(f"✅ Extracted activations: {activations.shape}")

    # Compute effective rank
    U, S, V = torch.svd(activations.T)
    total_var = (S ** 2).sum()
    cumsum = (S ** 2).cumsum(0)
    effective_rank_90 = (cumsum < 0.9 * total_var).sum().item() + 1
    effective_rank_95 = (cumsum < 0.95 * total_var).sum().item() + 1

    print(f"\nActivation statistics:")
    print(f"  Effective rank (90% var): {effective_rank_90}/{activations.size(1)}")
    print(f"  Effective rank (95% var): {effective_rank_95}/{activations.size(1)}")
    print(f"  Sparsity: {effective_rank_90/activations.size(1)*100:.1f}%")

    # ==============================================================================
    # PHASE 4: Train SAEs (Multiple Seeds)
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 4: Training SAEs on Sparse Fourier Activations")
    print("="*80)

    seeds = [42, 123, 456, 789, 1011][:args.n_sae_seeds]
    saes = []

    for i, seed in enumerate(seeds):
        print(f"\nTraining SAE {i+1}/{len(seeds)} (seed={seed})...")
        sae = train_sae(
            activations=activations,
            d_sae=args.d_sae,
            k=args.k,
            n_epochs=100,
            batch_size=1024,
            lr=1e-3,
            device=args.device,
            seed=seed
        )

        # Save SAE
        sae_path = args.output_dir / f'sae_seed_{seed}.pt'
        sae.save(sae_path)
        print(f"  ✅ Saved to {sae_path}")

        saes.append(sae.cpu())

    # ==============================================================================
    # PHASE 5: Compute PWMCC (Pairwise across all seeds)
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 5: Computing PWMCC (Feature Stability)")
    print("="*80)

    print(f"\nComputing pairwise PWMCC for {len(saes)} SAEs...")

    pwmcc_matrix = np.zeros((len(saes), len(saes)))

    for i in range(len(saes)):
        for j in range(i+1, len(saes)):
            pwmcc = compute_pwmcc(saes[i], saes[j])
            pwmcc_matrix[i, j] = pwmcc
            pwmcc_matrix[j, i] = pwmcc
            print(f"  Seed {seeds[i]} vs {seeds[j]}: PWMCC = {pwmcc:.4f}")

    # Compute mean PWMCC (off-diagonal)
    off_diag = pwmcc_matrix[np.triu_indices(len(saes), k=1)]
    mean_pwmcc = off_diag.mean()
    std_pwmcc = off_diag.std()

    print(f"\n" + "-"*80)
    print(f"MEAN PWMCC: {mean_pwmcc:.4f} ± {std_pwmcc:.4f}")
    print(f"-"*80)

    # ==============================================================================
    # PHASE 6: Compare to Baseline and Validate Hypothesis
    # ==============================================================================

    print("\n" + "="*80)
    print("RESULTS: Sparse vs Dense Ground Truth Comparison")
    print("="*80)

    # Load dense ground truth result (from main experiments)
    dense_pwmcc = 0.309  # From our 2-layer transformer results
    random_baseline = 0.300

    print(f"\n| Setup | Ground Truth | Theory Prediction | Empirical PWMCC |")
    print(f"|-------|--------------|-------------------|-----------------|")
    print(f"| 2-layer (dense) | eff_rank=80/128 (62.5%) | PWMCC ≈ 0.30 | {dense_pwmcc:.3f} ✅ |")
    print(f"| 1-layer (sparse) | ~{len(fourier_results['top_frequencies'][:5])} Fourier freqs | PWMCC > 0.70 | {mean_pwmcc:.3f} ", end="")

    if mean_pwmcc > 0.70:
        print("✅ |")
        print(f"\n🎉 HYPOTHESIS CONFIRMED!")
        print(f"   Sparse ground truth → High stability ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
        print(f"   This DEFINITIVELY validates Cui et al.'s identifiability theory!")
        validation_result = "confirmed"
    elif mean_pwmcc > 0.50:
        print("⚠️ |")
        print(f"\n⚠️  PARTIAL VALIDATION")
        print(f"   PWMCC improved ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
        print(f"   But below theoretical prediction (>0.70)")
        print(f"   Possible reasons:")
        print(f"   - Fourier structure not perfectly sparse (R²={fourier_results['primary_r2']:.3f})")
        print(f"   - SAE hyperparameters not optimal")
        print(f"   - Effective rank still too high ({effective_rank_90}/128)")
        validation_result = "partial"
    else:
        print("❌ |")
        print(f"\n❌ HYPOTHESIS NOT CONFIRMED")
        print(f"   PWMCC did not improve significantly ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
        print(f"   Possible issues:")
        print(f"   - Fourier structure may not be sufficiently sparse")
        print(f"   - SAE training may have failed")
        print(f"   - Theory may need refinement")
        validation_result = "failed"

    # ==============================================================================
    # Save All Results
    # ==============================================================================

    results = {
        'experiment': 'sparse_ground_truth_validation',
        'hypothesis': 'sparse_ground_truth -> high_sae_stability',
        'configuration': {
            'modulus': args.modulus,
            'n_samples': args.n_samples,
            'n_epochs': args.n_epochs,
            'n_sae_seeds': args.n_sae_seeds,
            'd_sae': args.d_sae,
            'k': args.k,
        },
        'transformer': {
            'architecture': '1-layer',
            'training_time_minutes': train_time / 60,
            'final_accuracy': training_stats['accuracies'][-1],
        },
        'fourier_validation': fourier_results,
        'activations': {
            'shape': list(activations.shape),
            'effective_rank_90': effective_rank_90,
            'effective_rank_95': effective_rank_95,
            'sparsity_percent': effective_rank_90 / activations.size(1) * 100,
        },
        'pwmcc_results': {
            'seeds': seeds,
            'matrix': pwmcc_matrix.tolist(),
            'mean': mean_pwmcc,
            'std': std_pwmcc,
        },
        'comparison': {
            'dense_setup_pwmcc': dense_pwmcc,
            'sparse_setup_pwmcc': mean_pwmcc,
            'improvement': mean_pwmcc - dense_pwmcc,
            'random_baseline': random_baseline,
        },
        'validation_result': validation_result,
    }

    with open(args.output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ All results saved to {args.output_dir}/")
    print(f"\nFiles created:")
    print(f"  - transformer/ (checkpoints)")
    print(f"  - sae_seed_*.pt (trained SAEs)")
    print(f"  - fourier_validation.json")
    print(f"  - results.json (complete results)")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
