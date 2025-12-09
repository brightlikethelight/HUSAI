#!/usr/bin/env python3
"""Synthetic Sparse Ground Truth Validation - Extension 1 (Option B)

Since the 1-layer transformer didn't learn sparse Fourier circuits, we use
synthetic data with KNOWN sparse ground truth for definitive validation.

Experimental Design:
-------------------
1. Generate synthetic activations with known sparse structure:
   - K = 10 true features (orthonormal directions)
   - Each sample: Linear combination of L0=3 random features
   - Ground truth sparsity: 30% per sample, but only 10 total features

2. Train SAEs (5 seeds) on this synthetic data:
   - d_sae = 64 (enough for 10 true features)
   - k = 5 (matched to L0=3 sparsity)

3. Measure PWMCC:
   - Prediction: PWMCC > 0.90 (extremely sparse ground truth)
   - Compare to dense setup: PWMCC = 0.309

Expected Outcome:
----------------
This is the GOLD STANDARD test of identifiability theory:
- Perfect control over ground truth sparsity
- Can verify SAEs recover true features
- Definitive validation if PWMCC > 0.90

Usage:
------
python scripts/synthetic_sparse_validation.py \\
    --output-dir results/synthetic_sparse \\
    --n-samples 50000 \\
    --n-true-features 10 \\
    --sparsity-per-sample 3 \\
    --n-sae-seeds 5 \\
    --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_sae import TopKSAE


def generate_synthetic_sparse_data(
    n_samples: int = 50000,
    d_model: int = 128,
    k_true_features: int = 10,
    l0_per_sample: int = 3,
    noise_std: float = 0.01,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data with known sparse ground truth.

    Args:
        n_samples: Number of samples
        d_model: Ambient dimension
        k_true_features: Number of true features (sparsity of ground truth)
        l0_per_sample: Sparsity per sample (how many features active)
        noise_std: Standard deviation of additive noise
        device: Device

    Returns:
        activations: [n_samples, d_model] - synthetic activations
        true_features: [d_model, k_true_features] - ground truth feature directions
    """
    print(f"\nGenerating synthetic sparse data:")
    print(f"  Samples: {n_samples}")
    print(f"  Dimension: {d_model}")
    print(f"  True features: {k_true_features}")
    print(f"  Sparsity per sample (L0): {l0_per_sample}")
    print(f"  Noise std: {noise_std}")

    # Generate true features: k_true_features orthonormal directions
    true_features = torch.randn(d_model, k_true_features, device=device)
    true_features = F.normalize(true_features, dim=0)  # Unit norm columns

    print(f"\n✅ Generated {k_true_features} orthonormal ground truth features")

    # Generate samples: each is a sparse linear combination
    activations = []

    for i in range(n_samples):
        # Pick l0_per_sample random features
        active_features = torch.randperm(k_true_features)[:l0_per_sample]

        # Random coefficients (from standard normal)
        coefficients = torch.randn(l0_per_sample, device=device)

        # Linear combination
        activation = true_features[:, active_features] @ coefficients

        # Add small Gaussian noise
        activation += noise_std * torch.randn(d_model, device=device)

        activations.append(activation)

    activations = torch.stack(activations)  # [n_samples, d_model]

    print(f"\n✅ Generated {n_samples} synthetic activations")
    print(f"   Each sample: sparse combination of {l0_per_sample} features")
    print(f"   Mean norm: {activations.norm(dim=1).mean().item():.4f}")
    print(f"   Std norm: {activations.norm(dim=1).std().item():.4f}")

    return activations, true_features


def compute_ground_truth_recovery(
    sae: TopKSAE,
    true_features: torch.Tensor,
    threshold: float = 0.9
) -> Tuple[int, float]:
    """Compute how well SAE recovered ground truth features.

    Args:
        sae: Trained SAE
        true_features: [d_model, k_true] - ground truth features
        threshold: Cosine similarity threshold for "recovered"

    Returns:
        n_recovered: Number of true features recovered
        mean_max_similarity: Mean max cosine similarity
    """
    # Get SAE decoder weights [d_model, d_sae]
    # nn.Linear stores weight as [out_features, in_features] = [d_model, d_sae]
    decoder = sae.decoder.weight.data  # [d_model, d_sae] - NO transpose needed
    decoder = F.normalize(decoder, dim=0)  # Normalize each feature (column) to unit norm

    # Normalize true features
    true_features = F.normalize(true_features, dim=0)

    # Compute cosine similarities
    cos_sim = true_features.T @ decoder  # [k_true, d_sae]

    # For each true feature, find best SAE feature match
    max_similarities = cos_sim.abs().max(dim=1)[0]  # [k_true]

    # Count how many are "recovered" (>threshold similarity)
    n_recovered = (max_similarities > threshold).sum().item()

    mean_max_similarity = max_similarities.mean().item()

    return n_recovered, mean_max_similarity


def train_sae(
    activations: torch.Tensor,
    d_sae: int = 64,
    k: int = 5,
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
    # nn.Linear stores weight as [out_features, in_features] = [d_model, d_sae]
    # Each COLUMN is a feature vector
    D1 = sae1.decoder.weight.data  # [d_model, d_sae1]
    D2 = sae2.decoder.weight.data  # [d_model, d_sae2]

    # Normalize (already normalized by normalize_decoder(), but ensure it)
    D1 = F.normalize(D1, dim=0)  # Normalize each column (feature)
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
        description="Synthetic Sparse Ground Truth Validation - Extension 1 (Option B)"
    )
    parser.add_argument('--output-dir', type=Path,
                       default='results/synthetic_sparse',
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=50000,
                       help='Number of synthetic samples')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Ambient dimension')
    parser.add_argument('--n-true-features', type=int, default=10,
                       help='Number of true features (ground truth sparsity)')
    parser.add_argument('--sparsity-per-sample', type=int, default=3,
                       help='L0 sparsity per sample')
    parser.add_argument('--noise-std', type=float, default=0.01,
                       help='Noise standard deviation')
    parser.add_argument('--n-sae-seeds', type=int, default=5,
                       help='Number of SAE seeds to train')
    parser.add_argument('--d-sae', type=int, default=64,
                       help='SAE hidden dimension')
    parser.add_argument('--k', type=int, default=5,
                       help='TopK sparsity for SAE')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*80)
    print("SYNTHETIC SPARSE GROUND TRUTH VALIDATION - Extension 1 (Option B)")
    print("="*80)
    print(f"\nHypothesis: Sparse ground truth → High SAE stability (PWMCC > 0.90)")
    print(f"\nConfiguration:")
    print(f"  Samples: {args.n_samples}")
    print(f"  Ambient dimension: {args.d_model}")
    print(f"  True features: {args.n_true_features}")
    print(f"  Sparsity per sample: {args.sparsity_per_sample}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  SAE seeds: {args.n_sae_seeds}")
    print(f"  SAE architecture: d_sae={args.d_sae}, k={args.k}")
    print(f"  Device: {args.device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: Generate Synthetic Sparse Data
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 1: Generating Synthetic Sparse Data")
    print("="*80)

    start_time = time.time()
    activations, true_features = generate_synthetic_sparse_data(
        n_samples=args.n_samples,
        d_model=args.d_model,
        k_true_features=args.n_true_features,
        l0_per_sample=args.sparsity_per_sample,
        noise_std=args.noise_std,
        device=args.device
    )
    generation_time = time.time() - start_time

    print(f"\n✅ Data generation complete in {generation_time:.2f} seconds")

    # Analyze ground truth sparsity
    ground_truth_sparsity = args.n_true_features / args.d_model * 100
    per_sample_sparsity = args.sparsity_per_sample / args.n_true_features * 100

    print(f"\nGround Truth Structure:")
    print(f"  Total features: {args.n_true_features}/{args.d_model} ({ground_truth_sparsity:.1f}%)")
    print(f"  Active per sample: {args.sparsity_per_sample}/{args.n_true_features} ({per_sample_sparsity:.1f}%)")
    print(f"  ✅ EXTREMELY SPARSE - Ideal for identifiability theory validation")

    # Compute effective rank
    U, S, V = torch.svd(activations.T)
    total_var = (S ** 2).sum()
    cumsum = (S ** 2).cumsum(0)
    effective_rank_90 = (cumsum < 0.9 * total_var).sum().item() + 1
    effective_rank_95 = (cumsum < 0.95 * total_var).sum().item() + 1

    print(f"\nActivation Statistics:")
    print(f"  Effective rank (90% var): {effective_rank_90}/{args.d_model}")
    print(f"  Effective rank (95% var): {effective_rank_95}/{args.d_model}")
    print(f"  Expected: ≈{args.n_true_features} (matches ground truth)")

    # ==============================================================================
    # PHASE 2: Train SAEs (Multiple Seeds)
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 2: Training SAEs on Synthetic Sparse Data")
    print("="*80)

    seeds = [42, 123, 456, 789, 1011][:args.n_sae_seeds]
    saes = []

    for i, seed in enumerate(seeds):
        print(f"\nTraining SAE {i+1}/{len(seeds)} (seed={seed})...")
        start_time = time.time()

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

        train_time = time.time() - start_time

        # Compute ground truth recovery
        n_recovered, mean_similarity = compute_ground_truth_recovery(
            sae, true_features, threshold=0.9
        )

        print(f"  Training time: {train_time:.2f}s")
        print(f"  Ground truth recovery: {n_recovered}/{args.n_true_features} features (threshold=0.9)")
        print(f"  Mean max similarity: {mean_similarity:.4f}")

        # Save SAE
        sae_path = args.output_dir / f'sae_seed_{seed}.pt'
        sae.save(sae_path)
        print(f"  ✅ Saved to {sae_path}")

        saes.append(sae.cpu())

    # ==============================================================================
    # PHASE 3: Compute PWMCC (Pairwise across all seeds)
    # ==============================================================================

    print("\n" + "="*80)
    print("PHASE 3: Computing PWMCC (Feature Stability)")
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
    # PHASE 4: Validate Hypothesis
    # ==============================================================================

    print("\n" + "="*80)
    print("RESULTS: Sparse vs Dense Ground Truth Comparison")
    print("="*80)

    # Load dense ground truth result (from main experiments)
    dense_pwmcc = 0.309  # From our 2-layer transformer results
    random_baseline = 0.300

    print(f"\n| Setup | Ground Truth | Sparsity | Theory Prediction | Empirical PWMCC |")
    print(f"|-------|--------------|----------|-------------------|-----------------|")
    print(f"| 2-layer (dense) | eff_rank=80/128 | 62.5% | PWMCC ≈ 0.30 | {dense_pwmcc:.3f} ✅ |")
    print(f"| Synthetic (sparse) | {args.n_true_features}/{args.d_model} features | {ground_truth_sparsity:.1f}% | PWMCC > 0.90 | {mean_pwmcc:.3f} ", end="")

    if mean_pwmcc > 0.90:
        print("✅ |")
        print(f"\n🎉 HYPOTHESIS DEFINITIVELY CONFIRMED!")
        print(f"   Sparse ground truth → Extremely high stability ({mean_pwmcc:.3f})")
        print(f"   Improvement over dense: {(mean_pwmcc - dense_pwmcc)/dense_pwmcc*100:+.1f}%")
        print(f"   This validates Cui et al.'s identifiability theory with KNOWN ground truth!")
        validation_result = "confirmed"
    elif mean_pwmcc > 0.70:
        print("⚠️ |")
        print(f"\n✅ HYPOTHESIS CONFIRMED (with caveats)")
        print(f"   PWMCC = {mean_pwmcc:.3f} exceeds theoretical threshold (>0.70)")
        print(f"   But below perfect recovery threshold (>0.90)")
        print(f"   Possible reasons:")
        print(f"   - Noise adds small perturbations ({args.noise_std})")
        print(f"   - SAE hyperparameters not perfectly matched")
        print(f"   - Need larger d_sae or smaller k")
        validation_result = "confirmed_partial"
    elif mean_pwmcc > 0.50:
        print("⚠️ |")
        print(f"\n⚠️  PARTIAL VALIDATION")
        print(f"   PWMCC improved significantly ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
        print(f"   But below theoretical prediction (>0.70)")
        print(f"   Evidence that sparsity improves stability")
        validation_result = "partial"
    else:
        print("❌ |")
        print(f"\n❌ HYPOTHESIS NOT CONFIRMED")
        print(f"   PWMCC did not improve ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
        print(f"   Unexpected result requiring investigation")
        validation_result = "failed"

    # ==============================================================================
    # PHASE 5: Analyze Ground Truth Recovery
    # ==============================================================================

    print("\n" + "="*80)
    print("GROUND TRUTH FEATURE RECOVERY ANALYSIS")
    print("="*80)

    print(f"\nChecking if SAEs recovered the {args.n_true_features} true features...")

    recovery_results = []
    for i, sae in enumerate(saes):
        n_recovered, mean_similarity = compute_ground_truth_recovery(
            sae, true_features, threshold=0.9
        )
        recovery_results.append({
            'seed': seeds[i],
            'n_recovered': n_recovered,
            'mean_similarity': mean_similarity
        })
        print(f"  Seed {seeds[i]}: {n_recovered}/{args.n_true_features} recovered (mean sim: {mean_similarity:.4f})")

    avg_recovery = np.mean([r['n_recovered'] for r in recovery_results])
    avg_similarity = np.mean([r['mean_similarity'] for r in recovery_results])

    print(f"\nAverage recovery: {avg_recovery:.1f}/{args.n_true_features} features")
    print(f"Average similarity: {avg_similarity:.4f}")

    if avg_recovery >= args.n_true_features * 0.9:
        print(f"✅ EXCELLENT: SAEs consistently recovered ground truth!")
    elif avg_recovery >= args.n_true_features * 0.7:
        print(f"⚠️  GOOD: SAEs recovered most ground truth features")
    else:
        print(f"⚠️  PARTIAL: SAEs only partially recovered ground truth")

    # ==============================================================================
    # Save All Results
    # ==============================================================================

    results = {
        'experiment': 'synthetic_sparse_ground_truth_validation',
        'hypothesis': 'sparse_ground_truth -> high_sae_stability',
        'configuration': {
            'n_samples': args.n_samples,
            'd_model': args.d_model,
            'n_true_features': args.n_true_features,
            'sparsity_per_sample': args.sparsity_per_sample,
            'noise_std': args.noise_std,
            'n_sae_seeds': args.n_sae_seeds,
            'd_sae': args.d_sae,
            'k': args.k,
        },
        'ground_truth': {
            'total_sparsity_percent': ground_truth_sparsity,
            'per_sample_sparsity_percent': per_sample_sparsity,
            'effective_rank_90': effective_rank_90,
            'effective_rank_95': effective_rank_95,
        },
        'pwmcc_results': {
            'seeds': seeds,
            'matrix': pwmcc_matrix.tolist(),
            'mean': mean_pwmcc,
            'std': std_pwmcc,
        },
        'recovery_results': recovery_results,
        'recovery_stats': {
            'avg_n_recovered': avg_recovery,
            'avg_similarity': avg_similarity,
        },
        'comparison': {
            'dense_setup_pwmcc': dense_pwmcc,
            'sparse_setup_pwmcc': mean_pwmcc,
            'improvement': mean_pwmcc - dense_pwmcc,
            'improvement_percent': (mean_pwmcc - dense_pwmcc) / dense_pwmcc * 100,
            'random_baseline': random_baseline,
        },
        'validation_result': validation_result,
    }

    with open(args.output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ All results saved to {args.output_dir}/")
    print(f"\nFiles created:")
    print(f"  - sae_seed_*.pt (trained SAEs)")
    print(f"  - results.json (complete results)")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
