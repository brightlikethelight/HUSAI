#!/usr/bin/env python3
"""Alternative stability metrics for trained vs random SAEs.

CRITICAL CONTEXT:
- PWMCC metric shows trained SAEs = random SAEs (both ~0.30)
- We need metrics where trained SAEs SHOULD outperform random

This script tests 5 alternative metrics:
1. Reconstruction Loss - trained should have MUCH lower loss than random
2. Feature Activation Sparsity - trained should have sparser, more efficient activations
3. Explained Variance - trained should explain more variance in activations
4. Feature Correlation Stability - correlation of feature activations across seeds
5. Top-k Activation Overlap - do same inputs activate same features across seeds?

Expected outcomes:
- Reconstruction Loss: trained << random (by orders of magnitude)
- Sparsity: trained should be sparser (lower L0, similar or lower L1)
- Explained Variance: trained >> random (close to 1.0 vs near 0)
- Feature Correlation: both should be low (no matching expected)
- Top-k Overlap: trained > random (same inputs → same features)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/alternative_stability_metrics.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_sae import TopKSAE, ReLUSAE
from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import create_dataloaders

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
SAES_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'

SEEDS = [42, 123, 456, 789, 1011]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAE architecture parameters
TOPK_CONFIG = {
    'd_model': 128,
    'd_sae': 1024,  # 8x expansion
    'k': 32
}

RELU_CONFIG = {
    'd_model': 128,
    'd_sae': 1024,  # 8x expansion
    'l1_coef': 0.001
}


@dataclass
class MetricResults:
    """Container for all metrics for a single SAE."""
    architecture: str
    seed: int
    is_random: bool
    mse_loss: float
    explained_variance: float
    l0_sparsity: float
    l1_norm: float
    frac_active_features: float
    feature_correlation: float = None
    topk_overlap: float = None


def get_activations(p=113, layer_idx=1, position=2, max_samples=5000):
    """Extract activations from transformer.

    Args:
        p: Modulus for dataset
        layer_idx: Which layer to extract from
        position: Which position (2 = after op token)
        max_samples: Maximum number of samples to extract

    Returns:
        Tensor of shape [n_samples, d_model]
    """
    # Try multiple possible paths
    possible_paths = [
        RESULTS_DIR / 'checkpoints' / 'transformer_grokking.pt',
        RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt',
        RESULTS_DIR / 'baseline_relu_seed42' / 'transformer_best.pt',
    ]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(f"Transformer model not found in any of: {possible_paths}")

    # Load model using ModularArithmeticTransformer's load_checkpoint method
    model, _ = ModularArithmeticTransformer.load_checkpoint(model_path, device=DEVICE)
    model.eval()
    print(f"  ✓ Loaded transformer from {model_path}")

    # Create dataset
    train_loader, _ = create_dataloaders(
        modulus=p,
        fraction=1.0,
        train_fraction=0.8,
        batch_size=256,
        seed=42,
        format="sequence",
        num_workers=0
    )

    activations = []
    n_samples = 0

    with torch.no_grad():
        for batch in train_loader:
            # Unpack the batch (x, y)
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch

            x_batch = x_batch.to(DEVICE)

            # Get activations using the model's method
            layer_acts = model.get_activations(x_batch, layer=layer_idx)

            # Extract specific position
            position_acts = layer_acts[:, position, :]
            activations.append(position_acts.cpu())

            n_samples += position_acts.shape[0]
            if max_samples is not None and n_samples >= max_samples:
                break

    activations = torch.cat(activations, dim=0)
    if max_samples is not None:
        activations = activations[:max_samples]

    return activations


def create_random_sae(architecture: str, seed: int):
    """Create a random (untrained) SAE for baseline comparison."""
    torch.manual_seed(seed)

    if architecture == 'topk':
        sae = TopKSAE(**TOPK_CONFIG)
    elif architecture == 'relu':
        sae = ReLUSAE(**RELU_CONFIG)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    sae.reset_parameters()
    return sae.to(DEVICE)


def load_trained_sae(architecture: str, seed: int):
    """Load a trained SAE checkpoint."""
    sae_path = SAES_DIR / f'{architecture}_seed{seed}' / 'sae_final.pt'

    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    if architecture == 'topk':
        sae = TopKSAE.load(sae_path, device=DEVICE)
    elif architecture == 'relu':
        sae = ReLUSAE.load(sae_path, device=DEVICE)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return sae


def compute_reconstruction_metrics(sae, activations: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction loss and explained variance."""
    sae.eval()

    with torch.no_grad():
        batch_size = 1024
        all_reconstructions = []

        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(DEVICE)
            reconstruction, _, _ = sae(batch)
            all_reconstructions.append(reconstruction.cpu())

        reconstructions = torch.cat(all_reconstructions, dim=0)

    # MSE loss
    mse = F.mse_loss(reconstructions, activations).item()

    # Explained variance: 1 - (Var(residual) / Var(original))
    residuals = activations - reconstructions
    total_variance = torch.var(activations, dim=0).sum().item()
    residual_variance = torch.var(residuals, dim=0).sum().item()

    if total_variance > 1e-8:
        explained_variance = 1 - (residual_variance / total_variance)
    else:
        explained_variance = 0.0

    return {
        'mse_loss': mse,
        'explained_variance': explained_variance
    }


def compute_sparsity_metrics(sae, activations: torch.Tensor) -> Dict[str, float]:
    """Compute sparsity metrics on latent activations."""
    sae.eval()

    with torch.no_grad():
        batch_size = 1024
        all_latents = []

        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(DEVICE)
            _, latents, _ = sae(batch)
            all_latents.append(latents.cpu())

        latents = torch.cat(all_latents, dim=0)

    # L0: average number of active features per sample
    active_mask = latents.abs() > 1e-6
    l0 = active_mask.float().sum(dim=1).mean().item()

    # L1: average L1 norm
    l1 = latents.abs().sum(dim=1).mean().item()

    # Fraction of features ever active across all samples
    ever_active = active_mask.any(dim=0)
    frac_active = ever_active.float().mean().item()

    return {
        'l0_sparsity': l0,
        'l1_norm': l1,
        'frac_active_features': frac_active
    }


def compute_feature_correlation_stability(
    saes: List[Tuple[str, int, any]],
    activations: torch.Tensor,
    n_samples: int = 1000
) -> float:
    """Compute correlation of feature activations across different SAE seeds.

    This measures whether different SAE instances learn similar feature representations.
    NOT the same as PWMCC (which measures feature matching).
    """
    activations_subset = activations[:n_samples].to(DEVICE)

    # Get latent activations for each SAE
    all_latents = []

    with torch.no_grad():
        for arch, seed, sae in saes:
            sae.eval()
            _, latents, _ = sae(activations_subset)
            # Flatten to [n_samples * d_sae]
            all_latents.append(latents.flatten().cpu().numpy())

    # Compute pairwise correlations
    correlations = []
    for i in range(len(all_latents)):
        for j in range(i + 1, len(all_latents)):
            corr = np.corrcoef(all_latents[i], all_latents[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0


def compute_topk_overlap(
    saes: List[Tuple[str, int, any]],
    activations: torch.Tensor,
    k: int = 32,
    n_samples: int = 1000
) -> float:
    """Compute overlap of top-k active features across SAE seeds.

    For the SAME input, do different SAE instances activate similar features?
    This is a direct measure of representational stability.
    """
    activations_subset = activations[:n_samples].to(DEVICE)

    # Get top-k active features for each SAE
    all_topk_sets = []

    with torch.no_grad():
        for arch, seed, sae in saes:
            sae.eval()
            _, latents, _ = sae(activations_subset)

            # For each sample, get indices of top-k features
            topk_indices = []
            for i in range(latents.shape[0]):
                sample_latents = latents[i].abs()
                topk_idx = torch.topk(sample_latents, k=min(k, sample_latents.shape[0]))[1]
                topk_indices.append(set(topk_idx.cpu().numpy()))

            all_topk_sets.append(topk_indices)

    # Compute Jaccard similarity for each sample across SAE pairs
    jaccard_similarities = []

    for sample_idx in range(n_samples):
        for i in range(len(all_topk_sets)):
            for j in range(i + 1, len(all_topk_sets)):
                set_i = all_topk_sets[i][sample_idx]
                set_j = all_topk_sets[j][sample_idx]

                intersection = len(set_i & set_j)
                union = len(set_i | set_j)

                if union > 0:
                    jaccard = intersection / union
                    jaccard_similarities.append(jaccard)

    return np.mean(jaccard_similarities) if jaccard_similarities else 0.0


def evaluate_sae(
    architecture: str,
    seed: int,
    is_random: bool,
    activations: torch.Tensor
) -> MetricResults:
    """Evaluate a single SAE on all individual metrics."""
    # Load SAE
    if is_random:
        sae = create_random_sae(architecture, seed)
        label = f"{architecture.upper()} Random (seed={seed})"
    else:
        sae = load_trained_sae(architecture, seed)
        label = f"{architecture.upper()} Trained (seed={seed})"

    print(f"  Evaluating {label}...")

    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(sae, activations)

    # Sparsity metrics
    sparsity_metrics = compute_sparsity_metrics(sae, activations)

    return MetricResults(
        architecture=architecture,
        seed=seed,
        is_random=is_random,
        **recon_metrics,
        **sparsity_metrics
    )


def compute_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Standard PWMCC (for reference)."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("ALTERNATIVE STABILITY METRICS: Trained vs Random SAEs")
    print("=" * 80)
    print()
    print("CRITICAL CONTEXT:")
    print("  - PWMCC metric shows trained SAEs = random SAEs (~0.30)")
    print("  - Testing alternative metrics where trained SHOULD outperform random")
    print()
    print("Metrics tested:")
    print("  1. Reconstruction Loss (trained << random)")
    print("  2. Sparsity (trained should be sparser)")
    print("  3. Explained Variance (trained >> random)")
    print("  4. Feature Correlation Stability")
    print("  5. Top-k Activation Overlap")
    print()
    print("=" * 80)
    print()

    # Extract activations
    print("Step 1: Loading transformer and extracting activations...")
    print("  (layer 1, position 2, max 5000 samples)")
    activations = get_activations(p=113, layer_idx=1, position=2, max_samples=5000)
    print(f"  ✓ Extracted activations: {activations.shape}")
    print()

    # Evaluate all SAEs
    print("Step 2: Evaluating SAEs on individual metrics...")
    print()

    all_results = []

    for architecture in ['topk', 'relu']:
        print(f"Architecture: {architecture.upper()}")
        print("-" * 40)

        # Trained SAEs
        for seed in SEEDS:
            result = evaluate_sae(architecture, seed, is_random=False, activations=activations)
            all_results.append(result)

        # Random SAEs (same seeds for fair comparison)
        for seed in SEEDS:
            result = evaluate_sae(architecture, seed, is_random=True, activations=activations)
            all_results.append(result)

        print()

    # Compute cross-seed metrics
    print("Step 3: Computing cross-seed stability metrics...")
    print()

    for architecture in ['topk', 'relu']:
        print(f"Architecture: {architecture.upper()}")
        print("-" * 40)

        # Load all trained SAEs for this architecture
        trained_saes = []
        for seed in SEEDS:
            sae = load_trained_sae(architecture, seed)
            trained_saes.append((architecture, seed, sae))

        # Load all random SAEs for this architecture
        random_saes = []
        for seed in SEEDS:
            sae = create_random_sae(architecture, seed)
            random_saes.append((architecture, seed, sae))

        # Feature correlation stability
        print(f"  Computing feature correlation stability...")
        trained_corr = compute_feature_correlation_stability(trained_saes, activations, n_samples=1000)
        random_corr = compute_feature_correlation_stability(random_saes, activations, n_samples=1000)

        # Top-k overlap
        print(f"  Computing top-k activation overlap...")
        k_val = TOPK_CONFIG['k'] if architecture == 'topk' else 32
        trained_overlap = compute_topk_overlap(trained_saes, activations, k=k_val, n_samples=1000)
        random_overlap = compute_topk_overlap(random_saes, activations, k=k_val, n_samples=1000)

        # Update results with cross-seed metrics
        for result in all_results:
            if result.architecture == architecture:
                if result.is_random:
                    result.feature_correlation = random_corr
                    result.topk_overlap = random_overlap
                else:
                    result.feature_correlation = trained_corr
                    result.topk_overlap = trained_overlap

        print(f"  ✓ Trained feature correlation: {trained_corr:.4f}")
        print(f"  ✓ Random feature correlation:  {random_corr:.4f}")
        print(f"  ✓ Trained top-k overlap:       {trained_overlap:.4f}")
        print(f"  ✓ Random top-k overlap:        {random_overlap:.4f}")
        print()

    # Create summary table
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame([
        {
            'Architecture': r.architecture.upper(),
            'Type': 'Trained' if not r.is_random else 'Random',
            'Seed': r.seed,
            'MSE Loss': r.mse_loss,
            'Explained Var': r.explained_variance,
            'L0 Sparsity': r.l0_sparsity,
            'L1 Norm': r.l1_norm,
            'Frac Active': r.frac_active_features,
            'Feature Corr': r.feature_correlation,
            'TopK Overlap': r.topk_overlap
        }
        for r in all_results
    ])

    # Aggregate statistics
    print("AGGREGATE STATISTICS (Mean ± Std)")
    print("-" * 80)
    print()

    for architecture in ['TOPK', 'RELU']:
        print(f"{architecture} Architecture:")
        print()

        # Trained stats
        trained_df = df[(df['Architecture'] == architecture) & (df['Type'] == 'Trained')]
        print(f"  TRAINED SAEs (n={len(trained_df)}):")
        print(f"    MSE Loss:          {trained_df['MSE Loss'].mean():.6f} ± {trained_df['MSE Loss'].std():.6f}")
        print(f"    Explained Var:     {trained_df['Explained Var'].mean():.4f} ± {trained_df['Explained Var'].std():.4f}")
        print(f"    L0 Sparsity:       {trained_df['L0 Sparsity'].mean():.2f} ± {trained_df['L0 Sparsity'].std():.2f}")
        print(f"    L1 Norm:           {trained_df['L1 Norm'].mean():.2f} ± {trained_df['L1 Norm'].std():.2f}")
        print(f"    Frac Active:       {trained_df['Frac Active'].mean():.4f} ± {trained_df['Frac Active'].std():.4f}")
        print(f"    Feature Corr:      {trained_df['Feature Corr'].iloc[0]:.4f}")
        print(f"    TopK Overlap:      {trained_df['TopK Overlap'].iloc[0]:.4f}")
        print()

        # Random stats
        random_df = df[(df['Architecture'] == architecture) & (df['Type'] == 'Random')]
        print(f"  RANDOM SAEs (n={len(random_df)}):")
        print(f"    MSE Loss:          {random_df['MSE Loss'].mean():.6f} ± {random_df['MSE Loss'].std():.6f}")
        print(f"    Explained Var:     {random_df['Explained Var'].mean():.4f} ± {random_df['Explained Var'].std():.4f}")
        print(f"    L0 Sparsity:       {random_df['L0 Sparsity'].mean():.2f} ± {random_df['L0 Sparsity'].std():.2f}")
        print(f"    L1 Norm:           {random_df['L1 Norm'].mean():.2f} ± {random_df['L1 Norm'].std():.2f}")
        print(f"    Frac Active:       {random_df['Frac Active'].mean():.4f} ± {random_df['Frac Active'].std():.4f}")
        print(f"    Feature Corr:      {random_df['Feature Corr'].iloc[0]:.4f}")
        print(f"    TopK Overlap:      {random_df['TopK Overlap'].iloc[0]:.4f}")
        print()

        # Comparison
        print(f"  COMPARISON (Trained vs Random):")
        mse_ratio = random_df['MSE Loss'].mean() / trained_df['MSE Loss'].mean()
        ev_diff = trained_df['Explained Var'].mean() - random_df['Explained Var'].mean()
        overlap_diff = trained_df['TopK Overlap'].iloc[0] - random_df['TopK Overlap'].iloc[0]

        print(f"    MSE Reduction:     {mse_ratio:.2f}x better (random/trained)")
        print(f"    EV Improvement:    {ev_diff:+.4f} (trained - random)")
        print(f"    Overlap Gain:      {overlap_diff:+.4f} (trained - random)")
        print()

    # Save detailed results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / 'alternative_stability_metrics.csv'
    df.to_csv(csv_path, index=False)

    print("=" * 80)
    print(f"✓ Saved detailed results to {csv_path}")
    print("=" * 80)
    print()

    # Key findings
    print("KEY FINDINGS:")
    print("-" * 80)

    # Check if trained outperforms random on each metric
    findings = []

    for arch in ['TOPK', 'RELU']:
        trained = df[(df['Architecture'] == arch) & (df['Type'] == 'Trained')]
        random = df[(df['Architecture'] == arch) & (df['Type'] == 'Random')]

        # MSE: lower is better
        if trained['MSE Loss'].mean() < random['MSE Loss'].mean():
            findings.append(f"✓ {arch}: Trained has LOWER reconstruction loss (better)")
        else:
            findings.append(f"✗ {arch}: Trained does NOT have lower reconstruction loss (UNEXPECTED)")

        # Explained Variance: higher is better
        if trained['Explained Var'].mean() > random['Explained Var'].mean():
            findings.append(f"✓ {arch}: Trained has HIGHER explained variance (better)")
        else:
            findings.append(f"✗ {arch}: Trained does NOT have higher explained variance (UNEXPECTED)")

        # Top-k Overlap: higher is better (more stable)
        if trained['TopK Overlap'].iloc[0] > random['TopK Overlap'].iloc[0]:
            findings.append(f"✓ {arch}: Trained has HIGHER top-k overlap (more stable)")
        else:
            findings.append(f"✗ {arch}: Trained does NOT have higher top-k overlap (concerning)")

    for finding in findings:
        print(f"  {finding}")

    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("  If trained SAEs outperform random on these metrics, training is working!")
    print("  If not, we have a fundamental problem with SAE training or evaluation.")
    print("=" * 80)


if __name__ == '__main__':
    main()
