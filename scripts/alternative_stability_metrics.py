#!/usr/bin/env python3
"""Test alternative stability metrics beyond PWMCC.

Since PWMCC = random baseline, we need metrics that can distinguish
trained from random SAEs.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/alternative_stability_metrics.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_sae import TopKSAE


def compute_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Standard PWMCC."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_high_match_fraction(decoder1: torch.Tensor, decoder2: torch.Tensor, 
                                 threshold: float = 0.7) -> float:
    """Fraction of features with a match above threshold."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_sim = cos_sim.abs().max(dim=1)[0]
    return (max_sim > threshold).float().mean().item()


def compute_mutual_nearest_neighbors(decoder1: torch.Tensor, decoder2: torch.Tensor,
                                      threshold: float = 0.5) -> float:
    """Fraction of features that are mutual nearest neighbors."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    
    # Best match from 1 to 2
    best_1to2 = cos_sim.abs().argmax(dim=1)
    # Best match from 2 to 1
    best_2to1 = cos_sim.abs().argmax(dim=0)
    
    # Check mutual: if feature i's best match is j, and j's best match is i
    n_features = decoder1.shape[1]
    mutual_count = 0
    for i in range(n_features):
        j = best_1to2[i].item()
        if best_2to1[j].item() == i:
            # Also check similarity is above threshold
            if cos_sim[i, j].abs().item() > threshold:
                mutual_count += 1
    
    return mutual_count / n_features


def compute_hungarian_matching(decoder1: torch.Tensor, decoder2: torch.Tensor) -> dict:
    """Optimal bipartite matching using Hungarian algorithm."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    
    # Hungarian algorithm minimizes cost, so use negative similarity
    cost_matrix = -cos_sim.abs().cpu().numpy()
    
    # Only use subset if too large (Hungarian is O(n³))
    n = min(500, cost_matrix.shape[0], cost_matrix.shape[1])
    cost_subset = cost_matrix[:n, :n]
    
    row_ind, col_ind = linear_sum_assignment(cost_subset)
    
    # Get matched similarities
    matched_sims = cos_sim.abs()[row_ind, col_ind].cpu().numpy()
    
    return {
        'mean_matched_sim': float(matched_sims.mean()),
        'median_matched_sim': float(np.median(matched_sims)),
        'fraction_above_0.5': float((matched_sims > 0.5).mean()),
        'fraction_above_0.7': float((matched_sims > 0.7).mean()),
        'fraction_above_0.9': float((matched_sims > 0.9).mean()),
    }


def compute_subspace_overlap(decoder1: torch.Tensor, decoder2: torch.Tensor, 
                              k: int = 100) -> float:
    """Overlap of top-k principal subspaces."""
    # SVD of decoders
    U1, S1, _ = torch.svd(decoder1)
    U2, S2, _ = torch.svd(decoder2)
    
    # Take top-k singular vectors
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    # Subspace overlap: ||U1_k.T @ U2_k||_F² / k
    overlap_matrix = U1_k.T @ U2_k
    overlap = (overlap_matrix ** 2).sum().item() / k
    
    return overlap


def main():
    print("=" * 70)
    print("ALTERNATIVE STABILITY METRICS ANALYSIS")
    print("=" * 70)
    print()
    
    d_model = 128
    d_sae = 1024
    k = 32
    
    # Create random SAEs
    print("Creating random SAEs...")
    torch.manual_seed(100)
    random_sae1 = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    torch.manual_seed(101)
    random_sae2 = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    
    # Load trained SAEs
    print("Loading trained SAEs...")
    trained_sae1 = TopKSAE.load(Path('results/saes/topk_seed42/sae_final.pt'))
    trained_sae2 = TopKSAE.load(Path('results/saes/topk_seed123/sae_final.pt'))
    
    print()
    print("=" * 70)
    print("METRIC COMPARISON: RANDOM vs TRAINED")
    print("=" * 70)
    
    # Compare metrics
    metrics = [
        ("PWMCC", compute_pwmcc),
        ("High Match Fraction (>0.7)", lambda d1, d2: compute_high_match_fraction(d1, d2, 0.7)),
        ("High Match Fraction (>0.5)", lambda d1, d2: compute_high_match_fraction(d1, d2, 0.5)),
        ("Mutual NN (>0.5)", lambda d1, d2: compute_mutual_nearest_neighbors(d1, d2, 0.5)),
        ("Mutual NN (>0.3)", lambda d1, d2: compute_mutual_nearest_neighbors(d1, d2, 0.3)),
        ("Subspace Overlap (k=50)", lambda d1, d2: compute_subspace_overlap(d1, d2, 50)),
        ("Subspace Overlap (k=100)", lambda d1, d2: compute_subspace_overlap(d1, d2, 100)),
    ]
    
    print(f"\n{'Metric':<35} {'Random':>12} {'Trained':>12} {'Diff':>12}")
    print("-" * 70)
    
    results = {}
    for name, metric_fn in metrics:
        random_val = metric_fn(random_sae1.decoder.weight.data, 
                               random_sae2.decoder.weight.data)
        trained_val = metric_fn(trained_sae1.decoder.weight.data,
                                trained_sae2.decoder.weight.data)
        diff = trained_val - random_val
        
        results[name] = {'random': random_val, 'trained': trained_val, 'diff': diff}
        
        # Highlight if trained > random
        marker = "✓" if diff > 0.01 else " "
        print(f"{name:<35} {random_val:>12.4f} {trained_val:>12.4f} {diff:>+12.4f} {marker}")
    
    # Hungarian matching (separate due to different output)
    print()
    print("Hungarian Matching Analysis:")
    print("-" * 70)
    
    random_hungarian = compute_hungarian_matching(random_sae1.decoder.weight.data,
                                                   random_sae2.decoder.weight.data)
    trained_hungarian = compute_hungarian_matching(trained_sae1.decoder.weight.data,
                                                    trained_sae2.decoder.weight.data)
    
    for key in random_hungarian:
        r_val = random_hungarian[key]
        t_val = trained_hungarian[key]
        diff = t_val - r_val
        marker = "✓" if diff > 0.01 else " "
        print(f"  {key:<30} {r_val:>12.4f} {t_val:>12.4f} {diff:>+12.4f} {marker}")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    improvements = sum(1 for r in results.values() if r['diff'] > 0.01)
    print(f"\nMetrics where Trained > Random (by >0.01): {improvements}/{len(results)}")
    
    if improvements == 0:
        print("\n❌ NO metrics show trained SAEs are more stable than random!")
        print("   This confirms the critical finding: standard SAE training")
        print("   does not produce reproducible features.")
    elif improvements < len(results) // 2:
        print("\n⚠️  Few metrics show improvement over random baseline.")
        print("   The stability problem is severe.")
    else:
        print("\n✓ Some metrics show trained > random.")
        print("   There may be SOME learned structure.")
    
    # Save results
    import json
    output = {
        'metrics': results,
        'hungarian_random': random_hungarian,
        'hungarian_trained': trained_hungarian,
    }
    
    output_path = Path('results/analysis/alternative_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
