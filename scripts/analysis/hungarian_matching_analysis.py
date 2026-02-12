#!/usr/bin/env python3
"""Hungarian Matching Analysis for SAE Feature Comparison.

This script implements the methodology from Paulo & Belrose (2025):
"Sparse Autoencoders Trained on the Same Data Learn Different Features"

Instead of just computing PWMCC (mean of max cosine similarity),
we use the Hungarian algorithm to find optimal 1-to-1 matching
between features, then classify features as "shared" or "orphan".

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/hungarian_matching_analysis.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
SAE_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'

SEEDS = [42, 123, 456, 789, 1011]


def load_decoder(seed: int) -> torch.Tensor:
    """Load decoder weights from SAE checkpoint."""
    for name in ['sae_final.pt', 'sae.pt']:
        path = SAE_DIR / f'topk_seed{seed}' / name
        if path.exists():
            ckpt = torch.load(path, map_location='cpu')
            return ckpt['model_state_dict']['decoder.weight']
    raise FileNotFoundError(f"No SAE found for seed {seed}")


def hungarian_matching(D1: torch.Tensor, D2: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal 1-to-1 matching using Hungarian algorithm.
    
    Args:
        D1: Decoder weights [d_model, d_sae]
        D2: Decoder weights [d_model, d_sae]
    
    Returns:
        row_ind: Indices in D1
        col_ind: Matched indices in D2
        similarities: Cosine similarities for matched pairs
    """
    # Normalize columns
    D1_norm = F.normalize(D1, dim=0)
    D2_norm = F.normalize(D2, dim=0)
    
    # Cosine similarity matrix [d_sae, d_sae]
    cos_sim = (D1_norm.T @ D2_norm).numpy()
    
    # Convert to cost matrix (Hungarian minimizes cost)
    # We want to maximize similarity, so use 1 - |cos_sim|
    cost_matrix = 1 - np.abs(cos_sim)
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Get similarities for matched pairs
    similarities = np.abs(cos_sim[row_ind, col_ind])
    
    return row_ind, col_ind, similarities


def classify_features(similarities: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Classify features as shared or orphan based on similarity threshold.
    
    Args:
        similarities: Cosine similarities for matched pairs
        threshold: Similarity threshold for "shared" classification
    
    Returns:
        shared_mask: Boolean mask for shared features
        orphan_mask: Boolean mask for orphan features
    """
    shared_mask = similarities >= threshold
    orphan_mask = ~shared_mask
    return shared_mask, orphan_mask


def analyze_feature_overlap():
    """Run Hungarian matching analysis on all SAE pairs."""
    print("=" * 70)
    print("HUNGARIAN MATCHING ANALYSIS")
    print("=" * 70)
    print()
    print("Methodology from Paulo & Belrose (2025):")
    print("'Sparse Autoencoders Trained on the Same Data Learn Different Features'")
    print()
    
    # Load all decoders
    print("Loading SAE decoders...")
    decoders = {}
    for seed in SEEDS:
        try:
            decoders[seed] = load_decoder(seed)
            print(f"  ✓ Loaded seed {seed}")
        except Exception as e:
            print(f"  ✗ Failed to load seed {seed}: {e}")
    
    if len(decoders) < 2:
        print("Not enough SAEs loaded!")
        return
    
    seeds = list(decoders.keys())
    n_saes = len(seeds)
    d_sae = decoders[seeds[0]].shape[1]
    
    print(f"\nAnalyzing {n_saes} SAEs with {d_sae} features each")
    print()
    
    # Analyze all pairs
    print("=" * 70)
    print("PAIRWISE ANALYSIS")
    print("=" * 70)
    print()
    
    all_similarities = []
    pair_results = []
    
    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            row_ind, col_ind, similarities = hungarian_matching(
                decoders[seed1], decoders[seed2]
            )
            
            all_similarities.extend(similarities)
            
            # Classify with different thresholds
            shared_05, orphan_05 = classify_features(similarities, 0.5)
            shared_07, orphan_07 = classify_features(similarities, 0.7)
            shared_03, orphan_03 = classify_features(similarities, 0.3)
            
            result = {
                'seed1': seed1,
                'seed2': seed2,
                'mean_similarity': float(similarities.mean()),
                'std_similarity': float(similarities.std()),
                'min_similarity': float(similarities.min()),
                'max_similarity': float(similarities.max()),
                'pct_shared_05': float(100 * shared_05.sum() / len(similarities)),
                'pct_shared_07': float(100 * shared_07.sum() / len(similarities)),
                'pct_shared_03': float(100 * shared_03.sum() / len(similarities)),
            }
            pair_results.append(result)
            
            print(f"Seed {seed1} vs {seed2}:")
            print(f"  Mean similarity: {result['mean_similarity']:.4f} ± {result['std_similarity']:.4f}")
            print(f"  Shared (>0.5): {result['pct_shared_05']:.1f}%")
            print(f"  Shared (>0.7): {result['pct_shared_07']:.1f}%")
            print()
    
    # Overall statistics
    all_similarities = np.array(all_similarities)
    
    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print()
    
    print("Matched similarity distribution:")
    print(f"  Mean: {all_similarities.mean():.4f}")
    print(f"  Std:  {all_similarities.std():.4f}")
    print(f"  Min:  {all_similarities.min():.4f}")
    print(f"  Max:  {all_similarities.max():.4f}")
    print()
    
    print("Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_similarities, p)
        print(f"  {p}th: {val:.4f}")
    print()
    
    # Compare with PWMCC
    print("=" * 70)
    print("COMPARISON WITH PWMCC")
    print("=" * 70)
    print()
    
    # Compute PWMCC for comparison
    pwmcc_values = []
    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            D1_norm = F.normalize(decoders[seed1], dim=0)
            D2_norm = F.normalize(decoders[seed2], dim=0)
            cos_sim = D1_norm.T @ D2_norm
            max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
            max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
            pwmcc = (max_1to2 + max_2to1) / 2
            pwmcc_values.append(pwmcc)
    
    print(f"PWMCC (mean of max): {np.mean(pwmcc_values):.4f} ± {np.std(pwmcc_values):.4f}")
    print(f"Hungarian (mean of matched): {all_similarities.mean():.4f} ± {all_similarities.std():.4f}")
    print()
    
    # The difference is because:
    # - PWMCC: For each feature, find the BEST match (may reuse matches)
    # - Hungarian: Find optimal 1-to-1 matching (no reuse)
    print("Note: PWMCC is higher because it allows 'reusing' good matches.")
    print("Hungarian matching enforces 1-to-1 correspondence.")
    print()
    
    # Classification summary
    print("=" * 70)
    print("FEATURE CLASSIFICATION")
    print("=" * 70)
    print()
    
    avg_shared_05 = np.mean([r['pct_shared_05'] for r in pair_results])
    avg_shared_07 = np.mean([r['pct_shared_07'] for r in pair_results])
    avg_shared_03 = np.mean([r['pct_shared_03'] for r in pair_results])
    
    print(f"Average % shared features:")
    print(f"  Threshold 0.3: {avg_shared_03:.1f}%")
    print(f"  Threshold 0.5: {avg_shared_05:.1f}%")
    print(f"  Threshold 0.7: {avg_shared_07:.1f}%")
    print()
    
    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    if avg_shared_05 < 10:
        print("⚠️  Very few features are shared across seeds (<10% at threshold 0.5)")
        print("   This indicates high seed dependence, consistent with Paulo & Belrose (2025).")
    elif avg_shared_05 < 50:
        print("⚠️  Moderate feature sharing ({:.1f}% at threshold 0.5)".format(avg_shared_05))
        print("   Some features are reproducible, but many are seed-dependent.")
    else:
        print("✓ High feature sharing ({:.1f}% at threshold 0.5)".format(avg_shared_05))
        print("   Most features are reproducible across seeds.")
    
    # Save results
    results = {
        'pair_results': pair_results,
        'overall': {
            'mean_similarity': float(all_similarities.mean()),
            'std_similarity': float(all_similarities.std()),
            'min_similarity': float(all_similarities.min()),
            'max_similarity': float(all_similarities.max()),
            'pwmcc_mean': float(np.mean(pwmcc_values)),
            'pwmcc_std': float(np.std(pwmcc_values)),
            'pct_shared_03': float(avg_shared_03),
            'pct_shared_05': float(avg_shared_05),
            'pct_shared_07': float(avg_shared_07),
        }
    }
    
    output_path = OUTPUT_DIR / 'hungarian_matching_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    analyze_feature_overlap()
