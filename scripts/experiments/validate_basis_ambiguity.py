#!/usr/bin/env python3
"""Validate the Basis Ambiguity Discovery

This script tests the key claim: SAEs learn the same SUBSPACE but different BASES.

If true:
- PWMCC (feature overlap) should be low (~0.26)
- Subspace overlap should be HIGH (>0.90)

This would confirm that SAE "instability" is basis ambiguity, not wrong features.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'


def compute_subspace_overlap(D1, D2, k=None):
    """
    Compute subspace overlap between two decoder matrices.
    
    Args:
        D1: [d_model, d_sae] decoder matrix
        D2: [d_model, d_sae] decoder matrix
        k: Number of principal components to compare
    
    Returns:
        Overlap in [0, 1]: 1.0 = perfect subspace match, 0.0 = orthogonal
    """
    U1, S1, _ = torch.svd(D1)
    U2, S2, _ = torch.svd(D2)
    
    if k is None:
        k = min(D1.shape[1], D2.shape[1])
    
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    overlap = (U1_k.T @ U2_k).pow(2).sum() / k
    return overlap.item()


def compute_pwmcc(D1, D2):
    """Compute PWMCC between two decoder matrices."""
    D1_norm = F.normalize(D1, dim=0)
    D2_norm = F.normalize(D2, dim=0)
    cos_sim = D1_norm.T @ D2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def load_sae_decoders(results_dir):
    """Load decoder matrices from saved SAEs."""
    decoders = []
    seeds = []
    for sae_file in sorted(results_dir.glob('sae_seed_*.pt')):
        checkpoint = torch.load(sae_file, map_location='cpu')
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            decoder = checkpoint['model_state_dict']['decoder.weight']
        else:
            decoder = checkpoint['decoder.weight']
        decoders.append(decoder)
        seed = int(sae_file.stem.split('_')[-1])
        seeds.append(seed)
    return decoders, seeds


def main():
    print("=" * 70)
    print("VALIDATING BASIS AMBIGUITY HYPOTHESIS")
    print("=" * 70)
    
    # Load synthetic sparse exact results
    exact_dir = RESULTS_DIR / 'synthetic_sparse_exact'
    if not exact_dir.exists():
        print(f"ERROR: {exact_dir} not found")
        return
    
    decoders, seeds = load_sae_decoders(exact_dir)
    print(f"\nLoaded {len(decoders)} SAEs from {exact_dir}")
    print(f"Seeds: {seeds}")
    print(f"Decoder shape: {decoders[0].shape}")
    
    # Compute pairwise metrics
    n = len(decoders)
    pwmcc_values = []
    subspace_overlaps = []
    
    print("\nPairwise comparisons:")
    print("-" * 50)
    
    for i in range(n):
        for j in range(i + 1, n):
            pwmcc = compute_pwmcc(decoders[i], decoders[j])
            overlap = compute_subspace_overlap(decoders[i], decoders[j])
            
            pwmcc_values.append(pwmcc)
            subspace_overlaps.append(overlap)
            
            print(f"  Seed {seeds[i]} vs {seeds[j]}: PWMCC={pwmcc:.3f}, Subspace={overlap:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    mean_pwmcc = np.mean(pwmcc_values)
    mean_overlap = np.mean(subspace_overlaps)
    
    print(f"\nMean PWMCC (feature overlap):    {mean_pwmcc:.3f} +/- {np.std(pwmcc_values):.3f}")
    print(f"Mean Subspace overlap:           {mean_overlap:.3f} +/- {np.std(subspace_overlaps):.3f}")
    
    print("\n" + "-" * 50)
    print("HYPOTHESIS TEST:")
    print("-" * 50)
    
    if mean_pwmcc < 0.35 and mean_overlap > 0.70:
        print("CONFIRMED: Basis Ambiguity!")
        print("  - Low feature overlap (PWMCC < 0.35)")
        print("  - High subspace overlap (> 0.70)")
        print("  - SAEs learn SAME subspace, DIFFERENT bases")
    elif mean_pwmcc < 0.35 and mean_overlap < 0.50:
        print("REJECTED: Different Subspaces")
        print("  - Low feature overlap AND low subspace overlap")
        print("  - SAEs are learning genuinely different things")
    else:
        print("INCONCLUSIVE")
        print(f"  - PWMCC: {mean_pwmcc:.3f}")
        print(f"  - Subspace: {mean_overlap:.3f}")
    
    # Also test on overparameterized
    print("\n" + "=" * 70)
    print("COMPARISON: Overparameterized SAEs")
    print("=" * 70)
    
    over_dir = RESULTS_DIR / 'synthetic_sparse'
    if over_dir.exists():
        over_decoders, over_seeds = load_sae_decoders(over_dir)
        print(f"\nLoaded {len(over_decoders)} SAEs from {over_dir}")
        
        over_pwmcc = []
        over_overlap = []
        
        for i in range(len(over_decoders)):
            for j in range(i + 1, len(over_decoders)):
                pwmcc = compute_pwmcc(over_decoders[i], over_decoders[j])
                overlap = compute_subspace_overlap(over_decoders[i], over_decoders[j], k=10)
                over_pwmcc.append(pwmcc)
                over_overlap.append(overlap)
        
        print(f"\nOverparameterized (d_sae=64, k=5):")
        print(f"  Mean PWMCC:    {np.mean(over_pwmcc):.3f}")
        print(f"  Mean Subspace: {np.mean(over_overlap):.3f}")
    
    # Save results
    results = {
        'exact_match': {
            'mean_pwmcc': float(mean_pwmcc),
            'std_pwmcc': float(np.std(pwmcc_values)),
            'mean_subspace_overlap': float(mean_overlap),
            'std_subspace_overlap': float(np.std(subspace_overlaps)),
            'hypothesis_confirmed': mean_pwmcc < 0.35 and mean_overlap > 0.70
        }
    }
    
    output_path = RESULTS_DIR / 'basis_ambiguity_validation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
