#!/usr/bin/env python3
"""Compute random baseline PWMCC for validation.

This critical control experiment measures what PWMCC we'd expect
between randomly initialized SAEs (no training). If random SAEs
show ~0.30 PWMCC, our trained SAE finding is meaningless.

Expected result: Random PWMCC should be much lower (~0.1-0.15)
to validate that 0.30 represents learned structure.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/compute_random_baseline.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.simple_sae import TopKSAE, ReLUSAE


def compute_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder weight matrices."""
    # Normalize
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    
    # Cosine similarity matrix
    cos_sim = d1_norm.T @ d2_norm
    
    # Symmetric PWMCC
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    
    return (max_1to2 + max_2to1) / 2


def main():
    print("=" * 70)
    print("RANDOM BASELINE PWMCC COMPUTATION")
    print("=" * 70)
    print()
    print("This experiment measures PWMCC between randomly initialized SAEs")
    print("to establish a baseline for comparison with trained SAEs.")
    print()
    
    # Match actual experimental setup
    d_model = 128
    d_sae = 1024
    k = 32
    n_random_saes = 10  # More than trained to get better estimate
    seeds = list(range(100, 100 + n_random_saes))
    
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  d_sae: {d_sae}")
    print(f"  k: {k}")
    print(f"  n_random_saes: {n_random_saes}")
    print(f"  seeds: {seeds}")
    print()
    
    # Create random SAEs
    print("Creating randomly initialized SAEs...")
    random_saes = []
    for seed in seeds:
        torch.manual_seed(seed)
        sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
        random_saes.append(sae)
    print(f"✅ Created {len(random_saes)} random SAEs")
    print()
    
    # Compute all pairwise PWMCCs
    print("Computing pairwise PWMCCs...")
    n = len(random_saes)
    pwmcc_values = []
    
    for i in range(n):
        for j in range(i + 1, n):
            decoder_i = random_saes[i].decoder.weight.data
            decoder_j = random_saes[j].decoder.weight.data
            pwmcc = compute_pwmcc(decoder_i, decoder_j)
            pwmcc_values.append(pwmcc)
            print(f"  SAE {seeds[i]} vs SAE {seeds[j]}: PWMCC = {pwmcc:.4f}")
    
    pwmcc_array = np.array(pwmcc_values)
    
    # Statistics
    print()
    print("=" * 70)
    print("RANDOM BASELINE RESULTS")
    print("=" * 70)
    print(f"\nNumber of pairs: {len(pwmcc_values)}")
    print(f"Mean PWMCC:      {pwmcc_array.mean():.4f}")
    print(f"Std PWMCC:       {pwmcc_array.std():.4f}")
    print(f"Min PWMCC:       {pwmcc_array.min():.4f}")
    print(f"Max PWMCC:       {pwmcc_array.max():.4f}")
    print(f"Median PWMCC:    {np.median(pwmcc_array):.4f}")
    
    # Compare to trained SAEs
    trained_pwmcc = 0.30  # Our finding
    print()
    print("=" * 70)
    print("COMPARISON TO TRAINED SAEs")
    print("=" * 70)
    print(f"\nTrained SAE PWMCC:  {trained_pwmcc:.4f}")
    print(f"Random SAE PWMCC:   {pwmcc_array.mean():.4f}")
    print(f"Difference:         {trained_pwmcc - pwmcc_array.mean():.4f}")
    print(f"Ratio:              {trained_pwmcc / pwmcc_array.mean():.2f}x")
    
    # Interpretation
    print()
    if pwmcc_array.mean() < 0.20:
        print("✅ GOOD: Random baseline is significantly lower than trained SAEs")
        print("   This validates that 0.30 PWMCC represents learned structure,")
        print("   not just random chance.")
    elif pwmcc_array.mean() < 0.25:
        print("⚠️  MODERATE: Random baseline is somewhat lower than trained SAEs")
        print("   The 0.30 finding has some validity but the margin is small.")
    else:
        print("❌ CONCERNING: Random baseline is close to trained SAE PWMCC")
        print("   This suggests 0.30 may not represent meaningful learning.")
    
    # Also compute theoretical expectation
    print()
    print("=" * 70)
    print("THEORETICAL ANALYSIS")
    print("=" * 70)
    print(f"\nFor random unit vectors in {d_model}-dimensional space:")
    print(f"  Expected |cos_sim| ≈ sqrt(2/π) / sqrt({d_model}) ≈ {np.sqrt(2/np.pi) / np.sqrt(d_model):.4f}")
    print(f"  With {d_sae} features, max over {d_sae} random cosines is higher")
    print(f"  Empirical random baseline: {pwmcc_array.mean():.4f}")
    
    # Save results
    results = {
        'n_random_saes': n_random_saes,
        'd_model': d_model,
        'd_sae': d_sae,
        'seeds': seeds,
        'pwmcc_values': pwmcc_values,
        'mean': float(pwmcc_array.mean()),
        'std': float(pwmcc_array.std()),
        'min': float(pwmcc_array.min()),
        'max': float(pwmcc_array.max()),
        'median': float(np.median(pwmcc_array)),
        'trained_comparison': {
            'trained_pwmcc': trained_pwmcc,
            'difference': float(trained_pwmcc - pwmcc_array.mean()),
            'ratio': float(trained_pwmcc / pwmcc_array.mean())
        }
    }
    
    import json
    output_path = Path('results/analysis/random_baseline.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
