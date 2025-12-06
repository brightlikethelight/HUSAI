#!/usr/bin/env python3
"""CRITICAL VERIFICATION: Compare trained SAE vs random SAE PWMCC.

This script directly tests the hypothesis that trained SAEs perform no better
than random SAEs. If PWMCC(trained) ≈ PWMCC(random), then training is ineffective.

Expected outcomes:
- If training works: trained_PWMCC >> random_PWMCC (e.g., 0.70 vs 0.30)
- If training fails: trained_PWMCC ≈ random_PWMCC (e.g., 0.30 vs 0.30)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/verify_random_baseline.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_sae import TopKSAE


def compute_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder weight matrices.

    Args:
        decoder1: Decoder weights [d_model, d_sae]
        decoder2: Decoder weights [d_model, d_sae]

    Returns:
        PWMCC score (0-1)
    """
    # Normalize each feature column to unit norm
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)

    # Cosine similarity matrix [d_sae, d_sae]
    cos_sim = d1_norm.T @ d2_norm

    # Symmetric PWMCC: average of max in both directions
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()

    return (max_1to2 + max_2to1) / 2


def load_trained_sae(sae_path: Path) -> TopKSAE:
    """Load a trained SAE from checkpoint (handles old checkpoint format)."""
    checkpoint = torch.load(sae_path, map_location='cpu')

    # Create SAE with architecture from checkpoint
    sae = TopKSAE(
        d_model=checkpoint['d_model'],
        d_sae=checkpoint['d_sae'],
        k=checkpoint['k'],
        aux_loss_coef=checkpoint.get('aux_loss_coef', 1.0 / checkpoint['k']),  # Default if missing
        dead_threshold=checkpoint.get('dead_threshold', 10)
    )

    # Load weights (handle old format with decoder.bias)
    state_dict = checkpoint['model_state_dict']
    # Remove decoder.bias if present (old format had it, current doesn't)
    if 'decoder.bias' in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != 'decoder.bias'}

    sae.load_state_dict(state_dict, strict=False)

    # Load feature counts if available
    if 'feature_counts' in checkpoint:
        sae.feature_counts = checkpoint['feature_counts']

    return sae


def create_random_sae(d_model: int, d_sae: int, k: int, seed: int) -> TopKSAE:
    """Create a random SAE with the same architecture."""
    torch.manual_seed(seed)
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    return sae


def main():
    print("=" * 80)
    print("CRITICAL VERIFICATION: TRAINED vs RANDOM SAE PWMCC")
    print("=" * 80)
    print()
    print("Research Question: Do trained SAEs learn stable features,")
    print("                   or is 0.30 PWMCC just random chance?")
    print()
    print("Hypothesis Test:")
    print("  H0 (null): PWMCC_trained = PWMCC_random (training is useless)")
    print("  H1 (alt):  PWMCC_trained > PWMCC_random (training works)")
    print()
    print("=" * 80)
    print()

    # Load trained SAEs
    BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
    RESULTS_DIR = BASE_DIR / 'results'

    # Use cross_layer_validation SAEs (Layer 0, well-documented)
    CROSS_LAYER_DIR = RESULTS_DIR / 'cross_layer_validation'
    trained_seeds = [42, 123, 456, 789, 1011]

    print("STEP 1: Loading trained SAEs")
    print("-" * 80)
    trained_saes = {}
    for seed in trained_seeds:
        sae_path = CROSS_LAYER_DIR / f'layer0_seed{seed}.pt'
        if sae_path.exists():
            sae = load_trained_sae(sae_path)
            trained_saes[seed] = sae
            print(f"  ✓ Loaded trained SAE (seed={seed})")
            print(f"    Architecture: d_model={sae.d_model}, d_sae={sae.d_sae}, k={sae.k}")

            # Check feature usage
            n_active = (sae.feature_counts > 0).sum().item()
            n_total = sae.d_sae
            print(f"    Features: {n_active}/{n_total} active ({100*n_active/n_total:.1f}%)")

    if not trained_saes:
        print("ERROR: No trained SAEs found!")
        return

    # Get architecture from first SAE
    first_sae = list(trained_saes.values())[0]
    d_model = first_sae.d_model
    d_sae = first_sae.d_sae
    k = first_sae.k

    print()
    print("STEP 2: Creating random SAEs")
    print("-" * 80)
    # Create random SAEs with DIFFERENT seeds (1000+ to avoid overlap)
    random_seeds = [1000 + i for i in range(len(trained_seeds))]
    random_saes = {}
    for seed in random_seeds:
        sae = create_random_sae(d_model=d_model, d_sae=d_sae, k=k, seed=seed)
        random_saes[seed] = sae
        print(f"  ✓ Created random SAE (seed={seed})")
        print(f"    Architecture: d_model={sae.d_model}, d_sae={sae.d_sae}, k={sae.k}")

    print()
    print("STEP 3: Computing PWMCC for TRAINED SAE pairs")
    print("-" * 80)
    trained_pwmcc_values = []
    trained_seed_list = list(trained_saes.keys())

    for i, seed1 in enumerate(trained_seed_list):
        for seed2 in trained_seed_list[i+1:]:
            decoder1 = trained_saes[seed1].decoder.weight.data
            decoder2 = trained_saes[seed2].decoder.weight.data
            pwmcc = compute_pwmcc(decoder1, decoder2)
            trained_pwmcc_values.append(pwmcc)
            print(f"  Trained seed {seed1} vs {seed2}: PWMCC = {pwmcc:.4f}")

    trained_pwmcc = np.array(trained_pwmcc_values)

    print()
    print("STEP 4: Computing PWMCC for RANDOM SAE pairs")
    print("-" * 80)
    random_pwmcc_values = []
    random_seed_list = list(random_saes.keys())

    for i, seed1 in enumerate(random_seed_list):
        for seed2 in random_seed_list[i+1:]:
            decoder1 = random_saes[seed1].decoder.weight.data
            decoder2 = random_saes[seed2].decoder.weight.data
            pwmcc = compute_pwmcc(decoder1, decoder2)
            random_pwmcc_values.append(pwmcc)
            print(f"  Random seed {seed1} vs {seed2}: PWMCC = {pwmcc:.4f}")

    random_pwmcc = np.array(random_pwmcc_values)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"TRAINED SAEs (n={len(trained_pwmcc)} pairs):")
    print(f"  Mean:   {trained_pwmcc.mean():.4f}")
    print(f"  Std:    {trained_pwmcc.std():.4f}")
    print(f"  Min:    {trained_pwmcc.min():.4f}")
    print(f"  Max:    {trained_pwmcc.max():.4f}")
    print(f"  Median: {np.median(trained_pwmcc):.4f}")
    print()
    print(f"RANDOM SAEs (n={len(random_pwmcc)} pairs):")
    print(f"  Mean:   {random_pwmcc.mean():.4f}")
    print(f"  Std:    {random_pwmcc.std():.4f}")
    print(f"  Min:    {random_pwmcc.min():.4f}")
    print(f"  Max:    {random_pwmcc.max():.4f}")
    print(f"  Median: {np.median(random_pwmcc):.4f}")
    print()
    print(f"DIFFERENCE:")
    print(f"  Trained - Random: {trained_pwmcc.mean() - random_pwmcc.mean():.4f}")
    print(f"  Effect size (Cohen's d): {(trained_pwmcc.mean() - random_pwmcc.mean()) / np.sqrt((trained_pwmcc.std()**2 + random_pwmcc.std()**2) / 2):.4f}")

    # Statistical test
    print()
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 80)
    print()

    # Mann-Whitney U test (non-parametric, robust to non-normality)
    statistic, p_value = stats.mannwhitneyu(trained_pwmcc, random_pwmcc, alternative='greater')

    print(f"Mann-Whitney U test (one-sided):")
    print(f"  H0: PWMCC_trained = PWMCC_random")
    print(f"  H1: PWMCC_trained > PWMCC_random")
    print(f"  U-statistic: {statistic:.2f}")
    print(f"  p-value: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"  ✓ REJECT H0 (p < {alpha})")
        print(f"    Trained SAEs have significantly higher PWMCC than random.")
    else:
        print(f"  ✗ FAIL TO REJECT H0 (p >= {alpha})")
        print(f"    Trained SAEs do NOT have significantly higher PWMCC than random.")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    difference = trained_pwmcc.mean() - random_pwmcc.mean()

    if difference > 0.10 and p_value < 0.01:
        print("✓ TRAINING WORKS")
        print(f"  Trained PWMCC ({trained_pwmcc.mean():.4f}) is significantly higher")
        print(f"  than random ({random_pwmcc.mean():.4f}), difference = {difference:.4f}")
        print(f"  SAEs are learning stable, reproducible features.")
    elif difference > 0.05 and p_value < 0.05:
        print("⚠️  WEAK EVIDENCE FOR TRAINING")
        print(f"  Trained PWMCC ({trained_pwmcc.mean():.4f}) is slightly higher")
        print(f"  than random ({random_pwmcc.mean():.4f}), difference = {difference:.4f}")
        print(f"  Effect size is small. Training may have limited effectiveness.")
    else:
        print("✗ TRAINING DOES NOT WORK")
        print(f"  Trained PWMCC ({trained_pwmcc.mean():.4f}) is NOT significantly")
        print(f"  different from random ({random_pwmcc.mean():.4f}), difference = {difference:.4f}")
        print(f"  CRITICAL FINDING: SAEs are NOT learning stable features!")
        print(f"  The 0.30 PWMCC represents random chance, not learned structure.")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("What does this mean?")
    print()
    if difference <= 0.05:
        print("1. PWMCC ≈ 0.30 is the RANDOM BASELINE for this architecture")
        print("   (d_model=128, d_sae=1024, k=32)")
        print()
        print("2. Training does NOT improve feature stability")
        print("   Different random seeds produce incompatible feature sets")
        print()
        print("3. This is a MAJOR finding:")
        print("   - SAEs may be fitting noise rather than structure")
        print("   - Feature interpretability claims are questionable")
        print("   - Reproducibility across seeds is impossible")
        print()
        print("4. Possible explanations:")
        print("   - Underconstraining: many equally good solutions")
        print("   - Rotation invariance: features rotate arbitrarily")
        print("   - Insufficient training signal from synthetic data")
        print("   - Architecture choice (TopK vs ReLU)")
    else:
        print("1. Training DOES improve feature stability")
        print(f"   Trained PWMCC is {difference:.4f} higher than random")
        print()
        print("2. SAEs learn reproducible features across seeds")
        print()
        print("3. Feature interpretability is more reliable")
        print()
        print("4. Further investigation needed:")
        print("   - Why is trained PWMCC still only ~0.30-0.40?")
        print("   - Can we improve stability further?")
        print("   - What causes remaining instability?")

    # Save results
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    results = {
        'architecture': {
            'd_model': d_model,
            'd_sae': d_sae,
            'k': k
        },
        'trained': {
            'seeds': trained_seed_list,
            'n_pairs': len(trained_pwmcc_values),
            'pwmcc_values': trained_pwmcc_values,
            'mean': float(trained_pwmcc.mean()),
            'std': float(trained_pwmcc.std()),
            'min': float(trained_pwmcc.min()),
            'max': float(trained_pwmcc.max()),
            'median': float(np.median(trained_pwmcc))
        },
        'random': {
            'seeds': random_seed_list,
            'n_pairs': len(random_pwmcc_values),
            'pwmcc_values': random_pwmcc_values,
            'mean': float(random_pwmcc.mean()),
            'std': float(random_pwmcc.std()),
            'min': float(random_pwmcc.min()),
            'max': float(random_pwmcc.max()),
            'median': float(np.median(random_pwmcc))
        },
        'comparison': {
            'difference': float(difference),
            'effect_size_cohens_d': float((trained_pwmcc.mean() - random_pwmcc.mean()) /
                                         np.sqrt((trained_pwmcc.std()**2 + random_pwmcc.std()**2) / 2))
        },
        'statistical_test': {
            'test': 'Mann-Whitney U (one-sided)',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'reject_null': bool(p_value < alpha)
        },
        'conclusion': 'training_works' if (difference > 0.05 and p_value < 0.05) else 'training_fails'
    }

    import json
    output_path = Path('results/analysis/trained_vs_random_pwmcc.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")

    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
