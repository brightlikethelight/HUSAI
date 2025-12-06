#!/usr/bin/env python3
"""Diagnostic script to investigate low Fourier overlap.

This script tests three critical hypotheses:
1. Is the Fourier ground truth correct? (Check transformer learned Fourier)
2. Is the dimension mismatch (113 vs 128) causing low overlap?
3. Is a random SAE baseline significantly different from trained SAEs?

Usage:
    python scripts/diagnose_fourier.py \
        --transformer results/transformer_5000ep/transformer_best.pt \
        --sae results/saes/topk_seed42/sae_final.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.models.simple_sae import TopKSAE
from src.analysis.fourier_validation import get_fourier_basis, compute_fourier_overlap


def test_dimension_mismatch(sae, fourier_basis, modulus=113):
    """Test if dimension mismatch (113 vs 128) explains low overlap.

    Returns:
        dict with overlap scores for different dimension configurations
    """
    print("\n" + "="*60)
    print("TEST 1: DIMENSION MISMATCH HYPOTHESIS")
    print("="*60)

    # Get SAE decoder weights [d_model, d_sae] = [128, 1024]
    decoder = sae.decoder.weight.data  # [128, 1024]

    # Current method (buggy - uses transpose)
    overlap_buggy = compute_fourier_overlap(decoder.T, fourier_basis)

    # Correct method (no transpose)
    overlap_correct = compute_fourier_overlap(decoder, fourier_basis)

    # Truncated method (only use Fourier dimensions)
    decoder_truncated = decoder[:modulus, :]  # [113, 1024]
    overlap_truncated = compute_fourier_overlap(decoder_truncated, fourier_basis)

    # Padded Fourier method (pad Fourier to 128 dims)
    fourier_padded = F.pad(fourier_basis, (0, 128 - modulus, 0, 0))  # [226, 128]
    overlap_padded = compute_fourier_overlap(decoder, fourier_padded)

    # Analyze energy distribution
    norms = decoder.norm(dim=1)  # [128]
    energy_fourier_dims = norms[:modulus].mean().item()
    energy_extra_dims = norms[modulus:].mean().item()
    energy_ratio = energy_extra_dims / energy_fourier_dims

    print(f"\nDecoder shape: {decoder.shape}")
    print(f"Fourier basis shape: {fourier_basis.shape}")
    print(f"\nEnergy distribution:")
    print(f"  Dims [0-{modulus-1}] (Fourier):  {energy_fourier_dims:.4f}")
    print(f"  Dims [{modulus}-127] (Extra):    {energy_extra_dims:.4f}")
    print(f"  Ratio (Extra/Fourier):          {energy_ratio:.1%}")

    print(f"\nOverlap scores:")
    print(f"  Current (buggy transpose):      {overlap_buggy:.4f}")
    print(f"  Correct (no transpose):         {overlap_correct:.4f}")
    print(f"  Truncated to Fourier dims:      {overlap_truncated:.4f}")
    print(f"  Padded Fourier to 128 dims:     {overlap_padded:.4f}")

    if energy_ratio > 0.2:
        print(f"\n⚠️  WARNING: {energy_ratio:.1%} of SAE energy is in non-Fourier dimensions!")
        print(f"   This suggests SAEs are learning features beyond Fourier basis.")

    return {
        'overlap_buggy': overlap_buggy,
        'overlap_correct': overlap_correct,
        'overlap_truncated': overlap_truncated,
        'overlap_padded': overlap_padded,
        'energy_ratio': energy_ratio
    }


def test_random_baseline(sae, fourier_basis, n_trials=5):
    """Compare trained SAE to random SAEs.

    Returns:
        dict with mean/std of random SAE overlap
    """
    print("\n" + "="*60)
    print("TEST 2: RANDOM BASELINE")
    print("="*60)

    random_overlaps = []

    for i in range(n_trials):
        # Create random SAE with same architecture
        random_sae = TopKSAE(d_model=sae.d_model, d_sae=sae.d_sae, k=sae.k)
        random_sae.reset_parameters()  # Random init

        # Compute overlap
        overlap = compute_fourier_overlap(
            random_sae.decoder.weight.data,
            fourier_basis
        )
        random_overlaps.append(overlap)

    mean_random = np.mean(random_overlaps)
    std_random = np.std(random_overlaps)

    # Get trained SAE overlap
    trained_overlap = compute_fourier_overlap(
        sae.decoder.weight.data,
        fourier_basis
    )

    print(f"\nRandom SAE overlap (n={n_trials}):")
    print(f"  Mean: {mean_random:.4f} ± {std_random:.4f}")
    print(f"  Range: [{min(random_overlaps):.4f}, {max(random_overlaps):.4f}]")

    print(f"\nTrained SAE overlap: {trained_overlap:.4f}")
    print(f"Improvement over random: {trained_overlap - mean_random:.4f} ({(trained_overlap/mean_random - 1)*100:+.1f}%)")

    if trained_overlap < mean_random + 2*std_random:
        print("\n❌ CRITICAL: Trained SAE is not significantly better than random!")
        print("   This suggests SAE is NOT learning Fourier structure.")
    else:
        print("\n✅ Trained SAE is significantly better than random.")
        print("   SAE is learning *something*, just not pure Fourier.")

    return {
        'random_mean': mean_random,
        'random_std': std_random,
        'trained': trained_overlap,
        'improvement': trained_overlap - mean_random
    }


def test_transformer_fourier(model, fourier_basis, modulus=113):
    """Check if the transformer itself learned Fourier circuits.

    This validates our ground truth assumption.

    Returns:
        dict with transformer Fourier overlap scores
    """
    print("\n" + "="*60)
    print("TEST 3: TRANSFORMER FOURIER VALIDATION")
    print("="*60)

    # Get embedding matrix for numerical tokens (0 to modulus-1)
    W_E = model.model.embed.W_E.data  # [vocab_size, d_model]
    W_E_numbers = W_E[:modulus, :]  # [113, 128]

    # Compute overlap (using only Fourier dimensions)
    W_E_fourier_dims = W_E_numbers[:, :modulus]  # [113, 113]
    overlap_embed = compute_fourier_overlap(W_E_fourier_dims.T, fourier_basis)

    # Try with all dimensions
    overlap_embed_full = compute_fourier_overlap(W_E_numbers.T, fourier_basis)

    print(f"\nEmbedding matrix analysis:")
    print(f"  W_E shape: {W_E.shape}")
    print(f"  Numerical tokens (0-{modulus-1}): {W_E_numbers.shape}")

    print(f"\nFourier overlap:")
    print(f"  Using Fourier dims [0-{modulus-1}]:  {overlap_embed:.4f}")
    print(f"  Using all dims [0-127]:             {overlap_embed_full:.4f}")

    if overlap_embed > 0.6:
        print("\n✅ EXCELLENT: Transformer embeddings show strong Fourier structure!")
        print("   Ground truth assumption is VALID.")
    elif overlap_embed > 0.4:
        print("\n⚠️  MODERATE: Transformer shows some Fourier structure.")
        print("   Ground truth is partially valid.")
    else:
        print("\n❌ CRITICAL: Transformer does NOT show Fourier structure!")
        print("   Our ground truth assumption may be WRONG.")
        print("   Either:")
        print("   1. Transformer hasn't grokked yet (check training metrics)")
        print("   2. Fourier basis computation is incorrect")
        print("   3. We're extracting from the wrong layer/position")

    return {
        'overlap_fourier_dims': overlap_embed,
        'overlap_full_dims': overlap_embed_full
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose low Fourier overlap in SAEs"
    )
    parser.add_argument(
        '--transformer',
        type=Path,
        required=True,
        help='Path to transformer checkpoint'
    )
    parser.add_argument(
        '--sae',
        type=Path,
        required=True,
        help='Path to trained SAE checkpoint'
    )
    parser.add_argument(
        '--modulus',
        type=int,
        default=113,
        help='Modulus for Fourier basis'
    )

    args = parser.parse_args()

    print("="*60)
    print("FOURIER DIAGNOSTIC SUITE")
    print("="*60)
    print(f"\nTransformer: {args.transformer}")
    print(f"SAE: {args.sae}")
    print(f"Modulus: {args.modulus}")

    # Load models
    print("\nLoading models...")
    model, _ = ModularArithmeticTransformer.load_checkpoint(args.transformer)
    sae = TopKSAE.load(args.sae)

    # Get Fourier basis
    print("Computing Fourier basis...")
    fourier_basis = get_fourier_basis(modulus=args.modulus)
    print(f"Fourier basis shape: {fourier_basis.shape}")

    # Run tests
    results = {}

    # Test 1: Dimension mismatch
    results['dimension_test'] = test_dimension_mismatch(sae, fourier_basis, args.modulus)

    # Test 2: Random baseline
    results['random_test'] = test_random_baseline(sae, fourier_basis)

    # Test 3: Transformer validation
    results['transformer_test'] = test_transformer_fourier(model, fourier_basis, args.modulus)

    # Summary and recommendations
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)

    # Check dimension mismatch
    if results['dimension_test']['energy_ratio'] > 0.3:
        print("\n1. DIMENSION MISMATCH:")
        print(f"   ⚠️  {results['dimension_test']['energy_ratio']:.1%} of SAE energy is in non-Fourier dims")
        print(f"   → Recommendation: Use padded Fourier basis for fair comparison")
        print(f"   → Expected overlap with padding: {results['dimension_test']['overlap_padded']:.4f}")

    # Check random baseline
    improvement = results['random_test']['improvement']
    if improvement < 0.05:
        print("\n2. RANDOM BASELINE:")
        print(f"   ❌ Trained SAE barely beats random (+{improvement:.4f})")
        print(f"   → CRITICAL: SAE is not learning Fourier structure!")
    else:
        print("\n2. RANDOM BASELINE:")
        print(f"   ✅ Trained SAE beats random by {improvement:.4f}")
        print(f"   → SAE is learning structure, but not Fourier")

    # Check transformer
    if results['transformer_test']['overlap_fourier_dims'] < 0.4:
        print("\n3. TRANSFORMER VALIDATION:")
        print(f"   ❌ Transformer overlap: {results['transformer_test']['overlap_fourier_dims']:.4f}")
        print(f"   → CRITICAL: Ground truth assumption may be WRONG!")
        print(f"   → Action: Verify transformer is grokked (check train/val accuracy)")
    else:
        print("\n3. TRANSFORMER VALIDATION:")
        print(f"   ✅ Transformer overlap: {results['transformer_test']['overlap_fourier_dims']:.4f}")
        print(f"   → Ground truth is valid")

    # Overall recommendation
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if results['transformer_test']['overlap_fourier_dims'] < 0.4:
        print("\n🛑 STOP: Investigate transformer training first")
        print("   The transformer may not have learned Fourier circuits.")
        print("   Action: Check training logs, verify grokking occurred")
    elif results['random_test']['improvement'] < 0.05:
        print("\n🛑 STOP: SAE is not learning structure")
        print("   Problem is likely in SAE training (hyperparameters, architecture)")
        print("   Action: Debug SAE training pipeline")
    elif results['dimension_test']['energy_ratio'] > 0.3:
        print("\n✅ PROCEED: Dimension mismatch is main issue")
        print("   SAEs are learning Fourier + extra structure in unused dims")
        print("   Actions:")
        print("   1. Re-compute all overlaps with padded Fourier basis")
        print(f"   2. Expected improvement: {results['dimension_test']['overlap_padded'] - results['dimension_test']['overlap_correct']:.4f}")
        print("   3. Investigate what SAEs learn in dims [113-127]")
    else:
        print("\n⚠️  UNCLEAR: No obvious cause found")
        print("   Actions:")
        print("   1. Try different extraction layer/position")
        print("   2. Test Fourier-aligned initialization (Experiment 2)")
        print("   3. Analyze specific SAE features vs Fourier components")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
