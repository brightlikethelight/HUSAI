#!/usr/bin/env python3
"""Validate Basis Ambiguity Hypothesis - Subspace Overlap Analysis

Background:
-----------
We discovered that SAEs trained on sparse ground truth data achieve:
- Feature-level PWMCC = 0.263 (low overlap at feature level)
- Ground truth recovery = 8.8/10 features (high subspace recovery)

This suggests "basis ambiguity": SAEs learn the CORRECT 10D subspace but
choose DIFFERENT bases within that subspace.

Hypothesis:
-----------
If basis ambiguity is the explanation, then:
1. Subspace overlap between SAE pairs should be HIGH (>0.90)
2. Feature-level PWMCC should remain LOW (~0.26)
3. The gap proves SAEs identify the subspace but not unique features

Mathematical Framework:
-----------------------
For decoder matrices D1, D2 ∈ R^(d_model × d_sae):
1. Extract top-k principal subspaces via SVD
2. Measure subspace overlap: ||U1_k^T U2_k||_F^2 / k
3. Compare to feature-level PWMCC

Expected Outcome:
-----------------
Subspace overlap >> Feature overlap validates identifiability theory:
- Sparse ground truth → unique subspace (high overlap)
- But infinitely many valid bases within subspace (low feature overlap)

Usage:
------
python scripts/validate_subspace_overlap.py \\
    --sae-dir results/synthetic_sparse_exact \\
    --output-file results/synthetic_sparse_exact/subspace_overlap_results.json \\
    --k 10

Author: brightliu@college.harvard.edu
Date: 2025-12-08
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

import torch
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_sae_decoder(checkpoint_path: Path) -> torch.Tensor:
    """Load decoder weight matrix from SAE checkpoint.

    Args:
        checkpoint_path: Path to SAE checkpoint (.pt file)

    Returns:
        decoder: [d_model, d_sae] decoder weight matrix
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = checkpoint['model_state_dict']['decoder.weight']
    return decoder


def compute_subspace_overlap(D1: torch.Tensor, D2: torch.Tensor, k: int) -> float:
    """Compute subspace overlap between two decoder matrices using SVD.

    The subspace overlap measures how much the top-k principal subspaces
    of two matrices overlap. A score of 1.0 means perfect alignment,
    while 0.0 means orthogonal subspaces.

    Mathematical Definition:
    ------------------------
    1. D1, D2 ∈ R^(d_model × d_sae) are decoder matrices
    2. SVD: D_i = U_i Σ_i V_i^T
    3. U_i_k ∈ R^(d_model × k) are top-k left singular vectors
    4. Overlap = ||U1_k^T U2_k||_F^2 / k

    This is the normalized Frobenius norm of the cross-product matrix,
    which measures the total squared cosine similarity between subspaces.

    Args:
        D1: [d_model, d_sae] first decoder matrix
        D2: [d_model, d_sae] second decoder matrix
        k: subspace dimension (number of top singular vectors)

    Returns:
        overlap: subspace overlap score in [0, 1]
    """
    # SVD to extract principal subspaces
    # Note: torch.svd is deprecated, use torch.linalg.svd
    U1, S1, Vh1 = torch.linalg.svd(D1, full_matrices=False)
    U2, S2, Vh2 = torch.linalg.svd(D2, full_matrices=False)

    # Extract top-k principal components
    U1_k = U1[:, :k]  # [d_model, k]
    U2_k = U2[:, :k]  # [d_model, k]

    # Compute subspace overlap
    # ||U1_k^T U2_k||_F^2 / k measures normalized squared overlap
    cross_product = U1_k.T @ U2_k  # [k, k]
    overlap = cross_product.pow(2).sum().item() / k

    return overlap


def compute_feature_level_pwmcc(D1: torch.Tensor, D2: torch.Tensor, top_k: int = None) -> float:
    """Compute feature-level pairwise maximum cosine correlation (PWMCC).

    This is the standard PWMCC metric that measures feature-level overlap
    by finding the best matching features via maximum cosine similarity.

    Args:
        D1: [d_model, d_sae] first decoder matrix
        D2: [d_model, d_sae] second decoder matrix
        top_k: if provided, only use top-k features (by norm)

    Returns:
        pwmcc: pairwise maximum cosine correlation
    """
    # Normalize decoder columns
    D1_norm = torch.nn.functional.normalize(D1, dim=0)  # [d_model, d_sae]
    D2_norm = torch.nn.functional.normalize(D2, dim=0)

    # Optionally filter to top-k features by norm
    if top_k is not None:
        norms1 = D1.norm(dim=0)
        norms2 = D2.norm(dim=0)
        top_k1 = norms1.topk(top_k).indices
        top_k2 = norms2.topk(top_k).indices
        D1_norm = D1_norm[:, top_k1]
        D2_norm = D2_norm[:, top_k2]

    # Compute cosine similarity matrix
    cos_sim = D1_norm.T @ D2_norm  # [d_sae, d_sae]

    # PWMCC: average of maximum similarities in both directions
    max_sim_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_sim_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    pwmcc = (max_sim_1to2 + max_sim_2to1) / 2

    return pwmcc


def compute_principal_angles(D1: torch.Tensor, D2: torch.Tensor, k: int) -> List[float]:
    """Compute principal angles between two subspaces.

    Principal angles θ_i ∈ [0, π/2] measure the canonical angles between
    two subspaces. Small angles indicate high alignment.

    Args:
        D1: [d_model, d_sae] first decoder matrix
        D2: [d_model, d_sae] second decoder matrix
        k: subspace dimension

    Returns:
        angles: list of k principal angles in radians
    """
    # Extract top-k subspaces
    U1, _, _ = torch.linalg.svd(D1, full_matrices=False)
    U2, _, _ = torch.linalg.svd(D2, full_matrices=False)
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    # Compute SVD of cross-product: U1_k^T U2_k
    _, singular_values, _ = torch.linalg.svd(U1_k.T @ U2_k, full_matrices=False)

    # Principal angles: θ_i = arccos(σ_i) where σ_i are singular values
    # Clamp to [0, 1] to handle numerical errors
    singular_values = torch.clamp(singular_values, 0, 1)
    angles = torch.acos(singular_values).tolist()

    return angles


def analyze_subspace_structure(D: torch.Tensor, k: int) -> Dict:
    """Analyze the structure of a single decoder's subspace.

    Args:
        D: [d_model, d_sae] decoder matrix
        k: subspace dimension

    Returns:
        analysis: dict with singular value spectrum and statistics
    """
    _, singular_values, _ = torch.linalg.svd(D, full_matrices=False)

    # Compute explained variance ratio
    total_variance = singular_values.pow(2).sum()
    variance_explained = (singular_values[:k].pow(2).sum() / total_variance).item()

    # Compute effective rank (using entropy)
    probs = singular_values.pow(2) / total_variance
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    effective_rank = np.exp(entropy)

    return {
        'singular_values': singular_values.tolist(),
        'top_k_singular_values': singular_values[:k].tolist(),
        'variance_explained_by_top_k': variance_explained,
        'effective_rank': effective_rank,
        'condition_number': (singular_values[0] / (singular_values[-1] + 1e-10)).item()
    }


def validate_subspace_overlap(
    sae_dir: Path,
    output_file: Path,
    k: int = 10,
    verbose: bool = True
) -> Dict:
    """Main validation function for basis ambiguity hypothesis.

    Args:
        sae_dir: directory containing SAE checkpoints
        output_file: path to save results JSON
        k: subspace dimension (should match n_true_features)
        verbose: whether to print detailed progress

    Returns:
        results: comprehensive analysis results
    """
    if verbose:
        print("=" * 80)
        print("SUBSPACE OVERLAP VALIDATION - Basis Ambiguity Hypothesis")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  SAE directory: {sae_dir}")
        print(f"  Subspace dimension (k): {k}")
        print(f"  Output file: {output_file}")

    # Load all SAE checkpoints
    sae_files = sorted(sae_dir.glob("sae_seed_*.pt"))
    if len(sae_files) == 0:
        raise ValueError(f"No SAE checkpoints found in {sae_dir}")

    if verbose:
        print(f"\n✅ Found {len(sae_files)} SAE checkpoints:")
        for f in sae_files:
            print(f"     {f.name}")

    # Load decoder matrices
    decoders = {}
    for sae_file in sae_files:
        seed = int(sae_file.stem.split('_')[-1])
        decoder = load_sae_decoder(sae_file)
        decoders[seed] = decoder

        if verbose:
            print(f"\n  Loaded seed {seed}: decoder shape = {decoder.shape}")

    seeds = sorted(decoders.keys())

    # Analyze individual decoder structures
    if verbose:
        print("\n" + "=" * 80)
        print("INDIVIDUAL DECODER ANALYSIS")
        print("=" * 80)

    decoder_analyses = {}
    for seed in seeds:
        analysis = analyze_subspace_structure(decoders[seed], k)
        decoder_analyses[seed] = analysis

        if verbose:
            print(f"\nSeed {seed}:")
            print(f"  Top-{k} variance explained: {analysis['variance_explained_by_top_k']:.4f}")
            print(f"  Effective rank: {analysis['effective_rank']:.2f}")
            print(f"  Condition number: {analysis['condition_number']:.2e}")
            print(f"  Top-5 singular values: {analysis['top_k_singular_values'][:5]}")

    # Compute pairwise subspace overlap
    if verbose:
        print("\n" + "=" * 80)
        print("PAIRWISE SUBSPACE OVERLAP ANALYSIS")
        print("=" * 80)

    subspace_overlaps = []
    feature_pwmccs = []
    principal_angles_all = []

    pairs = list(itertools.combinations(seeds, 2))

    for i, (seed1, seed2) in enumerate(pairs):
        D1 = decoders[seed1]
        D2 = decoders[seed2]

        # Compute subspace overlap
        overlap = compute_subspace_overlap(D1, D2, k)
        subspace_overlaps.append(overlap)

        # Compute feature-level PWMCC (using top-k features)
        pwmcc = compute_feature_level_pwmcc(D1, D2, top_k=k)
        feature_pwmccs.append(pwmcc)

        # Compute principal angles
        angles = compute_principal_angles(D1, D2, k)
        principal_angles_all.append(angles)

        if verbose:
            print(f"\nPair {i+1}/{len(pairs)}: Seed {seed1} vs Seed {seed2}")
            print(f"  Subspace overlap:     {overlap:.4f}")
            print(f"  Feature-level PWMCC:  {pwmcc:.4f}")
            print(f"  Gap (overlap - PWMCC): {overlap - pwmcc:.4f}")
            print(f"  Max principal angle:   {np.rad2deg(max(angles)):.2f}°")
            print(f"  Mean principal angle:  {np.rad2deg(np.mean(angles)):.2f}°")

    # Compute statistics
    subspace_overlap_mean = np.mean(subspace_overlaps)
    subspace_overlap_std = np.std(subspace_overlaps)
    feature_pwmcc_mean = np.mean(feature_pwmccs)
    feature_pwmcc_std = np.std(feature_pwmccs)
    gap_mean = subspace_overlap_mean - feature_pwmcc_mean

    # Summary statistics
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nSubspace Overlap (k={k}):")
        print(f"  Mean: {subspace_overlap_mean:.4f} ± {subspace_overlap_std:.4f}")
        print(f"  Min:  {min(subspace_overlaps):.4f}")
        print(f"  Max:  {max(subspace_overlaps):.4f}")

        print(f"\nFeature-level PWMCC (top-{k} features):")
        print(f"  Mean: {feature_pwmcc_mean:.4f} ± {feature_pwmcc_std:.4f}")
        print(f"  Min:  {min(feature_pwmccs):.4f}")
        print(f"  Max:  {max(feature_pwmccs):.4f}")

        print(f"\nGap Analysis:")
        print(f"  Mean gap: {gap_mean:.4f}")
        print(f"  Relative gap: {gap_mean / feature_pwmcc_mean * 100:.1f}%")

    # Hypothesis validation
    hypothesis_validated = subspace_overlap_mean > 0.90 and feature_pwmcc_mean < 0.35

    if verbose:
        print("\n" + "=" * 80)
        print("HYPOTHESIS VALIDATION")
        print("=" * 80)
        print(f"\nBasis Ambiguity Hypothesis:")
        print(f"  SAEs learn correct subspace but different bases")

        print(f"\nCriterion 1: Subspace overlap > 0.90")
        print(f"  Measured: {subspace_overlap_mean:.4f}")
        print(f"  Status: {'✅ PASS' if subspace_overlap_mean > 0.90 else '❌ FAIL'}")

        print(f"\nCriterion 2: Feature-level PWMCC < 0.35")
        print(f"  Measured: {feature_pwmcc_mean:.4f}")
        print(f"  Status: {'✅ PASS' if feature_pwmcc_mean < 0.35 else '❌ FAIL'}")

        print(f"\nOverall Result: {'✅ HYPOTHESIS VALIDATED' if hypothesis_validated else '❌ HYPOTHESIS REJECTED'}")

        if hypothesis_validated:
            print("\nInterpretation:")
            print("  SAEs successfully identify the unique 10D subspace (high overlap)")
            print("  But choose arbitrary orthonormal bases within it (low feature match)")
            print("  This confirms identifiability theory: subspace is unique, basis is not")

    # Compile results
    results = {
        'experiment': 'subspace_overlap_validation',
        'hypothesis': 'basis_ambiguity',
        'configuration': {
            'sae_dir': str(sae_dir),
            'k': k,
            'n_saes': len(seeds),
            'seeds': seeds
        },
        'individual_analyses': {
            str(seed): analysis for seed, analysis in decoder_analyses.items()
        },
        'pairwise_results': [
            {
                'seed1': int(seed1),
                'seed2': int(seed2),
                'subspace_overlap': float(overlap),
                'feature_pwmcc': float(pwmcc),
                'gap': float(overlap - pwmcc),
                'principal_angles_deg': [float(np.rad2deg(a)) for a in angles]
            }
            for (seed1, seed2), overlap, pwmcc, angles
            in zip(pairs, subspace_overlaps, feature_pwmccs, principal_angles_all)
        ],
        'statistics': {
            'subspace_overlap': {
                'mean': float(subspace_overlap_mean),
                'std': float(subspace_overlap_std),
                'min': float(min(subspace_overlaps)),
                'max': float(max(subspace_overlaps)),
                'all_values': [float(x) for x in subspace_overlaps]
            },
            'feature_pwmcc': {
                'mean': float(feature_pwmcc_mean),
                'std': float(feature_pwmcc_std),
                'min': float(min(feature_pwmccs)),
                'max': float(max(feature_pwmccs)),
                'all_values': [float(x) for x in feature_pwmccs]
            },
            'gap': {
                'mean': float(gap_mean),
                'relative_percent': float(gap_mean / feature_pwmcc_mean * 100)
            }
        },
        'validation': {
            'hypothesis': 'basis_ambiguity',
            'criterion_1': {
                'name': 'high_subspace_overlap',
                'threshold': 0.90,
                'measured': float(subspace_overlap_mean),
                'passed': bool(subspace_overlap_mean > 0.90)
            },
            'criterion_2': {
                'name': 'low_feature_pwmcc',
                'threshold': 0.35,
                'measured': float(feature_pwmcc_mean),
                'passed': bool(feature_pwmcc_mean < 0.35)
            },
            'overall': bool(hypothesis_validated)
        },
        'interpretation': {
            'subspace_identifiable': bool(subspace_overlap_mean > 0.90),
            'features_identifiable': bool(feature_pwmcc_mean > 0.90),
            'conclusion': 'basis_ambiguity_confirmed' if hypothesis_validated else 'hypothesis_rejected'
        }
    }

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n✅ Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate basis ambiguity hypothesis via subspace overlap analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--sae-dir',
        type=Path,
        default=Path('results/synthetic_sparse_exact'),
        help='Directory containing SAE checkpoints (default: results/synthetic_sparse_exact)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('results/synthetic_sparse_exact/subspace_overlap_results.json'),
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Subspace dimension (should match n_true_features, default: 10)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run validation
    results = validate_subspace_overlap(
        sae_dir=args.sae_dir,
        output_file=args.output_file,
        k=args.k,
        verbose=not args.quiet
    )

    # Print final summary
    if not args.quiet:
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nHypothesis: {'✅ VALIDATED' if results['validation']['overall'] else '❌ REJECTED'}")
        print(f"Subspace overlap: {results['statistics']['subspace_overlap']['mean']:.4f}")
        print(f"Feature PWMCC:    {results['statistics']['feature_pwmcc']['mean']:.4f}")
        print(f"Gap:              {results['statistics']['gap']['mean']:.4f}")


if __name__ == '__main__':
    main()
