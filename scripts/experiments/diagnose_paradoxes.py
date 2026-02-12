#!/usr/bin/env python3
"""
COMPREHENSIVE PARADOX DIAGNOSIS

This script investigates the three paradoxes discovered:
1. 88% ground truth recovery with 14% subspace overlap
2. 10th singular value drop
3. Gated architecture opposite trend

Key Questions:
- Is the ground truth recovery metric buggy?
- Is the subspace overlap calculation correct?
- What's the actual relationship between these metrics?
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'


def load_sae_decoders(results_dir: Path) -> Tuple[List[torch.Tensor], List[int]]:
    """Load decoder matrices from saved SAEs."""
    decoders = []
    seeds = []
    for sae_file in sorted(results_dir.glob('sae_seed_*.pt')):
        checkpoint = torch.load(sae_file, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            decoder = checkpoint['model_state_dict']['decoder.weight']
        else:
            decoder = checkpoint['decoder.weight']
        decoders.append(decoder)
        seed = int(sae_file.stem.split('_')[-1])
        seeds.append(seed)
    return decoders, seeds


def load_ground_truth(results_dir: Path) -> torch.Tensor:
    """Load ground truth features if saved."""
    gt_path = results_dir / 'ground_truth_features.pt'
    if gt_path.exists():
        return torch.load(gt_path, map_location='cpu')
    return None


# =============================================================================
# PARADOX 1: Ground Truth Recovery Metric Investigation
# =============================================================================

def compute_ground_truth_recovery_ORIGINAL(
    decoder: torch.Tensor,
    true_features: torch.Tensor,
    threshold: float = 0.9
) -> Tuple[int, float, torch.Tensor]:
    """ORIGINAL (potentially buggy) ground truth recovery."""
    # Original code normalizes along dim=1 (d_model)
    decoder_norm = F.normalize(decoder, dim=1)  # BUG? Should be dim=0?
    true_features_norm = F.normalize(true_features, dim=0)
    
    cos_sim = true_features_norm.T @ decoder_norm  # [k_true, d_sae]
    max_similarities = cos_sim.abs().max(dim=1)[0]
    n_recovered = (max_similarities > threshold).sum().item()
    mean_max_similarity = max_similarities.mean().item()
    
    return n_recovered, mean_max_similarity, max_similarities


def compute_ground_truth_recovery_FIXED(
    decoder: torch.Tensor,
    true_features: torch.Tensor,
    threshold: float = 0.9
) -> Tuple[int, float, torch.Tensor]:
    """FIXED ground truth recovery - normalize features correctly."""
    # Decoder shape: [d_model, d_sae]
    # Each column is a feature direction, so normalize along dim=0
    decoder_norm = F.normalize(decoder, dim=0)  # FIXED: normalize columns
    true_features_norm = F.normalize(true_features, dim=0)
    
    cos_sim = true_features_norm.T @ decoder_norm  # [k_true, d_sae]
    max_similarities = cos_sim.abs().max(dim=1)[0]
    n_recovered = (max_similarities > threshold).sum().item()
    mean_max_similarity = max_similarities.mean().item()
    
    return n_recovered, mean_max_similarity, max_similarities


def compute_ground_truth_recovery_STRICT(
    decoder: torch.Tensor,
    true_features: torch.Tensor,
    thresholds: List[float] = [0.7, 0.8, 0.9, 0.95, 0.99]
) -> Dict:
    """Compute recovery at multiple thresholds."""
    decoder_norm = F.normalize(decoder, dim=0)
    true_features_norm = F.normalize(true_features, dim=0)
    
    cos_sim = true_features_norm.T @ decoder_norm
    max_similarities = cos_sim.abs().max(dim=1)[0]
    
    results = {
        'max_similarities': max_similarities.tolist(),
        'mean_max_sim': max_similarities.mean().item(),
        'std_max_sim': max_similarities.std().item(),
        'min_max_sim': max_similarities.min().item(),
        'max_max_sim': max_similarities.max().item(),
    }
    
    for thresh in thresholds:
        n_recovered = (max_similarities > thresh).sum().item()
        results[f'recovered_at_{thresh}'] = n_recovered
    
    return results


# =============================================================================
# PARADOX 2: Subspace Overlap Investigation
# =============================================================================

def compute_subspace_overlap_detailed(
    D1: torch.Tensor,
    D2: torch.Tensor,
    k_values: List[int] = None
) -> Dict:
    """Compute subspace overlap at multiple k values."""
    U1, S1, _ = torch.svd(D1)
    U2, S2, _ = torch.svd(D2)
    
    if k_values is None:
        max_k = min(D1.shape[1], D2.shape[1])
        k_values = list(range(1, max_k + 1))
    
    results = {
        'singular_values_1': S1.tolist(),
        'singular_values_2': S2.tolist(),
        'overlaps': {}
    }
    
    for k in k_values:
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        
        # Subspace overlap: ||U1_k^T @ U2_k||_F^2 / k
        overlap = (U1_k.T @ U2_k).pow(2).sum() / k
        results['overlaps'][k] = overlap.item()
    
    return results


def analyze_singular_value_structure(decoder: torch.Tensor) -> Dict:
    """Analyze singular value structure of decoder."""
    U, S, V = torch.svd(decoder)
    
    # Compute variance explained
    total_var = (S ** 2).sum()
    var_explained = (S ** 2) / total_var
    cumsum_var = var_explained.cumsum(0)
    
    # Find effective ranks
    eff_rank_90 = (cumsum_var < 0.9).sum().item() + 1
    eff_rank_95 = (cumsum_var < 0.95).sum().item() + 1
    eff_rank_99 = (cumsum_var < 0.99).sum().item() + 1
    
    # Analyze gaps between consecutive singular values
    gaps = (S[:-1] - S[1:]) / S[:-1]  # Relative gaps
    
    return {
        'singular_values': S.tolist(),
        'variance_explained': var_explained.tolist(),
        'cumsum_variance': cumsum_var.tolist(),
        'effective_rank_90': eff_rank_90,
        'effective_rank_95': eff_rank_95,
        'effective_rank_99': eff_rank_99,
        'relative_gaps': gaps.tolist(),
        'max_gap_index': gaps.argmax().item(),
        'max_gap_value': gaps.max().item(),
    }


# =============================================================================
# PARADOX 3: PWMCC vs Subspace Overlap Relationship
# =============================================================================

def compute_pwmcc(D1: torch.Tensor, D2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder matrices."""
    D1_norm = F.normalize(D1, dim=0)
    D2_norm = F.normalize(D2, dim=0)
    cos_sim = D1_norm.T @ D2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def analyze_feature_matching(D1: torch.Tensor, D2: torch.Tensor) -> Dict:
    """Detailed analysis of feature matching between two SAEs."""
    D1_norm = F.normalize(D1, dim=0)
    D2_norm = F.normalize(D2, dim=0)
    cos_sim = D1_norm.T @ D2_norm  # [d_sae, d_sae]
    
    # Best matches in each direction
    max_1to2, idx_1to2 = cos_sim.abs().max(dim=1)
    max_2to1, idx_2to1 = cos_sim.abs().max(dim=0)
    
    # Check for one-to-one matching
    unique_matches_1to2 = len(set(idx_1to2.tolist()))
    unique_matches_2to1 = len(set(idx_2to1.tolist()))
    
    return {
        'pwmcc': (max_1to2.mean().item() + max_2to1.mean().item()) / 2,
        'mean_max_1to2': max_1to2.mean().item(),
        'mean_max_2to1': max_2to1.mean().item(),
        'std_max_1to2': max_1to2.std().item(),
        'std_max_2to1': max_2to1.std().item(),
        'unique_matches_1to2': unique_matches_1to2,
        'unique_matches_2to1': unique_matches_2to1,
        'd_sae': D1.shape[1],
        'pct_unique_1to2': unique_matches_1to2 / D1.shape[1] * 100,
        'pct_unique_2to1': unique_matches_2to1 / D2.shape[1] * 100,
        'distribution_1to2': {
            'min': max_1to2.min().item(),
            'max': max_1to2.max().item(),
            'median': max_1to2.median().item(),
            'pct_above_0.9': (max_1to2 > 0.9).float().mean().item() * 100,
            'pct_above_0.8': (max_1to2 > 0.8).float().mean().item() * 100,
            'pct_above_0.5': (max_1to2 > 0.5).float().mean().item() * 100,
        }
    }


# =============================================================================
# MAIN DIAGNOSIS
# =============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE PARADOX DIAGNOSIS")
    print("=" * 80)
    
    # Load exact match experiment data
    exact_dir = RESULTS_DIR / 'synthetic_sparse_exact'
    if not exact_dir.exists():
        print(f"ERROR: {exact_dir} not found")
        return
    
    decoders, seeds = load_sae_decoders(exact_dir)
    print(f"\nLoaded {len(decoders)} SAEs from {exact_dir}")
    print(f"Seeds: {seeds}")
    print(f"Decoder shape: {decoders[0].shape}")
    
    # Load results to get ground truth info
    with open(exact_dir / 'results.json') as f:
        results = json.load(f)
    
    n_true_features = results['configuration']['n_true_features']
    d_model = results['configuration']['d_model']
    
    print(f"\nExperiment config:")
    print(f"  n_true_features: {n_true_features}")
    print(f"  d_model: {d_model}")
    print(f"  d_sae: {results['configuration']['d_sae']}")
    print(f"  k: {results['configuration']['k']}")
    
    # ==========================================================================
    # DIAGNOSIS 1: Singular Value Structure
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS 1: Singular Value Structure")
    print("=" * 80)
    
    for i, decoder in enumerate(decoders):
        sv_analysis = analyze_singular_value_structure(decoder)
        print(f"\nSeed {seeds[i]}:")
        print(f"  Singular values: {[f'{s:.3f}' for s in sv_analysis['singular_values'][:12]]}")
        print(f"  Effective rank (90%): {sv_analysis['effective_rank_90']}")
        print(f"  Effective rank (95%): {sv_analysis['effective_rank_95']}")
        print(f"  Max gap at index: {sv_analysis['max_gap_index']} (gap={sv_analysis['max_gap_value']:.3f})")
        
        # Check for 10th singular value drop
        if len(sv_analysis['singular_values']) >= 10:
            s9 = sv_analysis['singular_values'][8]
            s10 = sv_analysis['singular_values'][9]
            drop = (s9 - s10) / s9 * 100
            print(f"  σ9={s9:.3f}, σ10={s10:.3f}, drop={drop:.1f}%")
    
    # ==========================================================================
    # DIAGNOSIS 2: Subspace Overlap at Different k
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS 2: Subspace Overlap at Different k")
    print("=" * 80)
    
    k_values = [1, 2, 3, 5, 7, 9, 10]
    
    for i in range(len(decoders)):
        for j in range(i + 1, len(decoders)):
            overlap_analysis = compute_subspace_overlap_detailed(
                decoders[i], decoders[j], k_values
            )
            
            print(f"\nSeed {seeds[i]} vs {seeds[j]}:")
            for k in k_values:
                print(f"  k={k}: overlap={overlap_analysis['overlaps'][k]:.3f}")
    
    # ==========================================================================
    # DIAGNOSIS 3: Feature Matching Analysis
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS 3: Feature Matching Analysis")
    print("=" * 80)
    
    for i in range(len(decoders)):
        for j in range(i + 1, len(decoders)):
            matching = analyze_feature_matching(decoders[i], decoders[j])
            
            print(f"\nSeed {seeds[i]} vs {seeds[j]}:")
            print(f"  PWMCC: {matching['pwmcc']:.3f}")
            print(f"  Unique matches 1→2: {matching['unique_matches_1to2']}/{matching['d_sae']} ({matching['pct_unique_1to2']:.1f}%)")
            print(f"  Unique matches 2→1: {matching['unique_matches_2to1']}/{matching['d_sae']} ({matching['pct_unique_2to1']:.1f}%)")
            print(f"  Max sim distribution: min={matching['distribution_1to2']['min']:.3f}, median={matching['distribution_1to2']['median']:.3f}, max={matching['distribution_1to2']['max']:.3f}")
            print(f"  % features with >0.9 match: {matching['distribution_1to2']['pct_above_0.9']:.1f}%")
            print(f"  % features with >0.5 match: {matching['distribution_1to2']['pct_above_0.5']:.1f}%")
    
    # ==========================================================================
    # DIAGNOSIS 4: Ground Truth Recovery Bug Check
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS 4: Ground Truth Recovery Bug Check")
    print("=" * 80)
    
    # Regenerate ground truth (same seed as experiment)
    torch.manual_seed(42)
    true_features = torch.randn(d_model, n_true_features)
    true_features = F.normalize(true_features, dim=0)
    
    print(f"\nRegenerated ground truth: shape={true_features.shape}")
    print(f"Ground truth orthogonality check: {(true_features.T @ true_features - torch.eye(n_true_features)).abs().max().item():.6f}")
    
    for i, decoder in enumerate(decoders):
        print(f"\nSeed {seeds[i]}:")
        
        # Original (buggy?) method
        n_orig, sim_orig, _ = compute_ground_truth_recovery_ORIGINAL(
            decoder, true_features, threshold=0.9
        )
        print(f"  ORIGINAL: {n_orig}/{n_true_features} recovered, mean_sim={sim_orig:.4f}")
        
        # Fixed method
        n_fixed, sim_fixed, _ = compute_ground_truth_recovery_FIXED(
            decoder, true_features, threshold=0.9
        )
        print(f"  FIXED:    {n_fixed}/{n_true_features} recovered, mean_sim={sim_fixed:.4f}")
        
        # Multi-threshold analysis
        strict_results = compute_ground_truth_recovery_STRICT(decoder, true_features)
        print(f"  Multi-threshold recovery:")
        for thresh in [0.7, 0.8, 0.9, 0.95, 0.99]:
            print(f"    >{thresh}: {strict_results[f'recovered_at_{thresh}']}/{n_true_features}")
        print(f"  Max similarities: {[f'{s:.3f}' for s in strict_results['max_similarities']]}")
    
    # ==========================================================================
    # DIAGNOSIS 5: The Core Paradox
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS 5: THE CORE PARADOX")
    print("=" * 80)
    
    # Compute all metrics for one pair
    D1, D2 = decoders[0], decoders[1]
    
    pwmcc = compute_pwmcc(D1, D2)
    overlap_k10 = compute_subspace_overlap_detailed(D1, D2, [10])['overlaps'][10]
    overlap_k9 = compute_subspace_overlap_detailed(D1, D2, [9])['overlaps'][9]
    
    n_rec_1, sim_1, _ = compute_ground_truth_recovery_FIXED(D1, true_features)
    n_rec_2, sim_2, _ = compute_ground_truth_recovery_FIXED(D2, true_features)
    
    print(f"\nSeed {seeds[0]} vs {seeds[1]}:")
    print(f"  PWMCC (feature overlap): {pwmcc:.3f}")
    print(f"  Subspace overlap (k=10): {overlap_k10:.3f}")
    print(f"  Subspace overlap (k=9):  {overlap_k9:.3f}")
    print(f"  GT recovery (SAE 1): {n_rec_1}/{n_true_features} (sim={sim_1:.3f})")
    print(f"  GT recovery (SAE 2): {n_rec_2}/{n_true_features} (sim={sim_2:.3f})")
    
    print("\n" + "-" * 40)
    print("PARADOX ANALYSIS:")
    print("-" * 40)
    
    if overlap_k9 > overlap_k10 * 1.5:
        print("✓ k=9 hypothesis SUPPORTED: 10th dimension is noise")
        print(f"  Overlap jumps from {overlap_k10:.3f} (k=10) to {overlap_k9:.3f} (k=9)")
    else:
        print("✗ k=9 hypothesis NOT supported")
    
    if sim_1 > 0.9 and overlap_k10 < 0.3:
        print("\n⚠️ PARADOX CONFIRMED:")
        print(f"  High GT recovery ({sim_1:.3f}) but low subspace overlap ({overlap_k10:.3f})")
        print("\nPossible explanations:")
        print("  1. SAEs find orthogonal projections of same features")
        print("  2. Ground truth metric allows many-to-one matching")
        print("  3. Features exist in different subspaces but still match GT")
    
    # ==========================================================================
    # SAVE DIAGNOSIS RESULTS
    # ==========================================================================
    
    diagnosis_results = {
        'experiment': 'paradox_diagnosis',
        'findings': {
            'pwmcc': pwmcc,
            'subspace_overlap_k10': overlap_k10,
            'subspace_overlap_k9': overlap_k9,
            'gt_recovery_sae1': {'n': n_rec_1, 'sim': sim_1},
            'gt_recovery_sae2': {'n': n_rec_2, 'sim': sim_2},
        }
    }
    
    output_path = RESULTS_DIR / 'paradox_diagnosis.json'
    with open(output_path, 'w') as f:
        json.dump(diagnosis_results, f, indent=2)
    
    print(f"\n✓ Diagnosis results saved to {output_path}")


if __name__ == '__main__':
    main()
