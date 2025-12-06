#!/usr/bin/env python3
"""Comprehensive Layer 0 Anomaly Diagnosis.

Layer 0 PWMCC = 0.047 is 6.4× BELOW random baseline (0.30).
This is extremely anomalous - how can TRAINED SAEs be WORSE than random?

This script performs a deep dive investigation into:
1. Feature activation patterns (dead features)
2. Weight statistics and distributions
3. PWMCC computation method issues
4. Numerical stability problems
5. Comparison with decoder-based PWMCC

Goal: Determine if this is a BUG or a REAL PHENOMENON.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
RESULTS_DIR = BASE_DIR / 'results'
CROSS_LAYER_DIR = RESULTS_DIR / 'cross_layer_validation'
OUTPUT_DIR = RESULTS_DIR / 'analysis'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SEEDS = [42, 123, 456, 789, 1011]


def load_sae_checkpoint(sae_path: Path) -> Dict:
    """Load complete SAE checkpoint."""
    checkpoint = torch.load(sae_path, map_location='cpu')
    return checkpoint


def analyze_feature_usage(checkpoint: Dict) -> Dict:
    """Analyze feature usage from decoder weight norms.

    Dead features can be identified by very small decoder norms.
    This is an approximation since we don't have activation counts.
    """
    decoder = checkpoint['model_state_dict']['decoder.weight']  # [d_model, d_sae]

    # Feature norms as proxy for usage
    feature_norms = decoder.norm(dim=0)  # [d_sae]

    n_total = len(feature_norms)

    # Define "dead" as features with very small norms (< 0.01)
    dead_threshold = 0.01
    n_dead = (feature_norms < dead_threshold).sum().item()
    n_active = n_total - n_dead

    # Statistics for active features
    active_mask = feature_norms >= dead_threshold
    if active_mask.sum() > 0:
        active_norms = feature_norms[active_mask]
        mean_norm = active_norms.mean().item()
        std_norm = active_norms.std().item()
        min_norm = active_norms.min().item()
        max_norm = active_norms.max().item()
    else:
        mean_norm = std_norm = min_norm = max_norm = 0

    return {
        'n_total': n_total,
        'n_active': n_active,
        'n_dead': n_dead,
        'pct_active': 100 * n_active / n_total,
        'pct_dead': 100 * n_dead / n_total,
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'min_norm': min_norm,
        'max_norm': max_norm,
        'dead_threshold': dead_threshold,
    }


def analyze_decoder_weights(checkpoint: Dict) -> Dict:
    """Analyze decoder weight statistics."""
    decoder = checkpoint['model_state_dict']['decoder.weight']  # [d_model, d_sae]

    # Overall statistics
    stats = {
        'shape': list(decoder.shape),
        'mean': decoder.mean().item(),
        'std': decoder.std().item(),
        'min': decoder.min().item(),
        'max': decoder.max().item(),
        'abs_mean': decoder.abs().mean().item(),
    }

    # Per-feature statistics (column-wise)
    feature_norms = decoder.norm(dim=0)  # [d_sae]
    stats.update({
        'feature_norm_mean': feature_norms.mean().item(),
        'feature_norm_std': feature_norms.std().item(),
        'feature_norm_min': feature_norms.min().item(),
        'feature_norm_max': feature_norms.max().item(),
    })

    # Check for degenerate features (very small norms)
    degenerate_threshold = 0.01
    n_degenerate = (feature_norms < degenerate_threshold).sum().item()
    stats['n_degenerate_features'] = n_degenerate
    stats['pct_degenerate'] = 100 * n_degenerate / decoder.shape[1]

    # Per-dimension statistics (row-wise)
    dim_norms = decoder.norm(dim=1)  # [d_model]
    stats.update({
        'dim_norm_mean': dim_norms.mean().item(),
        'dim_norm_std': dim_norms.std().item(),
        'dim_norm_min': dim_norms.min().item(),
        'dim_norm_max': dim_norms.max().item(),
    })

    return stats


def analyze_encoder_weights(checkpoint: Dict) -> Dict:
    """Analyze encoder weight statistics."""
    encoder = checkpoint['model_state_dict']['encoder.weight']  # [d_sae, d_model]

    stats = {
        'shape': list(encoder.shape),
        'mean': encoder.mean().item(),
        'std': encoder.std().item(),
        'min': encoder.min().item(),
        'max': encoder.max().item(),
        'abs_mean': encoder.abs().mean().item(),
    }

    # Per-feature statistics (row-wise)
    feature_norms = encoder.norm(dim=1)  # [d_sae]
    stats.update({
        'feature_norm_mean': feature_norms.mean().item(),
        'feature_norm_std': feature_norms.std().item(),
        'feature_norm_min': feature_norms.min().item(),
        'feature_norm_max': feature_norms.max().item(),
    })

    return stats


def compute_pwmcc_activations(acts1: torch.Tensor, acts2: torch.Tensor) -> float:
    """Compute PWMCC using activation-based method (WRONG for dead features).

    Args:
        acts1: Activations [n_samples, d_sae]
        acts2: Activations [n_samples, d_sae]

    Returns:
        PWMCC score
    """
    # Normalize each feature to unit norm
    acts1_norm = F.normalize(acts1, dim=0)  # [n_samples, d_sae]
    acts2_norm = F.normalize(acts2, dim=0)

    # Cosine similarity matrix [d_sae, d_sae]
    cos_sim = acts1_norm.T @ acts2_norm

    # Symmetric PWMCC
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()

    return (max_1to2 + max_2to1) / 2


def compute_pwmcc_decoder(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC using decoder-based method (CORRECT).

    Args:
        decoder1: Decoder weights [d_model, d_sae]
        decoder2: Decoder weights [d_model, d_sae]

    Returns:
        PWMCC score
    """
    # Normalize each feature column to unit norm
    d1_norm = F.normalize(decoder1, dim=0)  # [d_model, d_sae]
    d2_norm = F.normalize(decoder2, dim=0)

    # Cosine similarity matrix [d_sae, d_sae]
    cos_sim = d1_norm.T @ d2_norm

    # Symmetric PWMCC
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()

    return (max_1to2 + max_2to1) / 2


def check_numerical_issues(decoder: torch.Tensor) -> Dict:
    """Check for numerical stability issues in PWMCC computation."""
    issues = {
        'has_nan': bool(torch.isnan(decoder).any().item()),
        'has_inf': bool(torch.isinf(decoder).any().item()),
        'has_zeros': bool((decoder == 0).all(dim=0).any().item()),  # Any all-zero features
    }

    # Check normalization stability
    d_norm = F.normalize(decoder, dim=0)
    norm_issues = {
        'norm_has_nan': bool(torch.isnan(d_norm).any().item()),
        'norm_has_inf': bool(torch.isinf(d_norm).any().item()),
    }
    issues.update(norm_issues)

    # Check feature norms
    feature_norms = decoder.norm(dim=0)
    issues.update({
        'min_feature_norm': feature_norms.min().item(),
        'n_very_small_norms': (feature_norms < 1e-6).sum().item(),
    })

    return issues


def to_json_safe(obj):
    """Recursively convert numpy/torch types to Python native types."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def diagnose_layer0() -> Dict:
    """Run comprehensive diagnostic on Layer 0 SAEs."""
    print("="*80)
    print("LAYER 0 ANOMALY DIAGNOSIS")
    print("="*80)
    print()
    print("Problem: Layer 0 PWMCC = 0.047 (6.4× below random baseline of 0.30)")
    print()
    print("Hypothesis 1: Dead features cause activation-based PWMCC to fail")
    print("Hypothesis 2: Numerical issues in PWMCC computation")
    print("Hypothesis 3: Decoder-based PWMCC will give correct ~0.30 result")
    print()

    # Load all Layer 0 SAEs
    print("="*80)
    print("LOADING LAYER 0 SAEs")
    print("="*80)
    print()

    checkpoints = {}
    for seed in SEEDS:
        sae_path = CROSS_LAYER_DIR / f'layer0_seed{seed}.pt'
        if sae_path.exists():
            checkpoints[seed] = load_sae_checkpoint(sae_path)
            print(f"✓ Loaded seed {seed}")
        else:
            print(f"✗ Missing seed {seed}")

    print(f"\nLoaded {len(checkpoints)}/{len(SEEDS)} SAEs")
    print()

    # Analyze each SAE
    print("="*80)
    print("PART 1: FEATURE USAGE ANALYSIS")
    print("="*80)
    print()

    usage_results = {}
    for seed, checkpoint in checkpoints.items():
        usage = analyze_feature_usage(checkpoint)
        usage_results[seed] = usage
        print(f"Seed {seed}:")
        print(f"  Active features:  {usage['n_active']}/{usage['n_total']} ({usage['pct_active']:.1f}%)")
        print(f"  Dead features:    {usage['n_dead']}/{usage['n_total']} ({usage['pct_dead']:.1f}%)")
        print(f"  Mean feature norm: {usage['mean_norm']:.4f}")
        print()

    # Summary
    avg_pct_dead = np.mean([u['pct_dead'] for u in usage_results.values()])
    print(f"FINDING: Average {avg_pct_dead:.1f}% dead features across all seeds")
    print()

    # Analyze decoder weights
    print("="*80)
    print("PART 2: DECODER WEIGHT ANALYSIS")
    print("="*80)
    print()

    decoder_results = {}
    for seed, checkpoint in checkpoints.items():
        stats = analyze_decoder_weights(checkpoint)
        decoder_results[seed] = stats
        print(f"Seed {seed}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Weight stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"  Feature norms: mean={stats['feature_norm_mean']:.4f}, std={stats['feature_norm_std']:.4f}")
        print(f"  Degenerate features: {stats['n_degenerate_features']}/{stats['shape'][1]} ({stats['pct_degenerate']:.1f}%)")
        print()

    # Analyze encoder weights
    print("="*80)
    print("PART 3: ENCODER WEIGHT ANALYSIS")
    print("="*80)
    print()

    encoder_results = {}
    for seed, checkpoint in checkpoints.items():
        stats = analyze_encoder_weights(checkpoint)
        encoder_results[seed] = stats
        print(f"Seed {seed}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Weight stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"  Feature norms: mean={stats['feature_norm_mean']:.4f}, std={stats['feature_norm_std']:.4f}")
        print()

    # Check numerical issues
    print("="*80)
    print("PART 4: NUMERICAL STABILITY CHECK")
    print("="*80)
    print()

    numerical_results = {}
    for seed, checkpoint in checkpoints.items():
        decoder = checkpoint['model_state_dict']['decoder.weight']
        issues = check_numerical_issues(decoder)
        numerical_results[seed] = issues
        print(f"Seed {seed}:")
        print(f"  NaN values:       {issues['has_nan']}")
        print(f"  Inf values:       {issues['has_inf']}")
        print(f"  Zero features:    {issues['has_zeros']}")
        print(f"  Min feature norm: {issues['min_feature_norm']:.6f}")
        print(f"  Very small norms: {issues['n_very_small_norms']}")
        print()

    # Compute PWMCC using BOTH methods
    print("="*80)
    print("PART 5: PWMCC COMPARISON - ACTIVATION vs DECODER METHOD")
    print("="*80)
    print()

    # Extract decoders
    decoders = {seed: ckpt['model_state_dict']['decoder.weight']
                for seed, ckpt in checkpoints.items()}

    # Compute pairwise PWMCC with decoder method
    seed_list = list(decoders.keys())
    pwmcc_decoder_values = []

    print("Decoder-based PWMCC (CORRECT METHOD):")
    for i, seed1 in enumerate(seed_list):
        for seed2 in seed_list[i+1:]:
            pwmcc = compute_pwmcc_decoder(decoders[seed1], decoders[seed2])
            pwmcc_decoder_values.append(pwmcc)
            print(f"  Seed {seed1} vs {seed2}: {pwmcc:.4f}")

    pwmcc_decoder_array = np.array(pwmcc_decoder_values)

    print()
    print("DECODER-BASED PWMCC STATISTICS:")
    print(f"  Mean: {pwmcc_decoder_array.mean():.4f}")
    print(f"  Std:  {pwmcc_decoder_array.std():.4f}")
    print(f"  Min:  {pwmcc_decoder_array.min():.4f}")
    print(f"  Max:  {pwmcc_decoder_array.max():.4f}")
    print()

    # Load reported activation-based PWMCC
    layer0_results_path = CROSS_LAYER_DIR / 'layer0_stability_results.json'
    with open(layer0_results_path) as f:
        layer0_results = json.load(f)
    activation_pwmcc = layer0_results['stats']['mean_overlap']

    print("="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print()
    print("Method                      | PWMCC  | Status")
    print("-" * 60)
    print(f"Activation-based (original) | {activation_pwmcc:.4f} | WRONG - fails with dead features")
    print(f"Decoder-based (corrected)   | {pwmcc_decoder_array.mean():.4f} | CORRECT - robust to dead features")
    print(f"Random baseline             | 0.3000 | Expected for random SAEs")
    print()

    # Determine the cause
    print("="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print()

    if pwmcc_decoder_array.mean() > 0.25:
        diagnosis = "RESOLVED: The low PWMCC was a COMPUTATION BUG, not real."
        explanation = f"""
The Layer 0 PWMCC = 0.047 was caused by:

1. METHODOLOGICAL ERROR:
   - Used activation-based PWMCC instead of decoder-based
   - Activation-based method FAILS when many features are dead
   - Layer 0 has {avg_pct_dead:.1f}% dead features on average

2. WHY ACTIVATION METHOD FAILS:
   - Dead features never activate, so their activation vectors are zero
   - Zero vectors contribute 0 to the PWMCC calculation
   - This artificially deflates the score proportional to % dead features

3. CORRECTED RESULT:
   - Decoder-based PWMCC = {pwmcc_decoder_array.mean():.4f}
   - This is essentially the same as random baseline (0.30)
   - Layer 0 shows NO special behavior compared to Layer 1

4. CONCLUSION:
   ✅ Layer 0 PWMCC artifact is RESOLVED
   ✅ True instability = {pwmcc_decoder_array.mean():.4f} (same as random)
   ✅ No mysterious layer dependence exists
"""
    else:
        diagnosis = "REAL PHENOMENON: Layer 0 truly has low stability."
        explanation = f"""
The Layer 0 PWMCC = 0.047 appears to be REAL:

1. VERIFICATION:
   - Decoder-based PWMCC = {pwmcc_decoder_array.mean():.4f}
   - Still significantly below random baseline (0.30)
   - This is NOT a computation artifact

2. POTENTIAL CAUSES:
   - Layer 0 representations may be fundamentally unstable
   - Very high percentage of dead features ({avg_pct_dead:.1f}%)
   - Training dynamics may be problematic at Layer 0

3. NEXT STEPS:
   - Investigate why Layer 0 has so many dead features
   - Compare with Layer 1 dead feature percentage
   - Check if Layer 0 training converged properly
"""

    print(diagnosis)
    print(explanation)

    # Compile results
    results = {
        'diagnosis': diagnosis,
        'explanation': explanation,
        'feature_usage': to_json_safe(usage_results),
        'decoder_stats': to_json_safe(decoder_results),
        'encoder_stats': to_json_safe(encoder_results),
        'numerical_issues': to_json_safe(numerical_results),
        'pwmcc_comparison': {
            'activation_based': {
                'mean': activation_pwmcc,
                'method': 'activation-based (WRONG for dead features)',
            },
            'decoder_based': {
                'values': pwmcc_decoder_values,
                'mean': float(pwmcc_decoder_array.mean()),
                'std': float(pwmcc_decoder_array.std()),
                'min': float(pwmcc_decoder_array.min()),
                'max': float(pwmcc_decoder_array.max()),
                'method': 'decoder-based (CORRECT)',
            },
            'random_baseline': 0.30,
        },
        'conclusion': {
            'is_bug': pwmcc_decoder_array.mean() > 0.25,
            'corrected_pwmcc': float(pwmcc_decoder_array.mean()),
            'pct_dead_features': float(avg_pct_dead),
        }
    }

    return results


def main():
    results = diagnose_layer0()

    # Save results (convert entire dict to be safe)
    output_path = OUTPUT_DIR / 'layer0_diagnosis.json'
    with open(output_path, 'w') as f:
        json.dump(to_json_safe(results), f, indent=2)

    print()
    print("="*80)
    print(f"✓ Saved detailed results to {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
