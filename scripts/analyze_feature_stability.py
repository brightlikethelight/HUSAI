#!/usr/bin/env python3
"""Analyze feature stability across multiple trained SAEs.

This script loads multiple SAEs and computes feature overlap matrices,
comparing stability across different seeds and architectures.

Usage:
    # Analyze TopK SAEs trained with different seeds
    python scripts/analyze_feature_stability.py \
        --sae-dir results/saes/topk/ \
        --pattern "topk_seed*.pt" \
        --output results/analysis/topk_stability.pkl

    # Compare TopK vs ReLU architectures
    python scripts/analyze_feature_stability.py \
        --topk-dir results/saes/topk/ \
        --relu-dir results/saes/relu/ \
        --compare-architectures \
        --output results/analysis/architecture_comparison.pkl

Example Output:
    Loading 5 SAEs from results/saes/topk/...
    Computing PWMCC matrix...
    100%|████████████| 10/10 [00:05<00:00,  1.80it/s]
    
    Mean overlap: 0.742 ± 0.083
    Range: [0.621, 0.856]
    Above threshold (0.7): 70.0%
    
    Saved results to results/analysis/topk_stability.pkl
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_sae import TopKSAE, ReLUSAE
from src.analysis.feature_matching import (
    compute_feature_overlap_matrix,
    compute_feature_statistics,
    visualize_overlap_matrix,
    compare_architectures
)


class SimpleWrapper:
    """Minimal wrapper to match interface for feature matching."""
    def __init__(self, sae):
        self.sae = sae


def load_sae_from_checkpoint(path: Path, architecture: str = "topk"):
    """Load simple SAE from checkpoint.

    Args:
        path: Path to checkpoint .pt file
        architecture: "topk" or "relu"

    Returns:
        SimpleWrapper containing loaded SAE
    """
    checkpoint = torch.load(path, map_location='cpu')

    # SimpleSAE saves config as top-level keys, not nested
    d_model = checkpoint['d_model']
    d_sae = checkpoint['d_sae']

    if architecture == "topk":
        k = checkpoint['k']
        sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    elif architecture == "relu":
        l1_coef = checkpoint['l1_coef']
        sae = ReLUSAE(d_model=d_model, d_sae=d_sae, l1_coef=l1_coef)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    sae.load_state_dict(checkpoint['model_state_dict'])
    return SimpleWrapper(sae)


def load_saes_from_dir(
    sae_dir: Path,
    pattern: str = "*/sae_final.pt",
    architecture: str = "topk"
) -> List:
    """Load all SAE checkpoints matching pattern from directory.

    Args:
        sae_dir: Directory containing SAE checkpoint subdirectories
        pattern: Glob pattern for checkpoint files (default: */sae_final.pt)
        architecture: "topk" or "relu"

    Returns:
        saes: List of loaded SAE wrappers
    """
    sae_dir = Path(sae_dir)
    checkpoint_paths = sorted(sae_dir.glob(pattern))

    if not checkpoint_paths:
        raise ValueError(f"No SAE checkpoints found in {sae_dir} with pattern {pattern}")

    print(f"Loading {len(checkpoint_paths)} {architecture.upper()} SAEs from {sae_dir}...")

    saes = []
    for path in checkpoint_paths:
        print(f"  Loading {path.parent.name}/{path.name}...")
        sae = load_sae_from_checkpoint(path, architecture=architecture)
        saes.append(sae)

    print(f"✅ Successfully loaded {len(saes)} SAEs")
    return saes


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE feature stability across seeds/architectures"
    )
    
    # Input options
    parser.add_argument(
        "--sae-dir",
        type=Path,
        help="Directory containing SAE checkpoints"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/sae_final.pt",
        help="Glob pattern for SAE checkpoint files (default: */sae_final.pt)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="topk",
        choices=["topk", "relu"],
        help="SAE architecture type (default: topk)"
    )
    parser.add_argument(
        "--topk-dir",
        type=Path,
        help="Directory containing TopK SAE checkpoints (for architecture comparison)"
    )
    parser.add_argument(
        "--relu-dir",
        type=Path,
        help="Directory containing ReLU SAE checkpoints (for architecture comparison)"
    )
    
    # Analysis options
    parser.add_argument(
        "--compare-architectures",
        action="store_true",
        help="Compare TopK vs ReLU architectures (requires --topk-dir and --relu-dir)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/feature_stability.pkl"),
        help="Output file for results (pickle format)"
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare_architectures:
        if not args.topk_dir or not args.relu_dir:
            parser.error("--compare-architectures requires both --topk-dir and --relu-dir")
    elif not args.sae_dir:
        parser.error("Either --sae-dir or (--topk-dir and --relu-dir) required")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if args.compare_architectures:
        # Architecture comparison mode
        print("="*60)
        print("ARCHITECTURE COMPARISON MODE")
        print("="*60)
        
        # Load SAEs
        topk_saes = load_saes_from_dir(args.topk_dir, args.pattern, architecture="topk")
        relu_saes = load_saes_from_dir(args.relu_dir, args.pattern, architecture="relu")
        
        # Compare architectures
        comparison = compare_architectures(
            topk_saes,
            relu_saes,
            save_dir=args.save_plots if not args.no_plots else None
        )
        
        # Save results
        results = {
            "mode": "architecture_comparison",
            "topk_stats": comparison["topk"],
            "relu_stats": comparison["relu"],
            "topk_matrix": comparison["topk_matrix"],
            "relu_matrix": comparison["relu_matrix"],
            "n_topk_saes": len(topk_saes),
            "n_relu_saes": len(relu_saes),
        }
        
    else:
        # Single architecture mode
        print("="*60)
        print("FEATURE STABILITY ANALYSIS")
        print("="*60)
        
        # Load SAEs
        saes = load_saes_from_dir(args.sae_dir, args.pattern, architecture=args.architecture)
        
        # Compute overlap matrix
        print("\nComputing feature overlap matrix...")
        overlap_matrix = compute_feature_overlap_matrix(saes)
        
        # Compute statistics
        stats = compute_feature_statistics(saes, overlap_matrix)
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nNumber of SAEs: {len(saes)}")
        print(f"Mean overlap: {stats['mean_overlap']:.3f} ± {stats['std_overlap']:.3f}")
        print(f"Range: [{stats['min_overlap']:.3f}, {stats['max_overlap']:.3f}]")
        print(f"Median: {stats['median_overlap']:.3f}")
        print(f"Above threshold (0.7): {stats['above_threshold']:.1%}")
        print("="*60)
        
        # Interpretation
        mean = stats['mean_overlap']
        if mean > 0.7:
            print("\n✅ HIGH STABILITY: Features converged to similar representations")
            print("   This suggests SAEs are learning stable, meaningful features.")
        elif mean > 0.5:
            print("\n⚠️  MODERATE STABILITY: Some convergence but still variable")
            print("   Consider tuning hyperparameters or trying different architectures.")
        elif mean > 0.3:
            print("\n❌ LOW STABILITY: Similar to Paulo & Belrose baseline")
            print("   SAEs show significant variation across seeds (reproducibility issue).")
        else:
            print("\n❌ VERY LOW STABILITY: Features are essentially random")
            print("   Check SAE implementation and training procedure.")
        
        # Generate plots
        if not args.no_plots:
            save_dir = args.save_plots or args.output.parent
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Get labels from checkpoint filenames
            labels = [f"SAE{i+1}" for i in range(len(saes))]
            
            visualize_overlap_matrix(
                overlap_matrix,
                labels=labels,
                save_path=save_dir / "overlap_matrix.png"
            )
        
        # Save results
        results = {
            "mode": "single_architecture",
            "stats": stats,
            "overlap_matrix": overlap_matrix,
            "n_saes": len(saes),
        }
    
    # Save to pickle
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {args.output}")
    
    # Also save human-readable JSON (without matrices)
    json_output = args.output.with_suffix('.json')
    json_results = {k: v for k, v in results.items() 
                   if not k.endswith('_matrix')}
    with open(json_output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved summary to {json_output}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
