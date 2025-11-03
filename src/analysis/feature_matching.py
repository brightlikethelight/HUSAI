"""Feature matching and stability analysis for SAEs.

This module implements metrics for comparing features learned by different
SAEs, particularly focusing on stability across different random seeds.

Key metric: PWMCC (Pairwise Maximum Cosine Correlation)
- From Paulo & Belrose (2025): "Do SAEs Converge to Stable Features?"
- Measures feature overlap between two SAEs
- High PWMCC (>0.7): Features converged to similar representations
- Low PWMCC (<0.3): Features diverged (reproducibility crisis)

Example:
    >>> from src.models.sae import SAEWrapper
    >>> from src.analysis.feature_matching import compute_pwmcc, compute_feature_overlap_matrix
    >>>
    >>> # Train two SAEs with different seeds
    >>> sae1 = train_sae(config, seed=42)
    >>> sae2 = train_sae(config, seed=123)
    >>>
    >>> # Compute pairwise overlap
    >>> overlap = compute_pwmcc(sae1, sae2)
    >>> print(f"Feature overlap: {overlap:.3f}")
    >>>
    >>> # Compute full overlap matrix for multiple SAEs
    >>> saes = [train_sae(config, seed=s) for s in [42, 123, 456, 789, 1011]]
    >>> matrix = compute_feature_overlap_matrix(saes)
    >>> mean_overlap = matrix[np.triu_indices(5, k=1)].mean()
    >>> print(f"Mean pairwise overlap: {mean_overlap:.3f}")

References:
    Paulo & Belrose (2025): "Do SAEs Converge to Stable Features?"
    (Hypothetical paper modeling the reproducibility crisis)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


def compute_pwmcc(
    sae1,
    sae2,
    symmetric: bool = True
) -> float:
    """Compute Pairwise Maximum Cosine Correlation between two SAEs.
    
    For each feature in SAE1, finds the most similar feature in SAE2
    based on cosine similarity of decoder weights. Averages these maximum
    correlations (optionally symmetric).
    
    Args:
        sae1: First SAE (SAEWrapper or similar with .sae.decoder.weight)
        sae2: Second SAE
        symmetric: If True, compute (1→2 + 2→1) / 2. If False, only 1→2.
        
    Returns:
        PWMCC score in [0, 1]. Higher = more overlap.
        
    Example:
        >>> pwmcc = compute_pwmcc(sae1, sae2)
        >>> print(f"Overlap: {pwmcc:.3f}")
        >>> # 0.8 = very similar features
        >>> # 0.3 = Paulo & Belrose baseline (diverged)
        >>> # 0.1 = random
    """
    # Extract decoder weights (feature directions)
    # Shape: [d_model, d_sae]
    features1 = sae1.sae.decoder.weight.data.cpu()
    features2 = sae2.sae.decoder.weight.data.cpu()
    
    d_model1, d_sae1 = features1.shape
    d_model2, d_sae2 = features2.shape
    
    # Verify dimensions match
    assert d_model1 == d_model2, (
        f"Model dimensions must match: {d_model1} vs {d_model2}"
    )
    
    # Normalize features (for cosine similarity)
    features1_norm = F.normalize(features1, dim=0)  # [d_model, d_sae1]
    features2_norm = F.normalize(features2, dim=0)  # [d_model, d_sae2]
    
    # Compute cosine similarity matrix: [d_sae1, d_sae2]
    cos_sim = features1_norm.T @ features2_norm  # [d_sae1, d_sae2]
    
    # For each feature in SAE1, find maximum correlation with SAE2
    max_corr_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    
    if symmetric:
        # Also compute reverse direction (SAE2 → SAE1)
        max_corr_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
        
        # Symmetric PWMCC
        pwmcc = (max_corr_1to2 + max_corr_2to1) / 2
    else:
        pwmcc = max_corr_1to2
    
    return pwmcc


def compute_feature_overlap_matrix(
    saes: List,
    symmetric: bool = True,
    show_progress: bool = True
) -> np.ndarray:
    """Compute PWMCC matrix for multiple SAEs.
    
    Creates an N×N matrix where entry (i,j) is the PWMCC between
    SAE i and SAE j. Diagonal is 1.0 (self-similarity).
    
    Args:
        saes: List of trained SAEs
        symmetric: If True, use symmetric PWMCC
        show_progress: If True, show tqdm progress bar
        
    Returns:
        overlap_matrix: [N, N] numpy array of PWMCC values
        
    Example:
        >>> saes = [load_sae(f"seed{s}.pt") for s in [42, 123, 456, 789, 1011]]
        >>> matrix = compute_feature_overlap_matrix(saes)
        >>> print(f"Mean off-diagonal: {matrix[np.triu_indices(5, k=1)].mean():.3f}")
    """
    n = len(saes)
    matrix = np.zeros((n, n))
    
    # Create progress bar for upper triangle
    total_pairs = n * (n - 1) // 2
    pbar = tqdm(total=total_pairs, desc="Computing PWMCC", disable=not show_progress)
    
    # Compute upper triangle
    for i in range(n):
        matrix[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, n):
            pwmcc = compute_pwmcc(saes[i], saes[j], symmetric=symmetric)
            matrix[i, j] = pwmcc
            matrix[j, i] = pwmcc  # Symmetric
            pbar.update(1)
    
    pbar.close()
    return matrix


def compute_feature_statistics(
    saes: List,
    overlap_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute summary statistics for feature stability.
    
    Args:
        saes: List of trained SAEs
        overlap_matrix: Optional pre-computed matrix. If None, computes it.
        
    Returns:
        stats: Dictionary with:
            - mean_overlap: Mean off-diagonal PWMCC
            - std_overlap: Std of off-diagonal PWMCC
            - min_overlap: Minimum pairwise overlap
            - max_overlap: Maximum pairwise overlap
            - median_overlap: Median pairwise overlap
            - above_threshold: Fraction of pairs with PWMCC > 0.7
            
    Example:
        >>> stats = compute_feature_statistics(saes)
        >>> print(f"Mean overlap: {stats['mean_overlap']:.3f} ± {stats['std_overlap']:.3f}")
        >>> print(f"Above threshold (0.7): {stats['above_threshold']:.1%}")
    """
    if overlap_matrix is None:
        overlap_matrix = compute_feature_overlap_matrix(saes, show_progress=False)
    
    n = len(saes)
    
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    off_diagonal = overlap_matrix[triu_indices]
    
    stats = {
        "mean_overlap": float(off_diagonal.mean()),
        "std_overlap": float(off_diagonal.std()),
        "min_overlap": float(off_diagonal.min()),
        "max_overlap": float(off_diagonal.max()),
        "median_overlap": float(np.median(off_diagonal)),
        "above_threshold": float((off_diagonal > 0.7).mean()),
    }
    
    return stats


def analyze_feature_pairs(
    sae1,
    sae2,
    top_k: int = 10
) -> Tuple[List[Tuple[int, int, float]], torch.Tensor]:
    """Find top-k most similar feature pairs between two SAEs.
    
    Args:
        sae1: First SAE
        sae2: Second SAE
        top_k: Number of top pairs to return
        
    Returns:
        pairs: List of (feature1_idx, feature2_idx, correlation)
        similarity_matrix: Full [d_sae1, d_sae2] similarity matrix
        
    Example:
        >>> pairs, sim_matrix = analyze_feature_pairs(sae1, sae2, top_k=5)
        >>> for f1, f2, corr in pairs:
        >>>     print(f"Feature {f1} ↔ Feature {f2}: {corr:.3f}")
    """
    # Extract decoder weights
    features1 = sae1.sae.decoder.weight.data.cpu()
    features2 = sae2.sae.decoder.weight.data.cpu()
    
    # Normalize
    features1_norm = F.normalize(features1, dim=0)
    features2_norm = F.normalize(features2, dim=0)
    
    # Cosine similarity matrix
    cos_sim = features1_norm.T @ features2_norm  # [d_sae1, d_sae2]
    
    # Find top-k pairs
    cos_sim_abs = cos_sim.abs()
    flat_indices = cos_sim_abs.flatten().topk(top_k)[1]
    
    pairs = []
    d_sae1, d_sae2 = cos_sim.shape
    for flat_idx in flat_indices:
        i = flat_idx // d_sae2
        j = flat_idx % d_sae2
        correlation = cos_sim[i, j].item()
        pairs.append((i.item(), j.item(), correlation))
    
    return pairs, cos_sim


def visualize_overlap_matrix(
    overlap_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "SAE Feature Overlap (PWMCC)",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Visualize PWMCC matrix as a heatmap.
    
    Args:
        overlap_matrix: [N, N] PWMCC matrix
        labels: Optional labels for SAEs (e.g., ["seed42", "seed123", ...])
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size (width, height)
        
    Returns:
        fig: Matplotlib figure
        
    Example:
        >>> matrix = compute_feature_overlap_matrix(saes)
        >>> fig = visualize_overlap_matrix(
        ...     matrix,
        ...     labels=[f"seed{s}" for s in [42, 123, 456, 789, 1011]],
        ...     save_path=Path("results/overlap_matrix.png")
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "PWMCC"},
        xticklabels=labels if labels else range(len(overlap_matrix)),
        yticklabels=labels if labels else range(len(overlap_matrix)),
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("SAE", fontsize=12)
    ax.set_ylabel("SAE", fontsize=12)
    
    # Add statistics annotation
    n = len(overlap_matrix)
    triu_indices = np.triu_indices(n, k=1)
    off_diagonal = overlap_matrix[triu_indices]
    mean_overlap = off_diagonal.mean()
    std_overlap = off_diagonal.std()
    
    stats_text = f"Mean: {mean_overlap:.3f} ± {std_overlap:.3f}\n"
    stats_text += f"Min: {off_diagonal.min():.3f}, Max: {off_diagonal.max():.3f}"
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlap matrix to {save_path}")
    
    return fig


def compare_architectures(
    topk_saes: List,
    relu_saes: List,
    save_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """Compare feature stability between TopK and ReLU architectures.
    
    Args:
        topk_saes: List of TopK SAEs trained with different seeds
        relu_saes: List of ReLU SAEs trained with different seeds
        save_dir: Optional directory to save comparison plots
        
    Returns:
        comparison: Dictionary with stats for each architecture
        
    Example:
        >>> topk_saes = [load_sae(f"topk_seed{s}.pt") for s in seeds]
        >>> relu_saes = [load_sae(f"relu_seed{s}.pt") for s in seeds]
        >>> comparison = compare_architectures(topk_saes, relu_saes)
        >>> print(f"TopK mean overlap: {comparison['topk']['mean_overlap']:.3f}")
        >>> print(f"ReLU mean overlap: {comparison['relu']['mean_overlap']:.3f}")
    """
    # Compute overlap matrices
    print("Computing TopK overlap matrix...")
    topk_matrix = compute_feature_overlap_matrix(topk_saes)
    
    print("Computing ReLU overlap matrix...")
    relu_matrix = compute_feature_overlap_matrix(relu_saes)
    
    # Compute statistics
    topk_stats = compute_feature_statistics(topk_saes, topk_matrix)
    relu_stats = compute_feature_statistics(relu_saes, relu_matrix)
    
    comparison = {
        "topk": topk_stats,
        "relu": relu_stats,
        "topk_matrix": topk_matrix,
        "relu_matrix": relu_matrix
    }
    
    # Print comparison
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"\nTopK SAE (n={len(topk_saes)}):")
    print(f"  Mean overlap: {topk_stats['mean_overlap']:.3f} ± {topk_stats['std_overlap']:.3f}")
    print(f"  Range: [{topk_stats['min_overlap']:.3f}, {topk_stats['max_overlap']:.3f}]")
    print(f"  Above 0.7 threshold: {topk_stats['above_threshold']:.1%}")
    
    print(f"\nReLU SAE (n={len(relu_saes)}):")
    print(f"  Mean overlap: {relu_stats['mean_overlap']:.3f} ± {relu_stats['std_overlap']:.3f}")
    print(f"  Range: [{relu_stats['min_overlap']:.3f}, {relu_stats['max_overlap']:.3f}]")
    print(f"  Above 0.7 threshold: {relu_stats['above_threshold']:.1%}")
    
    # Statistical comparison
    improvement = topk_stats['mean_overlap'] - relu_stats['mean_overlap']
    print(f"\nTopK vs ReLU:")
    print(f"  Difference: {improvement:+.3f} ({improvement/relu_stats['mean_overlap']:+.1%})")
    
    if topk_stats['mean_overlap'] > 0.7:
        print(f"  ✅ TopK achieves high stability (>0.7)")
    elif topk_stats['mean_overlap'] > relu_stats['mean_overlap'] + 0.1:
        print(f"  ✅ TopK shows improvement over ReLU")
    else:
        print(f"  ⚠️  Both architectures show similar instability")
    
    print("="*60)
    
    # Create comparison visualizations
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Heatmaps
        visualize_overlap_matrix(
            topk_matrix,
            title="TopK SAE Feature Overlap",
            save_path=save_dir / "topk_overlap_matrix.png"
        )
        
        visualize_overlap_matrix(
            relu_matrix,
            title="ReLU SAE Feature Overlap",
            save_path=save_dir / "relu_overlap_matrix.png"
        )
        
        # Distribution comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        topk_off_diag = topk_matrix[np.triu_indices(len(topk_saes), k=1)]
        relu_off_diag = relu_matrix[np.triu_indices(len(relu_saes), k=1)]
        
        ax.hist(topk_off_diag, bins=20, alpha=0.5, label='TopK', color='blue')
        ax.hist(relu_off_diag, bins=20, alpha=0.5, label='ReLU', color='red')
        ax.axvline(0.3, color='black', linestyle='--', 
                   label='Paulo & Belrose baseline (0.3)')
        ax.axvline(0.7, color='green', linestyle='--', 
                   label='Stability threshold (0.7)')
        
        ax.set_xlabel('PWMCC', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Feature Overlap Distribution by Architecture', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(save_dir / "architecture_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plots to {save_dir}")
    
    return comparison


if __name__ == "__main__":
    print("Feature matching module - use via import")
    print("\nExample usage:")
    print("  from src.analysis.feature_matching import compute_pwmcc")
    print("  overlap = compute_pwmcc(sae1, sae2)")
    print("  print(f'Overlap: {overlap:.3f}')")
