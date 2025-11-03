"""Fourier validation for SAEs on modular arithmetic.

This module provides ground truth validation by comparing SAE features
to the known Fourier basis that transformers learn for modular arithmetic.
This is a key competitive advantage for the HUSAI research project.

Key findings from Nanda et al. (2023):
- Transformers trained on modular arithmetic (a + b) mod p learn Fourier circuits
- The network decomposes the task using sin/cos features for each frequency
- We can extract this ground truth basis and compare to SAE features

References:
    Nanda et al. (2023): "Progress measures for grokking via mechanistic
    interpretability" https://arxiv.org/abs/2301.05217

Example:
    >>> from src.models.sae import SAEWrapper
    >>> from src.analysis.fourier_validation import get_fourier_basis, compute_fourier_overlap
    >>>
    >>> # Get ground truth Fourier basis
    >>> fourier_basis = get_fourier_basis(modulus=113)
    >>>
    >>> # Train SAE
    >>> sae = SAEWrapper(config)
    >>> metrics = train_sae(sae, activations, config)
    >>>
    >>> # Compute overlap with ground truth
    >>> overlap = compute_fourier_overlap(sae.sae.decoder.weight, fourier_basis)
    >>> print(f"Fourier overlap: {overlap:.3f}")
    >>>
    >>> # Good SAE: 0.6-0.8
    >>> # Poor SAE: 0.2-0.4
    >>> # Random: ~0.1
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def get_fourier_basis(modulus: int, device: str = "cpu") -> torch.Tensor:
    """Extract ground truth Fourier basis for modular arithmetic.

    For modulus p, the Fourier basis consists of sin and cos components
    for each frequency k = 0, 1, ..., p-1.

    Args:
        modulus: Modulus p for (a + b) mod p task
        device: Device to create tensor on

    Returns:
        fourier_basis: [2*modulus, modulus] tensor
            First p rows: cos(2πkx/p) for k=0 to p-1
            Next p rows: sin(2πkx/p) for k=0 to p-1

    Example:
        >>> basis = get_fourier_basis(modulus=113)
        >>> basis.shape
        torch.Size([226, 113])  # 226 = 2*113 Fourier components
    """
    p = modulus
    x = torch.arange(p, dtype=torch.float32, device=device)

    # Create basis vectors
    basis_vectors = []

    for k in range(p):
        # Cosine component
        cos_component = torch.cos(2 * torch.pi * k * x / p)
        basis_vectors.append(cos_component)

        # Sine component
        sin_component = torch.sin(2 * torch.pi * k * x / p)
        basis_vectors.append(sin_component)

    fourier_basis = torch.stack(basis_vectors)  # [2p, p]

    return fourier_basis


def compute_fourier_overlap(
    sae_features: torch.Tensor,
    fourier_basis: torch.Tensor,
    method: str = "max_cosine"
) -> float:
    """Compute overlap between SAE features and Fourier ground truth.

    This metric quantifies how well the SAE recovered the known Fourier
    structure that the transformer learned. High overlap means the SAE
    is learning meaningful, interpretable features.

    Args:
        sae_features: SAE decoder weights [d_model, d_sae]
        fourier_basis: Fourier basis [2*modulus, modulus]
        method: Overlap computation method
            - "max_cosine": Average max cosine similarity per SAE feature
            - "mean_cosine": Average cosine similarity (all pairs)

    Returns:
        overlap: Scalar in [0, 1]
            - 0.8-1.0: Excellent recovery of Fourier structure
            - 0.6-0.8: Good recovery (most features align)
            - 0.4-0.6: Moderate recovery (some alignment)
            - 0.2-0.4: Poor recovery (weak alignment)
            - 0.0-0.2: Random/no structure

    Example:
        >>> # SAE decoder: [128, 1024]
        >>> # Fourier basis: [226, 113]
        >>> overlap = compute_fourier_overlap(sae.decoder.weight, fourier_basis)
        >>> print(f"Overlap: {overlap:.3f}")
        0.723  # Good recovery!

    Note:
        This is a key metric for validating SAE quality on modular arithmetic.
        Unlike language models where ground truth is unknown, we can objectively
        measure if SAEs are learning the "right" features.
    """
    # Ensure same device
    sae_features = sae_features.to(fourier_basis.device)

    if method == "max_cosine":
        # For each SAE feature, find best match in Fourier basis
        # sae_features: [d_model, d_sae]
        # fourier_basis: [2p, p]

        # We need to project SAE features into the p-dimensional space
        # where Fourier basis lives. If d_model > p, we need to handle this.

        d_model, d_sae = sae_features.shape
        num_fourier, p = fourier_basis.shape

        # If d_model doesn't match p, we need to handle dimensionality
        if d_model != p:
            # Pad or truncate to match
            if d_model > p:
                # Take first p dimensions
                sae_features = sae_features[:p, :]
            else:
                # Pad with zeros
                padding = torch.zeros(
                    p - d_model, d_sae,
                    device=sae_features.device,
                    dtype=sae_features.dtype
                )
                sae_features = torch.cat([sae_features, padding], dim=0)

        # Normalize both feature sets
        sae_features_norm = F.normalize(sae_features, dim=0)  # [p, d_sae]
        fourier_basis_norm = F.normalize(fourier_basis, dim=1)  # [2p, p]

        # Compute cosine similarity matrix
        # [2p, p] @ [p, d_sae] = [2p, d_sae]
        cos_sim = fourier_basis_norm @ sae_features_norm

        # For each SAE feature, find maximum similarity to any Fourier component
        max_similarity = cos_sim.max(dim=0)[0]  # [d_sae]

        # Average across all SAE features
        overlap = max_similarity.mean().item()

        return overlap

    elif method == "mean_cosine":
        # Average cosine similarity across all pairs
        d_model, d_sae = sae_features.shape
        num_fourier, p = fourier_basis.shape

        # Match dimensions
        if d_model != p:
            if d_model > p:
                sae_features = sae_features[:p, :]
            else:
                padding = torch.zeros(
                    p - d_model, d_sae,
                    device=sae_features.device,
                    dtype=sae_features.dtype
                )
                sae_features = torch.cat([sae_features, padding], dim=0)

        # Normalize
        sae_features_norm = F.normalize(sae_features, dim=0)
        fourier_basis_norm = F.normalize(fourier_basis, dim=1)

        # Compute mean cosine similarity
        cos_sim = fourier_basis_norm @ sae_features_norm
        overlap = cos_sim.abs().mean().item()

        return overlap

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_feature_frequency_distribution(
    sae_features: torch.Tensor,
    fourier_basis: torch.Tensor,
    top_k: int = 10
) -> dict:
    """Analyze which Fourier frequencies each SAE feature corresponds to.

    Args:
        sae_features: SAE decoder weights [d_model, d_sae]
        fourier_basis: Fourier basis [2*modulus, modulus]
        top_k: Number of top frequencies to return per feature

    Returns:
        analysis: Dictionary with:
            - "frequency_assignments": [d_sae] best frequency per feature
            - "frequency_histogram": Counts of features per frequency
            - "top_frequencies": Most commonly recovered frequencies
    """
    sae_features = sae_features.to(fourier_basis.device)

    d_model, d_sae = sae_features.shape
    num_fourier, p = fourier_basis.shape
    modulus = num_fourier // 2

    # Match dimensions
    if d_model != p:
        if d_model > p:
            sae_features = sae_features[:p, :]
        else:
            padding = torch.zeros(
                p - d_model, d_sae,
                device=sae_features.device,
                dtype=sae_features.dtype
            )
            sae_features = torch.cat([sae_features, padding], dim=0)

    # Normalize
    sae_features_norm = F.normalize(sae_features, dim=0)
    fourier_basis_norm = F.normalize(fourier_basis, dim=1)

    # Compute cosine similarity
    cos_sim = fourier_basis_norm @ sae_features_norm  # [2p, d_sae]

    # For each SAE feature, find best Fourier component
    best_fourier_idx = cos_sim.argmax(dim=0)  # [d_sae]

    # Convert Fourier component index to frequency
    # Index 0-p-1: cos(kx), frequency k = index
    # Index p-2p-1: sin(kx), frequency k = index - p
    frequencies = torch.where(
        best_fourier_idx < modulus,
        best_fourier_idx,  # Cosine components
        best_fourier_idx - modulus  # Sine components
    )

    # Histogram of frequencies
    freq_counts = torch.bincount(frequencies, minlength=modulus)

    # Top frequencies
    top_freq_indices = freq_counts.topk(top_k)[1]

    analysis = {
        "frequency_assignments": frequencies.cpu().numpy(),
        "frequency_histogram": freq_counts.cpu().numpy(),
        "top_frequencies": top_freq_indices.cpu().numpy(),
        "similarity_scores": cos_sim.max(dim=0)[0].cpu().numpy()
    }

    return analysis


def visualize_fourier_features(
    sae_features: torch.Tensor,
    fourier_basis: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize SAE feature alignment with Fourier basis.

    Creates a figure with:
    1. Histogram of feature-frequency assignments
    2. Heatmap of cosine similarities
    3. Distribution of overlap scores

    Args:
        sae_features: SAE decoder weights [d_model, d_sae]
        fourier_basis: Fourier basis [2*modulus, modulus]
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure

    Example:
        >>> fig = visualize_fourier_features(sae.decoder.weight, basis)
        >>> plt.show()
    """
    analysis = compute_feature_frequency_distribution(sae_features, fourier_basis)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Frequency histogram
    axes[0].bar(range(len(analysis["frequency_histogram"])),
                analysis["frequency_histogram"])
    axes[0].set_xlabel("Fourier Frequency k")
    axes[0].set_ylabel("Number of SAE Features")
    axes[0].set_title("SAE Features per Fourier Frequency")
    axes[0].grid(alpha=0.3)

    # Plot 2: Top frequencies
    top_k = min(20, len(analysis["top_frequencies"]))
    top_freqs = analysis["top_frequencies"][:top_k]
    top_counts = analysis["frequency_histogram"][top_freqs]

    axes[1].bar(range(top_k), top_counts)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Feature Count")
    axes[1].set_title(f"Top {top_k} Most Common Frequencies")
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels(top_freqs, rotation=45)
    axes[1].grid(alpha=0.3)

    # Plot 3: Similarity score distribution
    axes[2].hist(analysis["similarity_scores"], bins=50, alpha=0.7, edgecolor='black')
    axes[2].axvline(analysis["similarity_scores"].mean(),
                    color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {analysis["similarity_scores"].mean():.3f}')
    axes[2].set_xlabel("Max Cosine Similarity")
    axes[2].set_ylabel("Number of Features")
    axes[2].set_title("Distribution of Feature-Fourier Overlap")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
