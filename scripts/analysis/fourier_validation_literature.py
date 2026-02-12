#!/usr/bin/env python3
"""Validate Fourier learning using Nanda et al.'s exact methodology.

This script implements Nanda et al.'s approach from "Progress measures for grokking
via mechanistic interpretability" (ICLR 2023) to measure Fourier structure.

Key differences from our activation-based approach:
1. Analyzes WEIGHTS (embedding matrix) not activations
2. Applies DFT along vocab dimension
3. Computes variance explained (R²) against theoretical Fourier components

Usage:
    python scripts/fourier_validation_literature.py \
        --transformer results/transformer_5000ep/transformer_final.pt \
        --modulus 113
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer


def get_theoretical_fourier_components(modulus=113, key_freqs=[1, 5]):
    """Generate theoretical Fourier components for modular arithmetic.

    For modular addition (a + b) mod p, the key frequencies identified by
    Nanda et al. are typically k=1 and k=5 (or similar low frequencies).

    Args:
        modulus: Prime modulus (113)
        key_freqs: List of key frequency indices

    Returns:
        theoretical: [modulus, 2*len(key_freqs)] tensor
            Columns are [cos(2π·k₁·i/p), sin(2π·k₁·i/p), cos(2π·k₂·i/p), ...]
    """
    components = []

    for k in key_freqs:
        # Angles for frequency k
        indices = torch.arange(modulus).float()
        angles = 2 * torch.pi * k * indices / modulus

        # Cos and sin components
        cos_component = torch.cos(angles).unsqueeze(1)  # [modulus, 1]
        sin_component = torch.sin(angles).unsqueeze(1)  # [modulus, 1]

        components.append(cos_component)
        components.append(sin_component)

    # Stack all components
    theoretical = torch.cat(components, dim=1)  # [modulus, 2*len(key_freqs)]

    # Normalize
    theoretical = F.normalize(theoretical, dim=0)

    return theoretical


def compute_variance_explained(learned_embedding, theoretical_components):
    """Compute R² (variance explained) by projecting learned onto theoretical.

    This measures how much of the learned embedding's variance is captured
    by the theoretical Fourier components.

    Args:
        learned_embedding: [modulus, d_model] - learned W_E for numerical tokens
        theoretical_components: [modulus, n_components] - theoretical Fourier basis

    Returns:
        r_squared: float in [0, 1] - fraction of variance explained
        projection: [modulus, d_model] - projection of learned onto theoretical
    """
    # Center learned embedding
    learned_centered = learned_embedding - learned_embedding.mean(dim=0)

    # Project learned onto theoretical subspace
    # projection = theoretical @ (theoretical^T @ learned)
    theo_T_learned = theoretical_components.T @ learned_centered  # [n_components, d_model]
    projection = theoretical_components @ theo_T_learned  # [modulus, d_model]

    # Compute R²
    ss_total = torch.sum(learned_centered ** 2)
    ss_residual = torch.sum((learned_centered - projection) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared.item(), projection


def identify_key_frequencies(embedding, modulus=113, top_k=10):
    """Identify key frequencies by applying DFT and finding largest magnitudes.

    This replicates Nanda et al.'s approach of applying DFT along the vocab
    dimension and identifying sparse structure at specific frequencies.

    Args:
        embedding: [modulus, d_model] - embedding matrix for numerical tokens
        modulus: Vocabulary size
        top_k: Number of top frequencies to return

    Returns:
        top_freqs: List of frequency indices sorted by magnitude
        freq_norms: [modulus] tensor of frequency magnitudes
    """
    # Apply DFT along vocab dimension (dim=0)
    embedding_fourier = torch.fft.fft(embedding, dim=0)  # [modulus, d_model] complex

    # Compute magnitude for each frequency
    embedding_fourier_mag = torch.abs(embedding_fourier)  # [modulus, d_model]

    # Average across model dimension to get frequency strength
    freq_norms = embedding_fourier_mag.mean(dim=1)  # [modulus]

    # Find top frequencies
    top_freqs_idx = torch.argsort(freq_norms, descending=True)[:top_k]
    top_freqs = top_freqs_idx.tolist()

    return top_freqs, freq_norms


def plot_frequency_spectrum(freq_norms, top_freqs, output_path=None):
    """Plot frequency spectrum showing key frequencies.

    Args:
        freq_norms: [modulus] tensor of frequency magnitudes
        top_freqs: List of top frequency indices
        output_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    modulus = len(freq_norms)
    freqs = np.arange(modulus)

    # Plot all frequencies
    ax.bar(freqs, freq_norms.cpu().numpy(), alpha=0.6, label='All frequencies')

    # Highlight top frequencies
    top_norms = [freq_norms[i].item() if i in top_freqs else 0 for i in range(modulus)]
    ax.bar(freqs, top_norms, color='red', alpha=0.8, label=f'Top {len(top_freqs)} frequencies')

    ax.set_xlabel('Frequency index k')
    ax.set_ylabel('Average magnitude')
    ax.set_title('Frequency Spectrum of Embedding Matrix (Fourier Transform)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved frequency spectrum to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def analyze_embedding_fourier(model, modulus=113, top_k=5, verbose=True):
    """Analyze embedding matrix for Fourier structure using Nanda et al.'s method.

    Args:
        model: Trained ModularArithmeticTransformer
        modulus: Prime modulus (113)
        top_k: Number of top frequencies to analyze
        verbose: Print detailed results

    Returns:
        dict with analysis results
    """
    # Extract embedding matrix
    W_E = model.model.embed.W_E.data  # [vocab_size, d_model]

    # Only use numerical tokens (0 to modulus-1)
    W_E_numbers = W_E[:modulus, :]  # [modulus, d_model]

    if verbose:
        print(f"\nEmbedding matrix (numerical tokens):")
        print(f"  Shape: {W_E_numbers.shape}")
        print(f"  Mean norm: {W_E_numbers.norm(dim=1).mean().item():.4f}")
        print(f"  Std norm: {W_E_numbers.norm(dim=1).std().item():.4f}")

    # Identify key frequencies
    if verbose:
        print(f"\nApplying DFT along vocab dimension...")
    top_freqs, freq_norms = identify_key_frequencies(W_E_numbers, modulus, top_k)

    if verbose:
        print(f"\nTop {top_k} frequencies (by magnitude):")
        for i, freq_idx in enumerate(top_freqs[:top_k]):
            print(f"  {i+1}. Frequency k={freq_idx}: magnitude = {freq_norms[freq_idx].item():.4f}")

    # Compute variance explained by different numbers of frequencies
    results = {}

    for n_freqs in [2, 5, 10]:
        if n_freqs > top_k:
            continue

        # Get theoretical components for top n frequencies
        selected_freqs = top_freqs[:n_freqs]
        theoretical = get_theoretical_fourier_components(modulus, selected_freqs)

        # Compute R²
        r_squared, projection = compute_variance_explained(W_E_numbers, theoretical)

        results[f'r_squared_top{n_freqs}'] = r_squared

        if verbose:
            print(f"\nVariance explained by top {n_freqs} frequencies:")
            print(f"  R² = {r_squared:.4f} ({r_squared*100:.2f}%)")

            if r_squared > 0.9:
                print(f"  ✅ EXCELLENT: Strong Fourier structure!")
            elif r_squared > 0.6:
                print(f"  ⚠️  MODERATE: Partial Fourier structure")
            elif r_squared > 0.3:
                print(f"  ⚠️  WEAK: Limited Fourier structure")
            else:
                print(f"  ❌ POOR: No clear Fourier structure")

    # Store additional info
    results['top_frequencies'] = top_freqs
    results['frequency_norms'] = freq_norms.tolist()
    results['embedding_shape'] = list(W_E_numbers.shape)

    return results


def compare_to_nanda_et_al(r_squared, verbose=True):
    """Compare our results to Nanda et al.'s reported values.

    Nanda et al. reported 93.2% - 98.2% variance explained for their grokked models.

    Args:
        r_squared: Our computed R² value
        verbose: Print comparison

    Returns:
        str: "match", "partial", or "mismatch"
    """
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON TO NANDA ET AL. (ICLR 2023)")
        print("="*60)
        print(f"\nNanda et al. results (grokked models):")
        print(f"  R² = 0.932 - 0.982 (93.2% - 98.2%)")
        print(f"\nOur results:")
        print(f"  R² = {r_squared:.4f} ({r_squared*100:.2f}%)")
        print(f"\nDifference: {(r_squared - 0.95)*100:+.2f} percentage points")

    if r_squared > 0.9:
        status = "match"
        if verbose:
            print(f"\n✅ MATCH: Our transformer learned Fourier circuits!")
            print(f"   Comparable to Nanda et al.'s grokked models.")
    elif r_squared > 0.6:
        status = "partial"
        if verbose:
            print(f"\n⚠️  PARTIAL MATCH: Some Fourier structure, but weaker.")
            print(f"   May indicate incomplete grokking or different algorithm.")
    else:
        status = "mismatch"
        if verbose:
            print(f"\n❌ MISMATCH: Our transformer did NOT learn Fourier circuits!")
            print(f"   Substantially below Nanda et al.'s results.")

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Validate Fourier structure using Nanda et al.'s methodology"
    )
    parser.add_argument(
        '--transformer',
        type=Path,
        required=True,
        help='Path to transformer checkpoint'
    )
    parser.add_argument(
        '--modulus',
        type=int,
        default=113,
        help='Prime modulus for Fourier basis'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top frequencies to analyze'
    )
    parser.add_argument(
        '--plot',
        type=Path,
        default=None,
        help='Path to save frequency spectrum plot'
    )

    args = parser.parse_args()

    print("="*60)
    print("FOURIER VALIDATION - LITERATURE METHOD (Nanda et al. 2023)")
    print("="*60)
    print(f"\nTransformer checkpoint: {args.transformer}")
    print(f"Modulus: {args.modulus}")
    print(f"Top-k frequencies: {args.top_k}")

    # Load model
    print(f"\nLoading model...")
    model, metadata = ModularArithmeticTransformer.load_checkpoint(args.transformer)
    model.eval()

    if metadata:
        print(f"\nModel metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    # Analyze embedding
    print("\n" + "="*60)
    print("EMBEDDING MATRIX ANALYSIS")
    print("="*60)

    results = analyze_embedding_fourier(
        model,
        modulus=args.modulus,
        top_k=args.top_k,
        verbose=True
    )

    # Plot frequency spectrum
    if args.plot:
        freq_norms = torch.tensor(results['frequency_norms'])
        top_freqs = results['top_frequencies'][:args.top_k]
        plot_frequency_spectrum(freq_norms, top_freqs, args.plot)

    # Compare to Nanda et al.
    primary_r_squared = results.get('r_squared_top2', results.get('r_squared_top5'))
    status = compare_to_nanda_et_al(primary_r_squared, verbose=True)

    # Interpretation and next steps
    print("\n" + "="*60)
    print("INTERPRETATION & NEXT STEPS")
    print("="*60)

    if status == "match":
        print("\n✅ TRANSFORMER LEARNED FOURIER CIRCUITS")
        print("\nImplications:")
        print("  1. Our activation-based measurement was wrong approach")
        print("  2. SAE Fourier validation becomes meaningful again")
        print("  3. Can return to original research narrative")
        print("\nNext steps:")
        print("  1. Recompute SAE Fourier overlap using weight-based method")
        print("  2. Investigate why SAEs don't extract Fourier from weights")
        print("  3. Proceed with Phase 3 interventions")

    elif status == "partial":
        print("\n⚠️  TRANSFORMER LEARNED PARTIAL FOURIER STRUCTURE")
        print("\nImplications:")
        print("  1. Model may have partially grokked")
        print("  2. May use hybrid algorithm (Fourier + other)")
        print("  3. Need to investigate training dynamics")
        print("\nNext steps:")
        print("  1. Check training curves for grokking")
        print("  2. Test on held-out data (generalization)")
        print("  3. Decide if partial Fourier is sufficient for validation")

    else:  # mismatch
        print("\n❌ TRANSFORMER DID NOT LEARN FOURIER CIRCUITS")
        print("\nImplications:")
        print("  1. Our initial assumption was wrong")
        print("  2. Model solved task using different algorithm")
        print("  3. SAE instability findings remain valid (more general!)")
        print("\nNext steps:")
        print("  1. Accept revised research narrative")
        print("  2. Proceed with paper on SAE feature instability")
        print("  3. Drop Fourier validation from claims")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
