"""Simplified investigation of transformer algorithm."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset

# Suppress OpenMP warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    """Run simplified investigation."""
    print("Loading model...")

    # Path to trained model
    model_path = Path(__file__).parent.parent / 'results' / 'transformer_5000ep' / 'transformer_final.pt'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    device = 'cpu'  # Use CPU to avoid CUDA issues
    model, extras = ModularArithmeticTransformer.load_checkpoint(
        model_path, device=device
    )
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Config: {model.config}")
    print(f"Trained for {extras.get('epoch', 'unknown')} epochs")

    # Create dataset
    print("\nCreating dataset...")
    modulus = 113
    dataset = ModularArithmeticDataset(modulus=modulus, fraction=1.0, seed=42)

    # Test accuracy on a sample
    print("\nTesting model accuracy...")
    n_test = 100
    correct = 0

    with torch.no_grad():
        for i in range(n_test):
            tokens, label = dataset[i]
            tokens = tokens.unsqueeze(0).to(device)

            logits = model(tokens)
            # Predict at position -2 (the '=' position)
            pred = logits[0, -2, :modulus].argmax()

            if pred.item() == label.item():
                correct += 1

    accuracy = correct / n_test
    print(f"Accuracy on {n_test} samples: {accuracy:.4f}")

    # Analyze embeddings
    print("\n" + "="*80)
    print("ANALYZING EMBEDDING WEIGHTS")
    print("="*80)

    embed = model.model.embed.W_E.detach().cpu().numpy()  # [vocab_size, d_model]
    print(f"Embedding shape: {embed.shape}")
    print(f"Mean: {embed.mean():.4f}, Std: {embed.std():.4f}")

    # Analyze digit embeddings only
    digit_embeds = embed[:modulus]  # [113, d_model]

    # Check for Fourier structure
    print("\nChecking for Fourier structure in embeddings...")

    # Compute FFT for each dimension
    fft_peak_ratios = []
    for dim in range(digit_embeds.shape[1]):
        fft = np.fft.fft(digit_embeds[:, dim])
        fft_mag = np.abs(fft)

        # Exclude DC component
        fft_mag_no_dc = fft_mag[1:]

        # Ratio of max to mean
        max_mag = fft_mag_no_dc.max()
        mean_mag = fft_mag_no_dc.mean()

        if mean_mag > 1e-10:
            peak_ratio = max_mag / mean_mag
            fft_peak_ratios.append(peak_ratio)

    fft_peak_ratios = np.array(fft_peak_ratios)
    print(f"\nFFT Peak Ratios (indicates Fourier structure):")
    print(f"  Mean: {fft_peak_ratios.mean():.3f}")
    print(f"  Max: {fft_peak_ratios.max():.3f}")
    print(f"  Std: {fft_peak_ratios.std():.3f}")
    print(f"\nInterpretation:")
    print(f"  < 2.0: No Fourier structure")
    print(f"  2-5: Weak Fourier structure")
    print(f"  > 5.0: Strong Fourier structure")

    if fft_peak_ratios.mean() > 3.0:
        print(f"\n  → Result: STRONG Fourier structure detected")
    else:
        print(f"\n  → Result: NO strong Fourier structure")

    # Check for polynomial correlations
    print("\n" + "="*80)
    print("CHECKING FOR POLYNOMIAL BASIS")
    print("="*80)

    positions = np.arange(modulus)
    print("\nMax correlation with polynomial powers:")

    for power in range(1, 6):
        poly_basis = (positions ** power) % modulus  # Keep in modulus range
        correlations = []

        for dim in range(digit_embeds.shape[1]):
            corr = np.corrcoef(digit_embeds[:, dim], poly_basis)[0, 1]
            correlations.append(abs(corr))

        max_corr = max(correlations)
        print(f"  x^{power}: {max_corr:.3f}")

        if max_corr > 0.7:
            print(f"    → Strong polynomial structure!")

    # Analyze attention patterns
    print("\n" + "="*80)
    print("ANALYZING ATTENTION PATTERNS")
    print("="*80)

    # Sample one input
    tokens, label = dataset[0]  # 0 + 0 = 0
    tokens = tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        _, cache = model.model.run_with_cache(tokens)

    # Analyze each layer
    for layer in range(model.config.n_layers):
        print(f"\nLayer {layer}:")

        # Get attention pattern [1, n_heads, seq_len, seq_len]
        attn = cache[f'blocks.{layer}.attn.hook_pattern'][0]  # [n_heads, seq_len, seq_len]

        # Average over heads
        avg_attn = attn.mean(0).cpu().numpy()  # [seq_len, seq_len]

        # Show what position 5 (answer) attends to
        # Sequence: [BOS, a, +, b, =, c, EOS]
        # Positions: [0,  1, 2, 3, 4, 5, 6]
        answer_attention = avg_attn[5]  # What answer position attends to

        print(f"  Answer position attends to:")
        print(f"    BOS: {answer_attention[0]:.3f}")
        print(f"    a (pos 1): {answer_attention[1]:.3f}")
        print(f"    + (pos 2): {answer_attention[2]:.3f}")
        print(f"    b (pos 3): {answer_attention[3]:.3f}")
        print(f"    = (pos 4): {answer_attention[4]:.3f}")

    # Final hypothesis
    print("\n" + "="*80)
    print("HYPOTHESIS")
    print("="*80)

    if fft_peak_ratios.mean() > 3.0:
        print("\nMost likely algorithm: FOURIER CIRCUITS")
        print("The model uses discrete Fourier basis to represent numbers")
    else:
        print("\nMost likely algorithm: NON-FOURIER APPROACH")
        print("Possible alternatives:")
        print("  - Lookup table with learned compression")
        print("  - Polynomial basis representation")
        print("  - Dense distributed representation")
        print("  - Other algebraic structure")

    print("\nInvestigation complete!")


if __name__ == '__main__':
    main()
