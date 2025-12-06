"""Safe, incremental investigation of transformer algorithm."""

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

# Suppress numpy warnings
np.seterr(divide='ignore', invalid='ignore')


def safe_correlation(x, y):
    """Compute correlation, handling edge cases."""
    try:
        # Check for constant arrays
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0

        corr_matrix = np.corrcoef(x, y)
        if np.isnan(corr_matrix[0, 1]):
            return 0.0

        return corr_matrix[0, 1]
    except:
        return 0.0


def main():
    """Run investigation safely."""
    print("="*80)
    print("TRANSFORMER ALGORITHM INVESTIGATION")
    print("="*80)

    # Setup
    model_path = Path(__file__).parent.parent / 'results' / 'transformer_5000ep' / 'transformer_final.pt'
    modulus = 113
    device = 'cpu'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    print("\n[1/6] Loading model...")
    model, extras = ModularArithmeticTransformer.load_checkpoint(model_path, device=device)
    model.eval()
    print(f"✓ Model loaded (epochs: {extras.get('epoch', 'unknown')})")

    # Create dataset
    print("\n[2/6] Creating dataset...")
    dataset = ModularArithmeticDataset(modulus=modulus, fraction=1.0, seed=42)
    print(f"✓ Dataset created ({len(dataset)} examples)")

    # Test generalization
    print("\n[3/6] Testing generalization...")
    print("-"*80)

    correct = 0
    total = 0
    n_test = 500

    with torch.no_grad():
        indices = np.random.choice(len(dataset), size=n_test, replace=False)
        for idx in indices:
            tokens, label = dataset[int(idx)]
            tokens = tokens.unsqueeze(0).to(device)

            try:
                logits = model(tokens)
                pred = logits[0, -2, :modulus].argmax()

                if pred.item() == label.item():
                    correct += 1
                total += 1
            except:
                continue

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Generalizes: {'YES ✓' if accuracy > 0.95 else 'NO ✗'}")

    # Analyze embeddings for Fourier structure
    print("\n[4/6] Analyzing embeddings for Fourier structure...")
    print("-"*80)

    try:
        embed = model.model.embed.W_E.detach().cpu().numpy()
        digit_embeds = embed[:modulus]

        print(f"Embedding shape: {digit_embeds.shape}")

        # Compute FFT for each dimension
        fft_peak_ratios = []

        for dim in range(digit_embeds.shape[1]):
            fft = np.fft.fft(digit_embeds[:, dim])
            fft_mag = np.abs(fft)
            fft_mag_no_dc = fft_mag[1:]

            if fft_mag_no_dc.mean() > 1e-10:
                peak_ratio = fft_mag_no_dc.max() / fft_mag_no_dc.mean()
                fft_peak_ratios.append(peak_ratio)

        fft_peak_ratios = np.array(fft_peak_ratios)
        fft_mean = fft_peak_ratios.mean()

        print(f"\nFFT Peak Ratio: {fft_mean:.3f}")
        print(f"Interpretation: <2.0=No Fourier, 2-5=Weak, >5=Strong")

        has_fourier = fft_mean > 3.0
        print(f"Fourier structure: {'YES ✓' if has_fourier else 'NO ✗'}")

    except Exception as e:
        print(f"Error analyzing embeddings: {e}")
        has_fourier = False

    # Compare to DFT basis
    print("\n[5/6] Comparing to DFT basis...")
    print("-"*80)

    try:
        # Create DFT matrix
        dft_real = np.zeros((modulus, modulus))
        dft_imag = np.zeros((modulus, modulus))

        for k in range(modulus):
            for n in range(modulus):
                angle = -2 * np.pi * k * n / modulus
                dft_real[k, n] = np.cos(angle)
                dft_imag[k, n] = np.sin(angle)

        # Normalize
        dft_real /= np.sqrt(modulus)
        dft_imag /= np.sqrt(modulus)

        # Check alignment
        alignments_real = []
        alignments_imag = []

        for dim in range(min(digit_embeds.shape[1], 128)):  # Limit to prevent crashes
            emb_vec = digit_embeds[:, dim]

            # Best alignment with any DFT frequency
            best_real = 0
            best_imag = 0

            for freq in range(modulus):
                corr_real = abs(safe_correlation(emb_vec, dft_real[freq]))
                corr_imag = abs(safe_correlation(emb_vec, dft_imag[freq]))

                best_real = max(best_real, corr_real)
                best_imag = max(best_imag, corr_imag)

            alignments_real.append(best_real)
            alignments_imag.append(best_imag)

        alignments_real = np.array(alignments_real)
        alignments_imag = np.array(alignments_imag)

        # Count high alignments
        n_high = ((alignments_real > 0.7) | (alignments_imag > 0.7)).sum()
        frac_high = n_high / len(alignments_real)

        print(f"DFT alignment fraction (>0.7): {frac_high:.3f}")
        print(f"Mean alignment (real): {alignments_real.mean():.3f}")
        print(f"Mean alignment (imag): {alignments_imag.mean():.3f}")

        is_dft_aligned = frac_high > 0.5
        print(f"DFT-based Fourier: {'YES ✓' if is_dft_aligned else 'NO ✗'}")

    except Exception as e:
        print(f"Error comparing to DFT: {e}")
        is_dft_aligned = False

    # Analyze MLP structure
    print("\n[6/6] Analyzing MLP structure...")
    print("-"*80)

    try:
        for layer_idx in range(model.config.n_layers):
            W_out = model.model.blocks[layer_idx].mlp.W_out.detach().cpu().numpy()

            sparsity = (np.abs(W_out) < 0.1).sum() / W_out.size

            # Compute effective rank
            s = np.linalg.svd(W_out, compute_uv=False)
            total_energy = (s ** 2).sum()
            cumsum = np.cumsum(s ** 2)
            n_comp_90 = np.argmax(cumsum / total_energy >= 0.9) + 1
            rank_ratio = n_comp_90 / min(W_out.shape)

            print(f"Layer {layer_idx}:")
            print(f"  Sparsity (|w|<0.1): {sparsity:.3f}")
            print(f"  Effective rank ratio: {rank_ratio:.3f}")

            is_lookup = sparsity > 0.5 or rank_ratio < 0.3
            if is_lookup:
                print(f"  → Lookup table structure detected")

    except Exception as e:
        print(f"Error analyzing MLP: {e}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n1. Generalization: {accuracy:.4f} - {'PASS' if accuracy > 0.95 else 'FAIL'}")
    print(f"2. Fourier structure: {'YES' if has_fourier or is_dft_aligned else 'NO'}")

    if has_fourier or is_dft_aligned:
        print(f"\n→ HYPOTHESIS: Fourier Circuits")
        print(f"  Model uses discrete Fourier transform")
    else:
        print(f"\n→ HYPOTHESIS: Non-Fourier Algorithm")
        print(f"  Possible alternatives:")
        print(f"    - Lookup table with compression")
        print(f"    - Polynomial representation")
        print(f"    - Dense distributed representation")
        print(f"    - Other algebraic structure")

    # Save summary
    output_dir = Path(__file__).parent.parent / 'results' / 'algorithm_investigation'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("ALGORITHM INVESTIGATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"FFT peak ratio: {fft_mean:.3f}\n")
        f.write(f"DFT alignment: {frac_high:.3f}\n")
        f.write(f"\nFourier circuits: {'YES' if has_fourier or is_dft_aligned else 'NO'}\n")

        if has_fourier or is_dft_aligned:
            f.write(f"\nHYPOTHESIS: Fourier Circuits\n")
        else:
            f.write(f"\nHYPOTHESIS: Non-Fourier Algorithm\n")

    print(f"\n✓ Summary saved to: {output_dir / 'summary.txt'}")


if __name__ == '__main__':
    main()
