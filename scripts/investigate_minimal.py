"""Minimal investigation focusing on key questions."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_accuracy(model, dataset, modulus, n_samples=500):
    """Test model accuracy."""
    print("Testing model accuracy...")

    correct = 0
    with torch.no_grad():
        indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        for idx in indices:
            tokens, label = dataset[int(idx)]
            tokens = tokens.unsqueeze(0)

            logits = model(tokens)
            pred = logits[0, -2, :modulus].argmax()

            if pred.item() == label.item():
                correct += 1

    acc = correct / n_samples
    print(f"  Accuracy: {acc:.4f} ({correct}/{n_samples})")
    return acc


def analyze_fourier_structure(model, modulus):
    """Analyze if embeddings have Fourier structure."""
    print("\nAnalyzing Fourier structure...")

    # Get embeddings
    embed = model.model.embed.W_E.detach().cpu().numpy()
    digit_embeds = embed[:modulus]

    # Method 1: FFT peak ratio
    print("  Method 1: FFT peak analysis")
    fft_ratios = []

    for dim in range(digit_embeds.shape[1]):
        vec = digit_embeds[:, dim]

        fft = np.fft.fft(vec)
        fft_mag = np.abs(fft)[1:]  # Exclude DC

        if fft_mag.mean() > 1e-10:
            ratio = fft_mag.max() / fft_mag.mean()
            fft_ratios.append(ratio)

    fft_mean = np.mean(fft_ratios)
    print(f"    FFT peak ratio (mean): {fft_mean:.3f}")
    print(f"    Interpretation: <2.0=None, 2-5=Weak, >5=Strong")

    # Method 2: Check for specific Fourier frequencies
    print("  Method 2: Frequency component analysis")

    # Count dimensions with strong frequency components
    strong_freq_count = 0

    for dim in range(digit_embeds.shape[1]):
        vec = digit_embeds[:, dim]

        fft = np.fft.fft(vec)
        fft_mag = np.abs(fft)

        # Check if any single frequency dominates
        fft_mag_sorted = np.sort(fft_mag)[::-1]
        if len(fft_mag) > 1:
            # Ratio of largest to second largest
            if fft_mag_sorted[1] > 0:
                dom_ratio = fft_mag_sorted[0] / fft_mag_sorted[1]
                if dom_ratio > 2.0:  # One frequency dominates
                    strong_freq_count += 1

    frac_strong_freq = strong_freq_count / digit_embeds.shape[1]
    print(f"    Dimensions with dominant frequency: {strong_freq_count}/{digit_embeds.shape[1]}")
    print(f"    Fraction: {frac_strong_freq:.3f}")

    # Determine if Fourier
    has_fourier = fft_mean > 3.0 or frac_strong_freq > 0.3

    return {
        'fft_mean': fft_mean,
        'frac_strong_freq': frac_strong_freq,
        'has_fourier': has_fourier
    }


def analyze_mlp_sparsity(model):
    """Analyze MLP weight sparsity."""
    print("\nAnalyzing MLP structure...")

    results = []

    for layer_idx in range(model.config.n_layers):
        W_in = model.model.blocks[layer_idx].mlp.W_in.detach().cpu().numpy()
        W_out = model.model.blocks[layer_idx].mlp.W_out.detach().cpu().numpy()

        # Sparsity
        sparsity_in = (np.abs(W_in) < 0.1).mean()
        sparsity_out = (np.abs(W_out) < 0.1).mean()

        # Rank
        s_out = np.linalg.svd(W_out, compute_uv=False)
        total_var = (s_out ** 2).sum()
        cumvar = np.cumsum(s_out ** 2)
        n_comp_90 = np.where(cumvar / total_var >= 0.9)[0][0] + 1
        rank_ratio = n_comp_90 / min(W_out.shape)

        print(f"  Layer {layer_idx}:")
        print(f"    Sparsity (|w|<0.1): W_in={sparsity_in:.3f}, W_out={sparsity_out:.3f}")
        print(f"    Effective rank: {n_comp_90}/{min(W_out.shape)} ({rank_ratio:.3f})")

        results.append({
            'layer': layer_idx,
            'sparsity_out': sparsity_out,
            'rank_ratio': rank_ratio
        })

    return results


def check_attention_patterns(model, dataset, modulus):
    """Check what the attention focuses on."""
    print("\nAnalyzing attention patterns...")

    # Get a sample
    tokens, _ = dataset[0]
    tokens = tokens.unsqueeze(0)

    with torch.no_grad():
        _, cache = model.model.run_with_cache(tokens)

    # Analyze each layer
    for layer in range(model.config.n_layers):
        attn = cache[f'blocks.{layer}.attn.hook_pattern'][0]  # [n_heads, seq, seq]
        avg_attn = attn.mean(0).cpu().numpy()  # [seq, seq]

        # What does answer position (5) attend to?
        # Sequence: [BOS, a, +, b, =, c, EOS]
        answer_attn = avg_attn[5]

        print(f"  Layer {layer}: Answer attends to a={answer_attn[1]:.3f}, b={answer_attn[3]:.3f}")

    return True


def main():
    print("="*80)
    print("MINIMAL ALGORITHM INVESTIGATION")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model_path = Path(__file__).parent.parent / 'results' / 'transformer_5000ep' / 'transformer_final.pt'
    model, extras = ModularArithmeticTransformer.load_checkpoint(model_path, device='cpu')
    model.eval()

    print(f"Model: {model.config.n_layers} layers, {model.config.d_model} dim")
    print(f"Trained: {extras['epoch']} epochs")

    # Create dataset
    print("\nCreating dataset...")
    modulus = 113
    dataset = ModularArithmeticDataset(modulus=modulus, fraction=1.0, seed=42)

    # Run analyses
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80 + "\n")

    # 1. Test accuracy
    accuracy = test_accuracy(model, dataset, modulus)

    # 2. Fourier analysis
    fourier_results = analyze_fourier_structure(model, modulus)

    # 3. MLP analysis
    mlp_results = analyze_mlp_sparsity(model)

    # 4. Attention analysis
    check_attention_patterns(model, dataset, modulus)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n1. ACCURACY: {accuracy:.4f} - {'✓ PASS' if accuracy > 0.95 else '✗ FAIL'}")

    print(f"\n2. FOURIER STRUCTURE:")
    print(f"   FFT peak ratio: {fourier_results['fft_mean']:.3f}")
    print(f"   Dominant frequency fraction: {fourier_results['frac_strong_freq']:.3f}")
    print(f"   Has Fourier: {'YES ✓' if fourier_results['has_fourier'] else 'NO ✗'}")

    print(f"\n3. MLP STRUCTURE:")
    final_layer = mlp_results[-1]
    is_sparse = final_layer['sparsity_out'] > 0.5
    is_low_rank = final_layer['rank_ratio'] < 0.3
    print(f"   Sparse: {'YES' if is_sparse else 'NO'} (sparsity={final_layer['sparsity_out']:.3f})")
    print(f"   Low rank: {'YES' if is_low_rank else 'NO'} (ratio={final_layer['rank_ratio']:.3f})")

    print(f"\n4. HYPOTHESIS:")

    if fourier_results['has_fourier']:
        print("   → FOURIER CIRCUITS")
        print("     Uses discrete Fourier transform for modular arithmetic")
    elif is_sparse or is_low_rank:
        print("   → LOOKUP TABLE / COMPRESSED REPRESENTATION")
        print("     Uses structured weights to map inputs to outputs")
    else:
        print("   → DENSE DISTRIBUTED REPRESENTATION")
        print("     No clear mathematical structure (e.g., learned features)")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'algorithm_investigation'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'accuracy': accuracy,
        'fourier': fourier_results,
        'mlp': mlp_results,
    }

    torch.save(summary, output_dir / 'investigation_results.pt')

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(f"ALGORITHM INVESTIGATION SUMMARY\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"FFT peak ratio: {fourier_results['fft_mean']:.3f}\n")
        f.write(f"Has Fourier: {fourier_results['has_fourier']}\n")
        f.write(f"\n")

        if fourier_results['has_fourier']:
            f.write(f"HYPOTHESIS: Fourier Circuits\n")
        elif is_sparse or is_low_rank:
            f.write(f"HYPOTHESIS: Lookup Table\n")
        else:
            f.write(f"HYPOTHESIS: Dense Representation\n")

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
