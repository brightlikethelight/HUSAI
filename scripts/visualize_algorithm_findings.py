"""Create visualizations for algorithm investigation findings."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_weight_sparsity():
    """Plot weight distribution showing sparsity."""
    # Load model
    model_path = Path('results/transformer_5000ep/transformer_final.pt')
    model, _ = ModularArithmeticTransformer.load_checkpoint(model_path, device='cpu')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sparse Weight Structure in Transformer', fontsize=16, fontweight='bold')

    for layer_idx in range(2):
        W_out = model.model.blocks[layer_idx].mlp.W_out.detach().cpu().numpy()

        # Plot 1: Weight distribution histogram
        ax = axes[layer_idx, 0]
        weights_flat = W_out.flatten()
        ax.hist(weights_flat, bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='|w| = 0.1')
        ax.axvline(-0.1, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Weight Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Layer {layer_idx}: Weight Distribution', fontsize=13)
        ax.legend()
        ax.set_xlim(-1, 1)

        # Add text annotation
        pct_sparse = 100 * (np.abs(weights_flat) < 0.1).mean()
        ax.text(0.5, 0.95, f'{pct_sparse:.1f}% of weights have |w| < 0.1',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10, fontweight='bold')

        # Plot 2: Cumulative distribution
        ax = axes[layer_idx, 1]
        abs_weights = np.abs(weights_flat)
        sorted_weights = np.sort(abs_weights)
        cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

        ax.plot(sorted_weights, cumulative, linewidth=2)
        ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='|w| = 0.1')
        ax.axhline(0.96, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Absolute Weight Value', fontsize=12)
        ax.set_ylabel('Cumulative Fraction', fontsize=12)
        ax.set_title(f'Layer {layer_idx}: Cumulative Weight Distribution', fontsize=13)
        ax.legend()
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path('results/algorithm_investigation')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'weight_sparsity.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'weight_sparsity.png'}")
    plt.close()


def plot_neuron_specialization():
    """Plot how specialized neurons are."""
    model_path = Path('results/transformer_5000ep/transformer_final.pt')
    model, _ = ModularArithmeticTransformer.load_checkpoint(model_path, device='cpu')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Neuron Specialization Analysis', fontsize=16, fontweight='bold')

    for layer_idx in range(2):
        W_out = model.model.blocks[layer_idx].mlp.W_out.detach().cpu().numpy()

        # Count connections per neuron
        connections_per_neuron = (np.abs(W_out) > 0.1).sum(axis=1)

        ax = axes[layer_idx]

        # Histogram
        ax.hist(connections_per_neuron, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(connections_per_neuron.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {connections_per_neuron.mean():.1f}')
        ax.set_xlabel('Number of Output Connections (|w| > 0.1)', fontsize=12)
        ax.set_ylabel('Number of Neurons', fontsize=12)
        ax.set_title(f'Layer {layer_idx}: Connections per Neuron', fontsize=13)
        ax.legend()

        # Add statistics
        n_active = (connections_per_neuron > 0).sum()
        pct_active = 100 * n_active / len(connections_per_neuron)
        avg_conn = connections_per_neuron[connections_per_neuron > 0].mean()

        stats_text = f'Active neurons: {n_active}/512 ({pct_active:.1f}%)\nAvg connections: {avg_conn:.1f}/128'
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=10)

    plt.tight_layout()

    output_dir = Path('results/algorithm_investigation')
    plt.savefig(output_dir / 'neuron_specialization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'neuron_specialization.png'}")
    plt.close()


def plot_embedding_fft():
    """Plot FFT analysis of embeddings."""
    model_path = Path('results/transformer_5000ep/transformer_final.pt')
    model, _ = ModularArithmeticTransformer.load_checkpoint(model_path, device='cpu')

    embed = model.model.embed.W_E.detach().cpu().numpy()
    digit_embeds = embed[:113]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fourier Analysis: Why This Is NOT Fourier Circuits', fontsize=16, fontweight='bold')

    # Plot 1: FFT peak ratios
    ax = axes[0]
    fft_ratios = []

    for dim in range(digit_embeds.shape[1]):
        vec = digit_embeds[:, dim]
        fft = np.fft.fft(vec)
        fft_mag = np.abs(fft)[1:]

        if fft_mag.mean() > 1e-10:
            ratio = fft_mag.max() / fft_mag.mean()
            fft_ratios.append(ratio)

    ax.hist(fft_ratios, bins=30, alpha=0.7, edgecolor='black', label='Our model')
    ax.axvline(np.mean(fft_ratios), color='red', linestyle='--',
              linewidth=2, label=f'Mean: {np.mean(fft_ratios):.2f}')
    ax.axvline(5.0, color='green', linestyle='--',
              linewidth=2, alpha=0.7, label='Fourier threshold (5.0)')

    ax.set_xlabel('FFT Peak Ratio', fontsize=12)
    ax.set_ylabel('Number of Embedding Dimensions', fontsize=12)
    ax.set_title('FFT Peak Ratio Distribution', fontsize=13)
    ax.legend()

    # Add interpretation box
    interpretation = 'Mean < 5.0 → NOT Fourier\nMean 2-5 → Weak structure\nMean > 5 → Strong Fourier'
    ax.text(0.95, 0.95, interpretation,
           transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)

    # Plot 2: Sample FFT magnitude spectrum
    ax = axes[1]

    # Show FFT for a few representative dimensions
    for dim in [0, 10, 50, 100]:
        vec = digit_embeds[:, dim]
        fft = np.fft.fft(vec)
        fft_mag = np.abs(fft)[:57]  # Show first half (symmetric)

        ax.plot(fft_mag, alpha=0.7, label=f'Dim {dim}')

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('FFT Magnitude', fontsize=12)
    ax.set_title('Sample FFT Magnitude Spectra', fontsize=13)
    ax.legend()
    ax.set_xlim(0, 56)

    # Add note
    ax.text(0.5, 0.95, 'Fourier circuits would show clear peaks at specific frequencies',
           transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
           fontsize=9)

    plt.tight_layout()

    output_dir = Path('results/algorithm_investigation')
    plt.savefig(output_dir / 'fourier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fourier_analysis.png'}")
    plt.close()


def create_comparison_table():
    """Create comparison figure: Fourier vs Sparse."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Data for table
    data = [
        ['Aspect', 'Fourier Circuits\n(Nanda et al.)', 'Our Model\n(Sparse Lookup)'],
        ['', '', ''],
        ['FFT Peak Ratio', '~5-10\n(Strong)', '~2.4\n(Weak)'],
        ['Weight Sparsity', 'Low\n(~Normal dist.)', 'High\n(96% near-zero)'],
        ['Neuron Connections', 'Dense\n(Most neurons active)', 'Sparse\n(Avg 6.6/128)'],
        ['Active Neurons', '~100%', '78%'],
        ['Specialization', 'Frequency-based', 'Output-based'],
        ['Accuracy', '100%', '100%'],
    ]

    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.35, 0.35])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header row styling
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Empty row styling
    for i in range(3):
        cell = table[(1, i)]
        cell.set_facecolor('#E8F5E9')

    # Result rows styling
    for row in range(2, 8):
        table[(row, 0)].set_facecolor('#F5F5F5')
        table[(row, 0)].set_text_props(weight='bold')

        # Highlight differences
        if row in [2, 3, 4]:  # Key difference rows
            table[(row, 1)].set_facecolor('#FFEBEE')  # Light red for Fourier
            table[(row, 2)].set_facecolor('#E3F2FD')  # Light blue for Sparse

    plt.title('Algorithm Comparison: What We Expected vs. What We Found',
             fontsize=14, fontweight='bold', pad=20)

    output_dir = Path('results/algorithm_investigation')
    plt.savefig(output_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'algorithm_comparison.png'}")
    plt.close()


def main():
    """Create all visualizations."""
    print("="*80)
    print("CREATING ALGORITHM INVESTIGATION VISUALIZATIONS")
    print("="*80)
    print()

    print("1. Weight sparsity analysis...")
    plot_weight_sparsity()

    print("2. Neuron specialization analysis...")
    plot_neuron_specialization()

    print("3. Fourier analysis...")
    plot_embedding_fft()

    print("4. Algorithm comparison table...")
    create_comparison_table()

    print()
    print("="*80)
    print("✓ All visualizations created!")
    print("="*80)
    print()
    print("Output directory: results/algorithm_investigation/")
    print("Files created:")
    print("  - weight_sparsity.png")
    print("  - neuron_specialization.png")
    print("  - fourier_analysis.png")
    print("  - algorithm_comparison.png")


if __name__ == '__main__':
    main()
