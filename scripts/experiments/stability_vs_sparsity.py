#!/usr/bin/env python3
"""Stability vs Sparsity: Testing if stability peaks at optimal L0.

NOVEL RESEARCH QUESTION:
The "Sparse but Wrong" paper (arXiv:2508.16560) shows that:
- Low L0 → feature hedging → wrong features
- They propose decoder projection to detect this

We hypothesize that:
- Low L0 → hedging → UNSTABLE features (different seeds hedge differently)
- Optimal L0 → correct features → STABLE features
- High L0 → polysemanticity → UNSTABLE features

If true, STABILITY can be used as a proxy for feature correctness!

Experimental Design:
- Fix d_sae = 128 (matched to effective rank ~80)
- Vary k = [4, 8, 16, 24, 32, 48, 64]
- Train 5 SAEs per k with different seeds
- Measure PWMCC at each k

Expected Result:
- Stability should PEAK at some optimal k
- This would be the first paper to connect stability to sparsity selection

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/stability_vs_sparsity.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'stability_vs_sparsity'
FIGURES_DIR = BASE_DIR / 'figures'

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cpu'


class TopKSAE(nn.Module):
    """TopK SAE with proper decoder normalization."""
    
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = min(k, d_sae)
        
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()
    
    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_act, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        return latents
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        recon = self.decoder(latents)
        return recon, latents


def compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder matrices."""
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_feature_level_stability(d1: torch.Tensor, d2: torch.Tensor) -> np.ndarray:
    """Compute per-feature stability (max cosine sim to any feature in other SAE)."""
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    # For each feature in d1, find its best match in d2
    max_sim = cos_sim.abs().max(dim=1)[0].numpy()
    return max_sim


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 10) -> float:
    """Compute random PWMCC baseline."""
    pwmcc_values = []
    for _ in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        pwmcc_values.append(compute_pwmcc(d1, d2))
    return np.mean(pwmcc_values)


def load_activations() -> torch.Tensor:
    """Load transformer activations."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format='sequence')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    acts = []
    with torch.no_grad():
        for batch, _ in dataloader:
            a = model.get_activations(batch, layer=1)[:, -2, :]
            acts.append(a)
    
    return torch.cat(acts, dim=0)


def train_sae(acts: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30) -> Tuple[TopKSAE, float]:
    """Train SAE and return model and reconstruction loss."""
    torch.manual_seed(seed)
    
    sae = TopKSAE(d_model=128, d_sae=d_sae, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    
    dataset = TensorDataset(acts)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    final_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in dataloader:
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            
            epoch_loss += loss.item()
        
        final_loss = epoch_loss / len(dataloader)
    
    return sae, final_loss


def compute_activation_frequencies(sae: TopKSAE, acts: torch.Tensor) -> np.ndarray:
    """Compute activation frequency for each feature."""
    sae.eval()
    with torch.no_grad():
        latents = sae.encode(acts)
        # Count how often each feature is active (non-zero)
        active = (latents.abs() > 1e-6).float()
        frequencies = active.mean(dim=0).numpy()
    return frequencies


def run_experiment(acts: torch.Tensor, d_sae: int, k: int, n_seeds: int = 5) -> Dict:
    """Run stability experiment for given k."""
    print(f"  k={k}...")
    
    # Train SAEs
    saes = []
    losses = []
    for seed in range(n_seeds):
        sae, loss = train_sae(acts, d_sae, k, seed)
        saes.append(sae)
        losses.append(loss)
    
    # Compute pairwise PWMCC
    pwmcc_values = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            pwmcc = compute_pwmcc(
                saes[i].decoder.weight.data,
                saes[j].decoder.weight.data
            )
            pwmcc_values.append(pwmcc)
    
    # Compute feature-level stability (average across all pairs)
    feature_stabilities = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            stab = compute_feature_level_stability(
                saes[i].decoder.weight.data,
                saes[j].decoder.weight.data
            )
            feature_stabilities.append(stab)
    
    avg_feature_stability = np.mean(feature_stabilities, axis=0)
    
    # Compute activation frequencies (from first SAE)
    frequencies = compute_activation_frequencies(saes[0], acts)
    
    # Random baseline
    random_baseline = compute_random_baseline(128, d_sae)
    
    return {
        'k': k,
        'd_sae': d_sae,
        'pwmcc_mean': np.mean(pwmcc_values),
        'pwmcc_std': np.std(pwmcc_values),
        'random_baseline': random_baseline,
        'ratio': np.mean(pwmcc_values) / random_baseline,
        'recon_loss': np.mean(losses),
        'feature_stability_mean': np.mean(avg_feature_stability),
        'feature_stability_std': np.std(avg_feature_stability),
        'frequencies': frequencies.tolist(),
        'feature_stabilities': avg_feature_stability.tolist(),
    }


def main():
    print("=" * 70)
    print("STABILITY VS SPARSITY EXPERIMENT")
    print("Testing if stability peaks at optimal L0")
    print("=" * 70)
    
    # Load activations
    print("\nLoading activations...")
    acts = load_activations()
    print(f"✓ Loaded {len(acts)} samples")
    
    # Fixed d_sae, varying k
    d_sae = 128  # Matched to effective rank ~80
    k_values = [4, 8, 16, 24, 32, 48, 64]
    
    print(f"\nFixed d_sae={d_sae}, varying k...")
    results = []
    for k in k_values:
        result = run_experiment(acts, d_sae, k, n_seeds=5)
        results.append(result)
        print(f"    k={k:2d}: PWMCC={result['pwmcc_mean']:.4f} ({result['ratio']:.2f}×), loss={result['recon_loss']:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: PWMCC vs k
    ax1 = axes[0, 0]
    ks = [r['k'] for r in results]
    pwmccs = [r['pwmcc_mean'] for r in results]
    stds = [r['pwmcc_std'] for r in results]
    baselines = [r['random_baseline'] for r in results]
    
    ax1.errorbar(ks, pwmccs, yerr=stds, fmt='o-', color='steelblue', 
                 label='Trained PWMCC', markersize=8, capsize=5, linewidth=2)
    ax1.plot(ks, baselines, 'x--', color='gray', label='Random Baseline', markersize=8)
    
    # Find and mark peak
    peak_idx = np.argmax(pwmccs)
    ax1.axvline(x=ks[peak_idx], color='green', linestyle=':', alpha=0.7, 
                label=f'Peak at k={ks[peak_idx]}')
    
    ax1.set_xlabel('Sparsity (k)', fontsize=12)
    ax1.set_ylabel('PWMCC', fontsize=12)
    ax1.set_title('Stability vs Sparsity', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Ratio to random vs k
    ax2 = axes[0, 1]
    ratios = [r['ratio'] for r in results]
    
    bars = ax2.bar(range(len(ks)), ratios, color='steelblue')
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Random baseline')
    
    # Mark peak
    bars[peak_idx].set_color('green')
    
    ax2.set_xlabel('Sparsity (k)', fontsize=12)
    ax2.set_ylabel('PWMCC / Random Baseline', fontsize=12)
    ax2.set_title('Stability Improvement Over Random', fontsize=14)
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax2.annotate(f'{ratio:.2f}×',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 3: Reconstruction loss vs k
    ax3 = axes[1, 0]
    losses = [r['recon_loss'] for r in results]
    
    ax3.plot(ks, losses, 'o-', color='coral', markersize=8, linewidth=2)
    ax3.set_xlabel('Sparsity (k)', fontsize=12)
    ax3.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax3.set_title('Reconstruction vs Sparsity', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Stability vs Reconstruction tradeoff
    ax4 = axes[1, 1]
    
    scatter = ax4.scatter(losses, pwmccs, c=ks, cmap='viridis', s=100, edgecolors='black')
    
    # Add k labels
    for i, (loss, pwmcc, k) in enumerate(zip(losses, pwmccs, ks)):
        ax4.annotate(f'k={k}', (loss, pwmcc), xytext=(5, 5), 
                     textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Reconstruction Loss (MSE)', fontsize=12)
    ax4.set_ylabel('PWMCC', fontsize=12)
    ax4.set_title('Stability-Reconstruction Tradeoff', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Sparsity (k)')
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / 'stability_vs_sparsity.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {fig_path}")
    
    # Save results (without large arrays for JSON)
    output_results = []
    for r in results:
        output_results.append({
            'k': int(r['k']),
            'd_sae': int(r['d_sae']),
            'pwmcc_mean': float(r['pwmcc_mean']),
            'pwmcc_std': float(r['pwmcc_std']),
            'random_baseline': float(r['random_baseline']),
            'ratio': float(r['ratio']),
            'recon_loss': float(r['recon_loss']),
            'feature_stability_mean': float(r['feature_stability_mean']),
            'feature_stability_std': float(r['feature_stability_std']),
        })
    
    output_path = OUTPUT_DIR / 'stability_vs_sparsity_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Stability vs Sparsity")
    print("=" * 70)
    print(f"\nFixed d_sae={d_sae}")
    print()
    
    print("Results by sparsity (k):")
    print("-" * 60)
    for r in results:
        print(f"  k={r['k']:2d}: PWMCC={r['pwmcc_mean']:.4f} ({r['ratio']:.2f}×), loss={r['recon_loss']:.4f}")
    
    # Find optimal k
    optimal_idx = np.argmax([r['ratio'] for r in results])
    optimal_k = results[optimal_idx]['k']
    optimal_ratio = results[optimal_idx]['ratio']
    
    print()
    print("=" * 70)
    print("KEY FINDING:")
    print("=" * 70)
    print(f"\n✓ Stability PEAKS at k={optimal_k} ({optimal_ratio:.2f}× random)")
    print()
    
    if optimal_k != k_values[0] and optimal_k != k_values[-1]:
        print("✓ HYPOTHESIS SUPPORTED: Stability peaks at intermediate sparsity!")
        print("  - Too sparse (low k): Feature hedging → unstable")
        print("  - Too dense (high k): Polysemanticity → unstable")
        print("  - Optimal k: Correct features → stable")
        print()
        print("IMPLICATION: Stability can be used to select optimal L0!")
    else:
        print("✗ Stability does not peak at intermediate k")
        print("  Further investigation needed")


if __name__ == '__main__':
    main()
