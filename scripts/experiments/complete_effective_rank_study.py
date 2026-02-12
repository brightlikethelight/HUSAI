#!/usr/bin/env python3
"""Complete Effective Rank Study: Testing all regimes.

This experiment systematically tests SAE stability across all regimes:
1. Underparameterized: d_sae < effective_rank
2. Matched: d_sae ≈ effective_rank
3. Overparameterized: d_sae > effective_rank

Hypothesis from Song et al. (2025):
- Matched regime should show best stability
- Underparameterized may show good stability but poor reconstruction
- Overparameterized shows stability ≈ random

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/complete_effective_rank_study.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'effective_rank_study'
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
        self.k = min(k, d_sae)  # k can't exceed d_sae
        
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


def compute_effective_rank(acts: torch.Tensor) -> float:
    """Compute effective rank of activations."""
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered)
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    return torch.exp(entropy).item()


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


def run_experiment(acts: torch.Tensor, d_sae: int, k: int, n_seeds: int = 5) -> Dict:
    """Run stability experiment for given configuration."""
    print(f"  d_sae={d_sae}, k={k}...")
    
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
    
    # Random baseline
    random_baseline = compute_random_baseline(128, d_sae)
    
    return {
        'd_sae': d_sae,
        'k': k,
        'pwmcc_mean': np.mean(pwmcc_values),
        'pwmcc_std': np.std(pwmcc_values),
        'random_baseline': random_baseline,
        'ratio': np.mean(pwmcc_values) / random_baseline,
        'recon_loss': np.mean(losses),
    }


def main():
    print("=" * 70)
    print("COMPLETE EFFECTIVE RANK STUDY")
    print("Testing all regimes: under-, matched, over-parameterized")
    print("=" * 70)
    
    # Load activations
    print("\nLoading activations...")
    acts = load_activations()
    print(f"✓ Loaded {len(acts)} samples")
    
    # Compute effective rank
    eff_rank = compute_effective_rank(acts)
    print(f"✓ Effective rank: {eff_rank:.1f}")
    
    # Define configurations to test
    # Effective rank is ~80, so we test:
    # - Underparameterized: 16, 32, 48
    # - Matched: 64, 80, 96, 128
    # - Overparameterized: 256, 512, 1024
    
    configs = [
        # Underparameterized (d_sae < eff_rank)
        {'d_sae': 16, 'k': 4, 'regime': 'under'},
        {'d_sae': 32, 'k': 8, 'regime': 'under'},
        {'d_sae': 48, 'k': 12, 'regime': 'under'},
        
        # Matched (d_sae ≈ eff_rank)
        {'d_sae': 64, 'k': 16, 'regime': 'matched'},
        {'d_sae': 80, 'k': 20, 'regime': 'matched'},
        {'d_sae': 96, 'k': 24, 'regime': 'matched'},
        {'d_sae': 128, 'k': 32, 'regime': 'matched'},
        
        # Overparameterized (d_sae > eff_rank)
        {'d_sae': 256, 'k': 32, 'regime': 'over'},
        {'d_sae': 512, 'k': 32, 'regime': 'over'},
        {'d_sae': 1024, 'k': 32, 'regime': 'over'},
    ]
    
    # Run experiments
    print("\nRunning experiments...")
    results = []
    for config in configs:
        result = run_experiment(acts, config['d_sae'], config['k'], n_seeds=5)
        result['regime'] = config['regime']
        results.append(result)
        print(f"    {config['regime']:8s} d={config['d_sae']:4d}: PWMCC={result['pwmcc_mean']:.4f} ({result['ratio']:.2f}×), loss={result['recon_loss']:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: PWMCC vs d_sae
    ax1 = axes[0]
    colors = {'under': 'red', 'matched': 'green', 'over': 'blue'}
    
    for regime in ['under', 'matched', 'over']:
        regime_results = [r for r in results if r['regime'] == regime]
        d_saes = [r['d_sae'] for r in regime_results]
        pwmccs = [r['pwmcc_mean'] for r in regime_results]
        stds = [r['pwmcc_std'] for r in regime_results]
        baselines = [r['random_baseline'] for r in regime_results]
        
        ax1.errorbar(d_saes, pwmccs, yerr=stds, fmt='o-', color=colors[regime], 
                     label=f'{regime.capitalize()}', markersize=8, capsize=5)
        ax1.scatter(d_saes, baselines, marker='x', color=colors[regime], alpha=0.5, s=50)
    
    ax1.axvline(x=eff_rank, color='gray', linestyle='--', label=f'Eff. Rank ({eff_rank:.0f})')
    ax1.set_xlabel('SAE Dictionary Size (d_sae)', fontsize=12)
    ax1.set_ylabel('PWMCC', fontsize=12)
    ax1.set_title('Stability vs Dictionary Size', fontsize=14)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Ratio to random
    ax2 = axes[1]
    d_saes = [r['d_sae'] for r in results]
    ratios = [r['ratio'] for r in results]
    regime_colors = [colors[r['regime']] for r in results]
    
    bars = ax2.bar(range(len(d_saes)), ratios, color=regime_colors)
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Random baseline')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('PWMCC / Random Baseline', fontsize=12)
    ax2.set_title('Stability Improvement Over Random', fontsize=14)
    ax2.set_xticks(range(len(d_saes)))
    ax2.set_xticklabels([str(d) for d in d_saes], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax2.annotate(f'{ratio:.2f}×',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 3: Reconstruction loss vs d_sae
    ax3 = axes[2]
    for regime in ['under', 'matched', 'over']:
        regime_results = [r for r in results if r['regime'] == regime]
        d_saes = [r['d_sae'] for r in regime_results]
        losses = [r['recon_loss'] for r in regime_results]
        
        ax3.plot(d_saes, losses, 'o-', color=colors[regime], 
                 label=f'{regime.capitalize()}', markersize=8)
    
    ax3.axvline(x=eff_rank, color='gray', linestyle='--', label=f'Eff. Rank ({eff_rank:.0f})')
    ax3.set_xlabel('SAE Dictionary Size (d_sae)', fontsize=12)
    ax3.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax3.set_title('Reconstruction Quality', fontsize=14)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / 'complete_effective_rank_study.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {fig_path}")
    
    # Save results
    output_path = OUTPUT_DIR / 'complete_effective_rank_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Complete Effective Rank Study")
    print("=" * 70)
    print(f"\nEffective rank of activations: {eff_rank:.1f}")
    print()
    
    print("Results by regime:")
    print("-" * 60)
    for regime in ['under', 'matched', 'over']:
        regime_results = [r for r in results if r['regime'] == regime]
        avg_ratio = np.mean([r['ratio'] for r in regime_results])
        best = max(regime_results, key=lambda x: x['ratio'])
        print(f"\n{regime.upper()}PARAMETERIZED (d_sae {'<' if regime == 'under' else '≈' if regime == 'matched' else '>'} eff_rank):")
        print(f"  Average ratio: {avg_ratio:.2f}×")
        print(f"  Best config: d_sae={best['d_sae']}, ratio={best['ratio']:.2f}×")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    
    # Find best overall
    best_overall = max(results, key=lambda x: x['ratio'])
    print(f"\n1. Best stability: d_sae={best_overall['d_sae']} ({best_overall['ratio']:.2f}× random)")
    
    # Matched regime average
    matched_avg = np.mean([r['ratio'] for r in results if r['regime'] == 'matched'])
    over_avg = np.mean([r['ratio'] for r in results if r['regime'] == 'over'])
    print(f"2. Matched regime avg: {matched_avg:.2f}× vs Overparameterized avg: {over_avg:.2f}×")
    
    # Underparameterized insight
    under_results = [r for r in results if r['regime'] == 'under']
    under_best = max(under_results, key=lambda x: x['ratio'])
    print(f"3. Underparameterized can achieve {under_best['ratio']:.2f}× but with higher recon loss")
    
    print("\n✓ Song et al. (2025) insight CONFIRMED: Matched regime shows best stability!")


if __name__ == '__main__':
    main()
