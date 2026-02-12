#!/usr/bin/env python3
"""Feature-Level Stability Analysis: Which features are stable and why?

NOVEL RESEARCH QUESTION:
Prior work measures aggregate stability (PWMCC). But which INDIVIDUAL features
are stable vs unstable? And what predicts feature-level stability?

Key hypotheses to test:
1. FREQUENCY HYPOTHESIS: Frequent features (high activation rate) are more stable
   - Song et al. (2025) showed this on LLMs
   - Does it hold on algorithmic tasks without semantic structure?

2. MAGNITUDE HYPOTHESIS: Features with larger activations are more stable
   - Larger activations = stronger signal = better constrained

3. DECODER NORM HYPOTHESIS: Features with larger decoder norms are more stable
   - Larger norms = more "important" features

4. CORRELATION HYPOTHESIS: Features that correlate with task variables are more stable
   - Task-relevant features should be more constrained

This analysis will reveal:
- The distribution of feature-level stability
- What predicts which features are stable
- Whether stability is uniform or concentrated in certain features

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/feature_level_stability_analysis.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'feature_level_stability'
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


def load_activations_with_labels() -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Load transformer activations with task labels (a, b, answer)."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format='sequence')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    acts = []
    a_vals = []
    b_vals = []
    answers = []
    
    with torch.no_grad():
        for batch, labels in dataloader:
            a = model.get_activations(batch, layer=1)[:, -2, :]
            acts.append(a)
            # Extract a, b from batch (tokens 0 and 2)
            a_vals.append(batch[:, 0].numpy())
            b_vals.append(batch[:, 2].numpy())
            answers.append(labels.numpy())
    
    return (
        torch.cat(acts, dim=0),
        np.concatenate(a_vals),
        np.concatenate(b_vals),
        np.concatenate(answers)
    )


def train_sae(acts: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30) -> TopKSAE:
    """Train SAE and return model."""
    torch.manual_seed(seed)
    
    sae = TopKSAE(d_model=128, d_sae=d_sae, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    
    dataset = TensorDataset(acts)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        for (batch,) in dataloader:
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
    
    return sae


def compute_feature_properties(sae: TopKSAE, acts: torch.Tensor, 
                                a_vals: np.ndarray, b_vals: np.ndarray, 
                                answers: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute various properties for each feature."""
    sae.eval()
    with torch.no_grad():
        latents = sae.encode(acts)
    
    latents_np = latents.numpy()
    
    # 1. Activation frequency (how often each feature is active)
    active = (np.abs(latents_np) > 1e-6).astype(float)
    frequencies = active.mean(axis=0)
    
    # 2. Mean activation magnitude (when active)
    mean_magnitudes = np.zeros(sae.d_sae)
    for i in range(sae.d_sae):
        active_mask = active[:, i] > 0
        if active_mask.sum() > 0:
            mean_magnitudes[i] = np.abs(latents_np[active_mask, i]).mean()
    
    # 3. Decoder norm
    decoder_norms = torch.norm(sae.decoder.weight.data, dim=0).numpy()
    
    # 4. Correlation with task variables
    corr_with_a = np.zeros(sae.d_sae)
    corr_with_b = np.zeros(sae.d_sae)
    corr_with_answer = np.zeros(sae.d_sae)
    
    for i in range(sae.d_sae):
        if frequencies[i] > 0.01:  # Only compute for features that activate sometimes
            corr_with_a[i] = np.abs(np.corrcoef(latents_np[:, i], a_vals)[0, 1])
            corr_with_b[i] = np.abs(np.corrcoef(latents_np[:, i], b_vals)[0, 1])
            corr_with_answer[i] = np.abs(np.corrcoef(latents_np[:, i], answers)[0, 1])
    
    # Handle NaN correlations
    corr_with_a = np.nan_to_num(corr_with_a)
    corr_with_b = np.nan_to_num(corr_with_b)
    corr_with_answer = np.nan_to_num(corr_with_answer)
    
    return {
        'frequency': frequencies,
        'mean_magnitude': mean_magnitudes,
        'decoder_norm': decoder_norms,
        'corr_with_a': corr_with_a,
        'corr_with_b': corr_with_b,
        'corr_with_answer': corr_with_answer,
        'max_task_corr': np.maximum.reduce([corr_with_a, corr_with_b, corr_with_answer])
    }


def compute_feature_level_stability(saes: List[TopKSAE]) -> np.ndarray:
    """Compute stability for each feature (average max cosine sim across all pairs)."""
    n_seeds = len(saes)
    d_sae = saes[0].d_sae
    
    # For each feature in each SAE, compute its max cosine sim to any feature in other SAEs
    all_stabilities = []
    
    for i in range(n_seeds):
        d_i = saes[i].decoder.weight.data
        d_i_norm = F.normalize(d_i, dim=0)
        
        feature_stabilities = np.zeros(d_sae)
        count = 0
        
        for j in range(n_seeds):
            if i == j:
                continue
            
            d_j = saes[j].decoder.weight.data
            d_j_norm = F.normalize(d_j, dim=0)
            
            # Cosine similarity matrix
            cos_sim = (d_i_norm.T @ d_j_norm).abs().numpy()
            
            # For each feature in SAE i, find its best match in SAE j
            max_sim = cos_sim.max(axis=1)
            feature_stabilities += max_sim
            count += 1
        
        feature_stabilities /= count
        all_stabilities.append(feature_stabilities)
    
    # Average across all SAEs
    return np.mean(all_stabilities, axis=0)


def main():
    print("=" * 70)
    print("FEATURE-LEVEL STABILITY ANALYSIS")
    print("Which features are stable and why?")
    print("=" * 70)
    
    # Load activations with labels
    print("\nLoading activations with task labels...")
    acts, a_vals, b_vals, answers = load_activations_with_labels()
    print(f"✓ Loaded {len(acts)} samples")
    
    # Configuration
    d_sae = 128
    k = 32
    n_seeds = 5
    
    print(f"\nTraining {n_seeds} SAEs (d_sae={d_sae}, k={k})...")
    saes = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...")
        sae = train_sae(acts, d_sae, k, seed)
        saes.append(sae)
    
    # Compute feature-level stability
    print("\nComputing feature-level stability...")
    feature_stability = compute_feature_level_stability(saes)
    
    # Compute feature properties (from first SAE as representative)
    print("Computing feature properties...")
    properties = compute_feature_properties(saes[0], acts, a_vals, b_vals, answers)
    
    # Analyze correlations between stability and properties
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Filter to features that are active at least sometimes
    active_mask = properties['frequency'] > 0.01
    n_active = active_mask.sum()
    print(f"\nAnalyzing {n_active} active features (frequency > 1%)")
    
    correlations = {}
    for prop_name, prop_values in properties.items():
        if prop_name == 'frequency':
            # Use log frequency for better correlation
            log_freq = np.log10(prop_values[active_mask] + 1e-6)
            r, p = stats.pearsonr(log_freq, feature_stability[active_mask])
            correlations['log_frequency'] = (r, p)
        else:
            r, p = stats.pearsonr(prop_values[active_mask], feature_stability[active_mask])
            correlations[prop_name] = (r, p)
    
    print("\nCorrelation with feature stability:")
    print("-" * 50)
    for prop_name, (r, p) in sorted(correlations.items(), key=lambda x: -abs(x[1][0])):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {prop_name:20s}: r={r:+.3f} (p={p:.4f}) {sig}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Distribution of feature stability
    ax1 = axes[0, 0]
    ax1.hist(feature_stability, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(feature_stability), color='red', linestyle='--', 
                label=f'Mean: {np.mean(feature_stability):.3f}')
    ax1.axvline(x=np.median(feature_stability), color='orange', linestyle='--',
                label=f'Median: {np.median(feature_stability):.3f}')
    ax1.set_xlabel('Feature Stability (max cosine sim)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Feature-Level Stability', fontsize=12)
    ax1.legend()
    
    # Panel 2: Stability vs Frequency
    ax2 = axes[0, 1]
    freq = properties['frequency']
    scatter = ax2.scatter(freq[active_mask], feature_stability[active_mask], 
                          c=properties['max_task_corr'][active_mask], cmap='viridis',
                          alpha=0.6, s=30)
    ax2.set_xlabel('Activation Frequency', fontsize=11)
    ax2.set_ylabel('Feature Stability', fontsize=11)
    ax2.set_title(f'Stability vs Frequency\n(r={correlations["log_frequency"][0]:.3f})', fontsize=12)
    ax2.set_xscale('log')
    plt.colorbar(scatter, ax=ax2, label='Task Correlation')
    
    # Panel 3: Stability vs Mean Magnitude
    ax3 = axes[0, 2]
    mag = properties['mean_magnitude']
    ax3.scatter(mag[active_mask], feature_stability[active_mask], 
                c='steelblue', alpha=0.6, s=30)
    ax3.set_xlabel('Mean Activation Magnitude', fontsize=11)
    ax3.set_ylabel('Feature Stability', fontsize=11)
    r_mag = correlations['mean_magnitude'][0]
    ax3.set_title(f'Stability vs Magnitude\n(r={r_mag:.3f})', fontsize=12)
    
    # Panel 4: Stability vs Task Correlation
    ax4 = axes[1, 0]
    task_corr = properties['max_task_corr']
    ax4.scatter(task_corr[active_mask], feature_stability[active_mask],
                c='coral', alpha=0.6, s=30)
    ax4.set_xlabel('Max Task Correlation', fontsize=11)
    ax4.set_ylabel('Feature Stability', fontsize=11)
    r_task = correlations['max_task_corr'][0]
    ax4.set_title(f'Stability vs Task Relevance\n(r={r_task:.3f})', fontsize=12)
    
    # Panel 5: Stability vs Decoder Norm
    ax5 = axes[1, 1]
    dec_norm = properties['decoder_norm']
    ax5.scatter(dec_norm[active_mask], feature_stability[active_mask],
                c='green', alpha=0.6, s=30)
    ax5.set_xlabel('Decoder Norm', fontsize=11)
    ax5.set_ylabel('Feature Stability', fontsize=11)
    r_dec = correlations['decoder_norm'][0]
    ax5.set_title(f'Stability vs Decoder Norm\n(r={r_dec:.3f})', fontsize=12)
    
    # Panel 6: Summary bar chart of correlations
    ax6 = axes[1, 2]
    prop_names = list(correlations.keys())
    r_values = [correlations[p][0] for p in prop_names]
    colors = ['green' if r > 0 else 'red' for r in r_values]
    
    bars = ax6.barh(range(len(prop_names)), r_values, color=colors, alpha=0.7)
    ax6.set_yticks(range(len(prop_names)))
    ax6.set_yticklabels(prop_names)
    ax6.set_xlabel('Correlation with Stability (r)', fontsize=11)
    ax6.set_title('What Predicts Feature Stability?', fontsize=12)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_xlim(-0.5, 0.5)
    
    # Add significance markers
    for i, (prop, (r, p)) in enumerate(correlations.items()):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax6.annotate(sig, (r + 0.02 if r > 0 else r - 0.05, i), fontsize=10)
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / 'feature_level_stability_analysis.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {fig_path}")
    
    # Save results
    results = {
        'config': {'d_sae': d_sae, 'k': k, 'n_seeds': n_seeds},
        'feature_stability': {
            'mean': float(np.mean(feature_stability)),
            'std': float(np.std(feature_stability)),
            'median': float(np.median(feature_stability)),
            'min': float(np.min(feature_stability)),
            'max': float(np.max(feature_stability)),
        },
        'correlations': {k: {'r': float(v[0]), 'p': float(v[1])} 
                        for k, v in correlations.items()},
        'n_active_features': int(n_active),
    }
    
    output_path = OUTPUT_DIR / 'feature_level_stability_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find strongest predictor
    strongest = max(correlations.items(), key=lambda x: abs(x[1][0]))
    print(f"\n1. STRONGEST PREDICTOR: {strongest[0]}")
    print(f"   Correlation: r={strongest[1][0]:.3f} (p={strongest[1][1]:.4f})")
    
    # Test frequency hypothesis
    freq_r, freq_p = correlations['log_frequency']
    print(f"\n2. FREQUENCY HYPOTHESIS:")
    if freq_r > 0.1 and freq_p < 0.05:
        print(f"   ✓ SUPPORTED: Frequent features are more stable (r={freq_r:.3f})")
    else:
        print(f"   ✗ NOT SUPPORTED: No significant relationship (r={freq_r:.3f})")
    
    # Test task correlation hypothesis
    task_r, task_p = correlations['max_task_corr']
    print(f"\n3. TASK RELEVANCE HYPOTHESIS:")
    if task_r > 0.1 and task_p < 0.05:
        print(f"   ✓ SUPPORTED: Task-relevant features are more stable (r={task_r:.3f})")
    else:
        print(f"   ✗ NOT SUPPORTED: No significant relationship (r={task_r:.3f})")
    
    # Distribution insight
    print(f"\n4. STABILITY DISTRIBUTION:")
    print(f"   Mean: {np.mean(feature_stability):.3f}")
    print(f"   Std:  {np.std(feature_stability):.3f}")
    print(f"   Range: [{np.min(feature_stability):.3f}, {np.max(feature_stability):.3f}]")
    
    # Identify most stable features
    top_stable_idx = np.argsort(feature_stability)[-5:][::-1]
    print(f"\n5. MOST STABLE FEATURES:")
    for idx in top_stable_idx:
        print(f"   Feature {idx}: stability={feature_stability[idx]:.3f}, "
              f"freq={properties['frequency'][idx]:.3f}, "
              f"task_corr={properties['max_task_corr'][idx]:.3f}")


if __name__ == '__main__':
    main()
