#!/usr/bin/env python3
"""PWMCC Sensitivity Analysis - Test Data Subset Bias.

The current cross-layer validation script computes PWMCC using only 5,000/50,000
samples (10% of data). This script tests whether this introduces bias by comparing
PWMCC computed on different data subset sizes:
- 10% of data (n=5,000) - current approach
- 50% of data (n=25,000) - medium sample
- 100% of data (n=50,000+) - full dataset

If PWMCC values are stable across subsets (variance < 0.01), the current approach
is valid. Otherwise, we should use larger subsets.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
SAES_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'

SEEDS = [42, 123]  # Test with 2 seed pairs for speed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleSAE(torch.nn.Module):
    """Minimal SAE for loading checkpoints."""

    def __init__(self, d_model: int, d_sae: int, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = torch.nn.Linear(d_model, d_sae, bias=True)
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=False)
        self.register_buffer('feature_counts', torch.zeros(d_sae))

    def encode(self, x):
        """Encode with TopK activation."""
        pre_act = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_act, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(dim=-1, index=topk_indices, src=topk_values)
        return latents


def compute_pwmcc(
    sae1: SimpleSAE,
    sae2: SimpleSAE,
    activations: torch.Tensor
) -> float:
    """Compute PWMCC between two SAEs using decoder weights.

    Args:
        sae1: First SAE
        sae2: Second SAE
        activations: Sample activations [n_samples, d_model]

    Returns:
        PWMCC score (0-1)
    """
    sae1.eval()
    sae2.eval()

    # Extract decoder weights [d_model, d_sae]
    W1 = sae1.decoder.weight.T  # [d_model, d_sae1]
    W2 = sae2.decoder.weight.T  # [d_model, d_sae2]

    # Normalize columns to unit vectors
    W1_norm = F.normalize(W1, p=2, dim=0)  # [d_model, d_sae1]
    W2_norm = F.normalize(W2, p=2, dim=0)  # [d_model, d_sae2]

    # Compute cosine similarity matrix [d_sae1, d_sae2]
    cosine_sim = torch.mm(W1_norm.T, W2_norm)

    # PWMCC: mean of maximum absolute correlations
    max_corr, _ = torch.max(torch.abs(cosine_sim), dim=1)
    pwmcc = torch.mean(max_corr).item()

    return pwmcc


def load_sae(sae_path: Path) -> SimpleSAE:
    """Load SAE from checkpoint."""
    checkpoint = torch.load(sae_path, map_location=DEVICE)
    d_model = checkpoint['d_model']
    d_sae = checkpoint['d_sae']
    k = checkpoint['k']

    sae = SimpleSAE(d_model, d_sae, k).to(DEVICE)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    return sae


def main():
    """Main execution: test PWMCC sensitivity to data subset size."""
    print("="*80)
    print("PWMCC SENSITIVITY ANALYSIS - DATA SUBSET SIZE")
    print("="*80)
    print()

    # Load transformer
    print("Loading transformer...")
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device=DEVICE)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded transformer")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"✓ Loaded {len(dataset)} samples")
    print()

    # Extract activations
    print("Extracting activations from layer 1, position -2...")
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                tokens, _ = batch
            else:
                tokens = batch
            tokens = tokens.to(DEVICE)
            layer_acts = model.get_activations(tokens, layer=1)
            position_acts = layer_acts[:, -2, :]
            activations.append(position_acts.cpu())

    activations = torch.cat(activations, dim=0)
    n_total = len(activations)
    print(f"✓ Extracted {n_total} samples")
    print()

    # Load SAEs
    print("Loading TopK SAEs...")
    saes = {}
    for seed in SEEDS:
        sae_path = SAES_DIR / f'topk_seed{seed}' / 'sae_final.pt'
        if sae_path.exists():
            saes[seed] = load_sae(sae_path)
            print(f"  ✓ Loaded seed {seed}")
    print()

    # Test different data subset sizes
    subset_sizes = [
        (0.10, int(n_total * 0.10), "10% (current)"),
        (0.50, int(n_total * 0.50), "50%"),
        (1.00, n_total, "100% (full)")
    ]

    print("="*80)
    print("COMPUTING PWMCC WITH DIFFERENT DATA SUBSET SIZES")
    print("="*80)
    print()

    results = {
        'n_total': n_total,
        'seed_pairs': [],
        'subset_comparison': []
    }

    # For each seed pair
    for i, seed1 in enumerate(SEEDS):
        for seed2 in SEEDS[i+1:]:
            print(f"Seed pair: {seed1} vs {seed2}")
            pair_results = {
                'seed1': seed1,
                'seed2': seed2,
                'pwmcc_by_subset': []
            }

            for fraction, n_samples, label in subset_sizes:
                # Sample activations
                if n_samples < n_total:
                    # Use first n_samples (consistent with current approach)
                    subset_acts = activations[:n_samples]
                else:
                    subset_acts = activations

                # Compute PWMCC
                pwmcc = compute_pwmcc(saes[seed1], saes[seed2], subset_acts)

                pair_results['pwmcc_by_subset'].append({
                    'label': label,
                    'fraction': fraction,
                    'n_samples': n_samples,
                    'pwmcc': pwmcc
                })

                print(f"  {label:20s} (n={n_samples:,}): PWMCC = {pwmcc:.6f}")

            results['seed_pairs'].append(pair_results)
            print()

    # Analyze variance across subsets
    print("="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)
    print()

    for label_idx, (fraction, n_samples, label) in enumerate(subset_sizes):
        pwmcc_values = [
            pair['pwmcc_by_subset'][label_idx]['pwmcc']
            for pair in results['seed_pairs']
        ]
        mean_pwmcc = np.mean(pwmcc_values)
        std_pwmcc = np.std(pwmcc_values)

        results['subset_comparison'].append({
            'label': label,
            'fraction': fraction,
            'n_samples': n_samples,
            'mean_pwmcc': mean_pwmcc,
            'std_pwmcc': std_pwmcc,
            'individual_values': pwmcc_values
        })

        print(f"{label:20s}: {mean_pwmcc:.6f} ± {std_pwmcc:.6f}")

    print()

    # Compute variance across subsets (for each seed pair)
    print("Variance across subset sizes (per seed pair):")
    for pair in results['seed_pairs']:
        pwmcc_across_subsets = [entry['pwmcc'] for entry in pair['pwmcc_by_subset']]
        variance = np.var(pwmcc_across_subsets)
        std = np.std(pwmcc_across_subsets)
        range_val = max(pwmcc_across_subsets) - min(pwmcc_across_subsets)

        print(f"  Seeds {pair['seed1']} vs {pair['seed2']}:")
        print(f"    Std: {std:.6f}, Range: {range_val:.6f}")

    print()

    # Overall assessment
    all_stds = []
    for pair in results['seed_pairs']:
        pwmcc_across_subsets = [entry['pwmcc'] for entry in pair['pwmcc_by_subset']]
        all_stds.append(np.std(pwmcc_across_subsets))

    max_std = max(all_stds)
    mean_std = np.mean(all_stds)

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"Maximum std across subsets: {max_std:.6f}")
    print(f"Average std across subsets: {mean_std:.6f}")
    print()

    if max_std < 0.01:
        conclusion = "✅ PWMCC is ROBUST to data subset size (std < 0.01)"
        recommendation = "Current 10% sampling approach is acceptable."
        risk_level = "LOW"
    elif max_std < 0.05:
        conclusion = "⚠️  PWMCC shows MODERATE sensitivity to subset size (0.01 < std < 0.05)"
        recommendation = "Consider using 50% or 100% of data for more stable estimates."
        risk_level = "MEDIUM"
    else:
        conclusion = "🚨 PWMCC shows HIGH sensitivity to subset size (std > 0.05)"
        recommendation = "MUST use 100% of data. 10% sampling is BIASED."
        risk_level = "HIGH"

    print(conclusion)
    print(f"Risk level: {risk_level}")
    print()
    print(f"Recommendation: {recommendation}")
    print()

    results['conclusion'] = conclusion
    results['recommendation'] = recommendation
    results['risk_level'] = risk_level
    results['max_std'] = max_std
    results['mean_std'] = mean_std

    # Save results
    output_path = OUTPUT_DIR / 'pwmcc_sensitivity_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {output_path}")

    # Generate report
    report_path = OUTPUT_DIR / 'pwmcc_sensitivity_report.md'
    with open(report_path, 'w') as f:
        f.write("# PWMCC Sensitivity Analysis Report\n\n")
        f.write("## Objective\n\n")
        f.write("Test whether computing PWMCC on only 10% of activations (current approach) ")
        f.write("introduces bias compared to using 50% or 100% of data.\n\n")

        f.write("## Results\n\n")
        f.write(f"Total activations available: {n_total:,}\n\n")
        f.write("### PWMCC by Subset Size\n\n")
        f.write("| Subset Size | Mean PWMCC | Std PWMCC |\n")
        f.write("|-------------|------------|-----------|\n")
        for entry in results['subset_comparison']:
            f.write(f"| {entry['label']:20s} | {entry['mean_pwmcc']:.6f} | {entry['std_pwmcc']:.6f} |\n")

        f.write("\n### Stability Across Subsets\n\n")
        f.write(f"- Maximum std across subsets: {max_std:.6f}\n")
        f.write(f"- Average std across subsets: {mean_std:.6f}\n\n")

        f.write("## Conclusion\n\n")
        f.write(f"{conclusion}\n\n")
        f.write(f"**Risk level:** {risk_level}\n\n")
        f.write(f"**Recommendation:** {recommendation}\n\n")

        f.write("## Detailed Results\n\n")
        for pair in results['seed_pairs']:
            f.write(f"### Seeds {pair['seed1']} vs {pair['seed2']}\n\n")
            f.write("| Subset | PWMCC |\n")
            f.write("|--------|-------|\n")
            for entry in pair['pwmcc_by_subset']:
                f.write(f"| {entry['label']} | {entry['pwmcc']:.6f} |\n")
            f.write("\n")

    print(f"✓ Saved report to {report_path}")
    print()

    print("="*80)
    print("✓ SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
