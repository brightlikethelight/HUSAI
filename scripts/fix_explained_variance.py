#!/usr/bin/env python3
"""Fix Explained Variance Bug and Recompute for All SAEs.

CRITICAL BUG: The explained variance computation in multiple files uses scalar
variance across all dimensions, which is mathematically incorrect. This script:

1. Implements the CORRECT EV formula (per-dimension variance)
2. Loads all 10 existing SAE checkpoints (5 TopK + 5 ReLU)
3. Recomputes EV using correct formula
4. Compares old vs new values
5. Saves corrected metrics and generates report

Correct Formula:
    EV = 1 - mean(MSE_per_dim / Var_per_dim)

Where:
    - MSE_per_dim: Mean squared error per dimension (averaged over samples)
    - Var_per_dim: Variance per dimension (variance of input activations)

WRONG (current implementation):
    data_var = activations.var()  # Scalar across ALL dims
    error_var = (activations - recon).var()  # Scalar
    EV = 1 - (error_var / data_var)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
RESULTS_DIR = BASE_DIR / 'results'
SAES_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_ev_correct(
    activations: torch.Tensor,
    reconstructions: torch.Tensor
) -> float:
    """Compute explained variance using CORRECT formula.

    Args:
        activations: Input activations [N, d_model]
        reconstructions: Reconstructed activations [N, d_model]

    Returns:
        Explained variance (scalar)

    Formula:
        EV = 1 - mean(MSE_per_dim / Var_per_dim)
    """
    # Ensure 2D: [N, d_model]
    if activations.dim() > 2:
        N_total = activations.shape[0] * activations.shape[1]
        d_model = activations.shape[-1]
        activations = activations.reshape(N_total, d_model)
        reconstructions = reconstructions.reshape(N_total, d_model)

    # Per-dimension variance of input
    per_dim_var = activations.var(dim=0, unbiased=True)  # [d_model]

    # Per-dimension MSE
    per_dim_mse = ((activations - reconstructions) ** 2).mean(dim=0)  # [d_model]

    # Overall EV: average across dimensions
    # (Only compute for dims with non-zero variance)
    nonzero_var = per_dim_var > 1e-8
    if nonzero_var.sum() == 0:
        return 0.0

    ev_per_dim = 1 - (per_dim_mse[nonzero_var] / per_dim_var[nonzero_var])
    explained_var = ev_per_dim.mean().item()

    return explained_var


def compute_ev_wrong(
    activations: torch.Tensor,
    reconstructions: torch.Tensor
) -> float:
    """Compute explained variance using WRONG formula (current implementation).

    This is the BUGGY version - included for comparison.
    """
    data_var = activations.var()
    error_var = (activations - reconstructions).var()
    explained_var = 1 - (error_var / data_var)
    return explained_var.item()


class SimpleSAE(nn.Module):
    """Simple SAE implementation matching the trained models."""

    def __init__(self, d_model: int, d_sae: int, architecture: str = 'topk', k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.architecture = architecture
        self.k = k if architecture == 'topk' else None

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)  # No bias in decoder

        # Feature counts (for compatibility with saved checkpoints)
        self.register_buffer('feature_counts', torch.zeros(d_sae))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            reconstructed, latents, pre_activation
        """
        # Encode
        pre_activation = self.encoder(x)

        # Apply activation
        if self.architecture == 'topk':
            # TopK activation
            topk_values, topk_indices = torch.topk(pre_activation, k=self.k, dim=-1)
            latents = torch.zeros_like(pre_activation)
            latents.scatter_(dim=-1, index=topk_indices, src=topk_values)
        else:
            # ReLU activation
            latents = F.relu(pre_activation)

        # Decode
        reconstructed = self.decoder(latents)

        return reconstructed, latents, pre_activation


def load_transformer() -> ModularArithmeticTransformer:
    """Load trained transformer model."""
    print("Loading transformer...")
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"Transformer not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device=DEVICE)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded transformer: {config.n_layers} layers, {config.d_model} dims")
    return model


def extract_activations(
    model: ModularArithmeticTransformer,
    dataloader: DataLoader,
    layer: int = 1,
    position: int = -2
) -> torch.Tensor:
    """Extract activations from specific layer and position."""
    print(f"Extracting activations from layer {layer}, position {position}...")
    activations = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                tokens, _ = batch
            else:
                tokens = batch

            tokens = tokens.to(DEVICE)
            layer_acts = model.get_activations(tokens, layer=layer)
            position_acts = layer_acts[:, position, :]
            activations.append(position_acts.cpu())

    activations = torch.cat(activations, dim=0)
    print(f"✓ Extracted {activations.shape[0]} samples × {activations.shape[1]} dims")
    return activations


def recompute_sae_ev(
    sae_path: Path,
    activations: torch.Tensor,
    architecture: str
) -> Dict:
    """Recompute EV for a single SAE using both old and new formulas.

    Returns:
        Dict with: seed, architecture, ev_old, ev_correct, difference
    """
    # Load SAE checkpoint
    checkpoint = torch.load(sae_path, map_location=DEVICE)

    # Get metadata
    seed = checkpoint.get('seed', int(sae_path.parent.name.split('seed')[-1]))
    d_model = checkpoint.get('d_model', 128)
    d_sae = checkpoint.get('d_sae', 1024)

    # Initialize SAE
    if architecture == 'topk':
        k = checkpoint.get('k', 32)
        sae = SimpleSAE(d_model, d_sae, architecture='topk', k=k).to(DEVICE)
    else:  # relu
        sae = SimpleSAE(d_model, d_sae, architecture='relu').to(DEVICE)

    # Load weights
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    # Reconstruct all activations
    print(f"  Reconstructing {len(activations)} samples...")
    reconstructions = []

    with torch.no_grad():
        for i in range(0, len(activations), 1024):
            batch = activations[i:i+1024].to(DEVICE)
            recon, _, _ = sae(batch)
            reconstructions.append(recon.cpu())

    reconstructions = torch.cat(reconstructions, dim=0)

    # Compute EV using both methods
    ev_old = compute_ev_wrong(activations, reconstructions)
    ev_correct = compute_ev_correct(activations, reconstructions)
    difference = ev_old - ev_correct

    return {
        'seed': seed,
        'architecture': architecture,
        'ev_old_buggy': ev_old,
        'ev_correct': ev_correct,
        'absolute_difference': difference,
        'relative_difference_pct': (difference / ev_old) * 100 if ev_old != 0 else 0,
        'sae_path': str(sae_path)
    }


def main():
    """Main execution: recompute EV for all 10 SAEs."""
    print("="*80)
    print("FIXING EXPLAINED VARIANCE BUG - RECOMPUTING FOR ALL SAEs")
    print("="*80)
    print()

    # Load transformer
    model = load_transformer()
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"✓ Loaded {len(dataset)} training samples")
    print()

    # Extract activations (layer 1, position -2)
    activations = extract_activations(model, dataloader, layer=1, position=-2)
    print()

    # Recompute EV for all SAEs
    print("="*80)
    print("RECOMPUTING EXPLAINED VARIANCE")
    print("="*80)
    print()

    results = {
        'topk': [],
        'relu': []
    }

    for seed in SEEDS:
        # TopK SAE
        topk_path = SAES_DIR / f'topk_seed{seed}' / 'sae_final.pt'
        if topk_path.exists():
            print(f"Processing TopK SAE (seed={seed})...")
            result = recompute_sae_ev(topk_path, activations, architecture='topk')
            results['topk'].append(result)
            print(f"  Old EV (buggy):  {result['ev_old_buggy']:.6f}")
            print(f"  New EV (correct): {result['ev_correct']:.6f}")
            print(f"  Difference:       {result['absolute_difference']:.6f} ({result['relative_difference_pct']:.2f}%)")
            print()

        # ReLU SAE
        relu_path = SAES_DIR / f'relu_seed{seed}' / 'sae_final.pt'
        if relu_path.exists():
            print(f"Processing ReLU SAE (seed={seed})...")
            result = recompute_sae_ev(relu_path, activations, architecture='relu')
            results['relu'].append(result)
            print(f"  Old EV (buggy):  {result['ev_old_buggy']:.6f}")
            print(f"  New EV (correct): {result['ev_correct']:.6f}")
            print(f"  Difference:       {result['absolute_difference']:.6f} ({result['relative_difference_pct']:.2f}%)")
            print()

    # Compute summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()

    # TopK summary
    topk_old = [r['ev_old_buggy'] for r in results['topk']]
    topk_new = [r['ev_correct'] for r in results['topk']]
    topk_diff = [r['absolute_difference'] for r in results['topk']]

    print("TopK SAE (n=5):")
    print(f"  Old EV (buggy):  {np.mean(topk_old):.6f} ± {np.std(topk_old):.6f}")
    print(f"  New EV (correct): {np.mean(topk_new):.6f} ± {np.std(topk_new):.6f}")
    print(f"  Avg difference:   {np.mean(topk_diff):.6f} ± {np.std(topk_diff):.6f}")
    print()

    # ReLU summary
    relu_old = [r['ev_old_buggy'] for r in results['relu']]
    relu_new = [r['ev_correct'] for r in results['relu']]
    relu_diff = [r['absolute_difference'] for r in results['relu']]

    print("ReLU SAE (n=5):")
    print(f"  Old EV (buggy):  {np.mean(relu_old):.6f} ± {np.std(relu_old):.6f}")
    print(f"  New EV (correct): {np.mean(relu_new):.6f} ± {np.std(relu_new):.6f}")
    print(f"  Avg difference:   {np.mean(relu_diff):.6f} ± {np.std(relu_diff):.6f}")
    print()

    # Save results
    output_path = OUTPUT_DIR / 'ev_correction_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {output_path}")
    print()

    # Generate detailed report
    report_path = OUTPUT_DIR / 'ev_correction_report.md'
    with open(report_path, 'w') as f:
        f.write("# Explained Variance Correction Report\n\n")
        f.write("## Bug Description\n\n")
        f.write("The explained variance computation in multiple files used scalar variance ")
        f.write("across all dimensions, which is mathematically incorrect:\n\n")
        f.write("```python\n")
        f.write("# WRONG (buggy implementation):\n")
        f.write("data_var = activations.var()  # Scalar across ALL dimensions\n")
        f.write("error_var = (activations - reconstructed).var()\n")
        f.write("EV = 1 - (error_var / data_var)\n")
        f.write("```\n\n")
        f.write("The CORRECT formula computes per-dimension variance:\n\n")
        f.write("```python\n")
        f.write("# CORRECT:\n")
        f.write("per_dim_var = activations.var(dim=0)  # [d_model]\n")
        f.write("per_dim_mse = ((activations - reconstructed) ** 2).mean(dim=0)  # [d_model]\n")
        f.write("EV = 1 - (per_dim_mse / per_dim_var).mean()\n")
        f.write("```\n\n")

        f.write("## Impact Analysis\n\n")
        f.write(f"### TopK SAE (n={len(results['topk'])})\n\n")
        f.write("| Seed | Old EV (Buggy) | New EV (Correct) | Difference | % Change |\n")
        f.write("|------|----------------|------------------|------------|----------|\n")
        for r in results['topk']:
            f.write(f"| {r['seed']} | {r['ev_old_buggy']:.6f} | {r['ev_correct']:.6f} | "
                   f"{r['absolute_difference']:.6f} | {r['relative_difference_pct']:.2f}% |\n")
        f.write(f"\n**Mean ± Std:**\n")
        f.write(f"- Old EV: {np.mean(topk_old):.6f} ± {np.std(topk_old):.6f}\n")
        f.write(f"- New EV: {np.mean(topk_new):.6f} ± {np.std(topk_new):.6f}\n")
        f.write(f"- Difference: {np.mean(topk_diff):.6f} ± {np.std(topk_diff):.6f}\n\n")

        f.write(f"### ReLU SAE (n={len(results['relu'])})\n\n")
        f.write("| Seed | Old EV (Buggy) | New EV (Correct) | Difference | % Change |\n")
        f.write("|------|----------------|------------------|------------|----------|\n")
        for r in results['relu']:
            f.write(f"| {r['seed']} | {r['ev_old_buggy']:.6f} | {r['ev_correct']:.6f} | "
                   f"{r['absolute_difference']:.6f} | {r['relative_difference_pct']:.2f}% |\n")
        f.write(f"\n**Mean ± Std:**\n")
        f.write(f"- Old EV: {np.mean(relu_old):.6f} ± {np.std(relu_old):.6f}\n")
        f.write(f"- New EV: {np.mean(relu_new):.6f} ± {np.std(relu_new):.6f}\n")
        f.write(f"- Difference: {np.mean(relu_diff):.6f} ± {np.std(relu_diff):.6f}\n\n")

        f.write("## Interpretation\n\n")
        avg_topk_diff = np.mean(topk_diff)
        avg_relu_diff = np.mean(relu_diff)

        if abs(avg_topk_diff) < 0.01 and abs(avg_relu_diff) < 0.01:
            f.write("✅ **Minor impact**: Differences are < 1%, indicating bug had minimal effect.\n")
        elif abs(avg_topk_diff) < 0.05 and abs(avg_relu_diff) < 0.05:
            f.write("⚠️ **Moderate impact**: Differences are 1-5%, should update reported values.\n")
        else:
            f.write("🚨 **Major impact**: Differences are > 5%, significantly affects interpretation!\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. Update all reported EV values in paper with corrected values\n")
        f.write("2. Use corrected EV formula in all future experiments\n")
        f.write("3. Fix bug in source files:\n")
        f.write("   - `scripts/train_simple_sae.py:167-169`\n")
        f.write("   - `scripts/cross_layer_validation.py:253-255`\n")
        f.write("   - `src/training/train_sae.py:357-358`\n")
        f.write("4. Regenerate figures with corrected EV values\n")

    print(f"✓ Saved detailed report to {report_path}")
    print()

    print("="*80)
    print("✓ EV CORRECTION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review {report_path}")
    print(f"2. Update figures with corrected EV values")
    print(f"3. Fix bug in source files")


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    main()
