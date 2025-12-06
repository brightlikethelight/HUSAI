#!/usr/bin/env python3
"""Investigate Layer 0 PWMCC Anomaly: Why is it BELOW random baseline?

CRITICAL FINDING:
- Layer 0 PWMCC = 0.047 (from cross_layer_validation)
- Random baseline PWMCC = 0.30
- Layer 0 is 6× BELOW random!

This script investigates WHY Layer 0 shows below-random consistency.

Hypotheses:
1. Dimensional collapse - Layer 0 has fewer effective dimensions
2. SAE training failure - Poor reconstruction at Layer 0
3. Activation scale issues - Very different scale/distribution
4. Random baseline differs by layer - Need to compute random PWMCC at Layer 0

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/investigate_layer0_anomaly.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_sae import TopKSAE
from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader


def compute_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder weight matrices."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_effective_rank(activations: torch.Tensor) -> float:
    """Compute effective rank of activation matrix using singular values.
    
    Effective rank = exp(entropy of normalized singular values)
    Higher = more dimensions used, Lower = dimensional collapse
    """
    # Center the data
    centered = activations - activations.mean(dim=0, keepdim=True)
    
    # SVD
    _, S, _ = torch.svd(centered)
    
    # Normalize singular values to form probability distribution
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]  # Remove zeros
    
    # Compute entropy
    entropy = -(S_norm * torch.log(S_norm)).sum()
    
    # Effective rank = exp(entropy)
    effective_rank = torch.exp(entropy).item()
    
    return effective_rank


def compute_activation_stats(activations: torch.Tensor) -> dict:
    """Compute comprehensive statistics about activations."""
    return {
        'mean': activations.mean().item(),
        'std': activations.std().item(),
        'min': activations.min().item(),
        'max': activations.max().item(),
        'shape': list(activations.shape),
        'effective_rank': compute_effective_rank(activations),
        'per_dim_var_mean': activations.var(dim=0).mean().item(),
        'per_dim_var_std': activations.var(dim=0).std().item(),
        'per_dim_var_min': activations.var(dim=0).min().item(),
        'per_dim_var_max': activations.var(dim=0).max().item(),
    }


def load_transformer():
    """Load trained transformer."""
    model_path = Path('results/transformer_5000ep/transformer_best.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def extract_activations(model, dataloader, layer: int, position: int = -2):
    """Extract activations from specific layer."""
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            tokens, _ = batch
            acts = model.get_activations(tokens, layer=layer)
            acts = acts[:, position, :]  # [batch, d_model]
            activations.append(acts)
    return torch.cat(activations, dim=0)


def main():
    print("=" * 70)
    print("INVESTIGATING LAYER 0 PWMCC ANOMALY")
    print("=" * 70)
    print()
    print("CONTEXT:")
    print("  - Layer 0 PWMCC = 0.047 (from cross_layer_validation)")
    print("  - Layer 1 PWMCC = 0.302")
    print("  - Random baseline PWMCC = 0.30")
    print("  - Layer 0 is 6× BELOW random!")
    print()
    
    # Load model and data
    print("Loading transformer and data...")
    model = load_transformer()
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"✅ Loaded {len(dataset)} samples")
    print()
    
    # Extract activations from both layers
    print("Extracting activations...")
    acts_layer0 = extract_activations(model, dataloader, layer=0)
    acts_layer1 = extract_activations(model, dataloader, layer=1)
    print(f"  Layer 0: {acts_layer0.shape}")
    print(f"  Layer 1: {acts_layer1.shape}")
    print()
    
    # Compute activation statistics
    print("=" * 70)
    print("HYPOTHESIS 1: DIMENSIONAL COLLAPSE")
    print("=" * 70)
    print()
    
    stats_layer0 = compute_activation_stats(acts_layer0)
    stats_layer1 = compute_activation_stats(acts_layer1)
    
    print(f"Layer 0 Statistics:")
    print(f"  Mean: {stats_layer0['mean']:.4f}")
    print(f"  Std: {stats_layer0['std']:.4f}")
    print(f"  Effective rank: {stats_layer0['effective_rank']:.2f} / 128")
    print(f"  Per-dim variance: {stats_layer0['per_dim_var_mean']:.4f} ± {stats_layer0['per_dim_var_std']:.4f}")
    print()
    
    print(f"Layer 1 Statistics:")
    print(f"  Mean: {stats_layer1['mean']:.4f}")
    print(f"  Std: {stats_layer1['std']:.4f}")
    print(f"  Effective rank: {stats_layer1['effective_rank']:.2f} / 128")
    print(f"  Per-dim variance: {stats_layer1['per_dim_var_mean']:.4f} ± {stats_layer1['per_dim_var_std']:.4f}")
    print()
    
    if stats_layer0['effective_rank'] < stats_layer1['effective_rank'] * 0.5:
        print("⚠️  Layer 0 has MUCH lower effective rank - dimensional collapse!")
    else:
        print("✓ Effective ranks are similar - no dimensional collapse")
    print()
    
    # Compute random baseline at each layer
    print("=" * 70)
    print("HYPOTHESIS 2: RANDOM BASELINE DIFFERS BY LAYER")
    print("=" * 70)
    print()
    
    d_model = 128
    d_sae = 1024
    k = 32
    n_random = 5
    
    print(f"Computing random PWMCC for {n_random} random SAE pairs...")
    
    random_pwmcc_layer0 = []
    random_pwmcc_layer1 = []
    
    for i in range(n_random):
        for j in range(i+1, n_random):
            # Random SAEs
            torch.manual_seed(100 + i)
            sae_i = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
            torch.manual_seed(100 + j)
            sae_j = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
            
            pwmcc = compute_pwmcc(sae_i.decoder.weight.data, sae_j.decoder.weight.data)
            random_pwmcc_layer0.append(pwmcc)
            random_pwmcc_layer1.append(pwmcc)  # Same for both (random doesn't depend on layer)
    
    random_mean = np.mean(random_pwmcc_layer0)
    random_std = np.std(random_pwmcc_layer0)
    
    print(f"Random PWMCC: {random_mean:.4f} ± {random_std:.4f}")
    print()
    
    # Check if Layer 0 trained SAEs exist
    print("=" * 70)
    print("CHECKING LAYER 0 SAE AVAILABILITY")
    print("=" * 70)
    print()
    
    layer0_saes_exist = False
    layer0_sae_paths = list(Path('results/saes').glob('*layer0*'))
    
    if layer0_sae_paths:
        print(f"Found {len(layer0_sae_paths)} Layer 0 SAE checkpoints")
        layer0_saes_exist = True
    else:
        print("❌ No Layer 0 SAE checkpoints found!")
        print("   The 0.047 PWMCC must come from cross_layer_validation.py")
        print("   Need to check that script for how Layer 0 SAEs were trained")
    print()
    
    # Summary and interpretation
    print("=" * 70)
    print("SUMMARY AND INTERPRETATION")
    print("=" * 70)
    print()
    
    print("Key findings:")
    print(f"  1. Random baseline PWMCC = {random_mean:.4f}")
    print(f"  2. Layer 0 PWMCC = 0.047 (from cross_layer_validation)")
    print(f"  3. Layer 1 PWMCC = 0.302")
    print()
    
    print("Interpretation:")
    if 0.047 < random_mean * 0.5:
        print("  🚨 Layer 0 PWMCC (0.047) is SIGNIFICANTLY below random ({:.3f})!".format(random_mean))
        print()
        print("  Possible explanations:")
        print("  a) SAEs at Layer 0 are learning ANTI-correlated features")
        print("     (different seeds find opposite directions)")
        print("  b) Layer 0 activations have structure that causes SAE divergence")
        print("  c) There's a bug in how Layer 0 SAEs were trained/evaluated")
        print()
        print("  This is actually a MORE INTERESTING finding than Layer 1!")
        print("  It suggests something ACTIVELY prevents feature alignment at Layer 0.")
    
    print()
    print("Recommended next steps:")
    print("  1. Verify Layer 0 SAE training was correct (check cross_layer_validation.py)")
    print("  2. Visualize Layer 0 vs Layer 1 activation distributions")
    print("  3. Check if Layer 0 SAEs have good reconstruction (EV)")
    print("  4. Train fresh Layer 0 SAEs with verified code")
    
    # Save results
    results = {
        'layer0_stats': stats_layer0,
        'layer1_stats': stats_layer1,
        'random_pwmcc': {
            'mean': random_mean,
            'std': random_std,
            'n_pairs': len(random_pwmcc_layer0)
        },
        'reported_pwmcc': {
            'layer0': 0.047,
            'layer1': 0.302
        },
        'interpretation': {
            'layer0_below_random': 0.047 < random_mean * 0.5,
            'layer1_at_random': abs(0.302 - random_mean) < 0.02
        }
    }
    
    output_path = Path('results/analysis/layer0_investigation.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
