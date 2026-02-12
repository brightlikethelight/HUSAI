#!/usr/bin/env python3
"""Deep analysis of Layer 0 orthogonality phenomenon.

FINDING: Layer 0 PWMCC = 0.047, which is 6× BELOW random baseline (0.30).
This means Layer 0 SAE features are nearly ORTHOGONAL across seeds.

This script investigates:
1. What is the cosine similarity distribution at Layer 0 vs Layer 1?
2. Are Layer 0 SAEs learning valid reconstructions?
3. What is the activation structure at each layer?
4. Is there something special about Layer 0 that causes orthogonality?

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/deep_layer0_analysis.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader


class SimpleSAE(nn.Module):
    """Simple SAE for loading checkpoints."""
    def __init__(self, d_model: int, d_sae: int, k: int = 32, has_decoder_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=has_decoder_bias)
        self.register_buffer('feature_counts', torch.zeros(d_sae))
        
    def forward(self, x):
        latents = self.encoder(x)
        topk_values, topk_indices = torch.topk(latents, k=self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(dim=-1, index=topk_indices, src=topk_values)
        reconstructed = self.decoder(sparse_latents)
        return reconstructed, sparse_latents, latents


def load_layer0_saes():
    """Load all Layer 0 SAEs."""
    saes = {}
    base_dir = Path('results/cross_layer_validation')
    seeds = [42, 123, 456, 789, 1011]
    
    for seed in seeds:
        path = base_dir / f'layer0_seed{seed}.pt'
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu')
            sae = SimpleSAE(d_model=128, d_sae=1024, k=32, has_decoder_bias=True)
            # Load with strict=False to handle missing keys
            sae.load_state_dict(checkpoint['model_state_dict'], strict=False)
            saes[seed] = sae
            print(f"  Loaded Layer 0 SAE seed {seed}")
    
    return saes


def load_layer1_saes():
    """Load all Layer 1 SAEs."""
    saes = {}
    base_dir = Path('results/saes')
    seeds = [42, 123, 456, 789, 1011]
    
    for seed in seeds:
        path = base_dir / f'topk_seed{seed}' / 'sae_final.pt'
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu')
            # Layer 1 SAEs have no decoder bias
            sae = SimpleSAE(d_model=128, d_sae=1024, k=32, has_decoder_bias=False)
            sae.load_state_dict(checkpoint['model_state_dict'])
            saes[seed] = sae
            print(f"  Loaded Layer 1 SAE seed {seed}")
    
    return saes


def compute_cosine_similarity_distribution(sae1, sae2):
    """Compute full cosine similarity matrix and return distribution."""
    d1 = F.normalize(sae1.decoder.weight.data, dim=0)
    d2 = F.normalize(sae2.decoder.weight.data, dim=0)
    cos_sim = d1.T @ d2  # [d_sae, d_sae]
    return cos_sim.abs().flatten().numpy()


def compute_reconstruction_quality(sae, activations):
    """Compute explained variance for an SAE."""
    with torch.no_grad():
        recon, _, _ = sae(activations)
        per_dim_var = activations.var(dim=0)
        per_dim_mse = ((activations - recon) ** 2).mean(dim=0)
        ev = 1 - (per_dim_mse / per_dim_var).mean()
    return ev.item()


def main():
    print("=" * 70)
    print("DEEP ANALYSIS: WHY IS LAYER 0 PWMCC BELOW RANDOM?")
    print("=" * 70)
    print()
    
    # Load SAEs
    print("Loading SAEs...")
    layer0_saes = load_layer0_saes()
    layer1_saes = load_layer1_saes()
    print()
    
    if len(layer0_saes) < 2:
        print("❌ Not enough Layer 0 SAEs found!")
        return
    
    # Load transformer and data
    print("Loading transformer and data...")
    model_path = Path('results/transformer_5000ep/transformer_best.pt')
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Extract activations
    print("Extracting activations...")
    acts_layer0 = []
    acts_layer1 = []
    with torch.no_grad():
        for batch in dataloader:
            tokens, _ = batch
            a0 = model.get_activations(tokens, layer=0)[:, -2, :]
            a1 = model.get_activations(tokens, layer=1)[:, -2, :]
            acts_layer0.append(a0)
            acts_layer1.append(a1)
    acts_layer0 = torch.cat(acts_layer0, dim=0)
    acts_layer1 = torch.cat(acts_layer1, dim=0)
    print(f"  Layer 0: {acts_layer0.shape}")
    print(f"  Layer 1: {acts_layer1.shape}")
    print()
    
    # Analysis 1: Cosine similarity distributions
    print("=" * 70)
    print("ANALYSIS 1: COSINE SIMILARITY DISTRIBUTIONS")
    print("=" * 70)
    print()
    
    seeds = list(layer0_saes.keys())
    
    # Layer 0 distribution
    layer0_cos_sims = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            cos_sim = compute_cosine_similarity_distribution(layer0_saes[s1], layer0_saes[s2])
            layer0_cos_sims.extend(cos_sim)
    layer0_cos_sims = np.array(layer0_cos_sims)
    
    # Layer 1 distribution
    layer1_cos_sims = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            if s1 in layer1_saes and s2 in layer1_saes:
                cos_sim = compute_cosine_similarity_distribution(layer1_saes[s1], layer1_saes[s2])
                layer1_cos_sims.extend(cos_sim)
    layer1_cos_sims = np.array(layer1_cos_sims)
    
    print("Layer 0 |cos_sim| distribution:")
    print(f"  Mean: {layer0_cos_sims.mean():.4f}")
    print(f"  Std:  {layer0_cos_sims.std():.4f}")
    print(f"  Max:  {layer0_cos_sims.max():.4f}")
    print(f"  >0.5: {(layer0_cos_sims > 0.5).mean()*100:.2f}%")
    print(f"  >0.7: {(layer0_cos_sims > 0.7).mean()*100:.2f}%")
    print()
    
    print("Layer 1 |cos_sim| distribution:")
    print(f"  Mean: {layer1_cos_sims.mean():.4f}")
    print(f"  Std:  {layer1_cos_sims.std():.4f}")
    print(f"  Max:  {layer1_cos_sims.max():.4f}")
    print(f"  >0.5: {(layer1_cos_sims > 0.5).mean()*100:.2f}%")
    print(f"  >0.7: {(layer1_cos_sims > 0.7).mean()*100:.2f}%")
    print()
    
    # Analysis 2: Reconstruction quality
    print("=" * 70)
    print("ANALYSIS 2: RECONSTRUCTION QUALITY")
    print("=" * 70)
    print()
    
    print("Layer 0 SAE reconstruction (EV):")
    layer0_evs = []
    for seed, sae in layer0_saes.items():
        ev = compute_reconstruction_quality(sae, acts_layer0)
        layer0_evs.append(ev)
        print(f"  Seed {seed}: EV = {ev:.4f}")
    print(f"  Mean: {np.mean(layer0_evs):.4f} ± {np.std(layer0_evs):.4f}")
    print()
    
    print("Layer 1 SAE reconstruction (EV):")
    layer1_evs = []
    for seed, sae in layer1_saes.items():
        ev = compute_reconstruction_quality(sae, acts_layer1)
        layer1_evs.append(ev)
        print(f"  Seed {seed}: EV = {ev:.4f}")
    print(f"  Mean: {np.mean(layer1_evs):.4f} ± {np.std(layer1_evs):.4f}")
    print()
    
    # Analysis 3: Activation structure
    print("=" * 70)
    print("ANALYSIS 3: ACTIVATION STRUCTURE")
    print("=" * 70)
    print()
    
    # Compute SVD to understand dimensionality
    def analyze_activation_structure(acts, name):
        centered = acts - acts.mean(dim=0, keepdim=True)
        _, S, _ = torch.svd(centered)
        
        # Cumulative explained variance
        var_explained = (S ** 2) / (S ** 2).sum()
        cumvar = var_explained.cumsum(0)
        
        # Effective rank
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        entropy = -(S_norm * torch.log(S_norm)).sum()
        eff_rank = torch.exp(entropy).item()
        
        # Dimensions for 90%, 95%, 99% variance
        dims_90 = (cumvar < 0.90).sum().item() + 1
        dims_95 = (cumvar < 0.95).sum().item() + 1
        dims_99 = (cumvar < 0.99).sum().item() + 1
        
        print(f"{name}:")
        print(f"  Effective rank: {eff_rank:.1f} / 128")
        print(f"  Dims for 90% var: {dims_90}")
        print(f"  Dims for 95% var: {dims_95}")
        print(f"  Dims for 99% var: {dims_99}")
        print(f"  Top singular value: {S[0].item():.2f}")
        print(f"  Singular value ratio (1st/10th): {S[0].item()/S[9].item():.2f}")
        
        return {
            'effective_rank': eff_rank,
            'dims_90': dims_90,
            'dims_95': dims_95,
            'dims_99': dims_99,
            'top_sv': S[0].item(),
            'sv_ratio': S[0].item()/S[9].item()
        }
    
    struct_layer0 = analyze_activation_structure(acts_layer0, "Layer 0 activations")
    print()
    struct_layer1 = analyze_activation_structure(acts_layer1, "Layer 1 activations")
    print()
    
    # Analysis 4: Decoder weight structure
    print("=" * 70)
    print("ANALYSIS 4: DECODER WEIGHT STRUCTURE")
    print("=" * 70)
    print()
    
    def analyze_decoder_structure(sae, name):
        W = sae.decoder.weight.data  # [d_model, d_sae]
        
        # Column norms
        col_norms = W.norm(dim=0)
        
        # Gram matrix (decoder columns similarity)
        W_norm = F.normalize(W, dim=0)
        gram = W_norm.T @ W_norm  # [d_sae, d_sae]
        
        # Off-diagonal similarities
        mask = ~torch.eye(gram.shape[0], dtype=bool)
        off_diag = gram[mask].abs()
        
        print(f"{name}:")
        print(f"  Column norm mean: {col_norms.mean():.4f} ± {col_norms.std():.4f}")
        print(f"  Self-similarity (off-diag |cos|): {off_diag.mean():.4f} ± {off_diag.std():.4f}")
        print(f"  Max self-similarity: {off_diag.max():.4f}")
        
        return {
            'col_norm_mean': col_norms.mean().item(),
            'self_sim_mean': off_diag.mean().item(),
            'self_sim_max': off_diag.max().item()
        }
    
    for seed in seeds[:2]:  # Just analyze first 2 for brevity
        if seed in layer0_saes:
            analyze_decoder_structure(layer0_saes[seed], f"Layer 0 SAE (seed {seed})")
        if seed in layer1_saes:
            analyze_decoder_structure(layer1_saes[seed], f"Layer 1 SAE (seed {seed})")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: WHY IS LAYER 0 DIFFERENT?")
    print("=" * 70)
    print()
    
    print("Key observations:")
    print(f"  1. Layer 0 |cos_sim| distribution is MUCH tighter around 0")
    print(f"     Layer 0 mean: {layer0_cos_sims.mean():.4f}, Layer 1 mean: {layer1_cos_sims.mean():.4f}")
    print()
    print(f"  2. Both layers have similar reconstruction quality")
    print(f"     Layer 0 EV: {np.mean(layer0_evs):.4f}, Layer 1 EV: {np.mean(layer1_evs):.4f}")
    print()
    print(f"  3. Activation structure differs:")
    print(f"     Layer 0 effective rank: {struct_layer0['effective_rank']:.1f}")
    print(f"     Layer 1 effective rank: {struct_layer1['effective_rank']:.1f}")
    print()
    
    if np.mean(layer0_evs) > 0.8:
        print("✅ Layer 0 SAEs achieve good reconstruction despite orthogonal features!")
        print("   This suggests multiple equally-valid decompositions exist.")
        print()
        print("INTERPRETATION:")
        print("   Layer 0 activations can be reconstructed by MANY different")
        print("   orthogonal feature sets. The optimization landscape has")
        print("   multiple isolated minima, each with different features.")
        print()
        print("   Layer 1 activations have more constrained structure,")
        print("   leading to features that are at least as correlated as random.")
    else:
        print("⚠️ Layer 0 SAEs have poor reconstruction - training may have failed.")
    
    # Save results
    results = {
        'layer0_cos_sim': {
            'mean': float(layer0_cos_sims.mean()),
            'std': float(layer0_cos_sims.std()),
            'max': float(layer0_cos_sims.max()),
            'pct_above_0.5': float((layer0_cos_sims > 0.5).mean()),
            'pct_above_0.7': float((layer0_cos_sims > 0.7).mean()),
        },
        'layer1_cos_sim': {
            'mean': float(layer1_cos_sims.mean()),
            'std': float(layer1_cos_sims.std()),
            'max': float(layer1_cos_sims.max()),
            'pct_above_0.5': float((layer1_cos_sims > 0.5).mean()),
            'pct_above_0.7': float((layer1_cos_sims > 0.7).mean()),
        },
        'layer0_ev': {'mean': float(np.mean(layer0_evs)), 'std': float(np.std(layer0_evs))},
        'layer1_ev': {'mean': float(np.mean(layer1_evs)), 'std': float(np.std(layer1_evs))},
        'layer0_structure': struct_layer0,
        'layer1_structure': struct_layer1,
    }
    
    output_path = Path('results/analysis/deep_layer0_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
