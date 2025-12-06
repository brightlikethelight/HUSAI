#!/usr/bin/env python3
"""Proper Causal Ablation Experiment.

This script implements ACTUAL causal ablation by:
1. Hooking into the model's forward pass
2. Replacing activations with SAE-modified activations
3. Measuring downstream task performance change

This is the CORRECT way to measure causal effects, unlike the proxy metric
used in intervention_validation.py.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/proper_causal_ablation.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.models.simple_sae import TopKSAE
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
RESULTS_DIR = BASE_DIR / 'results'
SAE_DIR = RESULTS_DIR / 'saes'

SEEDS = [42, 123, 456, 789, 1011]


def load_transformer():
    """Load trained transformer."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_sae(seed: int) -> TopKSAE:
    """Load trained TopK SAE."""
    for name in ['sae_final.pt', 'sae.pt']:
        sae_path = SAE_DIR / f'topk_seed{seed}' / name
        if sae_path.exists():
            checkpoint = torch.load(sae_path, map_location='cpu')
            sae = TopKSAE(d_model=128, d_sae=1024, k=32)
            sae.load_state_dict(checkpoint['model_state_dict'])
            sae.eval()
            return sae
    raise FileNotFoundError(f"No SAE found for seed {seed}")


def compute_feature_stability(saes: List[TopKSAE]) -> np.ndarray:
    """Compute per-feature stability scores across SAEs."""
    n_saes = len(saes)
    d_sae = saes[0].decoder.weight.shape[1]
    
    stability_scores = np.zeros((n_saes, d_sae))
    
    for i in range(n_saes):
        features_i = saes[i].decoder.weight.data.cpu()
        features_i_norm = F.normalize(features_i, dim=0)
        
        max_correlations = []
        for j in range(n_saes):
            if i == j:
                continue
            features_j = saes[j].decoder.weight.data.cpu()
            features_j_norm = F.normalize(features_j, dim=0)
            
            cos_sim = features_i_norm.T @ features_j_norm
            max_corr = cos_sim.abs().max(dim=1)[0].numpy()
            max_correlations.append(max_corr)
        
        stability_scores[i] = np.mean(max_correlations, axis=0)
    
    return stability_scores


def proper_ablation_with_hook(
    model: ModularArithmeticTransformer,
    sae: TopKSAE,
    feature_idx: int,
    test_loader: DataLoader,
    layer: int = 1
) -> Tuple[float, float]:
    """Perform proper causal ablation using TransformerLens hooks.
    
    Returns:
        baseline_acc: Accuracy without ablation
        ablated_acc: Accuracy with feature ablated
    """
    model.eval()
    sae.eval()
    
    def ablation_hook(activations, hook):
        """Hook to modify activations by ablating SAE feature."""
        # activations shape: [batch, seq, d_model]
        batch_size, seq_len, d_model = activations.shape
        
        # Get answer position activations
        acts = activations[:, -2, :].clone()  # [batch, d_model]
        
        # Encode through SAE
        latents = sae.encode(acts)  # [batch, d_sae]
        
        # Apply TopK
        topk_values, topk_indices = torch.topk(latents, k=sae.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(dim=-1, index=topk_indices, src=topk_values)
        
        # Ablate specific feature
        sparse_latents[:, feature_idx] = 0.0
        
        # Decode back
        reconstructed = sae.decode(sparse_latents)  # [batch, d_model]
        
        # Replace answer position with modified activations
        activations[:, -2, :] = reconstructed
        
        return activations
    
    # Baseline accuracy (no hook)
    correct_baseline = 0
    total = 0
    
    with torch.no_grad():
        for batch, labels in test_loader:
            logits = model(batch)
            preds = logits[:, -2, :].argmax(dim=-1)
            correct_baseline += (preds == labels).sum().item()
            total += len(labels)
    
    baseline_acc = correct_baseline / total
    
    # Ablated accuracy using TransformerLens run_with_hooks
    correct_ablated = 0
    total = 0
    
    # Hook point name for residual stream after layer
    hook_point = f"blocks.{layer}.hook_resid_post"
    
    with torch.no_grad():
        for batch, labels in test_loader:
            # Run with hook
            logits = model.model.run_with_hooks(
                batch,
                fwd_hooks=[(hook_point, ablation_hook)]
            )
            preds = logits[:, -2, :].argmax(dim=-1)
            correct_ablated += (preds == labels).sum().item()
            total += len(labels)
    
    ablated_acc = correct_ablated / total
    
    return baseline_acc, ablated_acc


def run_proper_ablation_experiment():
    """Run proper causal ablation experiment."""
    print("=" * 70)
    print("PROPER CAUSAL ABLATION EXPERIMENT")
    print("=" * 70)
    print()
    print("This uses forward hooks to ACTUALLY modify activations,")
    print("unlike the proxy metric in intervention_validation.py")
    print()
    
    # Load model and SAEs
    print("Loading transformer...")
    model = load_transformer()
    
    print("Loading SAEs...")
    saes = []
    for seed in SEEDS:
        try:
            sae = load_sae(seed)
            saes.append(sae)
            print(f"  ✓ Loaded seed {seed}")
        except Exception as e:
            print(f"  ✗ Failed to load seed {seed}: {e}")
    
    if len(saes) < 2:
        print("Not enough SAEs loaded!")
        return
    
    # Load test data
    print("\nLoading test data...")
    dataset = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=999, format="sequence")
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"  Test set: {len(dataset)} samples")
    
    # Compute feature stability
    print("\nComputing feature stability scores...")
    stability_scores = compute_feature_stability(saes)
    
    # Select stable and unstable features
    n_features = 10
    flat_scores = stability_scores.flatten()
    sorted_idx = np.argsort(flat_scores)
    
    # Top 10% most stable
    stable_idx = sorted_idx[int(len(flat_scores) * 0.9):]
    stable_selected = np.random.choice(stable_idx, size=n_features, replace=False)
    
    # Bottom 10% least stable
    unstable_idx = sorted_idx[:int(len(flat_scores) * 0.1)]
    unstable_selected = np.random.choice(unstable_idx, size=n_features, replace=False)
    
    d_sae = stability_scores.shape[1]
    stable_features = [(int(idx // d_sae), int(idx % d_sae)) for idx in stable_selected]
    unstable_features = [(int(idx // d_sae), int(idx % d_sae)) for idx in unstable_selected]
    
    print(f"\nSelected {n_features} stable features:")
    stable_stab = [stability_scores[s, f] for s, f in stable_features]
    print(f"  Stability range: [{min(stable_stab):.3f}, {max(stable_stab):.3f}]")
    
    print(f"\nSelected {n_features} unstable features:")
    unstable_stab = [stability_scores[s, f] for s, f in unstable_features]
    print(f"  Stability range: [{min(unstable_stab):.3f}, {max(unstable_stab):.3f}]")
    
    # Run ablation experiments
    print("\n" + "=" * 70)
    print("RUNNING PROPER CAUSAL ABLATIONS")
    print("=" * 70)
    
    stable_effects = []
    print("\nAblating stable features...")
    for sae_idx, feat_idx in tqdm(stable_features, desc="Stable"):
        try:
            baseline, ablated = proper_ablation_with_hook(
                model, saes[sae_idx], feat_idx, test_loader
            )
            effect = baseline - ablated
            stable_effects.append(effect)
        except Exception as e:
            print(f"  Error: {e}")
            stable_effects.append(0.0)
    
    unstable_effects = []
    print("\nAblating unstable features...")
    for sae_idx, feat_idx in tqdm(unstable_features, desc="Unstable"):
        try:
            baseline, ablated = proper_ablation_with_hook(
                model, saes[sae_idx], feat_idx, test_loader
            )
            effect = baseline - ablated
            unstable_effects.append(effect)
        except Exception as e:
            print(f"  Error: {e}")
            unstable_effects.append(0.0)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS: PROPER CAUSAL ABLATION")
    print("=" * 70)
    
    stable_effects = np.array(stable_effects)
    unstable_effects = np.array(unstable_effects)
    
    print(f"\nStable features (n={len(stable_effects)}):")
    print(f"  Mean effect: {stable_effects.mean():.4f} ± {stable_effects.std():.4f}")
    print(f"  Median: {np.median(stable_effects):.4f}")
    print(f"  Range: [{stable_effects.min():.4f}, {stable_effects.max():.4f}]")
    
    print(f"\nUnstable features (n={len(unstable_effects)}):")
    print(f"  Mean effect: {unstable_effects.mean():.4f} ± {unstable_effects.std():.4f}")
    print(f"  Median: {np.median(unstable_effects):.4f}")
    print(f"  Range: [{unstable_effects.min():.4f}, {unstable_effects.max():.4f}]")
    
    # Statistical tests
    from scipy import stats
    
    # Test 1: Are unstable features causal?
    t_stat, p_val = stats.ttest_1samp(unstable_effects, 0)
    print(f"\nTest 1: Do unstable features have causal effects?")
    print(f"  One-sample t-test vs 0: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05 and unstable_effects.mean() > 0:
        print(f"  ✅ YES: Unstable features have significant causal effects!")
    else:
        print(f"  ❌ NO: Unstable features have no significant causal effects")
    
    # Test 2: Stable vs Unstable
    t_stat, p_val = stats.ttest_ind(stable_effects, unstable_effects)
    print(f"\nTest 2: Do stable features have LARGER effects?")
    print(f"  Two-sample t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Save results
    results = {
        'stable_effects': stable_effects.tolist(),
        'unstable_effects': unstable_effects.tolist(),
        'stable_features': stable_features,
        'unstable_features': unstable_features,
        'method': 'proper_causal_ablation_with_hooks'
    }
    
    output_path = RESULTS_DIR / 'proper_causal_ablation_results.pt'
    torch.save(results, output_path)
    print(f"\n✓ Results saved to {output_path}")
    
    # Comparison with proxy metric
    print("\n" + "=" * 70)
    print("COMPARISON WITH PROXY METRIC")
    print("=" * 70)
    print()
    print("Previous (proxy metric - activation magnitude × variance):")
    print("  Stable mean effect: 0.073")
    print("  Unstable mean effect: 0.047")
    print()
    print("Current (proper causal ablation):")
    print(f"  Stable mean effect: {stable_effects.mean():.4f}")
    print(f"  Unstable mean effect: {unstable_effects.mean():.4f}")


if __name__ == '__main__':
    run_proper_ablation_experiment()
