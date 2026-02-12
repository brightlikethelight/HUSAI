#!/usr/bin/env python3
"""Measure Feature Sensitivity for SAE Features.

Based on 2025 SAE literature showing many "interpretable" features have poor sensitivity.
This script tests whether features activate consistently on semantically similar inputs.

For modular arithmetic, we test:
1. Do features that activate on (a, b) also activate on (a, b') where b' ≈ b?
2. Do features show consistent activation patterns across similar inputs?

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/measure_feature_sensitivity.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer import ModularArithmeticTransformer
from src.models.simple_sae import TopKSAE, ReLUSAE
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
SAE_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SEEDS = [42, 123, 456, 789, 1011]
MODULUS = 113


def load_transformer():
    """Load trained transformer."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_sae(arch: str, seed: int) -> torch.nn.Module:
    """Load trained SAE."""
    if arch == 'topk':
        sae_path = SAE_DIR / f'topk_seed{seed}' / 'sae_final.pt'
        if not sae_path.exists():
            sae_path = SAE_DIR / f'topk_seed{seed}' / 'sae.pt'
        checkpoint = torch.load(sae_path, map_location='cpu')
        sae = TopKSAE(d_model=128, d_sae=1024, k=32)
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        sae_path = SAE_DIR / f'relu_seed{seed}' / 'sae_final.pt'
        checkpoint = torch.load(sae_path, map_location='cpu')
        sae = ReLUSAE(d_model=128, d_sae=1024, l1_coef=1e-3)
        sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()
    return sae


def extract_activations(model, inputs: torch.Tensor, layer: int = 1) -> torch.Tensor:
    """Extract activations from transformer."""
    with torch.no_grad():
        acts = model.get_activations(inputs, layer=layer)
        return acts[:, -2, :]  # Answer position


def get_sae_features(sae, activations: torch.Tensor) -> torch.Tensor:
    """Get SAE feature activations."""
    with torch.no_grad():
        if isinstance(sae, TopKSAE):
            pre_acts = sae.encode(activations)
            topk_values, topk_indices = torch.topk(pre_acts, k=sae.k, dim=-1)
            features = torch.zeros_like(pre_acts)
            features.scatter_(dim=-1, index=topk_indices, src=topk_values)
        else:
            features = F.relu(sae.encoder(activations))
    return features


def create_similar_inputs(a: int, b: int, modulus: int, n_similar: int = 5) -> List[Tuple[int, int]]:
    """Create semantically similar inputs for modular arithmetic.
    
    For (a, b), similar inputs are:
    - (a, b±1), (a, b±2), ... (nearby b values)
    - (a±1, b), (a±2, b), ... (nearby a values)
    """
    similar = []
    
    # Vary b
    for delta in range(-n_similar, n_similar + 1):
        if delta != 0:
            b_new = (b + delta) % modulus
            similar.append((a, b_new))
    
    # Vary a
    for delta in range(-n_similar, n_similar + 1):
        if delta != 0:
            a_new = (a + delta) % modulus
            similar.append((a_new, b))
    
    return similar


def compute_feature_sensitivity(
    model,
    sae,
    n_test_inputs: int = 100,
    n_similar: int = 3,
    top_k_features: int = 10
) -> Dict:
    """Compute feature sensitivity scores.
    
    For each test input:
    1. Find top-k activating features
    2. Generate similar inputs
    3. Check if same features activate on similar inputs
    4. Compute sensitivity = fraction of similar inputs where feature activates
    """
    # Generate test inputs
    np.random.seed(42)
    test_inputs = []
    for _ in range(n_test_inputs):
        a = np.random.randint(0, MODULUS)
        b = np.random.randint(0, MODULUS)
        test_inputs.append((a, b))
    
    # Create token sequences
    def create_tokens(a, b):
        # Format: [BOS, a, +, b, =, answer, EOS]
        # BOS=113, +=114, ==115, EOS=116
        answer = (a + b) % MODULUS
        return torch.tensor([[113, a, 114, b, 115, answer, 116]])
    
    feature_sensitivities = defaultdict(list)
    feature_activations = defaultdict(list)
    
    for a, b in test_inputs:
        # Get features for original input
        tokens = create_tokens(a, b)
        acts = extract_activations(model, tokens)
        features = get_sae_features(sae, acts)
        
        # Find top-k activating features
        top_features = features[0].topk(top_k_features)
        top_indices = top_features.indices.tolist()
        top_values = top_features.values.tolist()
        
        # Generate similar inputs
        similar_pairs = create_similar_inputs(a, b, MODULUS, n_similar)
        
        # Check activation on similar inputs
        for feat_idx, feat_val in zip(top_indices, top_values):
            if feat_val < 0.01:  # Skip inactive features
                continue
                
            n_activated = 0
            for a_sim, b_sim in similar_pairs:
                tokens_sim = create_tokens(a_sim, b_sim)
                acts_sim = extract_activations(model, tokens_sim)
                features_sim = get_sae_features(sae, acts_sim)
                
                # Check if feature activates (> 10% of original activation)
                if features_sim[0, feat_idx] > feat_val * 0.1:
                    n_activated += 1
            
            sensitivity = n_activated / len(similar_pairs)
            feature_sensitivities[feat_idx].append(sensitivity)
            feature_activations[feat_idx].append(feat_val)
    
    # Aggregate results
    results = {
        'per_feature': {},
        'overall': {}
    }
    
    all_sensitivities = []
    for feat_idx, sens_list in feature_sensitivities.items():
        mean_sens = np.mean(sens_list)
        std_sens = np.std(sens_list)
        mean_act = np.mean(feature_activations[feat_idx])
        
        results['per_feature'][int(feat_idx)] = {
            'mean_sensitivity': float(mean_sens),
            'std_sensitivity': float(std_sens),
            'mean_activation': float(mean_act),
            'n_samples': len(sens_list)
        }
        all_sensitivities.extend(sens_list)
    
    results['overall'] = {
        'mean_sensitivity': float(np.mean(all_sensitivities)),
        'std_sensitivity': float(np.std(all_sensitivities)),
        'n_features_tested': len(feature_sensitivities),
        'n_total_samples': len(all_sensitivities)
    }
    
    return results


def main():
    print("=" * 70)
    print("FEATURE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("Testing whether SAE features activate consistently on similar inputs.")
    print("High sensitivity = feature responds to semantic similarity")
    print("Low sensitivity = feature is input-specific (potentially spurious)")
    print()
    
    # Load transformer
    print("Loading transformer...")
    model = load_transformer()
    print("✓ Loaded transformer")
    print()
    
    # Analyze each architecture and seed
    all_results = {}
    
    for arch in ['topk', 'relu']:
        print(f"\n{'='*70}")
        print(f"ARCHITECTURE: {arch.upper()}")
        print(f"{'='*70}")
        
        arch_results = {}
        arch_sensitivities = []
        
        for seed in SEEDS:
            print(f"\nAnalyzing {arch} seed {seed}...")
            
            try:
                sae = load_sae(arch, seed)
                results = compute_feature_sensitivity(
                    model, sae,
                    n_test_inputs=50,  # Reduced for speed
                    n_similar=3,
                    top_k_features=10
                )
                
                arch_results[seed] = results
                arch_sensitivities.append(results['overall']['mean_sensitivity'])
                
                print(f"  Mean sensitivity: {results['overall']['mean_sensitivity']:.4f}")
                print(f"  Features tested: {results['overall']['n_features_tested']}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        if arch_sensitivities:
            print(f"\n{arch.upper()} SUMMARY:")
            print(f"  Mean sensitivity across seeds: {np.mean(arch_sensitivities):.4f} ± {np.std(arch_sensitivities):.4f}")
        
        all_results[arch] = arch_results
    
    # Compare with PWMCC
    print("\n" + "=" * 70)
    print("SENSITIVITY vs STABILITY COMPARISON")
    print("=" * 70)
    print()
    
    # Load PWMCC data
    try:
        with open(OUTPUT_DIR / 'feature_stability.json') as f:
            topk_stability = json.load(f)
        topk_pwmcc = topk_stability.get('mean_overlap', 0.302)
    except:
        topk_pwmcc = 0.302
    
    try:
        with open(OUTPUT_DIR / 'relu_feature_stability.json') as f:
            relu_stability = json.load(f)
        relu_pwmcc = relu_stability.get('mean_overlap', 0.300)
    except:
        relu_pwmcc = 0.300
    
    topk_sens = np.mean([r['overall']['mean_sensitivity'] for r in all_results.get('topk', {}).values()])
    relu_sens = np.mean([r['overall']['mean_sensitivity'] for r in all_results.get('relu', {}).values()])
    
    print("Architecture | PWMCC (stability) | Sensitivity")
    print("-" * 50)
    print(f"TopK         | {topk_pwmcc:.4f}           | {topk_sens:.4f}")
    print(f"ReLU         | {relu_pwmcc:.4f}           | {relu_sens:.4f}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    if topk_sens < 0.5 and relu_sens < 0.5:
        print("  ⚠️ Both architectures show LOW sensitivity (<0.5)")
        print("  Features may be input-specific rather than semantically meaningful")
    elif topk_sens > 0.7 and relu_sens > 0.7:
        print("  ✓ Both architectures show HIGH sensitivity (>0.7)")
        print("  Features respond consistently to similar inputs")
    else:
        print(f"  Mixed results: TopK={topk_sens:.2f}, ReLU={relu_sens:.2f}")
    
    # Save results
    output_path = OUTPUT_DIR / 'feature_sensitivity_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Feature sensitivity measures how consistently features activate")
    print("on semantically similar inputs (nearby values in modular arithmetic).")
    print()
    print(f"TopK mean sensitivity: {topk_sens:.4f}")
    print(f"ReLU mean sensitivity: {relu_sens:.4f}")
    print()
    print("Low sensitivity + low PWMCC suggests features are neither stable")
    print("nor semantically meaningful - they may be fitting noise.")


if __name__ == '__main__':
    main()
