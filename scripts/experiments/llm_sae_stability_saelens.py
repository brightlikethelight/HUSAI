#!/usr/bin/env python3
"""LLM SAE Stability Analysis using SAELens Pretrained SAEs

This script tests whether our stability findings transfer to LLM SAEs
using pretrained SAEs from SAELens (Gemma Scope, Llama Scope).

RESEARCH QUESTION:
On algorithmic tasks, stability DECREASES monotonically with L0.
Does this hold for LLMs, or do LLMs show a different pattern
(potentially with an optimal L0)?

APPROACH:
1. Load pretrained SAEs from SAELens for Gemma-2-2B or Llama-3.1-8B
2. Compare SAEs trained with different sparsity levels
3. Measure feature overlap/stability between SAEs at different layers

NOTE: This requires GPU and the following packages:
    pip install sae-lens transformer-lens transformers

GPU REQUIREMENTS:
- Gemma-2-2B: ~8GB VRAM minimum
- Llama-3.1-8B: ~16GB VRAM minimum

For users without GPU access, consider:
1. Google Colab (free T4 GPU)
2. Lambda Labs, Vast.ai, or RunPod (cheap GPU rental)
3. University HPC cluster

Usage:
    python scripts/experiments/llm_sae_stability_saelens.py
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

if DEVICE == 'cpu':
    print("\n" + "=" * 70)
    print("WARNING: No GPU detected!")
    print("This script requires a GPU to run efficiently.")
    print("Options:")
    print("  1. Use Google Colab (free T4 GPU)")
    print("  2. Rent GPU from Lambda Labs, Vast.ai, or RunPod")
    print("  3. Use university HPC cluster")
    print("=" * 70 + "\n")

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'llm_sae_stability'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import sae_lens
        print(f"✓ sae_lens version: {sae_lens.__version__}")
    except ImportError:
        missing.append("sae-lens")
    
    try:
        import transformer_lens
        print(f"✓ transformer_lens available")
    except ImportError:
        missing.append("transformer-lens")
    
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def list_available_saes():
    """List available pretrained SAEs from SAELens."""
    try:
        from sae_lens import SAE
        
        # Get list of available SAE releases
        print("\nAvailable SAE releases:")
        print("-" * 50)
        
        # Known good SAE releases for stability analysis
        recommended = [
            ("gemma-scope-2b-pt-res", "Gemma 2 2B residual stream SAEs"),
            ("gemma-scope-2b-pt-mlp", "Gemma 2 2B MLP SAEs"),
            ("llama-scope-8b-pt-res", "Llama 3.1 8B residual stream SAEs"),
        ]
        
        for release_id, description in recommended:
            print(f"  {release_id}: {description}")
        
        return recommended
        
    except Exception as e:
        print(f"Error listing SAEs: {e}")
        return []


def load_sae_and_model(release_id: str, sae_id: str):
    """Load a pretrained SAE and its corresponding model."""
    from sae_lens import SAE
    
    print(f"\nLoading SAE: {release_id}/{sae_id}")
    
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release_id,
        sae_id=sae_id,
        device=DEVICE
    )
    
    print(f"  ✓ Loaded SAE with {sae.cfg.d_sae} features")
    print(f"  ✓ Sparsity (L0): {sparsity if sparsity else 'unknown'}")
    
    return sae, cfg_dict, sparsity


def compare_sae_features(sae1, sae2) -> float:
    """Compute PWMCC between two SAE decoder matrices."""
    import torch.nn.functional as F
    
    d1 = sae1.W_dec.data  # [d_sae, d_model]
    d2 = sae2.W_dec.data
    
    # Normalize
    d1_norm = F.normalize(d1, dim=1)  # Normalize along d_model
    d2_norm = F.normalize(d2, dim=1)
    
    # Cosine similarity matrix
    cos_sim = d1_norm @ d2_norm.T  # [d_sae1, d_sae2]
    
    # PWMCC
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    
    return (max_1to2 + max_2to1) / 2


def analyze_gemma_scope():
    """Analyze stability across Gemma Scope SAEs."""
    from sae_lens import SAE
    
    print("\n" + "=" * 70)
    print("GEMMA SCOPE STABILITY ANALYSIS")
    print("=" * 70)
    
    # Gemma Scope has SAEs at different widths (sparsity levels)
    # We'll compare SAEs at the same layer but different widths
    
    results = []
    
    # Available widths for Gemma Scope
    # Format: gemma-scope-2b-pt-res-canonical / layer_X / width_YK
    widths = ["16k", "32k", "65k", "131k"]
    layer = 12  # Middle layer
    
    print(f"\nAnalyzing layer {layer} across widths: {widths}")
    
    saes = {}
    for width in widths:
        try:
            sae_id = f"layer_{layer}/width_{width}/canonical"
            sae, cfg, sparsity = load_sae_and_model(
                "gemma-scope-2b-pt-res-canonical",
                sae_id
            )
            saes[width] = {
                'sae': sae,
                'd_sae': sae.cfg.d_sae,
                'sparsity': sparsity
            }
        except Exception as e:
            print(f"  ⚠ Could not load {width}: {e}")
    
    if len(saes) < 2:
        print("Not enough SAEs loaded for comparison")
        return results
    
    # Compare SAEs at different widths
    print("\nComparing SAE features across widths:")
    print("-" * 50)
    
    width_list = list(saes.keys())
    for i, w1 in enumerate(width_list):
        for w2 in width_list[i+1:]:
            pwmcc = compare_sae_features(saes[w1]['sae'], saes[w2]['sae'])
            print(f"  {w1} vs {w2}: PWMCC = {pwmcc:.4f}")
            
            results.append({
                'comparison': f"{w1}_vs_{w2}",
                'width1': w1,
                'width2': w2,
                'd_sae1': saes[w1]['d_sae'],
                'd_sae2': saes[w2]['d_sae'],
                'pwmcc': pwmcc
            })
    
    return results


def analyze_cross_layer_stability():
    """Analyze how SAE features change across layers."""
    from sae_lens import SAE
    
    print("\n" + "=" * 70)
    print("CROSS-LAYER STABILITY ANALYSIS")
    print("=" * 70)
    
    results = []
    width = "16k"  # Use smallest for speed
    layers = [6, 12, 18]  # Early, middle, late
    
    print(f"\nAnalyzing layers {layers} at width {width}")
    
    saes = {}
    for layer in layers:
        try:
            sae_id = f"layer_{layer}/width_{width}/canonical"
            sae, cfg, sparsity = load_sae_and_model(
                "gemma-scope-2b-pt-res-canonical",
                sae_id
            )
            saes[layer] = sae
        except Exception as e:
            print(f"  ⚠ Could not load layer {layer}: {e}")
    
    if len(saes) < 2:
        print("Not enough layers loaded for comparison")
        return results
    
    # Compare adjacent layers
    print("\nComparing SAE features across layers:")
    print("-" * 50)
    
    layer_list = sorted(saes.keys())
    for i in range(len(layer_list) - 1):
        l1, l2 = layer_list[i], layer_list[i+1]
        pwmcc = compare_sae_features(saes[l1], saes[l2])
        print(f"  Layer {l1} vs {l2}: PWMCC = {pwmcc:.4f}")
        
        results.append({
            'layer1': l1,
            'layer2': l2,
            'pwmcc': pwmcc
        })
    
    return results


def main():
    print("=" * 70)
    print("LLM SAE STABILITY ANALYSIS")
    print("Testing if algorithmic task findings transfer to LLMs")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        print("\nAlternatively, run this script in Google Colab:")
        print("  1. Go to colab.research.google.com")
        print("  2. Create new notebook")
        print("  3. Enable GPU: Runtime > Change runtime type > GPU")
        print("  4. Install: !pip install sae-lens transformer-lens")
        print("  5. Copy and run this script")
        return
    
    if DEVICE == 'cpu':
        print("\nSkipping analysis (requires GPU)")
        print("See instructions above for GPU options.")
        return
    
    # Run analyses
    results = {
        'device': DEVICE,
        'analyses': {}
    }
    
    try:
        # List available SAEs
        list_available_saes()
        
        # Analyze Gemma Scope
        gemma_results = analyze_gemma_scope()
        results['analyses']['gemma_scope'] = gemma_results
        
        # Analyze cross-layer stability
        layer_results = analyze_cross_layer_stability()
        results['analyses']['cross_layer'] = layer_results
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_path = OUTPUT_DIR / 'llm_sae_stability_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTo fully test our hypothesis on LLMs, we need to:")
    print("1. Train multiple SAEs with SAME architecture but DIFFERENT seeds")
    print("2. Compare feature stability across seeds (like we did for algorithmic tasks)")
    print("3. Vary sparsity (L0) and measure stability at each level")
    print("\nThe pretrained SAEs from SAELens are useful for:")
    print("- Understanding feature structure at different widths")
    print("- Cross-layer feature comparison")
    print("- But NOT for seed-to-seed stability (only one SAE per config)")


if __name__ == '__main__':
    main()
