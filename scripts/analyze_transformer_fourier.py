#!/usr/bin/env python3
"""Analyze Fourier structure in trained transformer.

This script investigates WHERE Fourier circuits appear in the grokking transformer
for modular arithmetic. We check multiple locations:
- Embeddings (raw Fourier basis)
- Layer 0 attention output
- Layer 0 residual stream
- Layer 1 attention output  
- Layer 1 residual stream

This diagnostic will reveal where SAEs should be trained for best Fourier recovery.

Usage:
    python scripts/analyze_transformer_fourier.py \
        --checkpoint results/transformer_5000ep/transformer_best.pt \
        --modulus 113 \
        --output results/analysis/fourier_locations.json

Reference: Nanda et al. "Progress measures for grokking via mechanistic interpretability"
Shows that Fourier circuits appear in attention layers with cosine attention patterns.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.analysis.fourier_validation import get_fourier_basis, compute_fourier_overlap


def extract_activations_at_locations(
    model,
    inputs: torch.Tensor,
    locations: List[str]
) -> Dict[str, torch.Tensor]:
    """Extract activations from multiple locations in the transformer.
    
    Locations to check:
    - 'embed': Embedding layer output
    - 'blocks.0.attn': Layer 0 attention output
    - 'blocks.0.hook_resid_post': Layer 0 residual stream (after attn + MLP)
    - 'blocks.1.attn': Layer 1 attention output
    - 'blocks.1.hook_resid_post': Layer 1 residual stream (after attn + MLP)
    
    Args:
        model: Trained transformer
        inputs: Input token IDs [batch, seq_len]
        locations: List of hook names to extract from
        
    Returns:
        activations: Dict mapping location name to activations [batch, seq_len, d_model]
    """
    activations = {}
    
    # Register hooks to capture activations
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            # Store activation (detach to avoid gradients)
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Get the underlying HookedTransformer
    hooked_model = model.model
    
    # Register hooks for each location
    for loc in locations:
        if loc == 'embed':
            hook = hooked_model.embed.register_forward_hook(make_hook(loc))
            hooks.append(hook)
        elif 'blocks' in loc:
            # Parse block number and component
            parts = loc.split('.')
            block_num = int(parts[1])
            component = '.'.join(parts[2:])
            
            if component == 'attn':
                hook = hooked_model.blocks[block_num].attn.register_forward_hook(make_hook(loc))
                hooks.append(hook)
            elif component == 'hook_resid_post':
                hook = hooked_model.blocks[block_num].register_forward_hook(make_hook(loc))
                hooks.append(hook)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def analyze_fourier_at_location(
    activations: torch.Tensor,
    fourier_basis: torch.Tensor,
    position_idx: int = -2,
    location_name: str = ""
) -> Dict:
    """Analyze Fourier structure at a specific location.
    
    Args:
        activations: Activations [batch, seq_len, d_model]
        fourier_basis: Fourier basis [n_fourier, p] where p=modulus
        position_idx: Which token position to analyze (-2 = answer position)
        location_name: Name for this location
        
    Returns:
        results: Dict with overlap, statistics, etc.
    """
    # Extract activations at target position
    act_at_pos = activations[:, position_idx, :]  # [batch, d_model]
    
    # For transformer activations, we treat each sample as a "feature"
    # So we transpose to [d_model, batch] to match SAE decoder format [d_model, d_sae]
    # This allows us to use compute_fourier_overlap
    transposed_act = act_at_pos.T  # [d_model, batch]
    
    # Compute overlap with Fourier basis (this handles dimension mismatch internally)
    overlap = compute_fourier_overlap(transposed_act, fourier_basis)
    
    results = {
        'location': location_name,
        'overall_overlap': float(overlap),
        'activation_shape': list(activations.shape),
        'activation_mean': float(activations.mean()),
        'activation_std': float(activations.std()),
        'n_samples': act_at_pos.shape[0],
        'd_model': act_at_pos.shape[1],
    }
    
    return results


def generate_test_samples(modulus: int, n_samples: int = 1000, device: str = 'cpu'):
    """Generate test samples for modular arithmetic.
    
    Args:
        modulus: Modulus for arithmetic (e.g., 113)
        n_samples: Number of samples
        device: Device
        
    Returns:
        inputs: Input tokens [n_samples, seq_len]
        Format: [BOS, a, +, b, =, answer, EOS]
    """
    import random
    
    inputs = []
    for _ in range(n_samples):
        a = random.randint(0, modulus - 1)
        b = random.randint(0, modulus - 1)
        answer = (a + b) % modulus
        
        # Token IDs (assuming vocab: [0..p-1]=numbers, p=+, p+1==, p+2=BOS, p+3=EOS)
        bos_token = modulus + 2
        eos_token = modulus + 3
        plus_token = modulus
        equals_token = modulus + 1
        
        seq = [bos_token, a, plus_token, b, equals_token, answer, eos_token]
        inputs.append(seq)
    
    return torch.tensor(inputs, dtype=torch.long, device=device)


def visualize_results(results: List[Dict], save_path: Path):
    """Create visualization of Fourier overlap at different locations.
    
    Args:
        results: List of result dicts from analyze_fourier_at_location
        save_path: Where to save figure
    """
    locations = [r['location'] for r in results]
    overlaps = [r['overall_overlap'] for r in results]
    stds = [0.0] * len(overlaps)  # No per-sample stats available
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(locations))
    bars = ax.bar(x, overlaps, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    
    # Add horizontal line at 0.6 (target threshold)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target (0.6)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Current SAE (0.26)')
    
    # Color bars based on overlap level
    for i, (bar, overlap) in enumerate(zip(bars, overlaps)):
        if overlap >= 0.6:
            bar.set_color('green')
            bar.set_alpha(0.8)
        elif overlap >= 0.4:
            bar.set_color('yellow')
            bar.set_alpha(0.8)
        else:
            bar.set_color('red')
            bar.set_alpha(0.6)
    
    ax.set_xlabel('Location in Transformer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fourier Overlap', fontsize=12, fontweight='bold')
    ax.set_title('Fourier Circuit Analysis Across Transformer Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(locations, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, overlap) in enumerate(zip(bars, overlaps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{overlap:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Fourier structure in transformer")
    
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to trained transformer checkpoint')
    parser.add_argument('--modulus', type=int, default=113,
                       help='Modulus for arithmetic (default: 113)')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of test samples (default: 1000)')
    parser.add_argument('--output', type=Path, default='results/analysis/fourier_locations.json',
                       help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRANSFORMER FOURIER CIRCUIT ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Modulus: {args.modulus}")
    print(f"Test samples: {args.n_samples}")
    print()
    
    # Load transformer
    print("Loading transformer...")
    model, extras = ModularArithmeticTransformer.load_checkpoint(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    print(f"✅ Loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Get Fourier basis
    print("Computing Fourier basis...")
    fourier_basis = get_fourier_basis(modulus=args.modulus)
    print(f"✅ Fourier basis: {fourier_basis.shape}")
    print()
    
    # Generate test samples
    print("Generating test samples...")
    test_inputs = generate_test_samples(args.modulus, args.n_samples, args.device)
    print(f"✅ Generated {test_inputs.shape[0]} samples")
    print()
    
    # Locations to analyze
    locations = [
        'embed',
        'blocks.0.attn',
        'blocks.0.hook_resid_post',
        'blocks.1.attn',
        'blocks.1.hook_resid_post',
    ]
    
    print("Extracting activations from all locations...")
    activations = extract_activations_at_locations(model, test_inputs, locations)
    
    for loc, act in activations.items():
        print(f"  {loc}: {act.shape}")
    print()
    
    # Analyze Fourier overlap at each location
    print("="*80)
    print("ANALYZING FOURIER OVERLAP AT EACH LOCATION")
    print("="*80)
    
    results = []
    
    for loc in locations:
        print(f"\n[{loc}]")
        act = activations[loc]
        
        # Analyze at answer position (-2)
        result = analyze_fourier_at_location(
            act,
            fourier_basis,
            position_idx=-2,
            location_name=loc
        )
        
        results.append(result)
        
        print(f"  Fourier overlap: {result['overall_overlap']:.4f}")
        print(f"  Activation stats: mean={result['activation_mean']:.4f}, std={result['activation_std']:.4f}")
        print(f"  Shape: {result['activation_shape']}")
        
        # Interpret
        if result['overall_overlap'] >= 0.6:
            print(f"  ✅ STRONG Fourier structure - IDEAL for SAE training!")
        elif result['overall_overlap'] >= 0.4:
            print(f"  ⚠️  MODERATE Fourier structure - Could work for SAEs")
        else:
            print(f"  ❌ WEAK Fourier structure - Poor choice for SAE training")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Find best location
    best_result = max(results, key=lambda r: r['overall_overlap'])
    print(f"\n🏆 Best location: {best_result['location']}")
    print(f"   Fourier overlap: {best_result['overall_overlap']:.4f}")
    
    # Compare to current SAE location (blocks.1.hook_resid_post)
    current_sae_loc = 'blocks.1.hook_resid_post'
    current_result = next((r for r in results if r['location'] == current_sae_loc), None)
    
    if current_result:
        print(f"\n📍 Current SAE extraction location: {current_sae_loc}")
        print(f"   Fourier overlap: {current_result['overall_overlap']:.4f}")
        print(f"   Current SAE achieves: ~0.26 overlap")
        
        if current_result['overall_overlap'] >= 0.6:
            print(f"\n🔍 CRITICAL FINDING:")
            print(f"   Transformer HAS strong Fourier structure ({current_result['overall_overlap']:.3f})")
            print(f"   BUT SAEs only achieve ~0.26 overlap")
            print(f"   → SAEs fundamentally struggle to extract known circuits!")
            print(f"   → This is a MAJOR research finding about SAE limitations")
        elif best_result['overall_overlap'] >= 0.6:
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Train SAEs from '{best_result['location']}' instead")
            print(f"   Expected improvement: {best_result['overall_overlap']:.3f} vs {current_result['overall_overlap']:.3f}")
        else:
            print(f"\n⚠️  CONCERN:")
            print(f"   Weak Fourier structure everywhere (max: {best_result['overall_overlap']:.3f})")
            print(f"   Transformer may not have learned Fourier circuits properly")
            print(f"   → Consider retraining transformer or checking training setup")
    
    # Save results
    output_data = {
        'modulus': args.modulus,
        'n_samples': args.n_samples,
        'checkpoint': str(args.checkpoint),
        'results': results,
        'best_location': best_result['location'],
        'best_overlap': best_result['overall_overlap'],
        'current_sae_location': current_sae_loc,
        'current_sae_overlap': current_result['overall_overlap'] if current_result else None,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Create visualization
    viz_path = args.output.parent / 'fourier_locations.png'
    visualize_results(results, viz_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
