#!/usr/bin/env python3
"""Extract per-SAE metrics from trained SAE checkpoints.

This script loads all 10 SAE checkpoints and extracts:
- Explained variance (EV)
- L0 sparsity
- Architecture parameters
- Feature usage statistics

Outputs a comprehensive metrics table for the paper.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.dataset import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
RESULTS_DIR = BASE_DIR / 'results'
SAES_DIR = RESULTS_DIR / 'saes'
OUTPUT_DIR = RESULTS_DIR / 'analysis'

SEEDS = [42, 123, 456, 789, 1011]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_transformer():
    """Load the trained transformer model."""
    model_path = RESULTS_DIR / 'checkpoints' / 'transformer_grokking.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"Transformer model not found: {model_path}")

    # Load checkpoint to get config
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Initialize model from config
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device=DEVICE)

    # Load weights
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def get_activations(model, dataloader, layer_idx=1, position=2):
    """Extract activations from transformer.

    Args:
        model: Trained transformer (ModularArithmeticTransformer)
        dataloader: Data loader
        layer_idx: Which layer to extract from (0 or 1)
        position: Which position (2 = after op token)

    Returns:
        Tensor of shape [n_samples, d_model]
    """
    activations = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['input_ids'].to(DEVICE)

            # Use TransformerLens's get_activations method
            # Returns [batch, seq_len, d_model]
            layer_acts = model.get_activations(tokens, layer=layer_idx)

            # Extract specific position
            position_acts = layer_acts[:, position, :]
            activations.append(position_acts.cpu())

    return torch.cat(activations, dim=0)


def compute_sae_metrics(sae, activations):
    """Compute reconstruction and sparsity metrics for an SAE.

    Args:
        sae: Trained SAE model
        activations: Input activations [n_samples, d_model]

    Returns:
        Dict of metrics
    """
    sae.eval()

    with torch.no_grad():
        # Move activations to device in batches to avoid OOM
        batch_size = 1024
        all_reconstructions = []
        all_latents = []

        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(DEVICE)
            latents = sae.encode(batch)
            reconstructions = sae.decode(latents)

            all_reconstructions.append(reconstructions.cpu())
            all_latents.append(latents.cpu())

        reconstructions = torch.cat(all_reconstructions, dim=0)
        latents = torch.cat(all_latents, dim=0)

    # Explained variance
    residuals = activations - reconstructions
    total_variance = torch.var(activations)
    residual_variance = torch.var(residuals)
    explained_variance = 1 - (residual_variance / total_variance)

    # L0 sparsity (mean number of active features)
    l0 = (latents.abs() > 1e-6).float().sum(dim=1).mean()

    # L1 norm
    l1 = latents.abs().sum(dim=1).mean()

    # Mean squared error
    mse = torch.mean((activations - reconstructions) ** 2)

    return {
        'explained_variance': float(explained_variance),
        'l0_sparsity': float(l0),
        'l1_norm': float(l1),
        'mse': float(mse),
        'n_active_features': float(l0),
    }


def extract_all_metrics():
    """Extract metrics for all 10 SAEs."""
    print("="*70)
    print("EXTRACTING PER-SAE METRICS")
    print("="*70)
    print()

    # Load transformer
    print("Loading transformer...")
    model = load_transformer()
    print(f"✓ Loaded transformer from {RESULTS_DIR}/checkpoints/transformer_grokking.pt")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ModularArithmeticDataset(p=113, split='train', seed=42)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"✓ Loaded dataset: {len(dataset)} samples")
    print()

    # Extract activations
    print("Extracting transformer activations (layer 1, position 2)...")
    activations = get_activations(model, dataloader, layer_idx=1, position=2)
    print(f"✓ Extracted activations: {activations.shape}")
    print()

    # Process each SAE
    all_metrics = {
        'topk': [],
        'relu': []
    }

    for seed in SEEDS:
        print(f"Processing seed {seed}...")

        # TopK SAE
        topk_path = SAES_DIR / f'topk_seed{seed}' / 'sae_final.pt'
        if topk_path.exists():
            # Load SAE
            sae = TopKSAE(
                input_dim=128,
                latent_dim=1024,  # 8x expansion
                k=32
            ).to(DEVICE)

            checkpoint = torch.load(topk_path, map_location=DEVICE)
            sae.load_state_dict(checkpoint['model_state_dict'])

            # Compute metrics
            metrics = compute_sae_metrics(sae, activations)
            metrics['seed'] = seed
            metrics['architecture'] = 'topk'
            metrics['k'] = 32
            metrics['latent_dim'] = 1024

            all_metrics['topk'].append(metrics)
            print(f"  ✓ TopK: EV={metrics['explained_variance']:.4f}, L0={metrics['l0_sparsity']:.1f}")

        # ReLU SAE
        relu_path = SAES_DIR / f'relu_seed{seed}' / 'sae_final.pt'
        if relu_path.exists():
            # Load SAE
            sae = ReLUSAE(
                input_dim=128,
                latent_dim=1024,
                l1_coeff=0.001
            ).to(DEVICE)

            checkpoint = torch.load(relu_path, map_location=DEVICE)
            sae.load_state_dict(checkpoint['model_state_dict'])

            # Compute metrics
            metrics = compute_sae_metrics(sae, activations)
            metrics['seed'] = seed
            metrics['architecture'] = 'relu'
            metrics['l1_coeff'] = 0.001
            metrics['latent_dim'] = 1024

            all_metrics['relu'].append(metrics)
            print(f"  ✓ ReLU: EV={metrics['explained_variance']:.4f}, L0={metrics['l0_sparsity']:.1f}")

        print()

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'per_sae_metrics.json'

    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("="*70)
    print(f"✓ Saved metrics to {output_path}")
    print("="*70)
    print()

    # Print summary statistics
    print("SUMMARY STATISTICS")
    print("="*70)

    topk_ev = [m['explained_variance'] for m in all_metrics['topk']]
    relu_ev = [m['explained_variance'] for m in all_metrics['relu']]

    print(f"TopK Explained Variance:  {np.mean(topk_ev):.4f} ± {np.std(topk_ev):.4f}")
    print(f"ReLU Explained Variance:  {np.mean(relu_ev):.4f} ± {np.std(relu_ev):.4f}")
    print()

    topk_l0 = [m['l0_sparsity'] for m in all_metrics['topk']]
    relu_l0 = [m['l0_sparsity'] for m in all_metrics['relu']]

    print(f"TopK L0 Sparsity:         {np.mean(topk_l0):.1f} ± {np.std(topk_l0):.1f}")
    print(f"ReLU L0 Sparsity:         {np.mean(relu_l0):.1f} ± {np.std(relu_l0):.1f}")
    print("="*70)

    return all_metrics


if __name__ == '__main__':
    extract_all_metrics()
