#!/usr/bin/env python3
"""Phase 4.3: Training Dynamics Analysis.

This script analyzes WHEN features diverge during training by:
1. Training SAEs with checkpoints saved every N epochs
2. Computing PWMCC at each checkpoint
3. Tracking PWMCC evolution over training

Key questions to answer:
- Do features start aligned and diverge? Or diverge from start?
- Is there a critical epoch where divergence accelerates?
- Does PWMCC stabilize, or continue changing?

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/analyze_training_dynamics.py
    
Runtime: ~3-4 hours (3 seed pairs × ~1 hour each)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
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
OUTPUT_DIR = RESULTS_DIR / 'training_dynamics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEED_PAIRS = [(42, 123), (42, 456), (42, 789)]  # 3 pairs for comparison
CHECKPOINT_INTERVAL = 5  # Save every 5 epochs
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAE hyperparameters
D_MODEL = 128
D_SAE = 1024
K = 32
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
LAYER_IDX = 1
POSITION = -2


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder."""
    
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder()

    def normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm after updates."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.encoder(x)
        topk_values, topk_indices = torch.topk(latents, k=self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_indices, topk_values)
        return sparse_latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed, latents


def load_transformer() -> ModularArithmeticTransformer:
    """Load trained transformer model."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device=DEVICE)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def extract_activations(model: ModularArithmeticTransformer) -> torch.Tensor:
    """Extract activations from transformer."""
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    activations = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                tokens, _ = batch
            else:
                tokens = batch
            
            tokens = tokens.to(DEVICE)
            layer_acts = model.get_activations(tokens, layer=LAYER_IDX)
            position_acts = layer_acts[:, POSITION, :]
            activations.append(position_acts.cpu())
    
    return torch.cat(activations, dim=0)


def compute_decoder_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC using decoder-based method."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    
    return (max_1to2 + max_2to1) / 2


def train_sae_with_checkpoints(
    activations: torch.Tensor,
    seed: int,
    checkpoint_epochs: List[int]
) -> Dict[int, torch.Tensor]:
    """Train SAE and save decoder at specified epochs.
    
    Returns:
        Dict mapping epoch -> decoder weights
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    sae = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, k=K).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    checkpoints = {}
    
    # Save initial (epoch 0)
    checkpoints[0] = sae.decoder.weight.detach().cpu().clone()
    
    for epoch in range(NUM_EPOCHS):
        sae.train()
        for (batch_acts,) in dataloader:
            batch_acts = batch_acts.to(DEVICE)
            reconstructed, latents = sae(batch_acts)
            
            recon_loss = F.mse_loss(reconstructed, batch_acts)
            
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            
            # CRITICAL: Normalize decoder after every step
            sae.normalize_decoder()
        
        # Save checkpoint if needed
        if (epoch + 1) in checkpoint_epochs:
            checkpoints[epoch + 1] = sae.decoder.weight.detach().cpu().clone()
    
    return checkpoints


def analyze_seed_pair(
    activations: torch.Tensor,
    seed1: int,
    seed2: int,
    checkpoint_epochs: List[int]
) -> Dict:
    """Analyze PWMCC evolution for a seed pair."""
    print(f"\n{'='*60}")
    print(f"Analyzing seed pair: {seed1} vs {seed2}")
    print(f"{'='*60}")
    
    # Train both SAEs with checkpoints
    print(f"Training SAE (seed={seed1})...")
    checkpoints1 = train_sae_with_checkpoints(activations, seed1, checkpoint_epochs)
    
    print(f"Training SAE (seed={seed2})...")
    checkpoints2 = train_sae_with_checkpoints(activations, seed2, checkpoint_epochs)
    
    # Compute PWMCC at each checkpoint
    print("Computing PWMCC evolution...")
    pwmcc_evolution = {}
    
    for epoch in [0] + checkpoint_epochs:
        if epoch in checkpoints1 and epoch in checkpoints2:
            pwmcc = compute_decoder_pwmcc(checkpoints1[epoch], checkpoints2[epoch])
            pwmcc_evolution[epoch] = pwmcc
            print(f"  Epoch {epoch:2d}: PWMCC = {pwmcc:.4f}")
    
    return {
        'seed1': seed1,
        'seed2': seed2,
        'checkpoint_epochs': [0] + checkpoint_epochs,
        'pwmcc_evolution': pwmcc_evolution
    }


def plot_training_dynamics(results: List[Dict], output_path: Path):
    """Generate training dynamics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: PWMCC evolution for each seed pair
    ax1 = axes[0]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, result in enumerate(results):
        epochs = list(result['pwmcc_evolution'].keys())
        pwmcc_values = list(result['pwmcc_evolution'].values())
        label = f"Seed {result['seed1']} vs {result['seed2']}"
        ax1.plot(epochs, pwmcc_values, 'o-', color=colors[i], label=label, linewidth=2, markersize=6)
    
    ax1.axhline(y=0.30, color='gray', linestyle='--', alpha=0.7, label='Random baseline')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('PWMCC (Decoder-based)')
    ax1.set_title('Feature Consistency During Training')
    ax1.legend()
    ax1.set_ylim(0, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Right: Average PWMCC with error band
    ax2 = axes[1]
    
    # Compute average and std across seed pairs
    all_epochs = results[0]['checkpoint_epochs']
    avg_pwmcc = []
    std_pwmcc = []
    
    for epoch in all_epochs:
        values = [r['pwmcc_evolution'].get(epoch, np.nan) for r in results]
        values = [v for v in values if not np.isnan(v)]
        if values:
            avg_pwmcc.append(np.mean(values))
            std_pwmcc.append(np.std(values))
        else:
            avg_pwmcc.append(np.nan)
            std_pwmcc.append(np.nan)
    
    avg_pwmcc = np.array(avg_pwmcc)
    std_pwmcc = np.array(std_pwmcc)
    
    ax2.fill_between(all_epochs, avg_pwmcc - std_pwmcc, avg_pwmcc + std_pwmcc, 
                     alpha=0.3, color='#3498db')
    ax2.plot(all_epochs, avg_pwmcc, 'o-', color='#3498db', linewidth=2, markersize=6,
             label='Average PWMCC')
    ax2.axhline(y=0.30, color='gray', linestyle='--', alpha=0.7, label='Random baseline')
    
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('PWMCC (Decoder-based)')
    ax2.set_title('Average Feature Consistency (n=3 pairs)')
    ax2.legend()
    ax2.set_ylim(0, 0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("PHASE 4.3: TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seed pairs: {SEED_PAIRS}")
    print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} epochs")
    print()
    
    # Load transformer and extract activations
    print("Loading transformer and extracting activations...")
    model = load_transformer()
    activations = extract_activations(model)
    print(f"✓ Extracted {activations.shape[0]} samples × {activations.shape[1]} dims")
    
    # Define checkpoint epochs
    checkpoint_epochs = list(range(CHECKPOINT_INTERVAL, NUM_EPOCHS + 1, CHECKPOINT_INTERVAL))
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    
    # Analyze each seed pair
    results = []
    for seed1, seed2 in SEED_PAIRS:
        result = analyze_seed_pair(activations, seed1, seed2, checkpoint_epochs)
        results.append(result)
    
    # Save results
    results_path = OUTPUT_DIR / 'training_dynamics_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")
    
    # Generate visualization
    plot_training_dynamics(results, OUTPUT_DIR / 'training_dynamics.png')
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS SUMMARY")
    print("=" * 70)
    
    # Analyze key patterns
    for result in results:
        epochs = list(result['pwmcc_evolution'].keys())
        pwmcc_values = list(result['pwmcc_evolution'].values())
        
        initial_pwmcc = pwmcc_values[0]
        final_pwmcc = pwmcc_values[-1]
        max_pwmcc = max(pwmcc_values)
        max_epoch = epochs[pwmcc_values.index(max_pwmcc)]
        
        print(f"\nSeed {result['seed1']} vs {result['seed2']}:")
        print(f"  Initial (epoch 0): {initial_pwmcc:.4f}")
        print(f"  Final (epoch {epochs[-1]}): {final_pwmcc:.4f}")
        print(f"  Peak: {max_pwmcc:.4f} at epoch {max_epoch}")
        print(f"  Change: {final_pwmcc - initial_pwmcc:+.4f}")
    
    # Overall interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Check if features diverge or converge
    all_initial = [r['pwmcc_evolution'][0] for r in results]
    all_final = [r['pwmcc_evolution'][max(r['pwmcc_evolution'].keys())] for r in results]
    
    avg_initial = np.mean(all_initial)
    avg_final = np.mean(all_final)
    
    print(f"\nAverage initial PWMCC: {avg_initial:.4f}")
    print(f"Average final PWMCC: {avg_final:.4f}")
    print(f"Change: {avg_final - avg_initial:+.4f}")
    
    if avg_final > avg_initial + 0.02:
        print("\n→ Features CONVERGE during training (PWMCC increases)")
    elif avg_final < avg_initial - 0.02:
        print("\n→ Features DIVERGE during training (PWMCC decreases)")
    else:
        print("\n→ Features remain STABLE during training (PWMCC unchanged)")
    
    if abs(avg_final - 0.30) < 0.02:
        print("→ Final PWMCC matches random baseline (~0.30)")
    
    print("\n" + "=" * 70)
    print("✓ PHASE 4.3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
