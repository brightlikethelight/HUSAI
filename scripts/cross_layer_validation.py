#!/usr/bin/env python3
"""Cross-layer validation experiment for SAE feature stability.

This experiment validates that the ~0.30 PWMCC baseline is not layer-specific,
but holds across different transformer layers.

Experiment design:
1. Extract activations from layer 0 (vs layer 1 in original experiments)
2. Train 5 TopK SAEs on layer 0 with different random seeds
3. Compute pairwise PWMCC among all 5 SAEs
4. Compare results with layer 1 (original results)

Expected outcome:
- Layer 0 PWMCC ≈ 0.30 (similar to layer 1)
- This demonstrates layer-independence of the instability phenomenon

Runtime: ~6-8 hours on CPU (5 SAEs × ~90 min each)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
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
OUTPUT_DIR = RESULTS_DIR / 'cross_layer_validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configuration
LAYER_IDX = 0  # Test on layer 0 (first layer)
POSITION = 2   # Same position as original (after operator token)
SEEDS = [42, 123, 456, 789, 1011]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAE hyperparameters (match original TopK experiments)
D_MODEL = 128
D_SAE = 1024  # 8x expansion
K = 32        # Top-32 activation
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
BATCH_SIZE = 256


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Matches the original TopK SAE architecture used in layer 1 experiments.
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Encoder: d_model -> d_sae
        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        # Decoder: d_sae -> d_model
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize weights
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations with TopK sparsity.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            Sparse latents [batch, d_sae]
        """
        # Linear projection
        latents = self.encoder(x)

        # TopK activation (per sample)
        # Keep only top-k values, zero out rest
        topk_values, topk_indices = torch.topk(latents, k=self.k, dim=-1)

        # Create sparse tensor
        sparse_latents = torch.zeros_like(latents)
        sparse_latents.scatter_(-1, topk_indices, topk_values)

        return sparse_latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to reconstruction.

        Args:
            latents: Sparse latents [batch, d_sae]

        Returns:
            Reconstructed activations [batch, d_model]
        """
        return self.decoder(latents)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input activations [batch, d_model]

        Returns:
            reconstructed: Reconstructed activations [batch, d_model]
            latents: Sparse latent activations [batch, d_sae]
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed, latents


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

    print(f"✓ Loaded transformer (config: {config.n_layers} layers, {config.d_model} dims)")
    return model


def extract_activations(
    model: ModularArithmeticTransformer,
    dataloader: DataLoader,
    layer: int,
    position: int
) -> torch.Tensor:
    """Extract activations from specific layer and position.

    Args:
        model: Trained transformer
        dataloader: Data loader
        layer: Which layer to extract from
        position: Which token position

    Returns:
        Activations tensor [n_samples, d_model]
    """
    print(f"Extracting activations from layer {layer}, position {position}...")
    activations = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            # Dataset returns (tokens, labels) tuple
            if isinstance(batch, (tuple, list)):
                tokens, _ = batch
            else:
                tokens = batch

            tokens = tokens.to(DEVICE)

            # Get activations using TransformerLens API
            layer_acts = model.get_activations(tokens, layer=layer)

            # Extract specific position
            position_acts = layer_acts[:, position, :]
            activations.append(position_acts.cpu())

    activations = torch.cat(activations, dim=0)
    print(f"✓ Extracted {activations.shape[0]} samples × {activations.shape[1]} dims")

    return activations


def train_sae(
    activations: torch.Tensor,
    seed: int,
    save_dir: Path
) -> TopKSAE:
    """Train a single TopK SAE.

    Args:
        activations: Training activations [n_samples, d_model]
        seed: Random seed for reproducibility
        save_dir: Directory to save checkpoint

    Returns:
        Trained SAE model
    """
    print(f"\n{'='*70}")
    print(f"Training TopK SAE (seed={seed})")
    print(f"{'='*70}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    sae = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, k=K).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=LEARNING_RATE)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    best_loss = float('inf')
    training_log = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_ev = 0.0
        n_batches = 0

        sae.train()
        for (batch_acts,) in dataloader:
            batch_acts = batch_acts.to(DEVICE)

            # Forward pass
            reconstructed, latents = sae(batch_acts)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, batch_acts)

            # Backward pass
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                # Explained variance
                total_var = torch.var(batch_acts)
                residual_var = torch.var(batch_acts - reconstructed)
                explained_var = 1 - (residual_var / total_var)

                epoch_loss += recon_loss.item()
                epoch_ev += explained_var.item()
                n_batches += 1

        # Epoch statistics
        avg_loss = epoch_loss / n_batches
        avg_ev = epoch_ev / n_batches

        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'explained_variance': avg_ev
        })

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
                  f"Loss={avg_loss:.4f}, EV={avg_ev:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = {
                'model_state_dict': sae.state_dict(),
                'd_model': D_MODEL,
                'd_sae': D_SAE,
                'k': K,
                'seed': seed,
                'layer': LAYER_IDX,
                'position': POSITION,
                'best_loss': best_loss,
                'final_ev': avg_ev
            }

    # Save final checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f'layer{LAYER_IDX}_seed{seed}.pt'
    torch.save(best_checkpoint, checkpoint_path)

    # Save training log
    log_path = save_dir / f'layer{LAYER_IDX}_seed{seed}_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"✓ Saved SAE to {checkpoint_path}")
    print(f"  Final: Loss={best_loss:.4f}, EV={training_log[-1]['explained_variance']:.4f}")

    return sae


def compute_pwmcc(sae1: TopKSAE, sae2: TopKSAE, activations: torch.Tensor) -> float:
    """Compute Pairwise Maximum Cosine Correlation between two SAEs.

    Args:
        sae1: First SAE
        sae2: Second SAE
        activations: Sample activations to encode [n_samples, d_model]

    Returns:
        PWMCC score (0 to 1)
    """
    sae1.eval()
    sae2.eval()

    with torch.no_grad():
        # Encode with both SAEs (in batches to avoid OOM)
        batch_size = 1024
        latents1_list = []
        latents2_list = []

        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(DEVICE)
            latents1_list.append(sae1.encode(batch).cpu())
            latents2_list.append(sae2.encode(batch).cpu())

        latents1 = torch.cat(latents1_list, dim=0)  # [n_samples, d_sae]
        latents2 = torch.cat(latents2_list, dim=0)

        # Normalize to unit vectors
        latents1 = F.normalize(latents1, p=2, dim=0)  # Normalize along sample dimension
        latents2 = F.normalize(latents2, p=2, dim=0)

        # Compute cosine similarity matrix [d_sae1, d_sae2]
        cosine_sim = torch.mm(latents1.T, latents2)

        # PWMCC: for each feature in sae1, find max correlation with sae2
        max_corr, _ = torch.max(torch.abs(cosine_sim), dim=1)
        pwmcc = torch.mean(max_corr).item()

    return pwmcc


def run_cross_layer_experiment():
    """Run the complete cross-layer validation experiment."""
    print("="*70)
    print("CROSS-LAYER VALIDATION EXPERIMENT")
    print("="*70)
    print(f"Target layer: {LAYER_IDX}")
    print(f"Position: {POSITION}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print()

    # 1. Load transformer
    model = load_transformer()

    # 2. Load dataset
    print("\nLoading dataset...")
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"✓ Loaded {len(dataset)} training samples")

    # 3. Extract activations from layer 0
    activations = extract_activations(model, dataloader, layer=LAYER_IDX, position=POSITION)

    # 4. Train 5 TopK SAEs with different seeds
    print(f"\n{'='*70}")
    print(f"TRAINING {len(SEEDS)} TOPK SAEs")
    print(f"{'='*70}")

    trained_saes = {}
    for seed in SEEDS:
        sae = train_sae(activations, seed, save_dir=OUTPUT_DIR)
        trained_saes[seed] = sae

    # 5. Compute pairwise PWMCC
    print(f"\n{'='*70}")
    print("COMPUTING PAIRWISE PWMCC")
    print(f"{'='*70}")

    n = len(SEEDS)
    overlap_matrix = np.ones((n, n))
    pairwise_overlaps = {}

    for i, seed1 in enumerate(SEEDS):
        for j, seed2 in enumerate(SEEDS):
            if i < j:  # Only compute upper triangle
                print(f"Computing PWMCC: seed{seed1} vs seed{seed2}...")
                pwmcc = compute_pwmcc(
                    trained_saes[seed1],
                    trained_saes[seed2],
                    activations[:5000]  # Use subset for speed
                )
                overlap_matrix[i, j] = pwmcc
                overlap_matrix[j, i] = pwmcc  # Symmetric
                pairwise_overlaps[f"{seed1}_{seed2}"] = pwmcc
                print(f"  PWMCC = {pwmcc:.4f}")

    # 6. Compute statistics
    upper_triangle = overlap_matrix[np.triu_indices(n, k=1)]
    stats = {
        'layer': LAYER_IDX,
        'position': POSITION,
        'mean_overlap': float(np.mean(upper_triangle)),
        'std_overlap': float(np.std(upper_triangle)),
        'min_overlap': float(np.min(upper_triangle)),
        'max_overlap': float(np.max(upper_triangle)),
        'median_overlap': float(np.median(upper_triangle)),
        'n_saes': len(SEEDS),
        'seeds': SEEDS
    }

    # 7. Save results
    results = {
        'stats': stats,
        'pairwise_overlaps': pairwise_overlaps,
        'overlap_matrix': overlap_matrix.tolist()
    }

    # Save JSON
    json_path = OUTPUT_DIR / f'layer{LAYER_IDX}_stability_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save pickle (for overlap matrix)
    pkl_path = OUTPUT_DIR / f'layer{LAYER_IDX}_overlap_matrix.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(overlap_matrix, f)

    # 8. Compare with layer 1 results
    print(f"\n{'='*70}")
    print("CROSS-LAYER COMPARISON")
    print(f"{'='*70}")

    # Load layer 1 results
    layer1_path = RESULTS_DIR / 'analysis' / 'feature_stability.json'
    if layer1_path.exists():
        with open(layer1_path, 'r') as f:
            layer1_results = json.load(f)

        print(f"Layer 0 (new):  PWMCC = {stats['mean_overlap']:.4f} ± {stats['std_overlap']:.4f}")
        print(f"Layer 1 (orig): PWMCC = {layer1_results['stats']['mean_overlap']:.4f} ± {layer1_results['stats']['std_overlap']:.4f}")

        diff = abs(stats['mean_overlap'] - layer1_results['stats']['mean_overlap'])
        print(f"\nAbsolute difference: {diff:.4f}")

        if diff < 0.05:
            print("✓ Layer-independence CONFIRMED (difference < 0.05)")
        else:
            print("⚠ Significant layer difference detected")
    else:
        print("⚠ Layer 1 results not found, cannot compare")

    print(f"\n{'='*70}")
    print("✓ EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("  - layer0_stability_results.json")
    print("  - layer0_overlap_matrix.pkl")
    print("  - layer0_seed*.pt (5 SAE checkpoints)")
    print("  - layer0_seed*_log.json (5 training logs)")

    return results


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP issue on macOS

    results = run_cross_layer_experiment()
