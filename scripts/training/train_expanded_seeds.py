#!/usr/bin/env python3
"""Phase 3: Train additional SAEs to expand sample size from n=5 to n=15.

This script trains 10 additional TopK SAEs and 10 additional ReLU SAEs
with new random seeds to increase statistical power.

Original seeds: [42, 123, 456, 789, 1011]
New seeds: [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031]

Expected runtime: ~30 hours (20 SAEs × ~1.5 hours each)
Can be parallelized if multiple GPUs available.

Usage:
    # Train all new SAEs (sequential)
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/training/train_expanded_seeds.py
    
    # Train specific architecture
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/training/train_expanded_seeds.py --arch topk
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/training/train_expanded_seeds.py --arch relu
    
    # Train specific seed range
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/training/train_expanded_seeds.py --seeds 2022,2023,2024
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import sys
import time

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
SAE_DIR = RESULTS_DIR / 'saes'

# Configuration
ORIGINAL_SEEDS = [42, 123, 456, 789, 1011]
NEW_SEEDS = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031]
ALL_SEEDS = ORIGINAL_SEEDS + NEW_SEEDS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAE hyperparameters (match original experiments)
D_MODEL = 128
D_SAE = 1024  # 8x expansion
K = 32        # TopK sparsity
L1_COEF = 1e-3  # ReLU L1 coefficient
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
BATCH_SIZE = 256
LAYER_IDX = 1
POSITION = -2  # Answer position


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


class ReLUSAE(nn.Module):
    """ReLU Sparse Autoencoder with L1 penalty."""
    
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        
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
        return F.relu(self.encoder(x))
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed, latents


def load_transformer() -> ModularArithmeticTransformer:
    """Load trained transformer model."""
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Transformer not found: {model_path}")
    
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


def train_topk_sae(activations: torch.Tensor, seed: int, save_dir: Path) -> Dict:
    """Train a TopK SAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    sae = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, k=K).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    training_log = []
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_ev = 0.0
        n_batches = 0
        
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
            
            with torch.no_grad():
                total_var = torch.var(batch_acts)
                residual_var = torch.var(batch_acts - reconstructed)
                explained_var = 1 - (residual_var / total_var)
                
                epoch_loss += recon_loss.item()
                epoch_ev += explained_var.item()
                n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_ev = epoch_ev / n_batches
        
        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'explained_variance': avg_ev
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = sae.state_dict().copy()
            best_ev = avg_ev
    
    # Save checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / 'sae.pt'
    torch.save({
        'model_state_dict': best_state,
        'd_model': D_MODEL,
        'd_sae': D_SAE,
        'k': K,
        'seed': seed,
        'layer': LAYER_IDX,
        'position': POSITION,
        'best_loss': best_loss,
        'final_ev': best_ev,
        'architecture': 'topk'
    }, checkpoint_path)
    
    # Save training log
    log_path = save_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return {
        'seed': seed,
        'architecture': 'topk',
        'best_loss': best_loss,
        'final_ev': best_ev,
        'checkpoint_path': str(checkpoint_path)
    }


def train_relu_sae(activations: torch.Tensor, seed: int, save_dir: Path) -> Dict:
    """Train a ReLU SAE with L1 penalty."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    sae = ReLUSAE(d_model=D_MODEL, d_sae=D_SAE).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    training_log = []
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_ev = 0.0
        epoch_l0 = 0.0
        n_batches = 0
        
        sae.train()
        for (batch_acts,) in dataloader:
            batch_acts = batch_acts.to(DEVICE)
            reconstructed, latents = sae(batch_acts)
            
            recon_loss = F.mse_loss(reconstructed, batch_acts)
            l1_loss = L1_COEF * latents.abs().mean()
            total_loss = recon_loss + l1_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # CRITICAL: Normalize decoder after every step
            sae.normalize_decoder()
            
            with torch.no_grad():
                total_var = torch.var(batch_acts)
                residual_var = torch.var(batch_acts - reconstructed)
                explained_var = 1 - (residual_var / total_var)
                l0 = (latents > 0).float().sum(dim=-1).mean()
                
                epoch_loss += recon_loss.item()
                epoch_ev += explained_var.item()
                epoch_l0 += l0.item()
                n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_ev = epoch_ev / n_batches
        avg_l0 = epoch_l0 / n_batches
        
        training_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'explained_variance': avg_ev,
            'l0': avg_l0
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = sae.state_dict().copy()
            best_ev = avg_ev
            best_l0 = avg_l0
    
    # Save checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / 'sae.pt'
    torch.save({
        'model_state_dict': best_state,
        'd_model': D_MODEL,
        'd_sae': D_SAE,
        'l1_coef': L1_COEF,
        'seed': seed,
        'layer': LAYER_IDX,
        'position': POSITION,
        'best_loss': best_loss,
        'final_ev': best_ev,
        'final_l0': best_l0,
        'architecture': 'relu'
    }, checkpoint_path)
    
    # Save training log
    log_path = save_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return {
        'seed': seed,
        'architecture': 'relu',
        'best_loss': best_loss,
        'final_ev': best_ev,
        'final_l0': best_l0,
        'checkpoint_path': str(checkpoint_path)
    }


def compute_decoder_pwmcc(decoder1: torch.Tensor, decoder2: torch.Tensor) -> float:
    """Compute PWMCC using decoder-based method (CORRECT for sparse SAEs)."""
    d1_norm = F.normalize(decoder1, dim=0)
    d2_norm = F.normalize(decoder2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    
    return (max_1to2 + max_2to1) / 2


def compute_expanded_pwmcc(arch: str, seeds: List[int]) -> Dict:
    """Compute PWMCC matrix for all SAEs of given architecture."""
    # Load all decoders
    decoders = {}
    for seed in seeds:
        if arch == 'topk':
            # Try multiple naming conventions
            possible_paths = [
                SAE_DIR / f'topk_seed{seed}' / 'sae.pt',
                SAE_DIR / f'topk_seed{seed}' / 'sae_final.pt',
                SAE_DIR / f'topk_seed{seed}' / 'sae_best.pt',
            ]
        else:
            possible_paths = [
                SAE_DIR / f'relu_seed{seed}' / 'sae.pt',
                SAE_DIR / f'relu_seed{seed}' / 'sae_final.pt',
                SAE_DIR / f'relu_seed{seed}' / 'sae_best.pt',
            ]
        
        sae_path = None
        for p in possible_paths:
            if p.exists():
                sae_path = p
                break
        
        if sae_path is not None:
            checkpoint = torch.load(sae_path, map_location='cpu')
            decoders[seed] = checkpoint['model_state_dict']['decoder.weight']
        else:
            print(f"  Warning: No SAE found for {arch} seed {seed}")
    
    # Compute pairwise PWMCC
    seed_list = sorted(decoders.keys())
    n = len(seed_list)
    pwmcc_matrix = np.ones((n, n))
    pairwise_values = []
    
    for i, seed1 in enumerate(seed_list):
        for j, seed2 in enumerate(seed_list):
            if i < j:
                pwmcc = compute_decoder_pwmcc(decoders[seed1], decoders[seed2])
                pwmcc_matrix[i, j] = pwmcc
                pwmcc_matrix[j, i] = pwmcc
                pairwise_values.append(pwmcc)
    
    pairwise_array = np.array(pairwise_values)
    
    return {
        'architecture': arch,
        'seeds': seed_list,
        'n_saes': n,
        'n_pairs': len(pairwise_values),
        'pwmcc_matrix': pwmcc_matrix.tolist(),
        'stats': {
            'mean': float(pairwise_array.mean()),
            'std': float(pairwise_array.std()),
            'min': float(pairwise_array.min()),
            'max': float(pairwise_array.max()),
            'median': float(np.median(pairwise_array))
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train expanded SAE seeds')
    parser.add_argument('--arch', choices=['topk', 'relu', 'both'], default='both',
                        help='Architecture to train')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of seeds to train')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only run analysis on existing SAEs')
    args = parser.parse_args()
    
    # Parse seeds
    if args.seeds:
        seeds_to_train = [int(s) for s in args.seeds.split(',')]
    else:
        seeds_to_train = NEW_SEEDS
    
    print("=" * 70)
    print("PHASE 3: EXPANDED SEED TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds to train: {seeds_to_train}")
    print(f"Architecture(s): {args.arch}")
    print()
    
    if not args.analyze_only:
        # Load transformer and extract activations
        print("Loading transformer and extracting activations...")
        model = load_transformer()
        activations = extract_activations(model)
        print(f"✓ Extracted {activations.shape[0]} samples × {activations.shape[1]} dims")
        print()
        
        # Train SAEs
        results = []
        
        if args.arch in ['topk', 'both']:
            print("=" * 70)
            print("TRAINING TOPK SAEs")
            print("=" * 70)
            
            for seed in seeds_to_train:
                save_dir = SAE_DIR / f'topk_seed{seed}'
                if (save_dir / 'sae.pt').exists():
                    print(f"Seed {seed}: Already exists, skipping")
                    continue
                
                print(f"\nTraining TopK SAE (seed={seed})...")
                start_time = time.time()
                result = train_topk_sae(activations, seed, save_dir)
                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed/60:.1f} min | Loss={result['best_loss']:.4f}, EV={result['final_ev']:.4f}")
                results.append(result)
        
        if args.arch in ['relu', 'both']:
            print("\n" + "=" * 70)
            print("TRAINING RELU SAEs")
            print("=" * 70)
            
            for seed in seeds_to_train:
                save_dir = SAE_DIR / f'relu_seed{seed}'
                if (save_dir / 'sae.pt').exists():
                    print(f"Seed {seed}: Already exists, skipping")
                    continue
                
                print(f"\nTraining ReLU SAE (seed={seed})...")
                start_time = time.time()
                result = train_relu_sae(activations, seed, save_dir)
                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed/60:.1f} min | Loss={result['best_loss']:.4f}, EV={result['final_ev']:.4f}")
                results.append(result)
        
        # Save training summary
        summary_path = RESULTS_DIR / 'analysis' / 'expanded_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved training summary to {summary_path}")
    
    # Compute expanded PWMCC
    print("\n" + "=" * 70)
    print("COMPUTING EXPANDED PWMCC")
    print("=" * 70)
    
    all_seeds = ORIGINAL_SEEDS + seeds_to_train
    
    if args.arch in ['topk', 'both']:
        print("\nTopK PWMCC analysis...")
        topk_results = compute_expanded_pwmcc('topk', all_seeds)
        print(f"  N SAEs: {topk_results['n_saes']}")
        print(f"  N pairs: {topk_results['n_pairs']}")
        print(f"  PWMCC: {topk_results['stats']['mean']:.4f} ± {topk_results['stats']['std']:.4f}")
        
        # Save
        with open(RESULTS_DIR / 'analysis' / 'expanded_pwmcc_topk.json', 'w') as f:
            json.dump(topk_results, f, indent=2)
    
    if args.arch in ['relu', 'both']:
        print("\nReLU PWMCC analysis...")
        relu_results = compute_expanded_pwmcc('relu', all_seeds)
        print(f"  N SAEs: {relu_results['n_saes']}")
        print(f"  N pairs: {relu_results['n_pairs']}")
        print(f"  PWMCC: {relu_results['stats']['mean']:.4f} ± {relu_results['stats']['std']:.4f}")
        
        # Save
        with open(RESULTS_DIR / 'analysis' / 'expanded_pwmcc_relu.json', 'w') as f:
            json.dump(relu_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ PHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
