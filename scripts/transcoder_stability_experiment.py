#!/usr/bin/env python3
"""Transcoder vs SAE Stability: Testing if transcoders are more stable.

NOVEL RESEARCH QUESTION (based on arXiv:2501.18823):
The "Transcoders Beat Sparse Autoencoders for Interpretability" paper shows that:
- Transcoders reconstruct MLP OUTPUT from MLP INPUT (not just activations)
- This dual-constraint may lead to more stable feature learning
- Transcoders achieve better interpretability scores

We hypothesize that:
- Transcoders → more constrained optimization → HIGHER PWMCC
- SAEs → single reconstruction → LOWER PWMCC
- If transcoders match SAEs → stability is fundamental property

Experimental Design:
1. Implement Transcoder: MLP_input → Sparse latents → MLP_output
2. Train 5 transcoders with different seeds (42, 123, 456, 789, 1011)
3. Use same sparsity (k=32) as SAE baseline
4. Compute PWMCC between transcoder decoder matrices
5. Compare with SAE PWMCC results

Expected Outcome:
- Hypothesis 1: Transcoders MORE stable → PWMCC > SAE PWMCC
- Hypothesis 2: Transcoders SAME stability → fundamental limit
- Hypothesis 3: Transcoders LESS stable → dual constraint hurts

Reference:
- Transformer: /results/transformer_5000ep/transformer_best.pt
- SAE baseline: stability_vs_sparsity experiment (k=32)
- arXiv:2501.18823 (Transcoders Beat SAEs)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/transcoder_stability_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'transcoder_stability'
FIGURES_DIR = BASE_DIR / 'figures'

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cpu'


class Transcoder(nn.Module):
    """Transcoder: Reconstructs MLP output from MLP input via sparse latents.

    Key difference from SAE:
    - SAE: reconstructs x from x
    - Transcoder: reconstructs f(x) from x, where f = MLP

    This dual constraint (input → latents, latents → output) may lead to
    more stable feature learning across random seeds.

    Architecture:
        MLP_input → encoder (W_enc, b_enc) → pre_activation
        → TopK(pre_activation, k) → latents
        → decoder (W_dec) → MLP_output

    Training objective:
        minimize: MSE(predicted_mlp_output, actual_mlp_output)
        subject to: exactly k active latents per sample

    Args:
        d_model: Dimension of residual stream (input/output)
        d_sae: Dimension of sparse latent space
        k: Number of top activations to keep (sparsity)

    Reference:
        arXiv:2501.18823 - Transcoders Beat Sparse Autoencoders
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = min(k, d_sae)

        # Encoder: maps MLP input to latent pre-activations
        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        # Decoder: maps latents to MLP output
        # Following SAE best practices: no bias, unit-normalized columns
        self.decoder = nn.Linear(d_sae, d_model, bias=False)

        # Initialize weights
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (SAE best practice)."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, mlp_input: torch.Tensor) -> torch.Tensor:
        """Encode MLP input to sparse latents.

        Args:
            mlp_input: MLP input activations [batch, d_model]

        Returns:
            latents: Sparse latent representation [batch, d_sae]
        """
        pre_act = self.encoder(mlp_input)

        # TopK activation: keep top-k, zero others
        topk_values, topk_indices = torch.topk(pre_act, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_indices, topk_values)

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to predicted MLP output.

        Args:
            latents: Sparse latents [batch, d_sae]

        Returns:
            mlp_output_pred: Predicted MLP output [batch, d_model]
        """
        return self.decoder(latents)

    def forward(
        self,
        mlp_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: MLP input → sparse latents → predicted MLP output.

        Args:
            mlp_input: Input to MLP layer [batch, d_model]

        Returns:
            mlp_output_pred: Predicted MLP output [batch, d_model]
            latents: Sparse latent activations [batch, d_sae]
        """
        latents = self.encode(mlp_input)
        mlp_output_pred = self.decode(latents)
        return mlp_output_pred, latents

    def get_l0(self, latents: torch.Tensor) -> float:
        """Compute L0 sparsity (average active features)."""
        return (latents != 0).float().sum(dim=-1).mean().item()

    def save(self, path: Path):
        """Save transcoder checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'd_model': self.d_model,
            'd_sae': self.d_sae,
            'k': self.k,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = 'cpu'):
        """Load transcoder checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        transcoder = cls(
            d_model=checkpoint['d_model'],
            d_sae=checkpoint['d_sae'],
            k=checkpoint['k']
        )

        transcoder.load_state_dict(checkpoint['model_state_dict'])
        return transcoder.to(device)


class TopKSAE(nn.Module):
    """TopK SAE for comparison (same as in stability_vs_sparsity)."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = min(k, d_sae)

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_act, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        return latents

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        recon = self.decoder(latents)
        return recon, latents


def compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Compute Pairwise Maximum Cosine Correlation (PWMCC).

    PWMCC measures feature stability by finding best matches between
    two feature sets and averaging cosine similarities.

    Args:
        d1: Decoder weights from model 1 [d_model, d_sae]
        d2: Decoder weights from model 2 [d_model, d_sae]

    Returns:
        pwmcc: Average of bidirectional best-match similarities
    """
    # Normalize to unit vectors
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)

    # Compute all pairwise cosine similarities
    cos_sim = d1_norm.T @ d2_norm  # [d_sae, d_sae]

    # For each feature in d1, find best match in d2
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()

    # For each feature in d2, find best match in d1
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()

    # Average bidirectional matches
    return (max_1to2 + max_2to1) / 2


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 10) -> float:
    """Compute random PWMCC baseline (should be ~1/sqrt(d_model))."""
    pwmcc_values = []
    for _ in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        pwmcc_values.append(compute_pwmcc(d1, d2))
    return np.mean(pwmcc_values)


def load_mlp_activations() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MLP input and output activations from transformer.

    Returns:
        mlp_inputs: Activations before MLP [n_samples, d_model]
        mlp_outputs: Activations after MLP [n_samples, d_model]
    """
    print("\nLoading transformer...")
    model_path = RESULTS_DIR / 'transformer_5000ep' / 'transformer_best.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerConfig(**checkpoint['config'])
    model = ModularArithmeticTransformer(config, device='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Extracting MLP activations...")
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format='sequence')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    mlp_inputs_list = []
    mlp_outputs_list = []

    with torch.no_grad():
        for batch, _ in dataloader:
            # Run with cache to get all activations
            _, cache = model.model.run_with_cache(batch)

            # For layer 1:
            # - MLP input = residual stream BEFORE MLP (after attention)
            # - MLP output = MLP's contribution to residual stream
            layer = 1

            # Get residual stream before MLP (= after attention + layernorm)
            mlp_in = cache[f'blocks.{layer}.ln2.hook_normalized'][:, -2, :]

            # Get MLP output (what MLP adds to residual stream)
            mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][:, -2, :]

            mlp_inputs_list.append(mlp_in)
            mlp_outputs_list.append(mlp_out)

    mlp_inputs = torch.cat(mlp_inputs_list, dim=0)
    mlp_outputs = torch.cat(mlp_outputs_list, dim=0)

    print(f"✓ Loaded {len(mlp_inputs)} samples")
    print(f"  MLP input shape: {mlp_inputs.shape}")
    print(f"  MLP output shape: {mlp_outputs.shape}")

    return mlp_inputs, mlp_outputs


def train_transcoder(
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 256
) -> Tuple[Transcoder, float]:
    """Train a transcoder on MLP input/output pairs.

    Args:
        mlp_inputs: MLP input activations [n_samples, d_model]
        mlp_outputs: MLP output activations [n_samples, d_model]
        d_sae: Latent dimension
        k: Sparsity level
        seed: Random seed
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size

    Returns:
        transcoder: Trained Transcoder model
        final_loss: Final reconstruction loss
    """
    torch.manual_seed(seed)

    d_model = mlp_inputs.shape[1]
    transcoder = Transcoder(d_model=d_model, d_sae=d_sae, k=k)
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=lr)

    # Create dataset of (input, output) pairs
    dataset = TensorDataset(mlp_inputs, mlp_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for mlp_in, mlp_out in dataloader:
            # Forward: predict MLP output from MLP input
            mlp_out_pred, latents = transcoder(mlp_in)

            # Loss: MSE between predicted and actual MLP output
            loss = F.mse_loss(mlp_out_pred, mlp_out)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder (SAE best practice)
            transcoder.normalize_decoder()

            epoch_loss += loss.item()

        final_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"    Seed {seed}, Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

    return transcoder, final_loss


def train_sae_baseline(
    mlp_inputs: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 256
) -> Tuple[TopKSAE, float]:
    """Train baseline SAE (reconstructs MLP input from itself)."""
    torch.manual_seed(seed)

    d_model = mlp_inputs.shape[1]
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(mlp_inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for (x,) in dataloader:
            recon, latents = sae(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()

        final_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"    Seed {seed}, Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

    return sae, final_loss


def run_transcoder_experiment(
    mlp_inputs: torch.Tensor,
    mlp_outputs: torch.Tensor,
    d_sae: int,
    k: int,
    seeds: List[int]
) -> Dict:
    """Run transcoder stability experiment across multiple seeds.

    Args:
        mlp_inputs: MLP input activations
        mlp_outputs: MLP output activations
        d_sae: Latent dimension
        k: Sparsity level
        seeds: Random seeds for training

    Returns:
        Dictionary with PWMCC statistics and trained models
    """
    print(f"\n{'='*70}")
    print(f"TRANSCODER EXPERIMENT: d_sae={d_sae}, k={k}")
    print(f"{'='*70}")

    # Train transcoders
    transcoders = []
    losses = []

    print(f"\nTraining {len(seeds)} transcoders...")
    for seed in seeds:
        print(f"\n  Training transcoder with seed={seed}")
        transcoder, loss = train_transcoder(
            mlp_inputs, mlp_outputs, d_sae, k, seed
        )
        transcoders.append(transcoder)
        losses.append(loss)

        # Save checkpoint
        save_path = OUTPUT_DIR / f'transcoder_seed{seed}.pt'
        transcoder.save(save_path)

    # Compute pairwise PWMCC
    print("\nComputing PWMCC...")
    pwmcc_values = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pwmcc = compute_pwmcc(
                transcoders[i].decoder.weight.data,
                transcoders[j].decoder.weight.data
            )
            pwmcc_values.append(pwmcc)
            print(f"  PWMCC (seed {seeds[i]} vs {seeds[j]}): {pwmcc:.4f}")

    # Random baseline
    random_baseline = compute_random_baseline(128, d_sae)

    results = {
        'd_sae': d_sae,
        'k': k,
        'pwmcc_mean': np.mean(pwmcc_values),
        'pwmcc_std': np.std(pwmcc_values),
        'pwmcc_all': pwmcc_values,
        'random_baseline': random_baseline,
        'ratio': np.mean(pwmcc_values) / random_baseline,
        'recon_loss_mean': np.mean(losses),
        'recon_loss_std': np.std(losses),
    }

    print(f"\n{'='*70}")
    print(f"TRANSCODER RESULTS:")
    print(f"  PWMCC: {results['pwmcc_mean']:.4f} ± {results['pwmcc_std']:.4f}")
    print(f"  Random baseline: {results['random_baseline']:.4f}")
    print(f"  Ratio: {results['ratio']:.2f}×")
    print(f"  Reconstruction loss: {results['recon_loss_mean']:.6f}")
    print(f"{'='*70}")

    return results


def run_sae_baseline_experiment(
    mlp_inputs: torch.Tensor,
    d_sae: int,
    k: int,
    seeds: List[int]
) -> Dict:
    """Run baseline SAE stability experiment for comparison."""
    print(f"\n{'='*70}")
    print(f"BASELINE SAE EXPERIMENT: d_sae={d_sae}, k={k}")
    print(f"{'='*70}")

    # Train SAEs
    saes = []
    losses = []

    print(f"\nTraining {len(seeds)} SAEs...")
    for seed in seeds:
        print(f"\n  Training SAE with seed={seed}")
        sae, loss = train_sae_baseline(mlp_inputs, d_sae, k, seed)
        saes.append(sae)
        losses.append(loss)

        # Save checkpoint
        save_path = OUTPUT_DIR / f'sae_seed{seed}.pt'
        torch.save({
            'model_state_dict': sae.state_dict(),
            'd_model': sae.d_model,
            'd_sae': sae.d_sae,
            'k': sae.k,
        }, save_path)

    # Compute pairwise PWMCC
    print("\nComputing PWMCC...")
    pwmcc_values = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pwmcc = compute_pwmcc(
                saes[i].decoder.weight.data,
                saes[j].decoder.weight.data
            )
            pwmcc_values.append(pwmcc)
            print(f"  PWMCC (seed {seeds[i]} vs {seeds[j]}): {pwmcc:.4f}")

    # Random baseline
    random_baseline = compute_random_baseline(128, d_sae)

    results = {
        'd_sae': d_sae,
        'k': k,
        'pwmcc_mean': np.mean(pwmcc_values),
        'pwmcc_std': np.std(pwmcc_values),
        'pwmcc_all': pwmcc_values,
        'random_baseline': random_baseline,
        'ratio': np.mean(pwmcc_values) / random_baseline,
        'recon_loss_mean': np.mean(losses),
        'recon_loss_std': np.std(losses),
    }

    print(f"\n{'='*70}")
    print(f"SAE BASELINE RESULTS:")
    print(f"  PWMCC: {results['pwmcc_mean']:.4f} ± {results['pwmcc_std']:.4f}")
    print(f"  Random baseline: {results['random_baseline']:.4f}")
    print(f"  Ratio: {results['ratio']:.2f}×")
    print(f"  Reconstruction loss: {results['recon_loss_mean']:.6f}")
    print(f"{'='*70}")

    return results


def visualize_results(
    transcoder_results: Dict,
    sae_results: Dict,
    output_path: Path
):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: PWMCC comparison
    ax1 = axes[0]
    models = ['Transcoder', 'SAE']
    pwmccs = [
        transcoder_results['pwmcc_mean'],
        sae_results['pwmcc_mean']
    ]
    stds = [
        transcoder_results['pwmcc_std'],
        sae_results['pwmcc_std']
    ]
    baseline = transcoder_results['random_baseline']

    bars = ax1.bar(models, pwmccs, yerr=stds, capsize=10,
                   color=['steelblue', 'coral'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, label='Random Baseline')
    ax1.set_ylabel('PWMCC', fontsize=14, fontweight='bold')
    ax1.set_title('Feature Stability Comparison', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, pwmcc, std in zip(bars, pwmccs, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{pwmcc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel 2: Ratio to random
    ax2 = axes[1]
    ratios = [
        transcoder_results['ratio'],
        sae_results['ratio']
    ]

    bars2 = ax2.bar(models, ratios, color=['steelblue', 'coral'],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Random baseline')
    ax2.set_ylabel('PWMCC / Random Baseline', fontsize=14, fontweight='bold')
    ax2.set_title('Stability Improvement', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add ratio labels
    for bar, ratio in zip(bars2, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ratio:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel 3: Distribution of pairwise PWMCC
    ax3 = axes[2]
    trans_pwmccs = transcoder_results['pwmcc_all']
    sae_pwmccs = sae_results['pwmcc_all']

    positions = [1, 2]
    bp = ax3.boxplot([trans_pwmccs, sae_pwmccs], positions=positions,
                      widths=0.6, patch_artist=True,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

    # Color boxes
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.8)

    ax3.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, label='Random')
    ax3.set_xticks(positions)
    ax3.set_xticklabels(models, fontsize=12)
    ax3.set_ylabel('PWMCC', fontsize=14, fontweight='bold')
    ax3.set_title('PWMCC Distribution (All Pairs)', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")


def main():
    print("=" * 70)
    print("TRANSCODER VS SAE STABILITY EXPERIMENT")
    print("Testing if transcoders are more stable than SAEs")
    print("=" * 70)

    # Experimental parameters
    d_sae = 128  # Same as effective rank
    k = 32       # Same as our SAE experiments
    seeds = [42, 123, 456, 789, 1011]  # 5 seeds for statistics

    print(f"\nExperimental setup:")
    print(f"  d_sae: {d_sae}")
    print(f"  k: {k}")
    print(f"  seeds: {seeds}")

    # Load MLP activations
    mlp_inputs, mlp_outputs = load_mlp_activations()

    # Run transcoder experiment
    transcoder_results = run_transcoder_experiment(
        mlp_inputs, mlp_outputs, d_sae, k, seeds
    )

    # Run SAE baseline experiment
    sae_results = run_sae_baseline_experiment(
        mlp_inputs, d_sae, k, seeds
    )

    # Visualize comparison
    fig_path = FIGURES_DIR / 'transcoder_vs_sae_stability.pdf'
    visualize_results(transcoder_results, sae_results, fig_path)

    # Save results
    results = {
        'experimental_setup': {
            'd_sae': d_sae,
            'k': k,
            'seeds': seeds,
            'n_samples': len(mlp_inputs),
        },
        'transcoder': {
            'pwmcc_mean': float(transcoder_results['pwmcc_mean']),
            'pwmcc_std': float(transcoder_results['pwmcc_std']),
            'pwmcc_all': [float(x) for x in transcoder_results['pwmcc_all']],
            'random_baseline': float(transcoder_results['random_baseline']),
            'ratio': float(transcoder_results['ratio']),
            'recon_loss_mean': float(transcoder_results['recon_loss_mean']),
            'recon_loss_std': float(transcoder_results['recon_loss_std']),
        },
        'sae': {
            'pwmcc_mean': float(sae_results['pwmcc_mean']),
            'pwmcc_std': float(sae_results['pwmcc_std']),
            'pwmcc_all': [float(x) for x in sae_results['pwmcc_all']],
            'random_baseline': float(sae_results['random_baseline']),
            'ratio': float(sae_results['ratio']),
            'recon_loss_mean': float(sae_results['recon_loss_mean']),
            'recon_loss_std': float(sae_results['recon_loss_std']),
        }
    }

    output_path = OUTPUT_DIR / 'transcoder_vs_sae_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {output_path}")

    # Summary and interpretation
    print("\n" + "=" * 70)
    print("SUMMARY: TRANSCODER VS SAE STABILITY")
    print("=" * 70)
    print()

    print("RESULTS:")
    print("-" * 70)
    print(f"Transcoder PWMCC: {transcoder_results['pwmcc_mean']:.4f} ± {transcoder_results['pwmcc_std']:.4f}")
    print(f"SAE PWMCC:        {sae_results['pwmcc_mean']:.4f} ± {sae_results['pwmcc_std']:.4f}")
    print(f"Random baseline:  {transcoder_results['random_baseline']:.4f}")
    print()

    # Statistical comparison
    pwmcc_diff = transcoder_results['pwmcc_mean'] - sae_results['pwmcc_mean']
    pwmcc_diff_pct = 100 * pwmcc_diff / sae_results['pwmcc_mean']

    print("COMPARISON:")
    print("-" * 70)
    print(f"Difference: {pwmcc_diff:+.4f} ({pwmcc_diff_pct:+.1f}%)")
    print(f"Transcoder ratio: {transcoder_results['ratio']:.2f}×")
    print(f"SAE ratio:        {sae_results['ratio']:.2f}×")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print()

    # Threshold for "significantly more stable" (arbitrary: >5% improvement)
    if pwmcc_diff > 0.05:
        print("✓ HYPOTHESIS 1 SUPPORTED: Transcoders are MORE STABLE")
        print()
        print("  Key insights:")
        print("  - Dual constraint (input→output) stabilizes feature learning")
        print("  - Transcoders find more consistent decomposition across seeds")
        print("  - This supports arXiv:2501.18823's interpretability claims")
        print()
        print("  Implications:")
        print("  - Transcoders may be preferred for interpretability work")
        print("  - Stability correlates with task-relevant decomposition")
        print("  - MLP output constraint guides features to meaningful directions")

    elif pwmcc_diff < -0.05:
        print("✗ HYPOTHESIS 3: Transcoders are LESS STABLE")
        print()
        print("  Key insights:")
        print("  - Dual constraint may overfit to specific MLP behavior")
        print("  - SAE's self-reconstruction may be more robust")
        print("  - Needs further investigation")

    else:
        print("→ HYPOTHESIS 2: Transcoders have SAME STABILITY as SAEs")
        print()
        print("  Key insights:")
        print("  - Stability appears to be a fundamental property")
        print("  - Both methods find similarly stable decompositions")
        print("  - Transcoder's interpretability gains (arXiv:2501.18823)")
        print("    may come from different feature directions, not stability")
        print()
        print("  Implications:")
        print("  - Stability is intrinsic to the sparse coding problem")
        print("  - Choice between SAE/Transcoder should be based on")
        print("    interpretability quality, not stability concerns")

    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print()
    print("1. Analyze feature interpretability (do transcoders find better features?)")
    print("2. Compare feature activation patterns on test data")
    print("3. Test on different layers and transformer architectures")
    print("4. Investigate relationship between stability and interpretability")
    print()


if __name__ == '__main__':
    main()
