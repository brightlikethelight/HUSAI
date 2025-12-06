#!/usr/bin/env python3
"""Stability-aware SAE training based on Song et al. (2025).

This script implements multi-seed consistency training to improve SAE stability.
The key idea: train paired SAEs with different seeds but encourage them to learn
similar features via a consistency loss term.

Research question:
    Can we improve PWMCC from baseline 0.30 to higher values using stability-aware training?

Method:
    1. Train two SAEs simultaneously (seeds 42 and 123)
    2. Add consistency loss: encourage similar decoder directions
    3. Consistency metric: max cosine similarity between decoder columns
    4. Test different consistency loss weights (λ)

Expected outcome:
    If Song et al.'s approach works, PWMCC should increase above 0.30 baseline.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/stability_aware_sae.py --lambda 0.1

Reference:
    Song et al. (2025): "Stabilizing Sparse Autoencoders via Multi-seed Consistency"
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.simple_sae import TopKSAE


def compute_pwmcc(decoder1, decoder2):
    """Compute Pairwise Max Cosine Correlation (PWMCC) between two SAE decoders.

    This is the key stability metric from our research.
    PWMCC = 0.30 baseline (trained SAEs are as similar as random SAEs)

    Args:
        decoder1: [d_model, d_sae] decoder weight matrix
        decoder2: [d_model, d_sae] decoder weight matrix

    Returns:
        pwmcc: Scalar similarity score
    """
    # Normalize decoder columns
    decoder1 = F.normalize(decoder1, dim=0)  # [d_model, d_sae]
    decoder2 = F.normalize(decoder2, dim=0)  # [d_model, d_sae]

    # Compute all pairwise cosine similarities
    similarity = torch.mm(decoder1.T, decoder2)  # [d_sae, d_sae]

    # Max over each row (for decoder1 features, find best match in decoder2)
    max_sim_1to2 = similarity.max(dim=1)[0].mean()  # Average best match

    # Max over each column (for decoder2 features, find best match in decoder1)
    max_sim_2to1 = similarity.max(dim=0)[0].mean()  # Average best match

    # PWMCC is average of both directions
    pwmcc = (max_sim_1to2 + max_sim_2to1) / 2

    return pwmcc.item()


def compute_consistency_loss(sae1, sae2):
    """Compute consistency loss between two SAE decoders.

    Following Song et al.: encourage decoder columns to align across seeds.
    We use max cosine similarity as the consistency metric.

    Args:
        sae1: First SAE (seed 42)
        sae2: Second SAE (seed 123)

    Returns:
        loss: Consistency loss (negative PWMCC, so minimizing increases alignment)
    """
    decoder1 = sae1.decoder.weight.data  # [d_model, d_sae]
    decoder2 = sae2.decoder.weight.data  # [d_model, d_sae]

    # Normalize
    decoder1 = F.normalize(decoder1, dim=0)
    decoder2 = F.normalize(decoder2, dim=0)

    # Cosine similarity matrix
    similarity = torch.mm(decoder1.T, decoder2)  # [d_sae, d_sae]

    # Max similarity for each feature in sae1
    max_sim_1to2 = similarity.max(dim=1)[0]  # [d_sae]

    # Max similarity for each feature in sae2
    max_sim_2to1 = similarity.max(dim=0)[0]  # [d_sae]

    # Consistency loss: negative average max similarity
    # (minimizing this loss = maximizing similarity)
    loss = -(max_sim_1to2.mean() + max_sim_2to1.mean()) / 2

    return loss


def train_paired_saes(
    activations,
    d_model,
    d_sae,
    k,
    lambda_consistency,
    epochs=20,
    batch_size=256,
    lr=3e-4,
    seed1=42,
    seed2=123
):
    """Train two SAEs simultaneously with consistency loss.

    Args:
        activations: [n_samples, d_model] training data
        d_model: Input dimension
        d_sae: SAE hidden dimension
        k: TopK parameter
        lambda_consistency: Weight for consistency loss
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seed1: First random seed
        seed2: Second random seed

    Returns:
        sae1, sae2: Trained SAE pair
        final_pwmcc: Final PWMCC between the pair
    """
    print(f"\n{'='*70}")
    print(f"Training paired SAEs with consistency loss (λ={lambda_consistency})")
    print(f"{'='*70}")

    # Create two SAEs with different seeds
    torch.manual_seed(seed1)
    sae1 = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    torch.manual_seed(seed2)
    sae2 = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    # Separate optimizers for each SAE
    optimizer1 = torch.optim.Adam(sae1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(sae2.parameters(), lr=lr)

    # Dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        sae1.train()
        sae2.train()
        sae1.reset_feature_counts()
        sae2.reset_feature_counts()

        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_consistency = 0
        epoch_mse1 = 0
        epoch_mse2 = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for (batch,) in pbar:
            # Forward pass for both SAEs
            recon1, latents1, aux_loss1 = sae1(batch)
            recon2, latents2, aux_loss2 = sae2(batch)

            # Reconstruction losses
            mse_loss1 = F.mse_loss(recon1, batch)
            mse_loss2 = F.mse_loss(recon2, batch)

            # Base losses (reconstruction + sparsity)
            base_loss1 = mse_loss1 + aux_loss1
            base_loss2 = mse_loss2 + aux_loss2

            # Consistency loss (encourages similar features)
            if lambda_consistency > 0:
                consistency_loss = compute_consistency_loss(sae1, sae2)
            else:
                consistency_loss = torch.tensor(0.0)

            # Total losses
            loss1 = base_loss1 + lambda_consistency * consistency_loss
            loss2 = base_loss2 + lambda_consistency * consistency_loss

            # Backward pass for SAE 1
            optimizer1.zero_grad()
            loss1.backward(retain_graph=(lambda_consistency > 0))
            optimizer1.step()
            sae1.normalize_decoder()

            # Backward pass for SAE 2
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            sae2.normalize_decoder()

            # Metrics
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_consistency += consistency_loss.item()
            epoch_mse1 += mse_loss1.item()
            epoch_mse2 += mse_loss2.item()
            n_batches += 1

            pbar.set_postfix({
                'loss1': f'{loss1.item():.4f}',
                'loss2': f'{loss2.item():.4f}',
                'cons': f'{consistency_loss.item():.4f}'
            })

        # Epoch metrics
        avg_loss1 = epoch_loss1 / n_batches
        avg_loss2 = epoch_loss2 / n_batches
        avg_consistency = epoch_consistency / n_batches
        avg_mse1 = epoch_mse1 / n_batches
        avg_mse2 = epoch_mse2 / n_batches

        # Compute current PWMCC
        with torch.no_grad():
            current_pwmcc = compute_pwmcc(
                sae1.decoder.weight.data,
                sae2.decoder.weight.data
            )

        print(f"\n Epoch {epoch+1}/{epochs}:")
        print(f"   SAE1 - Loss: {avg_loss1:.4f}, MSE: {avg_mse1:.4f}")
        print(f"   SAE2 - Loss: {avg_loss2:.4f}, MSE: {avg_mse2:.4f}")
        print(f"   Consistency loss: {avg_consistency:.4f}")
        print(f"   Current PWMCC: {current_pwmcc:.4f}")

    # Final PWMCC
    with torch.no_grad():
        final_pwmcc = compute_pwmcc(
            sae1.decoder.weight.data,
            sae2.decoder.weight.data
        )

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final PWMCC: {final_pwmcc:.4f}")
    print(f"{'='*70}\n")

    return sae1, sae2, final_pwmcc


def evaluate_reconstruction_quality(sae, activations):
    """Evaluate SAE reconstruction quality.

    Args:
        sae: Trained SAE
        activations: Test data

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    sae.eval()

    with torch.no_grad():
        recon, latents, _ = sae(activations)

        # MSE
        mse = F.mse_loss(recon, activations).item()

        # Explained variance
        data_var = activations.var().item()
        error_var = (activations - recon).var().item()
        explained_var = 1 - (error_var / data_var)

        # L0 sparsity
        l0 = sae.get_l0(latents)

        # Dead neurons
        n_dead = len(sae.get_dead_neurons())
        dead_pct = n_dead / sae.d_sae * 100

    return {
        'mse': mse,
        'explained_variance': explained_var,
        'l0': l0,
        'dead_neurons': n_dead,
        'dead_pct': dead_pct
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stability-aware SAE training (Song et al. 2025)"
    )

    # Lambda values to test
    parser.add_argument(
        '--lambda',
        type=float,
        dest='lambda_consistency',
        default=0.1,
        help='Consistency loss weight (default: 0.1)'
    )
    parser.add_argument(
        '--test-all',
        action='store_true',
        help='Test all λ values: [0.0, 0.01, 0.1, 1.0]'
    )

    # Model architecture
    parser.add_argument(
        '--expansion',
        type=int,
        default=8,
        help='Expansion factor (d_sae = expansion * d_model)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=32,
        help='TopK k parameter'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )

    # Data
    parser.add_argument(
        '--activations',
        type=Path,
        default=Path('results/training_dynamics/activations_layer0.pt'),
        help='Path to activation data'
    )

    args = parser.parse_args()

    # Load activations
    print(f"Loading activations from {args.activations}...")
    activations = torch.load(args.activations)
    print(f"✅ Loaded activations: {activations.shape}")

    # Get dimensions
    n_samples, d_model = activations.shape
    d_sae = args.expansion * d_model

    print(f"\nExperiment setup:")
    print(f"  Samples: {n_samples}")
    print(f"  d_model: {d_model}")
    print(f"  d_sae: {d_sae} ({args.expansion}x expansion)")
    print(f"  k: {args.k}")
    print(f"  Epochs: {args.epochs}")

    # Lambda values to test
    if args.test_all:
        lambda_values = [0.0, 0.01, 0.1, 1.0]
    else:
        lambda_values = [args.lambda_consistency]

    print(f"\nTesting λ values: {lambda_values}")

    # Results storage
    results = []

    # Baseline: Train two independent SAEs (λ=0) if testing all
    if args.test_all:
        print("\n" + "="*80)
        print("BASELINE: Independent training (λ=0.0)")
        print("="*80)

        # Train SAE 1 (seed 42)
        torch.manual_seed(42)
        sae_baseline1 = TopKSAE(d_model=d_model, d_sae=d_sae, k=args.k)
        optimizer1 = torch.optim.Adam(sae_baseline1.parameters(), lr=args.lr)

        # Train SAE 2 (seed 123)
        torch.manual_seed(123)
        sae_baseline2 = TopKSAE(d_model=d_model, d_sae=d_sae, k=args.k)
        optimizer2 = torch.optim.Adam(sae_baseline2.parameters(), lr=args.lr)

        # Simple independent training
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(args.epochs):
            sae_baseline1.train()
            sae_baseline2.train()

            for (batch,) in tqdm(dataloader, desc=f"Baseline Epoch {epoch+1}/{args.epochs}"):
                # Train SAE 1
                optimizer1.zero_grad()
                recon1, _, aux1 = sae_baseline1(batch)
                loss1 = F.mse_loss(recon1, batch) + aux1
                loss1.backward()
                optimizer1.step()
                sae_baseline1.normalize_decoder()

                # Train SAE 2
                optimizer2.zero_grad()
                recon2, _, aux2 = sae_baseline2(batch)
                loss2 = F.mse_loss(recon2, batch) + aux2
                loss2.backward()
                optimizer2.step()
                sae_baseline2.normalize_decoder()

        # Compute baseline PWMCC
        baseline_pwmcc = compute_pwmcc(
            sae_baseline1.decoder.weight.data,
            sae_baseline2.decoder.weight.data
        )

        print(f"\n🎯 BASELINE PWMCC: {baseline_pwmcc:.4f}")

        # Evaluate quality
        metrics1 = evaluate_reconstruction_quality(sae_baseline1, activations)
        metrics2 = evaluate_reconstruction_quality(sae_baseline2, activations)

        results.append({
            'lambda': 0.0,
            'pwmcc': baseline_pwmcc,
            'metrics1': metrics1,
            'metrics2': metrics2
        })

        print(f"   SAE1 - MSE: {metrics1['mse']:.4f}, ExplVar: {metrics1['explained_variance']:.4f}, L0: {metrics1['l0']:.1f}")
        print(f"   SAE2 - MSE: {metrics2['mse']:.4f}, ExplVar: {metrics2['explained_variance']:.4f}, L0: {metrics2['l0']:.1f}")

    # Test stability-aware training with different λ values
    for lambda_val in lambda_values:
        if lambda_val == 0.0 and args.test_all:
            continue  # Already computed baseline

        print(f"\n{'='*80}")
        print(f"Testing λ = {lambda_val}")
        print(f"{'='*80}")

        # Train paired SAEs
        sae1, sae2, final_pwmcc = train_paired_saes(
            activations=activations,
            d_model=d_model,
            d_sae=d_sae,
            k=args.k,
            lambda_consistency=lambda_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        # Evaluate quality
        metrics1 = evaluate_reconstruction_quality(sae1, activations)
        metrics2 = evaluate_reconstruction_quality(sae2, activations)

        results.append({
            'lambda': lambda_val,
            'pwmcc': final_pwmcc,
            'metrics1': metrics1,
            'metrics2': metrics2
        })

        print(f"📊 Results for λ = {lambda_val}:")
        print(f"   PWMCC: {final_pwmcc:.4f}")
        print(f"   SAE1 - MSE: {metrics1['mse']:.4f}, ExplVar: {metrics1['explained_variance']:.4f}, L0: {metrics1['l0']:.1f}")
        print(f"   SAE2 - MSE: {metrics2['mse']:.4f}, ExplVar: {metrics2['explained_variance']:.4f}, L0: {metrics2['l0']:.1f}")

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS: Stability-Aware SAE Training")
    print("="*80)
    print(f"\n{'λ':<10} {'PWMCC':<10} {'Δ from baseline':<20} {'Avg MSE':<10} {'Avg ExplVar':<12}")
    print("-" * 80)

    baseline_pwmcc = results[0]['pwmcc'] if args.test_all else None

    for result in results:
        lambda_val = result['lambda']
        pwmcc = result['pwmcc']
        avg_mse = (result['metrics1']['mse'] + result['metrics2']['mse']) / 2
        avg_explvar = (result['metrics1']['explained_variance'] +
                      result['metrics2']['explained_variance']) / 2

        if baseline_pwmcc is not None and lambda_val > 0:
            delta = pwmcc - baseline_pwmcc
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            pct_change = (delta / baseline_pwmcc) * 100
            delta_display = f"{delta_str} ({pct_change:+.1f}%)"
        else:
            delta_display = "baseline" if lambda_val == 0 else "N/A"

        print(f"{lambda_val:<10.2f} {pwmcc:<10.4f} {delta_display:<20} {avg_mse:<10.4f} {avg_explvar:<12.4f}")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if args.test_all:
        baseline = results[0]['pwmcc']
        best_result = max(results[1:], key=lambda x: x['pwmcc'])
        best_lambda = best_result['lambda']
        best_pwmcc = best_result['pwmcc']
        improvement = best_pwmcc - baseline
        pct_improvement = (improvement / baseline) * 100

        print(f"\nBaseline PWMCC (λ=0.0): {baseline:.4f}")
        print(f"Best PWMCC (λ={best_lambda}): {best_pwmcc:.4f}")
        print(f"Improvement: +{improvement:.4f} ({pct_improvement:+.1f}%)")

        if improvement > 0.05:  # >5 percentage point improvement
            print("\n✅ SUCCESS! Stability-aware training significantly improves PWMCC.")
            print("   Song et al.'s consistency loss approach is effective.")
        elif improvement > 0.01:  # >1 percentage point
            print("\n✓ MODERATE SUCCESS. Consistency loss provides small improvement.")
            print("  Further tuning may yield better results.")
        else:
            print("\n❌ NEGATIVE RESULT. Consistency loss does not improve stability.")
            print("   This suggests SAE instability has deeper causes than Song et al. addressed.")
    else:
        pwmcc = results[0]['pwmcc']
        lambda_val = results[0]['lambda']

        print(f"\nPWMCC with λ={lambda_val}: {pwmcc:.4f}")

        if pwmcc > 0.35:
            print("   This is HIGHER than baseline 0.30 - promising!")
        elif pwmcc > 0.25:
            print("   Similar to baseline 0.30 - consistency loss had limited effect.")
        else:
            print("   This is LOWER than baseline 0.30 - consistency loss may have hurt.")

        print("\n💡 Run with --test-all to compare multiple λ values comprehensively.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
