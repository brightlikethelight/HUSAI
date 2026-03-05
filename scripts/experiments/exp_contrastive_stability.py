#!/usr/bin/env python3
"""Experiment 1.3: Stability-Promoting Contrastive Training.

Implements a multi-seed contrastive loss that encourages decoder column
alignment between simultaneously trained SAEs. Based on Song et al. (2025).

The loss is:
    L = MSE(x, x_hat) + lambda * alignment_penalty(decoder_A, decoder_B)

where alignment_penalty encourages each column in decoder_A to have a
high cosine similarity match in decoder_B (and vice versa).

This closes the paper's narrative loop: "here's the problem, here's why
it exists, and here's evidence it's solvable."

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_contrastive_stability.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from src.models.simple_sae import TopKSAE

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "contrastive_stability"
FIGURES_DIR = PROJECT_ROOT / "figures"
DEVICE = "cpu"
SEEDS = [42, 123, 456, 789, 1011]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_alignment_loss(decoder_a: nn.Linear, decoder_b: nn.Linear) -> torch.Tensor:
    """Compute alignment penalty between two decoder weight matrices.

    Encourages each column in decoder_a to have a high cosine-similarity
    match in decoder_b (and vice versa). This is a differentiable
    proxy for PWMCC.

    Returns:
        loss: scalar, lower means more aligned. Range [0, 2].
    """
    # decoder.weight: [d_model, d_sae]
    w_a = F.normalize(decoder_a.weight, dim=0)  # [d_model, d_sae]
    w_b = F.normalize(decoder_b.weight, dim=0)

    # Cosine similarity matrix: [d_sae_a, d_sae_b]
    cos_sim = w_a.T @ w_b

    # For each feature in A, find max similarity to B
    max_a_to_b = cos_sim.abs().max(dim=1)[0]  # [d_sae_a]
    # For each feature in B, find max similarity to A
    max_b_to_a = cos_sim.abs().max(dim=0)[0]  # [d_sae_b]

    # Alignment loss: 1 - mean_max_similarity (lower = more aligned)
    loss = 2.0 - max_a_to_b.mean() - max_b_to_a.mean()
    return loss


def load_activations() -> torch.Tensor:
    model_path = PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = TransformerConfig(**checkpoint["config"])
    model = ModularArithmeticTransformer(config, device=DEVICE)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    acts = []
    with torch.no_grad():
        for batch, _ in dataloader:
            a = model.get_activations(batch, layer=1)[:, -2, :]
            acts.append(a)
    return torch.cat(acts, dim=0)


def train_sae_pair_contrastive(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed_a: int,
    seed_b: int,
    lambda_align: float,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 256,
) -> Tuple[TopKSAE, TopKSAE, Dict]:
    """Train a pair of SAEs jointly with contrastive alignment loss.

    Both SAEs see the same data batches. Each computes its own
    reconstruction loss, plus a shared alignment penalty.
    """
    d_model = activations.shape[1]

    torch.manual_seed(seed_a)
    sae_a = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)

    torch.manual_seed(seed_b)
    sae_b = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)

    params = list(sae_a.parameters()) + list(sae_b.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"mse_a": [], "mse_b": [], "align_loss": [], "total_loss": []}

    for epoch in range(epochs):
        epoch_mse_a = 0.0
        epoch_mse_b = 0.0
        epoch_align = 0.0
        epoch_total = 0.0
        n_batches = 0

        for (batch,) in dataloader:
            batch = batch.to(DEVICE)

            # Forward both SAEs
            recon_a, latents_a, aux_a = sae_a(batch)
            recon_b, latents_b, aux_b = sae_b(batch)

            mse_a = F.mse_loss(recon_a, batch)
            mse_b = F.mse_loss(recon_b, batch)

            # Alignment loss
            align_loss = compute_alignment_loss(sae_a.decoder, sae_b.decoder)

            total = mse_a + mse_b + aux_a + aux_b + lambda_align * align_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            sae_a.normalize_decoder()
            sae_b.normalize_decoder()

            epoch_mse_a += mse_a.item()
            epoch_mse_b += mse_b.item()
            epoch_align += align_loss.item()
            epoch_total += total.item()
            n_batches += 1

        history["mse_a"].append(epoch_mse_a / n_batches)
        history["mse_b"].append(epoch_mse_b / n_batches)
        history["align_loss"].append(epoch_align / n_batches)
        history["total_loss"].append(epoch_total / n_batches)

    return sae_a, sae_b, history


def train_sae_standard(
    activations: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30,
) -> TopKSAE:
    """Train a standard (non-contrastive) SAE."""
    torch.manual_seed(seed)
    d_model = activations.shape[1]
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for _ in range(epochs):
        for (batch,) in dataloader:
            recon, latents, aux_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

    return sae


def main():
    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1.3: Contrastive Stability-Promoting Training")
    print("=" * 70)
    print(f"Output: {run_dir}")

    manifest = {
        "experiment": "contrastive_stability",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": DEVICE,
    }

    # ── Load activations ──
    print("\nLoading activations...")
    acts = load_activations()
    print(f"  Shape: {acts.shape}")

    d_sae = 1024
    k = 32
    epochs = 30

    # ── Baseline: standard training ──
    print(f"\nBaseline: Training {len(SEEDS)} standard SAEs...")
    standard_saes = [train_sae_standard(acts, d_sae, k, s, epochs) for s in SEEDS]

    standard_pwmcc = []
    for i in range(len(standard_saes)):
        for j in range(i + 1, len(standard_saes)):
            p = compute_pwmcc(
                standard_saes[i].decoder.weight.data,
                standard_saes[j].decoder.weight.data,
            )
            standard_pwmcc.append(p)
    print(f"  Standard PWMCC: {np.mean(standard_pwmcc):.4f} +/- {np.std(standard_pwmcc):.4f}")

    # ── Contrastive training at various lambda values ──
    lambda_values = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
    contrastive_results = []

    for lam in lambda_values:
        print(f"\nContrastive lambda={lam}:")

        # Train pairs of SAEs contrastively
        # Use seed pairs from SEEDS
        pair_saes = []
        pair_histories = []

        for i in range(0, len(SEEDS) - 1, 2):
            sa, sb = SEEDS[i], SEEDS[i + 1]
            print(f"  Training pair (seed {sa}, seed {sb})...")
            sae_a, sae_b, hist = train_sae_pair_contrastive(
                acts, d_sae, k, sa, sb, lambda_align=lam, epochs=epochs,
            )
            pair_saes.extend([sae_a, sae_b])
            pair_histories.append(hist)

        # Also train a solo SAE with last seed for odd-count
        if len(SEEDS) % 2 == 1:
            last_seed = SEEDS[-1]
            print(f"  Training solo (seed {last_seed})...")
            solo = train_sae_standard(acts, d_sae, k, last_seed, epochs)
            pair_saes.append(solo)

        # Compute pairwise PWMCC among all contrastive SAEs
        c_pwmcc = []
        for i in range(len(pair_saes)):
            for j in range(i + 1, len(pair_saes)):
                p = compute_pwmcc(
                    pair_saes[i].decoder.weight.data,
                    pair_saes[j].decoder.weight.data,
                )
                c_pwmcc.append(p)

        # Compute reconstruction quality
        recon_losses = []
        for sae in pair_saes:
            sae.eval()
            with torch.no_grad():
                recon, _, _ = sae(acts[:1000])
                recon_losses.append(F.mse_loss(recon, acts[:1000]).item())

        result = {
            "lambda": lam,
            "pwmcc_mean": float(np.mean(c_pwmcc)),
            "pwmcc_std": float(np.std(c_pwmcc)),
            "pwmcc_values": [float(v) for v in c_pwmcc],
            "recon_loss_mean": float(np.mean(recon_losses)),
            "recon_loss_std": float(np.std(recon_losses)),
            "n_saes": len(pair_saes),
        }
        contrastive_results.append(result)

        print(
            f"  PWMCC: {result['pwmcc_mean']:.4f} +/- {result['pwmcc_std']:.4f}, "
            f"recon: {result['recon_loss_mean']:.6f}"
        )

    manifest["sae_config"] = {"d_sae": d_sae, "k": k, "epochs": epochs}
    manifest["standard_baseline"] = {
        "pwmcc_mean": float(np.mean(standard_pwmcc)),
        "pwmcc_std": float(np.std(standard_pwmcc)),
    }
    manifest["contrastive_results"] = contrastive_results
    manifest["completed"] = utc_now()

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS: Contrastive Stability-Promoting Training")
    print("=" * 70)
    print(f"\nStandard baseline PWMCC: {np.mean(standard_pwmcc):.4f} +/- {np.std(standard_pwmcc):.4f}")
    print(f"\nContrastive training sweep:")
    print(f"  {'lambda':>8s}  {'PWMCC':>12s}  {'Recon Loss':>12s}  {'Improvement':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

    baseline_pwmcc = np.mean(standard_pwmcc)
    for r in contrastive_results:
        improvement = r["pwmcc_mean"] - baseline_pwmcc
        print(
            f"  {r['lambda']:8.3f}  "
            f"{r['pwmcc_mean']:8.4f}+/-{r['pwmcc_std']:.3f}  "
            f"{r['recon_loss_mean']:12.6f}  "
            f"{improvement:+12.4f}"
        )

    best = max(contrastive_results, key=lambda r: r["pwmcc_mean"])
    print(f"\nBest lambda: {best['lambda']} (PWMCC={best['pwmcc_mean']:.4f})")
    print(f"Improvement over standard: {best['pwmcc_mean'] - baseline_pwmcc:+.4f}")

    if best["pwmcc_mean"] > baseline_pwmcc + 0.05:
        print("\nCONFIRMED: Contrastive training improves SAE feature stability!")
    elif best["pwmcc_mean"] > baseline_pwmcc:
        print("\nMarginal improvement with contrastive training.")
    else:
        print("\nNo improvement - contrastive loss may need tuning.")
    print("=" * 70)

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        lambdas = [r["lambda"] for r in contrastive_results]
        pwmccs = [r["pwmcc_mean"] for r in contrastive_results]
        pwmcc_stds = [r["pwmcc_std"] for r in contrastive_results]
        recons = [r["recon_loss_mean"] for r in contrastive_results]

        # Panel 1: PWMCC vs lambda
        ax = axes[0]
        ax.errorbar(range(len(lambdas)), pwmccs, yerr=pwmcc_stds,
                     fmt="o-", color="#2980b9", capsize=5, markersize=8, label="Contrastive")
        ax.axhline(y=baseline_pwmcc, color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"Standard baseline ({baseline_pwmcc:.3f})")
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f"{l}" for l in lambdas], rotation=45)
        ax.set_xlabel("Alignment Lambda")
        ax.set_ylabel("PWMCC")
        ax.set_title("Stability vs Alignment Strength")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: Stability-reconstruction tradeoff
        ax = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
        for i, (pw, rc, lam) in enumerate(zip(pwmccs, recons, lambdas)):
            ax.scatter(rc, pw, s=150, c=[colors[i]], zorder=5, edgecolors="black")
            ax.annotate(f"$\\lambda$={lam}", (rc, pw), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

        # Add standard baseline
        std_recon = float(np.mean([
            F.mse_loss(sae(acts[:1000])[0], acts[:1000]).item()
            for sae in standard_saes
        ]))
        ax.scatter(std_recon, baseline_pwmcc, s=200, c="red", marker="*",
                   zorder=5, label="Standard baseline")

        ax.set_xlabel("Reconstruction Loss (MSE)")
        ax.set_ylabel("PWMCC (Feature Stability)")
        ax.set_title("Stability-Reconstruction Tradeoff")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_contrastive_stability.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nFigure saved: {fig_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate figure: {e}")

    # ── Save manifest ──
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
