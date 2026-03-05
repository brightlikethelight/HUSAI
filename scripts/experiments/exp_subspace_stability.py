#!/usr/bin/env python3
"""Experiment 2.2: Subspace Stability vs Feature Stability.

PWMCC (feature-level alignment) = random. But the *subspace* spanned by active
features might be stable even if individual features aren't.

We measure:
  - Feature stability: PWMCC (already known to be ~0.30)
  - Subspace stability: Grassmann distance / principal angle alignment between
    decoder weight subspaces across seeds

If subspace stability >> feature stability, SAEs learn the same "representation
room" but tile it with different basis vectors. This partially rehabilitates SAE
utility while confirming feature-level claims are unreliable.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_subspace_stability.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from src.models.simple_sae import TopKSAE

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "subspace_stability"
FIGURES_DIR = PROJECT_ROOT / "figures"
DEVICE = "cpu"
SEEDS = [42, 123, 456, 789, 1011]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder weight matrices."""
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """Compute principal angles between two subspaces.

    Args:
        U1: [d, k1] orthonormal basis for subspace 1
        U2: [d, k2] orthonormal basis for subspace 2

    Returns:
        angles: [min(k1,k2)] principal angles in radians
    """
    # Compute SVD of U1^T @ U2
    M = U1.T @ U2  # [k1, k2]
    _, S, _ = torch.svd(M)
    # Clamp to [0,1] for numerical stability
    S = S.clamp(0.0, 1.0)
    angles = torch.acos(S)
    return angles


def compute_grassmann_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute Grassmann distance between two subspaces.

    The Grassmann distance is the norm of the vector of principal angles.
    Ranges from 0 (identical subspaces) to sqrt(k)*pi/2 (orthogonal).
    """
    angles = compute_principal_angles(U1, U2)
    return float(angles.norm().item())


def compute_subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute subspace overlap (mean cos^2 of principal angles).

    Ranges from 0 (orthogonal) to 1 (identical subspace).
    """
    angles = compute_principal_angles(U1, U2)
    return float(torch.cos(angles).pow(2).mean().item())


def get_decoder_subspace(
    decoder_weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Extract top-k principal subspace from decoder weight matrix.

    Args:
        decoder_weight: [d_model, d_sae] decoder columns
        k: number of principal components to keep

    Returns:
        U: [d_model, k] orthonormal basis
    """
    # decoder_weight columns are the feature directions
    # SVD of the decoder to get principal subspace
    U, S, _ = torch.svd(decoder_weight)
    return U[:, :k]


def load_activations() -> torch.Tensor:
    """Load 2-layer transformer activations (standard setup)."""
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


def train_sae(
    activations: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30
) -> TopKSAE:
    """Train a TopK SAE."""
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
    print("EXPERIMENT 2.2: Subspace Stability vs Feature Stability")
    print("=" * 70)
    print(f"Output: {run_dir}")

    manifest = {
        "experiment": "subspace_stability",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": DEVICE,
    }

    # ── Load activations ──
    print("\nLoading activations...")
    acts = load_activations()
    print(f"  Shape: {acts.shape}")

    # ── Train SAEs ──
    d_sae = 1024
    k_topk = 32
    print(f"\nTraining {len(SEEDS)} SAEs (d_sae={d_sae}, k={k_topk})...")
    saes = []
    for s in SEEDS:
        sae = train_sae(acts, d_sae=d_sae, k=k_topk, seed=s)
        saes.append(sae)
        print(f"  Seed {s}: done")

    # ── Compute feature-level stability (PWMCC) ──
    print("\nComputing feature-level stability (PWMCC)...")
    pwmcc_values = []
    for i in range(len(saes)):
        for j in range(i + 1, len(saes)):
            p = compute_pwmcc(saes[i].decoder.weight.data, saes[j].decoder.weight.data)
            pwmcc_values.append(p)
    pwmcc_mean = float(np.mean(pwmcc_values))
    pwmcc_std = float(np.std(pwmcc_values))
    print(f"  PWMCC: {pwmcc_mean:.4f} +/- {pwmcc_std:.4f}")

    # ── Compute subspace stability at various ranks ──
    print("\nComputing subspace stability...")
    subspace_ranks = [8, 16, 32, 64, 96, 128]  # different subspace dimensions
    subspace_results = {}

    for rank in subspace_ranks:
        # Extract subspaces
        subspaces = []
        for sae in saes:
            U = get_decoder_subspace(sae.decoder.weight.data, k=rank)
            subspaces.append(U)

        # Compute pairwise Grassmann distances and subspace overlaps
        grassmann_vals = []
        overlap_vals = []
        for i in range(len(subspaces)):
            for j in range(i + 1, len(subspaces)):
                gd = compute_grassmann_distance(subspaces[i], subspaces[j])
                so = compute_subspace_overlap(subspaces[i], subspaces[j])
                grassmann_vals.append(gd)
                overlap_vals.append(so)

        # Random baseline for subspace overlap
        random_overlaps = []
        for _ in range(20):
            r1 = torch.randn(acts.shape[1], d_sae)
            r2 = torch.randn(acts.shape[1], d_sae)
            U1 = get_decoder_subspace(r1, k=rank)
            U2 = get_decoder_subspace(r2, k=rank)
            random_overlaps.append(compute_subspace_overlap(U1, U2))

        subspace_results[rank] = {
            "grassmann_mean": float(np.mean(grassmann_vals)),
            "grassmann_std": float(np.std(grassmann_vals)),
            "overlap_mean": float(np.mean(overlap_vals)),
            "overlap_std": float(np.std(overlap_vals)),
            "random_overlap_mean": float(np.mean(random_overlaps)),
            "random_overlap_std": float(np.std(random_overlaps)),
            "overlap_over_random": float(np.mean(overlap_vals)) / max(float(np.mean(random_overlaps)), 1e-10),
        }

        print(
            f"  rank={rank:3d}: "
            f"subspace_overlap={np.mean(overlap_vals):.4f}+/-{np.std(overlap_vals):.4f}, "
            f"random={np.mean(random_overlaps):.4f}, "
            f"ratio={subspace_results[rank]['overlap_over_random']:.2f}x"
        )

    manifest["sae_config"] = {"d_sae": d_sae, "k": k_topk, "epochs": 30}
    manifest["pwmcc"] = {
        "mean": pwmcc_mean,
        "std": pwmcc_std,
        "values": [float(v) for v in pwmcc_values],
    }
    manifest["subspace_stability"] = {
        str(k): v for k, v in subspace_results.items()
    }
    manifest["completed"] = utc_now()

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS: Subspace Stability vs Feature Stability")
    print("=" * 70)
    print(f"\nFeature-level stability (PWMCC): {pwmcc_mean:.4f} +/- {pwmcc_std:.4f}")
    print(f"\nSubspace stability (overlap, by rank):")
    for rank in subspace_ranks:
        r = subspace_results[rank]
        print(
            f"  rank={rank:3d}: {r['overlap_mean']:.4f} +/- {r['overlap_std']:.4f} "
            f"(random: {r['random_overlap_mean']:.4f}, "
            f"{r['overlap_over_random']:.2f}x)"
        )

    best_rank = max(subspace_results, key=lambda r: subspace_results[r]["overlap_over_random"])
    best_overlap = subspace_results[best_rank]["overlap_mean"]
    print(f"\nBest subspace stability at rank={best_rank}: {best_overlap:.4f}")
    print(f"  vs feature stability (PWMCC): {pwmcc_mean:.4f}")

    if best_overlap > pwmcc_mean + 0.1:
        print("\n  CONFIRMED: Subspace is more stable than individual features!")
        print("  SAEs learn the same 'representation room' but tile it differently.")
    else:
        print("\n  Both subspace and feature stability are similarly low.")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Subspace overlap vs rank
        ax = axes[0]
        ranks = sorted(subspace_results.keys())
        trained_overlaps = [subspace_results[r]["overlap_mean"] for r in ranks]
        trained_stds = [subspace_results[r]["overlap_std"] for r in ranks]
        random_overlaps = [subspace_results[r]["random_overlap_mean"] for r in ranks]

        ax.errorbar(ranks, trained_overlaps, yerr=trained_stds, fmt="o-",
                     color="#2980b9", label="Trained SAEs", capsize=5, markersize=8)
        ax.plot(ranks, random_overlaps, "x--", color="#95a5a6", label="Random baseline",
                markersize=8)
        ax.axhline(y=pwmcc_mean, color="#e74c3c", linestyle=":", linewidth=2,
                    label=f"Feature PWMCC ({pwmcc_mean:.3f})")
        ax.set_xlabel("Subspace Rank (k)")
        ax.set_ylabel("Subspace Overlap")
        ax.set_title("Subspace vs Feature Stability")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel 2: Ratio to random
        ax = axes[1]
        ratios = [subspace_results[r]["overlap_over_random"] for r in ranks]
        colors = ["#2ecc71" if r > 1.5 else "#f39c12" if r > 1.1 else "#e74c3c" for r in ratios]
        bars = ax.bar(range(len(ranks)), ratios, color=colors)
        ax.axhline(y=1.0, color="black", linestyle="--", label="Random baseline")
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("Subspace Rank (k)")
        ax.set_ylabel("Overlap / Random Baseline")
        ax.set_title("Stability Improvement Over Random")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar, ratio in zip(bars, ratios):
            ax.annotate(f"{ratio:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_subspace_stability.pdf"
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
