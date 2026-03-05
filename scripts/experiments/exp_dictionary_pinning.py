#!/usr/bin/env python3
"""Experiment 2.4: Dictionary Pinning / Warm-Start Stability.

If the problem is that random init sends each seed to a different basin,
what if you anchor some decoder columns?

Design:
  - Take a trained SAE's top-N most active decoder columns.
  - Initialize new SAE with those N columns fixed, remaining random.
  - Train with those columns frozen (or with strong regularizer).
  - Measure PWMCC between original SAE and pinned one.
  - Vary N from 0 to d_sae.

Expected: PWMCC increases with N, with a transition point where high
stability is achieved without sacrificing reconstruction.

This is a dead simple practical method for stabilizing SAEs.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_dictionary_pinning.py
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

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "dictionary_pinning"
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


def train_sae_standard(
    activations: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30,
) -> Tuple[TopKSAE, float]:
    torch.manual_seed(seed)
    d_model = activations.shape[1]
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for (batch,) in dataloader:
            recon, latents, aux_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n += 1
        final_loss = epoch_loss / max(n, 1)

    return sae, final_loss


def train_sae_pinned(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    reference_decoder: torch.Tensor,
    pinned_indices: List[int],
    epochs: int = 30,
    pin_mode: str = "freeze",
) -> Tuple[TopKSAE, float]:
    """Train SAE with some decoder columns pinned from a reference.

    Args:
        activations: training data
        d_sae, k: SAE hyperparams
        seed: random seed for unpinned columns
        reference_decoder: [d_model, d_sae] decoder from reference SAE
        pinned_indices: which columns to pin
        epochs: training epochs
        pin_mode: "freeze" (zero grad) or "regularize" (L2 penalty toward reference)
    """
    torch.manual_seed(seed)
    d_model = activations.shape[1]
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)

    # Initialize pinned columns from reference
    with torch.no_grad():
        for idx in pinned_indices:
            sae.decoder.weight.data[:, idx] = reference_decoder[:, idx].clone()
        sae.normalize_decoder()

    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    pinned_mask = torch.zeros(d_sae, dtype=torch.bool, device=DEVICE)
    pinned_mask[pinned_indices] = True

    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for (batch,) in dataloader:
            recon, latents, aux_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + aux_loss

            if pin_mode == "regularize":
                # L2 penalty to stay near reference for pinned columns
                ref_cols = reference_decoder[:, pinned_indices].to(DEVICE)
                curr_cols = sae.decoder.weight.data[:, pinned_indices]
                pin_loss = 0.1 * F.mse_loss(curr_cols, ref_cols)
                loss = loss + pin_loss

            optimizer.zero_grad()
            loss.backward()

            if pin_mode == "freeze":
                # Zero out gradients for pinned decoder columns
                if sae.decoder.weight.grad is not None:
                    sae.decoder.weight.grad[:, pinned_mask] = 0.0
                    # Also zero encoder rows corresponding to pinned features
                    if sae.encoder.weight.grad is not None:
                        sae.encoder.weight.grad[pinned_mask, :] = 0.0
                    if sae.encoder.bias.grad is not None:
                        sae.encoder.bias.grad[pinned_mask] = 0.0

            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()
            n += 1

        final_loss = epoch_loss / max(n, 1)

    return sae, final_loss


def main():
    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2.4: Dictionary Pinning / Warm-Start Stability")
    print("=" * 70)
    print(f"Output: {run_dir}")

    manifest = {
        "experiment": "dictionary_pinning",
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

    # ── Train reference SAE ──
    ref_seed = SEEDS[0]
    print(f"\nTraining reference SAE (seed {ref_seed})...")
    ref_sae, ref_loss = train_sae_standard(acts, d_sae, k, ref_seed, epochs)
    print(f"  Reference recon loss: {ref_loss:.6f}")

    # Find most active features in reference
    ref_sae.eval()
    with torch.no_grad():
        _, ref_latents, _ = ref_sae(acts[:2000])
    feature_activity = ref_latents.abs().mean(dim=0)  # [d_sae]
    sorted_features = feature_activity.argsort(descending=True).tolist()

    # ── Baseline: standard training with different seeds ──
    print("\nBaseline: standard training...")
    standard_pwmcc = []
    for s in SEEDS[1:]:
        sae, _ = train_sae_standard(acts, d_sae, k, s, epochs)
        p = compute_pwmcc(ref_sae.decoder.weight.data, sae.decoder.weight.data)
        standard_pwmcc.append(p)
    baseline_pwmcc = float(np.mean(standard_pwmcc))
    print(f"  Standard PWMCC vs reference: {baseline_pwmcc:.4f} +/- {np.std(standard_pwmcc):.4f}")

    # ── Sweep N (number of pinned columns) ──
    n_pinned_values = [0, 16, 32, 64, 128, 256, 512, 768, 1024]
    pinning_results = []

    for n_pinned in n_pinned_values:
        print(f"\nPinning {n_pinned}/{d_sae} columns...")
        pinned_indices = sorted_features[:n_pinned]

        pwmcc_vals = []
        losses = []
        for s in SEEDS[1:]:
            sae, loss = train_sae_pinned(
                acts, d_sae, k, s,
                reference_decoder=ref_sae.decoder.weight.data,
                pinned_indices=pinned_indices,
                epochs=epochs,
                pin_mode="freeze",
            )
            p = compute_pwmcc(ref_sae.decoder.weight.data, sae.decoder.weight.data)
            pwmcc_vals.append(p)
            losses.append(loss)

        result = {
            "n_pinned": n_pinned,
            "frac_pinned": n_pinned / d_sae,
            "pwmcc_mean": float(np.mean(pwmcc_vals)),
            "pwmcc_std": float(np.std(pwmcc_vals)),
            "recon_loss_mean": float(np.mean(losses)),
            "recon_loss_std": float(np.std(losses)),
            "improvement_over_baseline": float(np.mean(pwmcc_vals)) - baseline_pwmcc,
        }
        pinning_results.append(result)

        print(
            f"  PWMCC: {result['pwmcc_mean']:.4f} +/- {result['pwmcc_std']:.4f}, "
            f"recon: {result['recon_loss_mean']:.6f}, "
            f"improvement: {result['improvement_over_baseline']:+.4f}"
        )

    manifest["sae_config"] = {"d_sae": d_sae, "k": k, "epochs": epochs}
    manifest["ref_seed"] = ref_seed
    manifest["ref_loss"] = ref_loss
    manifest["baseline_pwmcc"] = baseline_pwmcc
    manifest["baseline_pwmcc_std"] = float(np.std(standard_pwmcc)),
    manifest["pinning_results"] = pinning_results
    manifest["completed"] = utc_now()

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS: Dictionary Pinning")
    print("=" * 70)
    print(f"\nBaseline (no pinning) PWMCC: {baseline_pwmcc:.4f}")
    print(f"Reference recon loss: {ref_loss:.6f}")
    print(f"\nPinning sweep:")
    print(f"  {'N pinned':>10s}  {'Fraction':>8s}  {'PWMCC':>12s}  {'Recon':>10s}  {'Improv':>10s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")

    for r in pinning_results:
        print(
            f"  {r['n_pinned']:10d}  {r['frac_pinned']:8.2%}  "
            f"{r['pwmcc_mean']:.4f}+/-{r['pwmcc_std']:.3f}  "
            f"{r['recon_loss_mean']:10.6f}  {r['improvement_over_baseline']:+10.4f}"
        )

    # Find transition point
    transition = None
    for r in pinning_results:
        if r["pwmcc_mean"] > 0.7:
            transition = r
            break

    if transition:
        print(
            f"\nTransition to high stability (>0.7): "
            f"N={transition['n_pinned']} ({transition['frac_pinned']:.0%})"
        )
        print(
            f"  Recon loss at transition: {transition['recon_loss_mean']:.6f} "
            f"(vs baseline {ref_loss:.6f})"
        )
    else:
        print("\nNo transition to >0.7 PWMCC found (may need more pinned features)")

    print("=" * 70)

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        n_pinned = [r["n_pinned"] for r in pinning_results]
        pwmccs = [r["pwmcc_mean"] for r in pinning_results]
        pwmcc_stds = [r["pwmcc_std"] for r in pinning_results]
        recons = [r["recon_loss_mean"] for r in pinning_results]

        # Panel 1: PWMCC vs N pinned
        ax = axes[0]
        ax.errorbar(n_pinned, pwmccs, yerr=pwmcc_stds, fmt="o-",
                     color="#2980b9", capsize=5, markersize=8)
        ax.axhline(y=baseline_pwmcc, color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"No pinning ({baseline_pwmcc:.3f})")
        ax.axhline(y=0.7, color="#27ae60", linestyle=":", linewidth=1,
                    label="Stability threshold (0.7)")
        ax.fill_between([0, d_sae], [0.7, 0.7], [1.0, 1.0], alpha=0.1, color="green")
        ax.set_xlabel("Number of Pinned Features")
        ax.set_ylabel("PWMCC (vs Reference)")
        ax.set_title("Stability vs Pinning")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-20, d_sae + 20)

        # Panel 2: Stability-reconstruction tradeoff
        ax = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_pinned)))
        for i, (pw, rc, np_val) in enumerate(zip(pwmccs, recons, n_pinned)):
            ax.scatter(rc, pw, s=100 + np_val / 5, c=[colors[i]], zorder=5,
                       edgecolors="black", alpha=0.8)
            ax.annotate(f"N={np_val}", (rc, pw), textcoords="offset points",
                       xytext=(5, 5), fontsize=7)

        ax.axhline(y=baseline_pwmcc, color="#e74c3c", linestyle="--", alpha=0.5)
        ax.axhline(y=0.7, color="#27ae60", linestyle=":", alpha=0.5)
        ax.set_xlabel("Reconstruction Loss (MSE)")
        ax.set_ylabel("PWMCC (Feature Stability)")
        ax.set_title("Stability-Reconstruction Tradeoff")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_dictionary_pinning.pdf"
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
