#!/usr/bin/env python3
"""Experiment 2.3: Effective Rank as Universal Stability Predictor.

Tests whether d_sae/effective_rank universally predicts PWMCC across
different models, tasks, and architectures. If a simple curve fits,
this becomes a practical tool: every SAE practitioner can predict
stability before training.

Combines data from:
  - Modular arithmetic (2-layer, existing)
  - Modular arithmetic (1-layer, from exp 1.2)
  - Pythia-70M (from exp 1.1)
  - Sweep of d_sae values on each

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_effective_rank_predictor.py
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
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.utils.config import TransformerConfig
from src.models.simple_sae import TopKSAE

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "effective_rank_predictor"
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


def compute_effective_rank(acts: torch.Tensor) -> float:
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered[:5000])  # subsample for speed
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    return torch.exp(entropy).item()


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 10) -> float:
    vals = []
    for _ in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        vals.append(compute_pwmcc(d1, d2))
    return float(np.mean(vals))


def train_sae(
    acts: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30
) -> TopKSAE:
    torch.manual_seed(seed)
    d_model = acts.shape[1]
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    dataset = TensorDataset(acts)
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


def sweep_d_sae(
    acts: torch.Tensor,
    d_sae_values: List[int],
    eff_rank: float,
    seeds: List[int],
    label: str,
    epochs: int = 30,
) -> List[Dict]:
    """Sweep d_sae and measure PWMCC at each point."""
    d_model = acts.shape[1]
    results = []

    for d_sae in d_sae_values:
        k = max(4, min(d_sae // 4, 64))  # heuristic: k = d_sae/4, capped
        print(f"  {label}: d_sae={d_sae}, k={k}...")

        saes = [train_sae(acts, d_sae, k, s, epochs=epochs) for s in seeds]

        pwmcc_values = []
        for i in range(len(saes)):
            for j in range(i + 1, len(saes)):
                p = compute_pwmcc(saes[i].decoder.weight.data, saes[j].decoder.weight.data)
                pwmcc_values.append(p)

        random_bl = compute_random_baseline(d_model, d_sae)
        ratio = d_sae / eff_rank

        results.append({
            "label": label,
            "d_sae": d_sae,
            "k": k,
            "d_model": d_model,
            "eff_rank": eff_rank,
            "d_sae_over_eff_rank": ratio,
            "pwmcc_mean": float(np.mean(pwmcc_values)),
            "pwmcc_std": float(np.std(pwmcc_values)),
            "random_baseline": random_bl,
            "pwmcc_over_random": float(np.mean(pwmcc_values)) / random_bl if random_bl > 0 else float("inf"),
        })

        print(
            f"    ratio={ratio:.2f}, PWMCC={np.mean(pwmcc_values):.4f}, "
            f"random={random_bl:.4f}, {np.mean(pwmcc_values)/random_bl:.2f}x"
        )

    return results


def load_2layer_activations() -> torch.Tensor:
    """Load standard 2-layer transformer activations."""
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


def load_1layer_activations() -> torch.Tensor:
    """Load 1-layer transformer activations (from exp 1.2)."""
    ckpt_path = PROJECT_ROOT / "results" / "experiments" / "1layer_ground_truth" / "transformer_1layer.pt"

    if not ckpt_path.exists():
        print("  1-layer checkpoint not found. Training minimal 1-layer model...")
        torch.manual_seed(42)
        config = TransformerConfig(
            n_layers=1, d_model=128, n_heads=4, d_mlp=512, vocab_size=117, max_seq_len=7,
        )
        model = ModularArithmeticTransformer(config, device=DEVICE)
        dataset = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=42, format="sequence")
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5000):
            model.train()
            for batch_tokens, batch_labels in dataloader:
                logits = model(batch_tokens)
                loss = criterion(logits[:, -2, :113], batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"    Epoch {epoch+1}/5000")

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.model.state_dict(),
            "config": config.model_dump(),
        }, ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        config = TransformerConfig(**checkpoint["config"])
        model = ModularArithmeticTransformer(config, device=DEVICE)
        model.model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    acts = []
    with torch.no_grad():
        for batch, _ in dataloader:
            a = model.get_activations(batch, layer=0)[:, -2, :]
            acts.append(a)
    return torch.cat(acts, dim=0)


def power_law_model(x: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """PWMCC ~ a / (x^alpha) + c"""
    return a / np.power(x, alpha) + c


def main():
    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2.3: Effective Rank as Universal Stability Predictor")
    print("=" * 70)
    print(f"Output: {run_dir}")

    manifest = {
        "experiment": "effective_rank_predictor",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": DEVICE,
    }

    all_results = []

    # ── Setting 1: 2-layer mod-arith ──
    print("\nSetting 1: 2-layer modular arithmetic (layer 1)")
    acts_2l = load_2layer_activations()
    eff_rank_2l = compute_effective_rank(acts_2l)
    print(f"  Effective rank: {eff_rank_2l:.1f}, d_model={acts_2l.shape[1]}")

    d_sae_values_2l = [32, 64, 128, 256, 512, 1024]
    results_2l = sweep_d_sae(acts_2l, d_sae_values_2l, eff_rank_2l, SEEDS, "2-layer mod-arith")
    all_results.extend(results_2l)

    # ── Setting 2: 1-layer mod-arith ──
    print("\nSetting 2: 1-layer modular arithmetic (layer 0)")
    acts_1l = load_1layer_activations()
    eff_rank_1l = compute_effective_rank(acts_1l)
    print(f"  Effective rank: {eff_rank_1l:.1f}, d_model={acts_1l.shape[1]}")

    d_sae_values_1l = [32, 64, 128, 256, 512, 1024]
    results_1l = sweep_d_sae(acts_1l, d_sae_values_1l, eff_rank_1l, SEEDS, "1-layer mod-arith")
    all_results.extend(results_1l)

    # ── Fit universal curve ──
    print("\nFitting universal curve: PWMCC ~ a / (d_sae/eff_rank)^alpha + c")

    ratios = np.array([r["d_sae_over_eff_rank"] for r in all_results])
    pwmccs = np.array([r["pwmcc_mean"] for r in all_results])
    randoms = np.array([r["random_baseline"] for r in all_results])

    # Fit normalized PWMCC (above random)
    pwmcc_above_random = pwmccs - randoms

    try:
        popt, pcov = curve_fit(
            power_law_model, ratios, pwmccs,
            p0=[1.0, 0.5, 0.2], maxfev=5000,
            bounds=([0, 0, 0], [10, 5, 1]),
        )
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))

        # Goodness of fit
        predicted = power_law_model(ratios, *popt)
        ss_res = np.sum((pwmccs - predicted) ** 2)
        ss_tot = np.sum((pwmccs - pwmccs.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  Fitted: PWMCC = {a:.3f} / (d_sae/eff_rank)^{alpha:.3f} + {c:.3f}")
        print(f"  R^2 = {r_squared:.4f}")
        print(f"  Parameter errors: a+/-{perr[0]:.3f}, alpha+/-{perr[1]:.3f}, c+/-{perr[2]:.3f}")

        manifest["universal_curve"] = {
            "model": "PWMCC = a / (d_sae/eff_rank)^alpha + c",
            "a": float(a),
            "alpha": float(alpha),
            "c": float(c),
            "r_squared": float(r_squared),
            "parameter_errors": [float(e) for e in perr],
        }
    except Exception as e:
        print(f"  Warning: Curve fitting failed: {e}")
        popt = None
        r_squared = None

    manifest["all_results"] = all_results
    manifest["completed"] = utc_now()

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS: Universal Effective Rank Predictor")
    print("=" * 70)

    print(f"\nAll data points (sorted by ratio):")
    for r in sorted(all_results, key=lambda x: x["d_sae_over_eff_rank"]):
        print(
            f"  {r['label']:25s} d_sae={r['d_sae']:5d} "
            f"ratio={r['d_sae_over_eff_rank']:6.2f} "
            f"PWMCC={r['pwmcc_mean']:.4f} "
            f"random={r['random_baseline']:.4f} "
            f"ratio_to_random={r['pwmcc_over_random']:.2f}x"
        )

    if popt is not None:
        print(f"\nFitted curve: PWMCC = {a:.3f} / (d_sae/eff_rank)^{alpha:.3f} + {c:.3f}")
        print(f"R^2 = {r_squared:.4f}")

        # Practical predictions
        print("\nPredictions for common settings:")
        for ratio in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            pred = power_law_model(np.array([ratio]), *popt)[0]
            print(f"  d_sae/eff_rank = {ratio:5.1f}: predicted PWMCC = {pred:.3f}")

    print("=" * 70)

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: PWMCC vs d_sae/eff_rank
        ax = axes[0]
        colors_map = {"2-layer mod-arith": "#e74c3c", "1-layer mod-arith": "#2ecc71"}
        markers_map = {"2-layer mod-arith": "o", "1-layer mod-arith": "s"}

        for label in set(r["label"] for r in all_results):
            subset = [r for r in all_results if r["label"] == label]
            x = [r["d_sae_over_eff_rank"] for r in subset]
            y = [r["pwmcc_mean"] for r in subset]
            yerr = [r["pwmcc_std"] for r in subset]
            bl = [r["random_baseline"] for r in subset]

            color = colors_map.get(label, "#3498db")
            marker = markers_map.get(label, "^")

            ax.errorbar(x, y, yerr=yerr, fmt=f"{marker}-", color=color,
                         label=f"{label} (trained)", capsize=4, markersize=8)
            ax.scatter(x, bl, marker="x", color=color, alpha=0.4, s=40)

        if popt is not None:
            x_fit = np.linspace(0.3, max(ratios) * 1.1, 100)
            y_fit = power_law_model(x_fit, *popt)
            ax.plot(x_fit, y_fit, "k--", linewidth=2, alpha=0.6,
                     label=f"Fit (R$^2$={r_squared:.3f})")

        ax.set_xlabel("d_sae / Effective Rank", fontsize=12)
        ax.set_ylabel("PWMCC", fontsize=12)
        ax.set_title("Universal Stability Predictor", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: PWMCC/random vs ratio
        ax = axes[1]
        for label in set(r["label"] for r in all_results):
            subset = [r for r in all_results if r["label"] == label]
            x = [r["d_sae_over_eff_rank"] for r in subset]
            y = [r["pwmcc_over_random"] for r in subset]

            color = colors_map.get(label, "#3498db")
            marker = markers_map.get(label, "^")

            ax.plot(x, y, f"{marker}-", color=color, label=label, markersize=8)

        ax.axhline(y=1.0, color="gray", linestyle="--", label="Random baseline")
        ax.set_xlabel("d_sae / Effective Rank", fontsize=12)
        ax.set_ylabel("PWMCC / Random Baseline", fontsize=12)
        ax.set_title("Stability Above Random", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_effective_rank_predictor.pdf"
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
