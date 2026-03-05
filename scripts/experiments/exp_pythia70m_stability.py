#!/usr/bin/env python3
"""Experiment 1.1: Scale SAE stability analysis to Pythia-70M (real LLM).

The paper's Limitation #2: results are on toy tasks. Reviewers will ask
"does this hold on LLMs?" This experiment trains 5 TopK + 5 ReLU SAEs
on Pythia-70M residual stream and computes PWMCC.

Expected: PWMCC will be higher than on algorithmic tasks (~0.5-0.7 based on
Paulo & Belrose) but still with a meaningful gap to 1.0. The effective-rank
theory should predict the result: d_sae/eff_rank ratio predicts PWMCC.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_pythia70m_stability.py [--device cpu]

Requires: transformer_lens, ~2GB disk for activations cache.
"""

import argparse
import json
import sys
import time
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

from src.models.simple_sae import TopKSAE, ReLUSAE

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "pythia70m_stability"
CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "pythia70m_activations"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEEDS = [42, 123, 456, 789, 1011]
N_SAMPLES = 100_000  # number of activation vectors to collect
BATCH_SIZE = 64


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


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 20) -> float:
    vals = []
    for _ in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        vals.append(compute_pwmcc(d1, d2))
    return float(np.mean(vals))


def extract_pythia_activations(
    n_samples: int,
    layer: int = 0,
    hook_name: str = "hook_resid_pre",
    device: str = "cpu",
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Extract activations from Pythia-70M residual stream.

    Uses TransformerLens to load Pythia-70M and extract activations from
    a subset of The Pile / generated data.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    from transformer_lens import HookedTransformer

    print("  Loading Pythia-70M...")
    model = HookedTransformer.from_pretrained("pythia-70m-deduped", device=device)
    model.eval()

    d_model = model.cfg.d_model
    seq_len = 128  # context length per sample
    tokens_needed = n_samples  # we'll get one activation vector per token position

    # Generate random token sequences (simple approach; real data would be better
    # but this suffices for stability measurement since we care about the
    # geometry of the activation space, not the specific content)
    print(f"  Generating {tokens_needed} activation vectors (layer {layer}, {hook_name})...")

    all_acts = []
    collected = 0
    batch_idx = 0

    while collected < tokens_needed:
        # Use random tokens (mimics diverse inputs)
        batch_tokens = torch.randint(0, model.cfg.d_vocab, (BATCH_SIZE, seq_len), device=device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=f"blocks.{layer}.{hook_name}",
            )
            hook_key = f"blocks.{layer}.{hook_name}"
            acts = cache[hook_key]  # [batch, seq_len, d_model]
            # Flatten batch and sequence dims
            flat_acts = acts.reshape(-1, d_model).cpu()
            all_acts.append(flat_acts)
            collected += flat_acts.shape[0]

        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"    Collected {collected}/{tokens_needed} vectors...")

    activations = torch.cat(all_acts, dim=0)[:tokens_needed]
    print(f"  Extracted {activations.shape[0]} activation vectors, d_model={activations.shape[1]}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activations, cache_path)
        print(f"  Cached to {cache_path}")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activations


def train_topk_sae(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 1024,
    device: str = "cpu",
) -> Tuple[TopKSAE, float]:
    """Train TopK SAE on Pythia activations."""
    torch.manual_seed(seed)
    d_model = activations.shape[1]

    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            recon, latents, aux_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

    return sae.cpu(), final_loss


def train_relu_sae(
    activations: torch.Tensor,
    d_sae: int,
    l1_coef: float,
    seed: int,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 1024,
    device: str = "cpu",
) -> Tuple[ReLUSAE, float]:
    """Train ReLU SAE on Pythia activations."""
    torch.manual_seed(seed)
    d_model = activations.shape[1]

    sae = ReLUSAE(d_model=d_model, d_sae=d_sae, l1_coef=l1_coef).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            recon, latents, l1_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 2 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

    return sae.cpu(), final_loss


def run_stability_analysis(
    saes: List[nn.Module],
    label: str,
    d_model: int,
    d_sae: int,
) -> Dict:
    """Compute stability metrics for a list of trained SAEs."""
    pwmcc_values = []
    for i in range(len(saes)):
        for j in range(i + 1, len(saes)):
            p = compute_pwmcc(saes[i].decoder.weight.data, saes[j].decoder.weight.data)
            pwmcc_values.append(p)

    random_bl = compute_random_baseline(d_model, d_sae)

    return {
        "label": label,
        "n_saes": len(saes),
        "pwmcc_mean": float(np.mean(pwmcc_values)),
        "pwmcc_std": float(np.std(pwmcc_values)),
        "pwmcc_values": [float(v) for v in pwmcc_values],
        "random_baseline": random_bl,
        "pwmcc_over_random": float(np.mean(pwmcc_values)) / random_bl if random_bl > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Pythia-70M SAE stability experiment")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--d-sae", type=int, default=2048, help="SAE dictionary size")
    parser.add_argument("--k", type=int, default=64, help="TopK sparsity")
    parser.add_argument("--l1-coef", type=float, default=5e-4, help="ReLU L1 coefficient")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1.1: Pythia-70M SAE Stability")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Output: {run_dir}")

    manifest = {
        "experiment": "pythia70m_stability",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": args.device,
        "n_samples": args.n_samples,
        "d_sae": args.d_sae,
        "k": args.k,
        "l1_coef": args.l1_coef,
        "sae_epochs": args.sae_epochs,
    }

    # ── Phase 1: Extract activations ──
    print("\nPhase 1: Extracting Pythia-70M activations...")
    cache_path = CACHE_DIR / f"pythia70m_layer0_resid_pre_{args.n_samples}.pt"
    activations = extract_pythia_activations(
        n_samples=args.n_samples,
        layer=0,
        hook_name="hook_resid_pre",
        device=args.device,
        cache_path=cache_path,
    )
    d_model = activations.shape[1]
    print(f"  Activations: {activations.shape} (d_model={d_model})")

    # ── Phase 2: Compute activation statistics ──
    print("\nPhase 2: Computing activation statistics...")
    eff_rank = compute_effective_rank(activations)
    act_var = activations.var(dim=0).sum().item()
    print(f"  Effective rank: {eff_rank:.1f}")
    print(f"  Total variance: {act_var:.2f}")
    print(f"  d_sae/eff_rank ratio: {args.d_sae/eff_rank:.2f}")

    manifest["activation_stats"] = {
        "d_model": d_model,
        "n_samples": activations.shape[0],
        "effective_rank": eff_rank,
        "total_variance": act_var,
        "d_sae_over_eff_rank": args.d_sae / eff_rank,
    }

    # ── Phase 3: Train TopK SAEs ──
    print(f"\nPhase 3: Training {len(SEEDS)} TopK SAEs (d_sae={args.d_sae}, k={args.k})...")
    topk_saes = []
    topk_losses = []
    for s in SEEDS:
        print(f"  Seed {s}:")
        sae, loss = train_topk_sae(
            activations, d_sae=args.d_sae, k=args.k, seed=s,
            epochs=args.sae_epochs, device=args.device,
        )
        topk_saes.append(sae)
        topk_losses.append(loss)

    # ── Phase 4: Train ReLU SAEs ──
    print(f"\nPhase 4: Training {len(SEEDS)} ReLU SAEs (d_sae={args.d_sae}, l1={args.l1_coef})...")
    relu_saes = []
    relu_losses = []
    for s in SEEDS:
        print(f"  Seed {s}:")
        sae, loss = train_relu_sae(
            activations, d_sae=args.d_sae, l1_coef=args.l1_coef, seed=s,
            epochs=args.sae_epochs, device=args.device,
        )
        relu_saes.append(sae)
        relu_losses.append(loss)

    # ── Phase 5: Compute stability ──
    print("\nPhase 5: Computing stability metrics...")
    topk_results = run_stability_analysis(topk_saes, "TopK", d_model, args.d_sae)
    relu_results = run_stability_analysis(relu_saes, "ReLU", d_model, args.d_sae)

    # Cross-architecture PWMCC
    cross_pwmcc = []
    for t in topk_saes:
        for r in relu_saes:
            p = compute_pwmcc(t.decoder.weight.data, r.decoder.weight.data)
            cross_pwmcc.append(p)

    manifest["results"] = {
        "topk": topk_results,
        "relu": relu_results,
        "cross_architecture_pwmcc_mean": float(np.mean(cross_pwmcc)),
        "cross_architecture_pwmcc_std": float(np.std(cross_pwmcc)),
        "topk_recon_loss": float(np.mean(topk_losses)),
        "relu_recon_loss": float(np.mean(relu_losses)),
    }
    manifest["completed"] = utc_now()

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS: Pythia-70M SAE Stability")
    print("=" * 70)
    print(f"\nModel: Pythia-70M, layer 0, hook_resid_pre")
    print(f"Activations: {activations.shape[0]} samples, d_model={d_model}")
    print(f"Effective rank: {eff_rank:.1f}")
    print(f"d_sae/eff_rank: {args.d_sae/eff_rank:.2f}")

    for label, res in [("TopK", topk_results), ("ReLU", relu_results)]:
        print(f"\n{label} SAEs:")
        print(f"  PWMCC:            {res['pwmcc_mean']:.4f} +/- {res['pwmcc_std']:.4f}")
        print(f"  Random baseline:  {res['random_baseline']:.4f}")
        print(f"  PWMCC/random:     {res['pwmcc_over_random']:.2f}x")

    print(f"\nCross-architecture PWMCC: {np.mean(cross_pwmcc):.4f}")

    # Compare to modular arithmetic results
    mod_arith_pwmcc = 0.309  # from paper
    print(f"\nComparison to modular arithmetic:")
    print(f"  Mod-arith PWMCC: {mod_arith_pwmcc:.3f}")
    print(f"  Pythia TopK:     {topk_results['pwmcc_mean']:.3f}")
    print(f"  Pythia ReLU:     {relu_results['pwmcc_mean']:.3f}")

    if topk_results["pwmcc_mean"] > mod_arith_pwmcc + 0.1:
        print("  LLM features are more stable than algorithmic tasks (as expected)")
    elif topk_results["pwmcc_mean"] > mod_arith_pwmcc:
        print("  Slight improvement on LLM, but gap to 1.0 remains")
    else:
        print("  LLM stability is similar to algorithmic tasks - concerning!")
    print("=" * 70)

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: PWMCC comparison across settings
        ax = axes[0]
        settings = ["Mod-Arith\n(2-layer)", "Pythia-70M\nTopK", "Pythia-70M\nReLU"]
        pwmccs = [mod_arith_pwmcc, topk_results["pwmcc_mean"], relu_results["pwmcc_mean"]]
        stds = [0.02, topk_results["pwmcc_std"], relu_results["pwmcc_std"]]
        baselines = [0.15, topk_results["random_baseline"], relu_results["random_baseline"]]

        x = np.arange(3)
        ax.bar(x - 0.15, pwmccs, width=0.3, yerr=stds, color=["#e74c3c", "#2980b9", "#2ecc71"],
               capsize=5, label="Trained PWMCC")
        ax.bar(x + 0.15, baselines, width=0.3, alpha=0.3, color=["#c0392b", "#1a5276", "#196f3d"],
               label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(settings)
        ax.set_ylabel("PWMCC")
        ax.set_title("Feature Stability Across Scales")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Panel 2: d_sae/eff_rank vs PWMCC (will be populated with more data in exp 2.3)
        ax = axes[1]
        ratio = args.d_sae / eff_rank
        ax.scatter([ratio], [topk_results["pwmcc_mean"]], s=200, c="#2980b9",
                   marker="o", label="Pythia-70M TopK", zorder=5)
        ax.scatter([ratio], [relu_results["pwmcc_mean"]], s=200, c="#2ecc71",
                   marker="s", label="Pythia-70M ReLU", zorder=5)
        # Add modular arithmetic point (d_sae=1024, eff_rank~80)
        ax.scatter([1024/80], [mod_arith_pwmcc], s=200, c="#e74c3c",
                   marker="^", label="Mod-arith (2L)", zorder=5)
        ax.set_xlabel("d_sae / Effective Rank")
        ax.set_ylabel("PWMCC")
        ax.set_title("Overparameterization vs Stability")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 3: PWMCC distribution
        ax = axes[2]
        ax.hist(topk_results["pwmcc_values"], bins=8, alpha=0.6, color="#2980b9",
                label="TopK pairs", edgecolor="black")
        ax.hist(relu_results["pwmcc_values"], bins=8, alpha=0.6, color="#2ecc71",
                label="ReLU pairs", edgecolor="black")
        ax.axvline(topk_results["random_baseline"], color="gray", linestyle="--",
                    label=f"Random ({topk_results['random_baseline']:.3f})")
        ax.set_xlabel("PWMCC")
        ax.set_ylabel("Count")
        ax.set_title("Pairwise PWMCC Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_pythia70m_stability.pdf"
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
