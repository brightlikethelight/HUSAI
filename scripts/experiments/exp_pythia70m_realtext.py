#!/usr/bin/env python3
"""Pythia-70M stability with real text + d_sae sweep.

Fixes random-token methodology: uses openwebtext for realistic activation distributions.
Adds d_sae sweep to test effective rank predictor on LLM scale.
"""
import json
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.simple_sae import TopKSAE, ReLUSAE

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "pythia70m_stability"
CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "pythia70m_activations"
FIGURES_DIR = PROJECT_ROOT / "figures"
SEEDS = [42, 123, 456, 789, 1011]


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_pwmcc(d1, d2):
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_effective_rank(acts):
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered[:5000])
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    return torch.exp(entropy).item()


def compute_random_baseline(d_model, d_sae, n_trials=20):
    vals = [compute_pwmcc(torch.randn(d_model, d_sae), torch.randn(d_model, d_sae)) for _ in range(n_trials)]
    return float(np.mean(vals))


def extract_activations_realtext(n_samples, layer=0, hook_name="hook_resid_pre", device="cpu", cache_path=None):
    if cache_path and cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    from transformer_lens import HookedTransformer
    from datasets import load_dataset

    print("  Loading Pythia-70M...")
    model = HookedTransformer.from_pretrained("pythia-70m-deduped", device=device)
    model.eval()
    d_model = model.cfg.d_model
    seq_len = 128
    batch_size = 32

    print("  Loading text dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    # Filter to non-empty lines with decent length
    texts = [t for t in ds["text"] if len(t.strip()) > 100]
    print(f"  Got {len(texts)} text passages")

    print(f"  Tokenizing and extracting activations (layer {layer}, {hook_name})...")
    all_acts = []
    collected = 0
    text_idx = 0

    while collected < n_samples and text_idx < len(texts):
        batch_texts = texts[text_idx:text_idx + batch_size]
        text_idx += batch_size

        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        if tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens.to(device),
                names_filter=f"blocks.{layer}.{hook_name}",
            )
            acts = cache[f"blocks.{layer}.{hook_name}"]
            flat_acts = acts.reshape(-1, d_model).cpu()
            all_acts.append(flat_acts)
            collected += flat_acts.shape[0]

        if text_idx % 200 == 0:
            print(f"    Collected {collected}/{n_samples} vectors...")

    activations = torch.cat(all_acts, dim=0)[:n_samples]
    print(f"  Extracted {activations.shape[0]} vectors, d_model={activations.shape[1]}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activations, cache_path)
        print(f"  Cached to {cache_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return activations


def train_sae(activations, arch, d_sae, k=64, l1_coef=5e-4, seed=42,
              epochs=10, lr=3e-4, batch_size=1024, device="cpu"):
    torch.manual_seed(seed)
    d_model = activations.shape[1]

    if arch == "topk":
        sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)
    else:
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
            recon, latents, aux = sae(batch)
            loss = F.mse_loss(recon, batch) + aux
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: loss={final_loss:.6f}")

    return sae.cpu(), final_loss


def run_stability(saes, label, d_model, d_sae):
    pwmcc_values = []
    for i in range(len(saes)):
        for j in range(i + 1, len(saes)):
            p = compute_pwmcc(saes[i].decoder.weight.data, saes[j].decoder.weight.data)
            pwmcc_values.append(p)
    random_bl = compute_random_baseline(d_model, d_sae)
    return {
        "label": label, "n_saes": len(saes),
        "pwmcc_mean": float(np.mean(pwmcc_values)),
        "pwmcc_std": float(np.std(pwmcc_values)),
        "pwmcc_values": [float(v) for v in pwmcc_values],
        "random_baseline": random_bl,
        "pwmcc_over_random": float(np.mean(pwmcc_values)) / random_bl if random_bl > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--sae-epochs", type=int, default=10)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"run_realtext_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1.1b: Pythia-70M Real Text + d_sae Sweep")
    print("=" * 70)

    manifest = {
        "experiment": "pythia70m_realtext_sweep",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": args.device,
    }

    # Phase 1: Extract real text activations
    print("\nPhase 1: Extracting activations from real text...")
    cache_path = CACHE_DIR / f"pythia70m_layer0_resid_pre_realtext_{args.n_samples}.pt"
    activations = extract_activations_realtext(
        n_samples=args.n_samples, layer=0, hook_name="hook_resid_pre",
        device=args.device, cache_path=cache_path,
    )
    d_model = activations.shape[1]

    # Phase 2: Activation stats
    print("\nPhase 2: Activation statistics...")
    eff_rank = compute_effective_rank(activations)
    print(f"  d_model={d_model}, effective_rank={eff_rank:.1f} ({eff_rank/d_model*100:.1f}%)")
    manifest["activation_stats"] = {
        "d_model": d_model,
        "effective_rank": eff_rank,
        "n_samples": activations.shape[0],
    }

    # Phase 3: d_sae sweep for both architectures
    d_sae_values = [256, 512, 1024, 2048, 4096]
    k_values = {256: 16, 512: 32, 1024: 64, 2048: 64, 4096: 64}
    sweep_results = []

    for d_sae in d_sae_values:
        k = k_values[d_sae]
        ratio = d_sae / eff_rank
        print(f"\n--- d_sae={d_sae}, k={k}, ratio={ratio:.2f} ---")

        # Train TopK SAEs
        print(f"  Training {len(SEEDS)} TopK SAEs...")
        topk_saes = []
        for s in SEEDS:
            sae, _ = train_sae(activations, "topk", d_sae, k=k, seed=s,
                               epochs=args.sae_epochs, device=args.device)
            topk_saes.append(sae)

        # Train ReLU SAEs
        print(f"  Training {len(SEEDS)} ReLU SAEs...")
        relu_saes = []
        for s in SEEDS:
            sae, _ = train_sae(activations, "relu", d_sae, l1_coef=5e-4, seed=s,
                               epochs=args.sae_epochs, device=args.device)
            relu_saes.append(sae)

        topk_res = run_stability(topk_saes, f"TopK d_sae={d_sae}", d_model, d_sae)
        relu_res = run_stability(relu_saes, f"ReLU d_sae={d_sae}", d_model, d_sae)

        print(f"  TopK PWMCC={topk_res['pwmcc_mean']:.4f} ({topk_res['pwmcc_over_random']:.2f}x random)")
        print(f"  ReLU PWMCC={relu_res['pwmcc_mean']:.4f} ({relu_res['pwmcc_over_random']:.2f}x random)")

        sweep_results.append({
            "d_sae": d_sae, "k": k, "d_sae_over_eff_rank": ratio,
            "topk": topk_res, "relu": relu_res,
        })

    manifest["sweep_results"] = sweep_results
    manifest["completed"] = utc_now()

    # Report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Pythia-70M, layer 0, real text, eff_rank={eff_rank:.1f}")
    header = f"{'d_sae':>6} {'ratio':>6} {'TopK PWMCC':>12} {'TopK/rand':>10} {'ReLU PWMCC':>12} {'ReLU/rand':>10}"
    print(header)
    for r in sweep_results:
        line = (f"{r['d_sae']:>6} {r['d_sae_over_eff_rank']:>6.2f} "
                f"{r['topk']['pwmcc_mean']:>12.4f} {r['topk']['pwmcc_over_random']:>10.2f}x "
                f"{r['relu']['pwmcc_mean']:>12.4f} {r['relu']['pwmcc_over_random']:>10.2f}x")
        print(line)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: d_sae/eff_rank vs PWMCC/random
        ax = axes[0]
        ratios = [r["d_sae_over_eff_rank"] for r in sweep_results]
        topk_ratios = [r["topk"]["pwmcc_over_random"] for r in sweep_results]
        relu_ratios = [r["relu"]["pwmcc_over_random"] for r in sweep_results]
        ax.plot(ratios, topk_ratios, "o-", color="#2980b9", label="TopK", markersize=8)
        ax.plot(ratios, relu_ratios, "s-", color="#2ecc71", label="ReLU", markersize=8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax.set_xlabel("d_sae / Effective Rank")
        ax.set_ylabel("PWMCC / Random")
        ax.set_title("Pythia-70M: Stability vs Overparameterization")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: Raw PWMCC with baselines
        ax = axes[1]
        topk_pwmcc = [r["topk"]["pwmcc_mean"] for r in sweep_results]
        relu_pwmcc = [r["relu"]["pwmcc_mean"] for r in sweep_results]
        topk_bl = [r["topk"]["random_baseline"] for r in sweep_results]
        d_saes = [r["d_sae"] for r in sweep_results]
        ax.plot(d_saes, topk_pwmcc, "o-", color="#2980b9", label="TopK trained")
        ax.plot(d_saes, relu_pwmcc, "s-", color="#2ecc71", label="ReLU trained")
        ax.plot(d_saes, topk_bl, "o--", color="#2980b9", alpha=0.3, label="Random baseline")
        ax.set_xlabel("d_sae")
        ax.set_ylabel("PWMCC")
        ax.set_title("Pythia-70M: Feature Stability")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_pythia70m_realtext_sweep.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nFigure: {fig_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate figure: {e}")

    # Save manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
