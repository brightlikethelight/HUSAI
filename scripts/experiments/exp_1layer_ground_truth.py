#!/usr/bin/env python3
"""Experiment 1.2: 1-Layer Transformer with Known Fourier Ground Truth.

Nanda et al. showed 1-layer transformers learn clean Fourier circuits (R^2 ~93-98%).
Our 2-layer transformer learns non-Fourier algorithms (R^2 ~2%).

This experiment creates a direct comparison:
  - 1-layer (sparse ground truth): SAEs should recover Fourier features => high PWMCC
  - 2-layer (dense ground truth):  SAEs reconstruct but features are random => low PWMCC

This validates Cui et al.'s identifiability theory experimentally:
when ground truth is sparse and satisfies Condition 1, SAE features are identifiable.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_1layer_ground_truth.py
"""

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

from src.data.modular_arithmetic import ModularArithmeticDataset
from src.models.transformer import ModularArithmeticTransformer
from src.utils.config import TransformerConfig
from src.models.simple_sae import TopKSAE, ReLUSAE
from src.analysis.fourier_validation import get_fourier_basis, compute_fourier_overlap

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "1layer_ground_truth"
FIGURES_DIR = PROJECT_ROOT / "figures"
DEVICE = "cpu"

# Reproducibility
SEEDS = [42, 123, 456, 789, 1011]
N_SEEDS = len(SEEDS)


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


def compute_effective_rank(acts: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of singular values."""
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered)
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    return torch.exp(entropy).item()


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 20) -> float:
    """Compute random PWMCC baseline."""
    vals = []
    for i in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        vals.append(compute_pwmcc(d1, d2))
    return float(np.mean(vals))


def compute_fourier_r_squared(
    activations: torch.Tensor,
    modulus: int,
    model: ModularArithmeticTransformer | None = None,
) -> float:
    """Compute R^2 of Fourier basis fit to activations.

    Uses the model's embedding matrix to define Fourier components in
    model space (d_model dimensions), following Nanda et al. (2023).

    The key insight: the Fourier basis F[k,x] = cos/sin(2*pi*k*x/p)
    lives in token-index space (dim=p). We map it to model space via
    the embedding matrix W_E to get Fourier directions in d_model dims.

    If model is not available, falls back to SVD-based R^2 using PCA
    of token-grouped activations (less precise but still informative).
    """
    p = modulus
    n_samples, d_model = activations.shape

    if model is not None:
        # Use embedding matrix to define Fourier basis in model space
        # W_E: [vocab_size, d_model] -- maps token IDs to embeddings
        W_E = model.model.embed.W_E.data.cpu()  # [vocab, d_model]
        # Only use digit tokens (0..p-1)
        W_E_digits = W_E[:p, :]  # [p, d_model]

        # Build Fourier basis in model space:
        # For each frequency k, the Fourier direction is:
        #   cos_dir[k] = sum_x cos(2*pi*k*x/p) * W_E[x]  (a d_model vector)
        #   sin_dir[k] = sum_x sin(2*pi*k*x/p) * W_E[x]  (a d_model vector)
        x = torch.arange(p, dtype=torch.float32)
        fourier_dirs = []
        for k in range(p):
            cos_coeffs = torch.cos(2 * torch.pi * k * x / p)  # [p]
            sin_coeffs = torch.sin(2 * torch.pi * k * x / p)  # [p]
            cos_dir = cos_coeffs @ W_E_digits  # [d_model]
            sin_dir = sin_coeffs @ W_E_digits  # [d_model]
            fourier_dirs.append(cos_dir)
            fourier_dirs.append(sin_dir)

        fourier_basis_model = torch.stack(fourier_dirs)  # [2p, d_model]
    else:
        # Fallback: use PCA directions as a proxy
        # This won't measure Fourier alignment specifically, but measures
        # how much variance is captured by the top-2p directions
        centered = activations - activations.mean(dim=0, keepdim=True)
        U, S, V = torch.svd(centered)
        fourier_basis_model = V[:, :min(2 * p, d_model)].T  # [k, d_model]

    # Center activations
    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    total_var = acts_centered.var(dim=0).sum().item()
    if total_var < 1e-12:
        return 0.0

    # Project activations onto Fourier basis in model space
    # Remove zero/tiny directions
    norms = fourier_basis_model.norm(dim=1, keepdim=True)
    valid = norms.squeeze() > 1e-8
    fb_valid = fourier_basis_model[valid]
    fb_norm = F.normalize(fb_valid, dim=1)  # [k, d_model]

    # Coefficients: [n, k]
    coeffs = acts_centered @ fb_norm.T
    # Reconstruction in model space: [n, d_model]
    reconstructed = coeffs @ fb_norm

    residual_var = (acts_centered - reconstructed).var(dim=0).sum().item()
    r_squared = 1.0 - residual_var / total_var
    return float(max(0.0, r_squared))


def train_transformer(
    n_layers: int,
    modulus: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 42,
    checkpoint_freq: int = 500,
) -> Tuple[ModularArithmeticTransformer, Dict]:
    """Train a transformer on modular arithmetic until grokking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab_size = modulus + 4  # BOS, EOS, EQUALS, PLUS

    config = TransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
        vocab_size=vocab_size,
        max_seq_len=7,
    )

    model = ModularArithmeticTransformer(config, device=DEVICE)

    dataset = ModularArithmeticDataset(
        modulus=modulus, fraction=0.3, seed=seed, format="sequence"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation set
    val_dataset = ModularArithmeticDataset(
        modulus=modulus, fraction=1.0, seed=seed + 1000, format="sequence"
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    metrics_history = []

    print(f"Training {n_layers}-layer transformer on mod-{modulus}...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_tokens, batch_labels in dataloader:
            batch_tokens = batch_tokens.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            logits = model(batch_tokens)
            # Predict at the answer position (position -2 in sequence format)
            answer_logits = logits[:, -2, :modulus]
            loss = criterion(answer_logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = answer_logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        train_acc = correct / total if total > 0 else 0.0

        # Validation every 100 epochs
        if epoch % 100 == 0 or epoch == epochs:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vb_tokens, vb_labels in val_loader:
                    vb_tokens = vb_tokens.to(DEVICE)
                    vb_labels = vb_labels.to(DEVICE)
                    vlogits = model(vb_tokens)
                    vpreds = vlogits[:, -2, :modulus].argmax(dim=-1)
                    val_correct += (vpreds == vb_labels).sum().item()
                    val_total += len(vb_labels)

            val_acc = val_correct / val_total if val_total > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 500 == 0:
                print(
                    f"  Epoch {epoch}/{epochs}: "
                    f"loss={epoch_loss/len(dataloader):.4f}, "
                    f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
                )

            metrics_history.append({
                "epoch": epoch,
                "loss": epoch_loss / len(dataloader),
                "train_acc": train_acc,
                "val_acc": val_acc,
            })

        # Early stop if grokking achieved
        if best_val_acc > 0.98 and epoch >= 1000:
            print(f"  Grokking achieved at epoch {epoch}! val_acc={best_val_acc:.3f}")
            break

    return model, {"best_val_acc": best_val_acc, "history": metrics_history}


def extract_activations(
    model: ModularArithmeticTransformer,
    modulus: int = 113,
    layer: int = 0,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract activations from the model."""
    dataset = ModularArithmeticDataset(
        modulus=modulus, fraction=1.0, seed=42, format="sequence"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_acts = []
    with torch.no_grad():
        for batch_tokens, _ in dataloader:
            batch_tokens = batch_tokens.to(DEVICE)
            acts = model.get_activations(batch_tokens, layer=layer)
            # Take activations at the answer position
            all_acts.append(acts[:, -2, :])

    return torch.cat(all_acts, dim=0)


def train_sae_simple(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 256,
) -> Tuple[TopKSAE, float]:
    """Train a TopK SAE on activations."""
    torch.manual_seed(seed)
    d_model = activations.shape[1]

    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in dataloader:
            recon, latents, aux_loss = sae(batch)
            loss = F.mse_loss(recon, batch) + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    return sae, final_loss


def run_stability_analysis(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seeds: List[int],
    sae_epochs: int = 30,
    label: str = "",
) -> Dict:
    """Train multiple SAEs and compute stability metrics."""
    print(f"  Training {len(seeds)} SAEs ({label})...")
    saes = []
    losses = []
    for s in seeds:
        sae, loss = train_sae_simple(activations, d_sae, k, seed=s, epochs=sae_epochs)
        saes.append(sae)
        losses.append(loss)

    # Compute pairwise PWMCC
    pwmcc_values = []
    for i in range(len(saes)):
        for j in range(i + 1, len(saes)):
            pwmcc = compute_pwmcc(
                saes[i].decoder.weight.data, saes[j].decoder.weight.data
            )
            pwmcc_values.append(pwmcc)

    # Random baseline
    d_model = activations.shape[1]
    random_bl = compute_random_baseline(d_model, d_sae)

    # Fourier overlap for each SAE
    fourier_basis = get_fourier_basis(activations.shape[1], device=DEVICE)
    fourier_overlaps = []
    for sae in saes:
        overlap = compute_fourier_overlap(sae.decoder.weight.data, fourier_basis)
        fourier_overlaps.append(overlap)

    return {
        "label": label,
        "d_sae": d_sae,
        "k": k,
        "n_seeds": len(seeds),
        "pwmcc_mean": float(np.mean(pwmcc_values)),
        "pwmcc_std": float(np.std(pwmcc_values)),
        "pwmcc_values": [float(v) for v in pwmcc_values],
        "random_baseline": random_bl,
        "pwmcc_over_random": float(np.mean(pwmcc_values)) / random_bl if random_bl > 0 else float("inf"),
        "recon_loss_mean": float(np.mean(losses)),
        "recon_loss_std": float(np.std(losses)),
        "fourier_overlap_mean": float(np.mean(fourier_overlaps)),
        "fourier_overlap_std": float(np.std(fourier_overlaps)),
        "fourier_overlaps": [float(v) for v in fourier_overlaps],
    }


def main():
    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1.2: 1-Layer Ground Truth Comparison")
    print("=" * 70)
    print(f"Output: {run_dir}")
    print()

    manifest = {
        "experiment": "1layer_ground_truth",
        "started": utc_now(),
        "seeds": SEEDS,
        "device": DEVICE,
    }

    # ── Phase 1: Train 1-layer transformer ──
    print("Phase 1: Training 1-layer transformer on mod-113...")
    ckpt_1layer = RESULTS_DIR / "transformer_1layer.pt"

    if ckpt_1layer.exists():
        print(f"  Loading cached checkpoint: {ckpt_1layer}")
        checkpoint = torch.load(ckpt_1layer, map_location=DEVICE)
        config_1l = TransformerConfig(**checkpoint["config"])
        model_1l = ModularArithmeticTransformer(config_1l, device=DEVICE)
        model_1l.model.load_state_dict(checkpoint["model_state_dict"])
        train_info_1l = checkpoint.get("train_info", {})
    else:
        model_1l, train_info_1l = train_transformer(
            n_layers=1, modulus=113, d_model=128, n_heads=4, d_mlp=512,
            epochs=10000, lr=1e-3, seed=42,
        )
        # Save checkpoint
        ckpt_1layer.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model_1l.model.state_dict(),
            "config": model_1l.config.model_dump(),
            "train_info": train_info_1l,
        }, ckpt_1layer)
        print(f"  Saved 1-layer checkpoint to {ckpt_1layer}")

    print(f"  1-layer val accuracy: {train_info_1l.get('best_val_acc', 'N/A')}")

    # ── Phase 2: Load existing 2-layer transformer ──
    print("\nPhase 2: Loading 2-layer transformer...")
    ckpt_2layer = PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt"
    checkpoint_2l = torch.load(ckpt_2layer, map_location=DEVICE)
    config_2l = TransformerConfig(**checkpoint_2l["config"])
    model_2l = ModularArithmeticTransformer(config_2l, device=DEVICE)
    model_2l.model.load_state_dict(checkpoint_2l["model_state_dict"])
    print("  Loaded 2-layer transformer.")

    # ── Phase 3: Extract activations ──
    print("\nPhase 3: Extracting activations...")

    # 1-layer: use layer 0 (the only layer)
    acts_1l = extract_activations(model_1l, modulus=113, layer=0)
    print(f"  1-layer activations: {acts_1l.shape}")

    # 2-layer: use layer 1 (standard for this project)
    acts_2l = extract_activations(model_2l, modulus=113, layer=1)
    print(f"  2-layer activations: {acts_2l.shape}")

    # ── Phase 4: Compute effective rank + Fourier R^2 ──
    print("\nPhase 4: Computing effective rank and Fourier R^2...")
    eff_rank_1l = compute_effective_rank(acts_1l)
    eff_rank_2l = compute_effective_rank(acts_2l)
    r2_1l = compute_fourier_r_squared(acts_1l, modulus=113, model=model_1l)
    r2_2l = compute_fourier_r_squared(acts_2l, modulus=113, model=model_2l)

    print(f"  1-layer: eff_rank={eff_rank_1l:.1f}, Fourier R^2={r2_1l:.4f}")
    print(f"  2-layer: eff_rank={eff_rank_2l:.1f}, Fourier R^2={r2_2l:.4f}")

    manifest["activation_stats"] = {
        "1layer": {
            "eff_rank": eff_rank_1l,
            "fourier_r2": r2_1l,
            "n_samples": acts_1l.shape[0],
            "d_model": acts_1l.shape[1],
        },
        "2layer": {
            "eff_rank": eff_rank_2l,
            "fourier_r2": r2_2l,
            "n_samples": acts_2l.shape[0],
            "d_model": acts_2l.shape[1],
        },
    }

    # ── Phase 5: Train SAEs and measure stability ──
    print("\nPhase 5: Training SAEs and measuring stability...")

    # Test at the standard d_sae=1024 (overparameterized for both)
    d_sae = 1024
    k = 32

    results_1l = run_stability_analysis(
        acts_1l, d_sae=d_sae, k=k, seeds=SEEDS, sae_epochs=30,
        label="1-layer (Fourier ground truth)",
    )
    results_2l = run_stability_analysis(
        acts_2l, d_sae=d_sae, k=k, seeds=SEEDS, sae_epochs=30,
        label="2-layer (non-Fourier)",
    )

    # Also test at d_sae matched to effective rank (the regime where
    # identifiability theory predicts stability should hold)
    print("\nPhase 5b: Matched-regime SAE sweep...")
    matched_sweep = {}
    for model_label, acts, er in [("1layer", acts_1l, eff_rank_1l), ("2layer", acts_2l, eff_rank_2l)]:
        sweep_results = []
        for d_sae_test in [16, 32, 64, 128, 256, 512, 1024]:
            k_test = max(4, min(d_sae_test // 4, 32))
            res = run_stability_analysis(
                acts, d_sae=d_sae_test, k=k_test, seeds=SEEDS, sae_epochs=30,
                label=f"{model_label} d_sae={d_sae_test}",
            )
            res["d_sae_over_eff_rank"] = d_sae_test / er
            sweep_results.append(res)
            print(f"    {model_label} d_sae={d_sae_test}: ratio={d_sae_test/er:.1f}, PWMCC={res['pwmcc_mean']:.4f}, random={res['random_baseline']:.4f}")
        matched_sweep[model_label] = sweep_results

    manifest["sae_config"] = {"d_sae": d_sae, "k": k, "sae_epochs": 30}
    manifest["results"] = {"1layer": results_1l, "2layer": results_2l}
    manifest["matched_sweep"] = matched_sweep
    manifest["completed"] = utc_now()

    # ── Phase 6: Report results ──
    print("\n" + "=" * 70)
    print("RESULTS: 1-Layer vs 2-Layer Ground Truth Comparison")
    print("=" * 70)

    for label, res, r2, er in [
        ("1-layer", results_1l, r2_1l, eff_rank_1l),
        ("2-layer", results_2l, r2_2l, eff_rank_2l),
    ]:
        print(f"\n{label} Transformer:")
        print(f"  Fourier R^2:         {r2:.4f}")
        print(f"  Effective rank:      {er:.1f}")
        print(f"  PWMCC:               {res['pwmcc_mean']:.4f} +/- {res['pwmcc_std']:.4f}")
        print(f"  Random baseline:     {res['random_baseline']:.4f}")
        print(f"  PWMCC/random:        {res['pwmcc_over_random']:.2f}x")
        print(f"  Fourier overlap:     {res['fourier_overlap_mean']:.4f} +/- {res['fourier_overlap_std']:.4f}")
        print(f"  Recon loss:          {res['recon_loss_mean']:.6f}")

    print("\n" + "-" * 70)
    print("KEY COMPARISON:")
    delta_pwmcc = results_1l["pwmcc_mean"] - results_2l["pwmcc_mean"]
    delta_fourier = results_1l["fourier_overlap_mean"] - results_2l["fourier_overlap_mean"]
    print(f"  PWMCC difference (1L - 2L): {delta_pwmcc:+.4f}")
    print(f"  Fourier overlap diff:       {delta_fourier:+.4f}")

    if results_1l["pwmcc_mean"] > results_2l["pwmcc_mean"] + 0.05:
        print("  CONFIRMED: 1-layer (sparse GT) shows higher SAE stability")
    else:
        print("  UNEXPECTED: No significant stability difference")

    if r2_1l > 0.5 and r2_2l < 0.1:
        print("  CONFIRMED: 1-layer learns Fourier, 2-layer does not")
    print("=" * 70)

    # ── Phase 7: Generate figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: PWMCC comparison
        ax = axes[0]
        labels = ["1-layer\n(Fourier GT)", "2-layer\n(non-Fourier)"]
        means = [results_1l["pwmcc_mean"], results_2l["pwmcc_mean"]]
        stds = [results_1l["pwmcc_std"], results_2l["pwmcc_std"]]
        baselines = [results_1l["random_baseline"], results_2l["random_baseline"]]

        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, width=0.4, color=["#2ecc71", "#e74c3c"],
                       capsize=5, label="Trained SAE PWMCC")
        ax.bar(x + 0.45, baselines, width=0.35, color=["#27ae60", "#c0392b"],
               alpha=0.3, label="Random baseline")
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("PWMCC")
        ax.set_title("SAE Feature Stability")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Panel 2: Fourier overlap
        ax = axes[1]
        fo_means = [results_1l["fourier_overlap_mean"], results_2l["fourier_overlap_mean"]]
        fo_stds = [results_1l["fourier_overlap_std"], results_2l["fourier_overlap_std"]]
        ax.bar(x, fo_means, yerr=fo_stds, width=0.5, color=["#2ecc71", "#e74c3c"], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fourier Overlap")
        ax.set_title("Ground Truth Recovery")
        ax.grid(axis="y", alpha=0.3)

        # Panel 3: Effective rank vs R^2
        ax = axes[2]
        ax.scatter([eff_rank_1l], [r2_1l], s=200, c="#2ecc71", marker="o",
                   label="1-layer", zorder=5, edgecolors="black")
        ax.scatter([eff_rank_2l], [r2_2l], s=200, c="#e74c3c", marker="s",
                   label="2-layer", zorder=5, edgecolors="black")
        ax.set_xlabel("Effective Rank")
        ax.set_ylabel("Fourier R^2")
        ax.set_title("Activation Structure")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_1layer_ground_truth.pdf"
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
