#!/usr/bin/env python3
"""Experiment 2.1: Intervention Stability -- Does Feature Instability Matter?

The paper argues instability is bad for interpretability, but doesn't test
whether *practical applications* of SAE features (steering, circuit analysis)
break across seeds. This is the "so what?" experiment.

Design:
  - Pick top features from SAE seed A (highest activation magnitude).
  - Use those feature directions for activation steering on the transformer.
  - Measure behavioral effect on model outputs.
  - Repeat with SAE seed B, C, etc.
  - Question: do you get the same behavioral effect from "equivalent" features?

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/exp_intervention_stability.py
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

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "intervention_stability"
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


def find_best_match(
    feature_idx: int,
    decoder_source: torch.Tensor,
    decoder_target: torch.Tensor,
) -> Tuple[int, float]:
    """Find the best matching feature in target decoder for a source feature."""
    src_col = F.normalize(decoder_source[:, feature_idx : feature_idx + 1], dim=0)
    tgt_norm = F.normalize(decoder_target, dim=0)
    cos_sim = (tgt_norm.T @ src_col).squeeze()  # [d_sae_target]
    best_idx = cos_sim.abs().argmax().item()
    best_sim = cos_sim[best_idx].item()
    return best_idx, best_sim


def load_transformer() -> ModularArithmeticTransformer:
    """Load the trained 2-layer transformer."""
    model_path = PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = TransformerConfig(**checkpoint["config"])
    model = ModularArithmeticTransformer(config, device=DEVICE)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_activations(model: ModularArithmeticTransformer) -> torch.Tensor:
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    acts = []
    with torch.no_grad():
        for batch, _ in dataloader:
            a = model.get_activations(batch, layer=1)[:, -2, :]
            acts.append(a)
    return torch.cat(acts, dim=0)


def train_sae(
    activations: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 30,
) -> TopKSAE:
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


def steer_and_measure(
    model: ModularArithmeticTransformer,
    steering_direction: torch.Tensor,
    steering_magnitude: float,
    layer: int,
    eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
    modulus: int = 113,
) -> Dict:
    """Apply activation steering and measure behavioral effects.

    Adds steering_direction * magnitude to the residual stream at the
    given layer and measures:
    - KL divergence from unsteered output
    - Accuracy change
    - Output distribution shift
    """
    model.eval()

    # Collect unsteered baseline
    baseline_logits_all = []
    steered_logits_all = []
    baseline_correct = 0
    steered_correct = 0
    total = 0

    for tokens, labels in eval_data:
        tokens = tokens.to(DEVICE)
        labels = labels.to(DEVICE)

        # Baseline (no steering)
        with torch.no_grad():
            baseline_logits = model(tokens)[:, -2, :modulus]
            baseline_preds = baseline_logits.argmax(dim=-1)
            baseline_correct += (baseline_preds == labels).sum().item()

        # Steered: add direction to residual stream via hook
        steering_vec = steering_direction.to(DEVICE) * steering_magnitude

        def hook_fn(module, input, output):
            # output: [batch, seq, d_model]
            modified = output.clone()
            modified[:, -2, :] = modified[:, -2, :] + steering_vec
            return modified

        hook_name = f"blocks.{layer}.hook_resid_post"
        handle = model.model.hook_dict[hook_name].register_forward_hook(hook_fn)

        with torch.no_grad():
            steered_logits = model(tokens)[:, -2, :modulus]
            steered_preds = steered_logits.argmax(dim=-1)
            steered_correct += (steered_preds == labels).sum().item()

        handle.remove()

        baseline_logits_all.append(baseline_logits)
        steered_logits_all.append(steered_logits)
        total += len(labels)

    baseline_logits_cat = torch.cat(baseline_logits_all, dim=0)
    steered_logits_cat = torch.cat(steered_logits_all, dim=0)

    # KL divergence
    bl_probs = F.softmax(baseline_logits_cat, dim=-1)
    st_probs = F.softmax(steered_logits_cat, dim=-1)
    kl_div = F.kl_div(
        st_probs.log().clamp(min=-100), bl_probs, reduction="batchmean"
    ).item()

    # Cosine similarity of output distributions
    cos_sim = F.cosine_similarity(
        bl_probs.mean(dim=0, keepdim=True),
        st_probs.mean(dim=0, keepdim=True),
    ).item()

    return {
        "baseline_acc": baseline_correct / total if total > 0 else 0,
        "steered_acc": steered_correct / total if total > 0 else 0,
        "acc_change": (steered_correct - baseline_correct) / total if total > 0 else 0,
        "kl_divergence": kl_div,
        "output_cosine_sim": cos_sim,
    }


def main():
    run_dir = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2.1: Intervention Stability")
    print("=" * 70)

    manifest = {
        "experiment": "intervention_stability",
        "started": utc_now(),
        "seeds": SEEDS,
    }

    # ── Load model and data ──
    print("\nLoading transformer and extracting activations...")
    model = load_transformer()
    acts = extract_activations(model)
    print(f"  Activations: {acts.shape}")

    # Prepare eval data
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    eval_data = [(t, l) for t, l in eval_loader]

    # ── Train SAEs ──
    d_sae = 1024
    k = 32
    print(f"\nTraining {len(SEEDS)} SAEs...")
    saes = {}
    for s in SEEDS:
        saes[s] = train_sae(acts, d_sae, k, s, epochs=30)
        print(f"  Seed {s}: done")

    # ── Select top features from reference SAE ──
    ref_seed = SEEDS[0]
    ref_sae = saes[ref_seed]
    n_features_to_test = 10
    steering_magnitudes = [1.0, 2.0, 5.0]

    # Find features with highest mean activation magnitude
    ref_sae.eval()
    with torch.no_grad():
        _, ref_latents, _ = ref_sae(acts[:2000])
    mean_activations = ref_latents.abs().mean(dim=0)  # [d_sae]
    top_features = mean_activations.topk(n_features_to_test).indices.tolist()

    print(f"\nTop {n_features_to_test} features from seed {ref_seed}: {top_features}")
    print(f"Mean activations: {mean_activations[top_features].tolist()}")

    # ── Run interventions ──
    print(f"\nRunning interventions across {len(SEEDS)} seeds...")
    intervention_results = {}

    for feat_rank, feat_idx in enumerate(top_features):
        feat_results = {}

        for seed in SEEDS:
            sae = saes[seed]

            if seed == ref_seed:
                matched_idx = feat_idx
                match_sim = 1.0
            else:
                matched_idx, match_sim = find_best_match(
                    feat_idx, ref_sae.decoder.weight.data, sae.decoder.weight.data,
                )

            # Get the steering direction from this SAE
            steering_dir = sae.decoder.weight.data[:, matched_idx]

            seed_results = {
                "matched_feature": matched_idx,
                "match_similarity": match_sim,
                "interventions": {},
            }

            for mag in steering_magnitudes:
                result = steer_and_measure(
                    model, steering_dir, mag, layer=1, eval_data=eval_data
                )
                seed_results["interventions"][str(mag)] = result

            feat_results[str(seed)] = seed_results

        intervention_results[str(feat_idx)] = feat_results

    manifest["sae_config"] = {"d_sae": d_sae, "k": k}
    manifest["ref_seed"] = ref_seed
    manifest["top_features"] = top_features
    manifest["steering_magnitudes"] = steering_magnitudes
    manifest["intervention_results"] = intervention_results
    manifest["completed"] = utc_now()

    # ── Analyze cross-seed consistency ──
    print("\n" + "=" * 70)
    print("RESULTS: Intervention Stability Across Seeds")
    print("=" * 70)

    consistency_scores = []

    for feat_idx in top_features:
        feat_data = intervention_results[str(feat_idx)]

        for mag in steering_magnitudes:
            # Collect KL divergences and accuracy changes across seeds
            kl_divs = []
            acc_changes = []
            for seed in SEEDS:
                inter = feat_data[str(seed)]["interventions"][str(mag)]
                kl_divs.append(inter["kl_divergence"])
                acc_changes.append(inter["acc_change"])

            # Consistency = 1 - CoV of behavioral effect
            kl_std = np.std(kl_divs)
            kl_mean = np.mean(kl_divs)
            cov = kl_std / max(kl_mean, 1e-10)
            consistency = max(0, 1 - cov)
            consistency_scores.append(consistency)

    mean_consistency = float(np.mean(consistency_scores))
    print(f"\nOverall intervention consistency: {mean_consistency:.4f}")
    print(f"  (1.0 = identical effects across seeds, 0.0 = completely different)")

    # Per-feature summary
    print(f"\nPer-feature summary (magnitude={steering_magnitudes[-1]}):")
    for feat_idx in top_features:
        feat_data = intervention_results[str(feat_idx)]
        mag = str(steering_magnitudes[-1])

        kl_divs = [feat_data[str(s)]["interventions"][mag]["kl_divergence"] for s in SEEDS]
        acc_changes = [feat_data[str(s)]["interventions"][mag]["acc_change"] for s in SEEDS]
        match_sims = [feat_data[str(s)]["match_similarity"] for s in SEEDS]

        print(
            f"  Feature {feat_idx}: "
            f"KL={np.mean(kl_divs):.4f}+/-{np.std(kl_divs):.4f}, "
            f"acc_change={np.mean(acc_changes):+.4f}+/-{np.std(acc_changes):.4f}, "
            f"match_sim={np.mean(match_sims):.3f}"
        )

    if mean_consistency < 0.5:
        print("\n  FINDING: Interventions are UNSTABLE across seeds!")
        print("  Feature-level instability has practical consequences.")
    else:
        print("\n  FINDING: Interventions are relatively STABLE across seeds.")
        print("  The subspace may be stable even if individual features aren't.")
    print("=" * 70)

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: KL divergence across seeds for each feature
        ax = axes[0]
        mag_key = str(steering_magnitudes[-1])
        for i, feat_idx in enumerate(top_features[:5]):
            feat_data = intervention_results[str(feat_idx)]
            kls = [feat_data[str(s)]["interventions"][mag_key]["kl_divergence"] for s in SEEDS]
            ax.scatter([i] * len(SEEDS), kls, alpha=0.7, s=50)
            ax.scatter([i], [np.mean(kls)], marker="D", s=100, c="black", zorder=5)

        ax.set_xticks(range(5))
        ax.set_xticklabels([f"F{idx}" for idx in top_features[:5]])
        ax.set_xlabel("Feature (from reference SAE)")
        ax.set_ylabel(f"KL Divergence (mag={steering_magnitudes[-1]})")
        ax.set_title("Intervention Effect Across Seeds")
        ax.grid(alpha=0.3)

        # Panel 2: Match similarity vs KL consistency
        ax = axes[1]
        for feat_idx in top_features:
            feat_data = intervention_results[str(feat_idx)]
            mean_sim = np.mean([feat_data[str(s)]["match_similarity"] for s in SEEDS])
            kls = [feat_data[str(s)]["interventions"][mag_key]["kl_divergence"] for s in SEEDS]
            kl_cov = np.std(kls) / max(np.mean(kls), 1e-10)
            ax.scatter(mean_sim, kl_cov, s=80, alpha=0.7)

        ax.set_xlabel("Mean Feature Match Similarity")
        ax.set_ylabel("KL Divergence CoV (lower = more consistent)")
        ax.set_title("Match Quality vs Intervention Consistency")
        ax.grid(alpha=0.3)

        # Panel 3: Accuracy change distribution
        ax = axes[2]
        for seed in SEEDS:
            acc_changes = [
                intervention_results[str(f)][str(seed)]["interventions"][mag_key]["acc_change"]
                for f in top_features
            ]
            ax.hist(acc_changes, bins=10, alpha=0.4, label=f"Seed {seed}")

        ax.set_xlabel("Accuracy Change")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Steering Effects")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "exp_intervention_stability.pdf"
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
