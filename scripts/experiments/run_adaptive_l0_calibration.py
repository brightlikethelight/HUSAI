#!/usr/bin/env python3
"""Adaptive L0 calibration for TopK SAE with retrain/evaluate cycle.

Protocol:
1) Search over k candidates at fixed d_sae using multi-seed training.
2) Compute trained-vs-random PWMCC with CIs and reconstruction metrics.
3) Select k via conservative objective (delta PWMCC lower CI bound) with EV floor.
4) Retrain selected k on an expanded seed set for final estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.extract_activations import extract_activations
from src.models.simple_sae import TopKSAE


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_or_extract_activations(
    cache_path: Path,
    transformer_checkpoint: Path,
    layer: int,
    batch_size: int,
    device: str,
    modulus: int,
    seed: int,
) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu").float()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    acts = extract_activations(
        model_path=transformer_checkpoint,
        layer=layer,
        position="answer",
        batch_size=batch_size,
        device=device,
        modulus=modulus,
        seed=seed,
    )
    acts = acts.cpu().float()
    torch.save(acts, cache_path)
    return acts


def pwmcc(decoder_a: torch.Tensor, decoder_b: torch.Tensor) -> float:
    a = F.normalize(decoder_a, dim=0)
    b = F.normalize(decoder_b, dim=0)
    cos = a.T @ b
    max_a = cos.abs().max(dim=1).values.mean().item()
    max_b = cos.abs().max(dim=0).values.mean().item()
    return float((max_a + max_b) / 2)


def pairwise_pwmcc(decoders: dict[int, torch.Tensor]) -> list[float]:
    values: list[float] = []
    for sa, sb in itertools.combinations(sorted(decoders.keys()), 2):
        values.append(pwmcc(decoders[sa], decoders[sb]))
    return values


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(values: list[float], n_bootstrap: int = 10000, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "ci95_low": lo,
        "ci95_high": hi,
        "n": int(arr.size),
    }


def random_decoders(d_model: int, d_sae: int, seeds: list[int]) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for seed in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 100000)
        out[seed] = torch.randn((d_model, d_sae), generator=g)
    return out


def train_topk(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> tuple[TopKSAE, dict[str, float]]:
    set_seed(seed)

    d_model = int(activations.shape[1])
    model = TopKSAE(d_model=d_model, d_sae=d_sae, k=min(k, d_sae)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loader = DataLoader(
        TensorDataset(activations),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, latents, _ = model(batch, compute_aux_loss=False)
            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

    model.eval()
    with torch.no_grad():
        full = activations.to(device)
        recon, latents, _ = model(full, compute_aux_loss=False)
        mse = float(F.mse_loss(recon, full).item())
        total_var = torch.var(full)
        residual_var = torch.var(full - recon)
        explained_var = float((1 - residual_var / total_var).item())
        l0 = float((latents != 0).float().sum(dim=-1).mean().item())

    return model.cpu(), {"mse": mse, "explained_variance": explained_var, "l0": l0}


def run_condition(
    condition_name: str,
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    bootstrap_samples: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: list[dict[str, Any]] = []
    decoders: dict[int, torch.Tensor] = {}

    start = time.time()
    for seed in seeds:
        model, metrics = train_topk(
            activations=activations,
            d_sae=d_sae,
            k=k,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        ckpt_path = checkpoint_dir / f"sae_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "d_model": model.d_model,
                "d_sae": model.d_sae,
                "k": model.k,
                "condition": condition_name,
            },
            ckpt_path,
        )

        decoder = model.decoder.weight.detach().float().cpu()
        if decoder.shape[0] > decoder.shape[1]:
            decoder = decoder.T
        decoders[seed] = decoder

        per_seed_metrics.append({
            "seed": seed,
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            **metrics,
        })

    trained_vals = pairwise_pwmcc(decoders)
    random_vals = pairwise_pwmcc(random_decoders(activations.shape[1], d_sae, seeds))

    trained = summarize(trained_vals, n_bootstrap=bootstrap_samples, seed=11)
    random = summarize(random_vals, n_bootstrap=bootstrap_samples, seed=23)
    ev = summarize([m["explained_variance"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=31)
    mse = summarize([m["mse"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=37)

    delta = trained["mean"] - random["mean"]
    delta_ci_low = trained["ci95_low"] - random["ci95_high"]
    ratio = trained["mean"] / random["mean"] if random["mean"] > 0 else float("nan")

    return {
        "condition": condition_name,
        "d_sae": d_sae,
        "k": k,
        "n_seeds": len(seeds),
        "trained_pwmcc": trained,
        "random_pwmcc": random,
        "delta_pwmcc": delta,
        "delta_pwmcc_ci_low_conservative": delta_ci_low,
        "ratio_pwmcc": ratio,
        "explained_variance": ev,
        "mse": mse,
        "runtime_sec": time.time() - start,
        "per_seed_metrics": per_seed_metrics,
        "trained_pairwise_pwmcc_values": trained_vals,
        "random_pairwise_pwmcc_values": random_vals,
    }


def pick_best(search_records: list[dict[str, Any]], ev_floor: float) -> dict[str, Any]:
    eligible = [r for r in search_records if r["explained_variance"]["mean"] >= ev_floor]
    if not eligible:
        eligible = search_records

    return max(
        eligible,
        key=lambda r: (r["delta_pwmcc_ci_low_conservative"], r["delta_pwmcc"], r["explained_variance"]["mean"]),
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "d_sae",
                "k",
                "n_seeds",
                "trained_pwmcc_mean",
                "trained_pwmcc_ci95_low",
                "trained_pwmcc_ci95_high",
                "random_pwmcc_mean",
                "random_pwmcc_ci95_low",
                "random_pwmcc_ci95_high",
                "delta_pwmcc",
                "delta_pwmcc_ci_low_conservative",
                "ratio_pwmcc",
                "explained_variance_mean",
                "explained_variance_ci95_low",
                "explained_variance_ci95_high",
                "mse_mean",
                "mse_ci95_low",
                "mse_ci95_high",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "condition": r["condition"],
                    "d_sae": r["d_sae"],
                    "k": r["k"],
                    "n_seeds": r["n_seeds"],
                    "trained_pwmcc_mean": r["trained_pwmcc"]["mean"],
                    "trained_pwmcc_ci95_low": r["trained_pwmcc"]["ci95_low"],
                    "trained_pwmcc_ci95_high": r["trained_pwmcc"]["ci95_high"],
                    "random_pwmcc_mean": r["random_pwmcc"]["mean"],
                    "random_pwmcc_ci95_low": r["random_pwmcc"]["ci95_low"],
                    "random_pwmcc_ci95_high": r["random_pwmcc"]["ci95_high"],
                    "delta_pwmcc": r["delta_pwmcc"],
                    "delta_pwmcc_ci_low_conservative": r["delta_pwmcc_ci_low_conservative"],
                    "ratio_pwmcc": r["ratio_pwmcc"],
                    "explained_variance_mean": r["explained_variance"]["mean"],
                    "explained_variance_ci95_low": r["explained_variance"]["ci95_low"],
                    "explained_variance_ci95_high": r["explained_variance"]["ci95_high"],
                    "mse_mean": r["mse"]["mean"],
                    "mse_ci95_low": r["mse"]["ci95_low"],
                    "mse_ci95_high": r["mse"]["ci95_high"],
                    "runtime_sec": r["runtime_sec"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive L0 calibration and retrain")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument(
        "--activations-cache",
        type=Path,
        default=PROJECT_ROOT / "results" / "activations" / "layer1_answer.pt",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--activation-seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-sae", type=int, default=128)
    parser.add_argument("--k-candidates", type=str, default="4,8,12,16,24,32,48,64")
    parser.add_argument("--search-seeds", type=str, default="42,123,456,789,1011")
    parser.add_argument("--retrain-seeds", type=str, default="42,123,456,789,1011,2022,2023,2024")
    parser.add_argument("--search-epochs", type=int, default=25)
    parser.add_argument("--retrain-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ev-floor", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "adaptive_l0_calibration",
    )
    args = parser.parse_args()

    k_candidates = parse_int_list(args.k_candidates)
    search_seeds = parse_int_list(args.search_seeds)
    retrain_seeds = parse_int_list(args.retrain_seeds)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "activations_cache": str(args.activations_cache),
        "layer": args.layer,
        "modulus": args.modulus,
        "activation_seed": args.activation_seed,
        "device": args.device,
        "d_sae": args.d_sae,
        "k_candidates": k_candidates,
        "search_seeds": search_seeds,
        "retrain_seeds": retrain_seeds,
        "search_epochs": args.search_epochs,
        "retrain_epochs": args.retrain_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "ev_floor": args.ev_floor,
        "bootstrap_samples": args.bootstrap_samples,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    acts = load_or_extract_activations(
        cache_path=args.activations_cache,
        transformer_checkpoint=args.transformer_checkpoint,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        modulus=args.modulus,
        seed=args.activation_seed,
    )

    search_records: list[dict[str, Any]] = []
    for k in k_candidates:
        print(f"[search] d_sae={args.d_sae}, k={k}")
        record = run_condition(
            condition_name=f"search_dsae{args.d_sae}_k{k}",
            activations=acts,
            d_sae=args.d_sae,
            k=k,
            seeds=search_seeds,
            epochs=args.search_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            bootstrap_samples=args.bootstrap_samples,
            checkpoint_dir=run_dir / "checkpoints" / f"search_k{k}",
        )
        search_records.append(record)

    best = pick_best(search_records, ev_floor=args.ev_floor)
    best_k = int(best["k"])
    print(f"[selection] best k = {best_k}")

    retrain_record = run_condition(
        condition_name=f"retrain_dsae{args.d_sae}_k{best_k}",
        activations=acts,
        d_sae=args.d_sae,
        k=best_k,
        seeds=retrain_seeds,
        epochs=args.retrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        checkpoint_dir=run_dir / "checkpoints" / f"retrain_k{best_k}",
    )

    write_csv(run_dir / "search_summary.csv", search_records)
    write_csv(run_dir / "retrain_summary.csv", [retrain_record])

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "config_hash": stable_hash(config_payload),
            "run_id": run_id,
        },
        "config": config_payload,
        "activation_stats": {
            "shape": list(acts.shape),
            "mean": float(acts.mean().item()),
            "std": float(acts.std().item()),
        },
        "search_records": search_records,
        "selection": {
            "criterion": "max delta_pwmcc_ci_low_conservative with EV floor",
            "ev_floor": args.ev_floor,
            "selected_k": best_k,
            "selected_record": best,
        },
        "retrain_record": retrain_record,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Adaptive L0 Calibration",
        "",
        f"- Run ID: `{run_id}`",
        f"- Git commit: `{payload['run_metadata']['git_commit']}`",
        f"- Selected k: `{best_k}`",
        "",
        "## Search Table",
        "",
        "| k | trained PWMCC (95% CI) | random PWMCC (95% CI) | delta | conservative delta LCB | EV | MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in sorted(search_records, key=lambda x: x["k"]):
        t = r["trained_pwmcc"]
        rb = r["random_pwmcc"]
        lines.append(
            "| "
            f"{r['k']} | {t['mean']:.6f} [{t['ci95_low']:.6f}, {t['ci95_high']:.6f}] | "
            f"{rb['mean']:.6f} [{rb['ci95_low']:.6f}, {rb['ci95_high']:.6f}] | "
            f"{r['delta_pwmcc']:.6f} | {r['delta_pwmcc_ci_low_conservative']:.6f} | "
            f"{r['explained_variance']['mean']:.4f} | {r['mse']['mean']:.6f} |"
        )

    rt = retrain_record
    rtt = rt["trained_pwmcc"]
    rtr = rt["random_pwmcc"]
    lines.extend(
        [
            "",
            "## Retrain at Selected k",
            "",
            "| condition | trained PWMCC (95% CI) | random PWMCC (95% CI) | delta | ratio | EV | MSE |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| k={best_k} | {rtt['mean']:.6f} [{rtt['ci95_low']:.6f}, {rtt['ci95_high']:.6f}] | "
                f"{rtr['mean']:.6f} [{rtr['ci95_low']:.6f}, {rtr['ci95_high']:.6f}] | "
                f"{rt['delta_pwmcc']:.6f} | {rt['ratio_pwmcc']:.4f} | "
                f"{rt['explained_variance']['mean']:.4f} | {rt['mse']['mean']:.6f} |"
            ),
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "run_metadata": payload["run_metadata"],
        "artifacts": [
            str((run_dir / "config.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "results.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "summary.md").relative_to(PROJECT_ROOT)),
            str((run_dir / "search_summary.csv").relative_to(PROJECT_ROOT)),
            str((run_dir / "retrain_summary.csv").relative_to(PROJECT_ROOT)),
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Adaptive L0 calibration complete")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
