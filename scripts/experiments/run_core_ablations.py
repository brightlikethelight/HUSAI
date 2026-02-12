#!/usr/bin/env python3
"""Run core SAE ablations (k sweep and d_sae sweep) with confidence intervals.

This script executes Phase 4c core ablations on a fixed transformer activation
cache, writes reproducible artifacts, and emits machine-readable tables.
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


def pwmcc(decoder_a: torch.Tensor, decoder_b: torch.Tensor) -> float:
    a = F.normalize(decoder_a, dim=0)
    b = F.normalize(decoder_b, dim=0)
    cos = a.T @ b
    max_a = cos.abs().max(dim=1).values.mean().item()
    max_b = cos.abs().max(dim=0).values.mean().item()
    return float((max_a + max_b) / 2)


def pairwise_pwmcc(decoders: dict[int, torch.Tensor]) -> list[float]:
    values: list[float] = []
    for seed_a, seed_b in itertools.combinations(sorted(decoders.keys()), 2):
        values.append(pwmcc(decoders[seed_a], decoders[seed_b]))
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


def train_topk_sae(
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
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=min(k, d_sae)).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    sae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, latents, _ = sae(batch, compute_aux_loss=False)
            mse = F.mse_loss(recon, batch)
            loss = mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

    sae.eval()
    with torch.no_grad():
        full = activations.to(device)
        recon, latents, _ = sae(full, compute_aux_loss=False)
        mse = float(F.mse_loss(recon, full).item())
        l0 = float((latents != 0).float().sum(dim=-1).mean().item())
        total_var = torch.var(full)
        residual_var = torch.var(full - recon)
        explained_var = float((1 - residual_var / total_var).item())

    metrics = {"mse": mse, "l0": l0, "explained_variance": explained_var}
    return sae.cpu(), metrics


def random_decoders(d_model: int, d_sae: int, seeds: list[int]) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for seed in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 100000)
        out[seed] = torch.randn((d_model, d_sae), generator=g)
    return out


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
        return torch.load(cache_path, map_location="cpu")

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
    torch.save(acts.cpu(), cache_path)
    return acts.cpu()


def run_condition_grid(
    grid_name: str,
    conditions: list[dict[str, int]],
    activations: torch.Tensor,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    output_dir: Path,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for cond in conditions:
        d_sae = cond["d_sae"]
        k = cond["k"]
        condition_id = f"{grid_name}_dsae{d_sae}_k{k}"
        print(f"\n[{grid_name}] Running condition {condition_id}")

        per_seed_metrics: list[dict[str, Any]] = []
        decoders: dict[int, torch.Tensor] = {}
        condition_dir = output_dir / "checkpoints" / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for seed in seeds:
            print(f"  - seed {seed}")
            sae, metrics = train_topk_sae(
                activations=activations,
                d_sae=d_sae,
                k=k,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
            )

            checkpoint_path = condition_dir / f"sae_seed{seed}.pt"
            torch.save(
                {
                    "model_state_dict": sae.state_dict(),
                    "d_model": sae.d_model,
                    "d_sae": sae.d_sae,
                    "k": sae.k,
                },
                checkpoint_path,
            )

            decoder = sae.decoder.weight.detach().float().cpu()
            if decoder.shape[0] > decoder.shape[1]:
                decoder = decoder.T
            decoders[seed] = decoder

            per_seed_metrics.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                    **metrics,
                }
            )

        trained_values = pairwise_pwmcc(decoders)
        random_values = pairwise_pwmcc(random_decoders(activations.shape[1], d_sae, seeds))

        trained_summary = summarize(trained_values, n_bootstrap=bootstrap_samples, seed=11)
        random_summary = summarize(random_values, n_bootstrap=bootstrap_samples, seed=23)
        mse_summary = summarize([m["mse"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=37)
        ev_summary = summarize([m["explained_variance"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=41)
        l0_summary = summarize([m["l0"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=43)

        runtime_sec = time.time() - t0
        records.append(
            {
                "grid": grid_name,
                "d_sae": d_sae,
                "k": k,
                "n_seeds": len(seeds),
                "trained_pwmcc": trained_summary,
                "random_pwmcc": random_summary,
                "delta_pwmcc": trained_summary["mean"] - random_summary["mean"],
                "ratio_pwmcc": (
                    trained_summary["mean"] / random_summary["mean"] if random_summary["mean"] > 0 else float("nan")
                ),
                "mse": mse_summary,
                "explained_variance": ev_summary,
                "l0": l0_summary,
                "runtime_sec": runtime_sec,
                "per_seed_metrics": per_seed_metrics,
                "trained_pairwise_pwmcc_values": trained_values,
                "random_pairwise_pwmcc_values": random_values,
            }
        )

    return records


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid",
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
        "ratio_pwmcc",
        "mse_mean",
        "mse_ci95_low",
        "mse_ci95_high",
        "explained_variance_mean",
        "explained_variance_ci95_low",
        "explained_variance_ci95_high",
        "l0_mean",
        "l0_ci95_low",
        "l0_ci95_high",
        "runtime_sec",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "grid": rec["grid"],
                    "d_sae": rec["d_sae"],
                    "k": rec["k"],
                    "n_seeds": rec["n_seeds"],
                    "trained_pwmcc_mean": rec["trained_pwmcc"]["mean"],
                    "trained_pwmcc_ci95_low": rec["trained_pwmcc"]["ci95_low"],
                    "trained_pwmcc_ci95_high": rec["trained_pwmcc"]["ci95_high"],
                    "random_pwmcc_mean": rec["random_pwmcc"]["mean"],
                    "random_pwmcc_ci95_low": rec["random_pwmcc"]["ci95_low"],
                    "random_pwmcc_ci95_high": rec["random_pwmcc"]["ci95_high"],
                    "delta_pwmcc": rec["delta_pwmcc"],
                    "ratio_pwmcc": rec["ratio_pwmcc"],
                    "mse_mean": rec["mse"]["mean"],
                    "mse_ci95_low": rec["mse"]["ci95_low"],
                    "mse_ci95_high": rec["mse"]["ci95_high"],
                    "explained_variance_mean": rec["explained_variance"]["mean"],
                    "explained_variance_ci95_low": rec["explained_variance"]["ci95_low"],
                    "explained_variance_ci95_high": rec["explained_variance"]["ci95_high"],
                    "l0_mean": rec["l0"]["mean"],
                    "l0_ci95_low": rec["l0"]["ci95_low"],
                    "l0_ci95_high": rec["l0"]["ci95_high"],
                    "runtime_sec": rec["runtime_sec"],
                }
            )


def write_markdown_table(path: Path, records: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| d_sae | k | trained PWMCC (95% CI) | random PWMCC (95% CI) | delta | ratio | EV | MSE | L0 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rec in records:
        t = rec["trained_pwmcc"]
        r = rec["random_pwmcc"]
        ev = rec["explained_variance"]
        mse = rec["mse"]
        l0 = rec["l0"]
        lines.append(
            "| "
            f"{rec['d_sae']} | {rec['k']} | "
            f"{t['mean']:.6f} [{t['ci95_low']:.6f}, {t['ci95_high']:.6f}] | "
            f"{r['mean']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | "
            f"{rec['delta_pwmcc']:.6f} | {rec['ratio_pwmcc']:.4f} | "
            f"{ev['mean']:.4f} | {mse['mean']:.6f} | {l0['mean']:.2f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core ablations with CIs")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="42,123,456")

    parser.add_argument("--k-values", type=str, default="8,16,32,64")
    parser.add_argument("--dsae-values", type=str, default="64,128,256,512")
    parser.add_argument("--fixed-k", type=int, default=32)
    parser.add_argument("--fixed-dsae", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4c_core_ablations",
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    k_values = parse_int_list(args.k_values)
    dsae_values = parse_int_list(args.dsae_values)

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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seeds": seeds,
        "k_values": k_values,
        "dsae_values": dsae_values,
        "fixed_k": args.fixed_k,
        "fixed_dsae": args.fixed_dsae,
        "bootstrap_samples": args.bootstrap_samples,
    }

    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    print("Loading or extracting activations...")
    activations = load_or_extract_activations(
        cache_path=args.activations_cache,
        transformer_checkpoint=args.transformer_checkpoint,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        modulus=args.modulus,
        seed=args.activation_seed,
    )
    activations = activations.float().cpu()
    print(f"Activation shape: {list(activations.shape)}")

    k_conditions = [{"d_sae": args.fixed_dsae, "k": k} for k in k_values]
    d_conditions = [{"d_sae": d, "k": args.fixed_k} for d in dsae_values]

    k_records = run_condition_grid(
        grid_name="k_sweep",
        conditions=k_conditions,
        activations=activations,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=run_dir,
        bootstrap_samples=args.bootstrap_samples,
    )

    d_records = run_condition_grid(
        grid_name="d_sae_sweep",
        conditions=d_conditions,
        activations=activations,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=run_dir,
        bootstrap_samples=args.bootstrap_samples,
    )

    write_summary_csv(run_dir / "k_sweep_summary.csv", k_records)
    write_summary_csv(run_dir / "d_sae_sweep_summary.csv", d_records)
    write_markdown_table(run_dir / "k_sweep_summary.md", k_records, "Core Ablation: K Sweep")
    write_markdown_table(run_dir / "d_sae_sweep_summary.md", d_records, "Core Ablation: d_sae Sweep")

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
            "shape": list(activations.shape),
            "mean": float(activations.mean().item()),
            "std": float(activations.std().item()),
        },
        "k_sweep": k_records,
        "d_sae_sweep": d_records,
    }

    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    manifest = {
        "run_metadata": payload["run_metadata"],
        "artifacts": [
            str((run_dir / "config.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "results.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "k_sweep_summary.csv").relative_to(PROJECT_ROOT)),
            str((run_dir / "d_sae_sweep_summary.csv").relative_to(PROJECT_ROOT)),
            str((run_dir / "k_sweep_summary.md").relative_to(PROJECT_ROOT)),
            str((run_dir / "d_sae_sweep_summary.md").relative_to(PROJECT_ROOT)),
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Core ablations complete")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
