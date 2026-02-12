#!/usr/bin/env python3
"""Consistency-first objective sweep via decoder alignment regularization.

Protocol:
1) Train a reference SAE (seed_ref) with standard MSE objective.
2) For each regularization coefficient lambda_consistency, train additional
   SAEs with loss = MSE + lambda * alignment_penalty(decoder, ref_decoder).
3) Evaluate cross-seed PWMCC, trained-vs-random gap, alignment-to-reference,
   and reconstruction metrics.
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


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def alignment_score(decoder: torch.Tensor, ref_decoder: torch.Tensor) -> float:
    d = F.normalize(decoder, dim=0)
    r = F.normalize(ref_decoder, dim=0)
    cos = d.T @ r
    return float(cos.abs().max(dim=1).values.mean().item())


def alignment_penalty(decoder: torch.Tensor, ref_decoder: torch.Tensor) -> torch.Tensor:
    d = F.normalize(decoder, dim=0)
    r = F.normalize(ref_decoder, dim=0)
    cos = d.T @ r
    return 1.0 - cos.abs().max(dim=1).values.mean()


def train_topk(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    lambda_consistency: float,
    ref_decoder: torch.Tensor | None,
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

    ref_decoder_device = None
    if ref_decoder is not None:
        ref_decoder_device = ref_decoder.to(device)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, latents, _ = model(batch, compute_aux_loss=False)
            mse = F.mse_loss(recon, batch)

            if ref_decoder_device is not None and lambda_consistency > 0:
                reg = alignment_penalty(model.decoder.weight, ref_decoder_device)
                loss = mse + lambda_consistency * reg
            else:
                loss = mse

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

    metrics = {"mse": mse, "explained_variance": explained_var, "l0": l0}
    return model.cpu(), metrics


def run_lambda_condition(
    lambda_consistency: float,
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed_ref: int,
    train_seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    bootstrap_samples: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Reference model always trained without consistency regularization.
    ref_model, ref_metrics = train_topk(
        activations=activations,
        d_sae=d_sae,
        k=k,
        seed=seed_ref,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        lambda_consistency=0.0,
        ref_decoder=None,
    )

    ref_decoder = ref_model.decoder.weight.detach().float().cpu()
    if ref_decoder.shape[0] > ref_decoder.shape[1]:
        ref_decoder = ref_decoder.T

    ref_ckpt = checkpoint_dir / f"sae_seed{seed_ref}.pt"
    torch.save(
        {
            "model_state_dict": ref_model.state_dict(),
            "d_model": ref_model.d_model,
            "d_sae": ref_model.d_sae,
            "k": ref_model.k,
            "seed": seed_ref,
            "lambda_consistency": 0.0,
        },
        ref_ckpt,
    )

    decoders: dict[int, torch.Tensor] = {seed_ref: ref_decoder}
    per_seed_metrics: list[dict[str, Any]] = [
        {
            "seed": seed_ref,
            "checkpoint": str(ref_ckpt.relative_to(PROJECT_ROOT)),
            "lambda_consistency": 0.0,
            "alignment_to_ref": 1.0,
            **ref_metrics,
        }
    ]

    start = time.time()
    for seed in train_seeds:
        model, metrics = train_topk(
            activations=activations,
            d_sae=d_sae,
            k=k,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            lambda_consistency=lambda_consistency,
            ref_decoder=ref_decoder,
        )

        decoder = model.decoder.weight.detach().float().cpu()
        if decoder.shape[0] > decoder.shape[1]:
            decoder = decoder.T

        ckpt = checkpoint_dir / f"sae_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "d_model": model.d_model,
                "d_sae": model.d_sae,
                "k": model.k,
                "seed": seed,
                "lambda_consistency": lambda_consistency,
            },
            ckpt,
        )

        decoders[seed] = decoder
        per_seed_metrics.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                "lambda_consistency": lambda_consistency,
                "alignment_to_ref": alignment_score(decoder, ref_decoder),
                **metrics,
            }
        )

    seeds_all = [seed_ref] + list(train_seeds)
    trained_vals = pairwise_pwmcc(decoders)
    random_vals = pairwise_pwmcc(random_decoders(activations.shape[1], d_sae, seeds_all))

    trained = summarize(trained_vals, n_bootstrap=bootstrap_samples, seed=13)
    random = summarize(random_vals, n_bootstrap=bootstrap_samples, seed=29)
    ev = summarize([m["explained_variance"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=31)
    mse = summarize([m["mse"] for m in per_seed_metrics], n_bootstrap=bootstrap_samples, seed=37)
    align_values = [m["alignment_to_ref"] for m in per_seed_metrics if m["seed"] != seed_ref]
    align = summarize(
        align_values,
        n_bootstrap=bootstrap_samples,
        seed=43,
    )

    delta = trained["mean"] - random["mean"]
    delta_ci_low = trained["ci95_low"] - random["ci95_high"]
    ratio = trained["mean"] / random["mean"] if random["mean"] > 0 else float("nan")

    return {
        "lambda_consistency": lambda_consistency,
        "d_sae": d_sae,
        "k": k,
        "n_models": len(seeds_all),
        "seed_ref": seed_ref,
        "train_seeds": train_seeds,
        "trained_pwmcc": trained,
        "random_pwmcc": random,
        "delta_pwmcc": delta,
        "delta_pwmcc_ci_low_conservative": delta_ci_low,
        "ratio_pwmcc": ratio,
        "explained_variance": ev,
        "mse": mse,
        "alignment_to_ref": align,
        "runtime_sec": time.time() - start,
        "per_seed_metrics": per_seed_metrics,
        "trained_pairwise_pwmcc_values": trained_vals,
        "random_pairwise_pwmcc_values": random_vals,
    }


def choose_lambda(records: list[dict[str, Any]], max_ev_drop: float) -> dict[str, Any]:
    baseline = None
    for r in records:
        if abs(r["lambda_consistency"]) < 1e-12:
            baseline = r
            break

    if baseline is None:
        baseline = min(records, key=lambda x: abs(x["lambda_consistency"]))

    ev_base = baseline["explained_variance"]["mean"]
    eligible = [r for r in records if (ev_base - r["explained_variance"]["mean"]) <= max_ev_drop]
    if not eligible:
        eligible = records

    return max(
        eligible,
        key=lambda r: (r["delta_pwmcc_ci_low_conservative"], r["delta_pwmcc"], r["alignment_to_ref"]["mean"]),
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_consistency",
                "d_sae",
                "k",
                "n_models",
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
                "alignment_to_ref_mean",
                "alignment_to_ref_ci95_low",
                "alignment_to_ref_ci95_high",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "lambda_consistency": r["lambda_consistency"],
                    "d_sae": r["d_sae"],
                    "k": r["k"],
                    "n_models": r["n_models"],
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
                    "alignment_to_ref_mean": r["alignment_to_ref"]["mean"],
                    "alignment_to_ref_ci95_low": r["alignment_to_ref"]["ci95_low"],
                    "alignment_to_ref_ci95_high": r["alignment_to_ref"]["ci95_high"],
                    "runtime_sec": r["runtime_sec"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency regularization sweep")
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
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed-ref", type=int, default=42)
    parser.add_argument("--train-seeds", type=str, default="123,456,789,1011")
    parser.add_argument("--lambdas", type=str, default="0.0,0.0001,0.0005,0.001,0.002")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-ev-drop", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "consistency_objective_sweep",
    )
    args = parser.parse_args()

    train_seeds = parse_int_list(args.train_seeds)
    lambdas = parse_float_list(args.lambdas)

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
        "k": args.k,
        "seed_ref": args.seed_ref,
        "train_seeds": train_seeds,
        "lambdas": lambdas,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_ev_drop": args.max_ev_drop,
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

    records: list[dict[str, Any]] = []
    for lam in lambdas:
        print(f"[lambda] {lam}")
        record = run_lambda_condition(
            lambda_consistency=lam,
            activations=acts,
            d_sae=args.d_sae,
            k=args.k,
            seed_ref=args.seed_ref,
            train_seeds=train_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            bootstrap_samples=args.bootstrap_samples,
            checkpoint_dir=run_dir / "checkpoints" / f"lambda_{lam}",
        )
        records.append(record)

    selected = choose_lambda(records, max_ev_drop=args.max_ev_drop)

    write_csv(run_dir / "sweep_summary.csv", records)

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
        "records": records,
        "selection": {
            "criterion": "max delta_pwmcc_ci_low_conservative under EV-drop constraint",
            "max_ev_drop": args.max_ev_drop,
            "selected_lambda": selected["lambda_consistency"],
            "selected_record": selected,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Consistency Objective Sweep",
        "",
        f"- Run ID: `{run_id}`",
        f"- Git commit: `{payload['run_metadata']['git_commit']}`",
        f"- Selected lambda: `{selected['lambda_consistency']}`",
        "",
        "| lambda | trained PWMCC (95% CI) | random PWMCC (95% CI) | delta | conservative delta LCB | EV | alignment-to-ref |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in sorted(records, key=lambda x: x["lambda_consistency"]):
        t = r["trained_pwmcc"]
        rb = r["random_pwmcc"]
        a = r["alignment_to_ref"]
        lines.append(
            "| "
            f"{r['lambda_consistency']} | {t['mean']:.6f} [{t['ci95_low']:.6f}, {t['ci95_high']:.6f}] | "
            f"{rb['mean']:.6f} [{rb['ci95_low']:.6f}, {rb['ci95_high']:.6f}] | "
            f"{r['delta_pwmcc']:.6f} | {r['delta_pwmcc_ci_low_conservative']:.6f} | "
            f"{r['explained_variance']['mean']:.4f} | {a['mean']:.6f} |"
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "run_metadata": payload["run_metadata"],
        "artifacts": [
            str((run_dir / "config.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "results.json").relative_to(PROJECT_ROOT)),
            str((run_dir / "summary.md").relative_to(PROJECT_ROOT)),
            str((run_dir / "sweep_summary.csv").relative_to(PROJECT_ROOT)),
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Consistency objective sweep complete")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
