#!/usr/bin/env python3
"""Assignment-aware consistency objective v2 with acceptance criteria.

v2 changes vs v1:
- Uses explicit one-to-one feature assignment (Hungarian matching) in the
  consistency regularizer instead of max-over-features alignment.
- Reports conservative delta lower-bound and acceptance gates.
- Supports optional external metric gate from a benchmark summary artifact.
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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.extract_activations import extract_activations  # noqa: E402
from src.models.simple_sae import TopKSAE  # noqa: E402


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


def assignment_alignment_penalty(
    decoder: torch.Tensor,
    ref_decoder: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Permutation-invariant alignment penalty via Hungarian matching."""
    d = F.normalize(decoder, dim=0)
    r = F.normalize(ref_decoder, dim=0)
    cos = d.T @ r

    with torch.no_grad():
        cost = (1.0 - cos.abs()).detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)
        row_t = torch.as_tensor(row_idx, device=cos.device, dtype=torch.long)
        col_t = torch.as_tensor(col_idx, device=cos.device, dtype=torch.long)

    matched = cos.abs()[row_t, col_t]
    mean_alignment = matched.mean()
    penalty = 1.0 - mean_alignment
    return penalty, float(mean_alignment.item())


def train_topk_assignment_v2(
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

    alignment_mean_running = 0.0
    alignment_batches = 0

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, latents, _ = model(batch, compute_aux_loss=False)
            mse = F.mse_loss(recon, batch)

            if ref_decoder_device is not None and lambda_consistency > 0:
                penalty, align = assignment_alignment_penalty(model.decoder.weight, ref_decoder_device)
                loss = mse + lambda_consistency * penalty
                alignment_mean_running += align
                alignment_batches += 1
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

    metrics = {
        "mse": mse,
        "explained_variance": explained_var,
        "l0": l0,
        "assignment_alignment_mean_train": (
            alignment_mean_running / alignment_batches if alignment_batches > 0 else float("nan")
        ),
    }
    return model.cpu(), metrics


def alignment_score_assignment(decoder: torch.Tensor, ref_decoder: torch.Tensor) -> float:
    _, align = assignment_alignment_penalty(decoder, ref_decoder)
    return align


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

    ref_model, ref_metrics = train_topk_assignment_v2(
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
            "consistency_objective": "assignment_hungarian_v2",
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
        model, metrics = train_topk_assignment_v2(
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
                "consistency_objective": "assignment_hungarian_v2",
            },
            ckpt,
        )

        decoders[seed] = decoder
        per_seed_metrics.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                "lambda_consistency": lambda_consistency,
                "alignment_to_ref": alignment_score_assignment(decoder, ref_decoder),
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
    align = summarize(align_values, n_bootstrap=bootstrap_samples, seed=43)

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


def write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lambda_consistency",
        "delta_pwmcc",
        "delta_pwmcc_ci_low_conservative",
        "ratio_pwmcc",
        "trained_pwmcc_mean",
        "trained_pwmcc_ci95_low",
        "trained_pwmcc_ci95_high",
        "random_pwmcc_mean",
        "random_pwmcc_ci95_low",
        "random_pwmcc_ci95_high",
        "explained_variance_mean",
        "alignment_to_ref_mean",
        "runtime_sec",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "lambda_consistency": rec["lambda_consistency"],
                    "delta_pwmcc": rec["delta_pwmcc"],
                    "delta_pwmcc_ci_low_conservative": rec["delta_pwmcc_ci_low_conservative"],
                    "ratio_pwmcc": rec["ratio_pwmcc"],
                    "trained_pwmcc_mean": rec["trained_pwmcc"]["mean"],
                    "trained_pwmcc_ci95_low": rec["trained_pwmcc"]["ci95_low"],
                    "trained_pwmcc_ci95_high": rec["trained_pwmcc"]["ci95_high"],
                    "random_pwmcc_mean": rec["random_pwmcc"]["mean"],
                    "random_pwmcc_ci95_low": rec["random_pwmcc"]["ci95_low"],
                    "random_pwmcc_ci95_high": rec["random_pwmcc"]["ci95_high"],
                    "explained_variance_mean": rec["explained_variance"]["mean"],
                    "alignment_to_ref_mean": rec["alignment_to_ref"]["mean"],
                    "runtime_sec": rec["runtime_sec"],
                }
            )


def extract_external_delta(path: Path | None) -> float | None:
    if path is None:
        return None
    if not path.exists():
        return None
    payload = json.loads(path.read_text())

    candidates: list[Any] = [
        (payload.get("summary") or {}).get("best_minus_llm_auc"),
        payload.get("best_minus_llm_auc"),
        (payload.get("delta_vs_matched_baseline") or {}).get("interpretability_score_mean_max"),
        (payload.get("delta_vs_matched_baseline") or {}).get("contrastive_score_mean_max"),
    ]
    for item in candidates:
        if isinstance(item, (float, int)):
            return float(item)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment-aware consistency v2 sweep")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "assignment_consistency_v2" / "layer1_answer_acts.pt",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--modulus", type=int, default=113)

    parser.add_argument("--seed-ref", type=int, default=42)
    parser.add_argument("--train-seeds", type=str, default="123,456,789,1011")
    parser.add_argument("--lambdas", type=str, default="0.0,0.01,0.05,0.1,0.2")

    parser.add_argument("--d-sae", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)

    parser.add_argument("--external-summary", type=Path, default=None)
    parser.add_argument("--min-delta-pwmcc", type=float, default=0.0)
    parser.add_argument("--min-delta-lcb", type=float, default=0.0)
    parser.add_argument("--max-ev-drop", type=float, default=0.05)
    parser.add_argument("--min-external-delta", type=float, default=0.0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4d_assignment_consistency_v2",
    )
    args = parser.parse_args()

    train_seeds = parse_int_list(args.train_seeds)
    lambdas = parse_float_list(args.lambdas)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    checkpoints_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    acts = load_or_extract_activations(
        cache_path=args.activation_cache,
        transformer_checkpoint=args.transformer_checkpoint,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        modulus=args.modulus,
        seed=args.seed_ref,
    )

    results: list[dict[str, Any]] = []
    for lam in lambdas:
        ckpt_dir = checkpoints_root / f"lambda_{lam}"
        rec = run_lambda_condition(
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
            checkpoint_dir=ckpt_dir,
        )
        results.append(rec)

    results_sorted = sorted(results, key=lambda r: r["lambda_consistency"])
    baseline = next((r for r in results_sorted if abs(r["lambda_consistency"]) < 1e-12), None)
    baseline_ev = baseline["explained_variance"]["mean"] if baseline else max(r["explained_variance"]["mean"] for r in results_sorted)

    best = max(results_sorted, key=lambda r: r["delta_pwmcc_ci_low_conservative"])
    ev_drop = baseline_ev - best["explained_variance"]["mean"]

    external_delta = extract_external_delta(args.external_summary)

    acceptance = {
        "min_delta_pwmcc": args.min_delta_pwmcc,
        "min_delta_lcb": args.min_delta_lcb,
        "max_ev_drop": args.max_ev_drop,
        "min_external_delta": args.min_external_delta,
        "best_lambda": best["lambda_consistency"],
        "best_delta_pwmcc": best["delta_pwmcc"],
        "best_delta_lcb": best["delta_pwmcc_ci_low_conservative"],
        "best_explained_variance": best["explained_variance"]["mean"],
        "baseline_explained_variance": baseline_ev,
        "ev_drop": ev_drop,
        "external_delta": external_delta,
        "gate_delta_pwmcc": bool(best["delta_pwmcc"] >= args.min_delta_pwmcc),
        "gate_delta_lcb": bool(best["delta_pwmcc_ci_low_conservative"] >= args.min_delta_lcb),
        "gate_ev_drop": bool(ev_drop <= args.max_ev_drop),
        "gate_external": bool(external_delta is None or external_delta >= args.min_external_delta),
    }
    acceptance["pass_all"] = bool(
        acceptance["gate_delta_pwmcc"]
        and acceptance["gate_delta_lcb"]
        and acceptance["gate_ev_drop"]
        and acceptance["gate_external"]
    )

    config_payload = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "activation_cache": str(args.activation_cache),
        "layer": args.layer,
        "modulus": args.modulus,
        "seed_ref": args.seed_ref,
        "train_seeds": train_seeds,
        "lambdas": lambdas,
        "d_sae": args.d_sae,
        "k": args.k,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "external_summary": str(args.external_summary) if args.external_summary else None,
        "acceptance": {
            "min_delta_pwmcc": args.min_delta_pwmcc,
            "min_delta_lcb": args.min_delta_lcb,
            "max_ev_drop": args.max_ev_drop,
            "min_external_delta": args.min_external_delta,
        },
    }

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "config_hash": stable_hash(config_payload),
            "run_id": run_id,
        },
        "config": config_payload,
        "results": results_sorted,
        "selection": {
            "strategy": "max conservative delta LCB",
            "best_lambda": best["lambda_consistency"],
        },
        "acceptance": acceptance,
    }

    out_json = run_dir / "results.json"
    out_csv = run_dir / "summary.csv"
    out_md = run_dir / "summary.md"

    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_summary_csv(out_csv, results_sorted)

    lines = [
        "# Assignment-Aware Consistency v2",
        "",
        f"- Run ID: `{run_id}`",
        f"- Git commit: `{payload['run_metadata']['git_commit']}`",
        f"- Best lambda (conservative delta): `{best['lambda_consistency']}`",
        "",
        "## Acceptance",
        "",
        f"- pass_all: `{acceptance['pass_all']}`",
        f"- gate_delta_pwmcc: `{acceptance['gate_delta_pwmcc']}`",
        f"- gate_delta_lcb: `{acceptance['gate_delta_lcb']}`",
        f"- gate_ev_drop: `{acceptance['gate_ev_drop']}`",
        f"- gate_external: `{acceptance['gate_external']}`",
        "",
        "| lambda | delta PWMCC | conservative delta LCB | EV mean | alignment mean |",
        "|---:|---:|---:|---:|---:|",
    ]

    for rec in results_sorted:
        lines.append(
            "| "
            f"{rec['lambda_consistency']:.4f} | "
            f"{rec['delta_pwmcc']:.6f} | "
            f"{rec['delta_pwmcc_ci_low_conservative']:.6f} | "
            f"{rec['explained_variance']['mean']:.6f} | "
            f"{rec['alignment_to_ref']['mean']:.6f} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print("Assignment consistency v2 complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
