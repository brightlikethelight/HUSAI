#!/usr/bin/env python3
"""Phase 4a reproduction: trained-vs-random PWMCC with manifest logging.

This script reproduces the core trained-vs-random stability comparison using
existing SAE checkpoints and random decoder controls under the same geometry.
It writes machine-readable artifacts and a run manifest for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SummaryStats:
    mean: float
    std: float
    min: float
    max: float
    median: float
    ci95_low: float
    ci95_high: float
    n: int


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def load_decoder(path: Path) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu")

    # Common checkpoint formats in this repo.
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "sae_state_dict" in ckpt:
            state = ckpt["sae_state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    decoder_key = None
    if "decoder.weight" in state:
        decoder_key = "decoder.weight"
    else:
        for key in state.keys():
            if key.endswith("decoder.weight"):
                decoder_key = key
                break

    if decoder_key is None:
        raise KeyError(f"decoder.weight not found in checkpoint: {path}")

    decoder = state[decoder_key].detach().float().cpu()
    if decoder.ndim != 2:
        raise ValueError(f"decoder.weight must be rank-2, got {decoder.shape} at {path}")

    # Expected orientation is [d_model, d_sae]. Transpose if clearly reversed.
    if decoder.shape[0] > decoder.shape[1]:
        decoder = decoder.T
    return decoder


def pwmcc(decoder_a: torch.Tensor, decoder_b: torch.Tensor) -> float:
    a = F.normalize(decoder_a, dim=0)
    b = F.normalize(decoder_b, dim=0)
    cos = a.T @ b
    max_a = cos.abs().max(dim=1).values.mean().item()
    max_b = cos.abs().max(dim=0).values.mean().item()
    return float((max_a + max_b) / 2)


def pairwise_pwmcc(decoders: dict[int, torch.Tensor]) -> tuple[list[dict[str, Any]], list[float]]:
    pair_rows: list[dict[str, Any]] = []
    values: list[float] = []

    for seed_a, seed_b in itertools.combinations(sorted(decoders.keys()), 2):
        value = pwmcc(decoders[seed_a], decoders[seed_b])
        pair_rows.append({"seed_a": seed_a, "seed_b": seed_b, "pwmcc": value})
        values.append(value)

    return pair_rows, values


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(values: list[float], n_bootstrap: int = 10000, seed: int = 0) -> SummaryStats:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return SummaryStats(
        mean=float(arr.mean()),
        std=std,
        min=float(arr.min()),
        max=float(arr.max()),
        median=float(np.median(arr)),
        ci95_low=lo,
        ci95_high=hi,
        n=int(arr.size),
    )


def cohens_d(a: list[float], b: list[float]) -> float:
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.size < 2 or xb.size < 2:
        return float("nan")

    var_a = xa.var(ddof=1)
    var_b = xb.var(ddof=1)
    pooled = math.sqrt(((xa.size - 1) * var_a + (xb.size - 1) * var_b) / (xa.size + xb.size - 2))
    if pooled == 0:
        return float("nan")
    return float((xa.mean() - xb.mean()) / pooled)


def find_checkpoint(sae_root: Path, seed: int) -> Path:
    candidate_dirs = [
        sae_root / f"topk_seed{seed}",
        sae_root / f"topk_layer1_seed{seed}",
    ]
    candidate_files = ["sae_final.pt", "sae.pt"]

    for directory in candidate_dirs:
        for filename in candidate_files:
            path = directory / filename
            if path.exists():
                return path

    raise FileNotFoundError(f"No TopK SAE checkpoint found for seed {seed} under {sae_root}")


def write_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_a", "seed_b", "pwmcc"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4a trained-vs-random reproduction")
    parser.add_argument(
        "--sae-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "saes",
        help="Directory containing trained SAE checkpoints",
    )
    parser.add_argument(
        "--trained-seeds",
        type=str,
        default="42,123,456,789,1011",
        help="Comma-separated list of trained seeds",
    )
    parser.add_argument(
        "--random-seeds",
        type=str,
        default="1000,1001,1002,1003,1004",
        help="Comma-separated list of random control seeds",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-sae", type=int, default=1024)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4a_trained_vs_random",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "trained_vs_random_pwmcc.json",
        help="Legacy analysis output path to keep existing tooling compatible",
    )
    args = parser.parse_args()

    trained_seeds = parse_int_list(args.trained_seeds)
    random_seeds = parse_int_list(args.random_seeds)

    sae_root = args.sae_root if args.sae_root.is_absolute() else (PROJECT_ROOT / args.sae_root)
    sae_root = sae_root.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained decoders.
    trained_decoders: dict[int, torch.Tensor] = {}
    checkpoint_manifest: list[dict[str, Any]] = []
    for seed in trained_seeds:
        ckpt_path = find_checkpoint(sae_root, seed).resolve()
        decoder = load_decoder(ckpt_path)
        trained_decoders[seed] = decoder
        checkpoint_manifest.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(ckpt_path),
                "shape": list(decoder.shape),
            }
        )

    # Random controls with matched geometry.
    random_decoders: dict[int, torch.Tensor] = {}
    for seed in random_seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        decoder = torch.randn((args.d_model, args.d_sae), generator=g)
        random_decoders[seed] = decoder

    trained_pairs, trained_values = pairwise_pwmcc(trained_decoders)
    random_pairs, random_values = pairwise_pwmcc(random_decoders)

    trained_stats = summarize(trained_values, n_bootstrap=args.bootstrap_samples, seed=17)
    random_stats = summarize(random_values, n_bootstrap=args.bootstrap_samples, seed=29)

    stat, p_value = mannwhitneyu(trained_values, random_values, alternative="greater")
    delta = trained_stats.mean - random_stats.mean
    ratio = trained_stats.mean / random_stats.mean if random_stats.mean > 0 else float("nan")
    d_value = cohens_d(trained_values, random_values)

    conclusion = "training_signal_present" if (delta > 0 and p_value < 0.05) else "no_training_signal"

    config_payload = {
        "trained_seeds": trained_seeds,
        "random_seeds": random_seeds,
        "d_model": args.d_model,
        "d_sae": args.d_sae,
        "k": args.k,
        "bootstrap_samples": args.bootstrap_samples,
    }

    payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *list(map(str, __import__("sys").argv))]),
            "config_hash": stable_hash(config_payload),
        },
        "architecture": {"d_model": args.d_model, "d_sae": args.d_sae, "k": args.k},
        "trained": {
            "seeds": trained_seeds,
            "n_pairs": len(trained_values),
            "pwmcc_values": trained_values,
            **asdict(trained_stats),
        },
        "random": {
            "seeds": random_seeds,
            "n_pairs": len(random_values),
            "pwmcc_values": random_values,
            **asdict(random_stats),
        },
        "comparison": {
            "difference": delta,
            "ratio": ratio,
            "effect_size_cohens_d": d_value,
        },
        "statistical_test": {
            "test": "Mann-Whitney U (one-sided, trained > random)",
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": 0.05,
            "reject_null": bool(p_value < 0.05),
        },
        "checkpoint_manifest": checkpoint_manifest,
        "conclusion": conclusion,
    }

    # Write artifacts.
    write_pairs_csv(args.output_dir / "trained_pairwise_pwmcc.csv", trained_pairs)
    write_pairs_csv(args.output_dir / "random_pairwise_pwmcc.csv", random_pairs)

    with (args.output_dir / "results.json").open("w") as f:
        json.dump(payload, f, indent=2)

    with args.analysis_output.open("w") as f:
        json.dump(payload, f, indent=2)

    summary_md = args.output_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Phase 4a Reproduction: Trained vs Random",
                "",
                f"- Timestamp (UTC): {payload['run_metadata']['timestamp_utc']}",
                f"- Git commit: `{payload['run_metadata']['git_commit']}`",
                f"- Config hash: `{payload['run_metadata']['config_hash']}`",
                "",
                "## Results",
                "",
                "| Group | Mean PWMCC | 95% CI | Std | N pairs |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Trained | {trained_stats.mean:.6f} | "
                    f"[{trained_stats.ci95_low:.6f}, {trained_stats.ci95_high:.6f}] | "
                    f"{trained_stats.std:.6f} | {trained_stats.n} |"
                ),
                (
                    f"| Random | {random_stats.mean:.6f} | "
                    f"[{random_stats.ci95_low:.6f}, {random_stats.ci95_high:.6f}] | "
                    f"{random_stats.std:.6f} | {random_stats.n} |"
                ),
                "",
                f"- Mean difference (trained - random): `{delta:.6f}`",
                f"- Ratio (trained/random): `{ratio:.4f}`",
                f"- Mann-Whitney U p-value (one-sided): `{p_value:.3e}`",
                f"- Cohen's d: `{d_value:.4f}`",
                f"- Conclusion: **{conclusion}**",
            ]
        )
        + "\n"
    )

    manifest = {
        "run_metadata": payload["run_metadata"],
        "artifacts": [
            str((args.output_dir / "results.json").relative_to(PROJECT_ROOT)),
            str((args.output_dir / "summary.md").relative_to(PROJECT_ROOT)),
            str((args.output_dir / "trained_pairwise_pwmcc.csv").relative_to(PROJECT_ROOT)),
            str((args.output_dir / "random_pairwise_pwmcc.csv").relative_to(PROJECT_ROOT)),
            str(args.analysis_output.relative_to(PROJECT_ROOT)),
        ],
    }
    with (args.output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print("Phase 4a reproduction complete")
    print(f"Results: {args.output_dir / 'results.json'}")
    print(f"Summary: {args.output_dir / 'summary.md'}")
    print(f"Legacy analysis output: {args.analysis_output}")


if __name__ == "__main__":
    main()
