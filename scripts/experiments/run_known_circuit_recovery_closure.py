#!/usr/bin/env python3
"""Known-circuit recovery closure track with trained-vs-random controls.

This script evaluates whether the modular-arithmetic transformer and selected SAE
checkpoints recover Fourier structure above random baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer import ModularArithmeticTransformer  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_ints(text: str) -> list[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected non-empty integer list")
    return [int(v) for v in values]


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": 0,
        }

    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
        half = 1.96 * std / math.sqrt(arr.size)
    else:
        std = 0.0
        half = 0.0

    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": int(arr.size),
    }


def top_frequencies_from_embedding(embedding: torch.Tensor, top_k: int) -> list[int]:
    # DFT on vocab dimension, as in modular-addition literature analyses.
    freq_mag = torch.abs(torch.fft.fft(embedding, dim=0)).mean(dim=1)
    idx = torch.argsort(freq_mag, descending=True)[:top_k]
    return [int(i) for i in idx.tolist()]


def build_fourier_basis(modulus: int, freqs: list[int]) -> torch.Tensor:
    components: list[torch.Tensor] = []
    idx = torch.arange(modulus, dtype=torch.float32)
    for k in freqs:
        angle = 2 * torch.pi * float(k) * idx / float(modulus)
        components.append(torch.cos(angle).unsqueeze(1))
        components.append(torch.sin(angle).unsqueeze(1))
    basis = torch.cat(components, dim=1)
    return F.normalize(basis, dim=0)


def variance_explained_r2(embedding: torch.Tensor, basis: torch.Tensor) -> float:
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    proj = basis @ (basis.T @ centered)
    ss_total = torch.sum(centered**2)
    ss_res = torch.sum((centered - proj) ** 2)
    return float((1.0 - ss_res / ss_total).item())


def transformer_fourier_r2(model: ModularArithmeticTransformer, modulus: int, top_k_freq: int, r2_freq_count: int) -> dict[str, Any]:
    w_e = model.model.embed.W_E.detach().float().cpu()[:modulus, :]
    top_freqs = top_frequencies_from_embedding(w_e, top_k=top_k_freq)
    selected = top_freqs[: max(1, min(r2_freq_count, len(top_freqs)))]
    basis = build_fourier_basis(modulus, selected)
    r2 = variance_explained_r2(w_e, basis)
    return {
        "r2": r2,
        "top_frequencies": top_freqs,
        "frequencies_used_for_r2": selected,
        "embedding_shape": list(w_e.shape),
    }


def load_decoder_matrix(checkpoint_path: Path) -> torch.Tensor:
    obj = torch.load(checkpoint_path, map_location="cpu")
    state = obj.get("model_state_dict") if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(state)}")

    if "decoder.weight" in state:
        dec = state["decoder.weight"].detach().float().cpu()
        # [d_model, d_sae]
        return dec
    if "W_dec" in state:
        dec = state["W_dec"].detach().float().cpu()
        # [d_sae, d_model] -> [d_model, d_sae]
        return dec.T.contiguous()
    raise KeyError(f"No decoder matrix found in {checkpoint_path}")


def sae_fourier_overlap(decoder_matrix: torch.Tensor, basis: torch.Tensor) -> float:
    # decoder_matrix: [d_model, d_sae], basis: [d_model, n_basis]
    d = F.normalize(decoder_matrix, dim=0)
    b = F.normalize(basis, dim=0)
    cos = torch.abs(d.T @ b)
    return float(cos.max(dim=1).values.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Known-circuit recovery closure")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--top-k-frequencies", type=int, default=10)
    parser.add_argument("--r2-frequency-count", type=int, default=5)
    parser.add_argument("--random-transformer-seeds", type=str, default="11,23,37,41,53")

    parser.add_argument(
        "--sae-checkpoint-glob",
        type=str,
        default="results/experiments/phase4d_assignment_consistency_v2/run_*/checkpoints/lambda_0.2/sae_seed*.pt",
    )
    parser.add_argument("--random-sae-seeds", type=str, default="101,103,107,109,113")

    parser.add_argument("--min-transformer-r2-delta", type=float, default=0.1)
    parser.add_argument("--min-sae-overlap-delta", type=float, default=0.0)
    parser.add_argument("--fail-on-gate-fail", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "known_circuit_recovery_closure",
    )
    args = parser.parse_args()

    args.transformer_checkpoint = to_abs(args.transformer_checkpoint)
    args.output_dir = to_abs(args.output_dir)

    if not args.transformer_checkpoint.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {args.transformer_checkpoint}")

    random_transformer_seeds = parse_ints(args.random_transformer_seeds)
    random_sae_seeds = parse_ints(args.random_sae_seeds)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trained_model, extras = ModularArithmeticTransformer.load_checkpoint(args.transformer_checkpoint, device="cpu")
    trained_model.eval()
    trained_fourier = transformer_fourier_r2(
        trained_model,
        modulus=args.modulus,
        top_k_freq=args.top_k_frequencies,
        r2_freq_count=args.r2_frequency_count,
    )

    random_transformer_r2: list[float] = []
    cfg = extras.get("config") if isinstance(extras, dict) else None
    if cfg is None:
        raise RuntimeError("Missing transformer config in checkpoint extras")

    for seed in random_transformer_seeds:
        torch.manual_seed(seed)
        rand_model = ModularArithmeticTransformer(cfg, device="cpu")
        rand_model.eval()
        rand_fourier = transformer_fourier_r2(
            rand_model,
            modulus=args.modulus,
            top_k_freq=args.top_k_frequencies,
            r2_freq_count=args.r2_frequency_count,
        )
        random_transformer_r2.append(float(rand_fourier["r2"]))

    random_transformer_stats = summary_stats(random_transformer_r2)
    transformer_r2_delta = float(trained_fourier["r2"]) - float(random_transformer_stats["mean"])
    transformer_r2_delta_lcb = float(trained_fourier["r2"]) - float(random_transformer_stats["ci95_high"])

    freqs_for_basis = [int(k) for k in trained_fourier["frequencies_used_for_r2"]]
    basis_full = build_fourier_basis(args.modulus, freqs_for_basis)

    sae_ckpts = sorted(PROJECT_ROOT.glob(args.sae_checkpoint_glob))
    trained_sae_overlap_values: list[float] = []
    sae_records: list[dict[str, Any]] = []
    d_model = None
    d_sae = None

    for ckpt in sae_ckpts:
        try:
            dec = load_decoder_matrix(ckpt)
        except Exception:
            continue

        if d_model is None:
            d_model = int(dec.shape[0])
            d_sae = int(dec.shape[1])

        if dec.shape[0] != basis_full.shape[0]:
            # Skip incompatible decoder widths.
            continue

        overlap = sae_fourier_overlap(dec, basis_full)
        trained_sae_overlap_values.append(overlap)
        sae_records.append(
            {
                "checkpoint": repo_rel(ckpt),
                "overlap": overlap,
                "d_model": int(dec.shape[0]),
                "d_sae": int(dec.shape[1]),
            }
        )

    trained_sae_stats = summary_stats(trained_sae_overlap_values)

    random_sae_overlap_values: list[float] = []
    if d_model is not None and d_sae is not None and sae_records:
        for seed in random_sae_seeds:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            rand_dec = torch.randn((d_model, d_sae), generator=gen)
            random_sae_overlap_values.append(sae_fourier_overlap(rand_dec, basis_full))

    random_sae_stats = summary_stats(random_sae_overlap_values)
    sae_overlap_delta = None
    sae_overlap_delta_lcb = None
    if sae_records and random_sae_overlap_values:
        sae_overlap_delta = float(trained_sae_stats["mean"]) - float(random_sae_stats["mean"])
        sae_overlap_delta_lcb = float(trained_sae_stats["ci95_low"]) - float(random_sae_stats["ci95_high"])

    gate_transformer = transformer_r2_delta_lcb >= args.min_transformer_r2_delta
    gate_sae = sae_overlap_delta_lcb is not None and sae_overlap_delta_lcb >= args.min_sae_overlap_delta
    pass_all = bool(gate_transformer and gate_sae)

    config_payload = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "modulus": args.modulus,
        "top_k_frequencies": args.top_k_frequencies,
        "r2_frequency_count": args.r2_frequency_count,
        "random_transformer_seeds": random_transformer_seeds,
        "sae_checkpoint_glob": args.sae_checkpoint_glob,
        "random_sae_seeds": random_sae_seeds,
        "min_transformer_r2_delta": args.min_transformer_r2_delta,
        "min_sae_overlap_delta": args.min_sae_overlap_delta,
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
        "transformer": {
            "trained": trained_fourier,
            "random_stats": random_transformer_stats,
            "random_values": random_transformer_r2,
            "delta_vs_random_mean": transformer_r2_delta,
            "delta_vs_random_lcb": transformer_r2_delta_lcb,
        },
        "sae": {
            "checkpoints_evaluated": len(sae_records),
            "records": sae_records,
            "trained_stats": trained_sae_stats,
            "random_stats": random_sae_stats,
            "random_values": random_sae_overlap_values,
            "delta_vs_random_mean": sae_overlap_delta,
            "delta_vs_random_lcb": sae_overlap_delta_lcb,
        },
        "gates": {
            "transformer_r2": gate_transformer,
            "sae_overlap": gate_sae,
            "pass_all": pass_all,
        },
    }

    out_json = run_dir / "closure_summary.json"
    out_md = run_dir / "closure_summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Known-Circuit Recovery Closure",
        "",
        f"- Run ID: `{run_id}`",
        f"- pass_all: `{pass_all}`",
        "",
        "## Transformer Fourier Recovery",
        "",
        f"- trained_r2: `{trained_fourier['r2']}`",
        f"- random_r2_mean: `{random_transformer_stats['mean']}`",
        f"- delta_vs_random_mean: `{transformer_r2_delta}`",
        f"- delta_vs_random_lcb: `{transformer_r2_delta_lcb}`",
        f"- gate_transformer_r2: `{gate_transformer}`",
        "",
        "## SAE Fourier Overlap",
        "",
        f"- checkpoints_evaluated: `{len(sae_records)}`",
        f"- trained_overlap_mean: `{trained_sae_stats['mean']}`",
        f"- random_overlap_mean: `{random_sae_stats['mean']}`",
        f"- delta_vs_random_mean: `{sae_overlap_delta}`",
        f"- delta_vs_random_lcb: `{sae_overlap_delta_lcb}`",
        f"- gate_sae_overlap: `{gate_sae}`",
        "",
        "## Artifacts",
        "",
        f"- JSON summary: `{repo_rel(out_json)}`",
        f"- Markdown summary: `{repo_rel(out_md)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("Known-circuit closure complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")

    if args.fail_on_gate_fail and not pass_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
