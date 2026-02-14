#!/usr/bin/env python3
"""Run a reproducible transcoder-vs-SAE stress evaluation.

This script trains a TopK transcoder and a matched TopK SAE on the same
activation source across multiple seeds, then compares feature stability
using PWMCC. It emits a gate-friendly summary artifact with `transcoder_delta`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.modular_arithmetic import create_dataloaders  # noqa: E402
from src.models.transformer import ModularArithmeticTransformer  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path) -> str:
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def position_to_index(position: str) -> int:
    mapping = {
        "answer": -2,
        "bos": 0,
        "first_operand": 1,
    }
    if position not in mapping:
        raise ValueError(f"Unsupported position: {position}")
    return mapping[position]


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def bootstrap_ci(values: list[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def pwmcc(decoder_a: torch.Tensor, decoder_b: torch.Tensor) -> float:
    """Pairwise maximum cosine correlation across feature columns."""
    a = F.normalize(decoder_a, dim=0)
    b = F.normalize(decoder_b, dim=0)
    cos = a.T @ b
    max_a = cos.abs().max(dim=1).values.mean().item()
    max_b = cos.abs().max(dim=0).values.mean().item()
    return float((max_a + max_b) / 2.0)


def pairwise_pwmcc(decoders: dict[int, torch.Tensor]) -> list[float]:
    out: list[float] = []
    for sa, sb in itertools.combinations(sorted(decoders.keys()), 2):
        out.append(pwmcc(decoders[sa], decoders[sb]))
    return out


def random_decoder_bank(d_model: int, d_sae: int, seeds: list[int]) -> dict[int, torch.Tensor]:
    bank: dict[int, torch.Tensor] = {}
    for seed in seeds:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 100_000)
        bank[seed] = torch.randn((d_model, d_sae), generator=gen)
    return bank


def load_mlp_io_activations(
    *,
    checkpoint: Path,
    layer: int,
    position: str,
    batch_size: int,
    device: str,
    modulus: int,
    seed: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model, _ = ModularArithmeticTransformer.load_checkpoint(checkpoint, device=device)
    model.eval()
    token_index = position_to_index(position)

    train_loader, val_loader = create_dataloaders(
        modulus=modulus,
        fraction=1.0,
        train_fraction=0.8,
        batch_size=batch_size,
        seed=seed,
        format="sequence",
        num_workers=0,
    )
    all_loader = DataLoader(
        torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    mlp_inputs: list[torch.Tensor] = []
    mlp_outputs: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for tokens, _ in all_loader:
            tokens = tokens.to(device)
            _, cache = model.model.run_with_cache(tokens)
            mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"][:, token_index, :]
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][:, token_index, :]

            mlp_inputs.append(mlp_in.detach().float().cpu())
            mlp_outputs.append(mlp_out.detach().float().cpu())

            collected += int(tokens.shape[0])
            if max_samples > 0 and collected >= max_samples:
                break

    x = torch.cat(mlp_inputs, dim=0)
    y = torch.cat(mlp_outputs, dim=0)
    if max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


class TopKEncoderDecoder(nn.Module):
    """Shared TopK sparse encoder with linear decoder."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = min(k, d_sae)
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = self.encoder(x)
        topk_values, topk_idx = torch.topk(pre_act, k=self.k, dim=-1, sorted=False)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_idx, topk_values)
        return latents

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        out = self.decoder(latents)
        return out, latents


@dataclass
class TrainMetrics:
    mse: float
    l0: float


def train_model(
    *,
    model: TopKEncoderDecoder,
    source: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> tuple[TopKEncoderDecoder, TrainMetrics]:
    set_seed(seed)
    model = model.to(device)
    model.train()

    dataset = TensorDataset(source, target)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            x = batch_x.to(device)
            y = batch_y.to(device)
            pred, _ = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

    model.eval()
    with torch.no_grad():
        x = source.to(device)
        y = target.to(device)
        pred, latents = model(x)
        mse = float(F.mse_loss(pred, y).item())
        l0 = float((latents != 0).float().sum(dim=-1).mean().item())

    return model.cpu(), TrainMetrics(mse=mse, l0=l0)


def save_checkpoint(
    *,
    path: Path,
    model: TopKEncoderDecoder,
    seed: int,
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "d_model": model.d_model,
            "d_sae": model.d_sae,
            "k": model.k,
            "seed": seed,
            "mode": mode,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcoder-vs-SAE stress evaluation")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--position", type=str, default="answer", choices=["answer", "bos", "first_operand"])
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all available samples")
    parser.add_argument("--d-sae", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_transcoder_stress",
    )
    args = parser.parse_args()

    args.transformer_checkpoint = resolve_path(args.transformer_checkpoint).resolve()
    args.output_dir = resolve_path(args.output_dir).resolve()

    if not args.transformer_checkpoint.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {args.transformer_checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    seeds = parse_int_list(args.seeds)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    activations_in, activations_out = load_mlp_io_activations(
        checkpoint=args.transformer_checkpoint,
        layer=args.layer,
        position=args.position,
        batch_size=args.batch_size,
        device=args.device,
        modulus=args.modulus,
        seed=min(seeds),
        max_samples=args.max_samples,
    )

    d_model = int(activations_in.shape[1])
    transcoder_decoders: dict[int, torch.Tensor] = {}
    sae_decoders: dict[int, torch.Tensor] = {}
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        transcoder = TopKEncoderDecoder(d_model=d_model, d_sae=args.d_sae, k=args.k)
        transcoder, transcoder_metrics = train_model(
            model=transcoder,
            source=activations_in,
            target=activations_out,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        transcoder_ckpt = ckpt_dir / f"transcoder_seed{seed}.pt"
        save_checkpoint(path=transcoder_ckpt, model=transcoder, seed=seed, mode="transcoder")
        transcoder_decoders[seed] = transcoder.decoder.weight.detach().float().cpu()

        sae = TopKEncoderDecoder(d_model=d_model, d_sae=args.d_sae, k=args.k)
        sae, sae_metrics = train_model(
            model=sae,
            source=activations_in,
            target=activations_in,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        sae_ckpt = ckpt_dir / f"sae_seed{seed}.pt"
        save_checkpoint(path=sae_ckpt, model=sae, seed=seed, mode="sae")
        sae_decoders[seed] = sae.decoder.weight.detach().float().cpu()

        per_seed.append(
            {
                "seed": seed,
                "transcoder": {
                    **asdict(transcoder_metrics),
                    "checkpoint": repo_rel(transcoder_ckpt),
                },
                "sae": {
                    **asdict(sae_metrics),
                    "checkpoint": repo_rel(sae_ckpt),
                },
            }
        )

    transcoder_pairwise = pairwise_pwmcc(transcoder_decoders)
    sae_pairwise = pairwise_pwmcc(sae_decoders)
    random_pairwise = pairwise_pwmcc(random_decoder_bank(d_model=d_model, d_sae=args.d_sae, seeds=seeds))

    transcoder_summary = summary_stats(transcoder_pairwise)
    sae_summary = summary_stats(sae_pairwise)
    random_summary = summary_stats(random_pairwise)
    transcoder_ci = bootstrap_ci(transcoder_pairwise, n_bootstrap=args.bootstrap_samples, seed=13)
    sae_ci = bootstrap_ci(sae_pairwise, n_bootstrap=args.bootstrap_samples, seed=17)

    delta = transcoder_summary["mean"] - sae_summary["mean"]
    ratio_transcoder_vs_random = (
        transcoder_summary["mean"] / random_summary["mean"] if random_summary["mean"] > 0 else float("nan")
    )
    ratio_sae_vs_random = sae_summary["mean"] / random_summary["mean"] if random_summary["mean"] > 0 else float("nan")

    config_payload = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "layer": args.layer,
        "position": args.position,
        "modulus": args.modulus,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "d_sae": args.d_sae,
        "k": args.k,
        "seeds": seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "bootstrap_samples": args.bootstrap_samples,
        "device": args.device,
        "run_id": run_id,
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
        "data": {
            "samples": int(activations_in.shape[0]),
            "d_model": d_model,
            "mlp_input_mean": float(activations_in.mean().item()),
            "mlp_output_mean": float(activations_out.mean().item()),
            "mlp_input_std": float(activations_in.std().item()),
            "mlp_output_std": float(activations_out.std().item()),
        },
        "per_seed": per_seed,
        "transcoder": {
            "pairwise_pwmcc_values": transcoder_pairwise,
            "summary": {**transcoder_summary, "ci95_low": transcoder_ci[0], "ci95_high": transcoder_ci[1]},
        },
        "sae": {
            "pairwise_pwmcc_values": sae_pairwise,
            "summary": {**sae_summary, "ci95_low": sae_ci[0], "ci95_high": sae_ci[1]},
        },
        "random_baseline": {
            "pairwise_pwmcc_values": random_pairwise,
            "summary": random_summary,
        },
        # Gate-compatible fields:
        "delta": delta,
        "transcoder_delta": delta,
        "metrics": {
            "transcoder_delta": delta,
            "ratio_transcoder_vs_random": ratio_transcoder_vs_random,
            "ratio_sae_vs_random": ratio_sae_vs_random,
        },
    }

    out_json = run_dir / "transcoder_stress_summary.json"
    out_md = run_dir / "transcoder_stress_summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Transcoder Stress Evaluation",
        "",
        f"- Run ID: `{run_id}`",
        f"- Samples: `{payload['data']['samples']}`",
        f"- d_model / d_sae / k: `{d_model}` / `{args.d_sae}` / `{args.k}`",
        f"- Seeds: `{seeds}`",
        "",
        "## Stability Summary (PWMCC)",
        "",
        f"- Transcoder mean (95% CI): `{transcoder_summary['mean']:.6f}` (`{transcoder_ci[0]:.6f}`, `{transcoder_ci[1]:.6f}`)",
        f"- SAE mean (95% CI): `{sae_summary['mean']:.6f}` (`{sae_ci[0]:.6f}`, `{sae_ci[1]:.6f}`)",
        f"- Random baseline mean: `{random_summary['mean']:.6f}`",
        f"- transcoder_delta (transcoder - SAE): `{delta:.6f}`",
        "",
        "## Ratios",
        "",
        f"- transcoder / random: `{ratio_transcoder_vs_random:.6f}`",
        f"- sae / random: `{ratio_sae_vs_random:.6f}`",
        "",
        "## Artifacts",
        "",
        f"- JSON summary: `{repo_rel(out_json)}`",
        f"- Markdown summary: `{repo_rel(out_md)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("Transcoder stress evaluation complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
