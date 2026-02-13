#!/usr/bin/env python3
"""Train a HUSAI TopK SAE from cached activation tensors.

This script is designed for fast benchmark integration workflows where activation
caches already exist (for example SAEBench model activation caches).

It produces a standard HUSAI checkpoint (`sae_final.pt`) compatible with the
repo's `TopKSAE.save()` format.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simple_sae import TopKSAE  # noqa: E402


@dataclass
class TrainConfig:
    activation_cache_dir: str
    activation_glob: str
    max_files: int
    max_rows_per_file: int
    max_total_rows: int
    d_sae: int
    k: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    mse: float
    aux: float
    l0: float
    explained_variance: float


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_rows(tensor: torch.Tensor, max_rows: int, generator: torch.Generator) -> torch.Tensor:
    if max_rows <= 0 or tensor.shape[0] <= max_rows:
        return tensor
    idx = torch.randperm(tensor.shape[0], generator=generator)[:max_rows]
    return tensor.index_select(0, idx)


def load_activation_bank(
    *,
    cache_dir: Path,
    activation_glob: str,
    max_files: int,
    max_rows_per_file: int,
    max_total_rows: int,
    seed: int,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    files = sorted(cache_dir.glob(activation_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No activation files matched {activation_glob} under {cache_dir}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    chunks: list[torch.Tensor] = []
    files_used: list[str] = []
    total_rows = 0
    d_model: int | None = None

    for path in files:
        acts = torch.load(path, map_location="cpu")
        if not isinstance(acts, torch.Tensor):
            raise TypeError(f"Activation file is not a tensor: {path}")
        if acts.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N, d_model], got {tuple(acts.shape)} in {path}")

        acts = acts.float()
        if d_model is None:
            d_model = int(acts.shape[1])
        elif acts.shape[1] != d_model:
            raise ValueError(
                f"Inconsistent activation width: expected {d_model}, got {acts.shape[1]} in {path}"
            )

        sampled = _sample_rows(acts, max_rows_per_file, generator)

        if max_total_rows > 0:
            remaining = max_total_rows - total_rows
            if remaining <= 0:
                break
            if sampled.shape[0] > remaining:
                sampled = sampled[:remaining]

        chunks.append(sampled)
        files_used.append(str(path))
        total_rows += int(sampled.shape[0])

    if not chunks:
        raise RuntimeError("No activations selected after sampling constraints.")

    bank = torch.cat(chunks, dim=0)
    meta = {
        "num_files_discovered": len(sorted(cache_dir.glob(activation_glob))),
        "num_files_used": len(files_used),
        "total_rows": int(bank.shape[0]),
        "d_model": int(bank.shape[1]),
    }
    return bank, files_used, meta


def train_topk_sae(
    *,
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[TopKSAE, list[EpochMetrics]]:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations shape [N, d_model], got {tuple(activations.shape)}")

    d_model = int(activations.shape[1])
    sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k).to(device)

    dataset = TensorDataset(activations)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    act_var = float(activations.var(dim=0).sum().item())
    metrics: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        sae.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_aux = 0.0
        epoch_l0 = 0.0
        batches = 0

        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            recon, latents, aux_loss = sae(batch)

            mse = F.mse_loss(recon, batch)
            loss = mse + aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += float(loss.item())
            epoch_mse += float(mse.item())
            epoch_aux += float(aux_loss.item())
            epoch_l0 += float((latents != 0).float().sum(dim=-1).mean().item())
            batches += 1

        mean_loss = epoch_loss / max(batches, 1)
        mean_mse = epoch_mse / max(batches, 1)
        mean_aux = epoch_aux / max(batches, 1)
        mean_l0 = epoch_l0 / max(batches, 1)
        ev = 1.0 - (mean_mse / act_var) if act_var > 0 else 0.0

        metrics.append(
            EpochMetrics(
                epoch=epoch,
                loss=mean_loss,
                mse=mean_mse,
                aux=mean_aux,
                l0=mean_l0,
                explained_variance=ev,
            )
        )

    return sae, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HUSAI TopK SAE from cached activations")
    parser.add_argument(
        "--activation-cache-dir",
        type=Path,
        default=Path("/tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped"),
        help="Directory containing cached activation .pt files",
    )
    parser.add_argument(
        "--activation-glob",
        type=str,
        default="*_blocks.0.hook_resid_pre.pt",
        help="Glob pattern inside activation cache dir",
    )
    parser.add_argument("--max-files", type=int, default=60, help="Max number of activation files to use (<=0 means all)")
    parser.add_argument("--max-rows-per-file", type=int, default=2048, help="Max sampled rows per activation file")
    parser.add_argument("--max-total-rows", type=int, default=120000, help="Global cap on rows after concatenation")
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "saes" / "husai_pythia70m_topk_seed42",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    config = TrainConfig(
        activation_cache_dir=str(args.activation_cache_dir),
        activation_glob=args.activation_glob,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
        d_sae=args.d_sae,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )

    activations, files_used, data_meta = load_activation_bank(
        cache_dir=args.activation_cache_dir,
        activation_glob=args.activation_glob,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
        seed=args.seed,
    )

    sae, history = train_topk_sae(
        activations=activations,
        d_sae=args.d_sae,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )

    checkpoint_path = output_dir / "sae_final.pt"
    sae.save(checkpoint_path)

    metrics_path = output_dir / "training_metrics.json"
    metrics_payload = [asdict(m) for m in history]
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")

    summary_payload = {
        "timestamp_utc": utc_now(),
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
        "data": data_meta,
        "source_files": files_used,
        "final_metrics": metrics_payload[-1] if metrics_payload else {},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    print("Training complete")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
