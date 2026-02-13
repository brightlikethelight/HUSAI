#!/usr/bin/env python3
"""Run a matched-budget architecture frontier sweep on external benchmarks.

Workflow per architecture/seed:
1) Train SAE on a shared activation budget.
2) Evaluate with HUSAI custom SAEBench adapter.
3) Optionally evaluate with HUSAI custom CE-Bench adapter.
4) Aggregate architecture frontier metrics with uncertainty summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.husai_custom_sae_adapter import dtype_from_name, normalize_architecture  # noqa: E402
from scripts.experiments.train_husai_sae_on_cached_activations import (  # noqa: E402
    DEFAULT_PYTHIA70M_ACTIVATION_CACHE,
    DEFAULT_SAEBENCH_MODEL_CACHE,
    load_activation_bank,
    set_seed,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_ints(text: str) -> list[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    return [int(v) for v in values]


def parse_architectures(text: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    return [normalize_architecture(v) or "topk" for v in values]


def maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def infer_dataset_names_from_files(files: list[str], hook_name: str) -> list[str]:
    suffix = f"_{hook_name}.pt"
    out: list[str] = []
    for raw in files:
        name = Path(raw).name
        if name.endswith(suffix):
            out.append(name[: -len(suffix)])
        else:
            out.append(Path(raw).stem)
    seen = set()
    deduped: list[str] = []
    for name in out:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def cmd_to_str(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def to_abs_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def to_repo_rel(path: Path) -> str:
    abs_path = to_abs_repo_path(path)
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


@dataclass
class TrainMetrics:
    mse: float
    explained_variance: float
    l0: float


def build_model(
    *,
    architecture: str,
    d_model: int,
    d_sae: int,
    k: int,
    model_name: str,
    hook_layer: int,
    hook_name: str,
    device: str,
    dtype: torch.dtype,
):
    from sae_bench.custom_saes.batch_topk_sae import BatchTopKSAE
    from sae_bench.custom_saes.jumprelu_sae import JumpReluSAE
    from sae_bench.custom_saes.relu_sae import ReluSAE
    from sae_bench.custom_saes.topk_sae import TopKSAE

    kwargs = {
        "d_in": d_model,
        "d_sae": d_sae,
        "model_name": model_name,
        "hook_layer": hook_layer,
        "hook_name": hook_name,
        "device": torch.device(device),
        "dtype": dtype,
    }

    if architecture == "topk":
        return TopKSAE(k=k, **kwargs)
    if architecture == "batchtopk":
        return BatchTopKSAE(k=k, **kwargs)
    if architecture == "relu":
        return ReluSAE(**kwargs)
    if architecture == "jumprelu":
        return JumpReluSAE(**kwargs)

    raise ValueError(f"Unsupported architecture: {architecture}")


def initialize_model_for_training(
    *,
    model: torch.nn.Module,
    activations: torch.Tensor,
    architecture: str,
    device: str,
) -> None:
    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(model.W_enc, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(model.W_dec, a=math.sqrt(5))
        model.W_dec.data = F.normalize(model.W_dec.data, dim=1)
        model.b_enc.zero_()

        # Center decoder bias on activation mean to avoid dead-start dynamics.
        act_mean = activations.mean(dim=0).to(device=device, dtype=model.b_dec.dtype)
        model.b_dec.copy_(act_mean)

        if architecture == "jumprelu" and hasattr(model, "threshold"):
            if isinstance(model.threshold, torch.nn.Parameter):
                model.threshold.data.zero_()
            else:
                model.threshold.zero_()


def train_single(
    *,
    architecture: str,
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    dtype: torch.dtype,
    model_name: str,
    hook_layer: int,
    hook_name: str,
    relu_l1_coef: float,
    jumprelu_l0_coef: float,
) -> tuple[torch.nn.Module, TrainMetrics]:
    set_seed(seed)

    d_model = int(activations.shape[1])
    model = build_model(
        architecture=architecture,
        d_model=d_model,
        d_sae=d_sae,
        k=k,
        model_name=model_name,
        hook_layer=hook_layer,
        hook_name=hook_name,
        device=device,
        dtype=dtype,
    )
    initialize_model_for_training(model=model, activations=activations, architecture=architecture, device=device)
    # BatchTopK requires thresholded inference, but threshold is only meaningful
    # after training/calibration. Train in explicit top-k mode first.
    if architecture == "batchtopk" and hasattr(model, "use_threshold"):
        model.use_threshold = False
    model.train()

    dataset = TensorDataset(activations)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            recon = model(batch)
            feats = model.encode(batch)
            mse = F.mse_loss(recon, batch)

            if architecture == "relu":
                sparse_penalty = relu_l1_coef * feats.abs().mean()
            elif architecture == "jumprelu":
                sparse_penalty = jumprelu_l0_coef * (feats > 0).float().mean()
            else:
                sparse_penalty = torch.tensor(0.0, device=batch.device)

            loss = mse + sparse_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if hasattr(model, "W_dec"):
                with torch.no_grad():
                    model.W_dec.data = F.normalize(model.W_dec.data, dim=1)

    if architecture == "batchtopk":
        calibrate_batchtopk_threshold(model=model, activations=activations, batch_size=batch_size, device=device)

    model.eval()
    with torch.no_grad():
        full = activations.to(device)
        recon = model(full)
        feats = model.encode(full)
        mse = float(F.mse_loss(recon, full).item())
        total_var = float(full.var().item())
        residual_var = float((full - recon).var().item())
        ev = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0
        l0 = float((feats > 0).float().sum(dim=-1).mean().item())

    return model.cpu(), TrainMetrics(mse=mse, explained_variance=ev, l0=l0)


def calibrate_batchtopk_threshold(
    *,
    model: torch.nn.Module,
    activations: torch.Tensor,
    batch_size: int,
    device: str,
) -> float:
    if not hasattr(model, "k") or not hasattr(model, "threshold") or not hasattr(model, "use_threshold"):
        raise ValueError("BatchTopK calibration requested on non-BatchTopK model")

    model.use_threshold = False
    k = int(model.k.item()) if hasattr(model.k, "item") else int(model.k)
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    kth_values: list[torch.Tensor] = []

    with torch.no_grad():
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            post_relu = F.relu((batch - model.b_dec) @ model.W_enc + model.b_enc)
            kth = torch.topk(post_relu, k=k, dim=-1, sorted=True).values[:, -1]
            kth_values.append(kth.detach().float().cpu())

    if not kth_values:
        raise ValueError("No activations available for BatchTopK threshold calibration")

    kth_concat = torch.cat(kth_values, dim=0)
    positive_kth = kth_concat[kth_concat > 0]
    if positive_kth.numel() > 0:
        threshold = float(positive_kth.median().item())
    else:
        threshold = 0.0
    model.threshold.copy_(torch.tensor(threshold, device=model.threshold.device, dtype=model.threshold.dtype))
    model.use_threshold = True
    return threshold


def write_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    architecture: str,
    d_model: int,
    d_sae: int,
    k: int,
    seed: int,
) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "d_model": d_model,
        "d_sae": d_sae,
        "architecture": architecture,
        "seed": seed,
    }
    if architecture in {"topk", "batchtopk"}:
        payload["k"] = k
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    cmd = cmd_to_str(command)
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def summary_stats(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-budget architecture frontier on external benchmarks")
    parser.add_argument("--activation-cache-dir", type=Path, default=DEFAULT_PYTHIA70M_ACTIVATION_CACHE)
    parser.add_argument("--activation-glob", type=str, default="*_blocks.0.hook_resid_pre.pt")
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-rows-per-file", type=int, default=2048)
    parser.add_argument("--max-total-rows", type=int, default=150000)

    parser.add_argument("--architectures", type=str, default="topk,relu,batchtopk,jumprelu")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")

    parser.add_argument("--run-saebench", action="store_true")
    parser.add_argument("--run-cebench", action="store_true")
    parser.add_argument("--cebench-repo", type=Path, default=None)
    parser.add_argument("--cebench-max-rows", type=int, default=None)

    parser.add_argument("--relu-l1-coef", type=float, default=1e-3)
    parser.add_argument("--jumprelu-l0-coef", type=float, default=1e-3)

    parser.add_argument(
        "--saebench-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "husai_saebench_probe_results_frontier",
    )
    parser.add_argument("--saebench-model-cache-path", type=Path, default=DEFAULT_SAEBENCH_MODEL_CACHE)
    parser.add_argument("--cebench-artifacts-path", type=Path, default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "ce_bench_artifacts_frontier")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4b_architecture_frontier_external",
    )
    args = parser.parse_args()

    args.activation_cache_dir = to_abs_repo_path(args.activation_cache_dir)
    args.saebench_results_path = to_abs_repo_path(args.saebench_results_path)
    args.saebench_model_cache_path = to_abs_repo_path(args.saebench_model_cache_path)
    args.cebench_artifacts_path = to_abs_repo_path(args.cebench_artifacts_path)
    args.output_dir = to_abs_repo_path(args.output_dir)
    if args.cebench_repo is not None:
        args.cebench_repo = to_abs_repo_path(args.cebench_repo)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    architectures = parse_architectures(args.architectures)
    seeds = parse_csv_ints(args.seeds)
    sae_dtype = dtype_from_name(args.dtype)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "external_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    activations, files_used, data_meta = load_activation_bank(
        cache_dir=args.activation_cache_dir,
        activation_glob=args.activation_glob,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
        max_total_rows=args.max_total_rows,
        seed=min(seeds) if seeds else 0,
    )
    d_model = int(activations.shape[1])
    dataset_names = infer_dataset_names_from_files(files_used, args.hook_name)

    records: list[dict[str, Any]] = []

    for arch in architectures:
        for seed in seeds:
            condition_id = f"{arch}_seed{seed}"
            ckpt_dir = checkpoints_dir / condition_id
            ckpt_path = ckpt_dir / "sae_final.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            model, train_metrics = train_single(
                architecture=arch,
                activations=activations,
                d_sae=args.d_sae,
                k=args.k,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                dtype=sae_dtype,
                model_name=args.model_name,
                hook_layer=args.hook_layer,
                hook_name=args.hook_name,
                relu_l1_coef=args.relu_l1_coef,
                jumprelu_l0_coef=args.jumprelu_l0_coef,
            )

            write_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                architecture=arch,
                d_model=d_model,
                d_sae=args.d_sae,
                k=args.k,
                seed=seed,
            )

            rec: dict[str, Any] = {
                "architecture": arch,
                "seed": seed,
                "checkpoint": to_repo_rel(ckpt_path),
                "train_metrics": asdict(train_metrics),
                "saebench": None,
                "cebench": None,
            }

            if args.run_saebench:
                out_dir = eval_dir / condition_id / "saebench"
                command = [
                    sys.executable,
                    "scripts/experiments/run_husai_saebench_custom_eval.py",
                    "--checkpoint",
                    str(ckpt_path),
                    "--architecture",
                    arch,
                    "--sae-release",
                    f"husai_{condition_id}",
                    "--model-name",
                    args.model_name,
                    "--hook-layer",
                    str(args.hook_layer),
                    "--hook-name",
                    args.hook_name,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--results-path",
                    str(args.saebench_results_path),
                    "--model-cache-path",
                    str(args.saebench_model_cache_path),
                    "--output-dir",
                    str(out_dir),
                    "--force-rerun",
                ]
                if dataset_names:
                    command.extend(["--dataset-names", ",".join(dataset_names)])
                rc, output = run_subprocess(command, PROJECT_ROOT)
                (logs_dir / f"{condition_id}_saebench.log").write_text(output)
                rec["saebench_returncode"] = rc
                summary_path = out_dir / "husai_custom_sae_summary.json"
                if summary_path.exists():
                    rec["saebench"] = json.loads(summary_path.read_text())

            if args.run_cebench:
                if args.cebench_repo is None:
                    raise ValueError("--cebench-repo is required when --run-cebench is set")
                out_dir = eval_dir / condition_id / "cebench"
                command = [
                    sys.executable,
                    "scripts/experiments/run_husai_cebench_custom_eval.py",
                    "--cebench-repo",
                    str(args.cebench_repo),
                    "--checkpoint",
                    str(ckpt_path),
                    "--architecture",
                    arch,
                    "--sae-release",
                    f"husai_{condition_id}",
                    "--model-name",
                    args.model_name,
                    "--hook-layer",
                    str(args.hook_layer),
                    "--hook-name",
                    args.hook_name,
                    "--device",
                    args.device,
                    "--sae-dtype",
                    args.dtype,
                    "--output-folder",
                    str(out_dir),
                    "--artifacts-path",
                    str(args.cebench_artifacts_path),
                ]
                if args.cebench_max_rows is not None:
                    command.extend(["--max-rows", str(args.cebench_max_rows)])
                rc, output = run_subprocess(command, PROJECT_ROOT)
                (logs_dir / f"{condition_id}_cebench.log").write_text(output)
                rec["cebench_returncode"] = rc
                summary_path = out_dir / "husai_custom_cebench_summary.json"
                if summary_path.exists():
                    rec["cebench"] = json.loads(summary_path.read_text())

            records.append(rec)

    aggregate: dict[str, Any] = {}
    for arch in architectures:
        rows = [r for r in records if r["architecture"] == arch]
        aggregate[arch] = {
            "train_mse": summary_stats([r["train_metrics"]["mse"] for r in rows]),
            "train_ev": summary_stats([r["train_metrics"]["explained_variance"] for r in rows]),
            "train_l0": summary_stats([r["train_metrics"]["l0"] for r in rows]),
            "saebench_best_auc": summary_stats(
                [
                    maybe_float((r.get("saebench") or {}).get("summary", {}).get("best_by_auc", {}).get("test_auc"))
                    for r in rows
                ]
            ),
            "saebench_best_minus_llm_auc": summary_stats(
                [maybe_float((r.get("saebench") or {}).get("summary", {}).get("best_minus_llm_auc")) for r in rows]
            ),
            "cebench_contrastive_max": summary_stats(
                [
                    maybe_float((r.get("cebench") or {}).get("custom_metrics", {}).get("contrastive_score_mean_max"))
                    for r in rows
                ]
            ),
            "cebench_independent_max": summary_stats(
                [
                    maybe_float((r.get("cebench") or {}).get("custom_metrics", {}).get("independent_score_mean_max"))
                    for r in rows
                ]
            ),
            "cebench_interpretability_max": summary_stats(
                [
                    maybe_float((r.get("cebench") or {}).get("custom_metrics", {}).get("interpretability_score_mean_max"))
                    for r in rows
                ]
            ),
        }

    payload = {
        "timestamp_utc": utc_now(),
        "command": " ".join(["python", *sys.argv]),
        "config": {
            "activation_cache_dir": str(args.activation_cache_dir),
            "activation_glob": args.activation_glob,
            "max_files": args.max_files,
            "max_rows_per_file": args.max_rows_per_file,
            "max_total_rows": args.max_total_rows,
            "architectures": architectures,
            "seeds": seeds,
            "d_sae": args.d_sae,
            "k": args.k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "dtype": args.dtype,
            "model_name": args.model_name,
            "hook_layer": args.hook_layer,
            "hook_name": args.hook_name,
            "run_saebench": args.run_saebench,
            "run_cebench": args.run_cebench,
            "cebench_repo": str(args.cebench_repo) if args.cebench_repo else None,
            "cebench_max_rows": args.cebench_max_rows,
            "relu_l1_coef": args.relu_l1_coef,
            "jumprelu_l0_coef": args.jumprelu_l0_coef,
            "saebench_results_path": str(args.saebench_results_path),
            "saebench_model_cache_path": str(args.saebench_model_cache_path),
            "cebench_artifacts_path": str(args.cebench_artifacts_path),
            "data_meta": data_meta,
            "source_files": files_used,
            "dataset_names_count": len(dataset_names),
            "run_id": run_id,
        },
        "records": records,
        "aggregate": aggregate,
    }

    out_json = run_dir / "results.json"
    out_md = run_dir / "summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Architecture Frontier External",
        "",
        f"- Run ID: `{run_id}`",
        f"- Architectures: `{architectures}`",
        f"- Seeds: `{seeds}`",
        f"- d_sae / k: `{args.d_sae}` / `{args.k}`",
        f"- Rows used: `{data_meta['total_rows']}`",
        "",
        "| architecture | train EV mean | SAEBench best-LLM AUC mean | CE-Bench interpretability max mean |",
        "|---|---:|---:|---:|",
    ]

    for arch in architectures:
        row = aggregate[arch]
        lines.append(
            "| "
            f"{arch} | "
            f"{row['train_ev']['mean']} | "
            f"{row['saebench_best_minus_llm_auc']['mean']} | "
            f"{row['cebench_interpretability_max']['mean']} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print("Architecture frontier run complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
