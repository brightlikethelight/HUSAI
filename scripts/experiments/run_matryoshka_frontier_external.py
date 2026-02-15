#!/usr/bin/env python3
"""Run a matched-budget Matryoshka TopK frontier on external benchmarks.

This adds a new architecture family under matched budget by training a TopK SAE
with nested-prefix reconstruction losses (Matryoshka-style objective).
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

from scripts.experiments.husai_custom_sae_adapter import dtype_from_name  # noqa: E402
from scripts.experiments.train_husai_sae_on_cached_activations import (  # noqa: E402
    DEFAULT_PYTHIA70M_ACTIVATION_CACHE,
    DEFAULT_SAEBENCH_MODEL_CACHE,
    load_activation_bank,
    set_seed,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path) -> str:
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def cmd_to_str(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(proc.returncode), proc.stdout


def infer_dataset_names(files: list[str], hook_name: str) -> list[str]:
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


@dataclass
class TrainMetrics:
    mse: float
    explained_variance: float
    l0: float
    prefix_mse: float


def build_topk_model(
    *,
    d_model: int,
    d_sae: int,
    k: int,
    model_name: str,
    hook_layer: int,
    hook_name: str,
    device: str,
    dtype: torch.dtype,
):
    from sae_bench.custom_saes.topk_sae import TopKSAE

    model = TopKSAE(
        d_in=d_model,
        d_sae=d_sae,
        k=k,
        model_name=model_name,
        hook_layer=hook_layer,
        hook_name=hook_name,
        device=torch.device(device),
        dtype=dtype,
    )
    return model


def train_matryoshka_topk(
    *,
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
    matryoshka_levels: list[int],
    matryoshka_coef: float,
) -> tuple[torch.nn.Module, TrainMetrics]:
    set_seed(seed)

    d_model = int(activations.shape[1])
    model = build_topk_model(
        d_model=d_model,
        d_sae=d_sae,
        k=k,
        model_name=model_name,
        hook_layer=hook_layer,
        hook_name=hook_name,
        device=device,
        dtype=dtype,
    )
    model.train()

    dataset = TensorDataset(activations)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    levels = sorted(set([lvl for lvl in matryoshka_levels if 0 < lvl <= d_sae]))
    if not levels:
        levels = [min(k, d_sae)]

    for _ in range(epochs):
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            recon = model(batch)
            feats = model.encode(batch)
            mse = F.mse_loss(recon, batch)

            prefix_losses: list[torch.Tensor] = []
            for lvl in levels:
                feats_prefix = feats.clone()
                feats_prefix[:, lvl:] = 0.0
                recon_prefix = feats_prefix @ model.W_dec + model.b_dec
                prefix_losses.append(F.mse_loss(recon_prefix, batch))

            prefix_loss = torch.stack(prefix_losses).mean() if prefix_losses else torch.tensor(0.0, device=batch.device)
            loss = mse + matryoshka_coef * prefix_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.W_dec.data = F.normalize(model.W_dec.data, dim=1)

    model.eval()
    with torch.no_grad():
        full = activations.to(device)
        recon = model(full)
        feats = model.encode(full)
        mse = float(F.mse_loss(recon, full).item())

        prefix_eval: list[float] = []
        for lvl in levels:
            feats_prefix = feats.clone()
            feats_prefix[:, lvl:] = 0.0
            recon_prefix = feats_prefix @ model.W_dec + model.b_dec
            prefix_eval.append(float(F.mse_loss(recon_prefix, full).item()))

        total_var = float(full.var().item())
        residual_var = float((full - recon).var().item())
        ev = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0
        l0 = float((feats > 0).float().sum(dim=-1).mean().item())

    return model.cpu(), TrainMetrics(mse=mse, explained_variance=ev, l0=l0, prefix_mse=float(np.mean(prefix_eval)))


def write_checkpoint(path: Path, model: torch.nn.Module, d_model: int, d_sae: int, k: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "d_model": d_model,
            "d_sae": d_sae,
            "k": k,
            "seed": seed,
            "architecture": "matryoshka",
            "consistency_objective": "matryoshka_prefix_reconstruction",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-budget Matryoshka frontier external run")
    parser.add_argument("--activation-cache-dir", type=Path, default=DEFAULT_PYTHIA70M_ACTIVATION_CACHE)
    parser.add_argument("--activation-glob", type=str, default="*_blocks.0.hook_resid_pre.pt")
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-rows-per-file", type=int, default=2048)
    parser.add_argument("--max-total-rows", type=int, default=150000)

    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--d-sae", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--matryoshka-levels", type=str, default="256,512,1024")
    parser.add_argument("--matryoshka-coef", type=float, default=0.2)

    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")

    parser.add_argument("--run-saebench", action="store_true")
    parser.add_argument("--run-cebench", action="store_true")
    parser.add_argument("--cebench-repo", type=Path, default=None)
    parser.add_argument("--cebench-max-rows", type=int, default=None)
    parser.add_argument("--cebench-matched-baseline-summary", type=Path, default=None)
    parser.add_argument("--saebench-datasets", type=str, default="")
    parser.add_argument("--saebench-dataset-limit", type=int, default=0)

    parser.add_argument(
        "--saebench-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "husai_saebench_probe_results_frontier_matryoshka",
    )
    parser.add_argument("--saebench-model-cache-path", type=Path, default=DEFAULT_SAEBENCH_MODEL_CACHE)
    parser.add_argument(
        "--cebench-artifacts-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "ce_bench_artifacts_frontier_matryoshka",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4b_matryoshka_frontier_external",
    )
    args = parser.parse_args()

    args.activation_cache_dir = to_abs(args.activation_cache_dir)
    args.saebench_results_path = to_abs(args.saebench_results_path)
    args.saebench_model_cache_path = to_abs(args.saebench_model_cache_path)
    args.cebench_artifacts_path = to_abs(args.cebench_artifacts_path)
    args.output_dir = to_abs(args.output_dir)
    if args.cebench_repo is not None:
        args.cebench_repo = to_abs(args.cebench_repo)
    if args.cebench_matched_baseline_summary is not None:
        args.cebench_matched_baseline_summary = to_abs(args.cebench_matched_baseline_summary)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.run_cebench and args.cebench_repo is None:
        raise ValueError("--cebench-repo is required when --run-cebench is set")

    seeds = parse_ints(args.seeds)
    levels = parse_ints(args.matryoshka_levels)
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
        seed=min(seeds),
    )
    d_model = int(activations.shape[1])

    dataset_names = infer_dataset_names(files_used, args.hook_name)
    if args.saebench_datasets:
        dataset_names = parse_csv_strings(args.saebench_datasets)
    if args.saebench_dataset_limit > 0:
        dataset_names = dataset_names[: args.saebench_dataset_limit]

    records: list[dict[str, Any]] = []

    for seed in seeds:
        condition_id = f"matryoshka_seed{seed}"
        ckpt_dir = checkpoints_dir / condition_id
        ckpt_path = ckpt_dir / "sae_final.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model, train_metrics = train_matryoshka_topk(
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
            matryoshka_levels=levels,
            matryoshka_coef=args.matryoshka_coef,
        )

        write_checkpoint(ckpt_path, model, d_model, args.d_sae, args.k, seed)

        rec: dict[str, Any] = {
            "architecture": "matryoshka",
            "seed": seed,
            "checkpoint": repo_rel(ckpt_path),
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
                "matryoshka",
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
            out_dir = eval_dir / condition_id / "cebench"
            command = [
                sys.executable,
                "scripts/experiments/run_husai_cebench_custom_eval.py",
                "--cebench-repo",
                str(args.cebench_repo),
                "--checkpoint",
                str(ckpt_path),
                "--architecture",
                "matryoshka",
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
            if args.cebench_matched_baseline_summary is not None:
                command.extend(["--matched-baseline-summary", str(args.cebench_matched_baseline_summary)])
            rc, output = run_subprocess(command, PROJECT_ROOT)
            (logs_dir / f"{condition_id}_cebench.log").write_text(output)
            rec["cebench_returncode"] = rc
            summary_path = out_dir / "husai_custom_cebench_summary.json"
            if summary_path.exists():
                rec["cebench"] = json.loads(summary_path.read_text())

        records.append(rec)

    aggregate = {
        "train_mse": summary_stats([r["train_metrics"]["mse"] for r in records]),
        "train_ev": summary_stats([r["train_metrics"]["explained_variance"] for r in records]),
        "train_l0": summary_stats([r["train_metrics"]["l0"] for r in records]),
        "train_prefix_mse": summary_stats([r["train_metrics"]["prefix_mse"] for r in records]),
        "saebench_best_minus_llm_auc": summary_stats(
            [maybe_float((r.get("saebench") or {}).get("summary", {}).get("best_minus_llm_auc")) for r in records]
        ),
        "cebench_interpretability_max": summary_stats(
            [
                maybe_float((r.get("cebench") or {}).get("custom_metrics", {}).get("interpretability_score_mean_max"))
                for r in records
            ]
        ),
        "cebench_interp_delta_vs_baseline": summary_stats(
            [
                maybe_float(
                    (r.get("cebench") or {}).get("delta_vs_matched_baseline", {}).get("interpretability_score_mean_max")
                )
                for r in records
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
            "architecture": "matryoshka",
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
            "cebench_matched_baseline_summary": (
                str(args.cebench_matched_baseline_summary) if args.cebench_matched_baseline_summary else None
            ),
            "saebench_datasets": parse_csv_strings(args.saebench_datasets),
            "saebench_dataset_limit": args.saebench_dataset_limit,
            "matryoshka_levels": levels,
            "matryoshka_coef": args.matryoshka_coef,
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
        "# Matryoshka Frontier External",
        "",
        f"- Run ID: `{run_id}`",
        f"- Seeds: `{seeds}`",
        f"- d_sae / k: `{args.d_sae}` / `{args.k}`",
        f"- Matryoshka levels: `{levels}`",
        f"- Matryoshka coef: `{args.matryoshka_coef}`",
        f"- Rows used: `{data_meta['total_rows']}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
        f"| train_ev | {aggregate['train_ev']['mean']} | {aggregate['train_ev']['std']} |",
        f"| train_prefix_mse | {aggregate['train_prefix_mse']['mean']} | {aggregate['train_prefix_mse']['std']} |",
        f"| saebench_best_minus_llm_auc | {aggregate['saebench_best_minus_llm_auc']['mean']} | {aggregate['saebench_best_minus_llm_auc']['std']} |",
        f"| cebench_interpretability_max | {aggregate['cebench_interpretability_max']['mean']} | {aggregate['cebench_interpretability_max']['std']} |",
        f"| cebench_interp_delta_vs_baseline | {aggregate['cebench_interp_delta_vs_baseline']['mean']} | {aggregate['cebench_interp_delta_vs_baseline']['std']} |",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("Matryoshka frontier run complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
