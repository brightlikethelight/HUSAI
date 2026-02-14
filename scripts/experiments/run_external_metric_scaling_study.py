#!/usr/bin/env python3
"""Run external-metric scaling studies over token budget, hook layer, and d_sae."""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def infer_dataset_names_from_files(files: list[Path], hook_name: str) -> list[str]:
    suffix = f"_{hook_name}.pt"
    names: list[str] = []
    for path in files:
        name = path.name
        if name.endswith(suffix):
            names.append(name[: -len(suffix)])
        else:
            names.append(path.stem)

    seen = set()
    deduped: list[str] = []
    for name in names:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="External metric scaling study")
    parser.add_argument(
        "--activation-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "sae_bench_model_cache" / "model_activations_pythia-70m-deduped",
    )
    parser.add_argument("--activation-glob-template", type=str, default="*_blocks.{layer}.hook_resid_pre.pt")
    parser.add_argument("--hook-name-template", type=str, default="blocks.{layer}.hook_resid_pre")

    parser.add_argument("--token-budgets", type=str, default="50000,100000,150000")
    parser.add_argument("--hook-layers", type=str, default="0")
    parser.add_argument("--d-sae-values", type=str, default="1024,2048")
    parser.add_argument("--seeds", type=str, default="42")

    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-rows-per-file", type=int, default=2048)

    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--run-saebench", action="store_true")
    parser.add_argument("--run-cebench", action="store_true")
    parser.add_argument("--cebench-repo", type=Path, default=None)
    parser.add_argument("--cebench-max-rows", type=int, default=None)
    parser.add_argument("--cebench-matched-baseline-summary", type=Path, default=None)

    parser.add_argument(
        "--saebench-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "husai_saebench_probe_results_scaling",
    )
    parser.add_argument(
        "--saebench-model-cache-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "sae_bench_model_cache",
    )
    parser.add_argument(
        "--cebench-artifacts-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "ce_bench_artifacts_scaling",
    )
    parser.add_argument("--saebench-datasets", type=str, default="")
    parser.add_argument("--saebench-dataset-limit", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_scaling_study",
    )
    args = parser.parse_args()

    args.activation_cache_dir = to_abs_repo_path(args.activation_cache_dir)
    args.saebench_results_path = to_abs_repo_path(args.saebench_results_path)
    args.saebench_model_cache_path = to_abs_repo_path(args.saebench_model_cache_path)
    args.cebench_artifacts_path = to_abs_repo_path(args.cebench_artifacts_path)
    args.output_dir = to_abs_repo_path(args.output_dir)
    if args.cebench_repo is not None:
        args.cebench_repo = to_abs_repo_path(args.cebench_repo)
    if args.cebench_matched_baseline_summary is not None:
        args.cebench_matched_baseline_summary = to_abs_repo_path(args.cebench_matched_baseline_summary)

    token_budgets = parse_ints(args.token_budgets)
    hook_layers = parse_ints(args.hook_layers)
    d_sae_values = parse_ints(args.d_sae_values)
    seeds = parse_ints(args.seeds)

    if args.run_cebench and args.cebench_repo is None:
        raise ValueError("--cebench-repo is required when --run-cebench is set")

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "external_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for token_budget, hook_layer, d_sae, seed in itertools.product(token_budgets, hook_layers, d_sae_values, seeds):
        cond_id = f"tok{token_budget}_layer{hook_layer}_dsae{d_sae}_seed{seed}"
        hook_name = args.hook_name_template.format(layer=hook_layer)
        activation_glob = args.activation_glob_template.format(layer=hook_layer)
        activation_files = sorted(args.activation_cache_dir.glob(activation_glob))

        if args.saebench_datasets:
            dataset_names = parse_csv_strings(args.saebench_datasets)
        else:
            dataset_names = infer_dataset_names_from_files(activation_files, hook_name)
        if args.saebench_dataset_limit > 0:
            dataset_names = dataset_names[: args.saebench_dataset_limit]

        if args.run_saebench and not dataset_names:
            raise ValueError(
                f"No SAEBench dataset names resolved for condition {cond_id}. "
                "Pass --saebench-datasets or ensure activation files match hook naming."
            )

        output_ckpt_dir = ckpt_dir / cond_id
        output_ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_ckpt_dir / "sae_final.pt"

        train_cmd = [
            sys.executable,
            "scripts/experiments/train_husai_sae_on_cached_activations.py",
            "--activation-cache-dir",
            str(args.activation_cache_dir),
            "--activation-glob",
            activation_glob,
            "--max-files",
            str(args.max_files),
            "--max-rows-per-file",
            str(args.max_rows_per_file),
            "--max-total-rows",
            str(token_budget),
            "--d-sae",
            str(d_sae),
            "--k",
            str(args.k),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output-dir",
            str(output_ckpt_dir),
        ]
        rc, output = run_subprocess(train_cmd, PROJECT_ROOT)
        (logs_dir / f"{cond_id}_train.log").write_text(output)

        rec: dict[str, Any] = {
            "condition_id": cond_id,
            "token_budget": token_budget,
            "hook_layer": hook_layer,
            "hook_name": hook_name,
            "d_sae": d_sae,
            "seed": seed,
            "train_returncode": rc,
            "train_summary": None,
            "activation_file_count": len(activation_files),
            "dataset_names": dataset_names,
            "saebench": None,
            "cebench": None,
            "checkpoint": to_repo_rel(checkpoint_path) if checkpoint_path.exists() else None,
        }

        train_summary_path = output_ckpt_dir / "summary.json"
        if train_summary_path.exists():
            rec["train_summary"] = json.loads(train_summary_path.read_text())

        if args.run_saebench and checkpoint_path.exists():
            saebench_out = eval_dir / cond_id / "saebench"
            sae_cmd = [
                sys.executable,
                "scripts/experiments/run_husai_saebench_custom_eval.py",
                "--checkpoint",
                str(checkpoint_path),
                "--architecture",
                "topk",
                "--sae-release",
                f"husai_scaling_{cond_id}",
                "--model-name",
                args.model_name,
                "--hook-layer",
                str(hook_layer),
                "--hook-name",
                hook_name,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--results-path",
                str(args.saebench_results_path),
                "--model-cache-path",
                str(args.saebench_model_cache_path),
                "--output-dir",
                str(saebench_out),
                "--force-rerun",
            ]
            if dataset_names:
                sae_cmd.extend(["--dataset-names", ",".join(dataset_names)])
            rc, output = run_subprocess(sae_cmd, PROJECT_ROOT)
            (logs_dir / f"{cond_id}_saebench.log").write_text(output)
            rec["saebench_returncode"] = rc
            summary_path = saebench_out / "husai_custom_sae_summary.json"
            if summary_path.exists():
                rec["saebench"] = json.loads(summary_path.read_text())

        if args.run_cebench and checkpoint_path.exists():
            cebench_out = eval_dir / cond_id / "cebench"
            ce_cmd = [
                sys.executable,
                "scripts/experiments/run_husai_cebench_custom_eval.py",
                "--cebench-repo",
                str(args.cebench_repo),
                "--checkpoint",
                str(checkpoint_path),
                "--architecture",
                "topk",
                "--sae-release",
                f"husai_scaling_{cond_id}",
                "--model-name",
                args.model_name,
                "--hook-layer",
                str(hook_layer),
                "--hook-name",
                hook_name,
                "--device",
                args.device,
                "--sae-dtype",
                args.dtype,
                "--output-folder",
                str(cebench_out),
                "--artifacts-path",
                str(args.cebench_artifacts_path),
            ]
            if args.cebench_max_rows is not None:
                ce_cmd.extend(["--max-rows", str(args.cebench_max_rows)])
            if args.cebench_matched_baseline_summary is not None:
                ce_cmd.extend(["--matched-baseline-summary", str(args.cebench_matched_baseline_summary)])
            rc, output = run_subprocess(ce_cmd, PROJECT_ROOT)
            (logs_dir / f"{cond_id}_cebench.log").write_text(output)
            rec["cebench_returncode"] = rc
            summary_path = cebench_out / "husai_custom_cebench_summary.json"
            if summary_path.exists():
                rec["cebench"] = json.loads(summary_path.read_text())

        records.append(rec)

    # Aggregate by each scaling axis.
    def extract_saebench_delta(record: dict[str, Any]) -> float | None:
        return maybe_float((record.get("saebench") or {}).get("summary", {}).get("best_minus_llm_auc"))

    def extract_cebench_interp(record: dict[str, Any]) -> float | None:
        return maybe_float((record.get("cebench") or {}).get("custom_metrics", {}).get("interpretability_score_mean_max"))

    def extract_cebench_delta(record: dict[str, Any]) -> float | None:
        return maybe_float(
            (record.get("cebench") or {}).get("delta_vs_matched_baseline", {}).get("interpretability_score_mean_max")
        )

    aggregates: dict[str, Any] = {
        "by_token_budget": {},
        "by_hook_layer": {},
        "by_d_sae": {},
    }

    for token_budget in token_budgets:
        rows = [r for r in records if r["token_budget"] == token_budget]
        aggregates["by_token_budget"][str(token_budget)] = {
            "saebench_best_minus_llm_auc": summary_stats([extract_saebench_delta(r) for r in rows]),
            "cebench_interpretability_max": summary_stats([extract_cebench_interp(r) for r in rows]),
            "cebench_interp_delta_vs_baseline": summary_stats([extract_cebench_delta(r) for r in rows]),
        }

    for hook_layer in hook_layers:
        rows = [r for r in records if r["hook_layer"] == hook_layer]
        aggregates["by_hook_layer"][str(hook_layer)] = {
            "saebench_best_minus_llm_auc": summary_stats([extract_saebench_delta(r) for r in rows]),
            "cebench_interpretability_max": summary_stats([extract_cebench_interp(r) for r in rows]),
            "cebench_interp_delta_vs_baseline": summary_stats([extract_cebench_delta(r) for r in rows]),
        }

    for d_sae in d_sae_values:
        rows = [r for r in records if r["d_sae"] == d_sae]
        aggregates["by_d_sae"][str(d_sae)] = {
            "saebench_best_minus_llm_auc": summary_stats([extract_saebench_delta(r) for r in rows]),
            "cebench_interpretability_max": summary_stats([extract_cebench_interp(r) for r in rows]),
            "cebench_interp_delta_vs_baseline": summary_stats([extract_cebench_delta(r) for r in rows]),
        }

    payload = {
        "timestamp_utc": utc_now(),
        "command": " ".join(["python", *sys.argv]),
        "config": {
            "activation_cache_dir": str(args.activation_cache_dir),
            "activation_glob_template": args.activation_glob_template,
            "hook_name_template": args.hook_name_template,
            "token_budgets": token_budgets,
            "hook_layers": hook_layers,
            "d_sae_values": d_sae_values,
            "seeds": seeds,
            "k": args.k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_files": args.max_files,
            "max_rows_per_file": args.max_rows_per_file,
            "model_name": args.model_name,
            "device": args.device,
            "dtype": args.dtype,
            "run_saebench": args.run_saebench,
            "run_cebench": args.run_cebench,
            "cebench_repo": str(args.cebench_repo) if args.cebench_repo else None,
            "cebench_max_rows": args.cebench_max_rows,
            "cebench_matched_baseline_summary": (
                str(args.cebench_matched_baseline_summary) if args.cebench_matched_baseline_summary else None
            ),
            "saebench_datasets": parse_csv_strings(args.saebench_datasets),
            "saebench_dataset_limit": args.saebench_dataset_limit,
            "run_id": run_id,
        },
        "records": records,
        "aggregates": aggregates,
    }

    out_json = run_dir / "results.json"
    out_md = run_dir / "summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# External Metric Scaling Study",
        "",
        f"- Run ID: `{run_id}`",
        f"- token budgets: `{token_budgets}`",
        f"- hook layers: `{hook_layers}`",
        f"- d_sae values: `{d_sae_values}`",
        f"- seeds: `{seeds}`",
        "",
        "## Aggregates by Token Budget",
        "",
        "| token_budget | SAEBench best-LLM AUC mean | CE-Bench interpretability max mean | CE-Bench interp delta vs baseline mean |",
        "|---:|---:|---:|---:|",
    ]

    for token_budget in token_budgets:
        row = aggregates["by_token_budget"][str(token_budget)]
        lines.append(
            "| "
            f"{token_budget} | "
            f"{row['saebench_best_minus_llm_auc']['mean']} | "
            f"{row['cebench_interpretability_max']['mean']} | "
            f"{row['cebench_interp_delta_vs_baseline']['mean']} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print("External scaling study complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
