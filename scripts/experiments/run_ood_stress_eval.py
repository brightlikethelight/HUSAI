#!/usr/bin/env python3
"""Run OOD stress evaluation for a HUSAI SAE checkpoint using SAEBench probes.

This script evaluates a single checkpoint on two disjoint dataset slices:
- in-domain (ID)
- out-of-domain (OOD)

It emits a gate-friendly summary artifact with `ood_drop` where:
`ood_drop = id_best_minus_llm_auc - ood_best_minus_llm_auc`
(higher means larger degradation under OOD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_husai_saebench_custom_eval import (  # noqa: E402
    infer_dataset_names_from_cache,
)


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


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def dataset_domain(name: str) -> str:
    # Strip numeric prefix: "123_foo_bar" -> "foo_bar"
    stripped = re.sub(r"^\d+_", "", name)
    if "_" in stripped:
        return stripped.split("_", 1)[0]
    return stripped


def auto_partition_datasets(
    datasets: list[str],
    ood_fraction: float,
    min_ood_datasets: int,
) -> tuple[list[str], list[str], list[str]]:
    if len(datasets) < 2:
        raise ValueError("Need at least two datasets for ID/OOD partitioning.")

    by_domain: dict[str, list[str]] = {}
    for name in datasets:
        dom = dataset_domain(name)
        by_domain.setdefault(dom, []).append(name)

    domains = sorted(by_domain.keys())
    if len(domains) < 2:
        # Fallback: split by dataset index.
        split = max(1, min(len(datasets) - 1, int(round(len(datasets) * ood_fraction))))
        id_names = datasets[:-split]
        ood_names = datasets[-split:]
        return id_names, ood_names, ["__index_split__"]

    ood_domain_count = int(round(len(domains) * ood_fraction))
    ood_domain_count = max(1, min(len(domains) - 1, ood_domain_count))
    ood_domains = domains[-ood_domain_count:]

    ood_names: list[str] = []
    id_names: list[str] = []
    for name in datasets:
        if dataset_domain(name) in ood_domains:
            ood_names.append(name)
        else:
            id_names.append(name)

    if len(ood_names) < min_ood_datasets:
        need = min_ood_datasets - len(ood_names)
        shift = id_names[-need:]
        id_names = id_names[:-need]
        ood_names = shift + ood_names

    if not id_names or not ood_names:
        raise ValueError("Auto partition produced empty ID or OOD slice.")

    return id_names, ood_names, ood_domains


def run_eval(
    *,
    label: str,
    args: argparse.Namespace,
    run_dir: Path,
    datasets: list[str],
) -> dict[str, Any]:
    output_dir = run_dir / label
    log_path = run_dir / "logs" / f"{label}.log"

    cmd = [
        sys.executable,
        "scripts/experiments/run_husai_saebench_custom_eval.py",
        "--checkpoint",
        str(args.checkpoint),
        "--sae-release",
        f"{args.sae_release}_{label}",
        "--model-name",
        args.model_name,
        "--hook-layer",
        str(args.hook_layer),
        "--hook-name",
        args.hook_name,
        "--reg-type",
        args.reg_type,
        "--setting",
        args.setting,
        "--ks",
        args.ks,
        "--dataset-names",
        ",".join(datasets),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--results-path",
        str(args.results_path),
        "--model-cache-path",
        str(args.model_cache_path),
        "--output-dir",
        str(output_dir),
    ]
    if args.architecture:
        cmd.extend(["--architecture", args.architecture])
    if args.binarize:
        cmd.append("--binarize")
    if args.force_rerun:
        cmd.append("--force-rerun")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)

    summary_path = output_dir / "husai_custom_sae_summary.json"
    summary = None
    best_minus_llm_auc = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        best_minus_llm_auc = (summary.get("summary") or {}).get("best_minus_llm_auc")

    return {
        "label": label,
        "returncode": int(proc.returncode),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)) if summary_path.exists() else None,
        "best_minus_llm_auc": best_minus_llm_auc,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD stress evaluation using SAEBench sparse probing")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", type=str, default="")
    parser.add_argument("--sae-release", type=str, default="husai_ood_stress")
    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")
    parser.add_argument("--reg-type", type=str, default="l1")
    parser.add_argument("--setting", type=str, default="normal")
    parser.add_argument("--ks", type=str, default="1,2,5")
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "husai_saebench_probe_results_ood",
    )
    parser.add_argument(
        "--model-cache-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "sae_bench_model_cache",
    )

    parser.add_argument("--id-datasets", type=str, default="")
    parser.add_argument("--ood-datasets", type=str, default="")
    parser.add_argument("--ood-fraction", type=float, default=0.25)
    parser.add_argument("--min-ood-datasets", type=int, default=8)

    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_ood_stress",
    )
    args = parser.parse_args()

    args.checkpoint = resolve_path(args.checkpoint).resolve()
    args.results_path = resolve_path(args.results_path).resolve()
    args.model_cache_path = resolve_path(args.model_cache_path).resolve()
    args.output_dir = resolve_path(args.output_dir).resolve()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    explicit_id = parse_csv_strings(args.id_datasets)
    explicit_ood = parse_csv_strings(args.ood_datasets)
    inferred_all = infer_dataset_names_from_cache(args.model_cache_path, args.model_name, args.hook_name)

    if explicit_id and explicit_ood:
        id_names = explicit_id
        ood_names = explicit_ood
        ood_domains = ["__explicit__"]
    elif explicit_id and not explicit_ood:
        inferred_set = set(inferred_all)
        id_names = explicit_id
        ood_names = [name for name in inferred_all if name in inferred_set and name not in set(id_names)]
        ood_domains = ["__explicit_id__"]
    elif explicit_ood and not explicit_id:
        inferred_set = set(inferred_all)
        ood_names = explicit_ood
        id_names = [name for name in inferred_all if name in inferred_set and name not in set(ood_names)]
        ood_domains = ["__explicit_ood__"]
    else:
        if not inferred_all:
            raise ValueError(
                "No datasets inferred from model cache. Pass --id-datasets/--ood-datasets or provide a valid --model-cache-path."
            )
        id_names, ood_names, ood_domains = auto_partition_datasets(
            inferred_all,
            ood_fraction=args.ood_fraction,
            min_ood_datasets=args.min_ood_datasets,
        )

    if not id_names or not ood_names:
        raise ValueError("ID/OOD dataset split is empty.")

    overlap = sorted(set(id_names).intersection(ood_names))
    if overlap:
        raise ValueError(f"ID and OOD dataset sets overlap: {overlap[:10]}")

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    id_result = run_eval(label="id", args=args, run_dir=run_dir, datasets=id_names)
    ood_result = run_eval(label="ood", args=args, run_dir=run_dir, datasets=ood_names)

    id_delta = id_result.get("best_minus_llm_auc")
    ood_delta = ood_result.get("best_minus_llm_auc")
    ood_drop = None
    relative_drop = None
    if isinstance(id_delta, (int, float)) and isinstance(ood_delta, (int, float)):
        ood_drop = float(id_delta) - float(ood_delta)
        if abs(float(id_delta)) > 1e-12:
            relative_drop = ood_drop / abs(float(id_delta))

    config_payload = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.architecture,
        "sae_release": args.sae_release,
        "model_name": args.model_name,
        "hook_layer": args.hook_layer,
        "hook_name": args.hook_name,
        "reg_type": args.reg_type,
        "setting": args.setting,
        "ks": args.ks,
        "device": args.device,
        "dtype": args.dtype,
        "results_path": str(args.results_path),
        "model_cache_path": str(args.model_cache_path),
        "output_dir": str(args.output_dir),
        "id_dataset_count": len(id_names),
        "ood_dataset_count": len(ood_names),
        "ood_domains": ood_domains,
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
        "datasets": {
            "id": id_names,
            "ood": ood_names,
            "ood_domains": ood_domains,
            "inferred_total_count": len(inferred_all),
        },
        "id_result": id_result,
        "ood_result": ood_result,
        # Gate-compatible fields:
        "ood_drop": ood_drop,
        "relative_drop": relative_drop,
        "drop": ood_drop,
        "metrics": {
            "ood_drop": ood_drop,
            "relative_drop": relative_drop,
            "id_best_minus_llm_auc": id_delta,
            "ood_best_minus_llm_auc": ood_delta,
        },
    }

    out_json = run_dir / "ood_stress_summary.json"
    out_md = run_dir / "ood_stress_summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# OOD Stress Evaluation",
        "",
        f"- Run ID: `{run_id}`",
        f"- ID dataset count: `{len(id_names)}`",
        f"- OOD dataset count: `{len(ood_names)}`",
        f"- OOD domains: `{ood_domains}`",
        "",
        "## Metric Summary",
        "",
        f"- ID best_minus_llm_auc: `{id_delta}`",
        f"- OOD best_minus_llm_auc: `{ood_delta}`",
        f"- ood_drop (ID - OOD): `{ood_drop}`",
        f"- relative_drop: `{relative_drop}`",
        "",
        "## Command Return Codes",
        "",
        f"- ID eval return code: `{id_result['returncode']}`",
        f"- OOD eval return code: `{ood_result['returncode']}`",
        "",
        "## Artifacts",
        "",
        f"- JSON summary: `{out_json.relative_to(PROJECT_ROOT)}`",
        f"- Markdown summary: `{out_md.relative_to(PROJECT_ROOT)}`",
        f"- ID log: `{id_result['log_path']}`",
        f"- OOD log: `{ood_result['log_path']}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("OOD stress evaluation complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
