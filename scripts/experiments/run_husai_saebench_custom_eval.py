#!/usr/bin/env python3
"""Run SAEBench sparse-probing SAE-probes on a HUSAI checkpoint.

This script loads a HUSAI checkpoint, maps it into a SAEBench custom SAE
object, and executes the official sparse_probing_sae_probes eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "results" / "cache" / "external_benchmarks"
DEFAULT_HUSAI_SAEBENCH_RESULTS = CACHE_ROOT / "husai_saebench_probe_results"
DEFAULT_SAEBENCH_MODEL_CACHE = CACHE_ROOT / "sae_bench_model_cache"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.husai_custom_sae_adapter import (  # noqa: E402
    build_custom_sae_from_checkpoint,
    dtype_from_name,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_ks(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--ks must include at least one integer")
    return [int(v) for v in values]


def parse_dataset_names(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values


def infer_dataset_names_from_cache(model_cache_path: Path, model_name: str, hook_name: str) -> list[str]:
    cache_dir = model_cache_path / f"model_activations_{model_name}"
    if not cache_dir.exists():
        return []

    suffix = f"_{hook_name}.pt"
    names: list[str] = []
    for path in sorted(cache_dir.glob(f"*{suffix}")):
        stem = path.name
        if stem.endswith(suffix):
            names.append(stem[: -len(suffix)])
    return names


def summarize_results(results: dict[str, dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    if not results:
        return {"error": "No results returned"}

    key = next(iter(results))
    payload = results[key]
    metrics = payload.get("eval_result_metrics", {})
    sae_metrics = metrics.get("sae", {})
    llm_metrics = metrics.get("llm", {})

    by_k: list[dict[str, Any]] = []
    for k in ks:
        by_k.append(
            {
                "k": k,
                "test_accuracy": sae_metrics.get(f"sae_top_{k}_test_accuracy"),
                "test_auc": sae_metrics.get(f"sae_top_{k}_test_auc"),
                "test_f1": sae_metrics.get(f"sae_top_{k}_test_f1"),
            }
        )

    valid_auc = [item for item in by_k if item.get("test_auc") is not None]
    best_by_auc = max(valid_auc, key=lambda x: x["test_auc"]) if valid_auc else None
    llm_auc = llm_metrics.get("llm_test_auc")

    return {
        "result_key": key,
        "llm_metrics": llm_metrics,
        "sae_metrics_by_k": by_k,
        "best_by_auc": best_by_auc,
        "best_minus_llm_auc": (
            (best_by_auc["test_auc"] - llm_auc) if best_by_auc and llm_auc is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HUSAI checkpoint with SAEBench sparse-probing SAE-probes")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to HUSAI sae_final.pt checkpoint")
    parser.add_argument("--architecture", type=str, default="", help="Optional architecture override (topk/relu/batchtopk/jumprelu)")
    parser.add_argument("--sae-release", type=str, default="husai_topk_custom")
    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")
    parser.add_argument("--reg-type", type=str, default="l1")
    parser.add_argument("--setting", type=str, default="normal")
    parser.add_argument("--ks", type=str, default="1,2,5")
    parser.add_argument("--dataset-names", type=str, default="")
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_HUSAI_SAEBENCH_RESULTS,
        help="Directory for raw SAE-probes JSON outputs",
    )
    parser.add_argument(
        "--model-cache-path",
        type=Path,
        default=DEFAULT_SAEBENCH_MODEL_CACHE,
        help="Model activation cache directory for SAEBench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_benchmark_official" / "husai_custom_saebench",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    ks = parse_ks(args.ks)
    dataset_names = parse_dataset_names(args.dataset_names)
    inferred_from_cache = False
    if not dataset_names:
        dataset_names = infer_dataset_names_from_cache(args.model_cache_path, args.model_name, args.hook_name)
        inferred_from_cache = True
    if not dataset_names:
        raise ValueError(
            "No dataset names resolved for SAEBench sparse probing; pass --dataset-names or provide a valid --model-cache-path."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_path.mkdir(parents=True, exist_ok=True)
    args.model_cache_path.mkdir(parents=True, exist_ok=True)

    architecture_override = args.architecture or None
    sae, sae_meta = build_custom_sae_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        hook_name=args.hook_name,
        device=args.device,
        dtype=dtype,
        architecture_override=architecture_override,
    )

    try:
        from sae_bench.evals.sparse_probing_sae_probes.main import (
            SparseProbingSaeProbesEvalConfig,
            run_eval,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import SAEBench sparse_probing_sae_probes. Ensure `sae-bench` is installed."
        ) from exc

    eval_config = SparseProbingSaeProbesEvalConfig(
        model_name=args.model_name,
        dataset_names=dataset_names,
        reg_type=args.reg_type,
        setting=args.setting,
        ks=ks,
        binarize=args.binarize,
        results_path=str(args.results_path),
        model_cache_path=str(args.model_cache_path),
        include_llm_baseline=True,
        baseline_method="logreg",
    )

    selected_saes = [(args.sae_release, sae)]
    results = run_eval(
        config=eval_config,
        selected_saes=selected_saes,
        device=args.device,
        output_path=str(args.output_dir),
        force_rerun=args.force_rerun,
    )

    summary = summarize_results(results, ks)

    run_payload = {
        "timestamp_utc": utc_now(),
        "command": " ".join(["python", *map(shlex.quote, sys.argv)]),
        "config": {
            "checkpoint": str(args.checkpoint),
            "architecture_override": architecture_override,
            "sae_release": args.sae_release,
            "model_name": args.model_name,
            "hook_layer": args.hook_layer,
            "hook_name": args.hook_name,
            "reg_type": args.reg_type,
            "setting": args.setting,
            "ks": ks,
            "dataset_names": dataset_names,
            "dataset_names_inferred_from_cache": inferred_from_cache,
            "dataset_count": len(dataset_names),
            "binarize": args.binarize,
            "device": args.device,
            "dtype": args.dtype,
            "results_path": str(args.results_path),
            "model_cache_path": str(args.model_cache_path),
            "force_rerun": args.force_rerun,
        },
        "config_hash": stable_hash(
            {
                "checkpoint": str(args.checkpoint),
                "architecture_override": architecture_override,
                "sae_release": args.sae_release,
                "model_name": args.model_name,
                "hook_layer": args.hook_layer,
                "hook_name": args.hook_name,
                "reg_type": args.reg_type,
                "setting": args.setting,
                "ks": ks,
                "dataset_names": dataset_names,
            }
        ),
        "sae_meta": sae_meta,
        "summary": summary,
    }

    summary_json = args.output_dir / "husai_custom_sae_summary.json"
    summary_md = args.output_dir / "husai_custom_sae_summary.md"
    summary_json.write_text(json.dumps(run_payload, indent=2) + "\n")

    lines = [
        "# HUSAI Custom SAE SAEBench Summary",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- SAE release id: `{args.sae_release}`",
        f"- Architecture: `{sae_meta.get('architecture')}`",
        f"- Model: `{args.model_name}`",
        f"- Hook: `{args.hook_name}`",
        f"- ks: `{ks}`",
        f"- dataset count: `{len(dataset_names)}`",
        f"- dataset source: `{'cache_inferred' if inferred_from_cache else 'explicit'}`",
        "",
        "## Best by AUC",
        "",
        f"- Best k by AUC: `{summary.get('best_by_auc', {}).get('k') if summary.get('best_by_auc') else None}`",
        f"- Best SAE test_auc: `{summary.get('best_by_auc', {}).get('test_auc') if summary.get('best_by_auc') else None}`",
        f"- LLM baseline test_auc: `{summary.get('llm_metrics', {}).get('llm_test_auc')}`",
        f"- Best minus LLM AUC: `{summary.get('best_minus_llm_auc')}`",
        "",
        "## Artifacts",
        "",
        f"- JSON summary: `{summary_json}`",
        f"- SAEBench output dir: `{args.output_dir}`",
        f"- Raw probe results path: `{args.results_path}`",
    ]
    summary_md.write_text("\n".join(lines) + "\n")

    print("HUSAI custom SAEBench eval complete")
    print(f"Output dir: {args.output_dir}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
