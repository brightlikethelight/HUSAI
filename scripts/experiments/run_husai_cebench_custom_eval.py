#!/usr/bin/env python3
"""Run CE-Bench directly on a HUSAI checkpoint via custom SAE object.

This adapter:
1) Loads a HUSAI checkpoint and maps it to a SAEBench custom SAE class.
2) Applies CE-Bench compatibility shims required by modern dependencies.
3) Executes CE-Bench evaluation for the custom SAE.
4) Emits a reproducibility summary and optional matched-baseline deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "results" / "cache" / "external_benchmarks"
DEFAULT_CEBENCH_ARTIFACTS = CACHE_ROOT / "ce_bench_artifacts"

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_cebench_compat import (  # noqa: E402
    clean_run_local_outputs,
    install_sae_lens_toolkit_shim,
    install_stw_stopwatch_shim,
    summarize_cebench_outputs,
)
from scripts.experiments.husai_custom_sae_adapter import (  # noqa: E402
    build_custom_sae_from_checkpoint,
    dtype_from_name,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_summary_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "contrastive_score_mean_max": summary.get("contrastive_score_mean_max"),
        "independent_score_mean_max": summary.get("independent_score_mean_max"),
        "interpretability_score_mean_max": summary.get("interpretability_score_mean_max"),
    }


def load_baseline_summary(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, float | None] | None]:
    if path is None:
        return None, None
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text())
    return payload, extract_summary_metrics(payload)


def diff_metrics(
    custom: dict[str, float | None], baseline: dict[str, float | None] | None
) -> dict[str, float | None] | None:
    if baseline is None:
        return None
    out: dict[str, float | None] = {}
    for key, value in custom.items():
        base = baseline.get(key)
        if value is None or base is None:
            out[key] = None
        else:
            out[key] = float(value) - float(base)
    return out


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path) -> str:
    abs_path = path.resolve() if not path.is_absolute() else path
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CE-Bench for a custom HUSAI checkpoint")
    parser.add_argument("--cebench-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", type=str, default="", help="Optional architecture override (topk/relu/batchtopk/jumprelu)")
    parser.add_argument("--sae-release", type=str, default="husai_topk_custom")
    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sae-dtype", type=str, default="float32")
    parser.add_argument(
        "--llm-dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_benchmark_official" / "husai_custom_cebench",
    )
    parser.add_argument("--artifacts-path", type=Path, default=DEFAULT_CEBENCH_ARTIFACTS)
    parser.add_argument("--matched-baseline-summary", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.cebench_repo = resolve_repo_path(args.cebench_repo).resolve()
    args.checkpoint = resolve_repo_path(args.checkpoint).resolve()
    args.output_folder = resolve_repo_path(args.output_folder).resolve()
    args.artifacts_path = resolve_repo_path(args.artifacts_path).resolve()
    if args.matched_baseline_summary is not None:
        args.matched_baseline_summary = resolve_repo_path(args.matched_baseline_summary).resolve()

    if not args.cebench_repo.exists():
        raise FileNotFoundError(f"CE-Bench repo not found: {args.cebench_repo}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if not install_sae_lens_toolkit_shim():
        raise RuntimeError(
            "Could not install sae_lens compatibility shim. "
            "Expected `sae_lens.loading.pretrained_saes_directory` in current env."
        )
    if not install_stw_stopwatch_shim():
        raise RuntimeError("Could not install stw.Stopwatch compatibility shim.")

    repo_str = str(args.cebench_repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.append(repo_str)

    try:
        import ce_bench.CE_Bench as ce_bench_mod
    except Exception as exc:
        raise RuntimeError(f"Failed to import CE-Bench from {args.cebench_repo}") from exc

    sae_dtype = dtype_from_name(args.sae_dtype)
    architecture_override = args.architecture or None
    sae, sae_meta = build_custom_sae_from_checkpoint(
        checkpoint_path=args.checkpoint,
        architecture_override=architecture_override,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        hook_name=args.hook_name,
        device=args.device,
        dtype=sae_dtype,
    )

    # CE-Bench enforces isinstance(sae, SAE); allow SAEBench custom classes too.
    base_sae_type = getattr(ce_bench_mod, "SAE", None)
    if base_sae_type is not None and not isinstance(base_sae_type, tuple):
        ce_bench_mod.SAE = (base_sae_type, type(sae))
    elif isinstance(base_sae_type, tuple) and type(sae) not in base_sae_type:
        ce_bench_mod.SAE = tuple([*base_sae_type, type(sae)])

    config = ce_bench_mod.AutoInterpEvalConfig(model_name=args.model_name)
    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = ce_bench_mod.activation_collection.LLM_NAME_TO_BATCH_SIZE[args.model_name]
    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = ce_bench_mod.activation_collection.LLM_NAME_TO_DTYPE[args.model_name]
    if args.random_seed is not None:
        config.random_seed = args.random_seed

    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.artifacts_path.mkdir(parents=True, exist_ok=True)

    dataset = ce_bench_mod.load_dataset("GulkoA/contrastive-stories-v4", split="train")
    dataset_rows_total = len(dataset)
    if args.max_rows is not None:
        n = min(max(0, int(args.max_rows)), dataset_rows_total)
        dataset = dataset.select(range(n))

    old_cwd = Path.cwd()
    run_error: Exception | None = None
    try:
        # CE-Bench writes relative outputs. Keep them isolated under this run folder.
        clean_run_local_outputs(args.output_folder)
        old_artifacts_path = getattr(ce_bench_mod.general_utils, "ARTIFACTS_PATH", None)
        ce_bench_mod.general_utils.ARTIFACTS_PATH = str(args.artifacts_path)
        os.chdir(args.output_folder)
        ce_bench_mod.run_eval_once(
            dataset=dataset,
            device=args.device,
            sae_release=args.sae_release,
            sae_id=sae,
            config=config,
        )
        if old_artifacts_path is not None:
            ce_bench_mod.general_utils.ARTIFACTS_PATH = old_artifacts_path
    except Exception as exc:
        run_error = exc
    finally:
        os.chdir(old_cwd)

    summary = summarize_cebench_outputs(args.output_folder)
    custom_metrics = extract_summary_metrics(summary)
    baseline_payload, baseline_metrics = load_baseline_summary(args.matched_baseline_summary)
    delta_vs_baseline = diff_metrics(custom_metrics, baseline_metrics)

    run_config_payload = {
        "cebench_repo": str(args.cebench_repo),
        "checkpoint": str(args.checkpoint),
        "architecture_override": architecture_override,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "sae_release": args.sae_release,
        "model_name": args.model_name,
        "hook_layer": args.hook_layer,
        "hook_name": args.hook_name,
        "device": args.device,
        "sae_dtype": args.sae_dtype,
        "llm_batch_size": config.llm_batch_size,
        "llm_dtype": config.llm_dtype,
        "random_seed": getattr(config, "random_seed", None),
        "dataset_name": "GulkoA/contrastive-stories-v4",
        "dataset_rows_total": dataset_rows_total,
        "dataset_rows_used": len(dataset),
        "max_rows": args.max_rows,
        "output_folder": str(args.output_folder),
        "artifacts_path": str(args.artifacts_path),
        "matched_baseline_summary": (
            str(args.matched_baseline_summary) if args.matched_baseline_summary is not None else None
        ),
    }

    run_payload = {
        "timestamp_utc": utc_now(),
        "command": " ".join(["python", *map(shlex.quote, sys.argv)]),
        "config_hash": stable_hash(run_config_payload),
        "config": run_config_payload,
        "sae_meta": sae_meta,
        "cebench_summary": summary,
        "custom_metrics": custom_metrics,
        "matched_baseline_metrics": baseline_metrics,
        "delta_vs_matched_baseline": delta_vs_baseline,
        "matched_baseline_payload": baseline_payload,
        "artifacts": {
            "cebench_metrics_summary_json": repo_rel(args.output_folder / "cebench_metrics_summary.json"),
            "cebench_metrics_summary_md": repo_rel(args.output_folder / "cebench_metrics_summary.md"),
        },
    }

    summary_json = args.output_folder / "husai_custom_cebench_summary.json"
    summary_md = args.output_folder / "husai_custom_cebench_summary.md"
    summary_json.write_text(json.dumps(run_payload, indent=2) + "\n")

    lines = [
        "# HUSAI Custom CE-Bench Summary",
        "",
        f"- Checkpoint: `{repo_rel(args.checkpoint)}`",
        f"- Checkpoint SHA256: `{run_config_payload['checkpoint_sha256']}`",
        f"- SAE release id: `{args.sae_release}`",
        f"- Model: `{args.model_name}`",
        f"- Hook: `{args.hook_name}` (layer `{args.hook_layer}`)",
        f"- Device: `{args.device}`",
        f"- Dataset rows used: `{run_config_payload['dataset_rows_used']}` / `{dataset_rows_total}`",
        f"- LLM dtype / batch: `{config.llm_dtype}` / `{config.llm_batch_size}`",
        "",
        "## Custom Metrics",
        "",
        f"- contrastive_score_mean.max: `{custom_metrics['contrastive_score_mean_max']}`",
        f"- independent_score_mean.max: `{custom_metrics['independent_score_mean_max']}`",
        f"- interpretability_score_mean.max: `{custom_metrics['interpretability_score_mean_max']}`",
    ]

    if baseline_metrics is not None:
        lines.extend(
            [
                "",
                "## Delta vs Matched Baseline",
                "",
                f"- baseline summary: `{repo_rel(args.matched_baseline_summary)}`",
                (
                    "- delta contrastive_score_mean.max "
                    f"(custom - baseline): `{delta_vs_baseline['contrastive_score_mean_max']}`"
                ),
                (
                    "- delta independent_score_mean.max "
                    f"(custom - baseline): `{delta_vs_baseline['independent_score_mean_max']}`"
                ),
                (
                    "- delta interpretability_score_mean.max "
                    f"(custom - baseline): `{delta_vs_baseline['interpretability_score_mean_max']}`"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CE-Bench metrics JSON: `{repo_rel(args.output_folder / 'cebench_metrics_summary.json')}`",
            f"- CE-Bench metrics markdown: `{repo_rel(args.output_folder / 'cebench_metrics_summary.md')}`",
            f"- Adapter summary JSON: `{repo_rel(summary_json)}`",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n")

    print("HUSAI custom CE-Bench eval complete")
    print(f"Output folder: {args.output_folder}")
    print(f"Summary JSON: {summary_json}")

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
