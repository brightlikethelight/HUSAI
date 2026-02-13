#!/usr/bin/env python3
"""Compatibility runner for CE-Bench in modern sae_lens environments.

Why this exists:
- CE-Bench currently imports `sae_lens.toolkit.pretrained_saes_directory`.
- In newer `sae_lens` releases (e.g. 6.x), this moved to
  `sae_lens.loading.pretrained_saes_directory`.

This script installs a runtime import shim, then executes the official
`ce_bench/CE_Bench.py` entrypoint with explicit args so runs are still
artifacted via our benchmark harness.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def install_sae_lens_toolkit_shim() -> bool:
    """Inject compatibility modules expected by CE-Bench into sys.modules."""
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except Exception:
        return False

    if "sae_lens.toolkit" not in sys.modules:
        sys.modules["sae_lens.toolkit"] = types.ModuleType("sae_lens.toolkit")

    shim_name = "sae_lens.toolkit.pretrained_saes_directory"
    shim_module = types.ModuleType(shim_name)
    shim_module.get_pretrained_saes_directory = get_pretrained_saes_directory
    sys.modules[shim_name] = shim_module
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CE-Bench with sae_lens compatibility shim")
    parser.add_argument("--cebench-repo", type=Path, required=True)
    parser.add_argument("--sae-regex-pattern", type=str, required=True)
    parser.add_argument("--sae-block-pattern", type=str, required=True)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_benchmark_official" / "cebench",
    )
    parser.add_argument("--artifacts-path", type=Path, default=Path("/tmp/ce_bench_artifacts"))
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument(
        "--llm-dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ce_bench_script = args.cebench_repo / "ce_bench" / "CE_Bench.py"
    if not ce_bench_script.exists():
        raise FileNotFoundError(f"CE-Bench entrypoint not found: {ce_bench_script}")

    shim_ok = install_sae_lens_toolkit_shim()
    if not shim_ok:
        raise RuntimeError(
            "Could not install sae_lens compatibility shim. "
            "Expected `sae_lens.loading.pretrained_saes_directory` in current env."
        )

    sys.path.insert(0, str(args.cebench_repo))

    argv = [
        str(ce_bench_script),
        "--sae_regex_pattern",
        args.sae_regex_pattern,
        "--sae_block_pattern",
        args.sae_block_pattern,
        "--output_folder",
        str(args.output_folder),
        "--artifacts_path",
        str(args.artifacts_path),
    ]
    if args.random_seed is not None:
        argv.extend(["--random_seed", str(args.random_seed)])
    if args.force_rerun:
        argv.append("--force_rerun")
    if args.llm_batch_size is not None:
        argv.extend(["--llm_batch_size", str(args.llm_batch_size)])
    if args.llm_dtype is not None:
        argv.extend(["--llm_dtype", args.llm_dtype])

    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.artifacts_path.mkdir(parents=True, exist_ok=True)

    print("Running CE-Bench with compatibility shim")
    print(f"CE-Bench repo: {args.cebench_repo}")
    print(f"Output folder: {args.output_folder}")
    print(f"Artifacts path: {args.artifacts_path}")

    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        runpy.run_path(str(ce_bench_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
