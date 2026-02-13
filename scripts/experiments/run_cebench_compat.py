#!/usr/bin/env python3
"""Compatibility runner for CE-Bench in modern dependency environments.

This wrapper addresses upstream API drift encountered when running CE-Bench:
1) `sae_lens.toolkit.pretrained_saes_directory` -> moved in modern sae_lens.
2) `stw.Stopwatch(start=...)` -> newer stw versions removed the `start` kwarg.

It installs parent-process shims, writes a `sitecustomize` shim for spawned
workers, and then executes CE-Bench while preserving harness artifacts.
"""

from __future__ import annotations

import argparse
import inspect
import os
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


def install_stw_stopwatch_shim() -> bool:
    """Patch stw.Stopwatch to accept `start=` for CE-Bench compatibility."""
    try:
        import stw
        base = stw.Stopwatch
    except Exception:
        return False

    try:
        signature = inspect.signature(base.__init__)
    except Exception:
        signature = None

    if signature and "start" in signature.parameters:
        return True

    class CompatStopwatch(base):
        def __init__(self, *args, start=None, **kwargs):
            super().__init__(*args, **kwargs)
            if start:
                start_fn = getattr(self, "start", None)
                if callable(start_fn):
                    start_fn()

    stw.Stopwatch = CompatStopwatch
    sys.modules["stw"].Stopwatch = CompatStopwatch
    return True


def write_sitecustomize_shim(output_folder: Path) -> Path:
    """Write a sitecustomize shim so multiprocessing workers inherit patches."""
    compat_dir = output_folder / ".cebench_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = compat_dir / "sitecustomize.py"
    sitecustomize_code = (
        "import inspect\n"
        "import sys\n"
        "import types\n"
        "# sae_lens toolkit alias shim\n"
        "try:\n"
        "    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory\n"
        "except Exception:\n"
        "    get_pretrained_saes_directory = None\n"
        "if get_pretrained_saes_directory is not None:\n"
        "    if 'sae_lens.toolkit' not in sys.modules:\n"
        "        sys.modules['sae_lens.toolkit'] = types.ModuleType('sae_lens.toolkit')\n"
        "    shim = types.ModuleType('sae_lens.toolkit.pretrained_saes_directory')\n"
        "    shim.get_pretrained_saes_directory = get_pretrained_saes_directory\n"
        "    sys.modules['sae_lens.toolkit.pretrained_saes_directory'] = shim\n"
        "# stw.Stopwatch(start=...) shim\n"
        "try:\n"
        "    import stw\n"
        "    _base = stw.Stopwatch\n"
        "    _sig = inspect.signature(_base.__init__)\n"
        "    if 'start' not in _sig.parameters:\n"
        "        class CompatStopwatch(_base):\n"
        "            def __init__(self, *args, start=None, **kwargs):\n"
        "                super().__init__(*args, **kwargs)\n"
        "                if start:\n"
        "                    start_fn = getattr(self, 'start', None)\n"
        "                    if callable(start_fn):\n"
        "                        start_fn()\n"
        "        stw.Stopwatch = CompatStopwatch\n"
        "        sys.modules['stw'].Stopwatch = CompatStopwatch\n"
        "except Exception:\n"
        "    pass\n"
    )
    sitecustomize_path.write_text(sitecustomize_code)
    return compat_dir


def prepend_pythonpath(path: Path) -> None:
    path_str = str(path)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        os.environ["PYTHONPATH"] = f"{path_str}:{existing}"
    else:
        os.environ["PYTHONPATH"] = path_str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CE-Bench with dependency-compatibility shims")
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

    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.artifacts_path.mkdir(parents=True, exist_ok=True)

    if not install_sae_lens_toolkit_shim():
        raise RuntimeError(
            "Could not install sae_lens compatibility shim. "
            "Expected `sae_lens.loading.pretrained_saes_directory` in current env."
        )
    if not install_stw_stopwatch_shim():
        raise RuntimeError("Could not install stw.Stopwatch compatibility shim.")

    # Ensure spawned Python workers import our shims via sitecustomize.
    compat_dir = write_sitecustomize_shim(args.output_folder)
    prepend_pythonpath(compat_dir)

    # Keep CE-Bench importable while avoiding local sae_lens shadowing.
    repo_str = str(args.cebench_repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.append(repo_str)

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

    print("Running CE-Bench with compatibility shim")
    print(f"CE-Bench repo: {args.cebench_repo}")
    print(f"Output folder: {args.output_folder}")
    print(f"Artifacts path: {args.artifacts_path}")
    print(f"sitecustomize shim dir: {compat_dir}")

    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        runpy.run_path(str(ce_bench_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
