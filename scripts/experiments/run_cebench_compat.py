#!/usr/bin/env python3
"""Compatibility runner for CE-Bench in modern dependency environments.

This wrapper addresses upstream API drift encountered when running CE-Bench:
1) `sae_lens.toolkit.pretrained_saes_directory` -> moved in modern sae_lens.
2) `stw.Stopwatch(start=...)` -> newer stw versions removed the `start` kwarg.

It installs parent-process shims, writes a `sitecustomize` shim for spawned
workers, executes CE-Bench, and exports a compact metrics summary artifact.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import runpy
import shutil
import sys
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "results" / "cache" / "external_benchmarks"
DEFAULT_CEBENCH_ARTIFACTS = CACHE_ROOT / "ce_bench_artifacts"


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
    """Patch stw.Stopwatch to accept `start=` and provide `stop()` compatibility."""
    try:
        import stw

        base = stw.Stopwatch
    except Exception:
        return False

    try:
        signature = inspect.signature(base.__init__)
    except Exception:
        signature = None

    if signature and "start" in signature.parameters and hasattr(base, "stop"):
        return True

    class CompatStopwatch(base):
        def __init__(self, *args, start=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._compat_stopped = False
            self._compat_stop_result = None
            if start:
                self.start()

        def start(self):
            now_fn = getattr(self, "now", None)
            now_value = now_fn() if callable(now_fn) else None
            if now_value is not None:
                self._start_time = now_value
                self._last_lap_time = now_value
            laps = getattr(self, "_laps", None)
            if isinstance(laps, list):
                laps.clear()
            self._compat_stopped = False
            self._compat_stop_result = None
            return self

        def stop(self):
            if self._compat_stopped:
                return self._compat_stop_result
            self._compat_stop_result = super().lap("stop")
            self._compat_stopped = True
            return self._compat_stop_result

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
        "# stw.Stopwatch(start=...) and stop() shim\n"
        "try:\n"
        "    import stw\n"
        "    _base = stw.Stopwatch\n"
        "    _sig = inspect.signature(_base.__init__)\n"
        "    if 'start' not in _sig.parameters or not hasattr(_base, 'stop'):\n"
        "        class CompatStopwatch(_base):\n"
        "            def __init__(self, *args, start=None, **kwargs):\n"
        "                super().__init__(*args, **kwargs)\n"
        "                self._compat_stopped = False\n"
        "                self._compat_stop_result = None\n"
        "                if start:\n"
        "                    self.start()\n"
        "            def start(self):\n"
        "                now_fn = getattr(self, 'now', None)\n"
        "                now_value = now_fn() if callable(now_fn) else None\n"
        "                if now_value is not None:\n"
        "                    self._start_time = now_value\n"
        "                    self._last_lap_time = now_value\n"
        "                laps = getattr(self, '_laps', None)\n"
        "                if isinstance(laps, list):\n"
        "                    laps.clear()\n"
        "                self._compat_stopped = False\n"
        "                self._compat_stop_result = None\n"
        "                return self\n"
        "            def stop(self):\n"
        "                if self._compat_stopped:\n"
        "                    return self._compat_stop_result\n"
        "                self._compat_stop_result = super().lap('stop')\n"
        "                self._compat_stopped = True\n"
        "                return self._compat_stop_result\n"
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


def clean_run_local_outputs(output_folder: Path) -> None:
    """Remove CE-Bench relative outputs so reruns stay deterministic."""
    scores_path = output_folder / "scores_dump.txt"
    interp_dir = output_folder / "interpretability_eval"

    if scores_path.exists():
        scores_path.unlink()
    if interp_dir.exists():
        shutil.rmtree(interp_dir)



def _copy_if_present(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def summarize_cebench_outputs(output_folder: Path) -> dict[str, Any]:
    """Collect CE-Bench result artifacts and emit a compact summary JSON/MD."""
    summary: dict[str, Any] = {
        "output_folder": str(output_folder),
        "results_json": None,
        "scores_dump": None,
        "total_rows": None,
        "contrastive_score_mean_max": None,
        "independent_score_mean_max": None,
        "interpretability_score_mean_max": None,
        "sae_release": None,
        "sae_id": None,
        "date": None,
        "scores_dump_line_count": None,
    }

    output_results = sorted((output_folder / "interpretability_eval").glob("**/results.json"))
    project_results = sorted((PROJECT_ROOT / "interpretability_eval").glob("**/results.json"))

    result_path: Path | None = output_results[0] if output_results else None
    if result_path is None and project_results:
        src = project_results[0]
        dst = output_folder / "captured_from_project_root" / src.relative_to(PROJECT_ROOT)
        if _copy_if_present(src, dst):
            result_path = dst

    output_scores = output_folder / "scores_dump.txt"
    project_scores = PROJECT_ROOT / "scores_dump.txt"
    scores_path: Path | None = output_scores if output_scores.exists() else None
    if scores_path is None and project_scores.exists():
        dst = output_folder / "captured_from_project_root" / "scores_dump.txt"
        if _copy_if_present(project_scores, dst):
            scores_path = dst

    if result_path is not None:
        summary["results_json"] = str(result_path)
        try:
            data = json.loads(result_path.read_text())
            summary["total_rows"] = data.get("total_rows")
            summary["sae_release"] = data.get("sae_release")
            summary["sae_id"] = data.get("sae_id")
            summary["date"] = data.get("date")
            summary["contrastive_score_mean_max"] = (
                data.get("contrastive_score_mean", {}) or {}
            ).get("max")
            summary["independent_score_mean_max"] = (
                data.get("independent_score_mean", {}) or {}
            ).get("max")
            summary["interpretability_score_mean_max"] = (
                data.get("interpretability_score_mean", {}) or {}
            ).get("max")
        except Exception as exc:
            summary["parse_error"] = str(exc)

    if scores_path is not None:
        summary["scores_dump"] = str(scores_path)
        try:
            with scores_path.open("r", encoding="utf-8") as f:
                summary["scores_dump_line_count"] = sum(1 for _ in f)
        except Exception as exc:
            summary["scores_dump_count_error"] = str(exc)

    summary_path = output_folder / "cebench_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    md_lines = [
        "# CE-Bench Metrics Summary",
        "",
        f"- output_folder: `{output_folder}`",
        f"- results_json: `{summary['results_json']}`",
        f"- scores_dump: `{summary['scores_dump']}`",
        f"- total_rows: `{summary['total_rows']}`",
        f"- contrastive_score_mean.max: `{summary['contrastive_score_mean_max']}`",
        f"- independent_score_mean.max: `{summary['independent_score_mean_max']}`",
        f"- interpretability_score_mean.max: `{summary['interpretability_score_mean_max']}`",
        f"- scores_dump_line_count: `{summary['scores_dump_line_count']}`",
        f"- sae_release: `{summary['sae_release']}`",
        f"- sae_id: `{summary['sae_id']}`",
        f"- date: `{summary['date']}`",
    ]
    if "parse_error" in summary:
        md_lines.append(f"- parse_error: `{summary['parse_error']}`")
    if "scores_dump_count_error" in summary:
        md_lines.append(f"- scores_dump_count_error: `{summary['scores_dump_count_error']}`")

    (output_folder / "cebench_metrics_summary.md").write_text("\n".join(md_lines) + "\n")
    return summary


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
    parser.add_argument("--artifacts-path", type=Path, default=DEFAULT_CEBENCH_ARTIFACTS)
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
    original_cwd = Path.cwd()
    run_error: Exception | None = None
    try:
        # CE-Bench writes relative outputs (interpretability_eval/scores_dump); keep them
        # scoped under this run's output folder instead of polluting repository root.
        os.chdir(args.output_folder)
        clean_run_local_outputs(args.output_folder)
        sys.argv = argv
        runpy.run_path(str(ce_bench_script), run_name="__main__")
    except Exception as exc:
        run_error = exc
    finally:
        os.chdir(original_cwd)
        sys.argv = old_argv

    summary = summarize_cebench_outputs(args.output_folder)
    print("CE-Bench summary written:", args.output_folder / "cebench_metrics_summary.json")
    print("CE-Bench key metrics:", json.dumps(summary, indent=2))

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
