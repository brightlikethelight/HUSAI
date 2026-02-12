#!/usr/bin/env python3
"""Build a SAEBench/CE-Bench-aligned evaluation slice from local artifacts.

This does not claim to run official SAEBench or CE-Bench suites. Instead it
maps this repo's measured signals to benchmark-style categories:
- consistency/control (trained vs random PWMCC)
- robustness (k and d_sae sweep trends)
- causal-faithfulness proxy (stable vs unstable intervention deltas)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def find_latest_json(root: Path, relative: str = "results.json") -> Path:
    candidates = sorted(root.glob(f"run_*/{relative}"))
    if not candidates:
        raise FileNotFoundError(f"No {relative} found under {root}")
    return candidates[-1]


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external benchmark-aligned slice")
    parser.add_argument(
        "--phase4a-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4a_trained_vs_random" / "results.json",
    )
    parser.add_argument(
        "--core-ablations-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4c_core_ablations",
    )
    parser.add_argument(
        "--causal-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "proper_causal_ablation_results.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_benchmark_slice",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load phase4a reproduction.
    if not args.phase4a_results.exists():
        raise FileNotFoundError(f"Phase4a result missing: {args.phase4a_results}")
    phase4a = json.loads(args.phase4a_results.read_text())

    # Load latest core ablations run.
    core_path = find_latest_json(args.core_ablations_root)
    core = json.loads(core_path.read_text())

    k_sweep = core.get("k_sweep", [])
    d_sae_sweep = core.get("d_sae_sweep", [])

    # Best configs by PWMCC delta and EV guardrail.
    def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = [r for r in records if r.get("explained_variance", {}).get("mean", -1.0) > 0.0]
        if not eligible:
            eligible = records
        if not eligible:
            return None
        return max(eligible, key=lambda r: r.get("delta_pwmcc", float("-inf")))

    best_k = best_record(k_sweep)
    best_d = best_record(d_sae_sweep)

    # Optional causal-faithfulness proxy.
    causal_proxy: dict[str, Any] = {
        "available": False,
        "method": "proper_causal_ablation_with_hooks",
        "stable_effect_mean": float("nan"),
        "unstable_effect_mean": float("nan"),
        "delta": float("nan"),
    }
    if args.causal_results.exists():
        obj = torch.load(args.causal_results, map_location="cpu")
        stable_effects = list(map(float, obj.get("stable_effects", [])))
        unstable_effects = list(map(float, obj.get("unstable_effects", [])))
        stable_mean = safe_mean(stable_effects)
        unstable_mean = safe_mean(unstable_effects)
        causal_proxy = {
            "available": True,
            "method": obj.get("method", "unknown"),
            "stable_effect_mean": stable_mean,
            "unstable_effect_mean": unstable_mean,
            "delta": stable_mean - unstable_mean,
            "n_stable": len(stable_effects),
            "n_unstable": len(unstable_effects),
        }

    trained_mean = float(phase4a["trained"]["mean"])
    random_mean = float(phase4a["random"]["mean"])
    p_value = float(phase4a["statistical_test"]["p_value"])

    consistency_pass = (trained_mean > random_mean) and (p_value < 0.05)
    robustness_pass = bool(best_k and best_k.get("delta_pwmcc", 0.0) > 0 and best_d and best_d.get("delta_pwmcc", 0.0) > 0)

    benchmark_slice = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "core_ablations_source": str(core_path.relative_to(PROJECT_ROOT)),
            "phase4a_source": str(args.phase4a_results.relative_to(PROJECT_ROOT)),
        },
        "protocol": {
            "name": "SAEBench/CE-Bench-aligned slice",
            "is_official_suite": False,
            "alignment_categories": [
                "consistency_vs_random_control",
                "hyperparameter_robustness",
                "causal_faithfulness_proxy",
            ],
            "notes": (
                "This is a protocol-aligned slice for this small-model repo. "
                "Use official SAEBench/CE-Bench tooling for leaderboard or SOTA claims."
            ),
        },
        "consistency_control": {
            "trained_pwmcc_mean": trained_mean,
            "trained_ci95": [
                float(phase4a["trained"]["ci95_low"]),
                float(phase4a["trained"]["ci95_high"]),
            ],
            "random_pwmcc_mean": random_mean,
            "random_ci95": [
                float(phase4a["random"]["ci95_low"]),
                float(phase4a["random"]["ci95_high"]),
            ],
            "p_value_one_sided": p_value,
            "delta": trained_mean - random_mean,
            "pass": consistency_pass,
        },
        "robustness": {
            "best_k_condition": best_k,
            "best_d_sae_condition": best_d,
            "pass": robustness_pass,
        },
        "causal_faithfulness_proxy": causal_proxy,
        "overall_readout": {
            "internal_gating_pass": bool(consistency_pass and robustness_pass),
            "ready_for_external_benchmark_claim": False,
            "blocking_note": (
                "Need official SAEBench/CE-Bench execution for external benchmark claims."
            ),
        },
    }

    output_json = args.output_dir / "benchmark_slice.json"
    output_md = args.output_dir / "benchmark_slice.md"
    manifest = args.output_dir / "manifest.json"

    output_json.write_text(json.dumps(benchmark_slice, indent=2) + "\n")

    best_k_str = "n/a"
    if best_k:
        best_k_str = (
            f"k={best_k['k']}, d_sae={best_k['d_sae']}, "
            f"delta={best_k['delta_pwmcc']:.6f}, ratio={best_k['ratio_pwmcc']:.4f}"
        )
    best_d_str = "n/a"
    if best_d:
        best_d_str = (
            f"d_sae={best_d['d_sae']}, k={best_d['k']}, "
            f"delta={best_d['delta_pwmcc']:.6f}, ratio={best_d['ratio_pwmcc']:.4f}"
        )

    output_md.write_text(
        "\n".join(
            [
                "# External Benchmark-Aligned Slice",
                "",
                "## Scope",
                "- Protocol: SAEBench/CE-Bench-aligned slice (not official suite run)",
                f"- Phase4a source: `{args.phase4a_results.relative_to(PROJECT_ROOT)}`",
                f"- Core ablations source: `{core_path.relative_to(PROJECT_ROOT)}`",
                "",
                "## Consistency Control",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Trained PWMCC mean | {trained_mean:.6f} |",
                f"| Random PWMCC mean | {random_mean:.6f} |",
                f"| Delta | {trained_mean - random_mean:.6f} |",
                f"| One-sided p-value | {p_value:.3e} |",
                f"| Pass | {consistency_pass} |",
                "",
                "## Robustness",
                f"- Best k-sweep condition: {best_k_str}",
                f"- Best d_sae-sweep condition: {best_d_str}",
                f"- Pass: {robustness_pass}",
                "",
                "## Causal Proxy",
                f"- Available: {causal_proxy['available']}",
                f"- Stable mean effect: {causal_proxy['stable_effect_mean']}",
                f"- Unstable mean effect: {causal_proxy['unstable_effect_mean']}",
                f"- Delta (stable - unstable): {causal_proxy['delta']}",
                "",
                "## Overall",
                (
                    f"- Ready for external benchmark claim: "
                    f"{benchmark_slice['overall_readout']['ready_for_external_benchmark_claim']}"
                ),
                "- Blocking note: Need official SAEBench/CE-Bench run before SOTA claims.",
            ]
        )
        + "\n"
    )

    manifest.write_text(
        json.dumps(
            {
                "run_metadata": benchmark_slice["run_metadata"],
                "artifacts": [
                    str(output_json.relative_to(PROJECT_ROOT)),
                    str(output_md.relative_to(PROJECT_ROOT)),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print("External benchmark-aligned slice complete")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
