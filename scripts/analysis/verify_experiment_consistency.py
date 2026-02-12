#!/usr/bin/env python3
"""Validate whether headline experiment conclusions match current artifacts.

This script creates a machine-readable and human-readable audit that is useful
for preventing drift between result JSONs and long-form writeups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def find_latest_result(root: Path) -> Path:
    candidates = sorted(root.glob("run_*/results.json"))
    if not candidates:
        raise FileNotFoundError(f"No run_*/results.json under {root}")
    return candidates[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def bootstrap_mean_diff_ci(
    x: list[float],
    y: list[float],
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    if arr_x.size == 0 or arr_y.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    xs = rng.choice(arr_x, size=(n_bootstrap, arr_x.size), replace=True).mean(axis=1)
    ys = rng.choice(arr_y, size=(n_bootstrap, arr_y.size), replace=True).mean(axis=1)
    diffs = xs - ys

    mean_diff = float(arr_x.mean() - arr_y.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def pick_adaptive_runs(root: Path) -> tuple[tuple[Path, dict[str, Any]], tuple[Path, dict[str, Any]]]:
    candidates = sorted(root.glob("run_*/results.json"))
    if not candidates:
        raise FileNotFoundError(f"No adaptive runs under {root}")

    search_runs: list[tuple[Path, dict[str, Any]]] = []
    control_runs: list[tuple[Path, dict[str, Any]]] = []

    for path in candidates:
        obj = load_json(path)
        k_candidates = obj.get("config", {}).get("k_candidates", [])
        if isinstance(k_candidates, list) and len(k_candidates) > 1:
            search_runs.append((path, obj))
        elif isinstance(k_candidates, list) and len(k_candidates) == 1:
            control_runs.append((path, obj))

    if not search_runs:
        raise RuntimeError("No adaptive search run found (k_candidates length > 1)")
    if not control_runs:
        raise RuntimeError("No adaptive control run found (single k_candidates value)")

    return search_runs[-1], control_runs[-1]


def find_consistency_baseline_and_selected(
    records: list[dict[str, Any]],
    selected_lambda: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = min(records, key=lambda r: abs(float(r["lambda_consistency"])))
    selected = min(records, key=lambda r: abs(float(r["lambda_consistency"]) - selected_lambda))
    return baseline, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify consistency of headline experiment claims")
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
        "--adaptive-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "adaptive_l0_calibration",
    )
    parser.add_argument(
        "--consistency-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "consistency_objective_sweep",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "experiment_consistency_report.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "experiment_consistency_report.md",
    )
    args = parser.parse_args()

    if not args.phase4a_results.exists():
        raise FileNotFoundError(f"Missing phase4a results: {args.phase4a_results}")

    phase4a = load_json(args.phase4a_results)
    core_path = find_latest_result(args.core_ablations_root)
    core = load_json(core_path)

    (adaptive_search_path, adaptive_search), (adaptive_control_path, adaptive_control) = pick_adaptive_runs(
        args.adaptive_root
    )

    consistency_path = find_latest_result(args.consistency_root)
    consistency = load_json(consistency_path)

    core_k_best = max(core.get("k_sweep", []), key=lambda r: float(r.get("delta_pwmcc", float("-inf"))))
    core_d_best = max(core.get("d_sae_sweep", []), key=lambda r: float(r.get("delta_pwmcc", float("-inf"))))

    phase4a_delta = float(phase4a["comparison"]["difference"])
    phase4a_p = float(phase4a["statistical_test"]["p_value"])

    adaptive_selected = adaptive_search.get("retrain_record", {})
    adaptive_control_record = adaptive_control.get("retrain_record", {})

    adaptive_gain, adaptive_gain_lo, adaptive_gain_hi = bootstrap_mean_diff_ci(
        adaptive_selected.get("trained_pairwise_pwmcc_values", []),
        adaptive_control_record.get("trained_pairwise_pwmcc_values", []),
        seed=37,
    )

    selected_lambda = float(consistency["selection"]["selected_lambda"])
    baseline_rec, selected_rec = find_consistency_baseline_and_selected(consistency["records"], selected_lambda)
    consistency_gain, consistency_gain_lo, consistency_gain_hi = bootstrap_mean_diff_ci(
        selected_rec.get("trained_pairwise_pwmcc_values", []),
        baseline_rec.get("trained_pairwise_pwmcc_values", []),
        seed=53,
    )

    checks = [
        {
            "id": "phase4a_training_signal",
            "pass": bool(phase4a_delta > 0 and phase4a_p < 0.05),
            "evidence": {
                "delta": phase4a_delta,
                "p_value": phase4a_p,
            },
        },
        {
            "id": "core_k_sweep_has_positive_delta",
            "pass": bool(float(core_k_best.get("delta_pwmcc", 0.0)) > 0.0),
            "evidence": {
                "best_k": int(core_k_best.get("k", -1)),
                "best_k_delta": float(core_k_best.get("delta_pwmcc", float("nan"))),
            },
        },
        {
            "id": "core_d_sae_sweep_has_positive_delta",
            "pass": bool(float(core_d_best.get("delta_pwmcc", 0.0)) > 0.0),
            "evidence": {
                "best_d_sae": int(core_d_best.get("d_sae", -1)),
                "best_d_delta": float(core_d_best.get("delta_pwmcc", float("nan"))),
            },
        },
        {
            "id": "adaptive_low_k_outperforms_k32_control",
            "pass": bool(adaptive_gain > 0 and adaptive_gain_lo > 0),
            "evidence": {
                "search_selected_k": int(adaptive_search.get("selection", {}).get("selected_k", -1)),
                "control_k": int(adaptive_control.get("selection", {}).get("selected_k", -1)),
                "trained_pwmcc_gain": adaptive_gain,
                "gain_ci95": [adaptive_gain_lo, adaptive_gain_hi],
            },
        },
        {
            "id": "consistency_regularizer_effect_is_small_or_unresolved",
            "pass": bool(consistency_gain_lo <= 0 <= consistency_gain_hi),
            "evidence": {
                "selected_lambda": selected_lambda,
                "gain": consistency_gain,
                "gain_ci95": [consistency_gain_lo, consistency_gain_hi],
            },
        },
    ]

    overall_pass = all(check["pass"] for check in checks)

    report = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "config_hash": stable_hash(
                {
                    "phase4a": str(args.phase4a_results),
                    "core": str(core_path),
                    "adaptive_search": str(adaptive_search_path),
                    "adaptive_control": str(adaptive_control_path),
                    "consistency": str(consistency_path),
                }
            ),
        },
        "inputs": {
            "phase4a": str(args.phase4a_results.relative_to(PROJECT_ROOT)),
            "core": str(core_path.relative_to(PROJECT_ROOT)),
            "adaptive_search": str(adaptive_search_path.relative_to(PROJECT_ROOT)),
            "adaptive_control": str(adaptive_control_path.relative_to(PROJECT_ROOT)),
            "consistency": str(consistency_path.relative_to(PROJECT_ROOT)),
        },
        "checks": checks,
        "overall_pass": overall_pass,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Experiment Consistency Report",
        "",
        f"- Timestamp (UTC): {report['run_metadata']['timestamp_utc']}",
        f"- Git commit: `{report['run_metadata']['git_commit']}`",
        f"- Overall pass: `{overall_pass}`",
        "",
        "## Inputs",
        f"- Phase4a: `{report['inputs']['phase4a']}`",
        f"- Core ablations: `{report['inputs']['core']}`",
        f"- Adaptive search: `{report['inputs']['adaptive_search']}`",
        f"- Adaptive control: `{report['inputs']['adaptive_control']}`",
        f"- Consistency sweep: `{report['inputs']['consistency']}`",
        "",
        "## Checks",
        "",
        "| check | pass | evidence |",
        "|---|---:|---|",
    ]

    for check in checks:
        evidence = json.dumps(check["evidence"], sort_keys=True)
        lines.append(f"| {check['id']} | {check['pass']} | `{evidence}` |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")

    print("Experiment consistency verification complete")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
