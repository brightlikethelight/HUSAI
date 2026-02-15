#!/usr/bin/env python3
"""Evaluate stress-gated release policy for SAE experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text())


def maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_phase4a_delta(payload: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if payload is None:
        return None, None

    delta = None
    lcb = None
    comp = payload.get("comparison") or {}
    trained = payload.get("trained") or {}
    random = payload.get("random") or {}

    if isinstance(comp.get("difference"), (int, float)):
        delta = float(comp.get("difference"))
    if isinstance(trained.get("ci95_low"), (int, float)) and isinstance(random.get("ci95_high"), (int, float)):
        lcb = float(trained.get("ci95_low")) - float(random.get("ci95_high"))
    return delta, lcb


def extract_legacy_external_delta(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    candidates: list[Any] = [
        (payload.get("summary") or {}).get("best_minus_llm_auc"),
        payload.get("best_minus_llm_auc"),
        (payload.get("delta_vs_matched_baseline") or {}).get("interpretability_score_mean_max"),
        (payload.get("delta_vs_matched_baseline") or {}).get("contrastive_score_mean_max"),
        (payload.get("delta_vs_matched_baseline") or {}).get("independent_score_mean_max"),
        payload.get("external_delta"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _pick_first_float(candidates: list[Any]) -> float | None:
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_external_metrics(payload: dict[str, Any] | None) -> dict[str, float | None]:
    if payload is None:
        return {
            "saebench_delta": None,
            "saebench_delta_ci95_low": None,
            "cebench_interp_delta_vs_baseline": None,
            "cebench_interp_delta_vs_baseline_ci95_low": None,
            "cebench_interpretability_max": None,
            "legacy_external_delta": None,
        }

    obj = payload
    selected = payload.get("selected_candidate")
    if isinstance(selected, dict):
        obj = selected

    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}

    saebench_delta = _pick_first_float(
        [
            metrics.get("saebench_delta"),
            obj.get("saebench_delta"),
            obj.get("best_minus_llm_auc"),
            (obj.get("summary") or {}).get("best_minus_llm_auc"),
        ]
    )

    saebench_delta_lcb = _pick_first_float(
        [
            metrics.get("saebench_delta_ci95_low"),
            obj.get("saebench_delta_ci95_low"),
        ]
    )

    cebench_delta = _pick_first_float(
        [
            metrics.get("cebench_interp_delta_vs_baseline"),
            metrics.get("cebench_delta"),
            obj.get("cebench_interp_delta_vs_baseline"),
            obj.get("cebench_delta"),
            (obj.get("delta_vs_matched_baseline") or {}).get("interpretability_score_mean_max"),
        ]
    )

    cebench_delta_lcb = _pick_first_float(
        [
            metrics.get("cebench_interp_delta_vs_baseline_ci95_low"),
            obj.get("cebench_interp_delta_vs_baseline_ci95_low"),
        ]
    )

    cebench_interp = _pick_first_float(
        [
            metrics.get("cebench_interpretability_max"),
            obj.get("cebench_interpretability_max"),
            (obj.get("custom_metrics") or {}).get("interpretability_score_mean_max"),
            (obj.get("cebench_summary") or {}).get("interpretability_score_mean_max"),
        ]
    )

    legacy_external_delta = extract_legacy_external_delta(obj)
    if legacy_external_delta is None:
        legacy_external_delta = saebench_delta if saebench_delta is not None else cebench_delta

    return {
        "saebench_delta": saebench_delta,
        "saebench_delta_ci95_low": saebench_delta_lcb,
        "cebench_interp_delta_vs_baseline": cebench_delta,
        "cebench_interp_delta_vs_baseline_ci95_low": cebench_delta_lcb,
        "cebench_interpretability_max": cebench_interp,
        "legacy_external_delta": legacy_external_delta,
    }


def extract_transcoder_delta(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    candidates: list[Any] = [
        payload.get("delta"),
        payload.get("transcoder_delta"),
        (payload.get("metrics") or {}).get("delta"),
        (payload.get("metrics") or {}).get("transcoder_delta"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_ood_drop(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    candidates: list[Any] = [
        payload.get("ood_drop"),
        payload.get("relative_drop"),
        payload.get("drop"),
        (payload.get("metrics") or {}).get("ood_drop"),
        (payload.get("metrics") or {}).get("relative_drop"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def threshold_gate(value: float | None, threshold: float, required: bool) -> bool:
    if value is None:
        return not required
    return float(value) >= float(threshold)


def evaluate_external_gates(
    *,
    external_mode: str,
    require_external: bool,
    min_external_delta: float,
    min_saebench_delta: float,
    min_cebench_delta: float,
    min_saebench_delta_lcb: float | None,
    min_cebench_delta_lcb: float | None,
    use_external_lcb: bool,
    external_metrics: dict[str, float | None],
) -> tuple[bool, bool, bool, str, float | None, float | None, float | None]:
    saebench_delta = external_metrics.get("saebench_delta")
    cebench_delta = external_metrics.get("cebench_interp_delta_vs_baseline")
    saebench_delta_lcb = external_metrics.get("saebench_delta_ci95_low")
    cebench_delta_lcb = external_metrics.get("cebench_interp_delta_vs_baseline_ci95_low")
    legacy_external_delta = external_metrics.get("legacy_external_delta")

    if use_external_lcb:
        saebench_gate_value = saebench_delta_lcb
        cebench_gate_value = cebench_delta_lcb
        saebench_threshold = (
            float(min_saebench_delta_lcb)
            if min_saebench_delta_lcb is not None
            else float(min_saebench_delta)
        )
        cebench_threshold = (
            float(min_cebench_delta_lcb)
            if min_cebench_delta_lcb is not None
            else float(min_cebench_delta)
        )
    else:
        saebench_gate_value = saebench_delta
        cebench_gate_value = cebench_delta
        saebench_threshold = float(min_saebench_delta)
        cebench_threshold = float(min_cebench_delta)

    gate_external_saebench = threshold_gate(saebench_gate_value, saebench_threshold, require_external)
    gate_external_cebench = threshold_gate(cebench_gate_value, cebench_threshold, require_external)

    mode_used = external_mode
    if external_mode == "saebench":
        gate_external = gate_external_saebench
    elif external_mode == "cebench":
        gate_external = gate_external_cebench
    elif external_mode == "joint":
        gate_external = gate_external_saebench and gate_external_cebench
    elif external_mode == "auto":
        # Prefer joint gating if both metrics are present.
        if use_external_lcb:
            if saebench_delta_lcb is not None and cebench_delta_lcb is not None:
                gate_external = gate_external_saebench and gate_external_cebench
                mode_used = "auto_joint_lcb"
            else:
                gate_external = False if require_external else True
                mode_used = "auto_lcb_missing"
        else:
            if saebench_delta is not None and cebench_delta is not None:
                gate_external = gate_external_saebench and gate_external_cebench
                mode_used = "auto_joint"
            else:
                gate_external = threshold_gate(legacy_external_delta, min_external_delta, require_external)
                mode_used = "auto_legacy"
    else:
        raise ValueError(f"Unsupported external_mode: {external_mode}")

    return (
        gate_external,
        gate_external_saebench,
        gate_external_cebench,
        mode_used,
        legacy_external_delta,
        saebench_gate_value,
        cebench_gate_value,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-gated release policy evaluator")
    parser.add_argument(
        "--phase4a-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4a_trained_vs_random" / "results.json",
    )
    parser.add_argument("--transcoder-results", type=Path, default=None)
    parser.add_argument("--ood-results", type=Path, default=None)
    parser.add_argument("--external-summary", type=Path, default=None)
    parser.add_argument("--external-candidate-json", type=Path, default=None)

    parser.add_argument("--min-trained-random-delta", type=float, default=0.0)
    parser.add_argument("--min-trained-random-lcb", type=float, default=0.0)
    parser.add_argument("--min-transcoder-delta", type=float, default=0.0)
    parser.add_argument("--max-ood-drop", type=float, default=0.1)
    parser.add_argument("--min-external-delta", type=float, default=0.0)
    parser.add_argument("--min-saebench-delta", type=float, default=0.0)
    parser.add_argument("--min-cebench-delta", type=float, default=0.0)
    parser.add_argument(
        "--min-saebench-delta-lcb",
        type=float,
        default=None,
        help="Optional LCB threshold for SAEBench delta when --use-external-lcb is set.",
    )
    parser.add_argument(
        "--min-cebench-delta-lcb",
        type=float,
        default=None,
        help="Optional LCB threshold for CE-Bench delta when --use-external-lcb is set.",
    )
    parser.add_argument(
        "--use-external-lcb",
        action="store_true",
        help="Gate external metrics using conservative CI lower bounds when available.",
    )
    parser.add_argument(
        "--external-mode",
        type=str,
        choices=["auto", "saebench", "cebench", "joint"],
        default="auto",
    )

    parser.add_argument("--require-transcoder", action="store_true")
    parser.add_argument("--require-ood", action="store_true")
    parser.add_argument("--require-external", action="store_true")
    parser.add_argument("--fail-on-gate-fail", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "release_stress_gates",
    )
    args = parser.parse_args()

    phase4a = load_json(args.phase4a_results)
    transcoder = load_json(args.transcoder_results)
    ood = load_json(args.ood_results)

    external_source = args.external_candidate_json if args.external_candidate_json is not None else args.external_summary
    external_payload = load_json(external_source)

    trained_delta, trained_lcb = extract_phase4a_delta(phase4a)
    transcoder_delta = extract_transcoder_delta(transcoder)
    ood_drop = extract_ood_drop(ood)
    external_metrics = extract_external_metrics(external_payload)

    gate_random = (
        trained_delta is not None
        and trained_lcb is not None
        and trained_delta >= args.min_trained_random_delta
        and trained_lcb >= args.min_trained_random_lcb
    )

    if transcoder_delta is None:
        gate_transcoder = not args.require_transcoder
    else:
        gate_transcoder = transcoder_delta >= args.min_transcoder_delta

    if ood_drop is None:
        gate_ood = not args.require_ood
    else:
        gate_ood = ood_drop <= args.max_ood_drop

    (
        gate_external,
        gate_external_saebench,
        gate_external_cebench,
        external_mode_used,
        legacy_external_delta,
        saebench_gate_value,
        cebench_gate_value,
    ) = evaluate_external_gates(
        external_mode=args.external_mode,
        require_external=args.require_external,
        min_external_delta=args.min_external_delta,
        min_saebench_delta=args.min_saebench_delta,
        min_cebench_delta=args.min_cebench_delta,
        min_saebench_delta_lcb=args.min_saebench_delta_lcb,
        min_cebench_delta_lcb=args.min_cebench_delta_lcb,
        use_external_lcb=args.use_external_lcb,
        external_metrics=external_metrics,
    )

    pass_all = bool(gate_random and gate_transcoder and gate_ood and gate_external)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "run_id": run_id,
        },
        "inputs": {
            "phase4a_results": repo_rel(args.phase4a_results),
            "transcoder_results": repo_rel(args.transcoder_results),
            "ood_results": repo_rel(args.ood_results),
            "external_summary": repo_rel(args.external_summary),
            "external_candidate_json": repo_rel(args.external_candidate_json),
            "external_source_used": repo_rel(external_source),
        },
        "thresholds": {
            "min_trained_random_delta": args.min_trained_random_delta,
            "min_trained_random_lcb": args.min_trained_random_lcb,
            "min_transcoder_delta": args.min_transcoder_delta,
            "max_ood_drop": args.max_ood_drop,
            "min_external_delta": args.min_external_delta,
            "min_saebench_delta": args.min_saebench_delta,
            "min_cebench_delta": args.min_cebench_delta,
            "min_saebench_delta_lcb": args.min_saebench_delta_lcb,
            "min_cebench_delta_lcb": args.min_cebench_delta_lcb,
            "use_external_lcb": args.use_external_lcb,
            "external_mode": args.external_mode,
            "require_transcoder": args.require_transcoder,
            "require_ood": args.require_ood,
            "require_external": args.require_external,
        },
        "metrics": {
            "trained_random_delta": trained_delta,
            "trained_random_delta_lcb": trained_lcb,
            "transcoder_delta": transcoder_delta,
            "ood_drop": ood_drop,
            "external_delta": legacy_external_delta,
            "saebench_delta": external_metrics.get("saebench_delta"),
            "saebench_delta_ci95_low": external_metrics.get("saebench_delta_ci95_low"),
            "cebench_interp_delta_vs_baseline": external_metrics.get("cebench_interp_delta_vs_baseline"),
            "cebench_interp_delta_vs_baseline_ci95_low": external_metrics.get("cebench_interp_delta_vs_baseline_ci95_low"),
            "cebench_interpretability_max": external_metrics.get("cebench_interpretability_max"),
            "external_gate_saebench_value": saebench_gate_value,
            "external_gate_cebench_value": cebench_gate_value,
        },
        "gates": {
            "random_model": gate_random,
            "transcoder": gate_transcoder,
            "ood": gate_ood,
            "external": gate_external,
            "external_saebench": gate_external_saebench,
            "external_cebench": gate_external_cebench,
            "external_mode_used": external_mode_used,
            "external_threshold_basis": "lcb" if args.use_external_lcb else "point",
            "pass_all": pass_all,
        },
    }

    out_json = run_dir / "release_policy.json"
    out_md = run_dir / "release_policy.md"

    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Stress-Gated Release Policy",
        "",
        f"- Run ID: `{run_id}`",
        f"- pass_all: `{pass_all}`",
        f"- external_mode_used: `{external_mode_used}`",
        f"- external_threshold_basis: `{'lcb' if args.use_external_lcb else 'point'}`",
        "",
        "## Gate Status",
        "",
        f"- random_model: `{gate_random}`",
        f"- transcoder: `{gate_transcoder}`",
        f"- ood: `{gate_ood}`",
        f"- external: `{gate_external}`",
        f"- external_saebench: `{gate_external_saebench}`",
        f"- external_cebench: `{gate_external_cebench}`",
        "",
        "## Metrics",
        "",
        f"- trained_random_delta: `{trained_delta}`",
        f"- trained_random_delta_lcb: `{trained_lcb}`",
        f"- transcoder_delta: `{transcoder_delta}`",
        f"- ood_drop: `{ood_drop}`",
        f"- external_delta (legacy): `{legacy_external_delta}`",
        f"- saebench_delta: `{external_metrics.get('saebench_delta')}`",
        f"- saebench_delta_ci95_low: `{external_metrics.get('saebench_delta_ci95_low')}`",
        f"- cebench_interp_delta_vs_baseline: `{external_metrics.get('cebench_interp_delta_vs_baseline')}`",
        f"- cebench_interp_delta_vs_baseline_ci95_low: `{external_metrics.get('cebench_interp_delta_vs_baseline_ci95_low')}`",
        f"- cebench_interpretability_max: `{external_metrics.get('cebench_interpretability_max')}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("Stress-gated release policy evaluation complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")

    if args.fail_on_gate_fail and not pass_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
