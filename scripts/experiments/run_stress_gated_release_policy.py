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


def extract_external_delta(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    candidates: list[Any] = [
        (payload.get("summary") or {}).get("best_minus_llm_auc"),
        payload.get("best_minus_llm_auc"),
        (payload.get("delta_vs_matched_baseline") or {}).get("interpretability_score_mean_max"),
        (payload.get("delta_vs_matched_baseline") or {}).get("contrastive_score_mean_max"),
        (payload.get("delta_vs_matched_baseline") or {}).get("independent_score_mean_max"),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
    return None


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

    parser.add_argument("--min-trained-random-delta", type=float, default=0.0)
    parser.add_argument("--min-trained-random-lcb", type=float, default=0.0)
    parser.add_argument("--min-transcoder-delta", type=float, default=0.0)
    parser.add_argument("--max-ood-drop", type=float, default=0.1)
    parser.add_argument("--min-external-delta", type=float, default=0.0)

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
    external = load_json(args.external_summary)

    trained_delta, trained_lcb = extract_phase4a_delta(phase4a)
    transcoder_delta = extract_transcoder_delta(transcoder)
    ood_drop = extract_ood_drop(ood)
    external_delta = extract_external_delta(external)

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

    if external_delta is None:
        gate_external = not args.require_external
    else:
        gate_external = external_delta >= args.min_external_delta

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
        },
        "thresholds": {
            "min_trained_random_delta": args.min_trained_random_delta,
            "min_trained_random_lcb": args.min_trained_random_lcb,
            "min_transcoder_delta": args.min_transcoder_delta,
            "max_ood_drop": args.max_ood_drop,
            "min_external_delta": args.min_external_delta,
            "require_transcoder": args.require_transcoder,
            "require_ood": args.require_ood,
            "require_external": args.require_external,
        },
        "metrics": {
            "trained_random_delta": trained_delta,
            "trained_random_delta_lcb": trained_lcb,
            "transcoder_delta": transcoder_delta,
            "ood_drop": ood_drop,
            "external_delta": external_delta,
        },
        "gates": {
            "random_model": gate_random,
            "transcoder": gate_transcoder,
            "ood": gate_ood,
            "external": gate_external,
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
        "",
        "## Gate Status",
        "",
        f"- random_model: `{gate_random}`",
        f"- transcoder: `{gate_transcoder}`",
        f"- ood: `{gate_ood}`",
        f"- external: `{gate_external}`",
        "",
        "## Metrics",
        "",
        f"- trained_random_delta: `{trained_delta}`",
        f"- trained_random_delta_lcb: `{trained_lcb}`",
        f"- transcoder_delta: `{transcoder_delta}`",
        f"- ood_drop: `{ood_drop}`",
        f"- external_delta: `{external_delta}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print("Stress-gated release policy evaluation complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")

    if args.fail_on_gate_fail and not pass_all:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
