#!/usr/bin/env python3
"""Select a release candidate from external benchmark runs.

This selector consumes frontier/scaling result JSON files, constructs per-checkpoint
candidate rows, computes a Pareto front over SAEBench and CE-Bench deltas, and
selects a final candidate using a weighted normalized score.

It also supports uncertainty-aware selection by grouping candidates across seeds
per condition and scoring conservative lower-confidence-bound (LCB) metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def maybe_float(value: Any) -> float | None:
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


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def infer_condition_id(record: dict[str, Any]) -> str:
    if isinstance(record.get("condition_id"), str):
        return str(record["condition_id"])
    arch = str(record.get("architecture", "unknown"))
    seed = record.get("seed", "na")
    return f"{arch}_seed{seed}"


def extract_train_ev(record: dict[str, Any]) -> float | None:
    train_metrics = record.get("train_metrics") or {}
    ev = maybe_float(train_metrics.get("explained_variance"))
    if ev is not None:
        return ev

    train_summary = record.get("train_summary") or {}
    final_metrics = train_summary.get("final_metrics") or {}
    return maybe_float(final_metrics.get("explained_variance"))


def extract_saebench_delta(record: dict[str, Any]) -> float | None:
    saebench = record.get("saebench") or {}
    summary = saebench.get("summary") or {}
    return maybe_float(summary.get("best_minus_llm_auc") or saebench.get("best_minus_llm_auc"))


def extract_cebench_interp(record: dict[str, Any]) -> float | None:
    cebench = record.get("cebench") or {}
    custom_metrics = cebench.get("custom_metrics") or {}
    value = maybe_float(custom_metrics.get("interpretability_score_mean_max"))
    if value is not None:
        return value

    cebench_summary = cebench.get("cebench_summary") or {}
    return maybe_float(cebench_summary.get("interpretability_score_mean_max"))


def extract_cebench_delta(record: dict[str, Any]) -> float | None:
    cebench = record.get("cebench") or {}
    delta = cebench.get("delta_vs_matched_baseline") or {}
    return maybe_float(delta.get("interpretability_score_mean_max"))


def extract_architecture(record: dict[str, Any]) -> str:
    arch = record.get("architecture")
    if isinstance(arch, str):
        return arch
    cebench = record.get("cebench") or {}
    sae_meta = cebench.get("sae_meta") or {}
    arch = sae_meta.get("architecture")
    if isinstance(arch, str):
        return arch
    saebench = record.get("saebench") or {}
    sae_meta = saebench.get("sae_meta") or {}
    arch = sae_meta.get("architecture")
    if isinstance(arch, str):
        return arch
    return "unknown"


def extract_hook_info(record: dict[str, Any]) -> tuple[int | None, str | None]:
    hook_layer = maybe_float(record.get("hook_layer"))
    hook_name = record.get("hook_name") if isinstance(record.get("hook_name"), str) else None

    cebench = record.get("cebench") or {}
    cfg = cebench.get("config") or {}
    if hook_layer is None:
        maybe_layer = maybe_float(cfg.get("hook_layer"))
        if maybe_layer is not None:
            hook_layer = maybe_layer
    if hook_name is None and isinstance(cfg.get("hook_name"), str):
        hook_name = str(cfg.get("hook_name"))

    saebench = record.get("saebench") or {}
    cfg = saebench.get("config") or {}
    if hook_layer is None:
        maybe_layer = maybe_float(cfg.get("hook_layer"))
        if maybe_layer is not None:
            hook_layer = maybe_layer
    if hook_name is None and isinstance(cfg.get("hook_name"), str):
        hook_name = str(cfg.get("hook_name"))

    return (int(hook_layer) if hook_layer is not None else None, hook_name)


def build_summary_paths(checkpoint: Path, condition_id: str) -> tuple[Path, Path]:
    run_dir = checkpoint.parents[2]
    saebench_summary = run_dir / "external_eval" / condition_id / "saebench" / "husai_custom_sae_summary.json"
    cebench_summary = run_dir / "external_eval" / condition_id / "cebench" / "husai_custom_cebench_summary.json"
    return saebench_summary, cebench_summary


def extract_candidates(results_path: Path, source: str) -> list[dict[str, Any]]:
    payload = load_json(results_path)
    records = payload.get("records") or []
    out: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        condition_id = infer_condition_id(rec)
        checkpoint_raw = rec.get("checkpoint")
        if not isinstance(checkpoint_raw, str):
            continue

        checkpoint = to_abs(Path(checkpoint_raw))
        saebench_summary_path, cebench_summary_path = build_summary_paths(checkpoint, condition_id)

        hook_layer, hook_name = extract_hook_info(rec)
        candidate = {
            "source": source,
            "source_results": repo_rel(results_path),
            "row_index": idx,
            "condition_id": condition_id,
            "architecture": extract_architecture(rec),
            "seed": rec.get("seed"),
            "hook_layer": hook_layer,
            "hook_name": hook_name,
            "checkpoint": repo_rel(checkpoint),
            "saebench_summary_path": repo_rel(saebench_summary_path),
            "cebench_summary_path": repo_rel(cebench_summary_path),
            "metrics": {
                "saebench_delta": extract_saebench_delta(rec),
                "cebench_interp_delta_vs_baseline": extract_cebench_delta(rec),
                "cebench_interpretability_max": extract_cebench_interp(rec),
                "train_explained_variance": extract_train_ev(rec),
            },
            "returncodes": {
                "saebench": rec.get("saebench_returncode"),
                "cebench": rec.get("cebench_returncode"),
            },
        }
        out.append(candidate)

    return out


def normalize_metric(values: list[float | None]) -> dict[int, float]:
    valid = [(i, v) for i, v in enumerate(values) if v is not None and not math.isnan(float(v))]
    if not valid:
        return {}

    only_vals = [float(v) for _, v in valid]
    lo = min(only_vals)
    hi = max(only_vals)
    if hi <= lo:
        return {i: 0.5 for i, _ in valid}
    return {i: (float(v) - lo) / (hi - lo) for i, v in valid}


def confidence_stats(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
        }

    mean = float(statistics.fmean(clean))
    if len(clean) > 1:
        std = float(statistics.stdev(clean))
        half_width = 1.96 * std / math.sqrt(len(clean))
    else:
        std = 0.0
        half_width = 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": len(clean),
    }


def pick_metric_from_stats(stats: dict[str, float | int | None], mode: str) -> float | None:
    if mode == "lcb":
        return maybe_float(stats.get("ci95_low"))
    return maybe_float(stats.get("mean"))


def infer_group_id(condition_id: str) -> str:
    # topk_seed42 -> topk, tok10000_layer0_dsae1024_seed42 -> tok10000_layer0_dsae1024
    return re.sub(r"_seed\d+$", "", condition_id)


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_sae = maybe_float((a.get("metrics") or {}).get("saebench_delta"))
    a_ce = maybe_float((a.get("metrics") or {}).get("cebench_interp_delta_vs_baseline"))
    b_sae = maybe_float((b.get("metrics") or {}).get("saebench_delta"))
    b_ce = maybe_float((b.get("metrics") or {}).get("cebench_interp_delta_vs_baseline"))

    if a_sae is None or a_ce is None or b_sae is None or b_ce is None:
        return False

    ge_all = a_sae >= b_sae and a_ce >= b_ce
    gt_any = a_sae > b_sae or a_ce > b_ce
    return ge_all and gt_any


def annotate_scores(
    candidates: list[dict[str, Any]],
    weight_saebench: float,
    weight_cebench: float,
    weight_train_ev: float,
) -> None:
    sae_vals = [maybe_float((c.get("metrics") or {}).get("saebench_delta")) for c in candidates]
    ce_vals = [maybe_float((c.get("metrics") or {}).get("cebench_interp_delta_vs_baseline")) for c in candidates]
    ev_vals = [maybe_float((c.get("metrics") or {}).get("train_explained_variance")) for c in candidates]

    sae_norm = normalize_metric(sae_vals)
    ce_norm = normalize_metric(ce_vals)
    ev_norm = normalize_metric(ev_vals)

    for i, cand in enumerate(candidates):
        score = 0.0
        detail: dict[str, float] = {}

        if i in sae_norm:
            score += weight_saebench * sae_norm[i]
            detail["saebench_norm"] = sae_norm[i]
        if i in ce_norm:
            score += weight_cebench * ce_norm[i]
            detail["cebench_delta_norm"] = ce_norm[i]
        if i in ev_norm:
            score += weight_train_ev * ev_norm[i]
            detail["train_ev_norm"] = ev_norm[i]

        selection = dict(cand.get("selection") or {})
        selection.update(
            {
                "joint_score": score,
                "normalized_components": detail,
                "weights": {
                    "saebench": weight_saebench,
                    "cebench_delta": weight_cebench,
                    "train_ev": weight_train_ev,
                },
            }
        )
        cand["selection"] = selection


def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_saebench_delta: float,
    min_cebench_delta: float,
    require_both_external: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for cand in candidates:
        metrics = cand.get("metrics") or {}
        sae = maybe_float(metrics.get("saebench_delta"))
        ce = maybe_float(metrics.get("cebench_interp_delta_vs_baseline"))
        rc = cand.get("returncodes") or {}

        if require_both_external and (sae is None or ce is None):
            continue
        if sae is not None and sae < min_saebench_delta:
            continue
        if ce is not None and ce < min_cebench_delta:
            continue
        if rc.get("saebench") not in (None, 0):
            continue
        if rc.get("cebench") not in (None, 0):
            continue

        out.append(cand)

    return out


def compute_pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for cand in candidates:
        dominated = any(dominates(other, cand) for other in candidates if other is not cand)
        cand.setdefault("selection", {})["is_pareto"] = not dominated
        if not dominated:
            front.append(cand)
    return front


def to_serializable(candidate: dict[str, Any]) -> dict[str, Any]:
    # candidates are already JSON-safe; return shallow copy to avoid accidental mutation.
    return json.loads(json.dumps(candidate))


def aggregate_condition_groups(
    candidates: list[dict[str, Any]],
    *,
    uncertainty_mode: str,
    min_seeds_per_group: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        condition_id = str(cand.get("condition_id", "unknown"))
        grouped[infer_group_id(condition_id)].append(cand)

    out: list[dict[str, Any]] = []
    for group_id, rows in grouped.items():
        if len(rows) < min_seeds_per_group:
            continue

        rep = max(
            rows,
            key=lambda c: float((c.get("selection") or {}).get("joint_score", float("-inf"))),
        )
        agg = to_serializable(rep)
        agg["source"] = "grouped"
        agg["sources"] = sorted({str(r.get("source", "unknown")) for r in rows})
        agg["group_id"] = group_id
        agg["condition_id"] = group_id
        agg["representative_condition_id"] = rep.get("condition_id")
        agg["seed_count"] = len(rows)
        agg["seed_candidates"] = [
            {
                "seed": r.get("seed"),
                "condition_id": r.get("condition_id"),
                "checkpoint": r.get("checkpoint"),
                "joint_score": (r.get("selection") or {}).get("joint_score"),
                "metrics": r.get("metrics"),
            }
            for r in sorted(rows, key=lambda r: str(r.get("seed")))
        ]

        metrics = dict(agg.get("metrics") or {})
        metric_specs = [
            "saebench_delta",
            "cebench_interp_delta_vs_baseline",
            "train_explained_variance",
            "cebench_interpretability_max",
        ]
        for key in metric_specs:
            stats = confidence_stats([maybe_float((r.get("metrics") or {}).get(key)) for r in rows])
            metrics[f"{key}_mean"] = maybe_float(stats.get("mean"))
            metrics[f"{key}_std"] = maybe_float(stats.get("std"))
            metrics[f"{key}_ci95_low"] = maybe_float(stats.get("ci95_low"))
            metrics[f"{key}_ci95_high"] = maybe_float(stats.get("ci95_high"))
            metrics[f"{key}_n"] = int(stats.get("n") or 0)
            # Primary metric used for filtering/pareto/scoring in grouped mode.
            metrics[key] = pick_metric_from_stats(stats, uncertainty_mode)

        agg["metrics"] = metrics
        agg["selection"] = {
            "grouped_from_seed_candidates": len(rows),
            "uncertainty_mode": uncertainty_mode,
            "representative_seed": rep.get("seed"),
        }
        out.append(agg)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best release candidate from external benchmark runs")
    parser.add_argument("--frontier-results", type=Path, action="append", default=[])
    parser.add_argument("--scaling-results", type=Path, action="append", default=[])

    parser.add_argument("--min-saebench-delta", type=float, default=-1e9)
    parser.add_argument("--min-cebench-delta", type=float, default=-1e9)
    parser.add_argument("--require-both-external", action="store_true")
    parser.add_argument(
        "--group-by-condition",
        dest="group_by_condition",
        action="store_true",
        help="Aggregate candidates across seeds by condition id stem (e.g. *_seed42 -> *).",
    )
    parser.add_argument(
        "--seed-level-selection",
        dest="group_by_condition",
        action="store_false",
        help="Disable grouped selection and score individual seed checkpoints.",
    )
    parser.add_argument(
        "--uncertainty-mode",
        type=str,
        choices=["point", "lcb"],
        default="lcb",
        help="Metric mode when grouping: point=mean, lcb=95% lower confidence bound.",
    )
    parser.add_argument("--min-seeds-per-group", type=int, default=3)

    parser.set_defaults(group_by_condition=True)

    parser.add_argument("--weight-saebench", type=float, default=0.5)
    parser.add_argument("--weight-cebench", type=float, default=0.5)
    parser.add_argument("--weight-train-ev", type=float, default=0.1)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "release_candidate_selection",
    )
    args = parser.parse_args()

    frontier_paths = [to_abs(p) for p in args.frontier_results]
    scaling_paths = [to_abs(p) for p in args.scaling_results]

    if not frontier_paths and not scaling_paths:
        raise ValueError("Pass at least one --frontier-results or --scaling-results file")

    output_dir = to_abs(args.output_dir)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    for path in frontier_paths:
        all_candidates.extend(extract_candidates(path, source="frontier"))
    for path in scaling_paths:
        all_candidates.extend(extract_candidates(path, source="scaling"))

    filtered = filter_candidates(
        all_candidates,
        min_saebench_delta=args.min_saebench_delta,
        min_cebench_delta=args.min_cebench_delta,
        require_both_external=args.require_both_external,
    )
    if not filtered:
        raise RuntimeError("No eligible candidates after filtering thresholds/returncodes")

    # Seed-level score annotation is reused when selecting representative checkpoints
    # for grouped candidates.
    annotate_scores(
        filtered,
        weight_saebench=args.weight_saebench,
        weight_cebench=args.weight_cebench,
        weight_train_ev=args.weight_train_ev,
    )

    selected_population = filtered
    if args.group_by_condition:
        selected_population = aggregate_condition_groups(
            filtered,
            uncertainty_mode=args.uncertainty_mode,
            min_seeds_per_group=max(1, args.min_seeds_per_group),
        )
        if not selected_population:
            raise RuntimeError(
                "No grouped candidates after applying --group-by-condition and --min-seeds-per-group"
            )
        annotate_scores(
            selected_population,
            weight_saebench=args.weight_saebench,
            weight_cebench=args.weight_cebench,
            weight_train_ev=args.weight_train_ev,
        )

    pareto = compute_pareto_front(selected_population)

    ordered = sorted(
        selected_population,
        key=lambda c: (
            bool((c.get("selection") or {}).get("is_pareto", False)),
            float((c.get("selection") or {}).get("joint_score", float("-inf"))),
            maybe_float((c.get("metrics") or {}).get("saebench_delta")) or float("-inf"),
            maybe_float((c.get("metrics") or {}).get("cebench_interp_delta_vs_baseline")) or float("-inf"),
        ),
        reverse=True,
    )

    selected = ordered[0]

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "run_id": run_id,
        },
        "config": {
            "frontier_results": [repo_rel(p) for p in frontier_paths],
            "scaling_results": [repo_rel(p) for p in scaling_paths],
            "min_saebench_delta": args.min_saebench_delta,
            "min_cebench_delta": args.min_cebench_delta,
            "require_both_external": args.require_both_external,
            "group_by_condition": args.group_by_condition,
            "uncertainty_mode": args.uncertainty_mode,
            "min_seeds_per_group": args.min_seeds_per_group,
            "weight_saebench": args.weight_saebench,
            "weight_cebench": args.weight_cebench,
            "weight_train_ev": args.weight_train_ev,
        },
        "counts": {
            "total_candidates": len(all_candidates),
            "eligible_seed_candidates": len(filtered),
            "eligible_candidates": len(selected_population),
            "pareto_count": len(pareto),
        },
        "selected_candidate": to_serializable(selected),
        "pareto_front": [to_serializable(c) for c in pareto],
        "ranked_candidates": [to_serializable(c) for c in ordered],
    }

    summary_json = run_dir / "selection_summary.json"
    selected_json = run_dir / "selected_candidate.json"
    pareto_json = run_dir / "pareto_front.json"
    table_json = run_dir / "candidate_table.json"
    summary_md = run_dir / "summary.md"

    summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    selected_json.write_text(json.dumps(payload["selected_candidate"], indent=2) + "\n")
    pareto_json.write_text(json.dumps(payload["pareto_front"], indent=2) + "\n")
    table_json.write_text(json.dumps(payload["ranked_candidates"], indent=2) + "\n")

    sel_metrics = selected.get("metrics") or {}
    sel_score = (selected.get("selection") or {}).get("joint_score")
    lines = [
        "# Release Candidate Selection",
        "",
        f"- Run ID: `{run_id}`",
        f"- Selection mode: `{'grouped' if args.group_by_condition else 'seed'}`",
        f"- Uncertainty mode: `{args.uncertainty_mode}`",
        f"- Eligible seed candidates: `{len(filtered)}`",
        f"- Eligible candidates: `{len(selected_population)}`",
        f"- Pareto candidates: `{len(pareto)}`",
        "",
        "## Selected Candidate",
        "",
        f"- source: `{selected.get('source')}`",
        f"- condition_id: `{selected.get('condition_id')}`",
        f"- group_id: `{selected.get('group_id')}`",
        f"- architecture: `{selected.get('architecture')}`",
        f"- seed: `{selected.get('seed')}`",
        f"- seed_count: `{selected.get('seed_count')}`",
        f"- checkpoint: `{selected.get('checkpoint')}`",
        f"- saebench_summary_path: `{selected.get('saebench_summary_path')}`",
        f"- cebench_summary_path: `{selected.get('cebench_summary_path')}`",
        f"- saebench_delta: `{sel_metrics.get('saebench_delta')}`",
        f"- saebench_delta_ci95_low: `{sel_metrics.get('saebench_delta_ci95_low')}`",
        f"- saebench_delta_ci95_high: `{sel_metrics.get('saebench_delta_ci95_high')}`",
        f"- cebench_interp_delta_vs_baseline: `{sel_metrics.get('cebench_interp_delta_vs_baseline')}`",
        f"- cebench_interp_delta_vs_baseline_ci95_low: `{sel_metrics.get('cebench_interp_delta_vs_baseline_ci95_low')}`",
        f"- cebench_interp_delta_vs_baseline_ci95_high: `{sel_metrics.get('cebench_interp_delta_vs_baseline_ci95_high')}`",
        f"- cebench_interpretability_max: `{sel_metrics.get('cebench_interpretability_max')}`",
        f"- train_explained_variance: `{sel_metrics.get('train_explained_variance')}`",
        f"- joint_score: `{sel_score}`",
        "",
        "## Artifacts",
        "",
        f"- selection_summary.json: `{repo_rel(summary_json)}`",
        f"- selected_candidate.json: `{repo_rel(selected_json)}`",
        f"- pareto_front.json: `{repo_rel(pareto_json)}`",
        f"- candidate_table.json: `{repo_rel(table_json)}`",
    ]
    summary_md.write_text("\n".join(lines) + "\n")

    print("Release candidate selection complete")
    print(f"Run dir: {run_dir}")
    print(f"Selected candidate: {selected.get('condition_id')} ({selected.get('source')})")
    print(f"Selected JSON: {selected_json}")


if __name__ == "__main__":
    main()
