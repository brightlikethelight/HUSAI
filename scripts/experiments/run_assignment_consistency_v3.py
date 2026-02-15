#!/usr/bin/env python3
"""Assignment-aware consistency objective v3 with external-aware selection.

v3 extends v2 by adding an optional external-evaluation stage for each lambda and
selecting checkpoints via a multi-objective Pareto + weighted score:
- internal consistency LCB (higher is better)
- EV drop vs baseline (lower is better)
- SAEBench delta (higher is better)
- CE-Bench delta vs matched baseline (higher is better)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_assignment_consistency_v2 import (  # noqa: E402
    load_or_extract_activations,
    parse_float_list,
    parse_int_list,
    run_lambda_condition,
)
from scripts.experiments.train_husai_sae_on_cached_activations import load_activation_bank  # noqa: E402


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


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def maybe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(proc.returncode), proc.stdout


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def checkpoint_d_model(checkpoint: Path) -> int | None:
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, dict):
        if isinstance(obj.get("d_model"), int):
            return int(obj["d_model"])
        state = obj.get("model_state_dict") or obj
    else:
        state = obj

    if isinstance(state, dict):
        if "decoder.weight" in state:
            dec = state["decoder.weight"]
            if isinstance(dec, torch.Tensor) and dec.ndim == 2:
                return int(dec.shape[0])
        if "W_dec" in state:
            dec = state["W_dec"]
            if isinstance(dec, torch.Tensor) and dec.ndim == 2:
                return int(dec.shape[1])
    return None


def infer_external_d_model_from_cache(model_cache_path: Path, model_name: str, hook_name: str) -> int | None:
    cache_dir = model_cache_path / f"model_activations_{model_name}"
    if not cache_dir.exists():
        return None
    files = sorted(cache_dir.glob(f"*_{hook_name}.pt"))
    if not files:
        return None
    acts = torch.load(files[0], map_location="cpu")
    if isinstance(acts, torch.Tensor) and acts.ndim == 2:
        return int(acts.shape[1])
    return None


def normalize(values: list[float | None]) -> dict[int, float]:
    clean = [(i, v) for i, v in enumerate(values) if v is not None]
    if not clean:
        return {}
    only_vals = [float(v) for _, v in clean]
    lo = min(only_vals)
    hi = max(only_vals)
    if hi <= lo:
        return {i: 0.5 for i, _ in clean}
    return {i: (float(v) - lo) / (hi - lo) for i, v in clean}


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    dims = [
        "internal_lcb",
        "ev_neg_drop",
        "saebench_delta",
        "cebench_delta",
    ]

    ge_all = True
    gt_any = False
    for dim in dims:
        av = maybe_float((a.get("selection_metrics") or {}).get(dim))
        bv = maybe_float((b.get("selection_metrics") or {}).get(dim))
        if av is None or bv is None:
            continue
        if av < bv:
            ge_all = False
            break
        if av > bv:
            gt_any = True

    return ge_all and gt_any


def summarize_external(saebench_summary: dict[str, Any] | None, cebench_summary: dict[str, Any] | None) -> dict[str, float | None]:
    saebench_delta = maybe_float((saebench_summary or {}).get("summary", {}).get("best_minus_llm_auc"))
    cebench_delta = maybe_float(
        (cebench_summary or {}).get("delta_vs_matched_baseline", {}).get("interpretability_score_mean_max")
    )
    cebench_interp = maybe_float(
        (cebench_summary or {}).get("custom_metrics", {}).get("interpretability_score_mean_max")
    )
    return {
        "saebench_delta": saebench_delta,
        "cebench_delta": cebench_delta,
        "cebench_interpretability_max": cebench_interp,
    }


def load_training_activations(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load activations from either modular cache or external activation cache."""
    if args.activation_cache_dir is not None:
        bank, files_used, data_meta = load_activation_bank(
            cache_dir=args.activation_cache_dir,
            activation_glob=args.activation_glob,
            max_files=args.max_files,
            max_rows_per_file=args.max_rows_per_file,
            max_total_rows=args.max_total_rows,
            seed=args.source_cache_seed if args.source_cache_seed is not None else args.seed_ref,
        )
        return bank.float(), {
            "source": "external_cache",
            "activation_cache_dir": repo_rel(args.activation_cache_dir),
            "activation_glob": args.activation_glob,
            "max_files": args.max_files,
            "max_rows_per_file": args.max_rows_per_file,
            "max_total_rows": args.max_total_rows,
            "source_cache_seed": args.source_cache_seed if args.source_cache_seed is not None else args.seed_ref,
            "data_meta": data_meta,
            "source_files": [repo_rel(Path(p)) for p in files_used],
        }

    acts = load_or_extract_activations(
        cache_path=args.activation_cache,
        transformer_checkpoint=args.transformer_checkpoint,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        modulus=args.modulus,
        seed=args.seed_ref,
    )
    return acts.float(), {
        "source": "modular_assignment",
        "activation_cache": repo_rel(args.activation_cache),
        "transformer_checkpoint": repo_rel(args.transformer_checkpoint),
        "layer": args.layer,
        "modulus": args.modulus,
    }


def pick_external_checkpoint(rec: dict[str, Any]) -> str | None:
    per_seed = rec.get("per_seed_metrics") or []
    if not per_seed:
        return None

    seed_ref = rec.get("seed_ref")
    non_ref = [m for m in per_seed if m.get("seed") != seed_ref]
    pool = non_ref if non_ref else per_seed

    best = max(
        pool,
        key=lambda m: (
            maybe_float(m.get("alignment_to_ref")) or float("-inf"),
            maybe_float(m.get("explained_variance")) or float("-inf"),
        ),
    )
    checkpoint = best.get("checkpoint")
    return str(checkpoint) if isinstance(checkpoint, str) else None


def evaluate_external_for_checkpoint(
    *,
    checkpoint: Path,
    lambda_id: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    ext_dir = run_dir / "external_eval" / lambda_id
    ext_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "checkpoint": repo_rel(checkpoint),
        "saebench": None,
        "cebench": None,
        "saebench_returncode": None,
        "cebench_returncode": None,
        "saebench_summary_path": None,
        "cebench_summary_path": None,
        "external_skip_reason": None,
    }

    ckpt_d_model = checkpoint_d_model(checkpoint)
    external_d_model = infer_external_d_model_from_cache(
        model_cache_path=args.saebench_model_cache_path,
        model_name=args.model_name,
        hook_name=args.hook_name,
    )
    if (
        (args.run_saebench or args.run_cebench)
        and ckpt_d_model is not None
        and external_d_model is not None
        and ckpt_d_model != external_d_model
    ):
        out["external_skip_reason"] = (
            f"d_model_mismatch checkpoint={ckpt_d_model} external_cache={external_d_model}"
        )
        return out

    if args.run_saebench:
        saebench_out = ext_dir / "saebench"
        command = [
            sys.executable,
            "scripts/experiments/run_husai_saebench_custom_eval.py",
            "--checkpoint",
            str(checkpoint),
            "--architecture",
            "topk",
            "--sae-release",
            f"husai_assignv3_{run_dir.name}_lambda{lambda_id}",
            "--model-name",
            args.model_name,
            "--hook-layer",
            str(args.hook_layer),
            "--hook-name",
            args.hook_name,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--results-path",
            str(args.saebench_results_path),
            "--model-cache-path",
            str(args.saebench_model_cache_path),
            "--output-dir",
            str(saebench_out),
        ]
        if args.saebench_datasets:
            command.extend(["--dataset-names", args.saebench_datasets])
        if args.force_rerun_external:
            command.append("--force-rerun")

        rc, text = run_subprocess(command, PROJECT_ROOT)
        (logs_dir / f"lambda_{lambda_id}_saebench.log").write_text(text)
        out["saebench_returncode"] = rc
        summary_path = saebench_out / "husai_custom_sae_summary.json"
        out["saebench_summary_path"] = repo_rel(summary_path) if summary_path.exists() else None
        out["saebench"] = load_json(summary_path)

    if args.run_cebench:
        cebench_out = ext_dir / "cebench"
        command = [
            sys.executable,
            "scripts/experiments/run_husai_cebench_custom_eval.py",
            "--cebench-repo",
            str(args.cebench_repo),
            "--checkpoint",
            str(checkpoint),
            "--architecture",
            "topk",
            "--sae-release",
            f"husai_assignv3_{run_dir.name}_lambda{lambda_id}",
            "--model-name",
            args.model_name,
            "--hook-layer",
            str(args.hook_layer),
            "--hook-name",
            args.hook_name,
            "--device",
            args.device,
            "--sae-dtype",
            args.dtype,
            "--output-folder",
            str(cebench_out),
            "--artifacts-path",
            str(args.cebench_artifacts_path),
        ]
        if args.cebench_max_rows is not None:
            command.extend(["--max-rows", str(args.cebench_max_rows)])
        if args.cebench_matched_baseline_summary is not None:
            command.extend(["--matched-baseline-summary", str(args.cebench_matched_baseline_summary)])

        rc, text = run_subprocess(command, PROJECT_ROOT)
        (logs_dir / f"lambda_{lambda_id}_cebench.log").write_text(text)
        out["cebench_returncode"] = rc
        summary_path = cebench_out / "husai_custom_cebench_summary.json"
        out["cebench_summary_path"] = repo_rel(summary_path) if summary_path.exists() else None
        out["cebench"] = load_json(summary_path)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment-aware consistency objective v3")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "assignment_consistency_v3" / "layer1_answer_acts.pt",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--modulus", type=int, default=113)

    parser.add_argument("--seed-ref", type=int, default=42)
    parser.add_argument("--train-seeds", type=str, default="123,456,789,1011")
    parser.add_argument("--lambdas", type=str, default="0.0,0.05,0.1,0.2,0.3")

    parser.add_argument(
        "--activation-cache-dir",
        type=Path,
        default=None,
        help="Optional external activation cache directory. If set, overrides modular extraction path.",
    )
    parser.add_argument(
        "--activation-glob",
        type=str,
        default="*_blocks.0.hook_resid_pre.pt",
        help="Glob pattern within --activation-cache-dir when external cache mode is enabled.",
    )
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--max-rows-per-file", type=int, default=2048)
    parser.add_argument("--max-total-rows", type=int, default=150000)
    parser.add_argument(
        "--source-cache-seed",
        type=int,
        default=None,
        help="Row-sampling seed for external cache loading (defaults to --seed-ref).",
    )

    parser.add_argument("--d-sae", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)

    parser.add_argument("--run-saebench", action="store_true")
    parser.add_argument("--run-cebench", action="store_true")
    parser.add_argument("--cebench-repo", type=Path, default=None)
    parser.add_argument("--cebench-max-rows", type=int, default=200)
    parser.add_argument("--cebench-matched-baseline-summary", type=Path, default=None)
    parser.add_argument("--saebench-datasets", type=str, default="")
    parser.add_argument("--model-name", type=str, default="pythia-70m-deduped")
    parser.add_argument("--hook-layer", type=int, default=0)
    parser.add_argument("--hook-name", type=str, default="blocks.0.hook_resid_pre")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--force-rerun-external", action="store_true")

    parser.add_argument(
        "--saebench-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "husai_saebench_probe_results_assignv3",
    )
    parser.add_argument(
        "--saebench-model-cache-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "sae_bench_model_cache",
    )
    parser.add_argument(
        "--cebench-artifacts-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cache" / "external_benchmarks" / "ce_bench_artifacts_assignv3",
    )

    parser.add_argument("--weight-internal-lcb", type=float, default=0.45)
    parser.add_argument("--weight-ev", type=float, default=0.15)
    parser.add_argument("--weight-saebench", type=float, default=0.2)
    parser.add_argument("--weight-cebench", type=float, default=0.2)

    parser.add_argument("--min-internal-lcb", type=float, default=0.0)
    parser.add_argument("--max-ev-drop", type=float, default=0.05)
    parser.add_argument("--min-saebench-delta", type=float, default=0.0)
    parser.add_argument("--min-cebench-delta", type=float, default=0.0)
    parser.add_argument("--require-external", action="store_true")
    parser.add_argument("--fail-on-acceptance-fail", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4d_assignment_consistency_v3",
    )
    args = parser.parse_args()

    args.transformer_checkpoint = to_abs(args.transformer_checkpoint)
    args.activation_cache = to_abs(args.activation_cache)
    args.output_dir = to_abs(args.output_dir)
    args.saebench_results_path = to_abs(args.saebench_results_path)
    args.saebench_model_cache_path = to_abs(args.saebench_model_cache_path)
    args.cebench_artifacts_path = to_abs(args.cebench_artifacts_path)
    if args.activation_cache_dir is not None:
        args.activation_cache_dir = to_abs(args.activation_cache_dir)
    if args.cebench_repo is not None:
        args.cebench_repo = to_abs(args.cebench_repo)
    if args.cebench_matched_baseline_summary is not None:
        args.cebench_matched_baseline_summary = to_abs(args.cebench_matched_baseline_summary)

    if args.run_cebench and args.cebench_repo is None:
        raise ValueError("--cebench-repo is required when --run-cebench is set")

    train_seeds = parse_int_list(args.train_seeds)
    lambdas = parse_float_list(args.lambdas)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    checkpoints_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    acts, activation_source = load_training_activations(args)

    records: list[dict[str, Any]] = []
    for lam in lambdas:
        ckpt_dir = checkpoints_root / f"lambda_{lam}"
        rec = run_lambda_condition(
            lambda_consistency=lam,
            activations=acts,
            d_sae=args.d_sae,
            k=args.k,
            seed_ref=args.seed_ref,
            train_seeds=train_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            bootstrap_samples=args.bootstrap_samples,
            checkpoint_dir=ckpt_dir,
        )
        records.append(rec)

    records = sorted(records, key=lambda r: r["lambda_consistency"])

    baseline = next((r for r in records if abs(r["lambda_consistency"]) < 1e-12), None)
    baseline_ev = baseline["explained_variance"]["mean"] if baseline else max(r["explained_variance"]["mean"] for r in records)

    for rec in records:
        checkpoint_rel = pick_external_checkpoint(rec)
        external_payload = None
        if checkpoint_rel is not None and (args.run_saebench or args.run_cebench):
            external_payload = evaluate_external_for_checkpoint(
                checkpoint=to_abs(Path(checkpoint_rel)),
                lambda_id=str(rec["lambda_consistency"]),
                args=args,
                run_dir=run_dir,
            )

        ev_drop = baseline_ev - rec["explained_variance"]["mean"]
        metrics = summarize_external(
            (external_payload or {}).get("saebench"),
            (external_payload or {}).get("cebench"),
        )

        rec["selected_checkpoint"] = checkpoint_rel
        rec["external_eval"] = external_payload
        rec["selection_metrics"] = {
            "internal_lcb": rec["delta_pwmcc_ci_low_conservative"],
            "ev_drop": ev_drop,
            "ev_neg_drop": -ev_drop,
            "saebench_delta": metrics["saebench_delta"],
            "cebench_delta": metrics["cebench_delta"],
            "cebench_interpretability_max": metrics["cebench_interpretability_max"],
        }

    internal_vals = [maybe_float((r.get("selection_metrics") or {}).get("internal_lcb")) for r in records]
    ev_vals = [maybe_float((r.get("selection_metrics") or {}).get("ev_neg_drop")) for r in records]
    sae_vals = [maybe_float((r.get("selection_metrics") or {}).get("saebench_delta")) for r in records]
    ce_vals = [maybe_float((r.get("selection_metrics") or {}).get("cebench_delta")) for r in records]

    internal_norm = normalize(internal_vals)
    ev_norm = normalize(ev_vals)
    sae_norm = normalize(sae_vals)
    ce_norm = normalize(ce_vals)

    for i, rec in enumerate(records):
        score = 0.0
        if i in internal_norm:
            score += args.weight_internal_lcb * internal_norm[i]
        if i in ev_norm:
            score += args.weight_ev * ev_norm[i]
        if i in sae_norm:
            score += args.weight_saebench * sae_norm[i]
        if i in ce_norm:
            score += args.weight_cebench * ce_norm[i]

        rec["selection"] = {
            "joint_score": score,
            "is_pareto": False,
            "weights": {
                "internal_lcb": args.weight_internal_lcb,
                "ev_neg_drop": args.weight_ev,
                "saebench_delta": args.weight_saebench,
                "cebench_delta": args.weight_cebench,
            },
        }

    for i, rec in enumerate(records):
        dominated = any(dominates(other, rec) for j, other in enumerate(records) if j != i)
        rec["selection"]["is_pareto"] = not dominated

    ranked = sorted(
        records,
        key=lambda r: (
            bool((r.get("selection") or {}).get("is_pareto", False)),
            float((r.get("selection") or {}).get("joint_score", float("-inf"))),
            maybe_float((r.get("selection_metrics") or {}).get("internal_lcb")) or float("-inf"),
            maybe_float((r.get("selection_metrics") or {}).get("saebench_delta")) or float("-inf"),
            maybe_float((r.get("selection_metrics") or {}).get("cebench_delta")) or float("-inf"),
        ),
        reverse=True,
    )

    best = ranked[0]
    best_metrics = best.get("selection_metrics") or {}

    gate_internal = bool((maybe_float(best_metrics.get("internal_lcb")) or float("-inf")) >= args.min_internal_lcb)
    gate_ev = bool((maybe_float(best_metrics.get("ev_drop")) or float("inf")) <= args.max_ev_drop)

    sae_delta = maybe_float(best_metrics.get("saebench_delta"))
    ce_delta = maybe_float(best_metrics.get("cebench_delta"))
    gate_sae = sae_delta is not None and sae_delta >= args.min_saebench_delta
    gate_ce = ce_delta is not None and ce_delta >= args.min_cebench_delta
    if not args.require_external:
        gate_sae = gate_sae or sae_delta is None
        gate_ce = gate_ce or ce_delta is None

    acceptance = {
        "best_lambda": best["lambda_consistency"],
        "best_checkpoint": best.get("selected_checkpoint"),
        "gate_internal_lcb": gate_internal,
        "gate_ev_drop": gate_ev,
        "gate_saebench": gate_sae,
        "gate_cebench": gate_ce,
        "min_internal_lcb": args.min_internal_lcb,
        "max_ev_drop": args.max_ev_drop,
        "min_saebench_delta": args.min_saebench_delta,
        "min_cebench_delta": args.min_cebench_delta,
        "require_external": args.require_external,
    }
    acceptance["pass_all"] = bool(gate_internal and gate_ev and gate_sae and gate_ce)

    config_payload = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "activation_cache": str(args.activation_cache),
        "activation_source": activation_source,
        "layer": args.layer,
        "modulus": args.modulus,
        "seed_ref": args.seed_ref,
        "train_seeds": train_seeds,
        "lambdas": lambdas,
        "d_sae": args.d_sae,
        "k": args.k,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "run_saebench": args.run_saebench,
        "run_cebench": args.run_cebench,
        "cebench_repo": str(args.cebench_repo) if args.cebench_repo is not None else None,
        "cebench_max_rows": args.cebench_max_rows,
        "cebench_matched_baseline_summary": (
            str(args.cebench_matched_baseline_summary) if args.cebench_matched_baseline_summary is not None else None
        ),
        "saebench_datasets": parse_csv_strings(args.saebench_datasets),
        "model_name": args.model_name,
        "hook_layer": args.hook_layer,
        "hook_name": args.hook_name,
        "dtype": args.dtype,
        "weights": {
            "internal_lcb": args.weight_internal_lcb,
            "ev_neg_drop": args.weight_ev,
            "saebench_delta": args.weight_saebench,
            "cebench_delta": args.weight_cebench,
        },
    }

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "git_commit": git_commit(),
            "command": " ".join(["python", *sys.argv]),
            "config_hash": stable_hash(config_payload),
            "run_id": run_id,
        },
        "config": config_payload,
        "records": records,
        "pareto_front": [r for r in records if (r.get("selection") or {}).get("is_pareto")],
        "ranked": ranked,
        "selected": best,
        "acceptance": acceptance,
    }

    out_json = run_dir / "results.json"
    out_md = run_dir / "summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Assignment-Aware Consistency v3",
        "",
        f"- Run ID: `{run_id}`",
        f"- Best lambda: `{best['lambda_consistency']}`",
        f"- Best checkpoint: `{best.get('selected_checkpoint')}`",
        f"- pass_all: `{acceptance['pass_all']}`",
        "",
        "## Acceptance",
        "",
        f"- gate_internal_lcb: `{acceptance['gate_internal_lcb']}`",
        f"- gate_ev_drop: `{acceptance['gate_ev_drop']}`",
        f"- gate_saebench: `{acceptance['gate_saebench']}`",
        f"- gate_cebench: `{acceptance['gate_cebench']}`",
        "",
        "| lambda | internal_lcb | ev_drop | saebench_delta | cebench_delta | joint_score | pareto |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rec in ranked:
        m = rec.get("selection_metrics") or {}
        sel = rec.get("selection") or {}
        lines.append(
            "| "
            f"{rec['lambda_consistency']:.4f} | "
            f"{m.get('internal_lcb')} | "
            f"{m.get('ev_drop')} | "
            f"{m.get('saebench_delta')} | "
            f"{m.get('cebench_delta')} | "
            f"{sel.get('joint_score')} | "
            f"{sel.get('is_pareto')} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print("Assignment consistency v3 complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")

    if args.fail_on_acceptance_fail and not acceptance["pass_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
