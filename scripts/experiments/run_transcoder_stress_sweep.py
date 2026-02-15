#!/usr/bin/env python3
"""Run a hyperparameter sweep for transcoder stress evaluation.

This wraps `run_transcoder_stress_eval.py` over a small grid and selects the
best condition by conservative delta lower bound:
    transcoder_delta_lcb = transcoder_ci95_low - sae_ci95_high
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_ints(text: str) -> list[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return [int(v) for v in values]


def parse_floats(text: str) -> list[float]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return [float(v) for v in values]


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def repo_rel(path: Path) -> str:
    abs_path = path.resolve()
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def cmd_to_str(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def run_cmd(command: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return int(proc.returncode), proc.stdout


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def extract_delta(summary: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if summary is None:
        return None, None

    delta = summary.get("transcoder_delta")
    if not isinstance(delta, (int, float)):
        delta = summary.get("delta")
    if not isinstance(delta, (int, float)):
        delta = (summary.get("metrics") or {}).get("transcoder_delta")

    transcoder_ci_low = (((summary.get("transcoder") or {}).get("summary") or {}).get("ci95_low"))
    sae_ci_high = (((summary.get("sae") or {}).get("summary") or {}).get("ci95_high"))
    if isinstance(transcoder_ci_low, (int, float)) and isinstance(sae_ci_high, (int, float)):
        delta_lcb = float(transcoder_ci_low) - float(sae_ci_high)
    else:
        delta_lcb = None

    return (float(delta) if isinstance(delta, (int, float)) else None, delta_lcb)


def latest_run_dir(root: Path) -> Path | None:
    runs = sorted(root.glob("run_*"))
    if not runs:
        return None
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyper-sweep for transcoder stress evaluation")
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "transformer_5000ep" / "transformer_best.pt",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--position", type=str, default="answer", choices=["answer", "bos", "first_operand"])
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--d-sae-values", type=str, default="128,256")
    parser.add_argument("--k-values", type=str, default="16,32")
    parser.add_argument("--epochs-values", type=str, default="12,20")
    parser.add_argument("--learning-rate-values", type=str, default="0.0003,0.001")

    parser.add_argument("--min-delta-lcb", type=float, default=0.0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_transcoder_stress_sweep",
    )
    args = parser.parse_args()

    args.transformer_checkpoint = to_abs(args.transformer_checkpoint)
    args.output_dir = to_abs(args.output_dir)

    if not args.transformer_checkpoint.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {args.transformer_checkpoint}")

    d_sae_values = parse_ints(args.d_sae_values)
    k_values = parse_ints(args.k_values)
    epochs_values = parse_ints(args.epochs_values)
    lr_values = parse_floats(args.learning_rate_values)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    logs_dir = run_dir / "logs"
    cond_root = run_dir / "conditions"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cond_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for d_sae in d_sae_values:
        for k in k_values:
            for epochs in epochs_values:
                for lr in lr_values:
                    condition_id = f"dsae{d_sae}_k{k}_ep{epochs}_lr{lr:g}"
                    condition_dir = cond_root / condition_id
                    condition_dir.mkdir(parents=True, exist_ok=True)

                    command = [
                        sys.executable,
                        "scripts/experiments/run_transcoder_stress_eval.py",
                        "--transformer-checkpoint",
                        str(args.transformer_checkpoint),
                        "--layer",
                        str(args.layer),
                        "--position",
                        args.position,
                        "--modulus",
                        str(args.modulus),
                        "--batch-size",
                        str(args.batch_size),
                        "--max-samples",
                        str(args.max_samples),
                        "--d-sae",
                        str(d_sae),
                        "--k",
                        str(k),
                        "--seeds",
                        args.seeds,
                        "--epochs",
                        str(epochs),
                        "--learning-rate",
                        str(lr),
                        "--bootstrap-samples",
                        str(args.bootstrap_samples),
                        "--device",
                        args.device,
                        "--output-dir",
                        str(condition_dir),
                    ]

                    rc, output = run_cmd(command, PROJECT_ROOT)
                    (logs_dir / f"{condition_id}.log").write_text(output)

                    latest = latest_run_dir(condition_dir)
                    summary_path = latest / "transcoder_stress_summary.json" if latest else None
                    summary = load_json(summary_path)
                    delta, delta_lcb = extract_delta(summary)

                    rec = {
                        "condition_id": condition_id,
                        "d_sae": d_sae,
                        "k": k,
                        "epochs": epochs,
                        "learning_rate": lr,
                        "returncode": rc,
                        "command": cmd_to_str(command),
                        "summary_path": repo_rel(summary_path) if summary_path is not None else None,
                        "delta": delta,
                        "delta_lcb": delta_lcb,
                    }
                    records.append(rec)

    valid = [r for r in records if r.get("returncode") == 0 and isinstance(r.get("delta_lcb"), (int, float))]
    if valid:
        best = max(valid, key=lambda r: float(r["delta_lcb"]))
    else:
        best = None

    payload = {
        "run_metadata": {
            "timestamp_utc": utc_now(),
            "command": " ".join(["python", *sys.argv]),
            "run_id": run_id,
        },
        "config": {
            "transformer_checkpoint": str(args.transformer_checkpoint),
            "layer": args.layer,
            "position": args.position,
            "modulus": args.modulus,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "seeds": args.seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "device": args.device,
            "d_sae_values": d_sae_values,
            "k_values": k_values,
            "epochs_values": epochs_values,
            "learning_rate_values": lr_values,
            "min_delta_lcb": args.min_delta_lcb,
        },
        "records": records,
        "best_condition": best,
        "gates": {
            "has_valid_condition": best is not None,
            "gate_delta_lcb": bool(best is not None and float(best["delta_lcb"]) >= float(args.min_delta_lcb)),
        },
    }
    payload["gates"]["pass_all"] = bool(payload["gates"]["has_valid_condition"] and payload["gates"]["gate_delta_lcb"])

    out_json = run_dir / "results.json"
    out_md = run_dir / "summary.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Transcoder Stress Sweep",
        "",
        f"- Run ID: `{run_id}`",
        f"- Total conditions: `{len(records)}`",
        f"- Valid conditions: `{len(valid)}`",
        f"- min_delta_lcb: `{args.min_delta_lcb}`",
        f"- pass_all: `{payload['gates']['pass_all']}`",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Condition",
                "",
                f"- condition_id: `{best['condition_id']}`",
                f"- delta: `{best['delta']}`",
                f"- delta_lcb: `{best['delta_lcb']}`",
                f"- summary_path: `{best['summary_path']}`",
                "",
            ]
        )

    lines.extend(
        [
            "| condition | delta | delta_lcb | rc |",
            "|---|---:|---:|---:|",
        ]
    )
    for rec in sorted(records, key=lambda r: (r.get("delta_lcb") is None, -(r.get("delta_lcb") or -1e9))):
        lines.append(
            f"| {rec['condition_id']} | {rec.get('delta')} | {rec.get('delta_lcb')} | {rec.get('returncode')} |"
        )

    out_md.write_text("\n".join(lines) + "\n")

    print("Transcoder stress sweep complete")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {out_md}")


if __name__ == "__main__":
    main()
