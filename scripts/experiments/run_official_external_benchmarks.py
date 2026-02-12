#!/usr/bin/env python3
"""Preflight and optional execution harness for official external benchmarks.

This script is intentionally conservative:
- It always writes a reproducibility manifest.
- It can export a local SAE checkpoint index for benchmark adapters.
- It runs official SAEBench/CE-Bench commands only when explicitly provided.

Rationale:
- This repository uses custom SAE checkpoints and small-model tasks.
- Official benchmark integration depends on external repos/packages and
  benchmark-specific command arguments that vary by setup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CommandResult:
    name: str
    attempted: bool
    success: bool
    returncode: int | None
    command: str | None
    cwd: str | None
    stdout_log: str | None
    stderr_log: str | None
    note: str


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def detect_saebench(repo_path: Path | None) -> dict[str, Any]:
    detected_repo = repo_path if repo_path and repo_path.exists() else None
    custom_runner = None
    if detected_repo:
        candidate = detected_repo / "sae_bench" / "custom_saes" / "run_all_evals_custom_saes.py"
        if candidate.exists():
            custom_runner = candidate

    return {
        "module_available": has_module("sae_bench"),
        "repo_path": str(detected_repo) if detected_repo else None,
        "custom_runner": str(custom_runner) if custom_runner else None,
        "recommended_refs": {
            "repo": "https://github.com/adamkarvonen/SAEBench",
            "custom_saes_readme": "sae_bench/custom_saes/README.md",
            "example_eval_cmd": (
                "python -m sae_bench.evals.sparse_probing.main "
                "--sae_regex_pattern <pattern> --sae_block_pattern <block> "
                "--model_name <model>"
            ),
        },
    }


def detect_cebench(repo_path: Path | None) -> dict[str, Any]:
    detected_repo = repo_path if repo_path and repo_path.exists() else None
    auto_script = None
    if detected_repo:
        candidate = detected_repo / "auto_script.sh"
        if candidate.exists():
            auto_script = candidate

    return {
        "module_available": has_module("ce_bench"),
        "repo_path": str(detected_repo) if detected_repo else None,
        "auto_script": str(auto_script) if auto_script else None,
        "recommended_refs": {
            "repo": "https://github.com/Yusen-Peng/CE-Bench",
            "paper": "https://arxiv.org/abs/2509.00691",
        },
    }


def collect_local_sae_index(sae_root: Path) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    if not sae_root.exists():
        return index

    checkpoint_paths: list[Path] = []
    checkpoint_paths.extend(sorted(sae_root.glob("topk_seed*/sae_final.pt")))
    checkpoint_paths.extend(sorted(sae_root.glob("topk_layer1_seed*/sae_final.pt")))

    for path in checkpoint_paths:
        rel = str(path.relative_to(PROJECT_ROOT))
        record: dict[str, Any] = {
            "checkpoint": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": {},
        }

        try:
            import torch

            obj = torch.load(path, map_location="cpu")
            state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else {}

            decoder = state.get("decoder.weight")
            if decoder is not None and hasattr(decoder, "shape"):
                record["metadata"]["decoder_shape"] = list(decoder.shape)
            if isinstance(obj, dict):
                for key in ("d_model", "d_sae", "k", "seed"):
                    if key in obj:
                        record["metadata"][key] = obj[key]
        except Exception as exc:
            record["metadata"]["load_error"] = str(exc)

        index.append(record)

    return index


def run_command(
    *,
    name: str,
    command: str | None,
    cwd: Path | None,
    logs_dir: Path,
    execute: bool,
) -> CommandResult:
    if not command:
        return CommandResult(
            name=name,
            attempted=False,
            success=False,
            returncode=None,
            command=None,
            cwd=str(cwd) if cwd else None,
            stdout_log=None,
            stderr_log=None,
            note="No command provided; preflight only.",
        )

    if not execute:
        return CommandResult(
            name=name,
            attempted=False,
            success=False,
            returncode=None,
            command=command,
            cwd=str(cwd) if cwd else None,
            stdout_log=None,
            stderr_log=None,
            note="Command provided but not executed (missing --execute).",
        )

    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )

    return CommandResult(
        name=name,
        attempted=True,
        success=proc.returncode == 0,
        returncode=proc.returncode,
        command=command,
        cwd=str(cwd) if cwd else None,
        stdout_log=str(stdout_path.relative_to(PROJECT_ROOT)),
        stderr_log=str(stderr_path.relative_to(PROJECT_ROOT)),
        note="completed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Official benchmark preflight/runner")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "experiments" / "phase4e_external_benchmark_official",
    )
    parser.add_argument(
        "--sae-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "saes",
    )
    parser.add_argument("--saebench-repo", type=Path, default=None)
    parser.add_argument("--cebench-repo", type=Path, default=None)
    parser.add_argument("--saebench-command", type=str, default="")
    parser.add_argument("--cebench-command", type=str, default="")
    parser.add_argument("--skip-saebench", action="store_true")
    parser.add_argument("--skip-cebench", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute provided commands. Without this flag, only preflight is written.",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "sae_root": str(args.sae_root),
        "saebench_repo": str(args.saebench_repo) if args.saebench_repo else None,
        "cebench_repo": str(args.cebench_repo) if args.cebench_repo else None,
        "saebench_command": args.saebench_command,
        "cebench_command": args.cebench_command,
        "skip_saebench": args.skip_saebench,
        "skip_cebench": args.skip_cebench,
        "execute": args.execute,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")

    saebench = detect_saebench(args.saebench_repo)
    cebench = detect_cebench(args.cebench_repo)
    local_sae_index = collect_local_sae_index(args.sae_root)

    command_results: list[CommandResult] = []

    if not args.skip_saebench:
        saebench_cwd = Path(saebench["repo_path"]) if saebench["repo_path"] else PROJECT_ROOT
        command_results.append(
            run_command(
                name="saebench",
                command=args.saebench_command.strip() or None,
                cwd=saebench_cwd,
                logs_dir=logs_dir,
                execute=args.execute,
            )
        )

    if not args.skip_cebench:
        cebench_cwd = Path(cebench["repo_path"]) if cebench["repo_path"] else PROJECT_ROOT
        command_results.append(
            run_command(
                name="cebench",
                command=args.cebench_command.strip() or None,
                cwd=cebench_cwd,
                logs_dir=logs_dir,
                execute=args.execute,
            )
        )

    preflight = {
        "timestamp_utc": utc_now(),
        "git_commit": git_commit(),
        "config_hash": stable_hash(config_payload),
        "saebench": saebench,
        "cebench": cebench,
        "local_sae_index_count": len(local_sae_index),
    }

    preflight_path = run_dir / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")

    sae_index_path = run_dir / "local_sae_index.json"
    sae_index_path.write_text(json.dumps(local_sae_index, indent=2) + "\n")

    command_payload = [asdict(r) for r in command_results]
    commands_path = run_dir / "commands.json"
    commands_path.write_text(json.dumps(command_payload, indent=2) + "\n")

    summary_lines = [
        "# Official External Benchmark Harness",
        "",
        f"- Run ID: `{run_id}`",
        f"- Git commit: `{preflight['git_commit']}`",
        f"- Config hash: `{preflight['config_hash']}`",
        "",
        "## Preflight",
        "",
        f"- SAEBench module available: `{saebench['module_available']}`",
        f"- SAEBench repo path: `{saebench['repo_path']}`",
        f"- CE-Bench module available: `{cebench['module_available']}`",
        f"- CE-Bench repo path: `{cebench['repo_path']}`",
        f"- Local SAE checkpoints indexed: `{len(local_sae_index)}`",
        "",
        "## Command Status",
    ]

    if not command_results:
        summary_lines.append("- No benchmark commands attempted.")
    else:
        summary_lines.append("")
        summary_lines.append("| name | attempted | success | returncode | note |")
        summary_lines.append("|---|---:|---:|---:|---|")
        for result in command_results:
            summary_lines.append(
                f"| {result.name} | {result.attempted} | {result.success} | "
                f"{result.returncode} | {result.note} |"
            )

    summary_lines.extend(
        [
            "",
            "## How to Execute",
            "",
            "Run with explicit official commands (examples):",
            "```bash",
            "python scripts/experiments/run_official_external_benchmarks.py \\",
            "  --saebench-repo /path/to/SAEBench \\",
            "  --cebench-repo /path/to/CE-Bench \\",
            "  --saebench-command \"<official SAEBench command>\" \\",
            "  --cebench-command \"<official CE-Bench command>\" \\",
            "  --execute",
            "```",
            "",
            "SAEBench reference command pattern from official docs:",
            "```bash",
            "python -m sae_bench.evals.sparse_probing.main \\",
            "  --sae_regex_pattern \"<pattern>\" \\",
            "  --sae_block_pattern \"<block>\" \\",
            "  --model_name <model>",
            "```",
        ]
    )

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    manifest = {
        "run_metadata": {
            "timestamp_utc": preflight["timestamp_utc"],
            "git_commit": preflight["git_commit"],
            "command": " ".join(["python", *map(shlex.quote, sys.argv)]),
            "config_hash": preflight["config_hash"],
            "run_id": run_id,
        },
        "artifacts": [
            str((run_dir / "config.json").relative_to(PROJECT_ROOT)),
            str(preflight_path.relative_to(PROJECT_ROOT)),
            str(sae_index_path.relative_to(PROJECT_ROOT)),
            str(commands_path.relative_to(PROJECT_ROOT)),
            str(summary_path.relative_to(PROJECT_ROOT)),
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Official benchmark harness complete")
    print(f"Run directory: {run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
