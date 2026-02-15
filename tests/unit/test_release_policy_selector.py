from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts" / "experiments" / "select_release_candidate.py"
RELEASE_GATE = ROOT / "scripts" / "experiments" / "run_stress_gated_release_policy.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_selector_prefers_joint_candidate(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_a = run_root / "frontier" / "run_aaa"
    run_b = run_root / "scaling" / "run_bbb"

    ckpt_a = run_a / "checkpoints" / "topk_seed42" / "sae_final.pt"
    ckpt_b = run_b / "checkpoints" / "tok10000_layer0_dsae1024_seed42" / "sae_final.pt"
    ckpt_a.parent.mkdir(parents=True, exist_ok=True)
    ckpt_b.parent.mkdir(parents=True, exist_ok=True)
    ckpt_a.write_text("x")
    ckpt_b.write_text("x")

    frontier_payload = {
        "records": [
            {
                "architecture": "topk",
                "seed": 42,
                "checkpoint": str(ckpt_a),
                "train_metrics": {"explained_variance": 0.9},
                "saebench": {"summary": {"best_minus_llm_auc": -0.04}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": -10.0},
                    "custom_metrics": {"interpretability_score_mean_max": 12.0},
                    "config": {"hook_layer": 0, "hook_name": "blocks.0.hook_resid_pre"},
                    "sae_meta": {"architecture": "topk"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            }
        ]
    }
    scaling_payload = {
        "records": [
            {
                "condition_id": "tok10000_layer0_dsae1024_seed42",
                "seed": 42,
                "hook_layer": 0,
                "hook_name": "blocks.0.hook_resid_pre",
                "checkpoint": str(ckpt_b),
                "train_summary": {"final_metrics": {"explained_variance": 0.8}},
                "saebench": {"summary": {"best_minus_llm_auc": -0.01}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": -2.0},
                    "custom_metrics": {"interpretability_score_mean_max": 20.0},
                    "config": {"hook_layer": 0, "hook_name": "blocks.0.hook_resid_pre"},
                    "sae_meta": {"architecture": "topk"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            }
        ]
    }

    frontier_json = tmp_path / "frontier_results.json"
    scaling_json = tmp_path / "scaling_results.json"
    _write_json(frontier_json, frontier_payload)
    _write_json(scaling_json, scaling_payload)

    out_dir = tmp_path / "selector_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--frontier-results",
            str(frontier_json),
            "--scaling-results",
            str(scaling_json),
            "--require-both-external",
            "--seed-level-selection",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs
    selected = json.loads((run_dirs[-1] / "selected_candidate.json").read_text())

    # Scaling candidate has much better CE-Bench delta and should win joint score.
    assert selected["condition_id"] == "tok10000_layer0_dsae1024_seed42"


def test_selector_grouped_lcb_prefers_stable_group(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_a = run_root / "frontier" / "run_aaa"

    ckpts = {}
    for arch in ("topk", "relu"):
        for seed in (42, 123):
            ckpt = run_a / "checkpoints" / f"{arch}_seed{seed}" / "sae_final.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text("x")
            ckpts[(arch, seed)] = ckpt

    payload = {
        "records": [
            {
                "architecture": "topk",
                "seed": 42,
                "checkpoint": str(ckpts[("topk", 42)]),
                "train_metrics": {"explained_variance": 0.85},
                "saebench": {"summary": {"best_minus_llm_auc": -0.005}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": 0.30},
                    "custom_metrics": {"interpretability_score_mean_max": 7.5},
                    "sae_meta": {"architecture": "topk"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            },
            {
                "architecture": "topk",
                "seed": 123,
                "checkpoint": str(ckpts[("topk", 123)]),
                "train_metrics": {"explained_variance": 0.85},
                "saebench": {"summary": {"best_minus_llm_auc": -0.095}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": -0.10},
                    "custom_metrics": {"interpretability_score_mean_max": 7.2},
                    "sae_meta": {"architecture": "topk"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            },
            {
                "architecture": "relu",
                "seed": 42,
                "checkpoint": str(ckpts[("relu", 42)]),
                "train_metrics": {"explained_variance": 0.82},
                "saebench": {"summary": {"best_minus_llm_auc": -0.040}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": 0.08},
                    "custom_metrics": {"interpretability_score_mean_max": 6.9},
                    "sae_meta": {"architecture": "relu"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            },
            {
                "architecture": "relu",
                "seed": 123,
                "checkpoint": str(ckpts[("relu", 123)]),
                "train_metrics": {"explained_variance": 0.82},
                "saebench": {"summary": {"best_minus_llm_auc": -0.042}},
                "cebench": {
                    "delta_vs_matched_baseline": {"interpretability_score_mean_max": 0.09},
                    "custom_metrics": {"interpretability_score_mean_max": 6.8},
                    "sae_meta": {"architecture": "relu"},
                },
                "saebench_returncode": 0,
                "cebench_returncode": 0,
            },
        ]
    }

    frontier_json = tmp_path / "frontier_results.json"
    _write_json(frontier_json, payload)

    out_dir = tmp_path / "selector_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--frontier-results",
            str(frontier_json),
            "--require-both-external",
            "--group-by-condition",
            "--uncertainty-mode",
            "lcb",
            "--min-seeds-per-group",
            "2",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs
    selected = json.loads((run_dirs[-1] / "selected_candidate.json").read_text())

    # Grouped-LCB mode should prefer the lower-variance relu group.
    assert selected["group_id"] == "relu"
    assert selected["metrics"]["saebench_delta_ci95_low"] is not None
    assert selected["metrics"]["cebench_interp_delta_vs_baseline_ci95_low"] is not None


def test_release_gate_joint_mode_with_candidate_json(tmp_path: Path) -> None:
    phase4a = {
        "comparison": {"difference": 0.01},
        "trained": {"ci95_low": 0.2},
        "random": {"ci95_high": 0.1},
    }
    transcoder = {"transcoder_delta": 0.02}
    ood = {"ood_drop": 0.01}
    candidate = {
        "metrics": {
            "saebench_delta": 0.005,
            "cebench_interp_delta_vs_baseline": -0.5,
            "cebench_interpretability_max": 9.0,
        }
    }

    phase4a_json = tmp_path / "phase4a.json"
    transcoder_json = tmp_path / "transcoder.json"
    ood_json = tmp_path / "ood.json"
    candidate_json = tmp_path / "candidate.json"
    _write_json(phase4a_json, phase4a)
    _write_json(transcoder_json, transcoder)
    _write_json(ood_json, ood)
    _write_json(candidate_json, candidate)

    out_dir = tmp_path / "gate_out"

    # Should fail in strict mode because CE-Bench threshold is unmet.
    proc_fail = subprocess.run(
        [
            sys.executable,
            str(RELEASE_GATE),
            "--phase4a-results",
            str(phase4a_json),
            "--transcoder-results",
            str(transcoder_json),
            "--ood-results",
            str(ood_json),
            "--external-candidate-json",
            str(candidate_json),
            "--external-mode",
            "joint",
            "--min-saebench-delta",
            "0.0",
            "--min-cebench-delta",
            "0.0",
            "--require-transcoder",
            "--require-ood",
            "--require-external",
            "--fail-on-gate-fail",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc_fail.returncode == 2, proc_fail.stdout + "\n" + proc_fail.stderr

    # Relax CE-Bench threshold to pass.
    proc_pass = subprocess.run(
        [
            sys.executable,
            str(RELEASE_GATE),
            "--phase4a-results",
            str(phase4a_json),
            "--transcoder-results",
            str(transcoder_json),
            "--ood-results",
            str(ood_json),
            "--external-candidate-json",
            str(candidate_json),
            "--external-mode",
            "joint",
            "--min-saebench-delta",
            "0.0",
            "--min-cebench-delta",
            "-1.0",
            "--require-transcoder",
            "--require-ood",
            "--require-external",
            "--fail-on-gate-fail",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc_pass.returncode == 0, proc_pass.stdout + "\n" + proc_pass.stderr


def test_release_gate_lcb_mode_uses_ci_fields(tmp_path: Path) -> None:
    phase4a = {
        "comparison": {"difference": 0.02},
        "trained": {"ci95_low": 0.2},
        "random": {"ci95_high": 0.1},
    }
    transcoder = {"transcoder_delta": 0.02}
    ood = {"ood_drop": 0.01}
    candidate = {
        "metrics": {
            "saebench_delta": 0.10,
            "saebench_delta_ci95_low": -0.02,
            "cebench_interp_delta_vs_baseline": 0.12,
            "cebench_interp_delta_vs_baseline_ci95_low": -0.03,
            "cebench_interpretability_max": 9.0,
        }
    }

    phase4a_json = tmp_path / "phase4a.json"
    transcoder_json = tmp_path / "transcoder.json"
    ood_json = tmp_path / "ood.json"
    candidate_json = tmp_path / "candidate.json"
    _write_json(phase4a_json, phase4a)
    _write_json(transcoder_json, transcoder)
    _write_json(ood_json, ood)
    _write_json(candidate_json, candidate)

    out_dir = tmp_path / "gate_out"

    # With LCB thresholds at 0.0, should fail despite positive point estimates.
    proc_fail = subprocess.run(
        [
            sys.executable,
            str(RELEASE_GATE),
            "--phase4a-results",
            str(phase4a_json),
            "--transcoder-results",
            str(transcoder_json),
            "--ood-results",
            str(ood_json),
            "--external-candidate-json",
            str(candidate_json),
            "--external-mode",
            "joint",
            "--min-saebench-delta",
            "0.0",
            "--min-cebench-delta",
            "0.0",
            "--use-external-lcb",
            "--min-saebench-delta-lcb",
            "0.0",
            "--min-cebench-delta-lcb",
            "0.0",
            "--require-transcoder",
            "--require-ood",
            "--require-external",
            "--fail-on-gate-fail",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc_fail.returncode == 2, proc_fail.stdout + "\n" + proc_fail.stderr

    # Relax LCB thresholds to pass.
    proc_pass = subprocess.run(
        [
            sys.executable,
            str(RELEASE_GATE),
            "--phase4a-results",
            str(phase4a_json),
            "--transcoder-results",
            str(transcoder_json),
            "--ood-results",
            str(ood_json),
            "--external-candidate-json",
            str(candidate_json),
            "--external-mode",
            "joint",
            "--min-saebench-delta",
            "0.0",
            "--min-cebench-delta",
            "0.0",
            "--use-external-lcb",
            "--min-saebench-delta-lcb",
            "-0.05",
            "--min-cebench-delta-lcb",
            "-0.05",
            "--require-transcoder",
            "--require-ood",
            "--require-external",
            "--fail-on-gate-fail",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc_pass.returncode == 0, proc_pass.stdout + "\n" + proc_pass.stderr


def test_selector_accepts_assignment_results(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    assign_run = run_root / "assignment" / "run_assign"

    ckpt_1 = assign_run / "checkpoints" / "lambda_0.1" / "sae_seed123.pt"
    ckpt_2 = assign_run / "checkpoints" / "lambda_0.3" / "sae_seed456.pt"
    ckpt_1.parent.mkdir(parents=True, exist_ok=True)
    ckpt_2.parent.mkdir(parents=True, exist_ok=True)
    ckpt_1.write_text("x")
    ckpt_2.write_text("x")

    assignment_payload = {
        "config": {"hook_layer": 0, "hook_name": "blocks.0.hook_resid_pre"},
        "records": [
            {
                "lambda_consistency": 0.1,
                "selected_checkpoint": str(ckpt_1),
                "explained_variance": {"mean": 0.61},
                "selection_metrics": {
                    "internal_lcb": 0.72,
                    "ev_drop": 0.02,
                    "saebench_delta": -0.03,
                    "cebench_delta": -12.0,
                    "cebench_interpretability_max": 14.0,
                },
                "external_eval": {
                    "saebench_returncode": 0,
                    "cebench_returncode": 0,
                    "saebench_summary_path": "saebench_a.json",
                    "cebench_summary_path": "cebench_a.json",
                },
            },
            {
                "lambda_consistency": 0.3,
                "selected_checkpoint": str(ckpt_2),
                "explained_variance": {"mean": 0.58},
                "selection_metrics": {
                    "internal_lcb": 0.80,
                    "ev_drop": 0.03,
                    "saebench_delta": -0.01,
                    "cebench_delta": -4.0,
                    "cebench_interpretability_max": 18.0,
                },
                "external_eval": {
                    "saebench_returncode": 0,
                    "cebench_returncode": 0,
                    "saebench_summary_path": "saebench_b.json",
                    "cebench_summary_path": "cebench_b.json",
                },
            },
        ],
    }

    assignment_json = tmp_path / "assignment_results.json"
    _write_json(assignment_json, assignment_payload)

    out_dir = tmp_path / "selector_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--assignment-results",
            str(assignment_json),
            "--require-both-external",
            "--seed-level-selection",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs
    selected = json.loads((run_dirs[-1] / "selected_candidate.json").read_text())

    assert selected["source"] == "assignment"
    assert selected["condition_id"] == "assignv3_lambda0.3_seed456"
