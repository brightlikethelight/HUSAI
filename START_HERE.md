# Start Here

Updated: 2026-02-15

This is the canonical entrypoint for understanding HUSAI today.

## 1) Current Truth

- Internal consistency improvements are real and reproducible.
- External deltas vs matched baselines are still negative in tested settings.
- Strict release gate currently fails (`pass_all=false`).

Canonical status files:
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- `results/analysis/experiment_consistency_report.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`

## 2) Read in This Order

1. `START_HERE.md`
2. `EXECUTIVE_SUMMARY.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `REPO_NAVIGATION.md`
5. `RUNBOOK.md`
6. `docs/CYCLE4_LITERATURE_ACTIONS.md`
7. `EXPERIMENT_LOG.md`

## 3) Research Questions

1. Can trained SAEs beat random controls on consistency?
2. Which architecture/sparsity/width choices improve consistency?
3. Do internal gains transfer to SAEBench/CE-Bench external gains?
4. Can a model pass strict release gates (random + transcoder + OOD + external)?

## 4) Core Experiment Tracks

- Internal baseline/ablations: `run_phase4a_reproduction.py`, `run_core_ablations.py`
- Objective design: `run_assignment_consistency_v2.py`
- External frontier/scaling: `run_architecture_frontier_external.py`, `run_external_metric_scaling_study.py`
- Stress and release policy: `run_transcoder_stress_eval.py`, `run_ood_stress_eval.py`, `run_stress_gated_release_policy.py`
- Queue orchestration: `run_b200_high_impact_queue.sh`

## 5) Final Cycle-3 Outcomes

Source: `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`

- Frontier multiseed: complete (`20` records).
- Scaling multiseed: complete (`24` records).
- Transcoder stress: complete; gate failed.
- OOD stress: complete; gate passed.
- External gate: failed.
- Overall strict gate: `pass_all=false`.

## 6) If You Read Only Three Files

1. `PROJECT_STUDY_GUIDE.md`
2. `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
3. `PROPOSAL_COMPLETENESS_REVIEW.md`
